from math import ceil
import math
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm
import wandb
import imageio
import torchvision.transforms.functional as TF
from scipy.ndimage import rotate

from transforms import PreProcessingPipeline
from aux import stitchLogits, createPatches
from metrics import Evaluator


def eval(model, device, dataset, config: dict, run_id: str):

	#-- parse config arguments --#

	trial = config['general']['trial']
	pre_processing = config['eval']['pre_processing']
	save_masks = config['eval']['save_masks']
	log = config['logging']['log']
	wandb_project = config['logging']['project']
	weights_dir = config['dirs']['weights_dir']
	out_masks_dir = config['dirs']['out_masks_dir']

	path2weights = os.path.join(os.path.abspath(weights_dir), model.name() + '_' + dataset.name() + '_' + 'trial' + str(trial).zfill(2))# + '_50e')
	out_masks_dir = os.path.join(os.path.abspath(out_masks_dir), dataset.name() + '/trial' + str(trial).zfill(2))

	if save_masks and not os.path.exists(out_masks_dir):
		os.makedirs(out_masks_dir)

	#-- model setup --#

	model.load_state_dict(torch.load(path2weights))

	#-- dataloaders setup --#

	ppp = PreProcessingPipeline(pre_processing, dataset_mean=dataset.channelMean(), dataset_std=dataset.channelStd(),
			     dataset_min=dataset.minValues(), dataset_max=dataset.maxValues())
	test_dataset = dataset.createTestSplit(ppp)
	test_dataloader = DataLoader(test_dataset, batch_size=1)

	#-- other setup --#

	metric = Evaluator(dataset.numClasses())

	#-- wandb setup --#

	if log:
		if run_id == None:
			wandb_config = {k: config[k] for k in config.keys() if k not in ['general', 'logging', 'dirs']}
			run_name = 'trial-' + str(trial).zfill(2) + '-eval'
			wandb.init(project = wandb_project, entity = 'mike-avg', name = run_name, config = wandb_config)
		else:
			wandb.init(id = run_id)

	#-- evaluation routine --#

	with torch.no_grad():

		# turn on eval mode
		model.eval()

		for idx, batch in enumerate(tqdm(test_dataloader)):

			img = batch['pixel_values']#.to(device)

			# augmentations
			images = [
				img,
				TF.hflip(img),
				TF.vflip(img),
				torch.rot90(img, 1, [2,3]),
				torch.rot90(img, 2, [2,3]),
				torch.rot90(img, 3, [2,3])
			]

			sum_logits = np.zeros([img.size(2), img.size(3), dataset.numClasses()])

			for im in range(len(images)):

				image = images[im]
				# print(image)

				# cropping
				if dataset.cropped():
					patches = createPatches(image[0], dataset.patchSize(), dataset.patchStride())
				else:
					patches = image
				
				logit_patches = np.zeros([patches.size(0), dataset.numClasses(), patches.size(2), patches.size(3)])

				for i in range(patches.size(0)):
					# inference
					p = patches[i]
					inputs = p.unsqueeze(0).to(device)

					logits = model(inputs)
					# logits, _ = model(inputs)
					
					# upsample logits because model outputs at 1/4 of the input resolution
					upsampled_logits = nn.functional.interpolate(logits, size=patches.shape[-2:], mode="bilinear", align_corners=False)
					# upsampled_logits = np.rollaxis(upsampled_logits.detach().cpu().numpy(), 1, 4)
					logit_patches[i] = upsampled_logits.detach().cpu().numpy()
				
				logit_patches = np.rollaxis(logit_patches, 1, 4)

				# stitch logits
				if dataset.cropped():
					stitched_logits = stitchLogits(logit_patches, (image[0].size(1), image[0].size(2), dataset.numClasses()), dataset.patchStride())
				else:
					stitched_logits = logit_patches.squeeze()

				# inverse transformation
				if im == 0:
					pass
				elif im == 1:
					stitched_logits = np.fliplr(stitched_logits)
				elif im == 2:
					stitched_logits = np.flipud(stitched_logits)
				elif im == 3:
					print(stitched_logits.shape)
					stitched_logits = rotate(stitched_logits, 270)
					print(stitched_logits.shape)
				elif im == 4:
					stitched_logits = rotate(stitched_logits, 180)
				elif im == 5:
					stitched_logits = rotate(stitched_logits, 90)

				print(im)
				sum_logits += stitched_logits

			# update metrics
			# sum_logits = sum_logits / 6
			prediction = np.argmax(sum_logits, axis=2)
			metric.add_batch(np.asarray(batch['labels'].squeeze(), np.uint8), prediction)

			# save visual results
			if save_masks:
				print('saving masks')
				colour_mask = np.ones((batch['labels'].size(1), batch['labels'].size(2), 3), np.uint8) # height, width, 3
				palette = dataset.colourPalette()
				for i in range(batch['labels'].size(1)):
					for j in range(batch['labels'].size(2)):
						colour_mask[i,j] = palette[prediction[i,j]]
				imageio.imwrite(out_masks_dir + '/' + str(idx).zfill(2) + '.png', colour_mask)


	# calculate metrics
	class_iou = metric.Intersection_over_Union()
	class_f1 = metric.F1()
	class_f1[np.isnan(class_f1)] = 0.

	class_idxs = list(range(dataset.numClasses()))
	if dataset.ignoreClass() is not None:
		class_idxs.pop(dataset.ignoreClass())

	metrics = {
		'overall_accuracy': np.nanmean(metric.OA()),
		'mean_iou': np.nanmean(class_iou[class_idxs]),
		'mean_f1': np.nanmean(class_f1[class_idxs]),
		'class_iou': class_iou,
		'class_f1': class_f1
	}

	# log metrics to wandb
	if log:

		class_metrics_table = wandb.Table(columns=['class', 'class_id'], data=list(dataset.label2id().items()))

		for key in list(metrics.keys()):
			if isinstance(metrics[key], np.ndarray):
				data = []
				data += [round(n, 4) for n in metrics[key]]
				class_metrics_table.add_column(name=key, data=data)
			else:
				wandb.log({key: round(metrics[key], 4)}, commit=False)
		
		wandb.log({"Per-class metrics": class_metrics_table})
	
	# print metrics
	print("\n---------------------")
	for key in list(metrics.keys()):
		print(key, np.around(metrics[key], 4) * 100)
	print("---------------------")
