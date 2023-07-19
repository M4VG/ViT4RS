import os
import torch
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm
import wandb
from copy import deepcopy
import numpy as np

from transforms import PreProcessingPipeline
from loss import lossFunctionBuilder
from metrics import Evaluator
from aux import set_seeds


def train(model, device, dataset, config: dict) -> str:

	#-- parse config arguments --#

	trial = config['general']['trial']
	pre_processing_train = config['train']['training_pre_processing']
	pre_processing_valid = config['train']['validation_pre_processing']
	batch_size = config['train']['batch_size']
	accumulation_steps = config['train']['accumulation_steps']
	loss_function = config['train']['loss_function']
	min_epochs = config['train']['min_epochs']
	training_patience = config['train']['patience']
	learning_rate = config['train']['learning_rate']
	log = config['logging']['log']
	wandb_project = config['logging']['project']
	weights_dir = config['dirs']['weights_dir']

	#-- set seeds for random number generators --#

	# set_seeds(63)
	set_seeds(20)

	#-- dataloaders setup --#

	ppp_valid = PreProcessingPipeline(pre_processing_valid, dataset_mean=dataset.channelMean(), dataset_std=dataset.channelStd(),
			     dataset_min=dataset.minValues(), dataset_max=dataset.maxValues())
	ppp_train = PreProcessingPipeline(pre_processing_train, dataset_mean=dataset.channelMean(), dataset_std=dataset.channelStd(),
			     dataset_min=dataset.minValues(), dataset_max=dataset.maxValues(), aux_dataset=dataset.createTrainingSplit(ppp_valid))

	train_dataset = dataset.createTrainingSplit(ppp_train)
	valid_dataset = dataset.createValidationSplit(ppp_valid)

	train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size)

	#-- metrics setup --#

	metric = Evaluator(dataset.numClasses())

	class_idxs = list(range(dataset.numClasses()))
	if dataset.ignoreClass() is not None:
		class_idxs.pop(dataset.ignoreClass())

	curr_score, prev_score, best_score = 0., 0., -1.
	train_losses, val_losses, meanIoUs, meanF1Scores, accuracies = [], [], [], [], []

	#-- training setup --#

	epoch = 0
	stop = False
	patience = training_patience
	optimizer = torch.optim.AdamW(model.parameters(), learning_rate)
	loss_fct = lossFunctionBuilder(loss_function, dataset.numClasses())

	#-- wandb setup --#

	run_id = None

	if log:
		wandb_config = {k: config[k] for k in config.keys() if k not in ['general', 'logging', 'dirs']}
		run_name = 'trial-' + str(trial).zfill(2)
		run_id = wandb.util.generate_id()
		wandb.init(project = wandb_project, entity = 'mike-avg', name = run_name, config = wandb_config, resume = True, id = run_id)
		# wandb.watch(model, log='all')

	#-- training loop --#

	while not stop:

		# turn on training mode
		model.train()

		epoch += 1
		print("\nEpoch:", epoch)

		# reset loss accumulator
		loss_accumulator = 0

		print('\nTraining...\n')

		# zero the parameter gradients
		optimizer.zero_grad()

		for idx, batch in enumerate(tqdm(train_dataloader)):

			# get the inputs
			inputs = batch["pixel_values"].to(device)

			# forward pass
			logits = model(inputs)
			# logits, logits_augmented = model(inputs)

			# upsample logits to original resolution
			upsampled_logits = nn.functional.interpolate(logits, size=inputs.shape[-2:], mode="bilinear", align_corners=False)
			# upsampled_logits_augmented = nn.functional.interpolate(logits_augmented, size=inputs.shape[-2:], mode="bilinear", align_corners=False)

			# calculate training loss
			loss = loss_fct(upsampled_logits, batch["labels"])
			# loss = loss_fct(upsampled_logits, upsampled_logits_augmented, batch["labels"])
			loss = loss / accumulation_steps

			# backwards pass
			loss.backward()

			# update weights with accumulated gradients
			if (idx + 1) % accumulation_steps == 0 or (idx + 1) == len(train_dataloader):
				optimizer.step()
				optimizer.zero_grad()

			# add batch loss to loss accumulator
			loss_accumulator += loss.item()
		
		# update avg train loss
		avg_loss = loss_accumulator / len(train_dataloader)
		train_losses.append(avg_loss)

		# evaluate
		with torch.no_grad():

			# turn on eval mode
			model.eval()

			# reset metrics and loss accumulator
			metric.reset()
			loss_accumulator = 0

			print('\nEvaluating...\n')

			for _, batch in enumerate(tqdm(valid_dataloader)):

				# inputs
				inputs = batch['pixel_values'].to(device)

				# inference
				logits = model(inputs)
				# logits, logits_augmented = model(inputs)
			
				# upsample logits and choose highest softmax score
				upsampled_logits = nn.functional.interpolate(logits, size=inputs.shape[-2:], mode="bilinear", align_corners=False)
				# upsampled_logits_augmented = nn.functional.interpolate(logits_augmented, size=inputs.shape[-2:], mode="bilinear", align_corners=False)
				predicted = upsampled_logits.argmax(dim=1)

				# calculate validation loss
				loss = loss_fct(upsampled_logits, batch["labels"])
				# loss = loss_fct(upsampled_logits, upsampled_logits_augmented, batch["labels"])
				loss_accumulator += loss.item()

				# update metrics
				metric.add_batch(batch['labels'].numpy(), predicted.detach().cpu().numpy())
	

		# update aux metrics vars
		class_f1 = metric.F1()
		class_f1[class_f1 == np.nan] = 0.
		curr_score = np.nanmean(class_f1[class_idxs])

		avg_loss = loss_accumulator / len(valid_dataloader)
		val_losses.append(avg_loss)
		meanIoUs.append(np.nanmean(metric.Intersection_over_Union()[class_idxs]))
		meanF1Scores.append(np.nanmean(class_f1[class_idxs]))
		accuracies.append(metric.OA())

		# stopping condition
		if epoch <= min_epochs:
			pass
		elif curr_score <= prev_score:
			patience -= 1
		else:
			patience = training_patience
		
		if epoch >= min_epochs and patience <= 0:
			stop = True
		
		print('> delta:', curr_score - prev_score)
		prev_score = curr_score

		# print metrics for this epoch
		print("---------------------")
		print('Average loss:', avg_loss)
		print('Mean IoU:', meanIoUs[-1])
		print('Mean F1-score:', meanF1Scores[-1])
		print('Overall accuracy:', accuracies[-1])
		print("---------------------")

		# log metrics to wandb
		if log:
			wandb.log({
				'epoch': epoch,
				'learning_rate': optimizer.param_groups[0]['lr'],
				'training': {
					'loss': train_losses[-1]
				},
				'validation': {
					'loss': val_losses[-1],
					'mIoU': meanIoUs[-1],
					'mean F1-score': meanF1Scores[-1],
					'accuracy': accuracies[-1]
				}
			})
		
		if curr_score > best_score:
			print('best model updated')
			best_score = curr_score
			best_model_dict = deepcopy(model.state_dict())
		
		if min_epochs > 50 and epoch == 50:
			weights_dir1 = os.path.join(os.path.abspath(weights_dir), model.name() + '_' + dataset.name() + '_' + 'trial' + str(trial).zfill(2) + '_50e')
			torch.save(best_model_dict, weights_dir1)
	
	#-- end training loop --#

	# save weights
	weights_dir2 = os.path.join(os.path.abspath(weights_dir), model.name() + '_' + dataset.name() + '_' + 'trial' + str(trial).zfill(2))
	torch.save(best_model_dict, weights_dir2)

	# return wandb run id
	return run_id
