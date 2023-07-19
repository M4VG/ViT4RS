from torch.utils.data import Dataset
import os
from PIL import Image
from enum import IntEnum
import torchvision.transforms.functional as TF
import numpy as np
import torch
import imageio
import skimage
import random

from aux import natural_keys


dataset_dir = {
	'potsdam': '/cfs/home/u121173/thesis/datasets/potsdam',
	'zurich_summer': '/cfs/home/u021173/thesis/datasets/zurich_summer'
}

class DatasetType(IntEnum):
	TRAIN = 0
	VALIDATION = 1
	TEST = 2


#################### GENERAL DATASET CLASSES ####################


class SemanticSegmentationDataset(Dataset):
	
	def __init__(self, root_dir: str, dataset_type: DatasetType, transform_fct, num_channels: int = 3, swap_channels: bool = False) -> None:

		self.root_dir = root_dir
		self.num_channels = num_channels
		self.dataset_type = dataset_type
		self.transform_fct = transform_fct
		self.swap_channels = swap_channels

		if self.dataset_type == DatasetType.TRAIN:
			sub_path = 'train'
		elif self.dataset_type == DatasetType.VALIDATION:
			sub_path = 'validation'
		elif self.dataset_type == DatasetType.TEST:
			sub_path = 'test'
		
		self.img_dir = os.path.join(self.root_dir, sub_path, 'images')
		self.ann_dir = os.path.join(self.root_dir, sub_path, 'annotations')
		
		# read images
		image_file_names = []
		for root, dirs, files in os.walk(self.img_dir):
			image_file_names.extend(files)
		self.images = sorted(image_file_names, key=natural_keys)
		
		# read annotations
		annotation_file_names = []
		for root, dirs, files in os.walk(self.ann_dir):
			annotation_file_names.extend(files)
		self.annotations = sorted(annotation_file_names, key=natural_keys)

		assert len(self.images) == len(self.annotations), "There must be as many images as there are segmentation maps"
		assert len(self.images) > 0, "No dataset found: check name, patch size and stride properties"


	def __len__(self) -> int:
		return len(self.images)


	def __getitem__(self, idx) -> dict:

		if self.swap_channels:
			channels = [3, 0, 1, 2]
		else:
			channels = [0, 1, 2, 3]

		img = imageio.imread(os.path.join(self.img_dir, self.images[idx]))[:,:,channels[:self.num_channels]]
		image = skimage.img_as_float32(img)
		image = np.rollaxis(image, 2, 0)
		segmentation_map = Image.open(os.path.join(self.ann_dir, self.annotations[idx]))
		image, segmentation_map = self.transform_fct(torch.tensor(image, dtype=torch.float32), TF.pil_to_tensor(segmentation_map).type(torch.long))
		return {'pixel_values': image, 'labels': segmentation_map}
	

	def getRandomImage(self) -> torch.tensor:

		path = random.choice(self.images)
		img = imageio.imread(os.path.join(self.img_dir, path))[:,:,:self.num_channels]
		image = skimage.img_as_float32(img)
		image = np.rollaxis(image, 2, 0)
		image = torch.tensor(image, dtype=torch.float32)
		return image



#################### SPECIFIC DATASET CLASSES ####################


class CustomDataset():

	def __init__(self, dataset_name: str, directory: str, input_channels: int, id2label: dict,
	      num_classes: int, colour_palette: list, ignore_class: int = None, channel_mean: tuple = None,
		  channel_std: tuple = None, crop: bool = False, patch_size: int = None, stride: int = None,
		  min_values: tuple = None, max_values: tuple = None) -> None:

		self.dataset_name = dataset_name
		self.dir = directory
		self.num_channels = input_channels
		self.id2label_dict = id2label
		self.label2id_dict = {v: k for k, v in self.id2label_dict.items()}
		self.num_classes = num_classes
		self.palette = colour_palette
		self.train_dataset = None
		self.valid_dataset = None
		self.test_dataset = None
		self.ignore_class = ignore_class
		self.channel_mean = channel_mean
		self.channel_std = channel_std
		self.crop = crop
		self.patch_size = patch_size
		self.stride = stride
		self.min_values = min_values
		self.max_values = max_values
	
	def name(self) -> str:
		return self.dataset_name
	
	def numChannels(self) -> str:
		return self.num_channels
	
	def id2label(self) -> dict:
		return self.id2label_dict
	
	def label2id(self) -> dict:
		return self.label2id_dict
	
	def numClasses(self) -> int:
		return self.num_classes

	def colourPalette(self) -> list:
		return self.palette
	
	def ignoreClass(self) -> int:
		return self.ignore_class
	
	def channelMean(self) -> tuple:
		return self.channel_mean
	
	def channelStd(self) -> tuple:
		return self.channel_std
	
	def cropped(self) -> bool:
		return self.crop
	
	def patchSize(self) -> int:
		return self.patch_size
	
	def patchStride(self) -> int:
		return self.stride

	def minValues(self) -> tuple:
		return self.min_values

	def maxValues(self) -> tuple:
		return self.max_values

	def createTrainingSplit(self, transform_fct) -> SemanticSegmentationDataset:
		pass

	def createValidationSplit(self, transform_fct) -> SemanticSegmentationDataset:
		pass

	def createTestSplit(self, transform_fct) -> SemanticSegmentationDataset:
		pass


class PotsdamDataset(CustomDataset):

	def __init__(self, input_channels: int = 3, crop: bool = False, patch_size: int = None, stride: int = None) -> None:

		# dataset info
		dataset_name = 'Potsdam'
		directory = dataset_dir['potsdam']
		if crop:
			directory += '_' + str(patch_size) + '_' + str(stride)
		num_channels = input_channels

		id2label_dict = {
			0: 'impervious_surfaces',
			1: 'building',
			2: 'low_vegetation',
			3: 'tree',
			4: 'car',
			5: 'clutter_background'
		}

		num_classes = len(id2label_dict)
		colour_palette = [[255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0], [255, 255, 0], [255, 0, 0]]
		ignore_class = 5
		channel_mean = (0.339419, 0.362922, 0.336925, 0.346662)
		channel_std = (0.140357, 0.138708, 0.144224, 0.14667)

		self.swap_channels = False
		# self.swap_channels = True

		# call superclass constructor
		super().__init__(dataset_name, directory, num_channels, id2label_dict, num_classes, colour_palette,
		   ignore_class, channel_mean, channel_std, crop, patch_size, stride)

	def createTrainingSplit(self, transform_fct) -> SemanticSegmentationDataset:
		self.train_dataset = SemanticSegmentationDataset(self.dir, DatasetType.TRAIN, transform_fct, num_channels=self.num_channels, swap_channels=self.swap_channels)
		return self.train_dataset

	def createValidationSplit(self, transform_fct) -> SemanticSegmentationDataset:
		self.valid_dataset = SemanticSegmentationDataset(self.dir, DatasetType.VALIDATION, transform_fct, num_channels=self.num_channels, swap_channels=self.swap_channels)
		return self.valid_dataset
	
	def createTestSplit(self, transform_fct) -> SemanticSegmentationDataset:
		self.test_dataset = SemanticSegmentationDataset(self.dir, DatasetType.TEST, transform_fct, num_channels=self.num_channels, swap_channels=self.swap_channels)
		return self.test_dataset


class ZurichSummerDataset(CustomDataset):

	def __init__(self, input_channels: int = 3, crop: bool = False, patch_size: int = None, stride: int = None) -> None:
		
		# dataset info
		dataset_name = 'ZurichSummer'
		directory = dataset_dir['zurich_summer']
		if crop:
			directory += '_' + str(patch_size) + '_' + str(stride)
		id2label_dict = {
			1: 'roads',
			2: 'buildings',
			3: 'trees',
			4: 'grass',
			5: 'bare_soil',
			6: 'water',
			7: 'rails',
			8: 'pools',
		}
		num_classes = len(id2label_dict)
		colour_palette = [[0, 0, 0], [100, 100, 100], [0, 125, 0], [0, 255, 0], [150, 80, 0], [0, 0, 150], [255, 255, 0], [150, 150, 255]]
		ignore_class = None
		channel_mean = (0.003155, 0.004310, 0.002663, 0.003133)
		channel_std = (0.001179, 0.001974, 0.001678, 0.00393)
		min_values = (0, 0, 0, 0)
		# max_values = tuple(np.array([2026, 2248, 2146, 3102])/65535)
		max_values = tuple(np.array([3102, 2026, 2248, 2146])/65535)

		# call superclass constructor
		super().__init__(dataset_name, directory, input_channels, id2label_dict, num_classes, colour_palette, 
		   ignore_class, channel_mean, channel_std, crop, patch_size, stride, min_values, max_values)
		
		# class specific properties
		# self.swap_channels = False
		self.swap_channels = True

	def createTrainingSplit(self, transform_fct) -> SemanticSegmentationDataset:
		self.train_dataset = SemanticSegmentationDataset(self.dir, DatasetType.TRAIN, transform_fct, num_channels=self.num_channels, swap_channels=self.swap_channels)
		return self.train_dataset

	def createValidationSplit(self, transform_fct) -> SemanticSegmentationDataset:
		self.valid_dataset = SemanticSegmentationDataset(self.dir, DatasetType.VALIDATION, transform_fct, num_channels=self.num_channels, swap_channels=self.swap_channels)
		return self.valid_dataset
	
	def createTestSplit(self, transform_fct) -> SemanticSegmentationDataset:
		self.test_dataset = SemanticSegmentationDataset(self.dir, DatasetType.TEST, transform_fct, num_channels=self.num_channels, swap_channels=self.swap_channels)
		return self.test_dataset



#################### DATASET BUILDER ####################


dataset_classes = {
	'potsdam': PotsdamDataset,
	'zurich_summer': ZurichSummerDataset
}


def datasetBuilder(dataset_config: dict) -> CustomDataset:

	dataset_name = dataset_config['name']
	input_channels = dataset_config['input_channels']
	crop = dataset_config['crop']
	patch_size = dataset_config['patch_size']
	stride = dataset_config['stride']

	assert input_channels in [3, 4], 'Input channels must be 3 or 4.'
	return dataset_classes[dataset_name](input_channels, crop, patch_size, stride)
