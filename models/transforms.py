import random
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import math
from fda import FDA_source_to_target
from skimage.exposure import match_histograms
import numpy as np


IMAGE_NET_MEAN = (0.485, 0.456, 0.406)
IMAGE_NET_STD = (0.229, 0.224, 0.225)


class PreProcessingPipeline:

	def __init__(self, config: list, input_size: int = None, dataset_mean: tuple = None,
	      dataset_std: tuple = None, dataset_min: tuple = None, dataset_max: tuple = None,
		  aux_dataset = None) -> None:

		self.functions = {
			'reduceMaskIndexes': self.reduceMaskIndexes,
			'randomCrop': self.randomCrop,
			'randomResizedCrop': self.randomResizedCrop,
			'colourTransforms': self.colourTransforms,
			'geometricTransforms': self.geometricTransforms,
			'normalizeImage': self.normalizeImage,
			'rgb2bgr': self.rgb2bgr,
			'contrastEnhancement': self.contrastEnhancement,
			'autocontrast': self.autocontrast,
			'equalizeHistogram': self.equalizeHistogram,
			'padding': self.padding,
			'fda': self.fda,
			'rhm': self.rhm,
			'rotate': self.rotate
		}

		self.config = config
		self.dataset_min = dataset_min
		self.dataset_max = dataset_max

		self.mean = list(IMAGE_NET_MEAN)
		self.std = list(IMAGE_NET_STD)

		if dataset_mean is not None and dataset_std is not None:
			self.mean.append(dataset_mean[3])
			self.std.append(dataset_std[3])

		if 'rhm' in self.config or 'fda' in config:
			assert aux_dataset is not None, 'To use RHM or FDA, a dataset must be provided'
			self.aux_dataset = aux_dataset
		
		if 'randomCrop' in self.config or 'padding' in self.config:
			assert input_size is not None, 'To use randomCrop or padding, an input size must be provided.'
			self.input_size = input_size


	def __call__(self, image, mask):

		# call selected preprocessing functions
		for f in self.config:
			func = self.functions[f]
			image, mask = func(image, mask)
		
		mask = mask.squeeze()
		
		return image, mask


	def reduceMaskIndexes(self, image, mask):
		mask[mask == 0] = 256
		mask = mask - 1
		return image, mask
	

	def randomCrop(self, image, mask):
		i, j, h, w = T.RandomCrop.get_params(image, (self.input_size, self.input_size))
		image = TF.crop(image, i, j, h, w)
		mask = TF.crop(mask, i, j, h, w)
		return image, mask


	def randomResizedCrop(self, image, mask):
		if random.random() > 0.5:
			i, j, h, w = T.RandomResizedCrop.get_params(image, scale=(0.3, 1), ratio=(1, 1))
			image = TF.resized_crop(image, i, j, h, w, size=image.shape[-2:])
			mask = TF.resized_crop(mask, i, j, h, w, size=image.shape[-2:], interpolation=T.InterpolationMode.NEAREST)
		return image, mask


	def colourTransforms(self, image, mask):

		# brightness adjustment
		if random.random() > 0:
			# factor = random.randint(2, 25) / 10
			image = TF.adjust_brightness(image, 2)
		
		# contrast adjustment
		if random.random() > 1:
			factor = random.randint(2, 25) / 10
			image = TF.adjust_contrast(image, factor)
		
		# hue adjustment
		if random.random() > 1:
			factor = random.randint(-5, 5) / 10
			image = TF.adjust_hue(image, factor)
		
		# saturation adjustment
		if random.random() > 1:
			factor = random.randint(2, 25) / 10
			image = TF.adjust_saturation(image, factor)
		
		return image, mask


	def rotate(self, image, mask):
		# image = TF.vflip(image)
		# mask  = TF.vflip(mask)
		image = TF.rotate(image, 180, fill=1)
		mask = TF.rotate(mask, 180, fill=255)
		return image, mask


	def geometricTransforms(self, image, mask):

		# horizontal flip
		if random.random() > 0.9:
			image = TF.hflip(image)
			mask  = TF.hflip(mask)

		# vertical flip
		if random.random() > 0.9:
			image = TF.vflip(image)
			mask  = TF.vflip(mask)

		# rotation 1
		if random.random() > 0.9:
			angle = random.choice([90, 180, 270])
			image = TF.rotate(image, angle, fill=1)
			mask = TF.rotate(mask, angle, fill=255)
		
		# rotation 2
		if random.random() > 0.9:
			i_h, i_w = image.size(1), image.size(2)
			# rotate
			angle = random.randint(-5, 5)
			image = TF.rotate(image, angle, fill=1)
			mask = TF.rotate(mask, angle, fill=255)
			# calculate cropping area
			crop_h, crop_w = self.calculateLargestRectangle(i_h, i_w, angle)
			# crop
			image = TF.center_crop(image, (crop_h, crop_w))
			mask = TF.center_crop(mask, (crop_h, crop_w))
			# resize
			image = TF.resize(image, (i_h, i_w))
			mask = TF.resize(mask, (i_h, i_w))

		return image, mask
	

	def normalizeImage(self, image, mask):
		c = image.size()[0]		# get number of channels in image
		image = T.Normalize(self.mean[:c], self.std[:c])(image)     # ImageNet mean and std
		return image, mask
	

	def rgb2bgr(self, image, mask):
		image = image[[2,1,0], :, :]
		return image, mask


	def contrastEnhancement(self, image, mask):

		if self.dataset_min is not None and self.dataset_max is not None:
			min_v = torch.tensor(self.dataset_min)
			max_v = torch.tensor(self.dataset_max)
		else:
			min_v = torch.min(image)
			max_v = torch.max(image)

		for i in range(image.size(0)):
			image[i] = (((image[i] - min_v[i]) / (max_v[i] - min_v[i])) * 1)
		
		return image, mask


	def autocontrast(self, image, mask):
		image = TF.autocontrast(image)
		return image, mask
	

	def equalizeHistogram(self, image, mask):
		image = TF.convert_image_dtype(image, dtype=torch.uint8)
		image = TF.equalize(image)
		image = TF.convert_image_dtype(image, dtype=torch.float32)
		return image, mask
	

	def padding(self, image, mask):
		# because we know we are only working with square images
		pad = int((self.input_size - image.shape[1]) / 2)
		image = TF.pad(image, padding=pad, fill=1)
		mask = TF.pad(mask, padding=pad, fill=255)
		return image, mask
	

	def fda(self, image, mask):

		if random.random() > 0.1:
			return image, mask

		# select random test image
		target_image = self.aux_dataset.getRandomImage()

		new_image = FDA_source_to_target(image, target_image, L=0.01)

		return new_image, mask
		

	def rhm(self, image, mask):

		if random.random() > 0.1:
			return image, mask

		# select random test image
		target_image = self.aux_dataset.getRandomImage()

		# match histograms
		matched = match_histograms(image.numpy(), target_image.numpy(), channel_axis=0)
		new_image = torch.tensor(matched)

		return new_image, mask
	

	########## auxiliar functions ##########


	def calculateLargestRectangle(self, h, w, angle):
		"""
		Given a rectangle of size wxh that has been rotated by 'angle',
		computes the width and height of the largest possible
		axis-aligned rectangle within the rotated rectangle.
		Original JS code by 'Andri' and Magnus Hoff from Stack Overflow
		Converted to Python by Aaron Snoswell
		Source: http://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders
		"""

		angle = math.radians(angle)
		quadrant = int(math.floor(angle / (math.pi / 2))) & 3
		sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
		alpha = (sign_alpha % math.pi + math.pi) % math.pi

		bb_w = w * math.cos(alpha) + h * math.sin(alpha)
		bb_h = w * math.sin(alpha) + h * math.cos(alpha)

		gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

		delta = math.pi - alpha - gamma

		length = h if (w < h) else w

		d = length * math.cos(alpha)
		a = d * math.sin(alpha) / math.sin(delta)

		y = a * math.cos(gamma)
		x = y * math.tan(gamma)

		return (int(bb_h - 2 * y), int(bb_w - 2 * x))
