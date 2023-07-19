from itertools import product
import random
import re
import numpy as np
from math import ceil
import torch


#-- set seed for random number generators --#

def set_seeds(seed: int) -> None:

	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	# torch.backends.cudnn.deterministic = True
	# torch.backends.cudnn.benchmark = False


#-- natural sorting in lists --#

def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]


#-- crop image into patches --#
# NOTE: pytorch implementation

def createPatches(img: torch.tensor, patchsize: int, stride: int) -> torch.tensor:

	i_h, i_w = img.size(-2), img.size(-1)
	i_c = img.size(0) if len(img.size()) == 3 else 1
	p_h = p_w = patchsize
	
	n_h = ceil((i_h - p_h) / stride + 1)
	n_w = ceil((i_w - p_w) / stride + 1)

	if len(img.size()) == 2:
		patches = torch.zeros((n_h * n_w, patchsize, patchsize), dtype=torch.long)
	else:
		patches = torch.zeros((n_h * n_w, img.size(0), patchsize, patchsize), dtype=torch.float32)

	for i, j in product(range(n_h), range(n_w)):

		# ignore last row and last column
		if i == n_h - 1 or j == n_w - 1:
			continue

		patches[i * n_w + j] = img[:, i * stride:i * stride + p_h, j * stride:j * stride + p_w]

	# last column - fixed width, variable height
	for i in range(n_h - 1):
		patches[(i+1) * n_w - 1] = img[:, i * stride:i * stride + p_h, -p_w:]
		
	# last row - fixed height, variable width
	for j in range(n_w - 1):
		patches[(n_h-1) * n_w + j] = img[:, -p_h:, j * stride:j * stride + p_w]

	# corner patch
	patches[-1] = img[:, -p_h:, -p_w:]

	return patches


#-- stitch logits into final prediction --#

def stitchLogits(patches, image_size, stride):

	i_h, i_w = image_size[:2]
	p_h, p_w = patches.shape[1:3]
	n_h = ceil((i_h - p_h) / stride + 1)
	n_w = ceil((i_w - p_w) / stride + 1)

	mean = np.zeros(image_size)
	patch_count = np.zeros(image_size)

	for p, (i, j) in zip(patches, product(range(n_h), range(n_w))):

		# ignore last row and last column
		if i == n_h - 1 or j == n_w - 1:
			continue

		# pixel count for calculating the mean
		patch_count[i * stride:i * stride + p_h, j * stride:j * stride + p_w] += ~np.isnan(p)
		# all NaN pixels set to 0
		ctignore = np.isnan(p)
		p[ctignore] = 0
		# sum pixel values
		mean[i * stride:i * stride + p_h, j * stride:j * stride + p_w] += p
		p[ctignore] = np.nan

	# last column - fixed width, variable height
	for i in range(n_h - 1):
		# select patch
		p = patches[(i + 1) * n_w - 1]
		# pixel count for calculating the mean
		patch_count[i * stride:i * stride + p_h, -p_w:] += ~np.isnan(p)
		# all NaN pixels set to 0
		ctignore = np.isnan(p)
		p[ctignore] = 0
		# sum pixel values
		mean[i * stride:i * stride + p_h, -p_w:] += p
		p[ctignore] = np.nan

	# last row - fixed height, variable width
	for j in range(n_w - 1):
		# select patch
		p = patches[(n_h - 1) * n_w + j]
		# pixel count for calculating the mean
		patch_count[-p_h:, j * stride:j * stride + p_w] += ~np.isnan(p)
		# all NaN pixels set to 0
		ctignore = np.isnan(p)
		p[ctignore] = 0
		# sum pixel values
		mean[-p_h:, j * stride:j * stride + p_w] += p
		p[ctignore] = np.nan
	
	# add corner patch (last one)
	p = patches[-1]
	# pixel count for calculating the mean
	patch_count[-p_h:, -p_w:] += ~np.isnan(p)
	# all NaN pixels set to 0
	ctignore = np.isnan(p)
	p[ctignore] = 0
	# sum pixel values
	mean[-p_h:, -p_w:] += p
	p[ctignore] = np.nan

	# divide the pixel sum by the pixel count to get the pixel mean
	mean = np.divide(mean, patch_count, out=np.zeros_like(mean), where=patch_count != 0)

	return mean


#-- expand labels to have class dimension --#

def expandLabels(labels: torch.tensor, num_classes: int) -> torch.tensor:

	# empty expanded labels tensor
	exp_labels = torch.zeros([num_classes, labels.size(0), labels.size(1), labels.size(2)], dtype=torch.long)

	# fill in class dimension for each class
	for i in range(num_classes):
		exp_labels[i][labels == i] = 1

	# move batch dimension forward: [batch, class, height, width]
	exp_labels = exp_labels.permute(1,0,2,3)

	return exp_labels
