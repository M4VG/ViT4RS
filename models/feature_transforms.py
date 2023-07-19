from itertools import product
import random
import torch


class FeatureTransform:

	def __init__(self, division=4, num_transforms=1) -> None:
		self.division = division
		self.num_transforms = num_transforms
		self.inverse_transforms = {
			self.rotate_90: self.rotate_270,
			self.rotate_180: self.rotate_180,
			self.rotate_270: self.rotate_90,
			self.flip_horizontal: self.flip_horizontal,
			self.flip_vertical: self.flip_vertical
		}
		self.transforms = list(self.inverse_transforms.keys())
		self.current_transforms = [None for i in range(num_transforms)]

	def transform_features(self, features_list: list):

		segments_list = [self.divide_segments(f) for f in features_list]

		for i in range(self.num_transforms):
			self.current_transforms[i] = random.choice(self.transforms)
			new_features_list = [self.current_transforms[i](f) for f in segments_list]
			stitched_features_list = [self.stitch_segments(f) for f in new_features_list]
			features_list = [torch.concat((features_list[j], stitched_features_list[j])) for j in range(len(features_list))]

		return features_list
	
	def invert_transformation(self, output_masks: torch.tensor):

		batch_size = int(output_masks.shape[0] / (self.num_transforms + 1))
		segments = self.divide_segments(output_masks[batch_size:])

		for i in range(self.num_transforms):
			segments[:, batch_size * i:batch_size * (i+1)] = self.inverse_transforms[self.current_transforms[i]](segments[:, batch_size * i:batch_size * (i+1)])

		stitched_masks = self.stitch_segments(segments)

		return torch.concat((output_masks[:batch_size], stitched_masks))

	def split_logits(self, logits: torch.tensor):

		batch_size = int(logits.shape[0] / (self.num_transforms + 1))
		logits_transformed = logits[batch_size:batch_size * 2]

		for i in range(self.num_transforms - 1):
			logits_transformed += logits[batch_size * (i+2):batch_size * (i+3)]

		return logits[:batch_size], torch.div(logits_transformed, self.num_transforms)		# original, transformed (avg)

	def divide_segments(self, features: torch.tensor):

		b, c, h, w = features.shape
		seg_size = int(h / self.division)
		n = self.division * self. division

		segments = torch.zeros([n, b, c, seg_size, seg_size], device=features.device)

		for i, j in product(range(self.division), range(self.division)):
			segments[i * self.division + j] = features[:, :, i * seg_size:(i+1) * seg_size, j * seg_size:(j+1) * seg_size]
		
		return segments
	
	def stitch_segments(self, segments: torch.tensor):

		n, b, c, h, w = segments.shape
		features_size = h * self.division

		stitched_features = torch.zeros([b, c, features_size, features_size], device=segments.device)

		for i, j in product(range(self.division), range(self.division)):
			stitched_features[:, :, i * h:(i+1) * h, j * w:(j+1) * w] = segments[i * self.division + j]
		
		return stitched_features

	def rotate_90(self, segments: torch.tensor):

		new_segments = torch.zeros_like(segments, device=segments.device)

		for i, j in product(range(self.division), range(self.division)):
			new_segments[i * self.division + j] = segments[j * self.division + (self.division-1 - i)]

		return new_segments

	def rotate_180(self, segments: torch.tensor):
		return torch.flip(segments, [0])

	def rotate_270(self, segments: torch.tensor):

		new_segments = torch.zeros_like(segments, device=segments.device)

		for i, j in product(range(self.division), range(self.division)):
			new_segments[i * self.division + j] = segments[(self.division-1 - j) * self.division + i]

		return new_segments
	
	def flip_horizontal(self, segments: torch.tensor):

		new_segments = torch.zeros_like(segments, device=segments.device)

		for i, j in product(range(self.division), range(self.division)):
			new_segments[i * self.division + j] = segments[i * self.division + (self.division-1 - j)]

		return new_segments
	
	def flip_vertical(self, segments: torch.tensor):

		new_segments = torch.zeros_like(segments, device=segments.device)

		for i, j in product(range(self.division), range(self.division)):
			new_segments[i * self.division + j] = segments[(self.division-1 - i) * self.division + j]

		return new_segments
	