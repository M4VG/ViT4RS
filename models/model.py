from transformers import SegformerForSemanticSegmentation, SegformerModel
import torch
import torch.nn as nn

from decoders.lawin.lawin_config import lawin_config
from decoders.lawin.lawin import LAWINHead
import decoders.dcfam.dcfam as dcfam
import decoders.ft_unetformer.ft_unetformer as ft_unetformer


#################### "ABSTRACT" MODEL WRAPPER CLASS ####################


class SegmentationModel(nn.Module):

	def __init__(self, model_name: str, model: nn.Module = None, encoder: nn.Module = None, decoder: nn.Module = None) -> None:
		
		super().__init__()
		self.model_name = model_name

		# complete model
		self.model = model

		# divided structure
		self.encoder = encoder
		self.decoder = decoder

	def name(self) -> str:
		return self.model_name
	
	def forward(self, inputs: torch.Tensor) -> torch.Tensor:
		return self.model(inputs)


#################### CONCRETE MODEL CLASSES ####################


class MiT_4channels(nn.Module):

	def __init__(self, variant: int, num_classes: int, weight_init: str) -> None:
		super().__init__()

		self.encoder = SegformerModel.from_pretrained('nvidia/mit-{}'.format(variant), num_labels=num_classes, num_channels=4, ignore_mismatched_sizes=True)
		sd = self.encoder.state_dict()

		# get pre-trained weights for the 1st conv layer
		pretrained_weights_layer1 = SegformerModel.from_pretrained('nvidia/mit-{}'.format(variant), num_labels=num_classes).state_dict()['encoder.patch_embeddings.0.proj.weight']

		if weight_init == 'zero':
			# create new weights with zero-initialized weights for channel 4 and pre-trained weights for the first 3 channels
			weights_layer1_shape = self.encoder.state_dict()['encoder.patch_embeddings.0.proj.weight'].size()
			new_weights_layer1 = torch.zeros(weights_layer1_shape)
			new_weights_layer1[:,:3,:,:] = pretrained_weights_layer1
			# replace layer 1 weights on the model
			sd['encoder.patch_embeddings.0.proj.weight'] = new_weights_layer1

		elif weight_init == 'random':
			# replace layer 1 weights on the model
			sd['encoder.patch_embeddings.0.proj.weight'][:,:3,:,:] = pretrained_weights_layer1
		
		elif weight_init == 'red_combined':
			# replace G and B weights
			sd['encoder.patch_embeddings.0.proj.weight'][:,1:3,:,:] = pretrained_weights_layer1[:,1:3,:,:]
			# replace R weights (2/3 of R)
			sd['encoder.patch_embeddings.0.proj.weight'][:,0,:,:] = pretrained_weights_layer1[:,0,:,:] * 2 / 3
			# replace IR weights (1/3 of R)
			sd['encoder.patch_embeddings.0.proj.weight'][:,3,:,:] = pretrained_weights_layer1[:,0,:,:] / 3
		
		elif weight_init == 'shift_spectrum':
			# R = 1/3 of R + 2/3 of G
			sd['encoder.patch_embeddings.0.proj.weight'][:,0,:,:] = (pretrained_weights_layer1[:,0,:,:] / 3) + (pretrained_weights_layer1[:,1,:,:] * 2 / 3)
			# G = 1/3 of G + 2/3 of B
			sd['encoder.patch_embeddings.0.proj.weight'][:,1,:,:] = (pretrained_weights_layer1[:,1,:,:] / 3) + (pretrained_weights_layer1[:,2,:,:] * 2 / 3)
			# B = 1/3 of B
			sd['encoder.patch_embeddings.0.proj.weight'][:,2,:,:] = pretrained_weights_layer1[:,2,:,:] / 3
			# IR = 2/3 of R
			sd['encoder.patch_embeddings.0.proj.weight'][:,3,:,:] = pretrained_weights_layer1[:,0,:,:] * 2 / 3

		self.encoder.load_state_dict(sd)
	
	def forward(self, inputs: torch.Tensor):
		return self.encoder(inputs, output_hidden_states=True)


class Segformer(SegmentationModel):

	def __init__(self, variant: int, num_classes: int, weight_init: str) -> None:

		model = SegformerForSemanticSegmentation.from_pretrained('nvidia/mit-{}'.format(variant), num_labels=num_classes)
		model_name = 'Segformer-{}'.format(variant)
		super().__init__(model_name=model_name, model=model)
	
	def forward(self, inputs: torch.Tensor) -> torch.Tensor:
		return self.model(inputs).logits


class Segformer_4channels(SegmentationModel):

	def __init__(self, variant: int, num_classes: int, weight_init: str) -> None:

		model = SegformerForSemanticSegmentation.from_pretrained('nvidia/mit-{}'.format(variant), num_labels=num_classes, num_channels=4, ignore_mismatched_sizes=True)
		model_name = 'Segformer-{}-4ch'.format(variant)
		sd = model.state_dict()

		# get pre-trained weights for the 1st conv layer
		pretrained_weights_layer1 = SegformerModel.from_pretrained('nvidia/mit-{}'.format(variant), num_labels=num_classes).state_dict()['encoder.patch_embeddings.0.proj.weight']

		if weight_init == 'zero':
			# create new weights with zero-initialized weights for channel 4 and pre-trained weights for the first 3 channels
			weights_layer1_shape = model.state_dict()['segformer.encoder.patch_embeddings.0.proj.weight'].size()
			new_weights_layer1 = torch.zeros(weights_layer1_shape)
			new_weights_layer1[:,:3,:,:] = pretrained_weights_layer1
			# replace layer 1 weights on the model
			sd['segformer.encoder.patch_embeddings.0.proj.weight'] = new_weights_layer1

		elif weight_init == 'random':
			# replace layer 1 weights on the model
			sd['segformer.encoder.patch_embeddings.0.proj.weight'][:,:3,:,:] = pretrained_weights_layer1
		
		elif weight_init == 'red_combined':
			# replace G and B weights
			sd['segformer.encoder.patch_embeddings.0.proj.weight'][:,1:3,:,:] = pretrained_weights_layer1[:,1:3,:,:]
			# replace R weights (2/3 of R)
			sd['segformer.encoder.patch_embeddings.0.proj.weight'][:,0,:,:] = pretrained_weights_layer1[:,0,:,:] * 2 / 3
			# replace IR weights (1/3 of R)
			sd['segformer.encoder.patch_embeddings.0.proj.weight'][:,3,:,:] = pretrained_weights_layer1[:,0,:,:] / 3
		
		elif weight_init == 'shift_spectrum':
			# R = 1/3 of R + 2/3 of G
			sd['segformer.encoder.patch_embeddings.0.proj.weight'][:,0,:,:] = (pretrained_weights_layer1[:,0,:,:] / 3) + (pretrained_weights_layer1[:,1,:,:] * 2 / 3)
			# G = 1/3 of G + 2/3 of B
			sd['segformer.encoder.patch_embeddings.0.proj.weight'][:,1,:,:] = (pretrained_weights_layer1[:,1,:,:] / 3) + (pretrained_weights_layer1[:,2,:,:] * 2 / 3)
			# B = 1/3 of B
			sd['segformer.encoder.patch_embeddings.0.proj.weight'][:,2,:,:] = pretrained_weights_layer1[:,2,:,:] / 3
			# IR = 2/3 of R
			sd['segformer.encoder.patch_embeddings.0.proj.weight'][:,3,:,:] = pretrained_weights_layer1[:,0,:,:] * 2 / 3

			# # R = 0.1 of R + 0.9 of G
			# sd['segformer.encoder.patch_embeddings.0.proj.weight'][:,0,:,:] = pretrained_weights_layer1[:,0,:,:] * 0.1 + pretrained_weights_layer1[:,1,:,:] * 0.9
			# # G = 0.1 of G + 0.9 of B
			# sd['segformer.encoder.patch_embeddings.0.proj.weight'][:,1,:,:] = pretrained_weights_layer1[:,1,:,:] * 0.1 + pretrained_weights_layer1[:,2,:,:] * 0.9
			# # B = 0.1 of B
			# sd['segformer.encoder.patch_embeddings.0.proj.weight'][:,2,:,:] = pretrained_weights_layer1[:,2,:,:] * 0.1
			# # IR = 0.9 of R
			# sd['segformer.encoder.patch_embeddings.0.proj.weight'][:,3,:,:] = pretrained_weights_layer1[:,0,:,:] * 0.9

		model.load_state_dict(sd)

		super().__init__(model_name=model_name, model=model)
	
	def forward(self, inputs: torch.Tensor) -> torch.Tensor:
		return self.model(inputs).logits


class Lawin(SegmentationModel):

	def __init__(self, variant: int, num_classes: int, weight_init: str) -> None:

		model_name = 'Lawin-{}'.format(variant)

		# encoder: MiT
		encoder = SegformerModel.from_pretrained('nvidia/mit-{}'.format(variant), num_labels=num_classes)

		# decoder: LawinASSP
		cfg = lawin_config['{}'.format(variant)]
		cfg['num_classes'] = num_classes
		decoder = LAWINHead(cfg)

		super().__init__(model_name=model_name, encoder=encoder, decoder=decoder)
	
	def forward(self, inputs: torch.Tensor) -> torch.Tensor:
		encoder_output = self.encoder(inputs, output_hidden_states=True)
		decoder_output = self.decoder(encoder_output.hidden_states)
		return decoder_output


class Lawin_4channels(SegmentationModel):

	def __init__(self, variant: int, num_classes: int, weight_init: str) -> None:

		model_name = 'Lawin-{}-4ch'.format(variant)

		# encoder: MiT
		encoder = MiT_4channels(variant, num_classes, weight_init)

		# decoder: LawinASSP
		cfg = lawin_config['{}'.format(variant)]
		cfg['num_classes'] = num_classes
		decoder = LAWINHead(cfg)

		super().__init__(model_name=model_name, encoder=encoder, decoder=decoder)
	
	def forward(self, inputs: torch.Tensor) -> torch.Tensor:
		encoder_output = self.encoder(inputs)
		decoder_output = self.decoder(encoder_output.hidden_states)
		return decoder_output


class DC_MiT(SegmentationModel):

	def __init__(self, variant: int, num_classes: int, weight_init: str) -> None:

		model_name = 'DC-MiT-{}'.format(variant)

		# encoder: MiT
		encoder = SegformerModel.from_pretrained('nvidia/mit-{}'.format(variant), num_labels=num_classes)

		# decoder: DCFAM
		decoder = dcfam.Decoder(encoder_channels=encoder.config.hidden_sizes, num_classes=num_classes)

		super().__init__(model_name=model_name, encoder=encoder, decoder=decoder)
	
	def forward(self, inputs: torch.Tensor) -> torch.Tensor:
		encoder_output = self.encoder(inputs, output_hidden_states=True)
		decoder_output = self.decoder(encoder_output.hidden_states)
		return decoder_output


class DC_MiT_4channels(SegmentationModel):

	def __init__(self, variant: int, num_classes: int, weight_init: str) -> None:

		model_name = 'DC-MiT-{}-4ch'.format(variant)

		# encoder: MiT
		encoder = MiT_4channels(variant, num_classes, weight_init)

		# decoder: DCFAM
		decoder = dcfam.Decoder(encoder_channels=encoder.encoder.config.hidden_sizes, num_classes=num_classes)

		super().__init__(model_name=model_name, encoder=encoder, decoder=decoder)
	
	def forward(self, inputs: torch.Tensor) -> torch.Tensor:
		encoder_output = self.encoder(inputs)
		decoder_output = self.decoder(encoder_output.hidden_states)
		return decoder_output


class MiT_UNetFormer(SegmentationModel):

	def __init__(self, variant: int, num_classes: int, weight_init: str) -> None:

		model_name = 'MiT-UNetFormer-{}'.format(variant)

		# encoder: MiT
		encoder = SegformerModel.from_pretrained('nvidia/mit-{}'.format(variant), num_labels=num_classes)

		# decoder: from FT-UNetFormer
		decoder = ft_unetformer.Decoder(encoder_channels=encoder.config.hidden_sizes, decode_channels=256, num_classes=num_classes)

		super().__init__(model_name=model_name, encoder=encoder, decoder=decoder)
	
	def forward(self, inputs: torch.Tensor) -> torch.Tensor:
		encoder_output = self.encoder(inputs, output_hidden_states=True)
		decoder_output = self.decoder(encoder_output.hidden_states)
		return decoder_output


class MiT_UNetFormer_4channels(SegmentationModel):

	def __init__(self, variant: int, num_classes: int, weight_init: str) -> None:

		model_name = 'MiT-UNetFormer-{}-4ch'.format(variant)

		# encoder: MiT
		encoder = MiT_4channels(variant, num_classes, weight_init)

		# decoder: from FT-UNetFormer
		decoder = ft_unetformer.Decoder(encoder_channels=encoder.encoder.config.hidden_sizes, decode_channels=256, num_classes=num_classes)

		super().__init__(model_name=model_name, encoder=encoder, decoder=decoder)
	
	def forward(self, inputs: torch.Tensor) -> torch.Tensor:
		encoder_output = self.encoder(inputs)
		decoder_output = self.decoder(encoder_output.hidden_states)
		return decoder_output


class DC_Swin(SegmentationModel):

	def __init__(self, variant: int, num_classes: int, weight_init: str) -> None:

		model_name = 'DC-Swin-{}'.format(variant)

		if variant == 'base':
			model = dcfam.dcswin_base(num_classes=num_classes)
		elif variant == 'small':
			model = dcfam.dcswin_small(num_classes=num_classes)
		elif variant == 'tiny':
			model = dcfam.dcswin_tiny(num_classes=num_classes)

		super().__init__(model_name=model_name, model=model)


class FT_UNetFormer(SegmentationModel):

	def __init__(self, variant: int, num_classes: int, weight_init: str) -> None:

		model_name = 'FT-UNetFormer-{}'.format(variant)

		if variant == 'base':
			model = ft_unetformer.FTUNetFormer(num_classes=num_classes)
		else:
			raise ValueError('FT-UNetFormer only has \'base\' variant.')

		super().__init__(model_name=model_name, model=model)



#################### MODEL BUILDER ####################


model_classes = {
	'segformer': Segformer,
	'segformer_4ch': Segformer_4channels,
	'lawin': Lawin,
	'lawin_4ch': Lawin_4channels,
	'dc_mit': DC_MiT,
	'dc_mit_4ch': DC_MiT_4channels,
	'dc_swin': DC_Swin,
	'mit_unetformer': MiT_UNetFormer,
	'mit_unetformer_4ch': MiT_UNetFormer_4channels,
	'ft_unetformer': FT_UNetFormer
}


def modelBuilder(model_name: str, num_classes: int, weight_init: str) -> SegmentationModel:

	variant = model_name.split('_')[-1]

	assert weight_init == 'zero'\
		or weight_init == 'random'\
		or weight_init == 'red_combined'\
		or weight_init == 'shift_spectrum', 'weight_init must be \'zero\', \'random\', \'red_combined\' or \'shift_spectrum\''

	return model_classes[model_name[:-(len(variant)+1)]](variant, num_classes, weight_init)
