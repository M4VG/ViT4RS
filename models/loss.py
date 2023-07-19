import torch
from aux import expandLabels


##### custom loss function abstract class #####

class CustomLossFunction(torch.nn.Module):

	def __init__(self, num_classes: int) -> None:
		super().__init__()
		self.num_classes = num_classes
	
	def forward(self, logits: torch.tensor, labels: torch.tensor) -> torch.tensor:
		pass


##### concrete custom loss function classes #####


class CrossEntropyLossWrapper(CustomLossFunction):

	def __init__(self, num_classes: int, ignore_index = 255) -> None:
		super(CrossEntropyLossWrapper, self).__init__(num_classes)
		self.loss_fct = torch.nn.CrossEntropyLoss(ignore_index = ignore_index)

		# weighted CE loss
		# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		# pot_class_distr = [0.2957, 0.2569, 0.2261, 0.1551, 0.0179, 0.0483]
		# pot_class_weight = torch.Tensor([0.5636, 0.6488, 0.7371, 1.0746, 9.3110, 3.4507]).to(device)
		# self.loss_fct = torch.nn.CrossEntropyLoss(weight = pot_class_weight, ignore_index = ignore_index)
	
	def forward(self, logits: torch.tensor, labels: torch.tensor) -> torch.tensor:
		return self.loss_fct(logits, labels.to(logits.device))

		# labels = labels.unsqueeze(1).to(logits.device)
		# sm = torch.nn.Softmax2d()
		# pred = sm(logits)

		# x = torch.zeros(logits.size(0), 1, logits.size(2), logits.size(3))
		# for batch in range(logits.size(0)):
		# 	x[batch] = pred[batch].gather(0, labels[0])

		# x = torch.mean(-torch.log(x))

		# return x#, self.loss_fct(logits, labels.squeeze(1))


class BCELossWrapper(CustomLossFunction):

	def __init__(self, num_classes: int, ignore_index = 255) -> None:
		super(BCELossWrapper, self).__init__(num_classes)
		self.loss_fct = torch.nn.BCEWithLogitsLoss()
	
	def forward(self, logits: torch.tensor, labels: torch.tensor) -> torch.tensor:
		return self.loss_fct(logits, labels.unsqueeze(1).type(torch.float32).to(logits.device))


class FocalTverskyLoss(CustomLossFunction):

	def __init__(self, num_classes: int, smooth=1, alpha=0.7, gamma=0.8, ignore_index = 255):
		super(FocalTverskyLoss, self).__init__(num_classes)
		self.smooth = smooth
		self.alpha = alpha
		self.gamma = gamma
		self.ignore_index = ignore_index
	
	def forward(self, logits: torch.tensor, labels: torch.tensor) -> torch.tensor:
		y_pred = torch.softmax(logits, dim=1)
		y_true = expandLabels(labels, self.num_classes).to(logits.device)
		assert y_true.shape == y_pred.shape, 'Label and prediction shapes do not match'
		return self.focal_tversky(y_true, y_pred)

	def focal_tversky(self, y_pred, y_true):
		pt_1 = self.tversky_index(y_true, y_pred)
		return torch.pow((1 - pt_1), self.gamma)

	def tversky_index(self, y_pred, y_true):
		y_true = torch.flatten(y_true)
		y_pred = torch.flatten(y_pred)
		indxs = (y_true != self.ignore_index)
		true_pos = torch.sum(y_true[indxs] * y_pred[indxs])
		false_neg = torch.sum(y_true[indxs] * (1 - y_pred[indxs]))
		false_pos = torch.sum((1 - y_true[indxs]) * y_pred[indxs])
		return (true_pos + self.smooth) / (true_pos + self.alpha * false_neg + (1 - self.alpha) * false_pos + self.smooth)


class LogCoshDiceLoss(CustomLossFunction):

	def __init__(self, num_classes: int, smooth=1., ignore_index = 255):
		super(LogCoshDiceLoss, self).__init__(num_classes)
		self.dice_loss = DiceLoss(num_classes, smooth, ignore_index)

	def forward(self, logits: torch.tensor, labels: torch.tensor) -> torch.tensor:
		x = self.dice_loss(logits, labels)
		return torch.log((torch.exp(x) + torch.exp(-x)) / 2.0)


class DiceLoss(CustomLossFunction):

	def __init__(self, num_classes: int, smooth=1., ignore_index = 255):
		super(DiceLoss, self).__init__(num_classes)
		self.smooth = smooth
		self.ignore_index = ignore_index
	
	def forward(self, logits: torch.tensor, labels: torch.tensor) -> torch.tensor:
		y_pred = torch.softmax(logits, dim=1)
		y_true = expandLabels(labels, self.num_classes).to(logits.device)
		assert y_true.shape == y_pred.shape, 'Label and prediction shapes do not match'
		return self.dice_loss(y_true, y_pred)

	def dice_loss(self, y_pred, y_true):
		loss = 1 - self.generalized_dice_coefficient(y_true, y_pred)
		return loss
	
	def generalized_dice_coefficient(self, y_pred, y_true):
		y_true = torch.flatten(y_true)
		y_pred = torch.flatten(y_pred)
		indxs = (y_true != self.ignore_index)
		intersection = torch.sum(y_true[indxs] * y_pred[indxs])
		score = (2. * intersection + self.smooth) / (torch.sum(y_true[indxs]) + torch.sum(y_pred[indxs]) + self.smooth)
		return score


class CrossEntropyDiceLoss(CustomLossFunction):

	def __init__(self, num_classes: int, smooth=1., ignore_index = 255):
		super(CrossEntropyDiceLoss, self).__init__(num_classes)
		self.cross_entropy_loss = CrossEntropyLossWrapper(num_classes, ignore_index)
		self.dice_loss = DiceLoss(num_classes, smooth, ignore_index)
	
	def forward(self, logits: torch.tensor, labels: torch.tensor) -> torch.tensor:
		return self.cross_entropy_loss(logits, labels) + self.dice_loss(logits, labels)


class CrossEntropyFocalTverskyLoss(CustomLossFunction):

	def __init__(self, num_classes: int, smooth=1., ignore_index = 255):
		super(CrossEntropyFocalTverskyLoss, self).__init__(num_classes)
		self.cross_entropy_loss = CrossEntropyLossWrapper(num_classes, ignore_index)
		self.focal_tversky_loss = FocalTverskyLoss(num_classes, smooth)
	
	def forward(self, logits: torch.tensor, labels: torch.tensor) -> torch.tensor:
		return self.cross_entropy_loss(logits, labels) + self.focal_tversky_loss(logits, labels)


################## Feature Augmentation Loss ##################

class LogitsMSELoss(CustomLossFunction):

	def __init__(self, num_classes: int):
		super(LogitsMSELoss, self).__init__(num_classes)
		self.loss_fct = torch.nn.MSELoss()
	
	def forward(self, logits_original: torch.tensor, logits_augmented: torch.tensor) -> torch.tensor:
		return self.loss_fct(logits_augmented, logits_original)


class CrossEntropyMSELoss(CustomLossFunction):

	def __init__(self, num_classes: int, ignore_index = 255):
		super(CrossEntropyMSELoss, self).__init__(num_classes)
		self.cross_entropy_loss_1 = CrossEntropyLossWrapper(num_classes, ignore_index)
		self.cross_entropy_loss_2 = CrossEntropyLossWrapper(num_classes, ignore_index)
		self.logits_mse_loss = LogitsMSELoss(num_classes)
	
	def forward(self, logits_original: torch.tensor, logits_augmented: torch.tensor, labels: torch.tensor) -> torch.tensor:
		return self.cross_entropy_loss_1(logits_original, labels) + self.cross_entropy_loss_2(logits_augmented, labels) + self.logits_mse_loss(logits_original, logits_augmented)
	

class CrossEntropyDiceMSELoss(CustomLossFunction):

	def __init__(self, num_classes: int, ignore_index = 255):
		super(CrossEntropyDiceMSELoss, self).__init__(num_classes)
		self.cross_entropy_dice_loss = CrossEntropyDiceLoss(num_classes=num_classes, ignore_index=ignore_index)
		self.logits_mse_loss = LogitsMSELoss(num_classes)
	
	def forward(self, logits_original: torch.tensor, logits_augmented: torch.tensor, labels: torch.tensor) -> torch.tensor:
		return self.cross_entropy_dice_loss(logits_original, labels) + self.logits_mse_loss(logits_original, logits_augmented)



#################### LOSS FUNCTION BUILDER ####################


function_classes = {
	'cross_entropy': CrossEntropyLossWrapper,
	'focal_tversky': FocalTverskyLoss,
	'log_cosh_dice': LogCoshDiceLoss,
	'binary_cross_entropy': BCELossWrapper,
	'dice': DiceLoss,
	'cross_entropy_dice': CrossEntropyDiceLoss,
	'cross_entropy_focal_tversky': CrossEntropyFocalTverskyLoss,
	'cross_entropy_mse': CrossEntropyMSELoss,
	'cross_entropy_dice_mse': CrossEntropyDiceMSELoss
}


def lossFunctionBuilder(function_name: str, num_classes: int) -> CustomLossFunction:
	return function_classes[function_name](num_classes)	# NOTE: does not work with args atm
