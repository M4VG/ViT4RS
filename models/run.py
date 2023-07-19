import yaml
import argparse
import pprint
import torch

import dataset
import model
import train
import eval

if __name__ == '__main__':

	# parse args
	parser = argparse.ArgumentParser()
	parser.add_argument("config_file", help="path to config file")
	args = parser.parse_args()

	# read config file
	with open(args.config_file, 'r') as file:
		config = yaml.safe_load(file)
	
	# print config
	pprint.pprint(config, compact=True, sort_dicts=False)
	print('\n')

	# create dataset
	dataset = dataset.datasetBuilder(config['dataset'])

	# create model
	model = model.modelBuilder(config['model']['name'], dataset.numClasses(), config['model']['weight_init'])

	# move model to GPU
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model.to(device)
	# print('> Current CUDA device: {}\n'.format(torch.cuda.current_device()))

	# print summary
	print('> Model:', model.name())
	print('> Dataset:', dataset.name())

	# wandb setup
	run_id = None

	# train
	if config['general']['do_train']:

		print('\n>>> Starting training\n')

		run_id = train.train(
			model,
			device,
			dataset,
			config
		)
	
	# eval
	if config['general']['do_eval']:

		print('\n>>> Starting evaluation\n')

		eval.eval(
			model,
			device,
			dataset,
			config,
			run_id
		)
	