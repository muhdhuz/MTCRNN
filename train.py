"""
A script for multi-tier conditional RNN training
"""
import os

import torch
import network.config as config
from network.model import CondRNN
from dataloader.dataloader import DataLoader


class Trainer:
	def __init__(self, args):
		self.args = args
		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

		self.model = CondRNN(args.cond_size, args.hidden_size,
							   args.mulaw_channels, args.n_layers, self.device,
							   lr=args.lr)

		self.data_loader = DataLoader(args.data_dir, args.sample_rate, args.seq_len, args.stride, 
									paramdir=args.param_dir, prop=args.prop,
									mulaw_channels=args.mulaw_channels,
									batch_size=args.batch_size)
		
		self.epoch_size = len(self.data_loader)

	def infinite_batch(self):
		while True:
			for inputs, targets in self.data_loader:
				yield inputs, targets

	def run(self):
		print("total steps in 1 epoch is", self.epoch_size)
		for step,(inputs, targets) in enumerate(self.infinite_batch(), start=1):
			try:
				inputs, targets = inputs.to(self.device), targets.to(self.device)
				print(inputs.shape)
				print(inputs)
				#print(targets.shape)
				loss = self.model.train(inputs, targets)

				print('[{0}/{1}] loss: {2}'.format(step, self.args.num_steps, loss))

				if step >= self.args.num_steps:
					break
				if step % self.args.checkpoint == 0:
					self.model.save(self.args.model_dir, step)

			except KeyboardInterrupt:
				break

		self.model.save(self.args.model_dir, step)



def prepare_output_dir(args):
	args.log_dir = os.path.join(args.output_dir, 'log')
	args.model_dir = os.path.join(args.output_dir, 'model')
	args.test_output_dir = os.path.join(args.output_dir, 'test')

	os.makedirs(args.log_dir, exist_ok=True)
	os.makedirs(args.model_dir, exist_ok=True)
	os.makedirs(args.test_output_dir, exist_ok=True)


if __name__ == '__main__':
	args = config.parse_args()
	print(args)

	prepare_output_dir(args)

	trainer = Trainer(args)

	trainer.run()
