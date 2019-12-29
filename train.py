"""
A script for multi-tier conditional RNN training
"""
import os
import time
from datetime import datetime

import torch
import network.config as config
from network.model import CondRNN
from dataloader.dataloader import DataLoader
from utils.myUtils import time_taken



class Trainer:
	def __init__(self, args):
		self.args = args
		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

		if 'audio' in args.generate:
			self.model = CondRNN(args.cond_size+args.gen_size, args.hidden_size,
								args.mulaw_channels, args.n_layers, self.device,
								lr=args.lr,paramonly=False,onehot=args.onehot)

		else:
			self.model = CondRNN(args.cond_size+args.gen_size, args.hidden_size,
								args.gen_size, args.n_layers, self.device,
								lr=args.lr,paramonly=True,onehot=args.onehot)

		self.data_loader = DataLoader(args.data_dir, args.sample_rate, args.seq_len, args.stride, 
									paramdir=args.param_dir, prop=args.prop, generate=args.generate,
									mulaw_channels=args.mulaw_channels,
									batch_size=args.batch_size,
									onehot=args.onehot)
		
		self.epoch_size = len(self.data_loader)

		if args.model_dir: #to resume from checkpoint
			self.model.load(args.model_dir, args.step)
			print("Resuming training from step",args.step)


	def infinite_batch(self):
		while True:
			for inputs, targets in self.data_loader:
				yield inputs, targets


	def run(self):
		print("Total steps in 1 epoch is", self.epoch_size)
		print("Starting training at: {:%Y-%m-%d %H:%M:%S}".format(datetime.now()))
		since = time.time()
		for step,(inputs, targets) in enumerate(self.infinite_batch(), start=1+args.step):
			try:
				inputs, targets = inputs.to(self.device), targets.to(self.device)
				loss = self.model.train(inputs, targets, self.args.tfr, self.args.temp)
				time_elapsed = time.time() - since
				print('{0} [{1}/{2}] loss: {3}'.format(time_taken(time_elapsed), step, self.args.num_steps+args.step, loss))

				if step >= (self.args.num_steps+args.step):
					break
				if step % self.args.checkpoint == 0:
					self.model.save(self.args.new_model_dir, step)

			except KeyboardInterrupt:
				break

		self.model.save(self.args.new_model_dir, step)


def prepare_output_dir(args):
	args.log_dir = os.path.join('output', args.output_dir, 'log')
	args.new_model_dir = os.path.join('output', args.output_dir, 'model')
	args.test_output_dir = os.path.join('output', args.output_dir, 'test')

	os.makedirs(args.log_dir, exist_ok=True)
	os.makedirs(args.new_model_dir, exist_ok=True)
	os.makedirs(args.test_output_dir, exist_ok=True)

	with open(args.log_dir+"/args.txt", "w") as text_file:
		print("{}".format(args), file=text_file)


if __name__ == '__main__':
	args = config.parse_args()
	print(args)

	prepare_output_dir(args)

	trainer = Trainer(args)

	trainer.run()
