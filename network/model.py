"""
Main model of MTCRNN
Calculate loss and optimizing
Interfaces train/generate to the network
"""
import os
import numpy as np

import torch
import torch.optim

from network.networks import RnnBlock,RnnBlockNorm
import torchvision.transforms as transform
import dataloader.transforms as tr



class CondRNN:
	def __init__(self, input_size, hidden_size, output_size, n_layers, device, lr=0.002, paramonly=False, onehot=False):
		self.device = device
		self.lr = lr
		self.paramonly = paramonly
		self.output_size = output_size
		self.onehot = onehot
		
		if self.paramonly:
			self.net = RnnBlockNorm(input_size, hidden_size, output_size, n_layers).to(self.device)
		else:
			self.net = RnnBlock(input_size, hidden_size, output_size, n_layers).to(self.device)
		print(self.net)

		self.loss = self._loss()
		self.optimizer = self._optimizer()


	def _loss(self):
		if self.paramonly:
			loss = torch.nn.MSELoss()
		else:
			loss = torch.nn.CrossEntropyLoss()
		return loss

	def _optimizer(self):
		return torch.optim.Adam(self.net.parameters(), lr=self.lr)

	def _train_step(self, input, target, hidden):
		"""
		Train 1 time
		:param inputs: Tensor[batch, timestep, channels]
		:param targets: Torch tensor [batch, channels, timestep]
		:return: float loss
		"""
		outputs, hidden = self.net(input,hidden,input.shape[0])
		loss = self.loss(outputs,target)

		return loss, hidden, outputs

	def train(self, inputs, targets, teacher_forcing_ratio, temperature):
		hidden = self.net.init_hidden(inputs.shape[0]).to(self.device)
		self.optimizer.zero_grad()
		sequence_loss = 0.
		input = inputs[:,0,:]

		for timestep in range(inputs.shape[1]):
			use_teacher_forcing = True if np.random.random() < teacher_forcing_ratio else False
			
			if use_teacher_forcing:
				loss, hidden, _ = self._train_step(input, torch.squeeze(targets[:,timestep],1), hidden)
				if timestep+1 < inputs.shape[1]:
					input = inputs[:,timestep+1,:]

			else:
				loss, hidden, output = self._train_step(input, torch.squeeze(targets[:,timestep],1), hidden)
				if timestep+1 < inputs.shape[1]:
					if self.paramonly:
						input = torch.cat((output.detach(),inputs[:,timestep+1,output.shape[1]:]),1)
					else:
						next_sample, _ = self.sample(output, temperature)
						input = torch.cat((next_sample.to(self.device),inputs[:,timestep+1,next_sample.shape[1]:]),1)

			sequence_loss += loss
		
		sequence_loss.backward()
		self.optimizer.step()
		return sequence_loss/inputs.shape[1] #return average sample loss 

	def build_hidden_state(self, inputs):
		hidden = self.net.init_hidden(inputs.shape[0]).to(self.device)
		
		if inputs.shape[1] > 1: #if priming with something with len>1
			for timestep in range(inputs.shape[1]-1):
				_, hidden = self.net(inputs[:,timestep,:],hidden,inputs.shape[0])  #build up hidden state
		return inputs[:,-1,:], hidden  #feed the last value as the initial value of the actual generation        

	def sample(self, output, temperature):
		"""sample from output layer"""
		log_output = torch.nn.functional.log_softmax(output,dim=1)

		#topv, topi = log_output.topk(1) #output topi is a mu-law index
		#mulaw_output2 = topi.detach().cpu().numpy()

		out_weights = log_output.div(temperature).exp()
		idx = torch.multinomial(out_weights, 1)
		mulaw_output = idx.detach().cpu().numpy()

		#encode for next step
		if self.onehot:
			mulaw_to_onehot = transform.Compose([tr.onehotEncode(self.output_size),tr.array2tensor(torch.FloatTensor)])
		else:
			mulaw_output_norm = mulaw_output/self.output_size #norm by no. of quantization channel -> [0,1]
			mulaw_to_onehot = tr.array2tensor(torch.FloatTensor)
		next_input = mulaw_to_onehot(mulaw_output_norm)
		
		#decode to get audio sample
		predicted_sample = tr.mulawDecode(self.output_size)(mulaw_output)
		return next_input, predicted_sample

	def generate(self, inputs, hidden, temperature):
		"""
		Generate 1 time
		:param inputs: Tensor[batch, timestep, channels]
		:return: Tensor[batch, timestep, channels]
		"""

		outputs, hidden = self.net(inputs,hidden,inputs.shape[0])
		if self.paramonly:
			next_input = outputs.detach().cpu()
			predicted_sample = outputs.detach().cpu().numpy()
		else:
			next_input, predicted_sample = self.sample(outputs,temperature)

		return next_input, predicted_sample, hidden

	@staticmethod
	def get_model_path(model_dir, step=0):
		basename = 'model'

		if step:
			return os.path.join(model_dir, '{0}_{1}.pkl'.format(basename, step))
		else:
			return os.path.join(model_dir, '{0}.pkl'.format(basename))

	def load(self, model_dir, step=0):
		"""
		Load pre-trained model
		:param model_dir:
		:param step:
		:return:
		"""
		print("Loading model from {0}".format(model_dir))

		model_path = self.get_model_path(model_dir, step)
		self.net.load_state_dict(torch.load(model_path, map_location=self.device))

	def save(self, model_dir, step=0):
		print("Saving model into {0}".format(model_dir))

		model_path = self.get_model_path(model_dir, step)
		torch.save(self.net.state_dict(), model_path)

