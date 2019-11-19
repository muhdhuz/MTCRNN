"""
Main model of WaveNet
Calculate loss and optimizing
"""
import os

import torch
import torch.optim

from network.networks import RnnBlock as RnnModule
import torchvision.transforms as transform
import dataloader.transforms as tr



class CondRNN:
	def __init__(self, input_size, hidden_size, output_size, n_layers, device, lr=0.002, paramonly=False, onehot=False):
		self.device = device
		self.net = RnnModule(input_size, hidden_size, output_size, n_layers).to(self.device)
		print(self.net)

		self.lr = lr
		self.loss = self._loss(paramonly)
		self.optimizer = self._optimizer()
		self.output_size = output_size
		self.paramonly = paramonly
		self.onehot = onehot

	def _loss(self,paramonly):
		if paramonly:
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

		return loss, hidden

	def train(self, inputs, targets):
		hidden = self.net.init_hidden(inputs.shape[0]).to(self.device)
		self.optimizer.zero_grad()
		sequence_loss = 0.

		for timestep in range(inputs.shape[1]):
			loss, hidden = self._train_step(inputs[:,timestep,:], torch.squeeze(targets[:,timestep],1), hidden)
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

	def sample(self, output):
		"""sample from output layer"""
		log_output = torch.nn.functional.log_softmax(output,dim=1)
		topv, topi = log_output.topk(1) #output topi is a mu-law index
		mulaw_output = topi.detach().cpu().numpy()

		#encode for next step
		if self.onehot:
			mulaw_to_onehot = transform.Compose([tr.onehotEncode(self.output_size),tr.array2tensor(torch.FloatTensor)])
		else:
			mulaw_output_norm = mulaw_output/self.output_size
			mulaw_to_onehot = tr.array2tensor(torch.FloatTensor)
		next_input = mulaw_to_onehot(mulaw_output_norm)
		
		#decode to get audio sample
		predicted_sample = tr.mulawDecode(self.output_size)(mulaw_output)
		return next_input, predicted_sample

	def generate(self, inputs, hidden):
		"""
		Generate 1 time
		:param inputs: Tensor[batch, timestep, channels]
		:return: Tensor[batch, timestep, channels]
		"""

		outputs, hidden = self.net(inputs,hidden,inputs.shape[0])
		if self.paramonly:
			next_input = outputs.detach()
			predicted_sample = outputs.detach().cpu().numpy()
		else:
			next_input, predicted_sample = self.sample(outputs)

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

