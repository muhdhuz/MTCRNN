"""
Main model of MTCRNN
Calculate loss and optimizing
Interfaces train/generate to the network
@muhammad huzaifah 2020/02/26 
"""
import os
import numpy as np

import torch
import torch.optim

import network.networks as nets
import torchvision.transforms as transform
import network.sampling as samp



class CondRNN:
	def __init__(self, input_size, hidden_size, output_size, n_layers, device, lr=0.005, paramonly=False, onehot=False, ntype=0, plstm=False):
		self.device = device
		self.lr = lr
		self.paramonly = paramonly
		self.output_size = output_size
		self.onehot = onehot
		self.ntype = ntype
		self.n_layers = n_layers
		self.plstm = plstm

		if self.paramonly:
			if ntype == 0:
				self.net = nets.BaseRnnBlock(input_size, hidden_size, output_size, n_layers, plstm, paramonly, 0).to(self.device)
			elif ntype == 1: 
				self.net = nets.SeperateMeanVar(input_size, hidden_size, output_size, n_layers, 1).to(self.device)
			elif ntype == 2: 
				self.net = nets.SeperateMeanVar(input_size, hidden_size, output_size, n_layers, 2).to(self.device)			
			elif ntype == 3: 
				self.net = nets.SeperateVariables(input_size, hidden_size, output_size, n_layers, 3).to(self.device)	
			elif ntype == 4: 
				self.net = nets.SeperateVariables(input_size, hidden_size, output_size, n_layers, 4).to(self.device)	
			elif ntype == 5:
				self.net = nets.MDN(input_size, hidden_size, output_size, n_layers).to(self.device)
		else:
			if ntype == 0:			
				self.net = nets.BaseRnnBlock(input_size, hidden_size, output_size, n_layers, plstm, paramonly, 0).to(self.device)
			elif ntype == 6:
				self.net = nets.BaseRnnBlock(input_size, hidden_size, output_size, n_layers, plstm, paramonly, 6).to(self.device)
		print(self.net)
		#for name, param in self.net.named_parameters():
		#	print(name, ':', param.requires_grad)
		#print(self.net.state_dict())

		self.loss = self._loss()
		self.optimizer = self._optimizer()


	def _loss(self):
		if self.paramonly:
			loss = torch.nn.MSELoss() #torch.nn.L1Loss() #
		else:
			loss = torch.nn.CrossEntropyLoss()
		return loss

	def _mdn_loss(self,prob):
		nll = -torch.log(torch.sum(prob, dim=1))
		#print("nll",nll)
		return torch.mean(nll)

	def _optimizer(self):
		return torch.optim.Adam(self.net.parameters(), lr=self.lr)

	def _train_step(self, input, target, hidden, *args, **kwargs):
		"""
		Train 1 time
		:param inputs: Tensor[batch, timestep, channels]
		:param targets: Torch tensor [batch, channels, timestep]
		:return: float loss
		"""
		outputs, hidden = self.net(input,hidden,input.shape[0],times=kwargs['times'],device=self.device)
		if self.paramonly:
			if self.ntype in (0,1,2,3,4):
				outputs = samp.sample_normal(*outputs) #out=(mu.sig,beta)
				loss = self.loss(outputs,target)
			elif self.ntype == 5:
				prob = samp.gaussian_probability(outputs[0],outputs[1],target)
				outputs = outputs[2] * prob  #note these outputs are not the actual target values
				loss = self._mdn_loss(outputs)
				#print("L",loss,loss.shape)
		else:
			loss = self.loss(outputs,target)

		return loss, hidden, outputs

	def _get_init_hidden(self, batch_size):
		hidden = self.net.init_hidden(batch_size).to(self.device)
		if self.plstm:
			hidden = [(hidden, hidden) for _ in range(self.n_layers)]
		if self.ntype == 1 or self.ntype ==2:
			hidden = (hidden,hidden,hidden)
		elif self.ntype == 3 or self.ntype ==4:
			hidden = [hidden for _ in range(self.output_size+1)]

		return hidden

	def train(self, inputs, targets, teacher_forcing_ratio, temperature):
		hidden = self._get_init_hidden(inputs.shape[0])

		self.optimizer.zero_grad()
		sequence_loss = 0.
		input = inputs[:,0,:]
			
		onetime = torch.arange(inputs.shape[1],dtype=torch.float) #plstm things
		times = onetime.repeat(inputs.shape[0],1).to(self.device)

		for timestep in range(inputs.shape[1]):
			use_teacher_forcing = True if np.random.random() < teacher_forcing_ratio else False
			
			if use_teacher_forcing:
				loss, hidden, _ = self._train_step(input, torch.squeeze(targets[:,timestep],1), hidden, times=times[:,timestep])
				if timestep+1 < inputs.shape[1]:
					input = inputs[:,timestep+1,:]

			else:
				loss, hidden, output = self._train_step(input, torch.squeeze(targets[:,timestep,:],1), hidden, times=times[:,timestep])
				if timestep+1 < inputs.shape[1]:
					if self.paramonly:
						input = torch.cat((output.detach(),inputs[:,timestep+1,output.shape[1]:]),1)
					else:
						next_sample, _, _ = samp.sample_multinomial(output, temperature, self.output_size, self.onehot)
						input = torch.cat((next_sample.to(self.device),inputs[:,timestep+1,next_sample.shape[1]:]),1)

			sequence_loss += loss
		
		sequence_loss.backward()
		self.optimizer.step()
		return sequence_loss/inputs.shape[1] #return average sample loss 

	def build_hidden_state(self, inputs, times):
		"""build hidden state during priming"""
		hidden = self._get_init_hidden(inputs.shape[0])
	
		if inputs.shape[1] > 1: #if priming with something with len>1
			for timestep in range(inputs.shape[1]-1):
				_, hidden = self.net(inputs[:,timestep,:],hidden,inputs.shape[0],times=times[:,timestep])  #build up hidden state
		return inputs[:,-1,:], hidden  #feed the last value as the initial value of the actual generation        

	def generate(self, inputs, hidden, temperature, time):
		"""
		Generate 1 time
		:param inputs: Tensor[batch, timestep, channels]
		:return: Tensor[batch, timestep, channels]
		"""
		outputs, hidden = self.net(inputs,hidden,inputs.shape[0],beta=temperature,times=time,device=self.device)
		if self.paramonly:
			mu, sig = self._process_raw(outputs)
			if self.ntype in (0,1,2,3,4):
				outputs = samp.sample_normal(*outputs) #out=(mu,sig,beta)
			elif self.ntype == 5:
				outputs = samp.sample_mixturedensity(*outputs) #out=(mu,sigma,pi)

			next_input = outputs.detach().cpu()
			predicted_sample = outputs.detach().cpu().numpy()
			p_cut = []
		else:
			next_input, predicted_sample, p_cut = samp.sample_multinomial(outputs,temperature, self.output_size, self.onehot, cutoff=0.9)
			mu, sig = [], []

		return next_input, predicted_sample, hidden, mu, sig, p_cut

	def _process_raw(self,tensor):
		mu, sig, _ = tensor
		return mu.detach().cpu().numpy(), sig.detach().cpu().numpy()

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
		model_dict = torch.load(model_path, map_location=self.device)
		self.net.load_state_dict(model_dict)

	def save(self, model_dir, step=0):
		print("Saving model into {0}".format(model_dir))

		model_path = self.get_model_path(model_dir, step)
		torch.save(self.net.state_dict(), model_path)

