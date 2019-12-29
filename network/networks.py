"""
Base network modules

@muhammad huzaifah 01/11/2019 
"""

import torch 
import torch.nn as nn
import torch.nn.functional as F

class RnnBlock(nn.Module):
	def __init__(self, input_size, hidden_size, output_size, n_layers):
		"""
		A stack of RNN layers with dense layers at input and output
		cond_size: size of conditional vector
		input_size: if input layer - no. of input audio channels+conditional vectors, 
					for one-hot audio=mu-law channels + cond vector size
		hidden_size: no. of hidden nodes for each GRU layer
		output_size: size of output, normally=256 for 8-bit mu-law if final layer
		n_layers: no of stacked GRU layers
		"""
		super(RnnBlock, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.n_layers = n_layers

		self.i2d = nn.Linear(self.input_size, self.input_size*3)
		self.d2h = nn.Linear(self.input_size*3, hidden_size)
		self.gru = nn.GRU(hidden_size, hidden_size, n_layers, batch_first=True)
		self.h2o = nn.Linear(hidden_size, output_size)
	

	# input and cv are each one sequence element 
	def forward(self, input, hidden, batch_size=1):     
		h1 = F.relu(self.i2d(input))
		h2 = self.d2h(h1)      
		h_out, hidden = self.gru(h2.view(batch_size,1,-1), hidden)  #hidden shape = (n_layers*n_directions,batch_size,hidden_size)
		output = self.h2o(h_out.view(batch_size,-1))                #h_out shape = (timestep,batch_size,hidden_size*n_directions)
		return output, hidden

	# initialize hiddens for each minibatch
	def init_hidden(self,batch_size=1):
		return torch.zeros(self.n_layers, batch_size, self.hidden_size, dtype=torch.float)#, device=device)


class RnnStack(nn.Module):
	def __init__(self, stack_size):
		super(RnnStack, self).__init__()
		self.stack_size = stack_size


class RnnBlockNorm(nn.Module):
	def __init__(self, input_size, hidden_size, output_size, n_layers):
		"""
		A stack of RNN layers with dense layers at input and output
		cond_size: size of conditional vector
		input_size: if input layer - no. of input audio channels+conditional vectors, 
					for one-hot audio=mu-law channels + cond vector size
		hidden_size: no. of hidden nodes for each GRU layer
		output_size: size of output, normally=256 for 8-bit mu-law if final layer
		n_layers: no of stacked GRU layers
		"""
		super(RnnBlockNorm, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.n_layers = n_layers

		self.i2d = nn.Linear(self.input_size, self.input_size*3)
		self.d2h = nn.Linear(self.input_size*3, hidden_size)
		self.gru = nn.GRU(hidden_size, hidden_size, n_layers, batch_first=True)
		self.h2n = nn.Linear(hidden_size, output_size)
		self.h2v = nn.Linear(hidden_size, output_size)
	

	def sample_normal(self, mu, logvar):
		"""sample from output layer consisting of parameters mean and variance to a normal distribution"""
		#print("OUT",output.shape)
		#print("OUT",output)
		#print("MEAN",output[:,:output.shape[1]//2])
		#print("STD",output[:,output.shape[1]//2:])
		#sigmoid_out = torch.nn.functional.sigmoid(output)
		#out = torch.normal(mean=sigmoid_out[:,:output.shape[1]//2], std=sigmoid_out[:,output.shape[1]//2:]) 
		#print("NORMAL",out.shape)
		#print("NORMAL",out)
		#print("MEAN",mu)
		std = F.softplus(logvar)  #log(1+ exp(x))
		eps = torch.randn_like(std)
		#print("std",std)
		out = mu + eps*std
		#print("out",out) 
		return out

	# input and cv are each one sequence element 
	def forward(self, input, hidden, batch_size=1):     
		h1 = F.relu(self.i2d(input))
		h2 = self.d2h(h1)
		h_out, hidden = self.gru(h2.view(batch_size,1,-1), hidden)  #hidden shape = (n_layers*n_directions,batch_size,hidden_size)
		mu = self.h2n(h_out.view(batch_size,-1))                #h_out shape = (timestep,batch_size,hidden_size*n_directions)
		sig = self.h2v(h_out.view(batch_size,-1)) 
		output = self.sample_normal(mu,sig) 
		return output, hidden

	# initialize hiddens for each minibatch
	def init_hidden(self,batch_size=1):
		return torch.zeros(self.n_layers, batch_size, self.hidden_size, dtype=torch.float)#, device=device)