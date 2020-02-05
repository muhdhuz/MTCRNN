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
		Outputs - logits to be fed into a softmax or cross entropy loss

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
	
	def forward(self, input, hidden, batch_size=1, **kwargs):     
		h1 = F.relu(self.i2d(input))
		h2 = self.d2h(h1)      
		h_out, hidden = self.gru(h2.view(batch_size,1,-1), hidden)  #hidden shape = (n_layers*n_directions,batch_size,hidden_size)
		output = self.h2o(h_out.view(batch_size,-1))                #h_out shape = (timestep,batch_size,hidden_size*n_directions)
		return output, hidden

	# initialize hiddens for each minibatch
	def init_hidden(self,batch_size=1):
		return torch.zeros(self.n_layers, batch_size, self.hidden_size, dtype=torch.float)


class RnnBlockNorm(nn.Module):
	def __init__(self, input_size, hidden_size, output_size, n_layers):
		"""
		A stack of RNN layers with dense layers at input and output. 
		Outputs - Predicts a mean and variance for a normal distribution that can be sampled using sample_normal.
		
		input_size: if input layer - combined size of generated+conditional vectors
		hidden_size: no. of hidden nodes for each GRU layer
		output_size: size of output, normally equal to no. of generated parameters
		n_layers: no. of stacked GRU layers
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
	

	def sample_normal(self, mu, logvar, beta):
		"""sample from output layer consisting of parameters mean and variance to a normal distribution"""
		#sigmoid_out = torch.nn.functional.sigmoid(output)
		#out = torch.normal(mean=sigmoid_out[:,:output.shape[1]//2], std=sigmoid_out[:,output.shape[1]//2:]) 

		std = F.softplus(logvar,beta=beta)  #1/beta*log(1+ exp(x/beta))
		eps = torch.randn_like(std)
		out = mu + eps*std
		return out

	def forward(self, input, hidden, batch_size=1, **kwargs): 
		if 'beta' in kwargs.keys():
			beta = kwargs['beta']
		else:
			beta = 1.0

		h1 = F.relu(self.i2d(input))
		h2 = self.d2h(h1)
		h_out, hidden = self.gru(h2.view(batch_size,1,-1), hidden)  #hidden shape = (n_layers*n_directions,batch_size,hidden_size)
		mu = self.h2n(h_out.view(batch_size,-1))                #h_out shape = (timestep,batch_size,hidden_size*n_directions)
		sig = self.h2v(h_out.view(batch_size,-1)) 
		output = self.sample_normal(mu,sig,beta) 
		return output, hidden

	# initialize hiddens for each minibatch
	def init_hidden(self,batch_size=1):
		return torch.zeros(self.n_layers, batch_size, self.hidden_size, dtype=torch.float)