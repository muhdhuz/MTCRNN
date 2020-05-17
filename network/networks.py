"""
Base network modules

@muhammad huzaifah 2020/02/26 
"""

import math
import torch 
import torch.nn as nn
import torch.nn.functional as F
from network.phased_lstm import PhasedLSTM

#************************
#common layers
#************************
class RnnMeanVar(nn.Module):
	def __init__(self, input_size, hidden_size, output_size, n_layers):
		"""
		rnn layer (GRU) -> linear - mu
						-> linear - variance
		"""
		super(RnnMeanVar, self).__init__()
		self.rec = nn.GRU(input_size, hidden_size, n_layers, batch_first=True)
		self.h2n = nn.Linear(hidden_size, output_size)
		self.h2v = nn.Linear(hidden_size, output_size)
	
	def forward(self, input, hidden, batch_size):
		h_out, hidden = self.rec(input,hidden)
		mu = self.h2n(h_out.view(batch_size,-1)).squeeze(1)
		sig = self.h2v(h_out.view(batch_size,-1)).squeeze(1)				
		return mu, sig, hidden



class BaseRnnBlock(nn.Module):
	def __init__(self, input_size, hidden_size, output_size, n_layers, plstm, paramonly, *args, **kwargs):
		"""
		config net type 0 
		This is the baseline model and the simplest.
		A stack of RNN layers with dense layers at input and output

		Outputs - audio: logits to be fed into a softmax or cross entropy loss
				  parameters: Predicts a mean and variance for a normal distribution that can be sampled using sample_normal.

		input_size: if input layer - combined size of generated+conditional vectors, 
					for one-hot audio=mu-law channels + cond vector size
		hidden_size: no. of hidden nodes for each GRU layer
		output_size: size of output, normally equal to no. of generated parameters
					 for audio normally=256 for 8-bit mu-law
		n_layers: no of stacked GRU layers
		"""
		super(BaseRnnBlock, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.n_layers = n_layers
		self.plstm = plstm
		self.paramonly = paramonly
		self.ntype = args[0]
		up_factor = 3

		if self.ntype == 6:	
			self.i2d = nn.Linear(self.input_size, self.input_size*up_factor)  #change this to sequential after audio model retrained
			self.d2h = nn.Linear(self.input_size*up_factor, hidden_size)
		else:
			self.i2d = nn.Linear(input_size, hidden_size)

		if plstm:
			self.plstm_layer = PhasedLSTM(hidden_size, hidden_size, n_layers)
		else:
			self.gru = nn.GRU(hidden_size, hidden_size, n_layers, batch_first=True)
		if paramonly:
			self.h2n = nn.Linear(hidden_size, output_size)
			self.h2v = nn.Linear(hidden_size, output_size)
		else:
			self.h2o = nn.Linear(hidden_size, output_size)
	
	def forward(self, input, hidden, batch_size=1, **kwargs):
		if 'beta' in kwargs.keys(): #beta param to control temperature for sampling
			beta = kwargs['beta']
		else:
			beta = 1.0

		h1 = F.relu(self.i2d(input)) 
		if self.ntype == 6:
			h1 = self.d2h(h1)      
		if self.plstm:
			h_out, hidden = self.plstm_layer(h1.view(batch_size,-1),kwargs['times'],hidden)  
		else:
			h_out, hidden = self.gru(h1.view(batch_size,1,-1),hidden)  #hidden shape = (n_layers*n_directions,batch_size,hidden_size)
		if self.paramonly:                   							   #h_out shape = (timestep,batch_size,hidden_size*n_directions)
			mu = self.h2n(h_out.view(batch_size,-1))
			sig = self.h2v(h_out.view(batch_size,-1))			
			output = (mu,sig,beta)
		else:
			output = self.h2o(h_out.view(batch_size,-1))
		
		return output, hidden

	# initialize hiddens for each minibatch
	def init_hidden(self,batch_size=1):
		if self.plstm:
			return torch.zeros(batch_size, self.hidden_size, dtype=torch.float)
		else:
			return torch.zeros(self.n_layers, batch_size, self.hidden_size, dtype=torch.float)


class SeperateMeanVar(nn.Module):
	def __init__(self, input_size, hidden_size, output_size, n_layers, *args, **kwargs):
		"""
		config net type 1 (no additional common GRU layer) or type 2 (additional GRU layer)
		Separate RNN pathways for mean and variances
		Outputs - Predicts a mean and variance for a normal distribution that can be sampled using sample_normal.
		
		input_size: if input layer - combined size of generated+conditional vectors
		hidden_size: no. of hidden nodes for each GRU layer
		output_size: size of output, normally equal to no. of generated parameters
		n_layers: no. of stacked GRU layers
		"""
		super(SeperateMeanVar, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.n_layers = n_layers
		self.ntype = args[0]

		self.i2d = nn.Linear(self.input_size, hidden_size)
		if self.ntype == 2:
			self.gru = nn.GRU(hidden_size, hidden_size, n_layers, batch_first=True)
		
		self.mu1 = nn.GRU(hidden_size, hidden_size, n_layers, batch_first=True)
		self.mu2 = nn.Linear(hidden_size, output_size)

		self.sigma1 = nn.GRU(hidden_size, hidden_size, n_layers, batch_first=True)
		self.sigma2 = nn.Linear(hidden_size, output_size)
	
	def forward(self, input, hidden, batch_size=1, **kwargs): 
		if 'beta' in kwargs.keys():
			beta = kwargs['beta']
		else:
			beta = 1.0

		h1 = F.relu(self.i2d(input))
		if self.ntype == 2:
			h1, hidden1 = self.gru(h1.view(batch_size,1,-1),hidden[0])
		else:
			hidden1 = None

		mu1out, hidden2 = self.mu1(h1.view(batch_size,1,-1), hidden[1]) 
		mu = self.mu2(mu1out.view(batch_size,-1)) 
		
		sigma1out, hidden3 = self.sigma1(h1.view(batch_size,1,-1), hidden[2])
		sigma = self.sigma2(sigma1out.view(batch_size,-1))
		
		output = (mu,sigma,beta) 
		
		return output, (hidden1,hidden2,hidden3)

	def init_hidden(self,batch_size=1):
		return torch.zeros(self.n_layers, batch_size, self.hidden_size, dtype=torch.float)


class SeperateVariables(nn.Module):
	def __init__(self, input_size, hidden_size, output_size, n_layers, *args, **kwargs):
		"""
		config net type 3 (no additional common GRU layer) or type 4 (additional GRU layer)
		Separate RNN pathways for each generated parameter
		Outputs - Predicts a mean and variance for a normal distribution that can be sampled using sample_normal.
		
		input_size: if input layer - combined size of generated+conditional vectors
		hidden_size: no. of hidden nodes for each GRU layer
		output_size: size of output, normally equal to no. of generated parameters
		n_layers: no. of stacked GRU layers
		"""
		super(SeperateVariables, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.n_layers = n_layers
		self.ntype = args[0]

		self.i2d = nn.Linear(self.input_size, hidden_size)
		if self.ntype == 4:
			self.gru = nn.GRU(hidden_size, hidden_size, n_layers, batch_first=True)
		
		self.out_pro = nn.ModuleList([RnnMeanVar(hidden_size,hidden_size,1,n_layers) for i in range(output_size)])

	
	def forward(self, input, hidden, batch_size=1, **kwargs): 
		if 'beta' in kwargs.keys():
			beta = kwargs['beta']
		else:
			beta = 1.0
		device = kwargs['device']
		mu = torch.empty(batch_size,self.output_size).to(device)
		sigma = torch.empty(batch_size,self.output_size).to(device)
		hidden_list = []

		h1 = F.relu(self.i2d(input))
		if self.ntype == 4:
			h1, hidden1 = self.gru(h1.view(batch_size,1,-1),hidden[0])	
		else:
			hidden1 = None
		hidden_list.append(hidden1)

		h1 = h1.view(batch_size,1,-1)
		for i, layer in enumerate(self.out_pro):
			mu[:,i], sigma[:,i], hidden_inner = layer(h1,hidden[i+1],batch_size)
			hidden_list.append(hidden_inner)
		
		output = (mu,sigma,beta) 
		
		return output, hidden_list

	def init_hidden(self,batch_size=1):
		return torch.zeros(self.n_layers, batch_size, self.hidden_size, dtype=torch.float)


class MDN(nn.Module):
	def __init__(self, input_size, hidden_size, output_size, n_layers, *args, **kwargs):
		"""
		config net type 5
		Mixture density network
		Outputs - means, variances, and mixture coeff to parameterize a mixture of Gaussians  
		
		input_size: if input layer - combined size of generated+conditional vectors
		hidden_size: no. of hidden nodes for each GRU layer
		output_size: size of output, normally equal to no. of generated parameters
		n_layers: no. of stacked GRU layers
		"""
		super(MDN, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.n_layers = n_layers
		self.n_gaussians = 3

		self.i2d = nn.Linear(self.input_size, hidden_size)
		self.gru = nn.GRU(hidden_size, hidden_size, n_layers, batch_first=True)
		self.tanh = nn.Tanh()
		
		self.mu = nn.Linear(hidden_size, output_size*self.n_gaussians)
		self.sigma = nn.Linear(hidden_size, output_size*self.n_gaussians)
		self.pi = nn.Linear(hidden_size, output_size*self.n_gaussians)
		self.pisoft = nn.Softmax(dim=1)
	
	def forward(self, input, hidden, batch_size=1, **kwargs): 
		if 'beta' in kwargs.keys():
			beta = kwargs['beta']
		else:
			beta = 1.0

		h1 = F.relu(self.i2d(input))
		h_out, hidden = self.gru(h1.view(batch_size,1,-1), hidden)
		h_out = self.tanh(h_out)

		pi = self.pi(h_out.view(batch_size,-1))
		pi = pi.view(-1, self.n_gaussians, self.output_size)
		pi = self.pisoft(pi)
		#print("pi",pi)
		
		mu = self.mu(h_out.view(batch_size,-1)) 
		mu = mu.view(-1, self.n_gaussians, self.output_size)
		#print("MU",mu.shape,mu)
		
		sig = self.sigma(h_out.view(batch_size,-1))
		sig = F.softplus(sig,beta=beta) #F.elu(sig) + 1.  #torch.exp(sig)
		sig = sig.view(-1, self.n_gaussians, self.output_size)
		#print("sig",sig)
		
		output = (mu,sig,pi) #self.sample(mu, sig, pi)
		#prob = self.gaussian_probability(mu,sig,target,batch_size) 
		#scaled_prob = pi * prob
		#output = torch.sum(scaled_prob, dim=1)
		#print("prob",prob)
		#print("O",output.shape,output)
		return output, hidden

	def init_hidden(self,batch_size=1):
		return torch.zeros(self.n_layers, batch_size, self.hidden_size, dtype=torch.float)
