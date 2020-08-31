"""
Different sampling methods to generate results.
@muhammad huzaifah 2020/02/26 
"""

import math
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transform
import dataloader.transforms as tr


def sample_normal(mu, sigma, beta):
	"""sample from output layer consisting of parameters mean and variance to a normal distribution"""

	std = F.softplus(sigma,beta=beta)  #1/beta*log(1+ exp(x/beta))
	#std = F.elu(sigma) + 1.
	eps = torch.randn_like(std)
	out = mu + eps*std
	return out  #torch.sigmoid()

def get_p(prob,cutoff):  #
	sorted_probs, sorted_indices = torch.sort(prob,descending=True)
	#print("prob",sorted_probs)
	cumulative_probs = torch.cumsum(sorted_probs, dim=-1).detach().cpu().numpy()
	#print("cum",cumulative_probs)
	
	#cutofftensor = np.full((cumulative_probs.shape[0], 1), cutoff)#cumulative_probs.new_full((cumulative_probs.shape[0], 1), cut_off)
	#print("off",cutofftensor)

	cutoff_index = np.apply_along_axis(lambda a: a.searchsorted(cutoff), axis=1, arr=cumulative_probs)
	cutoff_index = cutoff_index.reshape(cumulative_probs.shape[0],1)

	#index = np.searchsorted(cumulative_probs, cut_offtensor) #torch.searchsorted only available in pyotch 1.6
	#print("ind",cutoff_index)
	return cutoff_index


def sample_multinomial(output, temperature, output_size, onehot=False,cutoff=None):
	"""sample from output layer of an audio trained model. Takes in unscaled logits"""

	scaled_logits = output.div(temperature)
	out_weights = F.softmax(scaled_logits,dim=1)
	"""
	topv, topi = out_weights.topk(1) #output topi is a mu-law index
	chosen = torch.multinomial(topv, 1) #categorical distribution
	idx = torch.gather(topi, 1, chosen) 
	mulaw_output = idx.detach().cpu().numpy()
	"""
	if cutoff is not None:
		cutoff_index = get_p(out_weights,cutoff)
	else:
		cutoff_index = []
	
	idx = torch.multinomial(out_weights, 1) #categorical distribution
	mulaw_output = idx.detach().cpu().numpy()
	
	#encode for next step
	if onehot:
		mulaw_transform = transform.Compose([tr.onehotEncode(output_size),tr.array2tensor(torch.FloatTensor)])
	else:
		mulaw_output_norm = mulaw_output/output_size #norm by no. of quantization channel -> [0,1]
		mulaw_transform = tr.array2tensor(torch.FloatTensor)
	next_input = mulaw_transform(mulaw_output_norm)

	#decode to get audio sample
	predicted_sample = tr.mulawDecode(output_size)(mulaw_output)
	return next_input, predicted_sample, cutoff_index 


def gaussian_probability(mu, sigma, target):
	"""Returns the probability of `data` given MoG parameters `sigma` and `mu`. Use with MDN
	
	Arguments:
		sigma (BxGxO): The standard deviation of the Gaussians. B is the batch
			size, G is the number of Gaussians, and O is the number of
			dimensions per Gaussian.
		mu (BxGxO): The means of the Gaussians. B is the batch size, G is the
			number of Gaussians, and O is the number of dimensions per Gaussian.
		data (BxI): A batch of data. B is the batch size and I is the number of
			input dimensions.
	Returns:
		probabilities (BxG): The probability of each point in the probability
			of the distribution in the corresponding sigma/mu index.
	"""
	target = target.view(sigma.shape[0],1,-1).expand_as(sigma)
	#print("T",target.shape,target)
	ret = (torch.exp(-0.5 * (target - mu)**2 / sigma**2)) / (math.sqrt(2.0*math.pi)*sigma) 
	#print("R",ret.shape,ret)
	return ret


def sample_mixturedensity(mu, sigma, pi):
	"""samples from a mixture of Gaussians. Use with MDN"""
	mus = torch.empty(pi.shape[0],1,pi.shape[2])
	#print("M",mus,mus.shape)
	sigmas = torch.empty(pi.shape[0],1,pi.shape[2])
	#print("sigmas",sigmas,sigmas.shape)
	for feat in range(pi.shape[2]):
		idx = torch.multinomial(pi[:,:,feat], 1)#.squeeze(1)
		#print("idx",idx)
		idx = [torch.LongTensor(range(pi.shape[0])).unsqueeze(1), idx]
		mus[:,:,feat] = mu[:,:,feat][idx]
		sigmas[:,:,feat] = sigma[:,:,feat][idx]
	#print("M",mus,mus.shape)
	#print("sigmas",sigmas,sigmas.shape)

	eps = torch.randn_like(mus)  #batch,features
	out = mus + eps*sigmas
	out = out.squeeze(1)
	return out
	#print("o",out,out.shape)