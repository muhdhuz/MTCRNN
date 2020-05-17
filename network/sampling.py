"""
Different sampling methods to generate results.
@muhammad huzaifah 2020/02/26 
"""

import math
import torch 
import torch.nn as nn
import torch.nn.functional as F
import dataloader.transforms as tr


def sample_normal(mu, sigma, beta):
	"""sample from output layer consisting of parameters mean and variance to a normal distribution"""

	std = F.softplus(sigma,beta=beta)  #1/beta*log(1+ exp(x/beta))
	#std = F.elu(sigma) + 1.
	eps = torch.randn_like(std)
	out = mu + eps*std
	return out  #torch.sigmoid()


def sample_multinomial(output, temperature, output_size, onehot=False):
	"""sample from output layer of an audio trained model. Takes in unscaled logits"""
	#log_output = torch.nn.functional.log_softmax(output,dim=1)

	#topv, topi = log_output.topk(1) #output topi is a mu-law index
	#mulaw_output2 = topi.detach().cpu().numpy()

	#out_weights = log_output.div(temperature).exp()

	scaled_logits = output.div(temperature)
	out_weights = torch.nn.functional.softmax(scaled_logits,dim=1)

	idx = torch.multinomial(out_weights, 1) #categorical distribution
	mulaw_output = idx.detach().cpu().numpy()

	#encode for next step
	if onehot:
		mulaw_to_onehot = transform.Compose([tr.onehotEncode(output_size),tr.array2tensor(torch.FloatTensor)])
	else:
		mulaw_output_norm = mulaw_output/output_size #norm by no. of quantization channel -> [0,1]
		mulaw_to_onehot = tr.array2tensor(torch.FloatTensor)
	next_input = mulaw_to_onehot(mulaw_output_norm)
	
	#decode to get audio sample
	predicted_sample = tr.mulawDecode(output_size)(mulaw_output)
	return next_input, predicted_sample


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