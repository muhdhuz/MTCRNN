"""
Various audio/Pytorch data transformations
"""

import numpy as np
import torch

class onehotEncode:
	"""
	Array of class indices to one-hot encoding: [1,2,0] -> [[0 1 0],[0 0 1],[1 0 0]]
	channels: total no. of classes 
	"""
	def __init__(self,channels=256):
		self.channels = channels

	def __call__(self, input):
		one_hot = np.zeros((input.size, self.channels), dtype=float)
		one_hot[np.arange(input.size), input.astype(int).ravel()] = 1

		return one_hot


class onehotDecode:
	"""
	One-hot encoding to array of class indices
	"""
	def __init__(self,axis=1):
		self.axis = axis
	
	def __call__(self, input):
		decoded = np.argmax(input, axis=self.axis)

		return decoded


class mulawEncode: 
	"""
	Quantize waveform amplitudes.
	Reference: https://github.com/vincentherrmann/pytorch-wavenet/blob/master/audio_data.py
	norm: normalize mulaw between [0,quantization_channels] to [0,1]
	"""
	def __init__(self,quantization_channels=256,norm=False):
		self.quantization_channels = quantization_channels
		self.norm = norm

	def __call__(self, audio):
		mu = float(self.quantization_channels - 1)
		quantize_space = np.linspace(-1, 1, self.quantization_channels)

		quantized = np.sign(audio) * np.log(1 + mu * np.abs(audio)) / np.log(mu + 1)
		quantized = np.digitize(quantized, quantize_space) - 1

		if self.norm:
			quantized = quantized/self.quantization_channels

		return quantized


class mulawDecode:
	"""
	Recovers waveform from quantized values.
	Reference: https://github.com/vincentherrmann/pytorch-wavenet/blob/master/audio_data.py
	"""
	def __init__(self,quantization_channels=256,norm=False):
		self.quantization_channels = quantization_channels
		self.norm = norm

	def __call__(self, output):
		mu = float(self.quantization_channels - 1)
		if self.norm:
			 output = (output*self.quantization_channels).astype(int)

		expanded = (output / self.quantization_channels) * 2. - 1
		waveform = np.sign(expanded) * (
					np.exp(np.abs(expanded) * np.log(mu + 1)) - 1
				) / mu

		return waveform


class array2tensor:
	"""Convert ndarrays in sample to Tensors. Samples are assumed to be python dicts"""
	def __init__(self,dtype):
		self.dtype = dtype
	
	def __call__(self, sample):
		return torch.from_numpy(sample).type(self.dtype)


class dic2tensor:
	"""Convert ndarrays in sample to Tensors. Samples are assumed to be python dicts"""
	def __init__(self,dtype):
		self.dtype = dtype

	def __call__(self, sample):
		combined = np.stack([sample[i] for i in sample],axis=1)
		if len(sample) > 1:			  
			tensor_sample = torch.squeeze(torch.from_numpy(combined).type(self.dtype))
		else:
			tensor_sample = torch.from_numpy(combined).type(self.dtype)		
		
		return tensor_sample


class injectNoise:
	"""Add some noise to the signal. 
	If constant=True, constant noise in the range of (low,high) and scaled by weight added to all samples
	If constant=False, noise will be added according to a constant signal-to-noise-ratio controlled by weight. (low,high) will be ignored"""
	def __init__(self,low=-1,high=1,weight=0.1,constant=False):
		self.low = low
		self.high = high
		self.constant = constant
		if weight > 1:
			raise ValueError("weight has to be <= 1")
		else:
			self.noiseweight = weight
	
	def __call__(self, sample):
		if self.constant: 
			return sample + self.noiseweight * np.random.uniform(self.low, self.high, size=len(sample)).reshape(-1,1)
		else:
			return sample + self.noiseweight * np.random.uniform(sample.min(), sample.max(), size=len(sample)).reshape(-1,1)


class normalizeDim:
	"""Normalizes one dimension of a dictionary to [0,1].
		NOTE: pmin and pmax should correspond to the min and max AFTER the paramManager has loaded the param files.
	Args:
		pname - the name of the parameter to normalize - must match one used in the param file
		pmin - the value that will be normalized to 0
		pmax - the value that will be normalized to 1
	"""
	def __init__(self, pname, pmin, pmax):
		self.pname = pname
		self.pmin = pmin
		self.pmax = pmax
		self.scale= pmax-pmin

	def __call__(self, sample):		
		try:
			value = sample[self.pname]
		except KeyError:
		# Key is not present
			print("param file has no property {}".format(self.pname))
		if any((elem < self.pmin or elem > self.pmax) for elem in sample[self.pname]) :
			raise ValueError('NormalizeDim: an element was found outside [pmin={}, pmax={}] range'.format(self.pmin, self.pmax))
		sample[self.pname] = [(x-self.pmin)/self.scale for x in sample[self.pname]]
		return sample		