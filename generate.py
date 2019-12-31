"""
A script for generation
"""
import torch
import soundfile as sf
#import librosa 
import datetime
import numpy as np
import os

import network.config as config
from network.model import CondRNN
from dataloader.dataloader import DataLoader
from paramManager.paramManager import paramManager


class Generator:
	def __init__(self, args):
		self.args = args
		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

		if 'audio' in args.generate:
			self.model = CondRNN(args.cond_size+args.gen_size, args.hidden_size,
								args.mulaw_channels, args.n_layers, self.device,
								paramonly=False,onehot=args.onehot)

		else:
			self.model = CondRNN(args.cond_size+args.gen_size, args.hidden_size,
								args.gen_size, args.n_layers, self.device,
								paramonly=True,onehot=args.onehot)

		self.model.load(args.model_dir, args.step)

		if args.seed is not None:
			args.batch_size = 1
			args.data_dir = args.seed

		self.data_loader = DataLoader(args.data_dir, args.sample_rate, args.seq_len, args.stride, 
									paramdir=args.param_dir, prop=args.prop, generate=args.generate,
									mulaw_channels=args.mulaw_channels,
									batch_size=args.batch_size,
									onehot=args.onehot)

		if args.external_array is not None:
			self.param_array = np.load(args.external_array)


	def _random_primer(self,cond_size,length=1,seed=None):
		#NOT IMPLEMENTED
		"""make noisy priming signal
		primer shape: [batchsize,length,cond_size+1]"""
		np.random.seed(seed)
		myp=np.zeros([1,length,cond_size+1])
		myp[0,:,0] =.1*np.random.ranf([length]) #signal
		for dim in range(cond_size):
			myp[0,:,dim+1] =.1*np.random.ranf([length])#-.15 #signal
		#myp[0,:,1] = .45+.1*np.random.ranf([length])     #instrument
		#myp[0,:,2] = .05*np.random.ranf([length])    #volume
		#myp[0,:,3] = np.random.ranf([1])    #pitch
		return torch.tensor(myp, dtype=torch.float, device=cfg.device)


	def _get_seed_from_audio(self, filepath):
		self.data_loader = DataLoader(filepath, self.args.sample_rate, self.args.seq_len, self.args.stride, 
							paramdir=self.args.param_dir, prop=self.args.prop, generate=self.args.generate,
							mulaw_channels=self.args.mulaw_channels,
							batch_size=self.args.batch_size,
							onehot=self.args.onehot)


	def _save_to_audio_file(self, data):
		for i in range(data.shape[0]):
			sf.write(self.args.out+str(i)+'.wav', data[i], 16000, 'PCM_24')
			print('Saved wav file as {}'.format(self.args.out+str(i)))


	def generate(self,params=None,original_sr=None):
		outputs = []
		if self.args.paramvect == 'external':
			if params is None:
				params = self.param_array
				assert params is not None, "Please provide a parameter array for paramvect option external!" 
			if original_sr is None:
				original_sr = self.args.external_sr
				assert original_sr is not None, "Please provide the original sample rate of the parameters for paramvect option external!"
			_,params_re = paramManager.resample(params,original_sr,self.args.sample_rate)

		if self.args.seed is not None:
			self._get_seed_from_audio(self.args.seed)

		for inputs, _ in self.data_loader:
			print('priming...')
			input = inputs[:,:self.args.seq_len-self.args.length,:].to(self.device)
			next_input, hidden = self.model.build_hidden_state(input)
			print('DONE')

			for length in range(self.args.length):
				transformed_sample, predicted_sample, hidden = self.model.generate(next_input,hidden,self.args.temp)

				if predicted_sample.shape[1] > 1:
					predicted_sample = np.expand_dims(predicted_sample, axis=1) 
				outputs = np.concatenate((outputs, predicted_sample),axis=1) if len(outputs) else predicted_sample
				print('{0}/{1} samples are generated.'.format(outputs.shape[1], self.args.length))
				
				if self.args.paramvect == 'self':
					#if self.args.onehot:
					#	paramvect = inputs[:,self.args.seq_len-self.args.length+length,self.args.mulaw_channels:] #can merge this since gen_size will cover mulaw_channels case
					#else:
					paramvect = inputs[:,self.args.seq_len-self.args.length+length,self.args.gen_size:]
					next_input = torch.cat((transformed_sample,paramvect),dim=1).to(self.device)
				
				elif self.args.paramvect == 'external':
					if length < params_re.shape[1]: 
						paramvect = params_re[:,length,:]
					else:
						paramvect = params_re[:,-1,:] #if paramvect shorter than desired length just keep using the last value 
					next_input = torch.cat((transformed_sample,torch.from_numpy(paramvect).float()),dim=1).to(self.device)

				elif self.args.paramvect == 'none':
					next_input = transformed_sample.to(self.device)

			break

		if 'audio' in self.args.generate:
			self._save_to_audio_file(outputs)
		else:
			if self.args.save:
				np.save(self.args.out,outputs)

		if self.args.paramvect == 'self':
			#return: outputs, original cond params, original generated features
			#return outputs, inputs[:,self.args.seq_len-self.args.length:,self.args.gen_size:], inputs[:,self.args.seq_len-self.args.length:,:self.args.gen_size]
			return outputs, inputs[:,:,self.args.gen_size:], inputs[:,:,:self.args.gen_size]
		elif self.args.paramvect == 'external':
			#return: outputs, original cond params
			return outputs, params_re
		else:
			#return: outputs
			return outputs


if __name__ == '__main__':
	args = config.parse_args(is_training=False)
	print(args)

	generator = Generator(args)

	start_time = datetime.datetime.now()

	generator.generate()

	print('Generate took {0} seconds'.format(datetime.datetime.now() - start_time))

