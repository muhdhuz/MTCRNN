"""
A script for generation
"""
import torch
import soundfile as sf
import datetime
import time
import numpy as np
import os

import network.config as config
from network.model import CondRNN
from dataloader.dataloader import DataLoader
from paramManager.paramManager import paramManager
from utils.myUtils import ProgressBar


class Generator:
	def __init__(self, args):
		self.args = args
		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		print("using", self.device)
		#if torch.cuda.is_available():
		#	torch.cuda.empty_cache()

		if 'audio' in args.generate:
			self.model = CondRNN(args.cond_size+args.gen_size, args.hidden_size,
								args.mulaw_channels, args.n_layers, self.device,
								paramonly=False,onehot=args.onehot,
								ntype=args.net,plstm=args.plstm)
		else:
			self.model = CondRNN(args.cond_size+args.gen_size, args.hidden_size,
								args.gen_size, args.n_layers, self.device,
								paramonly=True,onehot=args.onehot,
								ntype=args.net,plstm=args.plstm)

		self.model.load(args.model_dir, args.step)

		if args.seed is not None:
			#self.args.batch_size = 1
			#self.args.data_dir = self.args.seed
			self._get_seed_from_audio(self.args.seed)

		else:
			self.data_loader = DataLoader(args.data_dir, args.sample_rate, args.seq_len, args.stride, 
										paramdir=args.param_dir, prop=args.prop, generate=args.generate,
										mulaw_channels=args.mulaw_channels,
										batch_size=args.batch_size,
										onehot=args.onehot)

		if args.external_array is not None:
			self.param_array = np.load(args.external_array)


	def _random_primer(self,batch_size,feature_size,length=1,seed=None):
		"""make noisy priming signal
		feature_size = cond_size+gen_size
		primer shape: [batchsize,length,feature_size]"""
		np.random.seed(seed)
		myp = np.random.ranf((batch_size,length,feature_size))

		return torch.tensor(myp, dtype=torch.float, device=self.device)


	def _get_seed_from_audio(self, filepath):
		self.data_loader = DataLoader(filepath, self.args.sample_rate, self.args.seq_len, self.args.stride, 
							paramdir=self.args.param_dir, prop=self.args.prop, generate=self.args.generate,
							mulaw_channels=self.args.mulaw_channels,
							batch_size=self.args.batch_size, shuffle=False,
							onehot=self.args.onehot,
							is_seeded=True,load_start=self.args.seed_start)


	def _save_to_audio_file(self, data):
		if data.shape[0] > 1:
			for i in range(data.shape[0]):
				sf.write(self.args.out+'_'+str(i)+'.wav', data[i], 16000, 'PCM_24')
				print('Saved wav file as {}'.format(self.args.out+'_'+str(i)))
		else:
			sf.write(self.args.out+'.wav', data[0], 16000, 'PCM_24')
			print('Saved wav file as {}'.format(self.args.out))


	def generate(self,params=None,original_sr=None):
		start_time = datetime.datetime.now()
		with torch.no_grad():
			outputs = []
			mus = []
			sigs = []
			if self.args.paramvect == 'external':
				if params is None:
					params = self.param_array
					assert params is not None, "Please provide a parameter array for paramvect option external!" 
				if original_sr is None:
					original_sr = self.args.external_sr
					assert original_sr is not None, "Please provide the original sample rate of the parameters for paramvect option external!"
				_,params_re = paramManager.resample(params,original_sr,self.args.sample_rate)
			
			for inputs, _ in self.data_loader:
				print('priming...')
				onetime = torch.arange(self.args.seq_len,dtype=torch.float) #plstm things
				times = onetime.repeat(self.args.batch_size,1).to(self.device)  #inputs.shape[0]
				print(times.shape)
				if self.args.rand_prime:
					input = self._random_primer(batch_size=self.args.batch_size,feature_size=self.args.cond_size+self.args.gen_size,
										length=self.args.seq_len-self.args.length,seed=self.args.rand_seed)
				else:
					input = inputs[:,:self.args.seq_len-self.args.length,:].to(self.device)
				next_input, hidden = self.model.build_hidden_state(input,times[:,:self.args.seq_len-self.args.length])
				print('DONE')
				progress = ProgressBar(self.args.length, fmt=ProgressBar.FULL)

				for length in range(self.args.length):
					timestep = times[:,self.args.seq_len-self.args.length+length-1]
					transformed_sample, predicted_sample, hidden, mu, sig = self.model.generate(next_input,hidden,self.args.temp,timestep)

					if predicted_sample.shape[1] > 1:
						predicted_sample = np.expand_dims(predicted_sample, axis=1)
						mu = np.expand_dims(mu, axis=1) 
						sig = np.expand_dims(sig, axis=1)  
					outputs = np.concatenate((outputs, predicted_sample),axis=1) if len(outputs) else predicted_sample
					mus = np.concatenate((mus, mu),axis=1) if len(mus) else mu
					sigs = np.concatenate((sigs, sig),axis=1) if len(sigs) else sig
					#print('{0}/{1} samples are generated.'.format(outputs.shape[1], self.args.length))
					progress()		
					
					if self.args.paramvect == 'self':
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

			progress.done()
			time.sleep(0.5)
			print('Generate took {0} seconds'.format(datetime.datetime.now() - start_time))
			if 'audio' in self.args.generate:
				self._save_to_audio_file(outputs)
			else:
				if self.args.save:
					np.save(self.args.out,outputs)

			if self.args.paramvect == 'self':
				#return: outputs, original cond params, original generated features
				#return outputs, inputs[:,self.args.seq_len-self.args.length:,self.args.gen_size:].numpy(), inputs[:,self.args.seq_len-self.args.length:,:self.args.gen_size].numpy()#, mus, sigs
				return outputs, inputs[:,:,self.args.gen_size:].numpy(), inputs[:,:,:self.args.gen_size].numpy()
			elif self.args.paramvect == 'external':
				#return: outputs, original cond params
				return outputs, params_re#, mus, sigs
			else:
				#return: outputs
				return outputs


if __name__ == '__main__':
	args = config.parse_args(is_training=False)
	print(args)

	generator = Generator(args)

	generator.generate()

