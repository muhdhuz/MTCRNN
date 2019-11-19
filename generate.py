"""
A script for WaveNet training

TO DO:
- take seed file 16000 in length, use first seq_len to build hidden state, use (length-seq_len) to generate
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


class Generator:
	def __init__(self, args):
		self.args = args
		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

		if args.paramonly:
			self.model = CondRNN(args.input_size, args.hidden_size,
								args.input_size, args.n_layers, self.device,
								paramonly=args.paramonly,
								onehot=args.onehot)
		else:
			self.model = CondRNN(args.input_size, args.hidden_size,
								args.mulaw_channels, args.n_layers, self.device,
								paramonly=args.paramonly,
								onehot=args.onehot)

		self.model.load(args.model_dir, args.step)

		if args.seed is not None:
			args.batch_size = 1
			args.data_dir = args.seed

		self.data_loader = DataLoader(args.data_dir, args.sample_rate, args.seq_len, args.stride, 
									paramdir=args.param_dir, prop=args.prop,
									mulaw_channels=args.mulaw_channels,
									batch_size=args.batch_size,
									paramonly=args.paramonly,
									onehot=args.onehot)

		#if args.paramvect is not None:
		#	self.paramvect = np.load(args.paramvect)


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
		audio, _ = sf.read(filepath)
		audio_length = len(audio)

		audio = utils.mu_law_encode(audio, self.args.in_channels)
		audio = utils.one_hot_encode(audio, self.args.in_channels)

		seed = self._make_seed(audio)

		return self._variable(seed).to(self.device), audio_length

	def _save_to_audio_file(self, data):
		for i in range(data.shape[0]):
		# = data[0].detach().cpu().numpy()
		#data = utils.one_hot_decode(data, axis=1)
		#audio = utils.mu_law_decode(data, self.args.in_channels)
			#print(data[i])
			sf.write(self.args.out+str(i)+'.wav', data[i], 16000, 'PCM_24')
			#librosa.output.write_wav(self.args.out+str(i), data[i], self.args.sample_rate)
			print('Saved wav file as {}'.format(self.args.out+str(i)))

		#return None librosa.get_duration(y=audio, sr=self.args.sample_rate)

	def generate(self):
		outputs = []

		#print("PA",paramvect.shape)
		#inputs, audio_length = self._get_seed_from_audio(self.args.seed)
		for inputs, _ in self.data_loader:
			#print("ORI IN",inputs)
			input = inputs[:,:self.args.seq_len-self.args.length,:].to(self.device)
			#print("INPUT",input.shape)
			#print("INPUT",input)
			next_input, hidden = self.model.build_hidden_state(input)
			#print("NEXT INPUT",next_input)
			for length in range(self.args.length):

				transformed_sample, predicted_sample, hidden = self.model.generate(next_input, hidden)
				#print("TS",transformed_sample)
				#print("PS",predicted_sample)
				if predicted_sample.shape[1] > 1:
					predicted_sample = np.expand_dims(predicted_sample, axis=1) 
				outputs = np.concatenate((outputs, predicted_sample),axis=1) if len(outputs) else predicted_sample
				#print("OUTSHAPE",outputs.shape)
				print('{0}/{1} samples are generated.'.format(outputs.shape[1], self.args.length))
				if self.args.paramvect == 'self':
					#paramvect_new = paramvect + 0.00001*length
					if self.args.onehot:
						paramvect = inputs[:,self.args.seq_len-self.args.length+length,self.args.mulaw_channels:]
					else:
						paramvect = inputs[:,self.args.seq_len-self.args.length+length,1:]
					#print("P",paramvect.shape)
					#print("P",paramvect)
					#batchparams = np.tile(paramvect_new,(self.args.batch_size,1))
					#paramvect_tensor = torch.from_numpy(batchparams).type(torch.FloatTensor)

					next_input = torch.cat((transformed_sample,paramvect),dim=1).to(self.device)
				elif self.args.paramvect == 'none':
					next_input = transformed_sample
				#print("NI",next_input) 
				#print("NI shape",next_input.shape)
			break
		#outputs = outputs[:, :self.args.length, :]
		#print(outputs.shape)
		if not self.args.paramonly:
			self._save_to_audio_file(outputs)
			#return outputs, original params, original audio
			if self.args.onehot:
				return outputs, inputs[:,self.args.seq_len-self.args.length:,self.args.mulaw_channels:], inputs[:,self.args.seq_len-self.args.length:,:self.args.mulaw_channels]
			else:
				return outputs, inputs[:,self.args.seq_len-self.args.length:,1:], inputs[:,self.args.seq_len-self.args.length:,0]
		else:
			#return outputs, original params
			return outputs, inputs[:,self.args.seq_len-self.args.length:,:]

if __name__ == '__main__':
	args = config.parse_args(is_training=False)
	print(args)

	generator = Generator(args)

	start_time = datetime.datetime.now()

	generator.generate()

	print('Generate took {0} seconds'.format(datetime.datetime.now() - start_time))

