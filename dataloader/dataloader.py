"""
The dataloader class below is a chunker and data loader that loads a part of an audio clip 
plus corresponding parameters (if any) from a param file. To be used with pytorch platform.
It can also load parameters directly from parameter files. 
Intended to be used for the MTCRNN project: https://github.com/muhdhuz/MTCRNN 

Algo for audio loading:
to load each group containing: 1 audio wav file
							   1 parameter json file
filenames will be contained within a directory tree, specify the root
1. parse directory. get path of all objects in each line and append to list
2. create a list of indices to draw samples (eg. index 52 = 4th wav file sample 390). this no. will also be the __len__
3. __getitem__:
	sample index
	load corresponding wav file
	pull out correct audio sample sequence
	load corresponding params
	convert audio to mu-law
	convert mu-law + params to tensor

@muhammad huzaifah 16/05/2020

Notes: Eventually hope to migrate to torchaudio for loading/transformations
	   Currently torchaudio dependency libsox is broken for windows

	   This dataloader has the following expanded capabilities:
	   - load/transform parameters directly from param file
	   - load a single audio file at a specified start point
	   - handle variable file lengths in data directory
"""

import torch
import torch.utils.data as data
import torchvision.transforms as transform

import os
import numpy as np
import math
import soundfile as sf 

from paramManager import paramManager
import dataloader.transforms as tr


def file2list(rootdir,extension):
	"""append to list all files in rootdir with given extension"""
	filelist = [os.path.join(rootdir, f) for f in os.listdir(rootdir) if f.endswith('.' + extension)]
	return filelist

def listDirectory_all(directory,fileExtList=None,topdown=True,regex=None):
	"""returns a list of all files in directory and all its subdirectories
	directory: either a single file or a directory. If directory fileExtList has to be supplied
	fileList: full path
	fnameList: basenames"""
	fileList = []
	fnameList = []
	if os.path.isdir(directory):
		for root, _, files in os.walk(directory, topdown=topdown):
			for name in files:
				if regex is not None:
					if ((os.path.splitext(name)[1] in fileExtList) and (regex in name)):
						fileList.append(os.path.join(root, name))
						fnameList.append(name)
				else:
					if (os.path.splitext(name)[1] in fileExtList):
						fileList.append(os.path.join(root, name))
						fnameList.append(name)
	else:
		fileList.append(directory)
		fnameList.append(os.path.basename(directory)) 
	return fileList , fnameList

def check_duration(filelist,allsame=False):
	"""use PySoundFile's info method to find the duration of all files in filelist
	input params: allsame=True if you expect all durations to be the same, then routine will test this
	returns: a list of all durations in filelist"""
	filedurations = [sf.info(file=f).duration for f in filelist]
	if allsame == True:
		assert filedurations.count(filedurations[0]) == len(filedurations), "File durations are not all the same!"
	return filedurations
	
def dataset_properties(filelist,sr,seqLen):
	"""return several dataset parameters 
	input params: filelist - list of filenames that forms the dataset
				  sr - sample rate of the audio
				  seqLen - length of each data section measured in samples
	"""
	fileLen = len(filelist)						#no of files in filelist
	fileDuration = check_duration(filelist)		#duration of each file in sec
	totalFileDuration = sum(fileDuration)		#total duration of all files in dataset
	totalSamples = int(totalFileDuration * sr)	 #combined total no of samples
	srInSec = 1/sr								#sampling rate in sec
	seqLenInSec = srInSec * seqLen				#length of 1 data sequence in sec
	return fileLen,fileDuration,totalFileDuration,totalSamples,srInSec,seqLenInSec
	
def create_sampling_index(totalSamples,stride):
	"""calculate total no. of data sections in dataset. Indices will later be used to draw sequences (eg. index 52 = 4th wav file sample 390)
	input params: totalSamples - combined total no of samples in dataset (get from dataset_properties)
				  stride: shift in no. of samples between adjacent data sections. (seqLen - stride) samples will overlap between adjacent sequences.  
	"""	
	indexLen = totalSamples // stride
	return indexLen

def choose_sequence_notsame(index,fileDuration,srInSec,stride):
	"""alternative algorithm to choose_sequence if the file durations are not all the same. 
	Current default since can also be used if file durations are all the same (and marginally faster)"""
	timestamp = index * srInSec * stride
	cummulduration = 0
	chooseFileIndex = -1
	for duration in fileDuration:
		cummulduration += duration
		chooseFileIndex += 1
		if cummulduration >= timestamp:
			break
	startoffset = timestamp - (cummulduration - fileDuration[chooseFileIndex]) #will load at this start time	
	return chooseFileIndex,startoffset
	
def load_sequence(filelist,chooseFileIndex,startoffset,frames,sr):
	"""load the correct section of audio. If len of audio < seqLen+1 (e.g. sections at the end of the file), then draw another section.
	if is_seeded, we draw 1 sample more than seqLen so can take input=y[:-1] and target=y[1:]
	else just draw seqLen (for example if using for generation)"""
	y,_ = sf.read(filelist[chooseFileIndex],frames=frames,start=round(startoffset*sr))			
	#if len(y) < seqLen+1:
	#	y = None
	return y


class AudioDataset(data.Dataset):	 
	def __init__(self, datadir, sr, seqLen, stride, 
				paramdir, prop, generate, extension, is_seeded, load_start, 
				transform=None, param_transform=None, target_transform=None):
		"""
		sr: sample rate of audio files in dataset
		seqLen: sequence length of each input data in no. of samples. must be less no. samples in each audio file
		stride: shift in no. of samples between adjacent data sequences. (seqLen - stride) samples will overlap between adjacent sequences
		datadir: root data directory
		extension: file extension of data
		paramdir: directory of parameter files
		prop: list of parameter keys to be used for input conditioning
		generate: list of parameters key/audio to be generated
		is_seeded: Boolean to specify whether to load data from seed file
		load_start: If is_seeded, specify start time point to load file
		transform: list of transformations on input audio sequence. use with torchvision.transforms.Compose() for multiple
		param_transform: list of transformations on parameters. use with torchvision.transforms.Compose() for multiple
		target_transform: list of transformations on target audio sequence. use with torchvision.transforms.Compose() for multiple 
		"""
		super(AudioDataset, self).__init__()
		if stride < 1:
			raise ValueError("stride has to be >= 1")
			
		self.filelist,self.fnamelist = listDirectory_all(directory=datadir,fileExtList=extension)
		self.datadir = datadir
		self.paramdir= paramdir
		self.prop = prop
		self.generate = generate
		self.is_seeded = is_seeded
		self.load_start = load_start
		self.fileLen,self.fileDuration,self.totalFileDuration,self.totalSamples,self.srInSec,self.seqLenInSec = dataset_properties(self.filelist,sr,seqLen)
		self.sr = sr
		self.seqLen = seqLen
		self.stride = stride
		self.transform = transform
		self.param_transform = param_transform
		self.target_transform = target_transform
		self.indexLen = create_sampling_index(self.totalSamples,self.stride)


	def __getitem__(self,index):
		whole_sequence = self.rand_sample(index)
		if self.is_seeded:
			sequence = whole_sequence
			target = whole_sequence
		else:
			assert len(whole_sequence) == self.seqLen+1, str(len(whole_sequence))
			sequence = whole_sequence[:-1]
			target = whole_sequence[1:]			
		
		if self.transform is not None:
			input = self.transform(sequence)
		
		if self.target_transform is not None:
			target = self.target_transform(target)

		if self.paramdir is not None and len(self.prop)>0:
			pm = paramManager.paramManager(self.datadir, self.paramdir)
			params = pm.getParams(self.filelist[self.chooseFileIndex]) 
			paramdict = pm.resampleAllParams(params,self.seqLen,self.startoffset,self.startoffset+self.seqLenInSec,self.prop,verbose=False)
			paramtensor = self.param_transform(paramdict)
			input = torch.cat((input,paramtensor),1)  #input dim: (batch,seq,feature), batch dimension wrapped in automatically in Dataloader later			
		
		return input, target

	def __len__(self):
		return self.indexLen 


	def rand_sample(self,index=None,verbose=False):
		if index is None:
			index = np.random.randint(self.indexLen)
		self.chooseFileIndex,self.startoffset = choose_sequence_notsame(index+1,self.fileDuration,self.srInSec,self.stride)
		if self.is_seeded:
			self.startoffset = self.load_start/self.sr
			self.load_length = self.seqLen
		else:
			self.load_length = self.seqLen+1 #since need to an offset of 1 sample between target and input
			while self.fileDuration[self.chooseFileIndex] < (self.startoffset + self.seqLenInSec + 1/self.sr):
				index = np.random.randint(self.indexLen)
				self.chooseFileIndex,self.startoffset = choose_sequence_notsame(index+1,self.fileDuration,self.srInSec,self.stride)
	
		if verbose:
			print('loading part of file:',self.filelist[self.chooseFileIndex],'starting at',self.startoffset)
		whole_sequence = load_sequence(self.filelist,self.chooseFileIndex,self.startoffset,self.load_length,self.sr) 
		whole_sequence = whole_sequence.reshape(-1,1)
		return whole_sequence
		
		"""
		
		if self.is_seeded:
			startoffset = self.load_start/sr
			chooseFileIndex = 0
			whole_sequence = load_sequence(self.filelist,chooseFileIndex,startoffset,self.seqLen,self.sr) 
			sequence = whole_sequence.reshape(-1,1)
			target = whole_sequence.reshape(-1,1)
		else:
			chooseFileIndex,startoffset = choose_sequence_notsame(index+1,self.fileDuration,self.srInSec,self.stride)
			#while whole_sequence is None: #if len(whole_sequence) < self.seqLen+1, pick another random section
			while self.fileDuration[chooseFileIndex] < (startoffset + self.seqLenInSec + 1/self.sr):
				index = np.random.randint(self.indexLen)
				chooseFileIndex,startoffset = choose_sequence_notsame(index+1,self.fileDuration,self.srInSec,self.stride)
				
			whole_sequence = load_sequence(self.filelist,chooseFileIndex,startoffset,self.seqLen+1,self.sr) 
			whole_sequence = whole_sequence.reshape(-1,1)
			assert len(whole_sequence) == self.seqLen+1, str(len(whole_sequence))
			sequence = whole_sequence[:-1]
			target = whole_sequence[1:]

		if self.transform is not None:
			input = self.transform(sequence)
		
		if self.target_transform is not None:
			target = self.target_transform(target)
		
		
		#for now if generating audio cannot generate anything else because of softmax output 
		if len(self.generate) > 1:
			self.generatelist = [n for n in self.generate if n != 'audio'] #grab the names of the other things to generate less audio
			pm = paramManager.paramManager(self.datadir, self.paramdir)
			params = pm.getParams(self.filelist[chooseFileIndex])
			generatedict = pm.resampleAllParams(params,self.seqLen+1,startoffset,startoffset+self.seqLenInSec,self.generatelist,verbose=False)
			if self.param_transform is not None:
				generatetensor = self.param_transform(generatedict)
				input = torch.cat((input,generatetensor[:-1]),1)
				target = torch.cat((target, generatetensor[1:]),1)
		"""



class ParamDataset(data.Dataset):	 
	def __init__(self, datadir, sr, seqLen, stride, 
				paramdir, prop, generate, extension, is_seeded, load_start,
				param_transform=None):
		"""
		sr: standardized sample rate of parameters in dataset. For parameters with different sr, will be resampled to a common value. 
		seqLen: sequence length of each input data in no. of samples. must be less than sr*duration
		stride: shift in no. of samples between adjacent data sequences. (seqLen - stride) samples will overlap between adjacent sequences
		datadir: root audio data directory
		extension: file extension of data
		paramdir: directory of parameter files
		prop: list of parameter keys to be used for input conditioning
		generate: list of parameters key/audio to be generated
		is_seeded: Boolean to specify whether to load data from seed file
		load_start: If is_seeded, specify start time point to load file
		transform: list of transformations on input audio sequence. use with torchvision.transforms.Compose() for multiple
		param_transform: list of transformations on parameters. use with torchvision.transforms.Compose() for multiple
		target_transform: list of transformations on target audio sequence. use with torchvision.transforms.Compose() for multiple 
		"""
		super(ParamDataset, self).__init__()
		if stride < 1:
			raise ValueError("stride has to be >= 1")
			
		self.filelist,self.fnamelist = listDirectory_all(directory=datadir,fileExtList=extension)
		self.datadir = datadir
		self.paramdir= paramdir
		self.prop = prop
		self.generate = generate
		self.is_seeded = is_seeded
		self.load_start = load_start
		self.fileLen,self.fileDuration,self.totalFileDuration,self.totalSamples,self.srInSec,self.seqLenInSec = dataset_properties(self.filelist,sr,seqLen)
		self.sr = sr
		self.seqLen = seqLen
		self.stride = stride
		self.param_transform = param_transform
		self.indexLen = create_sampling_index(self.totalSamples,self.stride)


	def __getitem__(self,index):
		generatedict, paramdict = self.rand_sample(index)
		generatetensor = self.param_transform(generatedict) #tensor containing params to be generated
		if len(self.prop)>0: 
			paramtensor = self.param_transform(paramdict) #tensor containing conditional params
			fulltensor = torch.cat((generatetensor,paramtensor),1)
		else:
			fulltensor = generatetensor
		if self.is_seeded:
			input = fulltensor
			target = generatetensor
		else:
			input = fulltensor[:-1]
			target = generatetensor[1:]

		return input, target

	def __len__(self):
		return self.indexLen 


	def rand_sample(self,index=None,verbose=False):
		if index is None:
			index = np.random.randint(self.indexLen)
		self.chooseFileIndex,self.startoffset = choose_sequence_notsame(index+1,self.fileDuration,self.srInSec,self.stride)
		if self.is_seeded:
			self.startoffset = self.load_start/self.sr
			self.load_length = self.seqLen
		else:
			self.load_length = self.seqLen+1
			while self.fileDuration[self.chooseFileIndex] < (self.startoffset + self.seqLenInSec + 1/self.sr):
				index = np.random.randint(self.indexLen)
				self.chooseFileIndex,self.startoffset = choose_sequence_notsame(index+1,self.fileDuration,self.srInSec,self.stride)	
		if verbose:
			print('loading part of file:',self.filelist[self.chooseFileIndex],'starting at',self.startoffset)

		pm = paramManager.paramManager(self.datadir, self.paramdir)
		self.params = pm.getParams(self.filelist[self.chooseFileIndex])
		generatedict = pm.resampleAllParams(self.params,self.load_length,self.startoffset,self.startoffset+self.seqLenInSec,self.generate,verbose=False)
		if len(self.prop)>0: 
			paramdict = pm.resampleAllParams(self.params,self.load_length,self.startoffset,self.startoffset+self.seqLenInSec,self.prop,verbose=False)

		return generatedict, paramdict



class DataLoader(data.DataLoader):
	def __init__(self, datadir, sr=16000, seqLen=512, stride=1, 
				paramdir=None, prop=None, generate=None, extension='.wav',
				mulaw_channels=256,
				batch_size=1, shuffle=True, num_workers=4, onehot=False,
				is_seeded=False,load_start=0):

		assert set(prop).isdisjoint(generate), 'Cannot repeat keywords in both prop and generate!'
		param_transform_list = []
		#if 'spec_centroid' in prop:
		#	param_transform_list.append(tr.normalizeDim('spec_centroid',0,8000))  #nyquist sr/2 
		
		param_transform_list.append(tr.dic2tensor(torch.FloatTensor))

		if 'audio' in generate:
			assert len(generate) == 1, 'Audio generation is incompatiple with simultaneous generation of other parameters!'
			if onehot:
				audio_transform_list =  [tr.mulawEncode(mulaw_channels,norm=False),tr.onehotEncode(mulaw_channels),tr.array2tensor(torch.FloatTensor)]
			else:
				audio_transform_list =  [tr.mulawEncode(mulaw_channels,norm=True),tr.array2tensor(torch.FloatTensor)]
			
			self.dataset = AudioDataset(datadir, sr, seqLen, stride,
					paramdir, prop, generate, extension, is_seeded, load_start,
					transform=transform.Compose(audio_transform_list),
					param_transform=transform.Compose(param_transform_list),
					target_transform=transform.Compose([tr.mulawEncode(mulaw_channels),tr.array2tensor(torch.LongTensor)]))

		else:
			assert paramdir is not None, 'Please provide [paramdir]!'
			#assert prop is not None, 'Please provide parameters to be used [prop]!'

			self.dataset = ParamDataset(datadir, sr, seqLen, stride,
					paramdir, prop, generate, extension, is_seeded, load_start,
					param_transform=transform.Compose(param_transform_list))

		super(DataLoader, self).__init__(self.dataset, batch_size, shuffle, num_workers=num_workers)




"""
for testing

loader = DataLoader(datadir="dataset",   
                    sr=16000,seqLen=10,stride=1,
                    batch_size=1,
                    paramdir="param",
                    prop=['rmse', 'centroid'],
                    generate=['audio'],is_seeded=False,load_start=0)

print(len(loader))
for step, (inputs,targets) in enumerate(loader):
    print(step)
    print(inputs.shape)
    print(inputs)
    print(targets.shape)

    if step == 1:
        break
"""

