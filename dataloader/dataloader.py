"""
Audio dataloader is a chunker and dataloader that loads a part of an audio clip 
plus corresponding parameters (if any) from a param file. To be used with pytorch platform.

to load each group containing: 1 audio wav file
							   1 parameter json file
filenames will be contained within a csv file (1 group per line), or directly loaded from a directory
1. parse csvfile/directory. get path of all objects in each line and append to list
2. create a list of indices to draw samples (eg. index 52 = 4th wav file sample 390). this no. will also be the __len__
3. __getitem__:
	sample index
	load corresponding wav file
	pull out correct audio sample sequence
	load corresponding params
	convert audio to mu-law
	convert mu-law + params to tensor

@muhammad huzaifah 27/08/2019

Notes: Eventually hope to migrate to torchaudio for loading/transformations
		Currently torchaudio dependency libsox is broken for windows
"""

import torch
import torch.utils.data as data
import torchvision.transforms as transform

import os
import csv
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
	
def load_sequence(filelist,chooseFileIndex,startoffset,seqLen,sr):
	"""load the correct section of audio. If len of audio < seqLen+1 (e.g. sections at the end of the file), then draw another section.
	We draw 1 sample more than seqLen so can take input=y[:-1] and target=y[1:]"""
	y,_ = sf.read(filelist[chooseFileIndex],frames=seqLen+1,start=round(startoffset*sr))			
	if len(y) < seqLen+1:
		y = None
	return y


class AudioDataset(data.Dataset):	 
	def __init__(self, datadir, sr, seqLen, stride, 
				paramdir=None, prop=None, extension='.wav', 
				transform=None, param_transform=None, target_transform=None):
		"""
		sr: sample rate of audio files in dataset
		seqLen: sequence length of each input data in no. of samples. must be less no. samples in each audio file
		stride: shift in no. of samples between adjacent data sequences. (seqLen - stride) samples will overlap between adjacent sequences
		datadir: root data directory
		extension: file extension of data. default='.wav'
		paramdir: directory of parameter files
		prop: list of parameter keys to be used for input conditioning
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
		self.fileLen,self.fileDuration,self.totalFileDuration,self.totalSamples,self.srInSec,self.seqLenInSec = dataset_properties(self.filelist,sr,seqLen)
		self.sr = sr
		self.seqLen = seqLen
		self.stride = stride
		self.transform = transform
		self.param_transform = param_transform
		self.target_transform = target_transform
		self.indexLen = create_sampling_index(self.totalSamples,self.stride)
			
	def __getitem__(self,index):
		chooseFileIndex,startoffset = choose_sequence_notsame(index+1,self.fileDuration,self.srInSec,self.stride)
		whole_sequence = load_sequence(self.filelist,chooseFileIndex,startoffset,self.seqLen,self.sr)
		while whole_sequence is None: #if len(whole_sequence) < self.seqLen+1, pick another random section
			index = np.random.randint(self.indexLen)
			chooseFileIndex,startoffset = choose_sequence_notsame(index+1,self.fileDuration,self.srInSec,self.stride)
			whole_sequence = load_sequence(self.filelist,chooseFileIndex,startoffset,self.seqLen,self.sr) 
		assert len(whole_sequence) == self.seqLen+1, str(len(whole_sequence))
		whole_sequence = whole_sequence.reshape(-1,1)
		sequence = whole_sequence[:-1]
		target = whole_sequence[1:]

		if self.transform is not None:
			input = self.transform(sequence)
		if self.target_transform is not None:
			target = self.target_transform(target)
		if self.paramdir is not None:
			pm = paramManager.paramManager(self.datadir, self.paramdir)
			params = pm.getParams(self.filelist[chooseFileIndex]) 
			paramdict = pm.resampleAllParams(params,self.seqLen,startoffset,startoffset+self.seqLenInSec,self.prop,verbose=False)
			if self.param_transform is not None:
				paramtensor = self.param_transform(paramdict)
				input = torch.cat((input,paramtensor),1)  #input dim: (batch,seq,feature)	
		else:
			if self.transform is None:
				input = torch.from_numpy(sequence).type(torch.FloatTensor)	

		return input, target

	def __len__(self):
		return self.indexLen 
		
	def rand_sample(self,index=None,transform=False):
		whole_sequence = None
		while whole_sequence is None:
			if index is None:
				index = np.random.randint(self.indexLen)
			chooseFileIndex,startoffset = choose_sequence_notsame(index+1,self.fileDuration,self.srInSec,self.stride)
			print('loading part of file:',self.filelist[chooseFileIndex])	
			whole_sequence = load_sequence(self.filelist,chooseFileIndex,startoffset,self.seqLen,self.sr)
		#whole_sequence = whole_sequence.reshape(-1,1)
		sequence = whole_sequence[:-1]
		if transform:
			trans_sequence = self.transform(sequence)
			return sequence, trans_sequence 
		return sequence


class ParamDataset(data.Dataset):	 
	def __init__(self, datadir, sr, seqLen, stride, 
				paramdir, prop, extension='.wav', 
				param_transform=None):
		"""
		sr: standardized sample rate of parameters in dataset. For parameters with different sr, will be resampled to a common value. 
		seqLen: sequence length of each input data in no. of samples. must be less than sr*duration
		stride: shift in no. of samples between adjacent data sequences. (seqLen - stride) samples will overlap between adjacent sequences
		datadir: root audio data directory
		extension: file extension of data. default='.wav'
		paramdir: directory of parameter files
		prop: list of parameter keys to be used for input conditioning
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
		self.fileLen,self.fileDuration,self.totalFileDuration,self.totalSamples,self.srInSec,self.seqLenInSec = dataset_properties(self.filelist,sr,seqLen)
		self.sr = sr
		self.seqLen = seqLen
		self.stride = stride
		self.param_transform = param_transform
		self.indexLen = create_sampling_index(self.totalSamples,self.stride)
			
	def __getitem__(self,index):
		chooseFileIndex,startoffset = choose_sequence_notsame(index+1,self.fileDuration,self.srInSec,self.stride)

		while self.fileDuration[chooseFileIndex] < (startoffset + self.seqLenInSec):
			index = np.random.randint(self.indexLen)
			chooseFileIndex,startoffset = choose_sequence_notsame(index+1,self.fileDuration,self.srInSec,self.stride)
		
		pm = paramManager.paramManager(self.datadir, self.paramdir)
		params = pm.getParams(self.filelist[chooseFileIndex]) 
		paramdict = pm.resampleAllParams(params,self.seqLen,startoffset,startoffset+self.seqLenInSec,self.prop,verbose=False)

		paramtensor = self.param_transform(paramdict)
		input = paramtensor[:-1]
		target = paramtensor[1:]

		return input, target

	def __len__(self):
		return self.indexLen 
		
	def rand_sample(self,index=None,transform=False):
		if index is None:
			index = np.random.randint(self.indexLen)
		chooseFileIndex,startoffset = choose_sequence_notsame(index+1,self.fileDuration,self.srInSec,self.stride)

		while self.fileDuration[chooseFileIndex] < (startoffset + self.seqLenInSec):
			index = np.random.randint(self.indexLen)
			chooseFileIndex,startoffset = choose_sequence_notsame(index+1,self.fileDuration,self.srInSec,self.stride)
		
		print('loading part of file:',self.filelist[chooseFileIndex],'starting at',startoffset)
		pm = paramManager.paramManager(self.datadir, self.paramdir)	
		params = pm.getParams(self.filelist[chooseFileIndex]) 
		paramdict = pm.resampleAllParams(params,self.seqLen,startoffset,startoffset+self.seqLenInSec,self.prop,verbose=False)

		if transform:
			trans_sequence = self.param_transform(paramdict)
			return paramdict, trans_sequence 
		return paramdict


class DataLoader(data.DataLoader):
	def __init__(self, datadir, sr=16000, seqLen=512, stride=1, 
				paramdir=None, prop=None, extension='.wav',
				mulaw_channels=256,
				batch_size=1, shuffle=True, num_workers=4, paramonly=False):

		param_transform_list = []
		if 'spec_centroid' in prop:
			param_transform_list.append(tr.normalizeDim('spec_centroid',0,8000))  #nyquist sr/2 
		
		param_transform_list.append(tr.dic2tensor(torch.FloatTensor))

		if paramonly:
			assert paramdir is not None, 'Please provide [paramdir]!'
			assert prop is not None, 'Please provide parameters to be used [prop]!'

			self.dataset = ParamDataset(datadir, sr, seqLen, stride,
					paramdir, prop, extension,
					param_transform=transform.Compose(param_transform_list))

		else:
			self.dataset = AudioDataset(datadir, sr, seqLen, stride,
					paramdir, prop, extension,
					transform=transform.Compose([tr.mulawEncode(mulaw_channels),tr.onehotEncode(mulaw_channels),tr.array2tensor(torch.FloatTensor)]),
					param_transform=transform.Compose(param_transform_list),
					target_transform=transform.Compose([tr.mulawEncode(mulaw_channels),tr.array2tensor(torch.LongTensor)]))

		super(DataLoader, self).__init__(self.dataset, batch_size, shuffle, num_workers=num_workers)


"""
from transforms import mulawnEncode,mulaw,array2tensor,dic2tensor	
sr = 16000
seqLen = 5
stride = 1

adataset = AudioDataset(sr,seqLen,stride,
			datadir="dataset",extension="wav",
			paramdir="dataparam",prop=['rmse','instID','midiPitch'],  
			transform=transform.Compose([mulawnEncode(256,0,1),array2tensor(torch.FloatTensor)]),
			param_transform=dic2tensor(torch.FloatTensor),
			target_transform=transform.Compose([mulaw(256),array2tensor(torch.LongTensor)]))

for i in range(len(adataset)):
	inp,target = adataset[i]
	print(inp)
	print(target)
	
	if i == 2:
		break 
"""

