
import numpy as np

import glob
import os
import sys
import re

#======================================================================

# Some utility functions
#======================================================================

#General use
#*************************************
def plot_signal(result,start=None,figsize=(20,1),grid=False,logy=False,start_min_max=[-1.,1.],save=False,savename='1',ssh=True):
	"""Convenience function to plot signals.
	start = a float or list of floats containing the value(s) where a vertical line will be plotted
	"""
	if ssh:
		import matplotlib
		matplotlib.use('Agg') 
	import matplotlib.pylab as plt
	fig, ax = plt.subplots(figsize=figsize)
	if logy:
		ax.semilogy(np.arange(len(result)), result)
	else:
		ax.plot(np.arange(len(result)), result) #just print one example from the batch
	if start is not None:
		if isinstance(start, (list,)):
			for value in start: 
				ax.vlines(x=value,ymin=start_min_max[0], ymax=start_min_max[1], color='r')
		else:
			ax.vlines(x=start,ymin=start_min_max[0], ymax=start_min_max[1], color='r')
	if grid:
		plt.grid()
	if save:
		fig.savefig(savename + '.png', format='png', bbox_inches = 'tight')
	else:
		plt.show()
	plt.close(fig)

def save_obj(obj, dir, name):
	"""to save an obj using pickle"""
	import pickle
	with open(dir + '/' + name + '.pkl', 'wb+') as f:
		pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(dir, name):
	"""to load a pickled object"""
	import pickle
	with open(dir + '/' + name + '.pkl', 'rb') as f:
		return pickle.load(f)

def chunker(seq, size):
	"""returns non-overlapping chunks of length size for given sequence.
	can be used in a loop -
	A = 'ABCDEFG'
	for group in chunker(A, 2):
		print(group)
	-> 'AB' 'CD' 'EF' 'G'"""
	return (seq[pos:pos + size] for pos in range(0, len(seq), size))

def findMinMax(input):
	"""return min and max of input array"""
	return np.amin(input),np.amax(input)

def config_parser(config_file,namespace=False):
	"""parse a json file into python with namespace variables"""
	import json
	try:
		from types import SimpleNamespace as Namespace
	except ImportError:
		from argparse import Namespace  # Python 2.x fallback
	
	with open(config_file) as f:
		if namespace:
			var = json.load(f, object_hook=lambda d: Namespace(**d))
		else:
			var = json.load(f)
	return var

def dictfilter(dic, keys):
	return {key:dic[key] for key in keys}

def print_both(file, *args):
	"""prints and write to file simultaneously"""
	toprint = ' '.join([str(arg) for arg in args])
	print(toprint)
	file.write(toprint+'\n')

#timing
#*************************************
class Timer:
	"""class to find elapsed time"""
	def __init__(self):
		from datetime import datetime
		self.cache = datetime.now()

	def check(self):
		now = datetime.now()
		duration = now - self.cache
		self.cache = now
		return duration.total_seconds()

	def reset(self):
		self.cache = datetime.now()

def time_taken(elapsed):
	"""To format time taken in hh:mm:ss. Use with time.monotic() or Timer class"""
	m, s = divmod(elapsed, 60)
	h, m = divmod(m, 60)
	return "%d:%02d:%02d" % (h, m, s)

def mydate():
	"""returns current date and time"""
	from datetime import datetime
	return (datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


class ProgressBar(object):
	"""incrementing progress bar
	Credits to: Romuald Brunet from https://stackoverflow.com/questions/3160699/python-progress-bar"""
	DEFAULT = 'Progress: %(bar)s %(percent)3d%%'
	FULL = '%(bar)s %(current)d/%(total)d (%(percent)3d%%) %(remaining)d to go'

	def __init__(self, total, width=40, fmt=DEFAULT, symbol='=',
				 output=sys.stderr):
		assert len(symbol) == 1

		self.total = total
		self.width = width
		self.symbol = symbol
		self.output = output
		self.fmt = re.sub(r'(?P<name>%\(.+?\))d',
			r'\g<name>%dd' % len(str(total)), fmt)

		self.current = 0

	def __call__(self):
		percent = self.current / float(self.total)
		size = int(self.width * percent)
		remaining = self.total - self.current
		bar = '[' + self.symbol * size + ' ' * (self.width - size) + ']'

		args = {
			'total': self.total,
			'bar': bar,
			'current': self.current,
			'percent': percent * 100,
			'remaining': remaining
		}
		print('\r' + self.fmt % args, file=self.output, end='')
		self.current += 1

	def done(self):
		self.current = self.total
		self()
		print('', file=self.output)


#file/folder manipulation
#*************************************
def mostRecent(strPat) :
	files = glob.glob(strPat)
	files.sort(key=os.path.getmtime)
	return files[-1]
	
def listDirectory(directory,fileExtList='.wav',regex=None):										 
	"""returns a list of file info objects in directory that contains extension in the list fileExtList (include the . in your extension string)
	and regex if specified
	fileList - fullpath from working directory to files in directory
	fnameList - basenames in directory (including extension)
	regex - a substring in the filename, if unspecified will list all files in directory"""	
	if regex is not None:
		fnameList = [os.path.normcase(f)
				for f in os.listdir(directory)
					if (not(f.startswith('.')) and (regex in f))] 
	else:
		fnameList = [os.path.normcase(f)
				for f in os.listdir(directory)
					if (not(f.startswith('.')))] 
	
	fileList = [os.path.join(directory, f) 
			   for f in fnameList
				if os.path.splitext(f)[1] in fileExtList]  
	return fileList , fnameList

def listDirectory_all(directory,fileExt='.wav',topdown=True):
    """returns a list of all files in directory and all its subdirectories
	directory can also take a single file.
    fileList: full path to file
    fnameList: basenames
    fnameList_noext: basenames with no extension"""
    fileList = []
    fnameList = []
    fnameList_noext = []
    if os.path.isdir(directory):
        for root, _, files in os.walk(directory, topdown=topdown):
            for name in files:
                if name.endswith(fileExt):
                    fileList.append(os.path.join(root, name))
                    fnameList.append(name)
                    fnameList_noext.append(os.path.splitext(name)[0])
    else:
        if directory.endswith(fileExt):
            fileList.append(directory)
            basename = os.path.basename(directory)
            fnameList.append(basename)
            fnameList_noext.append(os.path.splitext(basename)[0])       
    return fileList, fnameList, fnameList_noext

def mass_delete(directory,regex,topdown=True):
	"""deletes all files matching regex in directory and all its subdirectories"""
	deleteList = []
	for root, _, files in os.walk(directory, topdown=topdown):
		for name in files:
			if regex in name:
				deleteList.append(name)
				os.remove(os.path.join(root, name))
	return deleteList

def find_classes(dir):
	"""
	Finds the class folders in a dataset.
	input params: dir (string)- Root directory path.
	Returns: tuple (classes, class_to_idx) - where classes are relative to (dir), and class_to_idx is a dictionary.
	Ensures: No class is a subdirectory of another.
	"""
	if isinstance(dir, str):
		if sys.version_info >= (3, 5):
			# Faster and available in Python 3.5 and above
			classes = [d.name for d in os.scandir(dir) if d.is_dir()]
		else:
			classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
		classes.sort()
		class_to_idx = {classes[i]: i for i in range(len(classes))}
	elif isinstance(dir, (list,)):
		classes = [i for i in range(len(dir))]
		class_to_idx = {classes[i]: i for i in classes}
	return classes, class_to_idx

#nsynth dataset
#*************************************
def extract_nsynth_pitch(filename):
	"""function to extract midi pitch from nsynth dataset base filenames
	keyboard_acoustic_000-059-075.wav -> 59""" 
	n = re.findall(r'(?<=-).*?(?=-)', filename)[0]
	if (n[0]=='0') :
		midinum=int(n[1:])
	else :
		midinum=int(n)
	return midinum

def extract_nsynth_instrument(filename):
	"""function to extract midi pitch from nsynth dataset base filenames
	keyboard_acoustic_000-059-075.wav -> keyboard_acoustic""" 
	n = re.search('(.*?_.*?)_', filename).group(1)
	return n

def extract_nsynth_volume(filename):
	"""function to extract volume level from nsynth dataset base filenames
	keyboard_acoustic_000-059-075.wav -> 75""" 
	return filename[-6:-4]


#audio transforms
#*************************************
def MinMaxScaling(input,min,max,a=-1,b=1):
	"""rescales input array from [min,max] -> [a,b]"""
	output = a + (input-min)*(b-a)/(max-min)
	return output

def ScaleAudio(file,outdir,scaling_function,*args,**kwargs):
	import soundfile as sf
	"""loads file, applies scaling_function, saves file in specified directory
	file: path to file
	outdir: path to directory where scaled file will be saved"""
	y,samplerate = sf.read(file)
	scaled_y = scaling_function(y,*args,**kwargs)
	try:
		os.stat(outdir) # test for existence
	except:
		os.mkdir(outdir) # create if necessary   
	basename = os.path.basename(file)
	return sf.write(outdir + '/' + basename, scaled_y, samplerate)

def ScaleAudio_all(indir,outdir,scaling_function,*args,**kwargs):
	"""Wraps ScaleAudio to run on all files in indir.
	Outdir preserves folder structure of indir"""
	for dirpath, _, filenames in os.walk(indir):
		if filenames:
			for name in filenames:
				structure = os.path.join(outdir, os.path.relpath(dirpath, indir))
				if not os.path.isdir(structure):
					os.mkdir(structure)
				ScaleAudio(os.path.join(dirpath, name),structure,scaling_function,*args,**kwargs)


#neural network training
#*************************************
def LinearScheduler(start,end,steps):
	"""yields a value that decays linearly with steps.
	used for scheduled sampling"""
	for i in np.linspace(start,end,steps):
		yield i