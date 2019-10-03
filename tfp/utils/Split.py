import os
import glob
import math
import json
import numpy as np

#module level imports
from tfp.config.config import SPLIT_JSON_LOC

class Split:
		
	def __init__(self,location=None,sequence_length = 100, overlap = 0,split_size=20):
		# pramaters 
		self.folder_location = location
		self.split_size = split_size
		self.overlap = overlap
		self.seq_len = sequence_length
		self.split_string = str(sequence_length) + "_" + str(overlap) + "_" + str(split_size)
		self.strides = self.seq_len - math.ceil(self.overlap * self.seq_len / 100)
		## function
		self.getfiles = self.get_files()
		self.checkcomb = self.check_comb()
	
	def check_comb(self):
		## json file should be present in config folder
		found = False
		file = SPLIT_JSON_LOC
		with open(file) as jsonfile:
			data = json.load(jsonfile)
			if self.split_string in data.keys():
				found = True
			else:
				found = False
		return found

			
			
	def get_files(self):
		all_numpy_files_loc = [x for x in os.listdir(self.folder_location) if x[-3:] == "npy" or x[-3:] == "npz"]
		return np.asarray(all_numpy_files_loc)

	def gen_split(self):
		""" function to divide trails into train trials and split trails"""
		
		files = self.getfiles() 
		num_test_trails = math.floor(len(files) * self.split_size)
		shuffled = np.random.shuffle(files)
		train_trails = shuffled[num_test_trails:]
		test_trails = shuffled[:num_test_trails]
        
		return train_trails, test_trails
	def split(self):
		if self.checkcomb:
			with open(SPLIT_JSON_LOC) as jsonfile:
				data = json.load(jsonfile)
				train_split = data[self.split_string]['train_splits']
				test_split = data[self.split_string]['test_splits']
		else:
			train_splits, test_splits = self.gen_split()
			with open(SPLIT_JSON_LOC,'rw') as jsonfile:
				data = json.load(jsonfile)
				data[self.split_string] = { "train_splits" : train_splits, "test_splits":test_splits}
				json.dump(data,jsonfile)
		
		comp_data = []
		for file in train_split:
			loc = os.path.join(self.folder_location,file)
			data = np.load(loc)
			if data.shape[0] < self.seq_len:
				break
			#data = data[:-(data.shape[0] % self.seq_len)
				
			num_bat = (data.shape[0] - self.seq_len)//(self.stride) + 1
			for i in range(num_bat):
				comp_data.append(data[i * self.stride : self.seq_length + i * self.stride])

		comp_data.random.shuffle(comp_data)

		np.save(os.path.join(self.folder_location,'data','data.npy'), np.asarray(comp_data))

		return None
