import os
import glob
import math


class Split:
		
	def __init__(self,location=None,sequence_length = 100, overlap = 0,split_size=0.2):
		# pramaters 
		self.folder_location = location
		self.split_size = split_size
		self.overlap = overlap
		self.seq_len = sequence_length
		
		## function
		self.getfiles = self.get_files()
	
	def get_files(self):
		all_numpy_files_loc = [x if x[-3:] == "npy" or x[-3:] == "npz" for x in os.listdir(self.loaction)]
		return all_numpy_files_loc

	def gen_split(self):
		""" function to divide trails into train trials and split trails"""
		files = self.getfiles() 
		num_test_trails = math.floor(len(files) * self.split_size)
		
		
			
