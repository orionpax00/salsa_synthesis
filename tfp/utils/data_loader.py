import os
import glob
import math
import json
import numpy as np

from tfp.config import get_cfg_defaults


class LoadData(object):
	"""
		This class generate Training data using the folder created by getData class

		returns an array of shape (None, seg_len, num_joints, 3)
	"""

	def __init__(self,config):
		self.config = config

	def getdata(self):

		full_data = []

		for file in os.listdir(self.config.DATA.DATA_LOC):
			data = np.load(os.path.join(self.config.DATA.DATA_LOC, file))
			full_data.append(data)

		return full_data




