"""
Generic Utility functions
"""
import json
import os
import numpy as np

def removeDuplicateColumns(df):
	"""
	Removes duplicate columns from the passed dataframe
	Looks for columns that end with _x or _y
	"""
	cols = [c for c in df.columns if c[-2:] != '_y']
	df = df[cols]
	df = df.rename(columns=lambda x: x if x[-2:] != '_x' else x.replace('_x', ''))
	return df


def removeTrailingColumnNumbering(column_list):
	"""
	When pandas finds columns with same name, it numbers them
	This function receives a list of column names and removes the numbering if found
	Looks for columns that end with .1, .2, .3 and so on
	"""
	import re
	tmp = []
	for s in column_list:
		x = re.search('\.{1}\d+',s)
		if x != None:
			i = x.span()[0] #index of the .
			tmp.append(s[:i])
		else:
			tmp.append(s)

	return tmp

def createDestinationPath(destinationPath):
	"""
	Create folder at destinationPath for saving outputs
	"""

	if not isinstance(destinationPath, str):
		raise TypeError('destinationPath must be a string')

	# Create directory to save destinationPath
	if not os.path.exists(destinationPath):
		os.makedirs(destinationPath)

	if not os.path.exists(os.path.join(destinationPath, 'graphics')):
		os.makedirs(os.path.join(destinationPath, 'graphics'))



def sampleClassMasks(sampleMetadata):
	"""
	Returns a dictionary of boolean array defining locations of samples in each sampleclass.

	:return: key: value pairs, SampleClass: boolean array of location in data
	:rtype: dict
	"""

	sampleClassMasks = {}

	# If SampleClass available
	if hasattr(sampleMetadata, 'SampleClass'):
		stypes = sampleMetadata['SampleClass'].unique()

		for stype in stypes:
			sampleClassMasks[stype] = sampleMetadata['SampleClass'] == stype

	# Otherwise set all to unknown
	else:
		sampleClassMasks['Unknown'] = np.zeros(sampleMetadata.shape[0]).astype(bool)

	return sampleClassMasks