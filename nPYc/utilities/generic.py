"""
Generic Utility functions
"""
import json
import os
import numpy as np
from ..enumerations import AssayRole, SampleType

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


def inferSampleClass(sampleMetadata):
	"""
	Infers `SampleClass` - standardised NPC types based on SampleType/AssayRole combinations

	:return: sampleMetadata with addition of column 'SampleClass', note if already present this will be overwritten
	"""
	sampleMetadata.loc[(sampleMetadata['SampleType'] == SampleType.StudySample) & (
			sampleMetadata['AssayRole'] == AssayRole.Assay), 'SampleClass'] = 'Study Sample'
	sampleMetadata.loc[(sampleMetadata['SampleType'] == SampleType.StudyPool) & (
		sampleMetadata['AssayRole'] == AssayRole.PrecisionReference), 'SampleClass'] = 'Study Reference'
	sampleMetadata.loc[(sampleMetadata['SampleType'] == SampleType.ExternalReference) & (
		sampleMetadata['AssayRole'] == AssayRole.PrecisionReference), 'SampleClass'] = 'Long-Term Reference'
	sampleMetadata.loc[(sampleMetadata['SampleType'] == SampleType.StudyPool) & (
		sampleMetadata['AssayRole'] == AssayRole.LinearityReference), 'SampleClass'] = 'Linearity Reference'
	sampleMetadata.loc[(sampleMetadata['SampleType'] == SampleType.MethodReference) & (
		sampleMetadata['AssayRole'] == AssayRole.PrecisionReference), 'SampleClass'] = 'Method Reference'
	sampleMetadata.loc[(sampleMetadata['SampleType'] == SampleType.ProceduralBlank) & (
		sampleMetadata['AssayRole'] == AssayRole.Blank), 'SampleClass'] = 'Blank'
	sampleMetadata.loc[(sampleMetadata['SampleType'] == SampleType.UnknownType) & (
		sampleMetadata['AssayRole'] == AssayRole.UnknownRole), 'SampleClass'] = 'Unknown'

	return sampleMetadata


def sampleClassMasks(sampleMetadata, on='SampleClass'):
	"""
	Returns a dictionary of boolean array defining locations of samples in each unique entry of column 'on', which must be a column name of sampleMetadata

	:return: key: value pairs, SampleClass: boolean array of location in data
	:rtype: dict
	"""

	sampleClassMasks = {}

	# If SampleClass available
	if hasattr(sampleMetadata, on):
		stypes = sampleMetadata[on].unique()

		for stype in stypes:
			sampleClassMasks[stype] = sampleMetadata[on] == stype

	# Otherwise set all to unknown
	else:
		sampleClassMasks['Unknown'] = np.zeros(sampleMetadata.shape[0]).astype(bool)

	return sampleClassMasks