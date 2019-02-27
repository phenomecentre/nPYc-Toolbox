import scipy
import pandas
import numpy
import pickle
import sys
import unittest
import tempfile
import os
import copy
import random
import string

sys.path.append("..")
import nPYc
from datetime import datetime, timedelta
from nPYc.enumerations import VariableType

def generateTestDataset(noSamp, noFeat, dtype='Dataset', variableType=VariableType.Discrete, sop='Generic'):
	"""
	Generate a dataset object with random sample and feature numbers, and random contents.

	.. warning:: Objects returned by this function are not expected to be fully functional!

	:param int noSamp: Number of samples
	:param int noFeat: Number of features
	:param VariableType variableType: Type of enumerations
	
	"""
	if dtype == 'Dataset':
		data = nPYc.Dataset(sop=sop)
	elif dtype == 'MSDataset':
		data = nPYc.MSDataset('', fileType='empty', sop=sop)
	elif dtype == 'NMRDataset':
		data = nPYc.NMRDataset('', fileType='empty', sop=sop)
	elif dtype  == 'TargetedDataset':
		data = nPYc.TargetedDataset('', fileType='empty', sop=sop)
	else:
		raise ValueError

	data.intensityData = numpy.random.lognormal(size=(noSamp, noFeat)) + 1

	data.sampleMetadata = pandas.DataFrame(0, index=numpy.arange(noSamp), columns=['Sample File Name', 'SampleType', 'AssayRole', 'Acquired Time', 'Run Order', 'Dilution', 'Detector', 'Correction Batch'])

	data.sampleMetadata['SampleType'] = nPYc.enumerations.SampleType.StudySample
	data.sampleMetadata['AssayRole'] = nPYc.enumerations.AssayRole.Assay
	data.sampleMetadata['Run Order'] = numpy.arange(noSamp)
	data.sampleMetadata['Detector'] = numpy.arange(noSamp) * 5
	data.sampleMetadata['Batch'] = 1
	data.sampleMetadata['Correction Batch'] = 2
	data.sampleMetadata.loc[0:int(noSamp / 2), 'Correction Batch'] = 1
	data.sampleMetadata['Exclusion Details'] = ''

	data.sampleMetadata['Sample File Name'] = [randomword(10) for x in range(0, noSamp)]
	data.sampleMetadata['Sampling ID'] = [randomword(10) for x in range(0, noSamp)]
	data.sampleMetadata['Dilution'] = numpy.random.rand(noSamp)

	noClasses = numpy.random.randint(2, 5)
	classNames = [str(i) for i in range(0, noClasses)]
	classProbabilties = numpy.random.rand(noClasses)
	classProbabilties = classProbabilties / sum(classProbabilties)

	data.sampleMetadata['Classes'] = numpy.random.choice(classNames, size=noSamp, p=classProbabilties)

	data.sampleMetadata['Acquired Time'] = [d for d in datetime_range(datetime.now(), noSamp, timedelta(minutes=15))]
	#Ensure seconds are not recorded, otherwise its impossible to test datasets read with datasets recorded on the fly.
	data.sampleMetadata['Acquired Time'] = [datetime.strptime(d.strftime("%Y-%m-%d %H:%M"), "%Y-%m-%d %H:%M") for d in data.sampleMetadata['Acquired Time']]
	data.sampleMetadata['Acquired Time'] = data.sampleMetadata['Acquired Time'].dt.to_pydatetime()

	data.sampleMetadata.iloc[::10, 1] = nPYc.enumerations.SampleType.StudyPool
	data.sampleMetadata.iloc[::10, 2] = nPYc.enumerations.AssayRole.PrecisionReference

	data.sampleMetadata.iloc[5::10, 1] = nPYc.enumerations.SampleType.ExternalReference
	data.sampleMetadata.iloc[5::10, 2] = nPYc.enumerations.AssayRole.PrecisionReference

	if dtype == 'MSDataset' or dtype == 'Dataset':
		data.featureMetadata = pandas.DataFrame(0, index=numpy.arange(noFeat), columns=['m/z'])

		data.featureMetadata['m/z'] = (800 - 40) * numpy.random.rand(noFeat) + 40
		data.featureMetadata['Retention Time'] = (720 - 50) * numpy.random.rand(noFeat) + 50
		data.featureMetadata['Feature Name'] = [randomword(10) for x in range(0, noFeat)]
		data.Attributes['Feature Names'] = 'Feature Name'

	elif dtype == 'NMRDataset':
		data.featureMetadata = pandas.DataFrame(numpy.linspace(10, -1, noFeat), columns=('ppm',), dtype=float)
		data.featureMetadata['Feature Name'] = data.featureMetadata['ppm'].astype(str)
		data.sampleMetadata['Delta PPM'] = numpy.random.rand(noSamp)
		data.sampleMetadata['Line Width (Hz)'] = numpy.random.rand(noSamp)
		data.sampleMetadata['CalibrationFail'] = False
		data.sampleMetadata['LineWidthFail'] = False
		data.sampleMetadata['WaterPeakFail'] = False
		data.sampleMetadata['BaselineFail'] = False

		data.Attributes['Feature Names'] = 'ppm'

	data.VariableType = variableType
	data.initialiseMasks()

	return data


def randomword(length):
	# Function to generate random strings:

	validChars = string.ascii_letters + string.digits
	return ''.join(random.choice(validChars) for i in range(length))


def datetime_range(start, count, delta):
	current = start
	for i in range(0, count):
		yield current
		current += delta
