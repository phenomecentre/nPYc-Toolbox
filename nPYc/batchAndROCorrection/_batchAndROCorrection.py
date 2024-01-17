#!/usr/bin/env python
# -*- coding: utf-8 -*-

import types
import numpy
import scipy
import warnings
from scipy.signal import savgol_filter
import statsmodels.api as sm
lowess = sm.nonparametric.lowess
import logging
from scipy.signal import savgol_filter
import time
import sys
import copy
from datetime import datetime, timedelta
from ..objects._msDataset import MSDataset
from ..enumerations import AssayRole, SampleType


def correctMSdataset(data,
					 window=11,
					 method='LOWESS',
					 align='median',
					 parallelise=True,
					 excludeFailures=True,
					 correctionSampleType=SampleType.StudyPool):
	"""
	Conduct run-order correction and batch alignment on the :py:class:`~nPYc.objects.MSDataset` instance *data*, returning a new instance with corrected intensity values.

	Sample are seperated into batches according to the *'Correction Batch'* column in *data.sampleMetadata*.

	Samples are only corrected if they have a value in *'Correction Batch'* AND an *'AssayRole'*/*'SampleType'*
	combination not defined in 'samplesNotCorrected' (taken from the sop/data.Attributes['samplesNotCorrected'])

	:param data: MSDataset object with measurements to be corrected
	:type data: MSDataset
	:param int window: When calculating trends, consider this many reference samples, centred on the current position
	:param str method: Correction method, one of 'LOWESS' (default), 'SavitzkyGolay' or None for no correction
	:param str align: Average calculation of batch and feature intensity for correction, one of 'median' (default) or 'mean'
	:param bool parallelise: If ``True``, use multiple cores
	:param bool excludeFailures: If ``True``, remove features where a correct fit could not be calculated from the dataset
	:param enum correctionSampleType: Which SampleType to use for the correction, default SampleType.StudyPool
	:return: Duplicate of *data*, with run-order correction applied
	:rtype: MSDataset
	"""
	import copy

	# Check inputs
	if not isinstance(data, MSDataset):
		raise TypeError("data must be a MSDataset instance")
	if not isinstance(window, int) & (window>0):
		raise TypeError('window must be a positive integer')
	if method is not None:
		if not isinstance(method, str) & (method in {'LOWESS', 'SavitzkyGolay'}):
			raise ValueError('method must be == LOWESS or SavitzkyGolay')
	if not isinstance(align, str) & (align in {'mean', 'median', 'no'}):
		raise ValueError('align must be == mean, median or no')
	if not isinstance(parallelise, bool):
		raise TypeError("parallelise must be a boolean")
	if not isinstance(excludeFailures, bool):
		raise TypeError("excludeFailures must be a boolean")
	if not isinstance(correctionSampleType,SampleType):
		raise TypeError("correctionType must be a SampleType")

	# Define the samples to be corrected (only corrected if have value in 'Correction Batch' and not listed for
	# exclusion in 'samplesNotCorrected'
	samplesForCorrection = data.sampleMetadata['Correction Batch'].values.astype(float)

	for s in numpy.arange(len(data.Attributes['samplesNotCorrected']['SampleType'])):
		try:
			mask = (data.sampleMetadata['SampleType'] == SampleType[data.Attributes['samplesNotCorrected']['SampleType'][s]]) & \
				   (data.sampleMetadata['AssayRole'] == AssayRole[data.Attributes['samplesNotCorrected']['AssayRole'][s]])
			samplesForCorrection[mask] = numpy.nan
		except KeyError:
			raise KeyError('data.Attributes[\'samplesNotCorrected\'] must contain valid SampleType/AssayRole enumeration entries')

	with warnings.catch_warnings():
		warnings.simplefilter('ignore', category=RuntimeWarning)

		correctedP = _batchCorrectionHead(data.intensityData,
									 data.sampleMetadata['Run Order'].values,
									 (data.sampleMetadata['SampleType'].values == correctionSampleType) & (data.sampleMetadata['AssayRole'].values == AssayRole.PrecisionReference),
									 samplesForCorrection,
									 window=window,
									 method=method,
									 align=align,
									 parallelise=parallelise)

	correctedData = copy.deepcopy(data)
	correctedData.intensityData = correctedP[0]
	correctedData.fit = correctedP[1]
	correctedData.Attributes['Log'].append([datetime.now(),'Batch and run order correction applied'])

	return correctedData


def _batchCorrectionHead(data, runOrder, referenceSamples, batchList, window=11, method='LOWESS', align='median', parallelise=True, savePlots=False):
	"""
	Conduct run-order correction and batch alignment.

	:param data: Raw *n* × *m* numpy array of measurements to be corrected
	:type data: numpy.array
	:param runOrder: *n* item list of order of analysis
	:type runOrder: numpy.series
	:param referenceSamples: *n* element boolean array indicating reference samples to base the correction on
	:type referenceSamples: numpy.series
	:param batchList: *n* item list of correction batch, defines sample groupings into discrete batches for correction
	:type batchList: numpy.series
	:param int window: When calculating trends, use a consider this many reference samples, centred on the current position
	:param str method: Correction method, one of 'LOWESS' (default), 'SavitzkyGolay' or None for no correction
	:param str align: Average calculation of batch and feature intensity for correction, one of 'median' (default) or 'mean'
	"""
	# Validate inputs
	if not isinstance(data, numpy.ndarray):
		raise TypeError('data must be a numpy array')
	if not isinstance(runOrder, numpy.ndarray):
		raise TypeError('runOrder must be a numpy array')
	if not isinstance(referenceSamples, numpy.ndarray):
		raise TypeError('referenceSamples must be a numpy array')
	if not isinstance(batchList, numpy.ndarray):
		raise TypeError('batchList must be a numpy array')
	if not isinstance(window, int) & (window>0):
		raise TypeError('window must be a positive integer')
	if method is not None:
		if not isinstance(method, str) & (method in {'LOWESS', 'SavitzkyGolay'}):
			raise ValueError('method must be == LOWESS or SavitzkyGolay')	
	if not isinstance(align, str) & (align in {'mean', 'median', 'no'}):
			raise ValueError('align must be == mean, median or no')
	if not isinstance(parallelise, bool):
		raise TypeError('parallelise must be True or False')
	if not isinstance(savePlots, bool):
		raise TypeError('savePlots must be True or False')

	# Store paramaters in a dict to avoid arg lists going out of control
	parameters = dict()
	parameters['window'] = window
	parameters['method'] = method
	parameters['align'] = align

	if parallelise:
		# Set up multiprocessing enviroment
		import multiprocessing
		
		# Generate an index and set up pool
		# Use one less workers than CPU cores
		if multiprocessing.cpu_count()-1 <= 0:
			cores = 1
		else: 
			cores = multiprocessing.cpu_count()-1

		pool = multiprocessing.Pool(processes=cores)

		instances = range(0, cores)

		# Break features into no cores chunks
		featureIndex = _chunkMatrix(range(0, data.shape[1]), cores)

		# run _batchCorection
		##
		# At present pickle args and returns and reassemble  after - possiblly share memory in the future.
		##
		results2 = [pool.apply_async(_batchCorrection, args=(data, runOrder, referenceSamples, batchList, featureIndex, parameters, w)) for w in instances]

		results2 = [p.get(None) for p in results2]

		results = list()
		# Unpack results
		for instanceOutput in results2:
			for item in instanceOutput:
				results.append(item)

		# Shut down the pool
		pool.close()


	else:
		# Just run it
		# Iterate over features in one batch and correct them
		results = _batchCorrection(data, 
								   runOrder,
								   referenceSamples,
								   batchList,
								   range(0, data.shape[1]), # All features
								   parameters,
								   0)

	correctedData = numpy.empty_like(data)
	fits = numpy.empty_like(data)

	# Extract return values from tuple
	for (w, feature, fit) in results:
		correctedData[:, w] = feature
		fits[:, w] = fit

	return (correctedData, fits)


def _batchCorrection(data, runOrder, QCsamples, batchList, featureIndex, parameters, w):
	"""
	Break the dataset into batches to be corrected together.
	"""

	# Check if we have a list of lists, or just one list:
	if isinstance(featureIndex[0], range):
		featureList = featureIndex[w]
	else:
		featureList = range(0, len(featureIndex))

	# add results to this list:
	results = list()
	
	# Loop over all elements in featureList
	for i in featureList:

		# Create a matrix to be used with `nonlocal` to store fits
		try:
			feature = copy.deepcopy(data[:,i])
		except IndexError:
			feature = copy.deepcopy(data)
		fit = numpy.empty_like(feature)
		fit.fill(numpy.nan)
			
		# Identify number of unique batches
		batches = list(set(batchList))

		# Get overall average intensity
		if parameters['align'] == 'mean':
			featureAverage = numpy.mean(feature[QCsamples])
		elif parameters['align'] == 'median':
			featureAverage = numpy.median(feature[QCsamples])
		#else:
			#return numpy.zeros_like(data)
				
		# Iterate over batches.
		for batch in batches:
			# Skip the NaN batch
			if not numpy.isnan(batch):

				batchMask = numpy.squeeze(numpy.asarray(batchList == batch, 'bool'))

				if parameters['method'] == None:
					# Skip RO correction if method is none
					pass
				else:

					(feature[batchMask], fit[batchMask]) = runOrderCompensation(feature[batchMask],
																			runOrder[batchMask],
																			QCsamples[batchMask],
																			parameters)

				# Correct batch average to overall feature average
				if parameters['align'] == 'mean':
					batchMean = numpy.mean(feature[batchMask & QCsamples])
				elif parameters['align'] == 'median':
					batchMean = numpy.median(feature[batchMask & QCsamples])
				#else:
				#	batchMean = numpy.nan_like(feature[batchMask])
				if parameters['align'] != 'no':
					feature[batchMask] = numpy.divide(feature[batchMask], batchMean)
					feature[batchMask] = numpy.multiply(feature[batchMask], featureAverage)
				
#				# If negative data mark for exclusion (occurs when too many QCsamples have intensity==0)
#				if sum(feature[batchMask]<0) != 0:  # CJS 240816
#					exclude = exclude + '; negativeData=' + str(sum(feature[batchMask]<0))

#		results.append((i, feature, fit, exclude))  # CJS 240816
		results.append((i, feature, fit))

	return results


def runOrderCompensation(data, runOrder, referenceSamples, parameters):
	"""
	Model and remove longitudinal effects.
	"""

	# Break the QCs out of the dataset
	QCdata = data[referenceSamples]
	QCrunorder = runOrder[referenceSamples]

	# Select model
	# Optimisation of window would happen here.
	window = parameters['window']
	align = parameters['align']
	if parameters['method'] == 'LOWESS':
		(data, fit) = doLOESScorrection(QCdata, 
										QCrunorder, 
										data, 
										runOrder,
										align=align,
										window=window)
	elif parameters['method'] == 'SavitzkyGolay':
		(data, fit) = doSavitzkyGolayCorrection(QCdata, 
												QCrunorder, 
												data, 
												runOrder, 
												window=window)

	# Potentially exclude features with poor fits that retuned NaN &c here.
	
	return (data, fit)


def doLOESScorrection(QCdata, QCrunorder, data, runorder, align='median', window=11):
	"""
	Fit a LOWESS regression to the data.
	"""
	# Convert window number of samples to fraction of the dataset:
	noSamples = QCrunorder.shape

	if noSamples == 0:

		fit = numpy.zeros(shape=runorder.shape)
		corrected = data

	else:
		frac = window / float(numpy.squeeze(noSamples))
		frac = min([1, frac])
		# actually do the work
		z = lowess(QCdata, QCrunorder, frac=frac)

		# Divide by fit, then rescale to batch median
		fit = numpy.interp(runorder, z[:,0], z[:,1])
	
		# Fit can go negative if too many adjacent QC samples == 0; set any negative fit values to zero
		fit[fit < 0] = 0

		corrected = numpy.divide(data, fit)
		if align == 'median':
			corrected = numpy.multiply(corrected, numpy.median(QCdata))
		elif align == 'mean':
			corrected = numpy.multiply(corrected, numpy.mean(QCdata))

	return corrected, fit


def doSavitzkyGolayCorrection(QCdata, QCrunorder, data, runorder, window=11, polyOrder=3):
	"""
	Fit a Savitzky-Golay curve to the data.
	"""
	# Sort the array
	sortedRO = numpy.argsort(QCrunorder)
	sortedRO2 = QCrunorder[sortedRO]
	QCdataSorted = QCdata[sortedRO]

	# actually do the work
	z = savgol_filter(QCdataSorted, window, polyOrder)

	fit = numpy.interp(runorder, sortedRO2, z)

	corrected = numpy.divide(data, fit)
	corrected = numpy.multiply(corrected, numpy.median(QCdata))

	return corrected, fit


def optimiseCorrection(feature, optimise):
	"""
	Optimise the window function my mimising the output of `optimise(data)`
	"""
	pass


##
# Adapted from http://stackoverflow.com/questions/2130016/splitting-a-list-of-arbitrary-size-into-only-roughly-n-equal-parts
## 
def _chunkMatrix(seq, num):
	avg = round(len(seq) / float(num))
	out = []
	last = 0.0

	for i in range(0, num-1):
		out.append(seq[int(last):int(last + avg)])
		last += avg
	out.append(seq[int(last):max(seq)+1])

	return out
