import numpy
import pandas
import math
import os
from pathlib import PurePath
import datetime

from ..utilities._nmr import cutSec
from ..utilities._baseline import baseline


def interpolateSpectrum(spectrum, originalScale, targetScale):
	"""
	Interpolate spectra onto *targetScale* using python interp1d

	:param spectrum: the raw spectrum
	:type spectrum: array
	:param originalScale: the ppm
	:type originalScale:array
	:param targetScale: the scale
	:type targetScale: array
	:returns: interpolatedSpectra
	:rtype: array
	"""

	from scipy.interpolate import interp1d

	if spectrum.ndim == 2:
		noSpectra = spectrum.shape[0]
	elif spectrum.ndim == 1:
		noSpectra = 1
	else:
		raise ValueError("Interpolation is only supported for either a single or 2d array of 1D spectra.")

	if noSpectra == 1:
		interpolatedSpectrum = interp1d(originalScale,spectrum)(targetScale)

	else:
		interpolatedSpectrum = numpy.zeros(shape=(noSpectra, len(targetScale)))

		for i in range(noSpectra):
			interpolatedSpectrum[i,:] = interp1d(originalScale,spectrum[i,:], assume_sorted=True)(targetScale)

	return interpolatedSpectrum


def generateBaseName(sampleMetadata):
	"""
	Generate a sample base name for the loaded dataset by rounding by 10
	:param sampleMetadata:
	"""
	# Spliting expno from 'Sample File Name'
	def splitExpno(sampleFileName):
		output = sampleFileName.split(sep='/')
		return output[-1]

	# Rounding expno down by 10s
	def roundExpno(expno):
		return int(math.floor(expno / 10.0) * 10)

	def splitRack(sampleFileName):
		output = sampleFileName.split(sep='/')
		return output[-2]

	if not 'expno' in sampleMetadata.columns:
		expno = sampleMetadata.loc[:, 'Sample File Name'].apply(splitExpno)
		expno = pandas.to_numeric(expno)

	else:
		expno = sampleMetadata['expno']

	roundedExpno = expno.map(roundExpno)
	roundedExpno = roundedExpno.astype(str)

	return sampleMetadata.loc[:,'Sample File Name'].apply(splitRack).str.cat(roundedExpno, sep='/').values, expno.values


def cutSec(ppm, X, start, stop, featureMask):
	"""
	Temove defined regions from NMR spectra data
	input/output as per matlab version of code:

	% ppm (1,nv) = ppm scale for nv variables
	% X (ns,nv)  = NMR spectral data for ns samples
	% start (1,1) = ppm value for start of region to remove
	% stop (1,1) = ppm value for end of region to remove
	%
	% OUTPUT:
	% ppm (1,nr) = ppm scale with region from start:stop removed
	% X (ns,nr)  = NMR spectral with region from start:stop removed

	"""

	flip = 0
	if ppm[0] > ppm[-1]:
		flip = 1
		ppm = ppm[::-1]
		X = X[:, ::-1]

	# find first entry in ppm with >='start' valu
	start = (ppm >= start).nonzero()
	start = start[0][0]  # first entry
	stop = (ppm <= stop).nonzero()
	stop = stop[0][-1]  # last entry

	# currently setting featureMask will get rid of peaks in start:stop region BUT it also marks as excluded so have removed as inaccurately marking for exclusion when all we want to do is remove from intensityData not mark as exluded
	try:
		featureMask[0,
		start:stop] = False  # this may only occur on unit test data, not sure need to check but either way was causing issue
	except:
		featureMask[start:stop] = False
	if flip == 1:
		ppm = ppm[::-1]
		X = X[:, ::-1]
	return ppm, X, featureMask
	pass


def _qcCheckBaseline(spectrum, alpha):
	"""
	Baseline checks
	:param spectrum:
	:param alpha:
	:return:
	"""

	# Single threshold
	criticalThreshold = numpy.percentile(spectrum, 1 - alpha)
	# Number of points
	isOutlier = spectrum > criticalThreshold

	numpy.sum(isOutlier)/spectrum.shape[0]

	return isOutlier


# For now same as previous function, but keeping room for different algorithms
def _qcCheckWaterPeak(spectrum, alpha):
	"""
	Baseline checks
	:param spectrum:
	:param alpha:
	:return:
	"""

	# Single threshold
	criticalThreshold = numpy.percentile(spectrum, 1 - alpha)
	# Number of points
	isOutlier = spectrum > criticalThreshold

	numpy.sum(isOutlier) / spectrum.shape[0]

	return isOutlier
