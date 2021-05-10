import numpy
import pandas
import math
import os

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
		output = os.path.split(sampleFileName)
		return output[-1]

	# Rounding expno down by 10s
	def roundExpno(expno):
		return int(math.floor(expno / 10.0) * 10)

	def splitRack(sampleFileName):
		output = os.path.split(sampleFileName)
		return output[-2]

	if not 'expno' in sampleMetadata.columns:
		expno = sampleMetadata.loc[:, 'Sample File Name'].apply(splitExpno)
		expno = pandas.to_numeric(expno)

	else:
		expno = sampleMetadata['expno']

	roundedExpno = expno.map(roundExpno)
	roundedExpno = roundedExpno.astype(str)

	return sampleMetadata.loc[:, 'Sample File Name'].apply(splitRack).str.cat(roundedExpno, sep='/').values, expno.values


def qcCheckBaseline(spectrum, alpha):
	"""
	Baseline checks
	:param spectrum:
	:param alpha:
	:return:
	"""

	# Single threshold
	criticalThresholdUpper = numpy.percentile(spectrum, (1 - alpha)*100)
	criticalThresholdLower = numpy.percentile(spectrum, (alpha)*100)

	# check for each point if it outside the percentiles defined by alpha
	isOutlierPoint = spectrum > criticalThresholdUpper
	isOutlierPoint |= spectrum < criticalThresholdLower

	# Check proportion of points out per sample
	sumOut = numpy.sum(isOutlierPoint, 1)/spectrum.shape[1]
	# Check if proportion of points either above or below is higher than expected
	isOutlier = sumOut > 1 - alpha

	return isOutlier


# For now same as previous function, but keeping room for different algorithms
def qcCheckSolventPeak(spectrum, alpha):
	"""
	Solvent peak checks
	:param spectrum:
	:param alpha:
	:return:
	"""

	# Single threshold
	criticalThresholdUpper = numpy.percentile(spectrum, (1 - alpha) * 100)
	criticalThresholdLower = numpy.percentile(spectrum, (alpha) * 100)

	# check for each point if it outside the percentiles defined by alpha
	isOutlierPoint = spectrum > criticalThresholdUpper
	#isOutlierPoint |= spectrum < criticalThresholdLower

	# Check proportion of points out per sample
	sumOut = numpy.sum(isOutlierPoint, 1) / spectrum.shape[1]
	# Check if proportion of points either above or below is higher than expected
	isOutlier = sumOut > 1 - alpha

	return isOutlier
