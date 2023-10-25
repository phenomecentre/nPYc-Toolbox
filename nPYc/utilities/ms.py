"""
Utility functions.
"""
import numpy
import warnings
import pandas
from ..enumerations import AssayRole, SampleType

def rsd(data):
	"""
	Calculate percentage :term:`relative standard deviation` for each column in *data*.

	:math:`\mathit{{rsd(x)}} = \\frac{\mathit{\sigma_{x}}}{\mathit{\mu_{x}}} \\times 100`

	Where RSDs cannot be calculated, (i.e. means of zero), ``numpy.finfo(numpy.float64).max`` is returned.

	:param numpy.ndarray data: *n* by *m* numpy array of data, with features in columns, and samples in rows
	:return: *m* vector of RSDs
	:rtype: numpy.ndarray
	"""

	std = numpy.std(data, axis=0)

	# If std is zero, note it
	stdMask = std == 0
	std[stdMask] = 1

	rsd = numpy.multiply(numpy.divide(std, numpy.mean(data, axis=0)), 100)

	rsd[numpy.isnan(rsd)] = numpy.finfo(numpy.float64).max
	rsd[stdMask] = 0

	return rsd


def sequentialPrecision(data):
	"""
	Calculate percentage sequential precision for each column in *data*. Sequential precision for feature :math:`x` is defined as:

	:math:`\mathit{{sp(x)}} = \\frac{\sqrt{(\\frac{1}{n-1} \sum_{i=1}^{n-1} (x_{i+1} - x_i)^2)/2}}{\mu_{x}} \\times 100`

	:param numpy.ndarray data: *n* by *m* numpy array of measures, with features in columns, and samples in rows
	:return: *m* vector of sequential precision measures
	:rtype: numpy.ndarray
	"""
	# Calculate sample to sample difference
	sequentialDifference = numpy.diff(data, axis=0)

	# Calculate squared differences
	sequentialDifference = numpy.square(sequentialDifference)

	# Take the mean
	sequentialDifference = numpy.mean(sequentialDifference, axis=0)
	sequentialDifference = numpy.divide(sequentialDifference, 2.0)
	sequentialDifference = numpy.sqrt(sequentialDifference)

	# We handle divide by zeroes, so don't warn
	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		sequentialDifference = numpy.divide(sequentialDifference, numpy.mean(data, axis=0))

	sequentialDifference = numpy.multiply(sequentialDifference, 100)

	# Replace NaNs with float MAX
	sequentialDifference[numpy.isnan(sequentialDifference)] = numpy.finfo(numpy.float64).max

	return sequentialDifference


def generateLRmask(dataset):
	"""
	Generate a dictionary of masks for each Linearity Reference subset (i.e., for each batch 1-46 and 47-92), in order for mean correlation to dilution to be calculated.

	:param nPYc.MSDataset msData: Object containing dilution subsets to parse
	:return: LRoutput: Masks of Linearity Reference samples separated by batch
	"""
	from ..enumerations import AssayRole, SampleType

	if dataset.corrExclusions is None:
		raise ValueError('dataset.corrExclusions is not defined')

	# instantiate sample mask dictionary
	LRoutput = dict()

	if not 'Dilution Series' in dataset.sampleMetadata.columns:
		lrMask = (dataset.sampleMetadata['SampleType'].values == SampleType.StudyPool) & (dataset.sampleMetadata['AssayRole'].values == AssayRole.LinearityReference)
		lrMask = numpy.logical_and(lrMask, dataset.corrExclusions)
		LRoutput['All Dilution Samples'] = lrMask

	else:
		# determine number of batches
		batches = dataset.sampleMetadata['Batch'].unique()
		mask = pandas.notnull(batches)
		batches = batches[mask]

		# for each batch subset
		for batch in batches:

			dilutionSeries = dataset.sampleMetadata['Dilution Series'].unique()
			mask = pandas.notnull(dilutionSeries)
			dilutionSeries = dilutionSeries[mask]
			for series in dilutionSeries:
				seriesMask = numpy.logical_and(dataset.sampleMetadata['Batch'] == batch,
											   dataset.sampleMetadata['Dilution Series'] == series)

				seriesMask = numpy.logical_and(seriesMask,
											   dataset.corrExclusions)

				# Only store if there are any samples in this series
				if sum(seriesMask) > 0:
					name = 'Batch %s, series %s' % (str(batch), str(series))
					LRoutput[name] = seriesMask.values

	return LRoutput


def rsdsBySampleType(dataset, onlyPrecisionReferences=True, useColumn='SampleType'):
	"""
	Return percent RSDs calculated for the distinct class values in `useColumn`, defaults to the SampleType enums in 'SampleType'.

	:param Dataset dataset: Dataset object to generate RSDs for.
	:param bool onlyPrecisionReferences: If ``True`` only use samples with the 'AssayRole' PrecisionReference
	:returns: Dict of RSDs for each group
	:rtype: dict(str:numpy array)
	"""
	from ..enumerations import AssayRole
	from ..objects import Dataset

	if not isinstance(dataset, Dataset):
		raise TypeError('dataset must be an instance of Dataset.')

	if not useColumn in dataset.sampleMetadata.columns:
		raise KeyError("%s is not a column in sampleMetadata." % (useColumn))

	rsds = dict()
	sampleTypes = dataset.sampleMetadata[useColumn].unique()
	for sampleType in sampleTypes:

		if onlyPrecisionReferences:
			mask = numpy.logical_and(dataset.sampleMetadata[useColumn].values == sampleType,
									 dataset.sampleMetadata['AssayRole'].values == AssayRole.PrecisionReference)
		else:
			mask = dataset.sampleMetadata[useColumn].values == sampleType

		mask = numpy.logical_and(mask, dataset.sampleMask)

		if sum(mask) < 2:
			continue

		rsds[str(sampleType)] = rsd(dataset.intensityData[mask, :])

	return rsds

def generateTypeRoleMasks(sampleMetadata):
	"""
	Generate masks of standard NPC samples based on pre-defined combinations of 'SampleType' and 'AssayRole'
	"""

	# Number of samples
	ns = sampleMetadata.shape[0]

	try:
		ALLmask = numpy.ones(ns).astype(bool)
		SSmask = (sampleMetadata['SampleType'] == SampleType.StudySample) & (
					sampleMetadata['AssayRole'] == AssayRole.Assay)
		SPmask = (sampleMetadata['SampleType'] == SampleType.StudyPool) & (
					sampleMetadata['AssayRole'] == AssayRole.PrecisionReference)
		ERmask = (sampleMetadata['SampleType'] == SampleType.ExternalReference) & (
					sampleMetadata['AssayRole'] == AssayRole.PrecisionReference)
		SRDmask = (sampleMetadata['AssayRole'] == AssayRole.LinearityReference) & (
					sampleMetadata['SampleType'] == SampleType.StudyPool)
		Blankmask = sampleMetadata['SampleType'] == SampleType.ProceduralBlank

	except:
		ALLmask = numpy.zeros(ns).astype(bool)
		SSmask = numpy.zeros(ns).astype(bool)
		SPmask = numpy.zeros(ns).astype(bool)
		ERmask = numpy.zeros(ns).astype(bool)
		SRDmask = numpy.zeros(ns).astype(bool)
		Blankmask = numpy.zeros(ns).astype(bool)

	Unknownmask = (SSmask == False) & (SPmask == False) & (ERmask == False) & (SRDmask == False) & (Blankmask == False)

	TypeRoleMasks = {
		'ALLmask': ALLmask,
		'SSmask': SSmask,
		'SPmask': SPmask,
		'ERmask': ERmask,
		'SRDmask': SRDmask,
		'Blankmask': Blankmask,
		'Unknownmask': Unknownmask
	}

	return TypeRoleMasks

