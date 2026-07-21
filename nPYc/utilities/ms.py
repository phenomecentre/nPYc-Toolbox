"""
Utility functions.
"""
import numpy
import pandas

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


def generateLRmask(dataset):
	"""
	Generate a dictionary of masks for each Linearity Reference subset (i.e., for each batch 1-46 and 47-92), in order for mean correlation to dilution to be calculated.

	:param nPYc.MSDataset msData: Object containing dilution subsets to parse
	:return: LRoutput: Masks of Linearity Reference samples separated by batch
	"""
	from ..enumerations import AssayRole, SampleType

	# instantiate sample mask dictionary
	LRoutput = dict()

	if not 'Dilution Series' in dataset.sampleMetadata.columns:
		lrMask = (dataset.sampleMetadata['SampleType'].values == SampleType.StudyPool) & (dataset.sampleMetadata['AssayRole'].values == AssayRole.LinearityReference)
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