import numpy
import warnings
from ..enumerations import SampleType, AssayRole

def blankFilter(dataset, threshold=None):
	"""
	Generates a boolean mask of the features in *dataset* that is true where the average intensity is greater than that seen in procedural blank injections.

	If no procedural blank samples are present in *dataset* all features are marked as ``True``.

	:param MSDataset dataset: Dataset object to process
	:param threshold: If ``None`` attempt to read the theshold multiplier from *dataset.Attributes['threshold']*, otherwise use the value specified.
	:type threshold: None, False, or float
	:returns: Boolean mask where ``True`` indicates features above the blank threshold
	:rtype: numpy.ndarray
	"""

	if threshold is None:
		threshold = dataset.Attributes['blankThreshold']
	elif not (isinstance(threshold, float) or isinstance(threshold, bool)):
		raise TypeError("threshold must be either None, False, or a float, %s provided." % (type(threshold)))
	elif isinstance(threshold, bool) and threshold:
		raise TypeError("threshold must be either None, False, or a float, %s provided." % (type(threshold)))

	if threshold:
		blanksMask = dataset.sampleMetadata['SampleType'].values == SampleType.ProceduralBlank

		blanksMask = numpy.logical_and(blanksMask, dataset.sampleMask)

		if sum(blanksMask) <= 0:
			warnings.warn("No Procedural blank samples present, skipping blank filter.")
			threshold = False

	if threshold:
		# Calculate abundance in SP samples and SS
		sampleMask = numpy.logical_and(dataset.sampleMetadata['AssayRole'].values == AssayRole.Assay,
							 dataset.sampleMetadata['SampleType'].values == SampleType.StudySample)

		if sum(blanksMask) > 1:
			p95 = numpy.percentile(dataset.intensityData[blanksMask, :], 95, axis=0)
		else:
			p95 = dataset.intensityData[blanksMask, :]

		mask = numpy.mean(dataset.intensityData[sampleMask,:], axis=0) >= (p95 * threshold)

	else:
		mask = numpy.squeeze(numpy.ones([dataset.noFeatures, 1], dtype=bool), axis=1)

	return mask
