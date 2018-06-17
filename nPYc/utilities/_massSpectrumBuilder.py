import numpy
from datetime import datetime
from ._internal import _vcorrcoef
from ._conditionalJoin import conditionalJoin
from ._buildSpectrumFromQIfeature import buildMassSpectrumFromQIfeature

import copy

def massSpectrumBuilder(msData, correlationThreshold=0.95, rtWindow=20, simulatedSpecra=True):
	"""
	Combine individual features into pseudo-mass spectra, by looking for co-eluting features where the observed intensities correlate above *correlationThreshold* across the dataset.

	.. warning:: Care should be taken with datasets exhibiting strong run-order, batch, or sample concentration effects, as these effects may introduce strong correlations between all features in the dataset.

	:param MSDataset msData: MSdataset to process
	:param float correlationThreshold: Combine features correlated above this level
	:param float rtWindow: Combine features the coelute ± this range in seconds
	:param bool simulatedSpecra: If ``True``, generate simulated mass spectra for each component based on the constituent features
	:return: *msData* with features parsed into components
	:rtype: MSDataset
	"""
	##
	# Start with the peak of highest mean intensity
	##
	averageIntensities = numpy.mean(msData.intensityData, axis=0)
	ranking = numpy.argsort(averageIntensities)

	# Convert to mins for QI
	rtWindow = rtWindow / 60

	returnedData= copy.deepcopy(msData)
	returnedData.featureMetadata['Correlated Features'] = ''

	if simulatedSpecra:
		returnedData.featureMetadata['Mass Spectrum'] = None
		returnedData.featureMetadata['Mass Spectrum'] = returnedData.featureMetadata['Mass Spectrum'].astype(object)

	for currentFeature in ranking[::-1]:
		# Skip features already claimed
		if returnedData.featureMask[currentFeature]:
			##
			# Draw an RT window arround feature
			##
			rt = msData.featureMetadata['Retention Time'].iloc[currentFeature]
			rtMask = (msData.featureMetadata['Retention Time'].values > (rt - rtWindow)) & (msData.featureMetadata['Retention Time'].values < (rt + rtWindow))
			rtMask = numpy.logical_and(rtMask, returnedData.featureMask)

			# Mask the current feature
			rtMask[currentFeature] = False

			# Get index of our features in the dataset
			candidateFeatures = numpy.arange(0, msData.noFeatures, dtype=int)
			candidateFeatures = candidateFeatures[rtMask]

			##
			# Find additional features that correlate strongly
			##
			c = _vcorrcoef(msData.intensityData, msData.intensityData[:, currentFeature], featureMask=rtMask)

			##
			# Combine into one measure
			##
			# Mask out corelated features
			featuresToBeRemoved = candidateFeatures[c >= correlationThreshold]
			returnedData.featureMask[featuresToBeRemoved] = False

			for feature in featuresToBeRemoved:
				returnedData.featureMetadata.loc[currentFeature, 'Correlated Features'] = conditionalJoin(returnedData.featureMetadata['Correlated Features'].iloc[currentFeature],
																							returnedData.featureMetadata['Feature Name'].iloc[feature])

			if simulatedSpecra:
				# At the momment this only works for QI
				spectrum = list()
				if msData.Attributes['FeatureExtractionSoftware'] == 'Progenesis QI':
					featuresToBeRemoved = numpy.append(featuresToBeRemoved, currentFeature)
					for feature in featuresToBeRemoved:
						spectrum.append(buildMassSpectrumFromQIfeature(msData.featureMetadata.iloc[feature].to_dict()))
				spectrum = [item for sublist in spectrum for item in sublist]
				#returnedData.featureMetadata.set_value(currentFeature, 'Mass Spectrum', spectrum)
				returnedData.featureMetadata.at[currentFeature, 'Mass Spectrum'] = spectrum

	returnedData.applyMasks()
	returnedData.Attributes['Log'].append((datetime.now(), "Redundant features removed with rtWindow of: %f seconds and correlationThreshold of: %f." % (rtWindow * 60, correlationThreshold)))

	return returnedData
