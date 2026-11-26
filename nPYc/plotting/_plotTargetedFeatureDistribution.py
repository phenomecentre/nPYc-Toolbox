import matplotlib.pyplot as plt
from ..plotting._violinPlot import _violinPlotHelper
from ..utilities import sampleClassMasks
from matplotlib.colors import rgb2hex
import numpy
import math
import copy

def plotTargetedFeatureDistribution(datasetOriginal, labelFeaturesBy='Feature Name', orderFeaturesBy='Feature Name', featureMask=None, figures=None, savePath=None):
	"""
	Plot the distribution (violin plots) of a set of features, e.g., peakPantheR outputs, coloured by sample type

	:param datasetOriginal dataset: :py:class:`MSDataset`
	:param dict figures: If not ``None``, saves location of each figure for output in html report (see _generateMSReport.py)
	"""
   
	# Apply sample/feature masks if exclusions to be applied	
	dataset = copy.deepcopy(datasetOriginal)    
	if featureMask is not None:
		dataset.featureMask = featureMask
		dataset.applyMasks()

	# Set up for plotting in subplot figures 1x2
	nax = 3 # number of axis per figure
	nv = dataset.featureMetadata.shape[0]
	nf = math.ceil(nv/nax)
	plotNo = 0

	# Define sample type masks for all samples in dataset
	acquiredMasks = sampleClassMasks(dataset.sampleMetadata, on='SampleClass')
	sampleMasks = []
	palette = dataset.Attributes['sampleTypeColours']

	for key in acquiredMasks:

		# Use abbreviation if available
		if key in dataset.Attributes['sampleTypeAbbr']:
			sampleMasks.append((dataset.Attributes['sampleTypeAbbr'][key], acquiredMasks[key]))

		# Else use existing key
		else:
			sampleMasks.append((key, acquiredMasks[key]))

	# Check all keys are in the palette, otherwise add
	if not all(k in palette.keys() for k in acquiredMasks):
		colors = iter(plt.cm.rainbow(numpy.linspace(0, 1, len(acquiredMasks))))
		for u in acquiredMasks:
			palette[u] = rgb2hex(next(colors))

	# If order of features specified, plot features ordered by FeatureMask, then by featureMetadata 'orderFeaturesBy' column values
	if orderFeaturesBy:

		# Copy featureMetadata
		featureInfo = copy.deepcopy(dataset.featureMetadata)

		# Add 'Passing Selection' column
		if not hasattr(featureInfo, 'Passing Selection'):
			featureInfo['Passing Selection'] = dataset.featureMask

		featureInfo.sort_values(by=['Passing Selection', orderFeaturesBy], ascending=[False, True], inplace=True)

		sortIndex = featureInfo.index

	else:
		sortIndex = range(dataset.featureMetadata.shape[0])

	# Plot
	for figNo in range(nf):

		fig, axIXs = plt.subplots(1, nax, figsize=(dataset.Attributes['figureSize'][0], dataset.Attributes['figureSize'][1]/nax), dpi=dataset.Attributes['dpi'])

		for axNo in range(len(axIXs)):

			if plotNo >= nv:
				axIXs[axNo].axis('off')

			else:

				# Plot distribution of feature by sample type
				# Remove infinites and - infinites for targeted dataset.
				valid_values = numpy.isfinite(dataset.intensityData[:,sortIndex[plotNo]])

				currentFeatureSampleMasks = list()
				for maskIndex in range(len(sampleMasks)):
					currentFeatureSampleMasks.append((sampleMasks[maskIndex][0], sampleMasks[maskIndex][1] & valid_values))
				if valid_values.any():
					_violinPlotHelper(axIXs[axNo],
									  dataset.intensityData[:, sortIndex[plotNo]],
									  currentFeatureSampleMasks,
									  None, 'Sample Type', palette=palette, logy=False)

				axIXs[axNo].set_title(dataset.featureMetadata.loc[sortIndex[plotNo], labelFeaturesBy])

			# Advance plotNo
			plotNo = plotNo+1

		if savePath:
			if figures is not None:
				figures['featureDistribution_' + str(figNo)] = savePath + '_' + str(figNo) + '.' + dataset.Attributes['figureFormat']

			plt.savefig(savePath + '_' + str(figNo) + '.' + dataset.Attributes['figureFormat'], bbox_inches='tight', format=dataset.Attributes['figureFormat'], dpi=dataset.Attributes['dpi'])
			plt.close()
		else:
			plt.show()

	if figures is not None:
		return figures