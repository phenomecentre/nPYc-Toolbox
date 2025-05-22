import matplotlib.pyplot as plt
from ..plotting._violinPlot import _violinPlotHelper
from ..enumerations import AssayRole, SampleType
from ..utilities.ms import generateTypeRoleMasks
import numpy
import math
import copy

def plotTargetedFeatureDistribution(datasetOriginal, featureName='Feature Name', featureMask=None, sampleTypes=['SS', 'SP', 'ER'], logx=False, figures=None, savePath=None, figureFormat='png', dpi=72, figureSize=(11,7)):
	"""
	Plot the distribution (violin plots) of a set of features, e.g., peakPantheR outputs, coloured by sample type

	:param MSDataset dataset: :py:class:`MSDataset`
	:param bool logx: If ``True`` log-scale the x-axis
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

	# Define sample masks
	acquiredMasks = generateTypeRoleMasks(dataset.sampleMetadata)

	# Define sample masks
	sampleMasks = []
	palette = {}

	sTypeColourDict = {SampleType.StudySample: 'b', SampleType.StudyPool: 'g', SampleType.ExternalReference: 'r',
					   SampleType.MethodReference: 'm', SampleType.ProceduralBlank: 'c', 'Other': 'grey'}

	# Plot data coloured by sample type
	if sum(acquiredMasks['SSmask'] > 0) and 'SS' in sampleTypes:
		sampleMasks.append(('SS', acquiredMasks['SSmask']))
		palette['SS'] = sTypeColourDict[SampleType.StudySample]
	if sum(acquiredMasks['SPmask'] > 0) and 'SP' in sampleTypes:
		sampleMasks.append(('SR', acquiredMasks['SPmask']))
		palette['SR'] = sTypeColourDict[SampleType.StudyPool]
	if sum(acquiredMasks['ERmask'] > 0) and 'ER' in sampleTypes:
		sampleMasks.append(('LTR', acquiredMasks['ERmask']))
		palette['LTR'] = sTypeColourDict[SampleType.ExternalReference]

	# Plot
	for figNo in range(nf):

		fig, axIXs = plt.subplots(1, nax, figsize=(figureSize[0], figureSize[1]/nax), dpi=dpi)

		for axNo in range(len(axIXs)):

			if plotNo >= nv:
				axIXs[axNo].axis('off')

			else:

				# Plot distribution of feature by sample type
				# Remove infinites and - infinites for targeted dataset.
				valid_values = numpy.isfinite(dataset.intensityData[:,plotNo])

				currentFeatureSampleMasks = list()
				for maskIndex in range(len(sampleMasks)):
					currentFeatureSampleMasks.append((sampleMasks[maskIndex][0], sampleMasks[maskIndex][1] & valid_values))
				if valid_values.any():
					_violinPlotHelper(axIXs[axNo], dataset.intensityData[:, plotNo], currentFeatureSampleMasks, None, 'Sample Type', palette=palette, logy=False)

				axIXs[axNo].set_title(dataset.featureMetadata.loc[plotNo, featureName])

			# Advance plotNo
			plotNo = plotNo+1

		if savePath:
			if figures is not None:
				figures['featureDistribution_' + str(figNo)] = savePath + 'featureDistribution_' + str(figNo) + '.' + figureFormat

			plt.savefig(savePath + 'featureDistribution_' + str(figNo) + '.' + figureFormat, bbox_inches='tight', format=figureFormat, dpi=dpi)
			plt.close()
		else:
			plt.show()

	if figures is not None:
		return figures