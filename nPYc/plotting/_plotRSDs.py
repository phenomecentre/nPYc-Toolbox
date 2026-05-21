import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import seaborn as sns
import numpy
import pandas
import copy
import os
import plotly.graph_objs as go
import plotly
from .. import Dataset, MSDataset, NMRDataset
from ..enumerations import VariableType, SampleType, AssayRole
from ..utilities import rsd, sampleClassMasks
from ._plotVariableScatter import plotVariableScatter
from ..utilities._errorHandling import npycToolboxError


def plotRSDs(dataset, featureName='Feature Name', ratio=False, logx=True, xlim=None, withExclusions=False, sortOrder='rsdSP', savePath=None, featName=False, hLines=None, by='SampleClass'):
	"""
	plotRSDs(dataset, ratio=False, savePath=None, color=None \*\*kwargs)

	Visualise analytical *versus* biological variance.

	Plot RSDs calculated in study-reference samples (analytical variance), versus those calculated in study samples (biological variance). RSDs can be visualised either in absolute terms, or as a ratio to analytical variation (*ratio=*\ ``True``).

	:py:func:`plotRSDs` requires that the dataset have at least two samples with the :py:attr:`~nPYc.enumerations.AssayRole.PrecisionReference` :term:`assay role`, if present, RSDs calculated on independent sets of :py:attr:`~nPYc.enumerations.AssayRole.PrecisionReference` samples will also be plotted.

	:param Dataset dataset: Dataset object to plot, the object must have greater that one 'Study Sample' and 'Study-Reference Sample' defined
	:param bool ratio: If ``True`` plot the ratio of analytical variance to biological variance instead of raw values
	:param str featureName: featureMetadata column name by which to label features
	:param str sortOrder: featureMetadata column name by which to order features
	:param bool logx: If ``True`` plot RSDs on a log10 scaled axis
	:param xlim: Tuple of (min, max) RSD values to plot
	:type xlim: None or tuple(float, float)
	:param hLines: None or list of y positions at which to plot an horizontal line. Features are positioned from 1 to nFeat
	:type hLines: None or list
	:param savePath: If ``None`` plot interactively, otherwise save the figure to the path specified
	:type savePath: None or str
	:param bool featName: If ``True`` y-axis label is the feature Name, if ``False`` features are numbered.
	"""

	# Generate table of RSD values by sample type (by, default is 'SampleClass')
	rsdTable = _plotRSDsHelper(dataset,
							   featureName=featureName,
							   ratio=ratio,
							   withExclusions=withExclusions,
							   sortOrder=sortOrder,
							   by=by)

	cols = list(rsdTable.columns)
	cols.remove(featureName)

	# Check 'by' values are represented in the dataset.Attributes for plot colours and abbreviations
	for col in cols:
		if not col in dataset.Attributes['sampleTypeColours']:
			raise npycToolboxError('Unable to colour plot by: ' + str(col) + ' as not present in `dataset.Attributes["sampleTypeColours"]`')
		if not col in dataset.Attributes['sampleTypeAbbr']:
			raise npycToolboxError('Unable to label plot by: ' + str(col) + ' as not present in `dataset.Attributes["sampleTypeAbbr"]`')

	# Ensure we have 'Passing Selection' column in dataset object
	if not hasattr(dataset.featureMetadata, 'Passing Selection'):
		dataset.featureMetadata['Passing Selection'] = dataset.featureMask

	# If the featureMask has been applied, add a line to show failing features
	if hLines is not None:
		if dataset.featureMetadata.shape[0] != rsdTable.shape[0]:
			temp = [x for x in rsdTable['Feature Name'].values.tolist() if x in dataset.featureMetadata[featureName][dataset.featureMetadata['Passing Selection'] == False].values.tolist()]
			hLines = [len(temp)]

	# Plot
	if xlim:
		xLim = xlim
	else:
		minRSD = numpy.min(rsdTable[rsdTable.columns[1:]].values)
		maxRSD = numpy.max(rsdTable[rsdTable.columns[1:]].values)
		xLim = (minRSD, maxRSD)

	if logx:
		xlab = 'RSD (%)'
	else:
		xlab = 'RSD (%)'

	# Standardise naming for plotting conventions
	rsdTable.rename(columns={featureName: "yName"}, inplace=True)
	if featName:
		ylab = featureName
	else:
		ylab = 'Feature Number'

	plotVariableScatter(rsdTable,
						logX=logx,
						xLim=xLim,
						xLabel=xlab,
						yLabel=ylab,
						sTypeColourDict=dataset.Attributes['sampleTypeColours'],
						sTypeAbbrDict=dataset.Attributes['sampleTypeAbbr'],
						hLines=hLines,
						vLines=None,
						savePath=savePath,
						figureFormat=dataset.Attributes['figureFormat'],
						dpi=dataset.Attributes['dpi'],
						figureSize=dataset.Attributes['figureSize'])


def plotRSDsInteractive(dataset, featureName='Feature Name', ratio=False, withExclusions=False, sortOrder='rsdSP', logx=True, by='SampleClass', destinationPath=None, autoOpen=False):
	"""
	Plotly-based interactive version of :py:func:`plotRSDs`

	Visualise analytical *versus* biological variance.

	Plot RSDs calculated in study-reference samples (analytical variance), versus those calculated in study samples (biological variance). RSDs can be visualised either in absolute terms, or as a ratio to analytical variation (*ratio=*\ ``True``).

	:py:func:`plotRSDsInteractive` requires that the dataset have at least two samples with the :py:attr:`~nPYc.enumerations.AssayRole.PrecisionReference` :term:`assay role`, if present, RSDs calculated on independent sets of :py:attr:`~nPYc.enumerations.AssayRole.PrecisionReference` samples will also be plotted.

	:param Dataset dataset: Dataset object to plot, the object must have greater that one 'Study Sample' and 'Study-Reference Sample' defined
	:param str featureName: featureMetadata column name by which to label features
	:param bool ratio: If ``True`` plot the ratio of analytical variance to biological variance instead of raw values
	:param bool logx: If ``True`` plot RSDs on a log10 scaled axis

	"""

	# Generate table of RSD values by sample type (by, default is 'SampleClass')
	rsdTable = _plotRSDsHelper(dataset,
	                           featureName=featureName,
	                           ratio=ratio,
	                           withExclusions=withExclusions,
	                           sortOrder=sortOrder,
	                           by=by)

	cols = list(rsdTable.columns)
	cols.remove(featureName)

	# Check 'by' values are represented in the dataset.Attributes for plot colours and abbreviations
	for col in cols:
		if not col in dataset.Attributes['sampleTypeColours']:
			raise npycToolboxError('Unable to colour plot by: ' + str(col) + ' as not present in `dataset.Attributes["sampleTypeColours"]`')
		if not col in dataset.Attributes['sampleTypeAbbr']:
			raise npycToolboxError('Unable to label plot by: ' + str(col) + ' as not present in `dataset.Attributes["sampleTypeAbbr"]`')

	reversedIndex =  numpy.arange(len(rsdTable)-1,-1, -1)
	data = []

	for col in cols:
		plotData = go.Scatter(
			x=rsdTable[col].values,
			y=reversedIndex,
			mode='markers',
			text=rsdTable[featureName],
			name=dataset.Attributes['sampleTypeAbbr'][col],
			marker=dict(
				color=dataset.Attributes['sampleTypeColours'][col],
			),
			hoverinfo='x+text',
		)
		data.append(plotData)

	if logx:
		xaxis = dict(
					type='log',
					title='RSD (%)',
					autorange=True
					)
	else:
		xaxis = dict(
					title='RSD (%)'
					)

	layout = go.Layout(
				title='Feature RSDs',
				legend=dict(
					orientation="h"
				),
				hovermode = "closest",
				yaxis=dict(
						title='Feature Number'
					),
				xaxis=xaxis
				)

	figure = go.Figure(data=data, layout=layout)

	# Save to destinationPath
	if destinationPath:
		plotly.offline.plot(figure, filename=os.path.join(destinationPath, dataset.name + '_rsdSampletype.html'), auto_open=autoOpen)

	return figure


def _plotRSDsHelper(dataset, featureName='Feature Name', ratio=False, withExclusions=False, sortOrder='rsdSP', by='SampleClass'):

	if not dataset.VariableType == VariableType.Discrete:
		raise ValueError('Only datasets with discreetly sampled variables are supported.')

	if sum(dataset.sampleMetadata.loc[dataset.sampleMask, 'SampleType'].values == SampleType.StudySample) <= 2:
		raise ValueError('More than two Study Samples must be defined to calculate biological RSDs.')

	# Apply sample/feature masks if exclusions to be applied
	msData = copy.deepcopy(dataset)
	if withExclusions:
		msData.applyMasks()

	# Calculate RSD for any SampleClass with n > 3
	rsdVal = dict()
	rsdVal[featureName] = msData.featureMetadata.loc[:, featureName].values

	# Previously, the code was calculating RSD for only features with finite values,
	# commented out for now but could be re-instated if required

	# Define sample masks
	sampleMasks = sampleClassMasks(msData.sampleMetadata, on=by)

	for key in sampleMasks.keys():

		if sum(sampleMasks[key]) > 3:
			rsdVal[key] = rsd(msData.intensityData[sampleMasks[key], :])
			# finiteMask = (rsdVal[key] < numpy.finfo(numpy.float64).max)
			# rsdVal[key] = rsdVal[key][finiteMask]

			if ratio:
				rsdVal[key] = numpy.divide(rsdVal[key], msData.rsdSP)

	rsdTable = pandas.DataFrame(rsdVal)

	# If sortOrder, sort by FeatureMask, then order by featureMetadata 'sortOrder' column values
	if sortOrder:

		# Ensure we have 'Passing Selection' column in dataset object
		if not hasattr(msData.featureMetadata, 'Passing Selection'):
			msData.featureMetadata['Passing Selection'] = msData.featureMask

		# Add rsdSP if required (this can be empty for the first summary reports)
		if sortOrder == 'rsdSP':
			msData.featureMetadata['rsdSP'] = msData.rsdSP

		# Check that we have 'sortOrder' column in featureMetadata, and sort
		if hasattr(msData.featureMetadata, sortOrder):
			msData.featureMetadata.sort_values(by=['Passing Selection', sortOrder], ascending=[False, True], inplace=True)

		else:
			msData.featureMetadata.sort_values(by=['Passing Selection'], ascending=[False], inplace=True)

		sortIndex = msData.featureMetadata.index
		rsdTable = rsdTable.reindex(sortIndex)
		rsdTable.reset_index(drop=True, inplace=True)

	return rsdTable