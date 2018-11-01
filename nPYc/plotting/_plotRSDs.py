import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import seaborn as sns
import numpy
import pandas
import copy
import os
import plotly.graph_objs as go

from .. import Dataset, MSDataset, NMRDataset
from ..enumerations import VariableType, SampleType, AssayRole
from ..utilities import rsd
from ._plotVariableScatter import plotVariableScatter


def plotRSDs(dataset, featureName='Feature Name', ratio=False, logx=True, xlim=None, withExclusions=True, sortOrder=True, savePath=None, color=None, featName=False, hLines=None, figureFormat='png', dpi=72, figureSize=(11,7)):
	"""
	plotRSDs(dataset, ratio=False, savePath=None, color=None \*\*kwargs)

	Visualise analytical *versus* biological variance.

	Plot RSDs calculated in study-reference samples (analytical variance), versus those calculated in study samples (biological variance). RSDs can be visualised either in absolute terms, or as a ratio to analytical variation (*ratio=*\ ``True``).

	:py:func:`plotRSDs` requires that the dataset have at least two samples with the :py:attr:`~nPYc.enumerations.AssayRole.PrecisionReference` :term:`assay role`, if present, RSDs calculated on independent sets of :py:attr:`~nPYc.enumerations.AssayRole.PrecisionReference` samples will also be plotted.

	:param Dataset dataset: Dataset object to plot, the object must have greater that one 'Study Sample' and 'Study-Reference Sample' defined
	:param bool ratio: If ``True`` plot the ratio of analytical variance to biological variance instead of raw values
	:param str featureName: featureMetadata column name by which to label features
	:param bool logx: If ``True`` plot RSDs on a log10 scaled axis
	:param xlim: Tuple of (min, max) RSD values to plot
	:type xlim: None or tuple(float, float)
	:param hLines: None or list of y positions at which to plot an horizontal line. Features are positioned from 1 to nFeat
	:type hLines: None or list
	:param savePath: If ``None`` plot interactively, otherwise save the figure to the path specified
	:type savePath: None or str
	:param color: Allows the default colour pallet to be overridden
	:type color: None or seaborn.palettes._ColorPalette
	:param bool featName: If ``True`` y-axis label is the feature Name, if ``False`` features are numbered.
	"""
	rsdTable = _plotRSDsHelper(dataset, featureName=featureName, ratio=ratio, withExclusions=withExclusions, sortOrder=sortOrder)

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

	# Add Feature Name if required
	if featName:
		rsdTable['yName'] = rsdTable['Feature Name']
		ylab = 'Feature Name'
	else:
		ylab = 'Feature Number'

	plotVariableScatter(rsdTable, logX=logx, xLim=xLim, xLabel=xlab, yLabel=ylab, sampletypeColor=True, hLines=hLines, vLines=None, savePath=savePath, figureFormat=figureFormat, dpi=dpi, figureSize=figureSize)


def plotRSDsInteractive(dataset, featureName='Feature Name', ratio=False, logx=True):
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
	rsdTable = _plotRSDsHelper(dataset, featureName=featureName, ratio=ratio)

	reversedIndex =  numpy.arange(len(rsdTable)-1,-1, -1)
	data = []
	studySamples = go.Scatter(
		x = rsdTable[SampleType.StudySample].values,
		y = reversedIndex,
		mode = 'markers',
		text = rsdTable['Feature Name'],
		name = str(SampleType.StudySample),
		marker = dict(
			color = 'rgba(89, 117, 164, .8)',
		),
		hoverinfo = 'x+text',
	)
	data.append(studySamples)

	if SampleType.ExternalReference in rsdTable.columns:
		externalRef = go.Scatter(
			x = rsdTable[SampleType.ExternalReference].values,
			y = reversedIndex,
			mode = 'markers',
			text = rsdTable['Feature Name'],
			name = str(SampleType.ExternalReference),
			marker = dict(
				color = 'rgba(181, 93, 96, .8)',
			),
			hoverinfo = 'x+text',
		)
		data.append(externalRef)

	studyPool = go.Scatter(
		x = rsdTable[SampleType.StudyPool].values,
		y = reversedIndex,
		mode = 'markers',
		text = rsdTable['Feature Name'],
		name = str(SampleType.StudyPool),
		 marker = dict(
			color = 'rgba(95, 158, 110, .8)',
		),
		hoverinfo = 'x+text',
	)
	data.append(studyPool)
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

	return figure


def _plotRSDsHelper(dataset, featureName='Feature Name', ratio=False, withExclusions=True, sortOrder=True):  
	
	if not dataset.VariableType == VariableType.Discrete:
		raise ValueError('Only datasets with discreetly sampled variables are supported.')

	if sum(dataset.sampleMetadata.loc[dataset.sampleMask, 'SampleType'].values == SampleType.StudySample) <= 2:
		raise ValueError('More than two Study Samples must be defined to calculate biological RSDs.')

	## Calculate RSD for every SampleType with enough PrecisionReference samples.
	rsdVal = dict()

	precRefMask = dataset.sampleMetadata.loc[:, 'AssayRole'].values == AssayRole.PrecisionReference
	precRefMask = numpy.logical_and(precRefMask, dataset.sampleMask)
	sTypes = list(set(dataset.sampleMetadata.loc[precRefMask, 'SampleType'].values))

	if withExclusions:   
		rsdVal['Feature Name'] = dataset.featureMetadata.loc[dataset.featureMask, featureName].values
		rsdVal[SampleType.StudyPool] = dataset.rsdSP[dataset.featureMask]
		ssMask = (dataset.sampleMetadata['SampleType'].values == SampleType.StudySample) & dataset.sampleMask
		rsdList = rsd(dataset.intensityData[ssMask, :])
		rsdVal[SampleType.StudySample] = rsdList[dataset.featureMask]
	else:		
		rsdVal['Feature Name'] = dataset.featureMetadata.loc[:, featureName].values
		rsdVal[SampleType.StudyPool] = dataset.rsdSP
		ssMask = (dataset.sampleMetadata['SampleType'].values == SampleType.StudySample) & dataset.sampleMask
		rsdList = rsd(dataset.intensityData[ssMask, :])
		rsdVal[SampleType.StudySample] = rsdList		

	# Only keep features with finite values for SP and SS
	finiteMask = (rsdVal[SampleType.StudyPool] < numpy.finfo(numpy.float64).max)
	finiteMask = finiteMask & (rsdVal[SampleType.StudySample] < numpy.finfo(numpy.float64).max)

	for sType in sTypes:
		if not sTypes == SampleType.StudyPool:
			sTypeMask = dataset.sampleMetadata.loc[:, 'SampleType'].values == sType
			# precRefMask limits to Precision Reference and dataset.sampleMask
			sTypeMask = numpy.logical_and(sTypeMask, precRefMask)

			# minimum 3 points needed
			if sum(sTypeMask) >= 3:
				rsdList = rsd(dataset.intensityData[sTypeMask, :])
				if withExclusions:
					rsdVal[sType] = rsdList[dataset.featureMask]
				else:
					rsdVal[sType] = rsdList
				finiteMask = finiteMask & (rsdVal[sType] < numpy.finfo(numpy.float64).max)

	## apply finiteMask
	for sType in rsdVal.keys():
		rsdVal[sType] = rsdVal[sType][finiteMask]

	if ratio:
		rsdSP = copy.deepcopy(rsdVal[SampleType.StudyPool])
		for sType in sTypes:
			rsdVal[sType] = numpy.divide(rsdVal[sType], rsdSP)

	# reorder from largest to smallest RSD in Study Pool
	if sortOrder:
		sortIndex = reversed(numpy.argsort(rsdVal[SampleType.StudyPool]))
		rsdTable = pandas.DataFrame(rsdVal).reindex(sortIndex)
		rsdTable.reset_index(drop=True, inplace=True)
	else:
		rsdTable = pandas.DataFrame(rsdVal)

	return rsdTable
