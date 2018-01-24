# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import seaborn as sns
import numpy
import pandas as pd
from os.path import basename
from nPYc.enumerations import SampleType
import copy
import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.graph_objs as go


def plotPW(nmrData, savePath=None, figureFormat='png', dpi=72, figureSize=(11,7)):
	"""
	plotPW(nmrData, savePath=None, figureFormat='png', dpi=72, figureSize=(11,7))
	
	Visualise peak width values as box plot
	
	:param nmrData nmrdataset
		Dataset object
	:param title string
		Text for the figure title
	:param savePath None or str
		If None, plot interactively, otherwise attempt to save at this path
	:param format str
		Format to save figure
	:param dpi int
		Resolution to draw at
	:param figureSize tuple
		Specify size of figure
	"""

	# Load toolbox wide color scheme
	if 'sampleTypeColours' in nmrData.Attributes.keys():
		sTypeColourDict = copy.deepcopy(nmrData.Attributes['sampleTypeColours'])
		for stype in SampleType:
			if stype.name in sTypeColourDict.keys():
				sTypeColourDict[stype] = sTypeColourDict.pop(stype.name)
	else:
		sTypeColourDict = {SampleType.StudySample: 'b', SampleType.StudyPool: 'g', SampleType.ExternalReference: 'r',
						   SampleType.MethodReference: 'm', SampleType.ProceduralBlank: 'c', 'Other': 'grey'}

	sns.set_color_codes(palette='deep')	
	fig, ax = plt.subplots(1, figsize=figureSize, dpi=dpi)
	#all the pw values
	pw_ER=nmrData.sampleMetadata.iloc[nmrData.sampleMetadata['SampleType'].values == SampleType.ExternalReference]['Line Width (Hz)']
	pw_SP=nmrData.sampleMetadata.iloc[nmrData.sampleMetadata['SampleType'].values == SampleType.StudyPool]['Line Width (Hz)']
	pw_SS=nmrData.sampleMetadata.iloc[nmrData.sampleMetadata['SampleType'].values == SampleType.StudySample]['Line Width (Hz)']
	pw_unknown=nmrData.sampleMetadata.iloc[nmrData.sampleMetadata['SampleType'].isnull().values]['Line Width (Hz)']#this is the ones that either had missing info or for what ever reason the status column has a nan or null value and cannot identify as ltr, sr etc
	tempDF=pd.concat([pw_SS, pw_SP, pw_ER, pw_unknown], axis=1)#put them all together in a new df
	tempDF.columns = [SampleType.StudySample, SampleType.StudyPool, SampleType.ExternalReference, 'Other']
	tempDF.dropna(axis='columns', how='all', inplace=True) # remove empty columns
	
		#all the outliers only
	pw_ER=pw_ER.where(pw_ER>nmrData.Attributes['PWFailThreshold'], numpy.nan) #anything smaller than pw threshold set to NaN so doesnt plot
	pw_SP=pw_SP.where(pw_SP>nmrData.Attributes['PWFailThreshold'], numpy.nan)
	pw_SS=pw_SS.where(pw_SS>nmrData.Attributes['PWFailThreshold'], numpy.nan)
	pw_unknown=pw_unknown.where(pw_unknown>nmrData.Attributes['PWFailThreshold'], numpy.nan)
	tempDF_outliers=pd.concat([pw_SS, pw_SP, pw_ER, pw_unknown], axis=1)#put them all together in a new df
	tempDF_outliers.columns = ['SS', 'SP','ER', 'Unknown(missing status)']
	tempDF_outliers.dropna(axis='columns', how='all', inplace=True) # remove empty columns
	
	#plot fail threshold line only if there exists values greater than the fail (normally 1.4 set in SOP)
	if numpy.max(nmrData.sampleMetadata['Line Width (Hz)'])>nmrData.Attributes['PWFailThreshold']:#numpy.max(tempDF)[0]>nmrData.Attributes['PWFailThreshold']:
		plt.plot([-0.5,5], [nmrData.Attributes['PWFailThreshold'],nmrData.Attributes['PWFailThreshold']], 'r-', label='PW fail threshold')#-0.5 to 3 as on box plot as i understand the first plot is 1 at x so put to 3 as we have max of 3 categories normally (may have to revisit this)
		plt.legend(loc='best')#also we only need legend if the line is present

	if numpy.size(tempDF_outliers)>0:#we dont attempt to plot if their is no outliers
		sns.stripplot(data=tempDF_outliers, jitter=True, color="red")#overlay strip plot so can see the data points are outliers
	sns.violinplot(data=tempDF, cut=0, palette=sTypeColourDict) #cut:Set to 0 to limit the violin range within the range of the observed data
	plt.suptitle('Peak width values in Hz')
	plt.xlabel('peak width')
	plt.ylabel('Hz')		
	if savePath:
		plt.savefig(savePath, bbox_inches='tight', format=figureFormat, dpi=dpi)
		plt.close()
	else:
		plt.show()	


def plotLineWidth(nmrData, savePath=None, figureFormat='png', dpi=72, figureSize=(11,7)):
	"""
	plotLineWidth(nmrData, **kwargs)
	
	Visualise spectral line widths, plotting the median spectrum, the 95% variance, and any spectra where line width can not be calulated or exceeds the cutoff specified in ``nmrData.Attributes['LWpeakRange']``.

	:param NMRDataset nmrData: Dataset object
	:param savePath: If None, plot interactively, otherwise attempt to save at this path
	:type savePath: None or str
	"""

	localPPM, ppmMask, meanSpectrum, lowerPercentile, upperPercentile = nmrRangeHelper(nmrData, nmrData.Attributes['LWpeakRange'], percentiles=(5, 95))

	globalMask = numpy.ix_(nmrData.sampleMask, ppmMask)

	fig, ax = plt.subplots(1, 1, sharey=True)

	ax.plot(localPPM, meanSpectrum, color=(0.46,0.71,0.63))
	ax.fill_between(localPPM, lowerPercentile, y2=upperPercentile, color=(0,0.4,.3,0.2))

	##
	# Plot failures
	##
	for i in range(nmrData.noSamples):

		if nmrData.sampleMetadata.loc[i, 'Line Width (Hz)'] <= nmrData.Attributes['PWFailThreshold']:
			ax.plot(localPPM, nmrData.intensityData[i, ppmMask], color=(0.8,0.05,0.01,0.7))

		if numpy.isnan(nmrData.sampleMetadata.loc[i, 'Line Width (Hz)']):
			ax.plot(localPPM, nmrData.intensityData[i, ppmMask], color=(0.05,0.05,0.8,0.7))

	ax.axvline(nmrData.Attributes['calibrateTo'], color='k', linestyle='--')
	ax.set_xlabel('ppm')
	ax.invert_xaxis()

	##
	# Set up legend
	##
	variance = mpatches.Patch(color=(0,0.4,.3,0.2), label='Variance about the median')

	failures = mlines.Line2D([], [], color=(0.8,0.05,0.01,0.7), marker='',
							label='Exceeded linewidth cuttoff of %.2f Hz' % (nmrData.Attributes['PWFailThreshold']))
	uncalculated = mlines.Line2D([], [], color=(0.05,0.05,0.8,0.7), marker='',
							label='Linewidth could not be calculated')
	plt.legend(handles=[variance, failures, uncalculated])

	##
	# Save or draw
	##
	if savePath:
		fig.savefig(savePath, format=figureFormat, dpi=dpi)
		plt.close()
	else:
		plt.show()


def plotLineWidthInteractive(nmrData):
	"""
	Interactive Plotly version of py:func:`plotLineWidth`
	
	Visualise spectral line widths, plotting the median spectrum, the 95% variance, and any spectra where line width can not be calulated or exceeds the cutoff specified in ``nmrData.Attributes['LWpeakRange']``.

	:param NMRDataset nmrData: Dataset object
	:param savePath: If None, plot interactively, otherwise attempt to save at this path
	:type savePath: None or str
	"""
	localPPM, ppmMask, meanSpectrum, lowerPercentile, upperPercentile = nmrRangeHelper(nmrData, nmrData.Attributes['LWpeakRange'], percentiles=(5, 95))

	globalMask = numpy.ix_(nmrData.sampleMask, ppmMask)

	data = []
	failedCalculation = []
	failedCutoff = []

	##
	# Plot overall dataset variance
	##
	data = data + plotlyRangeHelper(localPPM, meanSpectrum, lowerPercentile, upperPercentile)

	trace = go.Scatter(
			x = nmrData.featureMetadata.loc[:, 'ppm'].values[ppmMask],
			y = nmrData.intensityData[0, ppmMask],
			name ='Mean Spectrum and variance',
			line = dict(
				color = ('rgb(117,182,160)')
			),
			mode = 'lines',
			visible = 'legendonly'
			)
	data.append(trace)
	trace = go.Scatter(
			x = nmrData.featureMetadata.loc[:, 'ppm'].values[ppmMask],
			y = nmrData.intensityData[0, ppmMask],
			line = dict(
				color = ('rgb(205, 12, 24)')
			),
			name = 'Spectra exceeding LW cutoff of %.2f Hz' % (nmrData.Attributes['PWFailThreshold']),
			mode = 'lines',
			visible = 'legendonly'
			)
	data.append(trace)
	trace = go.Scatter(
			x = nmrData.featureMetadata.loc[:, 'ppm'].values[ppmMask],
			y = nmrData.intensityData[0, ppmMask],
			line = dict(
				color = ('rgb(12, 12, 205)')
			),
			fillcolor = 'rgba(0,100,80,0.2)',
			name = 'Spectra where LW could not be calculated',
			mode = 'lines',
			visible = 'legendonly'
			)
	data.append(trace)

	##
	# Plot failures
	##
	for i in range(nmrData.noSamples):

		if nmrData.sampleMetadata.loc[i, 'Line Width (Hz)'] > nmrData.Attributes['PWFailThreshold']:

			trace = go.Scatter(
				x = nmrData.featureMetadata.loc[:, 'ppm'].values[ppmMask],
				y = nmrData.intensityData[i, ppmMask],
				line = dict(
					color = ('rgba(205, 12, 24, 0.7)')
				),
				text = '%.2f Hz,<br />%s' % (nmrData.sampleMetadata.loc[i, 'Line Width (Hz)'], nmrData.sampleMetadata.loc[i, 'Sample File Name']),
				hoverinfo = 'text',
				showlegend = False
			)

			failedCutoff.append(trace)

		if numpy.isnan(nmrData.sampleMetadata.loc[i, 'Line Width (Hz)']):

			trace = go.Scatter(
				x = nmrData.featureMetadata.loc[:, 'ppm'].values[ppmMask],
				y = nmrData.intensityData[i, ppmMask],
				line = dict(
					color = ('rgba(12, 12, 205, 0.7)')
				),
				text = nmrData.sampleMetadata.loc[i, 'Sample File Name'],
				hoverinfo = 'text',
				showlegend = False
			)
			failedCalculation.append(trace)

	data = data + failedCutoff
	data = data + failedCalculation

	layout = go.Layout(
				title='Line width',
				legend=dict(
					orientation="h"),
				yaxis = dict(
					showticklabels=False
				),
				hovermode = "closest",
				xaxis=dict(
						autorange='reversed'
					)
				)
	figure = go.Figure(data=data, layout=layout)

	return figure


def nmrRangeHelper(nmrData, bounds, percentiles=(5, 95)):
	"""
	
	"""
	ppmMask = (nmrData.featureMetadata.loc[:, 'ppm'].values >= bounds[0]) & (nmrData.featureMetadata.loc[:, 'ppm'].values <= bounds[1])
	localPPM = nmrData.featureMetadata.loc[ppmMask, 'ppm'].values

	globalMask = numpy.ix_(nmrData.sampleMask, ppmMask)

	medianSpectrum = numpy.median(nmrData.intensityData[globalMask], axis=0)
	lowerPercentile = numpy.percentile(nmrData.intensityData[globalMask], percentiles[0], axis=0)
	upperPercentile = numpy.percentile(nmrData.intensityData[globalMask], percentiles[1], axis=0)

	return localPPM, ppmMask, medianSpectrum, lowerPercentile, upperPercentile


def plotlyRangeHelper(localPPM, meanSpectrum, lowerPercentile, upperPercentile, xaxis=None):
	"""
	"""
	data = []
	trace = go.Scatter(
			x = localPPM,
			y = meanSpectrum,
			text ='Mean Spectrum',
			line = dict(
				color = ('rgb(117,182,160)')
			),
			mode = 'lines',
			hoverinfo = 'text',
			showlegend=False,
			xaxis=xaxis
			)
	data.append(trace)
	trace = go.Scatter(
			x = localPPM,
			y = upperPercentile,
			line = dict(
				color = ('rgba(117,182,160,0.2)')
			),
			text = '95% bound',
			mode = 'lines',
			hoverinfo = 'text',
			showlegend=False,
			xaxis=xaxis
			)
	data.append(trace)
	trace = go.Scatter(
			x = localPPM,
			y = lowerPercentile,
			line = dict(
				color = ('rgba(117,182,160,0.2)')
			),
			fillcolor = 'rgba(0,100,80,0.2)',
			fill = 'tonexty',
			text = '5% bound',
			mode = 'lines',
			hoverinfo = 'text',
			showlegend=False,
			xaxis=xaxis
			)
	data.append(trace)

	return data
