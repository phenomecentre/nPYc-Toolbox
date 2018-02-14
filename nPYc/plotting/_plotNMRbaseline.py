import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
import seaborn as sns
import numpy
import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.graph_objs as go

from ._nmrPlotting import nmrRangeHelper, plotlyRangeHelper

def plotBaseline(nmrData, savePath=None, figureFormat='png', dpi=72, figureSize=(11,7)):
	"""
	plotBaseline(nmrData, savePath=None, **kwargs)

	Plot spectral baseline at the high and low end of the spectrum. Visualise the median, bounds of 95% variance, and outliers.

	:param NMRDataset nmrData: Dataset object
	:param savePath: If None, plot interactively, otherwise attempt to save at this path
	:type savePath: None or str
	"""
	fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(15, 7), dpi=72)

	localPPM, ppmMask, meanSpectrum, lowerPercentile, upperPercentile = nmrRangeHelper(nmrData, (min(nmrData.Attributes['baselineCheckRegion'][0]),max(nmrData.Attributes['baselineCheckRegion'][0])), percentiles=(5, 95))
	ax2.plot(localPPM, meanSpectrum, color=(0.46,0.71,0.63))
	ax2.fill_between(localPPM, lowerPercentile, y2=upperPercentile, color=(0,0.4,.3,0.2))

	for i in range(nmrData.noSamples):

		if nmrData.sampleMetadata.loc[i, 'BaselineFail']:
			ax2.plot(localPPM, nmrData.intensityData[i, ppmMask], color=(0.05,0.05,0.8,0.7))

	localPPM, ppmMask, meanSpectrum, lowerPercentile, upperPercentile = nmrRangeHelper(nmrData, (min(nmrData.Attributes['baselineCheckRegion'][1]), max(nmrData.Attributes['baselineCheckRegion'][1])), percentiles=(5, 95))
	ax1.plot(localPPM, meanSpectrum, color=(0.46,0.71,0.63))
	ax1.fill_between(localPPM, lowerPercentile, y2=upperPercentile, color=(0,0.4,.3,0.2))

	for i in range(nmrData.noSamples):

		if nmrData.sampleMetadata.loc[i, 'BaselineFail']:
			ax1.plot(localPPM, nmrData.intensityData[i, ppmMask], color=(0.05,0.05,0.8,0.7))

	ax1.set_xlabel('ppm')
	ax1.invert_xaxis()
	ax1.get_yaxis().set_ticks([])

	ax2.set_xlabel('ppm')
	ax2.invert_xaxis()
	ax2.get_yaxis().set_ticks([])
	##
	# Set up legend
	##
	variance = patches.Patch(color=(0,0.4,0.3,0.2), label='Variance about the median')

	failures = lines.Line2D([], [], color=(0.05,0.05,0.8,0.7), marker='',
							label='Baseline failed on area')
	plt.legend(handles=[variance, failures])

	if savePath:
		plt.savefig(savePath, bbox_inches='tight', format=figureFormat, dpi=dpi)
		plt.close()
	else:
		plt.show()


def plotBaselineInteractive(nmrData):
	"""
	Interactive Plotly version of py:func:`plotBaseline`.

	Plot spectral baseline at the high and low end of the spectrum. Visualise the median, bounds of 95% variance, and outliers.

	:param NMRDataset nmrData: Dataset object
	"""
	data = []
	failedHigh = []
	failedLow = []

	localPPM, ppmMask, meanSpectrum, lowerPercentile, upperPercentile = nmrRangeHelper(nmrData, (min(nmrData.Attributes['baselineCheckRegion'][0]),max(nmrData.Attributes['baselineCheckRegion'][0])), percentiles=(5, 95))
	trace = plotlyRangeHelper(localPPM, meanSpectrum, lowerPercentile, upperPercentile, xaxis='x2')
	data = data + trace

	##
	# Plot failures low
	##
	for i in range(nmrData.noSamples):

		if nmrData.sampleMetadata.loc[i, 'BaselineFail']:
			trace = go.Scatter(
				x = nmrData.featureMetadata.loc[:, 'ppm'].values[ppmMask],
				y = nmrData.intensityData[i, ppmMask],
				line = dict(
					color = ('rgb(12, 12, 205)')
				),
				text = '%s' % (nmrData.sampleMetadata.loc[i, 'Sample File Name']),
				hoverinfo = 'text',
				showlegend = False,
				xaxis='x2'
			)
			failedLow.append(trace)

	localPPM, ppmMask, meanSpectrum, lowerPercentile, upperPercentile = nmrRangeHelper(nmrData, (min(nmrData.Attributes['baselineCheckRegion'][1]),max(nmrData.Attributes['baselineCheckRegion'][1])), percentiles=(5, 95))
	trace = plotlyRangeHelper(localPPM, meanSpectrum, lowerPercentile, upperPercentile)
	data = data + trace

	##
	# Plot failures high
	##
	for i in range(nmrData.noSamples):

		if nmrData.sampleMetadata.loc[i, 'BaselineFail']:

			trace = go.Scatter(
				x = nmrData.featureMetadata.loc[:, 'ppm'].values[ppmMask],
				y = nmrData.intensityData[i, ppmMask],
				line = dict(
					color = ('rgb(12, 12, 205)')
				),
				text = '%s' % (nmrData.sampleMetadata.loc[i, 'Sample File Name']),
				hoverinfo = 'text',
				showlegend = False
			)
			failedHigh.append(trace)

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
			color = ('rgb(12, 12, 205)')
		),
		fillcolor = 'rgba(0,100,80,0.2)',
		name = 'Spectra failing on baseline area',
		mode = 'lines',
		visible = 'legendonly'
		)
	data.append(trace)

	data = data + failedHigh
	data = data + failedLow

	layout = go.Layout(
				title='Baseline plot',
				legend=dict(
					orientation="h"),
				hovermode = "closest",
				xaxis=dict(
					domain = [0, 0.48],
					range=[(max(nmrData.Attributes['baselineCheckRegion'][1]),min(nmrData.Attributes['baselineCheckRegion'][1]))]
				),
				xaxis2=dict(
					domain = [0.52, 1],
					range=[(max(nmrData.Attributes['baselineCheckRegion'][0]),min(nmrData.Attributes['baselineCheckRegion'][0]))]
					),
				)

	figure = go.Figure(data=data, layout=layout)

	return figure
