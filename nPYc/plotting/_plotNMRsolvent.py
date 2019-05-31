import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
import seaborn as sns
import numpy
import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly.graph_objs as go

from ._nmrPlotting import nmrRangeHelper, plotlyRangeHelper

def plotSolventResonance(nmrData, savePath=None, figureFormat='png', dpi=72, figureSize=(11,7)):
	"""
	plotSolventResonance(nmrData, **kwargs)

	Plot the solvent region to be cut from the spectrum along with spectra failing solvent region checks.

	:param NMRDataset nmrData: Dataset to plot
	:param savePath: If ``None`` draw interactively, otherwise save to this path
	:type savePath: None or str
	"""
	bounds = (min(nmrData.Attributes['solventPeakCheckRegion'][0]), max(nmrData.Attributes['solventPeakCheckRegion'][1]))

	localPPM, ppmMask, meanSpectrum, lowerPercentile, upperPercentile = nmrRangeHelper(nmrData, bounds, percentiles=(5, 95))

	globalMask = numpy.ix_(nmrData.sampleMask, ppmMask)

	fig, ax = plt.subplots(1, 1, sharey=True, figsize=figureSize, dpi=dpi)

	ax.plot(localPPM, meanSpectrum, color=(0.46,0.71,0.63))

	ax.fill_between(localPPM, lowerPercentile, y2=upperPercentile, color=(0,0.4,.3,0.2))

	##
	# Plot failures
	##
	for i in range(nmrData.noSamples):

		if nmrData.sampleMetadata.loc[i, 'SolventPeakFail']:
				ax.plot(localPPM, nmrData.intensityData[i, ppmMask], color=(0.8,0.05,0.01,0.7))

		if nmrData.sampleMetadata.loc[i, 'SolventPeakFail']:
			ax.plot(localPPM, nmrData.intensityData[i, ppmMask], color=(0.05,0.05,0.8,0.7))


	ax.axvspan(max(nmrData.Attributes['solventPeakCheckRegion'][0]),
			   min(nmrData.Attributes['solventPeakCheckRegion'][1]), facecolor='k', alpha=0.2)
	# ax.set_xlabel('ppm')
	ax.invert_xaxis()
	ax.get_yaxis().set_ticks([])
	##
	# Set up legend
	##
	variance = patches.Patch(color=(0,0.4,.3,0.2), label='Variance about the median')
	water = patches.Patch(color=(0,0,0,0.2), label='Solvent region to be removed')
	failures = lines.Line2D([], [], color=(0.8,0.05,0.01,0.7), marker='',
									label='Solvent resonances failed on area')

	plt.legend(handles=[variance, water, failures])

	if savePath:
		plt.savefig(savePath, bbox_inches='tight', format=figureFormat, dpi=dpi)
		plt.close()
	else:
		plt.show()


def plotSolventResonanceInteractive(nmrData, title='Residual solvent resonance'):
	"""
	Ploty interactive version of :py:func:`plotSolventResonance`

	Plot the solvent region to be cut from the spectrum along with spectra failing solvent region checks.

	:param NMRDataset nmrData: Dataset to plot
	:returns: Plotly figure object to plot with iPlot
	"""
	data = []
	failed = []

	bounds = (min(nmrData.Attributes['solventPeakCheckRegion'][0]), max(nmrData.Attributes['solventPeakCheckRegion'][1]))

	localPPM, ppmMask, meanSpectrum, lowerPercentile, upperPercentile = nmrRangeHelper(nmrData, bounds, percentiles=(5, 95))
	trace = plotlyRangeHelper(localPPM, meanSpectrum, lowerPercentile, upperPercentile)
	data = data + trace

	##
	# Add fails
	##
	for i in range(nmrData.noSamples):

		if nmrData.sampleMetadata.loc[i, 'SolventPeakFail']:

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
			failed.append(trace)

	data = data + failed

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

	##
	# Add water region in layout
	##
	layout = go.Layout(
		title=title,
		legend=dict(
			orientation="h"),
		hovermode = "closest",
		xaxis=dict(
			range=(min(nmrData.Attributes['solventPeakCheckRegion'][0]), max(nmrData.Attributes['solventPeakCheckRegion'][1]))
		),
		yaxis = dict(
			showticklabels=False
		),
		shapes= [
			{
				'type': 'rect',
				'yref': 'paper',
				'x0': max(nmrData.Attributes['solventPeakCheckRegion'][0]),
				'y0': 0,
				'x1': min(nmrData.Attributes['solventPeakCheckRegion'][1]),
				'y1': 1,
				'line': {
					'color': 'rgb(0, 0, 0, 0)',
					'width': 0,
					},
				'fillcolor': 'rgba(128, 128, 128, 0.5)',
				},
			]
		)

	figure = go.Figure(data=data, layout=layout)

	return figure
