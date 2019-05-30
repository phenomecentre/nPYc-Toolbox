import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import numpy
import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.graph_objs as go

from ._nmrPlotting import nmrRangeHelper, plotlyRangeHelper

def plotCalibration(nmrData, savePath=None, figureFormat='png', dpi=72, figureSize=(11,7)):
	"""
	plotCalibration(nmrData, savePath=None, **kwargs)

	Visualise calibration of all spectra

	:param NMRDataset nmrData: Dataset object
	:param savePath: If None, plot interactively, otherwise attempt to save at this path
	:type savePath: None or str
	"""
	fig, ax = plt.subplots(1, figsize=figureSize, dpi=dpi)

	localPPM, ppmMask, meanSpectrum, lowerPercentile, upperPercentile = nmrRangeHelper(nmrData, nmrData.Attributes['ppmSearchRange'], percentiles=(5, 95))

	ax.plot(localPPM, meanSpectrum, color=(0.46,0.71,0.63))
	ax.fill_between(localPPM, lowerPercentile, y2=upperPercentile, color=(0,0.4,.3,0.2))

	# for i in range(nmrData.noSamples):
	# 	if nmrData.sampleMetadata.loc[i, 'Line Width (Hz)'] <= nmrData.Attributes['PWFailThreshold']:
	# 		ax.plot(localPPM, nmrData.intensityData[i, localPPM], color=(0.46,0.71,0.63))
	# 	else:
	# 		ax.plot(localPPM, nmrData.intensityData[i, localPPM], color=(0.05,0.05,0.8))

	for i in range(nmrData.noSamples):

		if nmrData.sampleMetadata.loc[i, 'CalibrationFail']:
			ax.plot(localPPM, nmrData.intensityData[i, ppmMask], color=(0.8, 0.05, 0.01, 0.7))

	ax.axvline(nmrData.Attributes['calibrateTo'], color='k', linestyle='--')
	variance = patches.Patch(color=(0,0.4,.3,0.2), label='Variance about the median')
	plt.legend(handles=[variance])

	plt.xlabel('ppm')
	ax.invert_xaxis()	
	ax.get_yaxis().set_ticks([])

	if savePath:
		plt.savefig(savePath, bbox_inches='tight', format=figureFormat, dpi=dpi)
		plt.close()
	else:
		plt.show()


def plotCalibrationInteractive(nmrData):
	"""
	Build Plotly figure of calibration
	
	:param NMRDataset nmrData: Dataset to visualise
	:returns: Plotly figure object for displaly with iplot()
	:rtype: plotly.graph_objs.Figure
	"""
	localPPM, ppmMask, meanSpectrum, lowerPercentile, upperPercentile = nmrRangeHelper(nmrData, nmrData.Attributes['ppmSearchRange'], percentiles=(5, 95))

	data = []
	failed = []
	##
	# Plot overall dataset variance
	##
	trace = plotlyRangeHelper(localPPM, meanSpectrum, lowerPercentile, upperPercentile)
	data = data + trace

	for i in range(nmrData.noSamples):

		if nmrData.sampleMetadata.loc[i, 'CalibrationFail']:

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
		x = [nmrData.Attributes['calibrateTo'], nmrData.Attributes['calibrateTo']],
		y = [0,1],
		name ='Calibration target position',
		line = dict(
			color = ('rgb(10, 10, 200)')
		),
		mode = 'lines',
		)
	data.append(trace)

	layout = go.Layout(
				#title='Chemical shift registration',
				legend=dict(
					orientation="h"),
				hovermode = "closest",
				xaxis=dict(
						autorange='reversed'
					),
				yaxis = dict(
					showticklabels=False
				),
				shapes=[
					# Line Vertical
						{
							'type': 'line',
							'xref': 'x',
							'yref': 'paper',
							'x0': nmrData.Attributes['calibrateTo'],
							'y0': 0,
							'x1': nmrData.Attributes['calibrateTo'],
							'y1': 1,
							'line': {
								'color': 'rgb(10, 10, 200)',
								'width': 3,
							},
						}
					]
				)
	figure = go.Figure(data=data, layout=layout)

	return figure
