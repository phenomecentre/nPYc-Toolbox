import plotly.plotly as py
import plotly.graph_objs as go
import seaborn
import numpy

from ..objects._nmrDataset import NMRDataset
from ..utilities._internal import _vcorrcoef

def correlationSpectroscopyInteractive(dataset, target, mode='SHY', correlationMethod='Pearson'):
	"""
	Conduct correlation spectroscopy analyses against the samples in *dataset*.

	Mode may be one of:
	- **SHY** Correlate features in *dataset* to values in *target*

	:param Dataset dataset: Correlations weill be projected into this dataset
	:param numpy.array target: Correlations are calculated to this
	:param str mode: Type of analysis to conduct
	:param str correlationMethod: Type of correlation to calculate, may be 'Pearson', or 'Spearman'
	:returns: Plotly figure
	:rtype:
	"""
	if mode.lower() == 'shy':
		colour = _vcorrcoef(dataset.intensityData[dataset.sampleMask, :], target, method=correlationMethod)

	magnitude = numpy.mean(dataset.intensityData[dataset.sampleMask, :],axis=0)
	
	plot = plotyShadedLineplot(dataset.featureMetadata.loc[dataset.featureMask, 'ppm'], magnitude[dataset.featureMask], colour[dataset.featureMask])
	
	if isinstance(dataset, NMRDataset):
		xaxis = 'reversed'
	else:
		xaxis= 'auto'
	
	layout = go.Layout(
		title=None,
		legend=dict(
			orientation="h"),
		hovermode = "closest",
		yaxis = dict(
			showticklabels=False
		),
		xaxis=dict(
			autorange=xaxis, 
			title='PPM'
		)
	)

	figure = go.Figure(data=plot, layout=layout)
	
	return figure


def plotyShadedLineplot(x, y, colour, shadeLevels=24):
	"""
	Hack arrund the fact that plotly cannot colour individual segments of a line plot seperatly
	"""
	colourScale = seaborn.color_palette("hls", n_colors=shadeLevels)
	
	minC = min(colour)
	maxC = max(colour)
	
	rangeC = maxC - minC
	stepC = rangeC / shadeLevels
	cutoff = minC

	plotData = list()
	for i in range(shadeLevels):

		mask = numpy.zeros_like(colour, dtype=bool)

		# Build the mask
		for j in range(1, len(colour) - 1):

			if (colour[j] >= cutoff) & (colour[j] <= (cutoff + stepC)):
				# Do this in a loop so we can ensure every line has at least 
				# three points and will be drawn.
				mask[(j-1):(j+2)] = True

		cutoff += stepC

		# Now loop through extracting contiguous chunks
		start = None
		rangeList = list()
		for j in range(len(colour)):
			if mask[j]:
				if start is None:
					start = j
			else:
				if start is not None:
					rangeList.append((start, j -1))
					start = None
		if start is not None:
			 rangeList.append((start, len(colour)))

		for traceRange in rangeList:
			trace = go.Scatter(
					x = x[traceRange[0]:traceRange[1]],
					y = y[traceRange[0]:traceRange[1]],
					mode = 'lines',
					text = colour[traceRange[0]:traceRange[1]],
					line = dict(
						color = 'rgb(%f, %f, %f)' % colourScale[i],
					),
					showlegend = False,
				)
			plotData.append(trace)
	return plotData
