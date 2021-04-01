import numpy
import re
import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objs as go
from ..enumerations import VariableType
from .. import Dataset, NMRDataset

def plotSpectralVariance(dataset, classes=None, quantiles=(25, 75), average='median', xlim=None, logy=False, title=None, savePath=None, figureFormat='png', dpi=72, figureSize=(11,7)):
	"""
	plotSpectralVariance(dataset, classes=None, quantiles=(25, 75), average='median', xlim=None, **kwargs)

	Plot the average spectral profile of dataset, optionally with the bounds of variance calculated from *quantiles* shaded. By specifying a column from *dataset.sampleMetadata* in the *classes* argument, individual averages and ranges will be plotted for each unique label in *dataset.sampleMetadata[classes]*.
	
	:param Dataset dataset: Data to plot
	:param classes: Plot by distinct classes specified
	:type classes: None or column in dataset.sampleMetadata
	:param quantiles: Plot these quantile bounds
	:type quantiles: None or (min, max)
	:param str average: Method to calculate average spectrum, defaults to 'median', may also be 'mean'
	:param xlim: Tuple of (min, max) values to scale the x-axis to
	:type xlim: None or (float, float)
	:param bool logy: If ``True`` plot intensities on a log10 scale
	:param str title: Text to title each plot with
	"""

	# Check we have a nPYc.Dataset
	if not isinstance(dataset, Dataset):
		raise TypeError('dataset must be a nPYc.Dataset subclass')
	
	# Check we have continuos data
	if dataset.VariableType != VariableType.Continuum:
		raise ValueError('dataset must have spectral variables')

	if not quantiles is None:
		if not len(quantiles) == 2:
			raise ValueError('quantiles must be a tuple of (low, high)')

	##
	# Set up plot
	##
	fig = plt.figure(figsize=figureSize, dpi=dpi)
	ax = plt.subplot(1,1,1)

	##
	# call helper to draw into axis:
	##
	_plotSpectralVarianceHelper(ax, dataset, classes=classes, quantiles=quantiles, average=average, xlim=xlim)

	if title is not None:
		ax.set_title(title)
	if logy:
		ax.set_yscale('symlog', nonpositive='clip')
	else:
		ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
	##
	# Save or draw
	##
	if savePath:
		plt.savefig(savePath, format=figureFormat, dpi=dpi)
		plt.close()
	else:
		plt.show()


def _plotSpectralVarianceHelper(ax, dataset, classes=None, quantiles=(25, 75), average='median', xlim=None):
	"""
	Draws variance plot into ax
	"""
	##
	# If plotting classses, find them here
	##
	classMask = dict()
	if classes is not None:
		if not classes in dataset.sampleMetadata.columns:
			raise ValueError('%s not in dataset.sampleMetadata.' % str(classes))

		uniqueClasses = dataset.sampleMetadata[classes].unique()
		for thisClass in uniqueClasses:
			classMask[thisClass] = (dataset.sampleMetadata[classes].values == thisClass) & dataset.sampleMask
	else:
		classMask['All'] = dataset.sampleMask

	# If xlimits, trim data
	featureMask = dataset.featureMask
	if xlim:
		featureMask = (dataset.featureMetadata['ppm'].values > xlim[0]) & (dataset.featureMetadata['ppm'].values < xlim[1]) & featureMask
	##
	# Loop through classes
	##
	intensityData = dataset.intensityData[:, featureMask]
	for thisClass in classMask.keys():
		# Find average of class
		if average == 'median':
			averageSpectrum = numpy.median(intensityData[classMask[thisClass], :], axis=0)
		elif average == 'mean':
			averageSpectrum = numpy.mean(intensityData[classMask[thisClass], :], axis=0)

		# Draw the average
		base_line, = ax.plot(dataset.featureMetadata.loc[featureMask, 'ppm'].values, averageSpectrum, label=thisClass)

		if quantiles is not None:
			# Find quantile range
			quantileRange = numpy.percentile(intensityData[classMask[thisClass], :], quantiles, axis=0)

			# Draw the range
			ax.fill_between(dataset.featureMetadata.loc[featureMask, 'ppm'], quantileRange[0, :], y2=quantileRange[1, :], alpha=0.5, facecolor=base_line.get_color())

	if classes:
		ax.legend()

	ax.set_xlabel('ppm')

	if isinstance(dataset, NMRDataset):
		ax.invert_xaxis()


def plotSpectralVarianceInteractive(dataset, classes=None, quantiles=(25, 75), average='mean', xlim=None, title=None):
	"""
	Plot the average spectral profile of dataset, optionally with the bounds of variance calculated from *quantiles* shaded. By specifying a column from *dataset.sampleMetadata* in the *classes* argument, individual averages and ranges will be plotted for each unique label in *dataset.sampleMetadata[classes]*.
	
	:param Dataset dataset: Data to plot
	:param classes: Plot by distinct classes specified
	:type classes: None or column in dataset.sampleMetadata
	:param quantiles: Plot these quantile bounds
	:type quantiles: None or (min, max)
	:param str average: Method to calculate average spectrum, defaults to 'median', may also be 'mean'
	:param xlim: Tuple of (min, max) values to scale the x-axis to
	:type xlim: None or (float, float)
	"""
	# Check we have a nPYc.Dataset
	if not isinstance(dataset, Dataset):
		raise TypeError('dataset must be a nPYc.Dataset subclass')

	# Check we have continuos data
	if dataset.VariableType != VariableType.Continuum:
		raise ValueError('dataset must have spectral variables')

	if not quantiles is None:
		if not len(quantiles) == 2:
			raise ValueError('quantiles must be a tuple of (low, high)')

	##
	# If plotting classses, find them here
	##
	classMask = dict()
	if classes is not None:
		if not classes in dataset.sampleMetadata.columns:
			raise ValueError('%s not in dataset.sampleMetadata.' % str(classes))

		uniqueClasses = dataset.sampleMetadata[classes].unique()
		for thisClass in uniqueClasses:
			classMask[thisClass] = (dataset.sampleMetadata[classes].values == thisClass) & dataset.sampleMask
	else:
		classMask['All'] = dataset.sampleMask

	# If xlimits, trim data
	featureMask = dataset.featureMask

	data = list()

	if xlim:
		featureMask = (dataset.featureMetadata['ppm'].values > xlim[0]) & (dataset.featureMetadata['ppm'].values < xlim[1]) & featureMask
	##
	# Loop through classes
	##
	colours = plotly.colors.DEFAULT_PLOTLY_COLORS
	colourIndex = 0
	colourParser = re.compile('rgb\((\d+),\W?(\d+),\W?(\d+)\)')
	for thisClass in classMask.keys():

		localMask = numpy.ix_(numpy.logical_and(dataset.sampleMask, classMask[thisClass]),
							  featureMask)

		# Find average of class
		if average == 'median':
			averageSpectrum = numpy.median(dataset.intensityData[localMask], axis=0)
		elif average == 'mean':
			averageSpectrum = numpy.mean(dataset.intensityData[localMask], axis=0)

		quantileRange = numpy.percentile(dataset.intensityData[localMask], quantiles, axis=0)

		trace = go.Scattergl(
				x = dataset.featureMetadata.loc[featureMask, 'ppm'],
				y = averageSpectrum,
				line = dict(
					color = colours[colourIndex]
				),
				text = thisClass,
				mode = 'lines',
				hoverinfo = 'text',
				name = thisClass,
				legendgroup=thisClass
				)

		classColour = colourParser.match(colours[colourIndex])
		classColour  = 'rgba(%s, %s, %s, 0.2)' % classColour.groups()

		data.append(trace)
		trace = go.Scattergl(
				x = dataset.featureMetadata.loc[featureMask, 'ppm'],
				y = quantileRange[0],
				line = dict(
					color = 'rgba(0,0,0,0)'
				),
				text = False,
				mode = 'lines',
				hoverinfo = 'none',
				showlegend = False,
				legendgroup=thisClass
				)
		data.append(trace)
		trace = go.Scattergl(
				x = dataset.featureMetadata.loc[featureMask, 'ppm'],
				y = quantileRange[1],
				line = dict(
					color = 'rgba(0,0,0,0)'
				),
				fillcolor = classColour,
				fill = 'tonexty',
				text =  False,
				mode = 'lines',
				hoverinfo = 'none',
				showlegend=False,
				legendgroup=thisClass
				)
		data.append(trace)
		colourIndex += 1
		if colourIndex >= len(colours):
			colourIndex = 0

	if True:
		xaxis = 'reversed'
	else:
		xaxis= 'auto'

	layout = go.Layout(
		title=title,
		legend=dict(
			orientation="h"),
		hovermode = "closest",
		yaxis = dict(
			showticklabels=False
		),
		xaxis=dict(
			autorange=xaxis
		)
		)

	figure = go.Figure(data=data, layout=layout)

	return figure
