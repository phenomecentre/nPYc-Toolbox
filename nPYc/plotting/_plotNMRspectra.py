import numpy
import plotly.graph_objs as go

from ..objects._nmrDataset import NMRDataset
from ..enumerations import VariableType


def plotSpectraInteractive(dataset, samples=None, xlim=None, featureNames=None, sampleLabels='Sample ID'):
	"""
	Plot spectra from *dataset*.

	:param Dataset dataset: Dataset to plot from
	:param samples: Index of samples to plot, if ``None`` plot all spectra
	:type samples: None or list of int
	:param xlim: Tuple of (minimum value, maximum value) defining a feature range to plot
	:type xlim: (float, float)
	"""
	if not dataset.VariableType == VariableType.Spectral:
		raise TypeError('Variables in dataset must be continuous.')

	if featureNames is None:
		featureNames = dataset.Attributes['Feature Names']
	elif featureNames not in dataset.featureMetadata.columns:
		raise KeyError('featureNames=%s is not a column in dataset.featureMetadata.' % (featureNames))
	if sampleLabels not in dataset.sampleMetadata.columns:
		raise KeyError('sampleLabels=%s is not a column in dataset.sampleMetadata.' % (sampleLabels))

	##
	# Filter features
	##
	featureMask = dataset.featureMask
	if xlim is not None:
		featureMask = (dataset.featureMetadata[featureNames].values > xlim[0]) & \
					  (dataset.featureMetadata[featureNames].values < xlim[1]) & \
					  featureMask
	features = dataset.featureMetadata.loc[featureMask, 'ppm'].values.squeeze()

	X = dataset.intensityData[:, featureMask]

	##
	# Filter samples
	##
	sampleMask = dataset.sampleMask
	if samples is None:
		samples = numpy.arange(X.shape[0])[sampleMask]
	elif isinstance(samples, int):
		samples = [samples]
	elif isinstance(samples, numpy.ndarray):
		sampleMask = sampleMask & samples
		samples = numpy.arange(dataset.noSamples)[sampleMask]

	data = list()
	if X.ndim == 1:
		trace = go.Scattergl(
			x = features,
			y = X,
			name = dataset.sampleMetadata.loc[samples, sampleLabels],
			mode = 'lines',
			)
		data.append(trace)
	else:
		for i in samples:
			trace = go.Scattergl(
				x = features,
				y = X[i, :],
				name = str(dataset.sampleMetadata.loc[i, sampleLabels]),
				mode = 'lines',
				)
			data.append(trace)

	if isinstance(dataset, NMRDataset):
		autorange = 'reversed'
	else:
		autorange = True

	layout = go.Layout(
				# title='',
				legend=dict(
					orientation="h"),
				hovermode = "closest",
				xaxis=dict(
						autorange=autorange
					),
				yaxis = dict(
					showticklabels=False
				),
				)
	figure = go.Figure(data=data, layout=layout)

	return figure
