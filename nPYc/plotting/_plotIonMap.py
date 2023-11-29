import numpy
import matplotlib.pyplot as plt
import plotly.graph_objs as go

from ..objects import MSDataset
from ..enumerations import VariableType

def plotIonMap(msData, useRetention=True, title=None, savePath=None, xlim=None, ylim=None, logx=False, logy=False, figureFormat='png', dpi=72, figureSize=(11,7)):
	"""
	plotIonMap(msData, \*\*kwargs):

	Visualise features in a MSDataset, to visualise the features in terms of the raw data.

	Plotting requires the presence of 'm/z' and 'Retention Time' columns in the :py:attr:`~nPYc.objects.Dataset.featureMetadata` table. If both 'm/z' and retention time are present, a 2D ion map is generated, otherwise a 1D mass-spectrum is plotted.

	:param MSDataset msData: Dataset object to visualise
	:param bool useRetention: If ``False`` ignore any Retention Time information and plot a 1D mass spectrum
	"""
	##
	# Check inputs
	##
	if not isinstance(msData, MSDataset):
		raise TypeError('msData must be an instance of MSDataset.')
	if not 'm/z' in msData.featureMetadata.columns:
		raise KeyError('msData must have m/z in the featureMetadata to plot.')

	fig, ax = plt.subplots(figsize=figureSize, dpi=dpi)

	if ('Retention Time' in msData.featureMetadata.columns) & useRetention:
		_plotIonMap(ax, msData, xlim, ylim)
	else:
		_plotMassSpectrum(ax, msData, xlim, ylim)
		useRetention = False

	if xlim:
		ax.set_xlim(xlim)
	elif useRetention:
		ax.set_xlim([0, msData.featureMetadata['Retention Time'].max()+msData.featureMetadata['Retention Time'].min()])
	if ylim:
		ax.set_ylim(ylim)
	elif useRetention:
		ax.set_ylim([0, msData.featureMetadata['m/z'].max()+msData.featureMetadata['m/z'].min()])

	if logy:
		ax.set_yscale('symlog')
	if logx:
		ax.set_xscale('symlog')
	if title:
		fig.suptitle(title)

	# Save or show
	if savePath:
		plt.savefig(savePath, bbox_inches='tight', format=figureFormat, dpi=dpi)
		plt.close()
	else:
		plt.show()


def _plotIonMap(ax, msData, xlim, ylim):
	##
	# Mask out data based on limits
	##
	featureMask = msData.featureMask
	if xlim:
		featureMask = (msData.featureMetadata['Retention Time'].values > xlim[0]) & (msData.featureMetadata['Retention Time'].values < xlim[1]) & featureMask
	if ylim:
		featureMask = (msData.featureMetadata['m/z'].values > ylim[0]) & (msData.featureMetadata['m/z'].values < ylim[1]) & featureMask

	# Alpha determined by intensity
	alphas = numpy.median(msData.intensityData[:, featureMask], axis=0)
	alphas[alphas==0] = numpy.min(alphas[alphas!=0])
	alphas = numpy.log(alphas)

	cb = ax.scatter(msData.featureMetadata.loc[featureMask, 'Retention Time'], msData.featureMetadata.loc[featureMask, 'm/z'],
					c = alphas, cmap = plt.colormaps.get_cmap('Blues'),
					alpha=0.3, edgecolors='k')

	cbar = plt.colorbar(cb)
	cbar.set_label('Log median intensity')

	ax.set_xlabel('Retention Time')
	ax.set_ylabel('m/z')


def _plotMassSpectrum(ax, msData, xlim, ylim):
	##
	# Plot a 1D spectrum
	##
	featureMask = msData.featureMask
	if xlim:
		featureMask = (msData.featureMetadata['m/z'].values > xlim[0]) & (msData.featureMetadata['m/z'].values < xlim[1]) & featureMask

	intensities = numpy.median(msData.intensityData[:, featureMask], axis=0)

	if msData.VariableType == VariableType.Discrete:
		ax.vlines(msData.featureMetadata.loc[featureMask, 'm/z'], [0], intensities)
	elif msData.VariableType == VariableType.Spectral:
		ax.plot(msData.featureMetadata.loc[featureMask, 'm/z'], intensities)

	ax.set_ylabel('Median intensity')
	ax.set_xlabel('m/z')


def plotIonMapInteractive(dataset, title=None, xlim=None, ylim=None, logx=False, logy=False, featureName='Feature Name'):
	"""
	Visualise features in a MSDataset, as an ion map.

	Plotting requires the presence of 'm/z' and 'Retention Time' columns in the :py:attr:`~nPYc.objects.Dataset.featureMetadata` table.

	:param MSDataset msData: Dataset object to visualise
	"""

	if featureName not in dataset.featureMetadata.columns:
		raise ValueError('%s is not a column in dataset.featureMetadata' % (featureName))

	if logx:
		logx = 'log'
	else:
		logx = None
	if logy:
		logy = 'log'
	else:
		logy = None

	featureMask = dataset.featureMask

	if xlim is not None:
		featureMask = (dataset.featureMetadata['Retention Time'].values > xlim[0]) & \
					  (dataset.featureMetadata['Retention Time'].values < xlim[1]) & \
					  featureMask
	if ylim is not None:
		featureMask = (dataset.featureMetadata['m/z'].values > ylim[0]) & \
					  (dataset.featureMetadata['m/z'].values < ylim[1]) & \
					  featureMask

	data = list()
	ionMap = go.Scatter(
		x = dataset.featureMetadata.loc[featureMask, 'Retention Time'],
		y = dataset.featureMetadata.loc[featureMask, 'm/z'],
		mode = 'markers',
		text = dataset.featureMetadata.loc[featureMask, featureName],
		hoverinfo = 'x, y, text',
		showlegend = False
		)

	data.append(ionMap)
	Xlabel = 'Retention Time'
	Ylabel = 'm/z'

	layout = {
		'xaxis' : dict(
			title = Xlabel,
			type = logx
			),
		'yaxis' : dict(
			title = Ylabel,
			type = logy
			),
		'title' : 'Ion map for: ' + dataset.name,
		'hovermode' : 'closest',
	}

	figure = go.Figure(data=data, layout=layout)

	return figure
