import plotly
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import numpy
import pandas
import copy
from ..objects._msDataset import MSDataset
from ._violinPlot import _violinPlotHelper
from ..enumerations import AssayRole, SampleType
from ..utilities.generic import createDestinationPath
from ..objects import Dataset
from matplotlib.colors import rgb2hex
import matplotlib.dates as mdates
from matplotlib.dates import MO, TU, WE, TH, FR, SA, SU
from matplotlib.dates import WeekdayLocator
from matplotlib.dates import DateFormatter
from matplotlib import gridspec
from matplotlib.patches import Rectangle
import os
import datetime

def plotIntensity(dataset, addViolin=True, addBatchShading=False,
			colourBy='SampleClass', colourType='categorical',
			colourDict=None, markerDict=None, abbrDict=None,
			logy=False, title='',
			withExclusions=True, savePath=None,
			figureFormat='png', dpi=72, figureSize=(11,7), opacity=.6):
	"""
	Visualise TIC for all or a subset of features coloured by either dilution value or detector voltage.
	With the option to shade by batch.

	.. note:: addViolin and colourByDetectorVoltage are mutually exclusive.

	:param MSDataset dataset: Dataset object
	:param bool addViolin: If ``True`` adds violin plots of TIC distribution pre and post correction split by sample type
	:param bool addBatchShading: If ``True`` shades plot according to sample batch
	:param str colourBy:
	:param str colourType:
	:param dict colourDict:
	:param dict markerDict:
	:param dict abbrDict:
	:param bool logy: If ``True`` plot y on a log scale
	:param str title: Title for the plot
	:param bool withExclusions: If ``False``, discard masked features from the sum
	:param savePath: If ``None`` plot interactively, otherwise save the figure to the path specified
	:type savePath: None or str
	:param str figureFormat: If saving the plot, use this format
	:param int dpi: Plot resolution
	:param figureSize: Dimensions of the figure
	:type figureSize: tuple(float, float)
	"""

	# Check inputs
	if not isinstance(dataset, Dataset):
		raise TypeError('dataset must be an instance of nPYc.Dataset')

	if colourBy not in dataset.sampleMetadata.columns:
		raise ValueError('colourBy must be a column in dataset.sampleMetadata')

	if not (('Acquired Time' in dataset.sampleMetadata.columns) or ('Run Order' in dataset.sampleMetadata.columns)):
		raise ValueError("'Acquired Time' or 'Run Order' must be columns in dataset.sampleMetadata")

	if not isinstance(colourType, str) & (colourType in {'categorical', 'continuous', 'continuousCentered'}):
		raise ValueError('colourType must be == ' + str({'categorical', 'continuous', 'continuousCentered'}))

	# Apply sample/feature masks if exclusions to be applied
	msData = copy.deepcopy(dataset)
	if withExclusions:
		msData.applyMasks()

	# List unique classes in msData.sampleMetadata[colourBy]
	uniq_classes = msData.sampleMetadata[colourBy].unique()
	uniq = [str(i) for i in uniq_classes]

	if colourType == 'categorical':

		# If colourDict check colour defined for every unique entry in class
		if colourDict is not None:
			if not all(k in colourDict.keys() for k in uniq):
				raise ValueError('If colourDict is specified every unique entry in ' + colourBy + ' must be a key in colourDict')
		# Otherwise create colour dict
		else:
			colourDict = {}
			colors = iter(plt.cm.rainbow(numpy.linspace(0, 1, len(uniq))))
			for u in uniq:
				colourDict[u] = rgb2hex(next(colors))

		# If markerDict check colour defined for every unique entry in class
		if markerDict is not None:
			if not all(k in markerDict.keys() for k in uniq):
				raise ValueError('If markerDict is specified every unique entry in ' + colourBy + ' must be a key in markerDict')
		else:
			markerDict = {}
			for u in uniq:
				markerDict[u] = 'o'

		# If abbrDict check abbr defined for every unique entry in class
		if abbrDict is not None:
			if not all(k in abbrDict.keys() for k in uniq):
				raise ValueError('If abbrDict is specified every unique entry in ' + colourBy + ' must be a key in abbrDict')
		else:
			abbrDict = {}
			for u in uniq:
				abbrDict[u] = u

	#sns.set_color_codes(palette='deep')
	fig = plt.figure(figsize=figureSize, dpi=dpi)
	gs = gridspec.GridSpec(1, 5)

	if addViolin:
		ax = plt.subplot(gs[0,:-1])
		ax2 = plt.subplot(gs[0,-1])
	else:
		ax = plt.subplot(gs[0,:-1])

	# Mask features with inf values
	tempFeatureMask = numpy.sum(numpy.isfinite(msData.intensityData), axis=0)
	tempFeatureMask = tempFeatureMask < msData.intensityData.shape[0]
	tempFeatureMask = (tempFeatureMask==False)

	# Use 'Acquired Time' if it exists:
	if ('Acquired Time' in msData.sampleMetadata.columns):

		# X axis limits for formatting
		minX = msData.sampleMetadata['Acquired Time'].loc[msData.sampleMetadata['Run Order'] == min(msData.sampleMetadata['Run Order'])].values
		maxX = msData.sampleMetadata['Acquired Time'].loc[msData.sampleMetadata['Run Order'] == max(msData.sampleMetadata['Run Order'])].values
		delta = maxX - minX
		days = delta.astype('timedelta64[D]')
		days = days / numpy.timedelta64(1, 'D')
		if days < 7:
			loc = WeekdayLocator(byweekday=(MO, TU, WE, TH, FR, SA, SU))
		else:
			loc = WeekdayLocator(byweekday=(MO, SA))
		formatter = DateFormatter('%d/%m/%y')

		# Ensure 'Acquired Time' is datetime.datetime, if it's already a datetime it will trigger an AttributeError
		try:
			acqTime = numpy.array([xtime.to_pydatetime() for xtime in msData.sampleMetadata['Acquired Time'].tolist()])
		except AttributeError:
			acqTime = numpy.array(msData.sampleMetadata['Acquired Time'].tolist())

	# Otherwise use 'Run Order'
	else:
		acqTime = msData.sampleMetadata['Run Order']

	tic = numpy.sum(msData.intensityData[:, tempFeatureMask == True], axis=1)

	# Colour by categorical class
	if colourType == 'categorical':
		palette = {}
		sampleMasks = []
		for u in uniq:
			sc = ax.scatter(acqTime[msData.sampleMetadata[colourBy] == u],
							tic[msData.sampleMetadata[colourBy] == u],
							marker=markerDict[u],
							s=30,
							c=colourDict[u],
							alpha=opacity,
							label=u)

			if addViolin:
				sampleMasks.append((abbrDict[u], msData.sampleMetadata[colourBy] == u))
				palette[abbrDict[u]] = colourDict[u]

		if addViolin:
			limits = ax.get_ylim()
			_violinPlotHelper(ax2, tic, sampleMasks, None, 'Sample Type',
							  palette=palette, ylimits=limits, logy=logy)

	# Colour by continuous class
	else:

		cmap = plt.cm.RdYlBu_r

		if colourType == 'continuous':
			mincol = numpy.nanmin(msData.sampleMetadata[colourBy])
			maxcol = numpy.nanmax(msData.sampleMetadata[colourBy])

		else:
			maxcol = numpy.max([numpy.abs(numpy.max(msData.sampleMetadata[colourBy])), numpy.abs(numpy.min(msData.sampleMetadata[colourBy]))])
			mincol = -maxcol

		sc = ax.scatter(acqTime,
						tic,
						c=msData.sampleMetadata[colourBy],
						cmap=cmap,
						vmin=mincol,
						vmax=maxcol,
						alpha=opacity)

	# Shade by automatically defined batches (if required)
	if addBatchShading:

		# Unique batches
		if 'Correction Batch' in msData.sampleMetadata.columns:
			batches = (numpy.unique(msData.sampleMetadata['Correction Batch'].values[
										~numpy.isnan(msData.sampleMetadata['Correction Batch'].values)])).astype(int)
		else:
			batches = (numpy.unique(
				msData.sampleMetadata['Batch'].values[
					~numpy.isnan(msData.sampleMetadata['Batch'].values)])).astype(int)

		# Define the colours (different for each batch) and get axis y-limits
		cmap = plt.get_cmap('gnuplot')
		colors = [cmap(i) for i in numpy.linspace(0, 1, len(batches)+1)]
		ymin, ymax = ax.get_ylim()
		colIX = 1

		# Add shading for each batch
		for i in batches:

			# Create rectangle x coordinates
			start = acqTime[msData.sampleMetadata['Run Order'] == min(msData.sampleMetadata[msData.sampleMetadata['Correction Batch'].values==i]['Run Order'])]
			end = acqTime[msData.sampleMetadata['Run Order'] == max(msData.sampleMetadata[msData.sampleMetadata['Correction Batch'].values==i]['Run Order'])]

			# Convert to matplotlib date representation
			if ('Acquired Time' in msData.sampleMetadata.columns):
				start = mdates.date2num(start)
				end = mdates.date2num(end)
			else:
				start = start.values[0]
				end = end.values[0]

			# Plot rectangle
			rect = Rectangle((start, ymin), end-start, abs(ymin)+abs(ymax), color=colors[colIX], alpha=opacity, zorder=0)#,label='Batch %d' % (i))
			ax.add_patch(rect)
			colIX = colIX + 1

	# Annotate figure
	ax.set_ylabel('Sum of Feature Intensities')
	ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=45)
	if ('Acquired Time' in msData.sampleMetadata.columns):
		ax.set_xlabel('Acquisition Date')
		ax.set_xlim(minX, maxX)
		ax.xaxis.set_major_locator(loc)
		ax.xaxis.set_major_formatter(formatter)
	else:
		ax.set_xlabel('Run Order')
	try:
		ax.set_ylim(ymin, ymax)
	except:
		pass
	if logy:
		ax.set_yscale('log', nonpositive='clip')
		ax.set_ylabel('Sum of Feature Intensities (log scale)')
	else:
		ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

	if colourType in {'continuous', 'continuousCentered'}:
		cbar = plt.colorbar(sc)
		cbar.set_label(colourBy)
		#leg = ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
	elif addViolin is False:
		ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

	fig.suptitle(title)

	# Save or output
	if savePath:
		try:
			plt.savefig(savePath, bbox_inches='tight', format=figureFormat, dpi=dpi)

		except UnboundLocalError:
			plt.savefig(savePath, bbox_inches='tight', format=figureFormat, dpi=dpi)
		plt.close()
	else:

		plt.show()


def plotIntensityInteractive(dataset,
							 x='Run Order',
							 y='Sum of Feature Intensities',
							 labelBy='Run Order',
							 colourBy='Correction Batch',
							 colourDict=None,
							 markerDict=None,
							 withExclusions=True,
							 destinationPath=None,
							 autoOpen=True,
							 opacity=.6):
	"""
	Interactively visualise sum of all feature intensities, or intensity for a given feature with plotly, provides tooltips to allow identification of samples.

	:param MSDataset dataset: Dataset object
	:param str x: X-axis of plot, either ``Run Order`` or ``Acquired Time``
	:param str y: Y-axis of plot, either ``Sum of Feature Intensities`` for sum of all features, or a specific feature name
	:param str labelBy: dataset.sampleMetadata column entry to display in tooltips
	:param str colourBy: dataset.sampleMetadata column entry to colour data points by
	:param dict colourDict:
	:param dict markerDict:
	:param bool withExclusions: If ``True``, only report on features and samples not masked by the sample and feature masks
	:param str destinationPath: file path to save html version of plot
	:param bool autoOpen: If ``True``, opens html version of plot
	:return: Data object to use with plotly
	"""

	# Apply sample/feature masks if exclusions to be applied
	msData = copy.deepcopy(dataset)
	if withExclusions:
		msData.applyMasks()

	# Checks
	if not (y in msData.featureMetadata['Feature Name'].values) | (y == 'Sum of Feature Intensities'):
		raise ValueError("y must be either a value in dataset.featureMetadata['Feature Name'] or 'Sum of Feature Intensities'")

	if not ((x in msData.sampleMetadata.columns) & (x in {'Run Order', 'Acquired Time'})):
		raise ValueError("x must be \'Run Order\' or \'Acquired Time\', and must be present as a column in dataset.sampleMetadata")

	if labelBy not in dataset.sampleMetadata.columns:
		raise ValueError('labelBy must be a column in dataset.sampleMetadata')

	if colourBy not in dataset.sampleMetadata.columns:
		raise ValueError('colourBy must be a column in dataset.sampleMetadata')

	# Create destinationPath for saving outputs
	if destinationPath:
		createDestinationPath(destinationPath)

	# Data preparation
	ns = len(msData.sampleMask)
	classes = msData.sampleMetadata[colourBy]
	hovertext = msData.sampleMetadata['Sample File Name'].str.cat(classes.astype(str), sep='; ' + colourBy + ': ')
	plotnans = classes.isnull().values
	data = []

	# Extract y values
	if y == 'Sum of Feature Intensities':
		tempFeatureMask = numpy.sum(numpy.isfinite(msData.intensityData), axis=0)
		tempFeatureMask = tempFeatureMask < msData.intensityData.shape[0]
		values = numpy.sum(msData.intensityData[:, tempFeatureMask == False], axis=1)

	else:
		feature = msData.featureMetadata.loc[msData.featureMetadata['Feature Name'] == y].index[0]
		values = msData.intensityData[:, feature]

	# Ensure all values in colourBy column have the same type

	# list of all types in column; and set of unique types
	mylist = list(type(classes[i]) for i in range(ns))
	myset = set(mylist)

	# if time pass
	if any(my == pandas.Timestamp for my in myset) or any(my == datetime.datetime for my in myset):
		pass

	# else if mixed type convert to string
	elif len(myset) > 1:
		classes = classes.astype(str)

	# If colourBy=='SampleClass' - NPC derived classes, then use default colours and markers
	if colourBy=='SampleClass':
		colourDict = msData.Attributes['sampleTypeColours']
		markerDict = msData.Attributes['sampleTypeMarkers']

	# Plot NaN values in gray
	if sum(plotnans != 0):
		NaNplot = go.Scatter(
			x=msData.sampleMetadata.loc[plotnans == True, x],
			y=values[plotnans == True],
			mode='markers',
			marker=dict(
				color='rgb(180, 180, 180)',
				symbol='circle',
				),
			text=hovertext[plotnans == True],
			name='NA',
			hoverinfo='text',
			showlegend=True, opacity=opacity
			)
		data.append(NaNplot)

	# Plot numeric values with a colorbar
	if classes.dtype in (int, float):
		CLASSplot = go.Scatter(
			x=msData.sampleMetadata.loc[plotnans == False, x],
			y=values[plotnans == False],
			mode='markers',
			marker=dict(
				colorscale='Portland',
				color=classes[plotnans == False],
				symbol='circle',
				showscale=True
				),
			text=hovertext[plotnans == False],
			hoverinfo='text',
			showlegend=False,
			opacity=opacity
			)
		data.append(CLASSplot)

	# Plot categorical values by unique groups
	else:
		uniq = numpy.unique(classes[plotnans == False])
		if colourDict is None:

			colourDict = {}
			colors = iter(plt.cm.rainbow(numpy.linspace(0, 1, len(uniq))))
			for u in uniq:
				colourDict[u] = rgb2hex(next(colors))

		if markerDict is None:

			markerDict = {}
			for u in uniq:
				markerDict[u] = 'diamond-dot'

		for i in uniq:
			CLASSplot = go.Scatter(
				x=msData.sampleMetadata.loc[classes == i, x],
				y=values[classes == i],
				mode='markers',
				marker=dict(
					color=colourDict[i],
					symbol=markerDict[i]
					),
				text=hovertext[classes == i],
				name=str(i),
				hoverinfo='text',
				showlegend=True,
				opacity=opacity
				)
			data.append(CLASSplot)

	# Overlay SR and LTR if columns present
	if (colourBy != 'SampleClass') & ('SampleType' in msData.sampleMetadata.columns) & ('AssayRole' in msData.sampleMetadata.columns):
		SRmask = ((msData.sampleMetadata['SampleType'].values == SampleType.StudyPool) &
				  (msData.sampleMetadata['AssayRole'].values == AssayRole.PrecisionReference)) #SPmask
		LTRmask = ((msData.sampleMetadata['SampleType'].values == SampleType.ExternalReference) &
				   (msData.sampleMetadata['AssayRole'].values == AssayRole.PrecisionReference)) # ERmask

		SRplot = go.Scatter(
			x=msData.sampleMetadata.loc[SRmask, x],
			y=values[SRmask],
			mode='markers',
			marker=dict(
				color='darkgreen',
				symbol='1',
			),
			text=hovertext[SRmask],
			name='Study Reference',
			hoverinfo='text',
			showlegend=True
		)
		data.append(SRplot)

		LTRplot = go.Scatter(
			x=msData.sampleMetadata.loc[LTRmask, x],
			y=values[LTRmask],
			mode='markers',
			marker=dict(
				color='darkorange',
				symbol='2',

			),
			text=hovertext[LTRmask],
			name='Long-Term Reference',
			hoverinfo='text',
			showlegend=True
		)
		data.append(LTRplot)

	# Add annotation
	layout = {
		'xaxis': dict(
			title=x,
		),
		'yaxis': dict(
			title=y
		),
		'title': y + ' coloured by ' + colourBy,
		'legend': dict(
			yanchor='middle',
			xanchor='right'
			),
		'hovermode': 'closest'
	}

	figure = go.Figure(data=data, layout=layout)

	if destinationPath:
		saveTemp = msData.name + '_' + y.replace('/', '') + '_colourBy_' + colourBy + '.html'
		plotly.offline.plot(figure, filename=os.path.join(destinationPath, saveTemp), auto_open=autoOpen)

	return figure
