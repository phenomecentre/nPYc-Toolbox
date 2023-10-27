import plotly
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import seaborn as sns
import numpy
import pandas
import copy
from ..objects._msDataset import MSDataset
from ._violinPlot import _violinPlotHelper
from ..enumerations import AssayRole, SampleType
from ..utilities.generic import createDestinationPath
import matplotlib.dates as mdates
from matplotlib.dates import MO, TU, WE, TH, FR, SA, SU
from matplotlib.dates import WeekdayLocator
from matplotlib.dates import DateFormatter
from matplotlib import gridspec
from matplotlib.patches import Rectangle
import os
import datetime

def plotTIC(msData, addViolin=True, addBatchShading=False, addLineAtGaps=False, colourByDetectorVoltage=False, logy=False, title='', withExclusions=True, savePath=None, figureFormat='png', dpi=72, figureSize=(11,7)):
	"""
	Visualise TIC for all or a subset of features coloured by either dilution value or detector voltage. With the option to shade by batch.

	.. note:: addViolin and colourByDetectorVoltage are mutually exclusive.

	:param MSDataset msData: Dataset object
	:param bool addViolin: If ``True`` adds violin plots of TIC distribution pre and post correction split by sample type
	:param bool addBatchShading: If ``True`` shades plot according to sample batch
	:param bool addLineAtGaps: If ``True`` adds line where acquisition time is greater than double the norm
	:param bool colourByDetectorVoltage: If ``True`` colours points by detector voltage, else colours by dilution
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
	if (addViolin) & (colourByDetectorVoltage):
		raise ValueError('addViolin and colourByDetectorVoltage cannot both be True')

	sns.set_color_codes(palette='deep')
	fig = plt.figure(figsize=figureSize, dpi=dpi)
	gs = gridspec.GridSpec(1, 5)

	if addViolin:
		ax = plt.subplot(gs[0,:-1])
		ax2 = plt.subplot(gs[0,-1])
	else:
		ax = plt.subplot(gs[0,:-1])

	# Load toolbox wide color scheme
	if 'sampleTypeColours' in msData.Attributes.keys():
		sTypeColourDict = copy.deepcopy(msData.Attributes['sampleTypeColours'])
		for stype in SampleType:
			if stype.name in sTypeColourDict.keys():
				sTypeColourDict[stype] = sTypeColourDict.pop(stype.name)
	else:
		sTypeColourDict = {SampleType.StudySample: 'b', SampleType.StudyPool: 'g', SampleType.ExternalReference: 'r',
							SampleType.MethodReference: 'm', SampleType.ProceduralBlank: 'c', 'Other': 'grey'}

	# Mask features with inf values
	tempFeatureMask = numpy.sum(numpy.isfinite(msData.intensityData), axis=0)
	tempFeatureMask = tempFeatureMask < msData.intensityData.shape[0]
	tempFeatureMask = (tempFeatureMask==False)

	if withExclusions:
		tempFeatureMask = numpy.logical_and(tempFeatureMask, msData.featureMask)
		tempSamplesMask = msData.sampleMask

	else:
		tempSamplesMask = numpy.ones(shape=msData.sampleMask.shape, dtype=bool)

	# Define sample types
	SSmask = ((msData.sampleMetadata['SampleType'].values == SampleType.StudySample) & (msData.sampleMetadata['AssayRole'].values == AssayRole.Assay)) & tempSamplesMask
	SPmask = ((msData.sampleMetadata['SampleType'].values == SampleType.StudyPool) & (msData.sampleMetadata['AssayRole'].values == AssayRole.PrecisionReference)) & tempSamplesMask
	ERmask = ((msData.sampleMetadata['SampleType'].values == SampleType.ExternalReference) & (msData.sampleMetadata['AssayRole'].values == AssayRole.PrecisionReference)) & tempSamplesMask
	LRmask = ((msData.sampleMetadata['SampleType'].values == SampleType.StudyPool) & (msData.sampleMetadata['AssayRole'].values == AssayRole.LinearityReference)) & tempSamplesMask

	# Use 'Acquired Time' if it exists:
	if ('Acquired Time' in msData.sampleMetadata.columns):

		# X axis limits for formatting
		minX = msData.sampleMetadata['Acquired Time'].loc[msData.sampleMetadata['Run Order'] == min(msData.sampleMetadata['Run Order'][SSmask | SPmask | ERmask | LRmask])].values
		maxX = msData.sampleMetadata['Acquired Time'].loc[msData.sampleMetadata['Run Order'] == max(msData.sampleMetadata['Run Order'][SSmask | SPmask | ERmask | LRmask])].values
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

	# If colouring by detector voltage
	if colourByDetectorVoltage:

		# Generate sample change in detector voltage
		detectorDiff = msData.sampleMetadata[['Detector', 'Run Order']].sort_values(by='Run Order')['Detector'].diff().sort_index()
		detectorDiff[0] = 0  			# no detector diff for first sample
		cMax = max(abs(detectorDiff))  	# colorbar symmetrical around 0
		cMin = -cMax

		# Plot TIC for different sample types, colored by change in detector voltage
		if cMax != 0:
			if sum(SSmask != 0):
				sc = ax.scatter(acqTime[SSmask], tic[SSmask], marker='o', c=detectorDiff[SSmask], cmap=plt.cm.get_cmap('bwr'), vmin=cMin, vmax=cMax, label='Study Sample', edgecolors='grey')
			if sum(SPmask != 0):
				sc = ax.scatter(acqTime[SPmask], tic[SPmask], marker='v', s=30, linewidth=0.9, c=detectorDiff[SPmask], cmap=plt.cm.get_cmap('bwr'), vmin=cMin, vmax=cMax, label='Study Reference', edgecolors='grey')
			if sum(ERmask != 0):
				sc = ax.scatter(acqTime[ERmask], tic[ERmask], marker='^', s=30, linewidth=0.9, c=detectorDiff[ERmask], cmap=plt.cm.get_cmap('bwr'), vmin=cMin, vmax=cMax, label='Long-Term Reference', edgecolors='grey')
			if sum(LRmask != 0):
				sc = ax.scatter(acqTime[LRmask], tic[LRmask], marker='s', c=detectorDiff[LRmask], cmap=plt.cm.get_cmap('bwr'), vmin=cMin, vmax=cMax, label='Serial Dilution', edgecolors='grey')

		# For the specific case where there is no detector voltage and colorscale collapses
		else:
			if sum(SSmask != 0):
				sc = ax.scatter(acqTime[SSmask], tic[SSmask], marker='o', c='w', cmap=plt.cm.get_cmap('bwr'), vmin=cMin, vmax=cMax, label='Study Sample', edgecolors='grey')
			if sum(SPmask != 0):
				sc = ax.scatter(acqTime[SPmask], tic[SPmask], marker='v', s=30, linewidth=0.9, c='w', cmap=plt.cm.get_cmap('bwr'), vmin=cMin, vmax=cMax, label='Study Reference', edgecolors='grey')
			if sum(ERmask != 0):
				sc = ax.scatter(acqTime[ERmask], tic[ERmask], marker='^', s=30, linewidth=0.9, c='w', cmap=plt.cm.get_cmap('bwr'), vmin=cMin, vmax=cMax, label='Long-Term Reference', edgecolors='grey')
			if sum(LRmask != 0):
				sc = ax.scatter(acqTime[LRmask], tic[LRmask], marker='s', c='w', cmap=plt.cm.get_cmap('bwr'), vmin=cMin, vmax=cMax, label='Serial Dilution', edgecolors='grey')

	# Colour by sample type
	else:

		# Plot TIC for different sample types
		if sum(SSmask != 0):
			sc = ax.scatter(acqTime[SSmask], tic[SSmask], marker='o', s=30, c=sTypeColourDict[SampleType.StudySample], label='Study Sample')
		if sum(SPmask != 0):
			sc = ax.scatter(acqTime[SPmask], tic[SPmask], marker='v', s=30, c=sTypeColourDict[SampleType.StudyPool], label='Study Reference')
		if sum(ERmask != 0):
			sc = ax.scatter(acqTime[ERmask], tic[ERmask], marker='^', s=30, c=sTypeColourDict[SampleType.ExternalReference], label='Long-Term Reference')
		if sum(LRmask != 0):
			sc = ax.scatter(acqTime[LRmask], tic[LRmask], marker='s', s=30, c=sTypeColourDict[SampleType.MethodReference], label='Serial Dilution')


	# Shade by automatically defined batches (if required)
	if addBatchShading:

		sampleMask = SSmask | SPmask | ERmask

		# Unique batches
		if 'Correction Batch' in msData.sampleMetadata.columns:
			batches = (numpy.unique(msData.sampleMetadata.loc[sampleMask, 'Correction Batch'].values[~numpy.isnan(msData.sampleMetadata.loc[sampleMask, 'Correction Batch'].values)])).astype(int)
		else:
			batches = (numpy.unique(msData.sampleMetadata.loc[sampleMask, 'Batch'].values[~numpy.isnan(msData.sampleMetadata.loc[sampleMask, 'Batch'].values)])).astype(int)

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
			rect = Rectangle((start, ymin), end-start, abs(ymin)+abs(ymax), color=colors[colIX], alpha=0.4, label='Batch %d' % (i), zorder=0)
			ax.add_patch(rect)
			colIX = colIX + 1
	else:
		# Still might need the batch information even if not using it for batch shading
		sampleMask = SSmask | SPmask | ERmask
		if 'Correction Batch' in msData.sampleMetadata.columns:
			batches = (numpy.unique(msData.sampleMetadata.loc[sampleMask, 'Correction Batch'].values[~numpy.isnan(msData.sampleMetadata.loc[sampleMask, 'Correction Batch'].values)])).astype(int)
		else:
			batches = (numpy.unique(msData.sampleMetadata.loc[sampleMask, 'Batch'].values[~numpy.isnan(msData.sampleMetadata.loc[sampleMask, 'Batch'].values)])).astype(int)

	# Add violin plot of data distribution (if required)
	if addViolin:

		sampleMasks = list()
		palette = {}
		if sum(SSmask)>0:
			sampleMasks.append(('SS', SSmask))
			palette['SS'] = sTypeColourDict[SampleType.StudySample]
		if sum(SPmask)>0:
			sampleMasks.append(('SR', SPmask))
			palette['SR'] = sTypeColourDict[SampleType.StudyPool]
		if sum(ERmask)>0:
			sampleMasks.append(('LTR', ERmask))
			palette['LTR'] = sTypeColourDict[SampleType.ExternalReference]
		if sum(LRmask)>0:
			sampleMasks.append(('SRD', LRmask))
			palette['SRD'] = sTypeColourDict[SampleType.MethodReference]

		limits = ax.get_ylim()

		_violinPlotHelper(ax2, tic, sampleMasks, None, 'Sample Type', palette=palette, ylimits=limits, logy=logy)

		#sns.despine(trim=True, ax=ax2)

	# Annotate figure
	ax.set_ylabel('Sum of all Feature Intensities')
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
		ax.set_ylabel('TIC (log scale)')
	else:
		ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

	if colourByDetectorVoltage and cMax != 0:
		cbaxes = fig.add_axes([0.81, 0.15, 0.03, 0.64 - (len(batches) * 0.04)]) # shorter color bar as more batches are present
		cbar = plt.colorbar(sc, cax=cbaxes)
		cbar.set_label('Change in Detector Voltage')
		leg = ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
	elif addViolin==False:
		ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

	fig.suptitle(title)

	# Save or output
	if savePath:
		try:
			plt.savefig(savePath, bbox_extra_artists=(leg, ), bbox_inches='tight', format=figureFormat, dpi=dpi)
		except UnboundLocalError:
			plt.savefig(savePath, bbox_inches='tight', format=figureFormat, dpi=dpi)
		plt.close()
	else:

		plt.show()


def plotTICinteractive(dataset, x='TIC', y='Run Order', labelBy='Run Order', colourBy='Correction Batch', withExclusions=True, destinationPath=None, autoOpen=True):
	"""
	Interactively visualise TIC or intensity for a given feature with plotly, provides tooltips to allow identification of samples.

	:param MSDataset dataset: Dataset object
	:param str x: X-axis of plot, either ``Run Order`` or ``Acquired Time``
	:param str y: Y-axis of plot, either ``TIC`` for sum of all features, or a specific feature name
	:param str labelBy: dataset.sampleMetadata column entry to display in tooltips
	:param str colourBy: dataset.sampleMetadata column entry to colour data points by
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
	if not (y in msData.featureMetadata['Feature Name'].values) | (y == 'TIC'):
		raise ValueError("y must be either a value in dataset.featureMetadata['Feature Name'] or 'TIC'")

	if not ((x in msData.sampleMetadata.columns) & (x in {'Run Order', 'Acquired Time'})):
		raise ValueError("x must be \'Run Order\' or \'Acquired Time\', and must be present as a column in dataset.sampleMetadata")

	if labelBy not in dataset.sampleMetadata.columns:
		raise ValueError('labelBy must be a column in dataset.sampleMetadata')

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
	if y == 'TIC':
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
			showlegend=True
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
			showlegend=False
			)
		data.append(CLASSplot)

	# Plot categorical values by unique groups
	else:
		uniq = numpy.unique(classes[plotnans == False])
		for i in uniq:
			CLASSplot = go.Scatter(
				x=msData.sampleMetadata.loc[classes == i, x],
				y=values[classes == i],
				mode='markers',
				marker=dict(
					colorscale='Portland',
					symbol='circle',
					),
				text=hovertext[classes == i],
				name=i,
				hoverinfo='text',
				showlegend=True
				)
			data.append(CLASSplot)

	# Overlay SR and LTR if columns present
	if ('SampleType' in msData.sampleMetadata.columns) & ('AssayRole' in msData.sampleMetadata.columns):
		SRmask = ((msData.sampleMetadata['SampleType'].values == SampleType.StudyPool) & (msData.sampleMetadata['AssayRole'].values == AssayRole.PrecisionReference))
		LTRmask = ((msData.sampleMetadata['SampleType'].values == SampleType.ExternalReference) & (msData.sampleMetadata['AssayRole'].values == AssayRole.PrecisionReference))

		SRplot = go.Scatter(
			x=msData.sampleMetadata.loc[SRmask, x],
			y=values[SRmask],
			mode='markers',
			marker=dict(
				color='rgb(63, 158, 108)',
				symbol='x',
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
				color='rgb(198, 83, 83)',
				symbol='x'
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
