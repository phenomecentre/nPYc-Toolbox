import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy
import pandas
from ..objects._msDataset import MSDataset
from ..utilities import generateLRmask
from ..utilities._internal import _vcorrcoef 
from ._violinPlot import _violinPlotHelper
from ..enumerations import AssayRole, SampleType
import matplotlib.dates as mdates
from matplotlib.dates import MO, TU, WE, TH, FR, SA, SU
from matplotlib.dates import WeekdayLocator
from matplotlib.dates import DateFormatter
from matplotlib import gridspec
from matplotlib.patches import Rectangle
from cycler import cycler
import os
import re
import copy


def histogram(values, inclusionVector=None, quantiles=None, title='', xlabel='', histBins=100, color=None, logy=False, logx=False, xlim=None, savePath=None, figureFormat='png', dpi=72, figureSize=(11,7)):
	"""
	histogram(values, inclusionVector=None, quantiles=None, histBins=100, color=None, logy=False, logx=False, **kwargs)

	Plot a histogram of values, optionally segmented according to observed quantiles.

	Quantiles can be calculated on a second *inclusionVector* when specified.

	:param values: Values to plot
	:type values: numpy.array or dict
	:param inclusionVector: Optional second vector with same size as values, used to select quantiles for plotting.
	:type inclusionVector: None or numpy.array
	:param quantiles: List of quantile bounds to segment the histogram by
	:type quantiles: None or List
	:param str title: Title for the plot
	:param str xlabel: Label for the X-axis
	:param int histBins: Number of bins to break the histgram into
	:param color: List of specific colours to use for plotting
	:type color: None or List
	:param bool logy: If ``True`` plot y on a log scale
	:param bool logx: If ``True`` plot x on a log scale
	:param xlim: Specify upper and lower bounds of the X axis
	:type xlim: tuple of int
	"""

	fig = plt.figure(figsize=figureSize, dpi=dpi)
	ax = plt.subplot(1,1,1)

	# Set the colorpalette
	if color is not None:
		sns.set_color_codes(palette='deep')
		ax.set_prop_cycle(cycler('color', color))
	elif quantiles is not None:
		flatui = ["#16a085", "#3498db", "#707b7c"]#, "#d2b4de", "#aeb6bf"]
		ax.set_prop_cycle(cycler('color', flatui))

	# Set masks etc if required (not currently possible when values is a dictionary)
	if not isinstance(values, dict):

		# If we don't have a matching pair of vectors use values for both.
		if not numpy.size(inclusionVector) == numpy.size(values):
			inclusionVector = values

		# If we are limiting axes, delete elements outof bounds
		if not xlim is None:
			mask = numpy.logical_and(values >= xlim[0], values <= xlim[1])
			inclusionVector = inclusionVector[mask]
			values = values[mask]

		# Remove non-finite elements
		maskFinite = numpy.logical_and(numpy.isfinite(inclusionVector), numpy.isfinite(values))
		inclusionVector = inclusionVector[maskFinite]
		values = values[maskFinite]

		minVal = numpy.nanmin(values)
		maxVal = numpy.nanmax(values)

	# Calculate ranges for dict entries early.
	else:

		if not inclusionVector is None:
			raise ValueError("Cannot provide an inclusion vector when plotting groups.")

		# Set min and max values
		if not xlim is None:
			minVal = xlim[0]
			maxVal = xlim[1]
		else:
			minVal = numpy.nan
			maxVal = numpy.nan
			for key in values:
				if (numpy.isnan(minVal)) | (minVal > numpy.nanmin(values[key])):
					minVal = numpy.nanmin(values[key])
				if (numpy.isnan(maxVal)) | (maxVal < numpy.nanmax(values[key])):
					maxVal = numpy.nanmax(values[key])

		label = values.keys()
		
	# If log scale for x
	if logx == True:
		
		if minVal == 0:
			minVal = numpy.finfo(numpy.float64).epsneg

		if minVal < 0:
			logx = False
			nbins = histBins
			xscale = 'linear'
		else:
			nbins = 10 ** numpy.linspace(numpy.log10(minVal), numpy.log10(maxVal), histBins)
			xscale = 'log'
	else:
		nbins = histBins
		xscale = 'linear'


	# If we are plotting multiple histograms on the same axis
	if isinstance(values, dict):

		for key in values:
			localValues = values[key]

			# If we are limiting axes, delete elements outof bounds
			if not xlim is None:
				mask = numpy.logical_and(localValues >= xlim[0], localValues <= xlim[1])
				localValues = localValues[mask]		
				
			# If we are plotting on a log scale, convert any 0 values to numpy.finfo(numpy.float64).epsneg
			if logx == True:
				localValues[localValues == 0] = numpy.finfo(numpy.float64).epsneg
				
			ax.hist(localValues,
				alpha=.4,
				range=(minVal, maxVal),
				label=key,
				bins=nbins)

	# If we are segmenting by quantiles
	elif quantiles:

		# Find bounds in inclusion vector
		quantiles = numpy.percentile(inclusionVector, quantiles)

		label = "Below {0:,.2f}".format(quantiles[0])
		mask = inclusionVector <= quantiles[0]
		if sum(mask) <= 1:
			plt.plot([],
					 label=label)
		else:
			ax.hist(values[mask],
					 alpha=.4,
					 label=label,
					 bins=nbins)

		for i in range(0, len(quantiles)-1):
			label = "Between {0:,.2f} and {1:,.2f}".format(quantiles[i], quantiles[i+1])
			mask = (inclusionVector > quantiles[i]) & (inclusionVector <= quantiles[i+1])
			if sum(mask) <= 1:
				plt.plot([],
						 label=label)
			else:
				ax.hist(values[mask], 
						 alpha=.4,
						 label=label,
						 bins=nbins)

		label = "Above {0:,.2f}".format(quantiles[-1])
		mask = inclusionVector > quantiles[-1]
		if sum(mask) <= 1:
			plt.plot([],
					label=label)
		else:
			ax.hist(values[mask],
					 alpha=.4,
					 label=label,
					 bins=nbins)
	else:
		if len(values) <= 1:
			plt.plot([])
		else:
			ax.hist(values,
					label='',
					bins=nbins)

	ax.set_ylabel('Count')
	if logy:
		ax.set_yscale('log', nonpositive='clip')
	if not xlim is None:
		ax.set_xlim(xlim)
	ax.set_xlabel(xlabel)	
	if 'label' in locals():
		ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
	ax.set_xscale(xscale)
	fig.suptitle(title)

	if savePath:
		plt.savefig(savePath, bbox_inches='tight', format=figureFormat, dpi=dpi)
		plt.close()
	else:
		plt.show()


def plotTICinteractive(dataset, plottype='Sample Type', labelby='Run Order', withExclusions=True):
	"""
	Interactively visualise TIC (coloured by batch and sample type) with plotly, provides tooltips to allow identification of samples.

	Plots may be of two types:
	* **'Sample Type'** to plot by sample type and coloured by batch
	* **'Linearity Reference'** to plot LR samples coloured by dilution

	:param MSDataset msData: Dataset object
	:param str plottype: Select plot type, may be either ``Sample Type`` or ``Linearity Reference``
	:return: Data object to use with plotly
	"""
	import plotly.graph_objs as go
	from ..utilities import generateLRmask

	if not isinstance(plottype, str) & (plottype in {'Sample Type', 'Serial Dilution'}):
		raise ValueError('plottype must be == \'Sample Type\', \'Serial Dilution\'')

	# Apply sample/feature masks if exclusions to be applied
	msData = copy.deepcopy(dataset)
	if withExclusions:
		msData.applyMasks()

	# Generate TIC
	tempFeatureMask = numpy.sum(numpy.isfinite(msData.intensityData), axis=0)
	tempFeatureMask = tempFeatureMask < msData.intensityData.shape[0]
	tic = numpy.sum(msData.intensityData[:,tempFeatureMask==False], axis=1)

	# Plot by 'Run Order' if 'Acquired Time' not available
	if ('Acquired Time' in msData.sampleMetadata.columns):
		plotby='Acquired Time'
	elif ('Run Order' in msData.sampleMetadata.columns):
		plotby = 'Run Order'
	else:
		print('Acquired Time/Run Order data (columns in dataset.sampleMetadata) not available to plot')
		return
		
	if plottype=='Sample Type': # Plot TIC for SR samples coloured by batch
	
		SSmask = ((msData.sampleMetadata['SampleType'].values == SampleType.StudySample) & (msData.sampleMetadata['AssayRole'].values == AssayRole.Assay))
		SPmask = ((msData.sampleMetadata['SampleType'].values == SampleType.StudyPool) & (msData.sampleMetadata['AssayRole'].values == AssayRole.PrecisionReference))
		ERmask = ((msData.sampleMetadata['SampleType'].values == SampleType.ExternalReference) & (msData.sampleMetadata['AssayRole'].values == AssayRole.PrecisionReference))
	
		SSplot = go.Scatter(
			x = msData.sampleMetadata[plotby][SSmask],
			y = tic[SSmask],
			mode = 'markers',
			marker = dict(
				colorscale = 'Portland',
				color = msData.sampleMetadata['Correction Batch'][SSmask],
				symbol = 'circle'
				),
				name = 'Study Sample',
				text = msData.sampleMetadata[labelby][SSmask]
				)

		SRplot = go.Scatter(
			x = msData.sampleMetadata[plotby][SPmask],
			y = tic[SPmask],
			mode = 'markers',
			marker = dict(
				color = 'rgb(63, 158, 108)',
				symbol = 'cross'
				),
			name = 'Study Reference',
			text = msData.sampleMetadata[labelby][SPmask]
			)
			
		LTRplot = go.Scatter(
			x = msData.sampleMetadata[plotby][ERmask],
			y = tic[ERmask],
			mode = 'markers',
			marker = dict(
				color = 'rgb(198, 83, 83)',
				symbol = 'cross'
				),
			name = 'Long-Term Reference',
			text = msData.sampleMetadata[labelby][ERmask]
			)
		
		data = [SSplot, SRplot, LTRplot]
		Xlabel = plotby
		title = 'TIC by Sample Type Coloured by Batch'
	
	if plottype=='Serial Dilution': # Plot TIC for LR samples coloured by dilution

		LRmask = ((msData.sampleMetadata['SampleType'].values == SampleType.StudyPool) & (msData.sampleMetadata['AssayRole'].values == AssayRole.LinearityReference))
		
		if hasattr(msData, 'corrExclusions'):
			
			if msData.corrExclusions is not None:
				SUBSETSmask = generateLRmask(msData)
			
				for element in msData.corrExclusions:
					if element in SUBSETSmask:
						LRmask[SUBSETSmask[element]==True] = False
				
		tic = tic[LRmask]
		runIX = numpy.argsort(msData.sampleMetadata['Run Order'][LRmask].values)
		runIX = numpy.argsort(runIX)
		labels = msData.sampleMetadata['Sample File Name'][LRmask].values
		
		LRplot = go.Scatter(
			x = runIX,
			y = tic,
			mode = 'markers',
			marker = dict(
				colorscale = 'Portland',
				color = msData.sampleMetadata['Dilution'][LRmask],
				symbol = 'circle'
				),
			text = labels
			)
		
		data = [LRplot]
		Xlabel = 'Condensed Run Order'
		title = 'TIC of Dilution Series Samples Coloured by Dilution'
		
		# Add annotation
	layout = {
		'xaxis' : dict(
			title = Xlabel,
			),
		'yaxis' : dict(
			title = 'TIC'
			),
		'title' : title,	
		'hovermode' : 'closest',
	}


	fig = {
		'data': data,
		'layout': layout,
	}


	return fig


def plotLRTIC(msData, sampleMask=None, colourByDetectorVoltage=False, title='', label=False, savePath=None, figureFormat='png', dpi=72, figureSize=(11,7)):
	"""
	Visualise TIC for linearity reference (LR) samples (either all or a subset) coloured by either dilution value or detector voltage.

	:param MSDataset msData: Dataset object
	:param sampleMask: Defines subset of samples to plot, if ``None`` use *msData's* built-in sampleMask
	:type sampleMask: None or array of bool
	:param bool colourByDetectorVoltage: If ``True`` colours points by detector voltage, else colours by dilution
	:param str title: Title for the plot
	:param bool label: If ``True``, labels points with run order values
	:param savePath: If ``None``, plot interactively, otherwise attempt to save at this path.
	:type savePath: None or str
	:param str format: Format to save figure
	:param int dpi: Resolution to draw at
	:param tuple figureSize: Specify size of figure
	:type figureSize: tuple(float, float)
	"""

	if sampleMask is None:
		sampleMask = numpy.ones(msData.sampleMask.shape).astype(bool)
	
	plt.figure(figsize=figureSize, dpi=dpi)
	ax = plt.subplot(1,1,1)
	
	# Plot TIC for LR samples coloured by sample dilution
	tic = numpy.sum(msData.intensityData, axis=1)
	LRmask = (msData.sampleMetadata['SampleType'].values == SampleType.StudyPool) & (msData.sampleMetadata['AssayRole'].values == AssayRole.LinearityReference) & (sampleMask)
	tic = tic[LRmask]
	runIX = numpy.argsort(msData.sampleMetadata['Run Order'][LRmask].values)
	runIX = numpy.argsort(runIX)
	
	if colourByDetectorVoltage:
		detectorDiff = msData.sampleMetadata[['Detector', 'Run Order']].sort_values(by='Run Order')['Detector'].diff().sort_index()
		detectorDiff[0] = 0  					# no detector diff for first sample
		cMax = max(abs(detectorDiff[LRmask]))  # colorbar symmetrical around 0
		cMin = -cMax
		sc = ax.scatter(runIX, tic,
			c = detectorDiff[LRmask],
			cmap = plt.cm.get_cmap('bwr'),
			vmin=cMin,
			vmax=cMax,
			edgecolors='grey')
	else:
		ax.scatter(runIX, tic,
			c = msData.sampleMetadata['Dilution'][LRmask],
			cmap = plt.cm.jet,
			edgecolors='grey')
		
	# Add sample labels
	if label==True:
		labels = msData.sampleMetadata['Run Order'][LRmask].values
		labels = labels[runIX]
		for i, txt in enumerate(labels):
			ax.annotate(txt, (runIX[i], tic[i]))
	 
	# Add a line where samples are not adjacent
	sampletime = [x - runIX[i - 1] for i, x in enumerate(runIX)]
	sampletime[0] = 1
	sampleBreaks = [i for i, v in enumerate(sampletime) if v > 1]
	ymin, ymax = ax.get_ylim()
	
	for sample in sampleBreaks:
		plt.plot([sample-0.5, sample-0.5], [ymin, ymax], color='k', linestyle='-', linewidth=0.5)

	ax.set_xlabel('Run Order')
	ax.set_ylabel('TIC')
	ax.set_title(title)
	ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
	if colourByDetectorVoltage:
		cbar = plt.colorbar(sc)
		cbar.set_label('Change in Detector Voltage')

	if savePath:
		plt.savefig(savePath, bbox_inches='tight', format=figureFormat, dpi=dpi)
		plt.close()
	else:
		plt.show()


def plotCorrelationToLRbyFeature(msData, featureMask=None, title='', maxNo=5, savePath=None, figureFormat='png', dpi=72, figureSize=(11, 7)):
	"""
	Summary plots of correlation to dilution for a subset of features, separated by sample batch. Each figure includes a scatter plot of feature intensity vs dilution, TIC of LR and surrounding SP samples, and a heatmap of correlation to dilution for each LR batch subset, overall, and mean.

	:param MSDataset msData: Dataset object
	:param featureMask: Limits plotting to a subset of features, if ``None`` use *msData's* built-in sampleMask
	:type featureMask: None or array of bool
	:param str title: Title for the plot
	:param int maxNo: Optional number of features to plot (default=10, i.e., 10 randomly selected features in *featureList* will be plotted)
	:param savePath: If ``None``, plot interactively, otherwise attempt to save at this path.
	:type savePath: None or str
	:param str figureFormat: Format to save figure
	:param int dpi: Resolution to draw at
	:param tuple figureSize: Specify size of figure
	:type figureSize: tuple(float, float)
	"""
	
	# Create directory if doesn't exist
	if not savePath is None:
		if not os.path.exists(savePath):
			os.makedirs(savePath)
	
	# Define feature masks
	if featureMask is None:
		featureMask = numpy.ones(msData.featureMask.shape).astype(bool)
	featureList = [i for i, x in enumerate(featureMask) if x]
	featureList = numpy.asarray(featureList)
		
	# Initiate counter if maxNo
	if not maxNo is None:
		if maxNo<len(featureList):
			featureList = numpy.random.permutation(featureList)[:maxNo]

	# Define sample mask and run order
	LRmask = (msData.sampleMetadata['SampleType'].values == SampleType.StudyPool) & (msData.sampleMetadata['AssayRole'].values == AssayRole.LinearityReference)
	runIX = numpy.argsort(msData.sampleMetadata['Run Order'][LRmask].values)
	runIX = numpy.argsort(runIX)
	
	
	# Generate LRbatchmask (list of all sample LR subsets) and correlation to dilution for each
	LRbatchmask = generateLRmask(msData)
	corALL = numpy.zeros([len(LRbatchmask), len(msData.featureMask)])
	corLRbyBatch = {}
	i = 0
	for key in LRbatchmask:
		corALL[i,:] = _vcorrcoef(msData.intensityData, msData.sampleMetadata['Dilution'].values, method=msData.Attributes['corrMethod'], sampleMask = LRbatchmask[key])
		corLRbyBatch[key] = corALL[i,:]
		i = i+1
	
	LRbatchmask['MeanOverall'] = LRmask
	corLRbyBatch['MeanOverall'] = numpy.mean(corALL, axis=0)	
		
	for feature in featureList:
		
		if 'LRcorVals' in locals():
			del LRcorVals
		
		saveName = title + 'Correlation to Dilution Feature ' + str(feature) + '.png'
		
		fig = plt.figure(figsize=figureSize, dpi=dpi)		
		gs = gridspec.GridSpec(1, 5)
		ax1 = plt.subplot(gs[0,:-1])
		ax2 = plt.subplot(gs[0, -1:])
		
		# Plot scatter of LR intensity coloured by dilution
		ax1.scatter(runIX, msData.intensityData[LRmask, feature],
				c = msData.sampleMetadata['Dilution'][LRmask],
				cmap = plt.cm.jet)
		
		# Add a line where samples are not adjacent
		sampletime = [x - runIX[i - 1] for i, x in enumerate(runIX)]
		sampletime[0] = 1
		sampleBreaks = [i for i, v in enumerate(sampletime) if v > 1]
		ymin, ymax = ax1.get_ylim()
	
		for sample in sampleBreaks:
			ax1.plot([sample-0.5, sample-0.5], [ymin, ymax], color='k', linestyle='-', linewidth=0.5)

		ax1.set_xlabel('Run Order')
		ax1.set_ylabel('Intensity')
		ax1.set_title(title + 'Feature ' + msData.featureMetadata['Feature Name'][feature] + ' (Index ' + str(feature) +')')
		ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))


		# Add headmap of correlation to dilution for each batch subset and the mean
		
		# Generate correlation to dilution for each batch subset and mean overall
		for key in sorted(LRbatchmask):		
			if 'LRcorVals' not in locals():
				LRcorVals = pandas.DataFrame({'Feature': [feature], 'LR subset' : [key], 'Correlation' : [corLRbyBatch[key][feature]]})
			else:
				LRcorVals = LRcorVals.append({'Feature': feature, 'LR subset' : key, 'Correlation' : corLRbyBatch[key][feature]}, ignore_index=True)				

		# Plot heatmap of correlation to dilution value
		LRcorVals = LRcorVals.pivot('LR subset','Feature','Correlation')
		ax2 = sns.heatmap(LRcorVals, annot=True, fmt='.3g', vmin=-1, vmax=1, cmap='seismic', cbar=False)


		# Save or show
		if savePath != None:
			plt.savefig(os.path.join(savePath, saveName), bbox_inches='tight', format=figureFormat, dpi=dpi)
			plt.close()
		else:
			plt.show()
