import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import seaborn as sns
import numpy
import pandas
from ..objects._msDataset import MSDataset
from ..enumerations import AssayRole, SampleType
from ._violinPlot import _violinPlotHelper
import matplotlib.dates as mdates
from matplotlib.dates import MO, TU, WE, TH, FR, SA, SU
from matplotlib.dates import WeekdayLocator
from matplotlib.dates import DateFormatter
from matplotlib import gridspec
import os
import copy

def plotBatchAndROCorrection(dataset, datasetcorrected, featureList, addViolin=True,
							 colourBy='SampleClass', colourDict=None, markerDict=None,
							 abbrDict=None, opacity=.6,
							 sampleAnnotation=None, logy=False, title='',
							 withExclusions=True, savePath=None,
							 figureFormat='png', dpi=72, figureSize=(11,7)):
	"""
	Visualise the run-order correction applied to features, by plotting the values before and after correction, along with the fit calculated.
	
	:param MSDataset dataset: Dataset prior to correction
	:param MSDataset datasetcorrected: Dataset post-correction
	:param featureList: List of ints specifying indices of features to plot
	:type featureList: list[int,]
	:param bool addViolin: If ``true``, plot distributions as violin plots in addition to the longitudinal trend
	:param dict sampleAnnotation: Samples for annotation in plot, must include fields 'rank': index (int) and 'id': sample name (str, as in msData.sampleMetadata['Sample File Name']). For example, item['AbundanceSamples'] in featureID.py. 
	:param bool logy: If ``True`` plot intensities on a log10 scale
	:param str title: Text to title each plot with
	:param savePath: If ``None`` plot interactively, otherwise save the figures to the path specified
	:type savePath: None or str
	"""
	# Check inputs
	# Check dimensions of msData the same as msDatacorrected

	# TODO: implement plotting features by Run Order rather than by Acquired Time
	if ('Acquired Time' not in dataset.sampleMetadata.columns):
		raise NotImplementedError

	try:
		iterator = iter(featureList)
	except TypeError:
		# not iterable
		iterator = iter([featureList])
	else:
		pass

	# Apply sample/feature masks if exclusions to be applied
	msData = copy.deepcopy(dataset)
	msDatacorrected = copy.deepcopy(datasetcorrected)
	if withExclusions:
		msData.applyMasks()
		msDatacorrected.applyMasks()

		# Check that dimensions are the same
		try:
			# Attempting to add arrays ar1 and ar2
			msData.intensityData + msDatacorrected.intensityData
		except ValueError as ve:
			print(ve)
			# If ValueError occurs (arrays have different dimensions), return "Different dimensions"
			return "msData and msDatacorrected must have the same dimensions"

		# List unique classes in msData.sampleMetadata[colourBy]
		uniq_classes = msData.sampleMetadata[colourBy].unique()
		uniq = [str(i) for i in uniq_classes]

		# If colourDict check colour defined for every unique entry in class
		if colourDict is not None:
			if not all(k in colourDict.keys() for k in uniq):
				raise ValueError(
					'If colourDict is specified every unique entry in ' + colourBy + ' must be a key in colourDict')
		# Otherwise create colour dict
		else:
			colourDict = {}
			color = iter(cm.rainbow(numpy.linspace(0, 1, len(uniq))))
			for i in range(len(uniq)):
				colourDict[uniq[i]] = next(color)

		# If markerDict check colour defined for every unique entry in class
		if markerDict is not None:
			if not all(k in markerDict.keys() for k in uniq):
				raise ValueError(
					'If markerDict is specified every unique entry in ' + colourBy + ' must be a key in markerDict')
		else:
			markerDict = {}
			for u in uniq:
				markerDict[u] = 'o'

		# If abbrDict check abbr defined for every unique entry in class
		if abbrDict is not None:
			if not all(k in abbrDict.keys() for k in uniq):
				raise ValueError(
					'If abbrDict is specified every unique entry in ' + colourBy + ' must be a key in abbrDict')
		else:
			abbrDict = {}
			for u in uniq:
				abbrDict[u] = u

	# Use 'Acquired Time' of it exists
	if ('Acquired Time' in msData.sampleMetadata.columns):

		# X axis limits for formatting
		minX = msData.sampleMetadata['Acquired Time'].loc[
			msData.sampleMetadata['Run Order'] == min(msData.sampleMetadata['Run Order'])].values
		maxX = msData.sampleMetadata['Acquired Time'].loc[
			msData.sampleMetadata['Run Order'] == max(msData.sampleMetadata['Run Order'])].values
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

	# Get and sort the fit data
	localBatch = msDatacorrected.sampleMetadata['Correction Batch'].values
	localFit = msDatacorrected.fit

	sortedRO = numpy.argsort(acqTime)
	acqTimeSorted = acqTime[sortedRO]
	fitSorted = localFit[sortedRO, :]
	localBatch = localBatch[sortedRO]

	batches = (numpy.unique(localBatch[~numpy.isnan(localBatch)])).astype(int)

	# define the colours (different for each batch) and get axis y-limits
	cmap = plt.get_cmap('gnuplot')
	colors = [cmap(i) for i in numpy.linspace(0, 1, len(batches) + 1)]

	for feature in iterator:
		
		# Validate inputs
		if not isinstance(feature, int):
			raise TypeError("feature number %s is not an integer." % type(feature))
		if not feature <= msData.intensityData.shape[1]:
			raise ValueError("feature (%s) greater than number of features in msData (%s)." % (feature, msData.intensityData.shape[1]))
			
		sns.set_color_codes(palette='deep')
		fig = plt.figure(figsize=figureSize, dpi=dpi)
		gs = gridspec.GridSpec(2, 5)

		if addViolin == False:
			ax = plt.subplot(gs[:,:-1])
		else:
			ax = plt.subplot(gs[:,:-1])
			ax2 = plt.subplot(gs[0,-1])
			ax3 = plt.subplot(gs[1,-1])

		# Plot feature intensity for different sample types
		palette = {}
		sampleMasks = []

		for u in uniq:

			# Plot uncorrected data
			ax.scatter(acqTime[msData.sampleMetadata[colourBy] == u],
							msData.intensityData[msData.sampleMetadata[colourBy] == u, feature],
							marker=markerDict[u],
							s=30,
							c=colourDict[u],
							alpha=opacity, # opacity
							label=u)

			# Plot arrows (base to point = before to after correction)
			SS = numpy.where(msData.sampleMetadata[colourBy] == u)
			temp = SS[0]
			for sample in temp:
				ax.annotate('', xy=(acqTime[sample],
									msDatacorrected.intensityData[sample, feature]),
							xytext=(acqTime[sample],
									msData.intensityData[sample, feature]),
							arrowprops=dict(edgecolor=colourDict[u],
											facecolor=colourDict[u], alpha=0.5,
											arrowstyle='-|>', shrinkA=0, shrinkB=0),
							clip_on=True)

			# Save masks for violinplots
			if addViolin:
				sampleMasks.append((abbrDict[u], msData.sampleMetadata[colourBy] == u))
				palette[abbrDict[u]] = colourDict[u]

		# Plot fit coloured by batch
		colIX = 1
		for i in batches:
			ax.plot(acqTimeSorted[localBatch==i],
					fitSorted[localBatch==i,feature], c=colors[colIX],
					alpha=0.6, label='Fit for batch ' + str(colIX))
			colIX = colIX + 1
						
		# Add sample annotation if required
		if sampleAnnotation is not None:
			for sample in sampleAnnotation:
				sampleIX = msData.sampleMetadata[msData.sampleMetadata['Sample File Name']==sample['id']].index
				sampleIX = int(sampleIX[0])
				ax.text(acqTime[sampleIX],
					msDatacorrected.intensityData[sampleIX, feature], 
					str(sample['rank']), 
					horizontalalignment='center', 
					verticalalignment='bottom')			

		# ax formatting
		ax.set_ylabel('Feature Intensity')
		if ('Acquired Time' in msData.sampleMetadata.columns):
			ax.set_xlabel('Acquisition Date')
			ax.set_xlim(minX, maxX)
			ax.xaxis.set_major_locator(loc)
			ax.xaxis.set_major_formatter(formatter)
		else:
			ax.set_xlabel('Run Order')
		labels = ax.get_xticklabels() 
		for label in labels:
			label.set_rotation(30) 
			label.set_horizontalalignment('right')
		if logy:
			ax.set_yscale('log', nonpositive='clip')
			ax.set_ylabel('Feature Intensity (log scale)')
		else:
			ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
		if addViolin:
			ax.legend()
		else:
			ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
		ax.set_title(title)

		if addViolin: # If required, violin plot of data distribution

			limits = ax.get_ylim()

			_violinPlotHelper(ax2, msData.intensityData[:, feature], sampleMasks, 'Pre-correction', None, palette=palette, ylimits=limits, logy=logy)
			_violinPlotHelper(ax3, msDatacorrected.intensityData[:, feature], sampleMasks, 'Post-correction', None, palette=palette, ylimits=limits, logy=logy)

		# figure formatting
		fig.tight_layout()
		
		if savePath is not None:
			if '.' in savePath:
				saveTo = savePath
			else:
				fileName = str(feature).zfill(5) + '_' + str(numpy.squeeze(msData.featureMetadata.loc[feature, 'Feature Name'])).replace('/', '-') + '.' + figureFormat
				saveTo = os.path.join(savePath, fileName)
			plt.savefig(saveTo, format=figureFormat, dpi=dpi)
			plt.close(fig)
		else:
			plt.show()
