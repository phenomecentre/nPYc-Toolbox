import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy
import pandas
from ..objects._msDataset import MSDataset
from ..utilities import generateLRmask
from ..enumerations import AssayRole, SampleType
from ._violinPlot import _violinPlotHelper
import matplotlib.dates as mdates
from matplotlib.dates import MO, TU, WE, TH, FR, SA, SU
from matplotlib.dates import WeekdayLocator
from matplotlib.dates import DateFormatter
from matplotlib import gridspec
from matplotlib.patches import Rectangle
import os
import copy

def plotBatchAndROCorrection(msData, msDatacorrected, featureList, addViolin=True, sampleAnnotation=None, logy=False, title='', savePath=None, figureFormat='png', dpi=72, figureSize=(11,7)):
	"""
	Visualise the run-order correction applied to features, by plotting the values before and after correction, along with the fit calculated.
	
	:param MSDataset msData: Dataset prior to correction
	:param MSDataset msDatacorrected: Dataset post-correction
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

	try:
		iterator = iter(featureList)
	except TypeError:
		# not iterable
		iterator = iter([featureList])
	else:
		pass

	# Load toolbox wide color scheme
	if 'sampleTypeColours' in msData.Attributes.keys():
		sTypeColourDict = copy.deepcopy(msData.Attributes['sampleTypeColours'])
		for stype in SampleType:
			if stype.name in sTypeColourDict.keys():
				sTypeColourDict[stype] = sTypeColourDict.pop(stype.name)
	else:
		sTypeColourDict = {SampleType.StudySample: 'b', SampleType.StudyPool: 'g', SampleType.ExternalReference: 'r',
							SampleType.MethodReference: 'm', SampleType.ProceduralBlank: 'c', 'Other': 'grey'}

	# Define sample types and exclude masked samples
	SSmask = (msData.sampleMetadata['SampleType'].values == SampleType.StudySample) & (msData.sampleMetadata['AssayRole'].values == AssayRole.Assay)
	SPmask = (msData.sampleMetadata['SampleType'].values == SampleType.StudyPool) & (msData.sampleMetadata['AssayRole'].values == AssayRole.PrecisionReference)
	ERmask = (msData.sampleMetadata['SampleType'].values == SampleType.ExternalReference) & (msData.sampleMetadata['AssayRole'].values == AssayRole.PrecisionReference)
	LRmask = (msData.sampleMetadata['SampleType'].values == SampleType.StudyPool) & (msData.sampleMetadata['AssayRole'].values == AssayRole.LinearityReference)

	# Get and sort the fit data
	localRO = msData.sampleMetadata['Acquired Time'].values
	localBatch = msDatacorrected.sampleMetadata['Correction Batch'].values
	localFit = msDatacorrected.fit

	sortedRO = numpy.argsort(localRO)
	sortedRO2 = localRO[sortedRO]
	fitSorted = localFit[sortedRO,:]
	localBatch = localBatch[sortedRO]
	
	batches = (numpy.unique(localBatch[~numpy.isnan(localBatch)])).astype(int)

	# define the colours (different for each batch) and get axis y-limits
	cmap = plt.get_cmap('gnuplot')
	colors = [cmap(i) for i in numpy.linspace(0, 1, len(batches)+1)]

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


		# Plot feature intensity for different sample types

		# SS
		if sum(SSmask) > 0:

			# Plot data
			ax.plot_date([pandas.to_datetime(d) for d in msData.sampleMetadata.loc[SSmask, 'Acquired Time']], msData.intensityData[SSmask, feature], c=sTypeColourDict[SampleType.StudySample], fmt='o', ms=4, alpha=0.5, label='Study Sample')

			# Plot arrows (base to point = before to after correction)
			SS = numpy.where(SSmask==True)
			temp = SS[0]
			for sample in temp:
				ax.annotate('', xy=(mdates.date2num(msData.sampleMetadata.loc[sample, 'Acquired Time']),
						msDatacorrected.intensityData[sample, feature]),
						xytext=(mdates.date2num(msData.sampleMetadata.loc[sample, 'Acquired Time']),
						msData.intensityData[sample, feature]),
						arrowprops=dict(edgecolor=sTypeColourDict[SampleType.StudySample], facecolor=sTypeColourDict[SampleType.StudySample], alpha=0.5, arrowstyle = '-|>', shrinkA=0, shrinkB=0),
						clip_on=True)

		# SR
		if sum(SPmask) > 0:

			ax.plot_date([pandas.to_datetime(d) for d in msData.sampleMetadata.loc[SPmask, 'Acquired Time']], msData.intensityData[SPmask, feature], c=sTypeColourDict[SampleType.StudyPool], fmt='o', ms=4, alpha=0.9, label='Study Reference')

			SP = numpy.where(SPmask==True)
			temp = SP[0]
			for sample in temp:
				ax.annotate('', xy=(mdates.date2num(msData.sampleMetadata.loc[sample, 'Acquired Time']),
						msDatacorrected.intensityData[sample, feature]),
						xytext=(mdates.date2num(msData.sampleMetadata.loc[sample, 'Acquired Time']),
						msData.intensityData[sample, feature]),
						arrowprops=dict(edgecolor=sTypeColourDict[SampleType.StudyPool], facecolor=sTypeColourDict[SampleType.StudyPool], alpha=0.9, arrowstyle = '-|>', shrinkA=0, shrinkB=0),
						clip_on=True)

		# LTR
		if sum(ERmask) > 0:

			ax.plot_date([pandas.to_datetime(d) for d in msData.sampleMetadata.loc[ERmask, 'Acquired Time']], msData.intensityData[ERmask, feature], c=sTypeColourDict[SampleType.ExternalReference], fmt='o', ms=4, alpha=0.9, label='Long-Term Reference')

			ER = numpy.where(ERmask==True)
			temp = ER[0]
			for sample in temp:
				ax.annotate('', xy=(mdates.date2num(msData.sampleMetadata.loc[sample, 'Acquired Time']),
						msDatacorrected.intensityData[sample, feature]),
						xytext=(mdates.date2num(msData.sampleMetadata.loc[sample, 'Acquired Time']),
						msData.intensityData[sample, feature]),
						arrowprops=dict(edgecolor=sTypeColourDict[SampleType.ExternalReference], facecolor=sTypeColourDict[SampleType.ExternalReference], alpha=0.9, arrowstyle = '-|>', shrinkA=0, shrinkB=0),
						clip_on=True)


		# SRD
		if sum(LRmask) > 0:

			ax.plot_date([pandas.to_datetime(d) for d in msData.sampleMetadata.loc[LRmask, 'Acquired Time']], msData.intensityData[LRmask, feature], c=sTypeColourDict[SampleType.MethodReference], fmt='s', ms=4, alpha=0.9, label='Serial Dilution')


		# Plot fit coloured by batch
		colIX = 1
		for i in batches:
			ax.plot([pandas.to_datetime(d) for d in sortedRO2[localBatch==i]], fitSorted[localBatch==i,feature], c=colors[colIX], alpha=0.9, label='Fit for batch ' + str(colIX))
			colIX = colIX + 1
						
		# Add sample annotation if required
		if sampleAnnotation is not None:
			for sample in sampleAnnotation:
				sampleIX = msData.sampleMetadata[msData.sampleMetadata['Sample File Name']==sample['id']].index
				sampleIX = int(sampleIX[0])
				ax.text(mdates.date2num(msDatacorrected.sampleMetadata.loc[sampleIX, 'Acquired Time']), 
					msDatacorrected.intensityData[sampleIX, feature], 
					str(sample['rank']), 
					horizontalalignment='center', 
					verticalalignment='bottom')			

		# ax formatting
		ax.set_xlim([[pandas.to_datetime(d) for d in minX][0], [pandas.to_datetime(d) for d in maxX][0]])
		ax.set_xlabel('Acquisition Date')
		ax.set_ylabel('Feature Intensity')
		ax.xaxis.set_major_locator(loc)
		ax.xaxis.set_major_formatter(formatter)
		labels = ax.get_xticklabels() 
		for label in labels:
			label.set_rotation(30) 
			label.set_horizontalalignment('right')
		if logy:
			ax.set_yscale('symlog')
		else:
			ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
		if addViolin:
			ax.legend()
		else:
			ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
		ax.set_title(title)

		if addViolin == True: # If required, violin plot of data distribution

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
