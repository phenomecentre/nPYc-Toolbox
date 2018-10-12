# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 13:27:55 2016

@author: cs401
"""
import matplotlib
from matplotlib import gridspec
import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex
from matplotlib.collections import LineCollection
import matplotlib.dates as mdates
#from matplotlib.dates import AutoDateFormatter
#from matplotlib.dates import AutoDateLocator
import seaborn as sns
import plotly.graph_objs as go
import pandas
import numpy
from ._violinPlot import _violinPlotHelper
from pyChemometrics.ChemometricsPCA import ChemometricsPCA
from ._plotDiscreteLoadings import plotDiscreteLoadings
from ..objects import Dataset
from ..enumerations import SampleType
import copy

def plotScree(R2, Q2=None, title = '', xlabel='', ylabel='', savePath=None, figureFormat='png', dpi=72, figureSize=(11,7)):
	"""
	Plot a barchart of variance explained (R2) and predicted (Q2) (if available) for each PCA component.

	:param numpy.array R2: PCA R2 values
	:param numpy.array Q2: PCA Q2 values
	:param str title: Title for the plot
	:param str xlabel: Label for the x-axis
	:param str ylabel: Label for the y-axis
	"""

	fig, ax = plt.subplots(figsize=figureSize, dpi=dpi)

	ind = numpy.arange(len(R2))
	width = 0.35

	ax.bar(ind, R2, width, color='#3498db', alpha=.4, label='R2')

	if Q2 is not None:
		ax.bar(ind+width, Q2, width, color='#16a085', alpha=.4, label='Q2')

	ax.set_ylabel(ylabel)
	ax.set_xlabel(xlabel)
	ax.set_xticks(ind + width)
	ax.set_xticklabels(numpy.arange(1,len(R2)+1))
	ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
	fig.suptitle(title)

	if savePath:
		plt.savefig(savePath, bbox_inches='tight', format=figureFormat, dpi=dpi)
		plt.close()
	else:
		plt.show()

def plotScores(pcaModel, classes=None, classType=None, components=None, alpha = 0.05, plotAssociation=None, title ='', xlabel='', figures=None, savePath=None, figureFormat='png', dpi=72, figureSize=(11, 7)):
	"""
	Plot PCA scores for each pair of components in PCAmodel, coloured by values defined in classes, and with Hotelling's T2 ellipse (95%)

	:param ChemometricsPCA pcaModel: PCA model object (scikit-learn based)
	:param pandas.Series classes: Measurement/groupings associated with each sample, e.g., BMI/treatment status
	:param str classType: Type of data in ``classes``, either 'Plot Sample Type', 'categorical' or 'continuous', must be specified if classes is not ``None``. If ``classType`` is 'Plot Sample Type', ``classes`` expects 'Study Sample', 'Study Pool', 'External Reference', 'Linearity Reference' or 'Sample'.
	:param components: If ``None`` plots all components in model, else plots those specified in components
	:type components: tuple (int, int)
	:param float alpha: Significance value for plotting Hotellings ellipse
	:param bool plotAssociation: If ``True``, plots the association between each set of PCA scores and the metadata values
	:param numpy.array significance: Significance of association of scores from each component with values in classes from correlation or Kruskal-Wallis test for example (see multivariateReport.py)
	:param str title: Title for the plot
	:param str xlabel: Label for the x-axis
	:param dict figures: If not ``None``, saves location of each figure for output in html report (see multivariateReport.py)
	"""

	# Check inputs
	if not isinstance(pcaModel, ChemometricsPCA):
		raise TypeError('PCAmodel must be an instance of ChemometricsPCA')

	if classes is not None and classType is None:
		raise ValueError('If classes is specified, classType must be')

	from matplotlib.patches import Ellipse

	# Preparation
	values = pcaModel.scores
	ns, nc = values.shape

	if components is None:
		components = numpy.ones([nc]).astype(bool)
	components = numpy.where(components==True)
	components = components[0]
	# TODO: fix this so can plot if model only has one component
	if len(components)==1:
		temp = numpy.arange(0,nc)
		components = numpy.append(components, min(temp[temp!=components]))
	nc = len(components)

	if title != '':
		title = title.translate ({ord(c): " " for c in "!@#$%^&*()[]{};:,./<>?\|`~-=_+"})
		plotTitle = title + ' '
		title = ''.join(title.split())
	else:
		plotTitle = ''

	if classes is None:
		classes = pandas.Series('Sample' for i in range(ns))
		classType = 'categorical'

	if classType == 'categorical':
		classes = classes.astype(str)

	uniq = classes.unique()
	try:
		uniq.sort()
	except:
		pass

	# Calculate critical value for Hotelling's T2
	#Fval = f.ppf(0.95, 2, ns-2)
	# Plot scores for each pair of components
	for i in numpy.arange(0,nc,2):

		if i+1 >= nc:
			j = 0
		else:
			j = i+1

		if plotAssociation is not None:
			fig = plt.figure(figsize=figureSize, dpi=dpi)
			gs = gridspec.GridSpec(2, 10)
			ax = plt.subplot(gs[:,3:])
			ax1 = plt.subplot(gs[0,:2])
			ax2 = plt.subplot(gs[1,:2])
		else:
			fig, ax = plt.subplots(figsize=figureSize, dpi=dpi)

		# If colouring by Sample Type - use standard reporting colours
		if classType == 'Plot Sample Type':

			## Try loading toolbox wide color scheme
			# value just in case
			sTypeColourDict = {SampleType.StudySample: 'b', SampleType.StudyPool: 'g', SampleType.ExternalReference: 'r',
							   SampleType.MethodReference: 'm', SampleType.ProceduralBlank: 'c', 'Other': 'grey'}
			# load from the SOP as we do not have access to object
			try:
				from .._toolboxPath import toolboxPath
				import json
				import os
				import copy

				with open(os.path.join(toolboxPath(), 'StudyDesigns', 'SOP', 'Generic.json')) as data_file:
					attributes = json.load(data_file)
				# convert key names to SampleType enum
				if 'sampleTypeColours' in attributes.keys():
					sTypeColourDict = copy.deepcopy(attributes['sampleTypeColours'])
					for stype in SampleType:
						if stype.name in sTypeColourDict.keys():
							sTypeColourDict[stype] = sTypeColourDict.pop(stype.name)
			except:
				pass

			sns.set_color_codes(palette='deep')
			if any(classes == 'Study Sample'):
				ax.scatter(values[classes.values == 'Study Sample', components[i]], values[classes.values == 'Study Sample', components[j]], c=sTypeColourDict[SampleType.StudySample], label='Study Sample')
			if any(classes == 'Study Pool'):
				ax.scatter(values[classes.values == 'Study Pool', components[i]], values[classes.values == 'Study Pool', components[j]], c=sTypeColourDict[SampleType.StudyPool], label='Study Pool')
			if any(classes == 'External Reference'):
				ax.scatter(values[classes.values == 'External Reference', components[i]], values[classes.values == 'External Reference', components[j]], c=sTypeColourDict[SampleType.ExternalReference], label='External Reference')
			if any(classes == 'Linearity Reference'):
				ax.scatter(values[classes.values == 'Linearity Reference', components[i]], values[classes.values == 'Linearity Reference', components[j]], c=sTypeColourDict[SampleType.MethodReference], label='Linearity Reference')
			if any(classes == 'Sample'):
				ax.scatter(values[classes.values == 'Sample', components[i]], values[classes.values == 'Sample', components[j]], c=sTypeColourDict['Other'], label='Sample')
			ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

		elif classType == 'categorical':

			colors_sns = {}
			# First plot any nans
			if any(u in {'nan', 'NaN', 'NaT', '', 'NA'} for u in uniq):
				nans = [i for i, x in enumerate(classes) if x in {'nan', 'NaN', 'NaT', '', 'NA'}]
				ax.scatter(values[nans, components[i]], values[nans, components[j]], c='#D3D3D3', label='NA')
				nans = [u in {'nan', 'NaN', 'NaT', '', 'NA'} for u in uniq]
				nans = [i for i, x in enumerate(uniq) if x not in {'nan', 'NaN', 'NaT', '', 'NA'}]
				uniqnonan = uniq[nans]
				colors_sns['NA'] = '#D3D3D3'

			else:
				uniqnonan = uniq

			# Plot remaining categories
			classIX=0
			colors = iter(plt.cm.rainbow(numpy.linspace(0,1,len(uniqnonan))))
			for u in uniqnonan:
				c = rgb2hex(next(colors))
				if classIX<20:
					ax.scatter(values[classes.values == u, components[i]], values[classes.values == u, components[j]], c=c, label=u)#olors[classIX], label=u)
				elif classIX==len(uniqnonan)-1:
					ax.scatter(values[classes.values == u, components[i]], values[classes.values == u, components[j]], c='0', alpha=0, label='...')
					ax.scatter(values[classes.values == u, components[i]], values[classes.values == u, components[j]], c=c, label=u)#colors[classIX], label=u)
				else:
					ax.scatter(values[classes.values == u, components[i]], values[classes.values == u, components[j]], c=c, label='_nolegend_')#colors[classIX], label='_nolegend_')
				classIX = classIX+1
				colors_sns[u] = c

			if plotAssociation is not None:

				nonans = [i for i, x in enumerate(classes) if x in {'nan', 'NaN', 'NaT', '', 'NA'}]
				plotClasses = classes.copy()
				plotClasses[nonans] = 'NA'
				tempdata = {plotTitle: plotClasses,
						   'PC'+str(components[i]+1): values[:, components[i]],
						   'PC'+str(components[j]+1): values[:, components[j]]}
				tempdata = pandas.DataFrame(tempdata, columns=[plotTitle, 'PC'+str(components[i]+1), 'PC'+str(components[j]+1)])

				# Association for component[i]
				ax1 = sns.swarmplot(x=plotTitle, y='PC'+str(components[i]+1), data=tempdata, ax=ax1, palette=colors_sns)
				ax1.set(xticklabels=[])
				ax1.set(xlabel='')

				# Association for component[j]
				ax2 = sns.swarmplot(x=plotTitle, y='PC'+str(components[j]+1), data=tempdata, ax=ax2, palette=colors_sns)
				ax2.set(xticklabels=[])

			ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

		elif classType == 'continuous':

			plotnans = classes.isnull().values
			if sum(plotnans != 0):
				ax.scatter(values[plotnans==True, components[i]], values[plotnans==True, components[j]], c='#D3D3D3', label='NA')
				ax.legend()

			cb = ax.scatter(values[plotnans==False, components[i]], values[plotnans==False, components[j]], c=classes[plotnans==False], cmap=plt.cm.rainbow)
			cbar = plt.colorbar(cb, ax=ax)
			cbar.set_label(title)

			if plotAssociation is not None:

				xvalnan = numpy.unique(classes[plotnans==False])
				xvalnan = xvalnan[-1] + xvalnan[-1] - xvalnan[-2]

				# Association for component[i]
				ax1.scatter(classes[plotnans==False], values[plotnans==False, components[i]], c=classes[plotnans==False], cmap=plt.cm.rainbow)
				ax1.scatter(numpy.ones([sum(plotnans),1])*xvalnan, values[plotnans, components[i]], c='#D3D3D3')
				ax1.set_ylabel('PC' + str(components[i]+1))
				ax1.set(xticklabels=[])

				# Association for component[j]
				ax2.scatter(classes[plotnans==False], values[plotnans==False, components[j]], c=classes[plotnans==False], cmap=plt.cm.rainbow)
				ax2.scatter(numpy.ones([sum(plotnans),1])*xvalnan, values[plotnans, components[j]], c='#D3D3D3')
				ax2.set_xlabel(plotTitle)
				ax2.set_ylabel('PC' + str(components[j]+1))

		# Add Hotelling's T2
		hotelling_ellipse = pcaModel.hotelling_T2(comps=numpy.array([components[i], components[j]]), alpha=alpha)

		#a = numpy.sqrt(numpy.var(values[:,components[i]])*Fval*2*((ns-1)/(ns-2)));
		#b = numpy.sqrt(numpy.var(values[:,components[j]])*Fval*2*((ns-1)/(ns-2)));
		ellipse = Ellipse(xy=(0, 0), width=hotelling_ellipse[0]*2, height=hotelling_ellipse[1]*2,
			edgecolor='k', fc='None', lw=2)
		ax.add_patch(ellipse)

		# Annotate
		ylabel = 'PC' + str(components[j]+1) + ' (' + '{0:.2f}'.format(pcaModel.modelParameters['VarExpRatio'][components[j]] * 100) + '%)'
		xlabel = 'PC' + str(components[i]+1) + ' (' + '{0:.2f}'.format(pcaModel.modelParameters['VarExpRatio'][components[i]] * 100) + '%)'
		if plotAssociation is not None:
			ylabel = ylabel + ' significance: ' + '{0:.2f}'.format(plotAssociation[components[j]])
			xlabel = xlabel + ' significance: ' + '{0:.2f}'.format(plotAssociation[components[i]])
		ax.set_ylabel(ylabel)
		ax.set_xlabel(xlabel)
		fig.suptitle(plotTitle + 'PC' + str(components[i]+1) + ' vs PC' + str(components[j]+1))

		# Save or show
		if savePath:

			if figures is not None:
				saveTemp = title + 'PC' + str(components[i]+1) + 'vsPC' + str(components[j]+1)
				figures[saveTemp] = savePath + saveTemp + '.' + figureFormat
			else:
				saveTemp = ''
			plt.savefig(savePath + saveTemp + '.' + figureFormat, bbox_inches='tight', format=figureFormat, dpi=dpi)
			plt.close()

		else:
			plt.show()

	# Return figures if saving for output in html report
	if figures is not None:
		return figures


def plotOutliers(values, runOrder, sampleType=None, addViolin=False, Fcrit=None, FcritAlpha=None, PcritPercentile=None, title='', xlabel='Run Order', ylabel='', savePath=None, figureFormat='png', dpi=72, figureSize=(11,7)):
	"""
	Plot scatter plot of PCA outlier stats sumT (strong) or DmodX (moderate), with a line at [25, 50, 75, 95, 99] quantiles and at a critical value if specified

	:param numpy.array values: dModX or sum of scores, measure of 'fit' for each sample
	:param numpy.array runOrder: Order of sample acquisition (samples are plotted in this order)
	:param pandas.Series sampleType: Sample type of each sample, must be from 'Study Sample', 'Study Pool', 'External Reference', or 'Sample' (see multivariateReport.py)
	:param bool addViolin: If True adds a violin plot of distribution of values
	:param float Fcrit: If not none, plots a line at Fcrit
	:param float FcritAlpha: Alpha value for Fcrit (for legend)
	:param float PcritPercentile: If not none, plots a line at this quantile
	:param str title: Title for the plot
	:param str xlabel: Label for the x-axis
	"""

	# Preparation
	if isinstance(sampleType, (str, type(None))):
		sampleType = pandas.Series(['Sample' for i in range(0, len(values))], name='sampleType')

	quantiles = [25, 50, 75, 95, 99]
	
	# Plot line at PcritPercentile in red if present
	if PcritPercentile is not None:
		if PcritPercentile in quantiles:
			quantiles.remove(PcritPercentile)
	
	quantilesVals = numpy.percentile(values, quantiles)

	## Try loading toolbox wide color scheme
	# value just in case
	sTypeColourDict = {SampleType.StudySample: 'b', SampleType.StudyPool: 'g', SampleType.ExternalReference: 'r',
					   SampleType.MethodReference: 'm', SampleType.ProceduralBlank: 'c', 'Other': 'grey'}
	# load from the SOP as we do not have access to object
	try:
		from .._toolboxPath import toolboxPath
		import json
		import os
		import copy

		with open(os.path.join(toolboxPath(), 'StudyDesigns', 'SOP', 'Generic.json')) as data_file:
			attributes = json.load(data_file)
		# convert key names to SampleType enum
		if 'sampleTypeColours' in attributes.keys():
			sTypeColourDict = copy.deepcopy(attributes['sampleTypeColours'])
			for stype in SampleType:
				if stype.name in sTypeColourDict.keys():
					sTypeColourDict[stype] = sTypeColourDict.pop(stype.name)
	except:
		pass

	sns.set_color_codes(palette='deep')
	plt.figure(figsize=figureSize, dpi=dpi)
	gs = gridspec.GridSpec(1, 5)

	if addViolin == False:
		ax = plt.subplot(gs[0,:])
	else:
		ax = plt.subplot(gs[0,:-2])
		ax2 = plt.subplot(gs[0,-1])
	sampleMasks = []
	palette = {}
	
	# Plot data coloured by sample type
	if any(sampleType == 'Study Sample'):
		ax.scatter(runOrder[sampleType.values == 'Study Sample'], values[sampleType.values == 'Study Sample'], c=sTypeColourDict[SampleType.StudySample], label='Study Sample')
		sampleMasks.append(('SS', sampleType.values=='Study Sample'))
		palette['SS'] = sTypeColourDict[SampleType.StudySample]
	if any(sampleType == 'Study Pool'):
		ax.scatter(runOrder[sampleType.values == 'Study Pool'], values[sampleType.values == 'Study Pool'], c=sTypeColourDict[SampleType.StudyPool], label='Study Pool')
		sampleMasks.append(('SP', sampleType.values=='Study Pool'))
		palette['SP'] = sTypeColourDict[SampleType.StudyPool]
	if any(sampleType == 'External Reference'):
		ax.scatter(runOrder[sampleType.values == 'External Reference'], values[sampleType.values == 'External Reference'], c=sTypeColourDict[SampleType.ExternalReference], label='External Reference')
		sampleMasks.append(('ER', sampleType.values=='External Reference'))
		palette['ER'] = sTypeColourDict[SampleType.ExternalReference]
	if any(sampleType == 'Sample'):
		ax.scatter(runOrder[sampleType.values == 'Sample'], values[sampleType.values == 'Sample'], c=sTypeColourDict['Other'], label='Sample')
		sampleMasks.append(('Sample', sampleType.values=='Sample'))
		palette['Sample'] = sTypeColourDict['Other']
		
	xmin, xmax = ax.get_xlim()

	# TODO: DmodX from pyChemometrics, what about the other measure?

	# Plot lines at quantiles
	for q in numpy.arange(0, len(quantiles)):
		ax.plot([xmin, xmax],[quantilesVals[q], quantilesVals[q]], 'k--', label='Q'+str(quantiles[q]))

	# Add line at Fcrit critical value
	if Fcrit:
		ax.plot([xmin, xmax],[Fcrit, Fcrit], 'c--', label='Fcrit (' + str(FcritAlpha) + ')')
		
	# Add line at PcritPercentage critical value
	if PcritPercentile:
		Pcrit = numpy.percentile(values, PcritPercentile)
		ax.plot([xmin, xmax],[Pcrit, Pcrit], 'r--', label='Q'+str(PcritPercentile))	

	# Annotate
	ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	ax.set_xlim([xmin, xmax])

	# If required, violin plot of data distribution
	if addViolin == True:

		limits = ax.get_ylim()

		_violinPlotHelper(ax2, values, sampleMasks, None, 'Sample Type', palette=palette, ylimits=limits, logy=False)

		ax2.yaxis.set_ticklabels([])

		sns.despine(trim=True, ax=ax2)

	# Save or show
	if savePath:
		plt.savefig(savePath, bbox_inches='tight', format=figureFormat, dpi=dpi)
		plt.close()
	else:
		plt.show()


def plotLoadings(pcaModel, msData, title='', figures=None, savePath=None, figureFormat='png', dpi=72, figureSize=(11, 7)):
	"""
	Plot PCA loadings for each component in PCAmodel. For NMR data plots the median spectrum coloured by the loading. For MS data plots an ion map (rt vs. mz) coloured by the loading.

	:param ChemometricsPCA pcaModel: PCA model object (scikit-learn based)
	:param Dataset msData: Dataset object
	:param str title: Title for the plot
	:param dict figures: If not ``None``, saves location of each figure for output in html report (see multivariateReport.py)
	"""

	nc = pcaModel.scores.shape[1]

	if ((msData.VariableType.name == 'Discrete') & (hasattr(msData.featureMetadata, 'Retention Time'))):

		Xvals = msData.featureMetadata['Retention Time']
		Xlabel = 'Retention Time'

		Yvals = msData.featureMetadata['m/z']
		Ylabel = 'm/z'

	elif ((msData.VariableType.name == 'Continuum') & (hasattr(msData.featureMetadata, 'ppm'))):

		Xvals = msData.featureMetadata['ppm']
		Xlabel = chr(948)+ '1H'

		Yvals = numpy.median(msData.intensityData, axis=0)
		Ylabel = 'Median Intensity'

	elif msData.VariableType.name == 'Discrete':
		compStep = 3

		if savePath:
			saveTemp = title + 'PCAloadings'
			saveTo = savePath + saveTemp + '.' + figureFormat

			if figures is not None:
				figures[saveTemp] = saveTo
		else:
			saveTo = None
		plotDiscreteLoadings(msData, pcaModel, nbComponentPerRow=compStep, savePath=saveTo, figureFormat=figureFormat, dpi=dpi)

		return figures

	else:
		print('add this functionality!!!')


	for i in numpy.arange(0,nc):

		cVect = pcaModel.loadings[i, :]
		orig_cmap = plt.cm.RdYlBu_r # Red for high, Blue for negative, and we will have a very neutral yellow for 0
		maxcol = numpy.max(cVect) # grab the maximum
		mincol = numpy.min(cVect) # Grab the minimum
		new_cmap = _shiftedColorMap(orig_cmap, start=0, midpoint=1 - maxcol/(maxcol + numpy.abs(mincol)), stop=1, name='new')

		fig, ax = plt.subplots(figsize=figureSize, dpi=dpi)

		if ((msData.VariableType.name == 'Discrete') & (hasattr(msData.featureMetadata, 'Retention Time'))):

			# To set the alpha of each point to be associated with the weight of the loading, generate an array where each row corresponds to a feature, the
			# first three columns to the colour of the point, and the last column to the alpha value
			from matplotlib.colors import Normalize
			import matplotlib.cm as cm

			# Return the colours for each feature
			norm = Normalize(vmin=mincol, vmax=maxcol)
			cb = cm.ScalarMappable(norm=norm, cmap=new_cmap)
			cVectAlphas = numpy.zeros((pcaModel.loadings.shape[1], 4))
			cIX = 0
			for c in cVect:
				cVectAlphas[cIX,:] = cb.to_rgba(cVect[cIX])
				cIX = cIX+1

			# Set the alpha (min 0.2, max 1)
			cVectAlphas[:,3] = (((abs(cVect) - numpy.min(abs(cVect))) * (1 - 0.2)) / (numpy.max(abs(cVect)) - numpy.min(abs(cVect)))) + 0.2
			if any(cVectAlphas[:,3] > 1):
				cVectAlphas[cVectAlphas[:,3]>1,3] = 1

			# Plot
			ax.scatter(Xvals, Yvals, color=cVectAlphas)#, edgecolors='k')
			cb.set_array(cVect)
			ax.set_xlim([min(msData.featureMetadata['Retention Time'])-1, max(msData.featureMetadata['Retention Time'])+1])

		elif ((msData.VariableType.name == 'Continuum') & (hasattr(msData.featureMetadata, 'ppm'))):

			# The rasterized ... I don't think it made a big difference, but this was me trying to improve zooming/panning performance. We can compare again without it,
			# as I dont remember if my final conclusion was "rasterized is important therefore leave it " or "doesn't matter leave it"
			ax.set_rasterized(True)

			lvector = cVect
			points = numpy.array([Xvals, Yvals]).transpose().reshape(-1,1,2)
			segs = numpy.concatenate([points[:-1],points[1:]],axis=1)

			cb = LineCollection(segs, cmap=new_cmap)
			cb.set_array(lvector)
			plt.gca().add_collection(cb) # add the collection to the plot
			plt.xlim(Xvals.min()-0.4, Xvals.max() + 0.4) # line collections don't auto-scale the plot
			plt.ylim(Yvals.min()*1.2, Yvals.max()*1.2)
			plt.gca().invert_xaxis()

		cbar = plt.colorbar(cb)
		cbar.set_label('Loadings')
		ax.set_xlabel(Xlabel)
		ax.set_ylabel(Ylabel)
		fig.suptitle('PCA Loadings for PC' + str(i+1))

		if savePath:

			if figures is not None:
				saveTemp = title + 'PCAloadingsPC' + str(i+1)
				figures[saveTemp] = savePath + saveTemp + '.' + figureFormat
			else:
				saveTemp = ''
			plt.savefig(savePath + saveTemp + '.' + figureFormat, bbox_inches='tight', format=figureFormat, dpi=dpi)
			plt.close()

		else:
			plt.show()

	if figures is not None:
		return figures


def plotScoresInteractive(dataTrue, pcaModel, colourBy, components=[1, 2], alpha=0.05, withExclusions=False):
	"""
	Interactively visualise PCA scores (coloured by a given sampleMetadata field, and for a given pair of components) with plotly, provides tooltips to allow identification of samples.
	
	:param Dataset dataTrue: Dataset
	:param PCA object pcaModel: PCA model object (scikit-learn based)
	:param str colourBy: **sampleMetadata** field name to of which values to colour samples by
	:param list components: List of two integers, components to plot
	:param float alpha: Significance value for plotting Hotellings ellipse
	:param bool withExclusions: If ``True``, only report on features and samples not masked by the sample and feature masks; must match between data and pcaModel
	"""

	import datetime

	# Check inputs
	if not isinstance(dataTrue, Dataset):
		raise TypeError('dataTrue must be an instance of nPYc.Dataset')
		
	if not isinstance(pcaModel, ChemometricsPCA):
		raise TypeError('PCAmodel must be a ChemometricsPCA object')

	values = pcaModel.scores
	ns, nc = values.shape
	sampleMetadata = dataTrue.sampleMetadata.copy()

	if colourBy not in sampleMetadata.columns:
		raise ValueError('colourBy must be a column in dataTrue.sampleMetadata')

	if (not all(isinstance(item, int) for item in components)) | (len(components) > 2):
		raise TypeError('components must be a list of two integer values')

	if not all(item <= nc for item in components):
		raise ValueError('integer values in components must not exceed the number of components in the model')

	components = [component - 1 for i, component in enumerate(components)] # Reduce components by one (account for python indexing)

	# Preparation
	if withExclusions:
		sampleMetadata = sampleMetadata.loc[dataTrue.sampleMask]
		sampleMetadata.reset_index(drop=True, inplace=True)
		
	if hasattr(pcaModel, '_npyc_dataset_shape'):
		if pcaModel._npyc_dataset_shape['NumberSamples'] != sampleMetadata.shape[0]:
			raise ValueError('Data dimension mismatch: Number of samples and features in the nPYc Dataset do not match'
							 'the numbers present when PCA was fitted. Verify if withExclusions argument is matching.')
	else:
		raise ValueError('Fit a PCA model beforehand using exploratoryAnalysisPCA.')
	
	classes = sampleMetadata[colourBy]
	hovertext = sampleMetadata['Sample File Name'].str.cat(classes.astype(str), sep='; ' + colourBy + ': ') # Save text to show in tooltips
	data = []

	# Ensure all values in column have the same type

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
	plotnans = classes.isnull().values
	if sum(plotnans != 0):
		NaNplot = go.Scatter(
			x = values[plotnans==True, components[0]],
			y = values[plotnans==True, components[1]],
			mode = 'markers',
			marker = dict(
				color = 'rgb(180, 180, 180)',
				symbol = 'circle',
				),
			text = hovertext[plotnans==True],
			hoverinfo = 'text',
			showlegend = False
			)
		data.append(NaNplot)

	# Plot numeric values with a colorbar
	if classes.dtype in (int, float):
		CLASSplot = go.Scatter(
			x = values[plotnans==False, components[0]],
			y = values[plotnans==False, components[1]],
			mode = 'markers',
			marker = dict(
				colorscale = 'Portland',
				color = classes[plotnans==False],
				symbol = 'circle',
				showscale = True
				),
			text = hovertext[plotnans==False],
			hoverinfo = 'text',
			showlegend = False
			)

	# Plot categorical values by unique groups
	else:
		uniq, indices = numpy.unique(classes, return_inverse=True)
		CLASSplot = go.Scatter(
			x = values[plotnans==False, components[0]],
			y = values[plotnans==False, components[1]],
			mode = 'markers',
			marker = dict(
				colorscale = 'Portland',
				color = indices[plotnans==False],
				symbol = 'circle',
				),
			text = hovertext[plotnans==False],
			hoverinfo = 'text',
			showlegend = False
			)

	data.append(CLASSplot)
	

	hotelling_ellipse = pcaModel.hotelling_T2(comps=numpy.array([components[0], components[1]]), alpha=alpha)

	layout = {
		'shapes' : [
			{
			'type': 'circle',
			'xref': 'x',
			'yref': 'y',
			'x0': 0 - hotelling_ellipse[0],
			'y0': 0 - hotelling_ellipse[1],
			'x1': 0 + hotelling_ellipse[0],
			'y1': 0 + hotelling_ellipse[1],
			}
		],
		'xaxis' : dict(
			title = 'PC' + str(components[0]+1) + ' (' + '{0:.2f}'.format(pcaModel.modelParameters['VarExpRatio'][components[0]] * 100) + '%)'
			),
		'yaxis' : dict(
			title = 'PC' + str(components[1]+1) + ' (' + '{0:.2f}'.format(pcaModel.modelParameters['VarExpRatio'][components[1]] * 100) + '%)'
			),
		'title' : 'Coloured by ' + colourBy,
		'legend' : dict(
			yanchor='middle',
			xanchor='right'
			),
		'hovermode' : 'closest'
	}


	figure = go.Figure(data=data, layout=layout)

	return figure


def plotLoadingsInteractive(dataTrue, pcaModel, component=1, withExclusions=False):
	"""
	Interactively visualise PCA loadings (for a given pair of components) with plotly, provides tooltips to allow identification of features.

	For MS data, plots RT vs. mz; for NMR plots ppm vs spectral intensity. Plots are coloured by the weight of the loadings.
	
	:param Dataset dataTrue: Dataset
	:param ChemometricsPCA pcaModel: PCA model object (scikit-learn based)
	:param int component: Component(s) to plot (one component (int) or list of two integers)
	:param bool withExclusions: If ``True``, only report on features and samples not masked by the sample and feature masks; must match between data and pcaModel
	"""

	# Separate one or two components plot
	if isinstance(component, list):
		multiPC = True
	else:
		multiPC = False

	# Check inputs
	if not isinstance(dataTrue, Dataset):
		raise TypeError('dataTrue must be an instance of nPYc.Dataset')
		
	if not isinstance(pcaModel, ChemometricsPCA):
		raise TypeError('PCAmodel must be a ChemometricsPCA object')

	# Preparation
	nc = pcaModel.scores.shape[1]

	if withExclusions:
		dataMasked = copy.deepcopy(dataTrue)
		dataMasked.applyMasks()
	else:
		dataMasked = dataTrue

	featureMetadata = copy.deepcopy(dataMasked.featureMetadata)

	if hasattr(pcaModel, '_npyc_dataset_shape'):
		if pcaModel._npyc_dataset_shape['NumberFeatures'] != featureMetadata.shape[0]:
			raise ValueError('Data dimension mismatch: Number of samples and features in the nPYc Dataset do not match'
							 'the numbers present when PCA was fitted. Verify if withExclusions argument is matching.')
	else:
		raise ValueError('Fit a PCA model beforehand using exploratoryAnalysisPCA.')


	# check single PC
	if not multiPC:
		if not isinstance(component, int):
			raise TypeError('component must be a single integer value')

		component = component - 1  # Reduce component by one (account for python indexing

		if component >= nc:
			raise ValueError('integer value in component must not exceed the number of components in the model')

		# Set up colour and tooltip values
		cVect = pcaModel.loadings[component, :]
		W_str = ["%.4f" % i for i in cVect]  # Format text for tooltips
		maxcol = numpy.max(abs(cVect))


	# check multi PC
	else:
		if ((not all(isinstance(item, int) for item in component)) | (len(component) > 2)):
			raise TypeError('component must be a list of two integer values')

		component = [cpt - 1 for i, cpt in enumerate(component)]  # Reduce component by one (account for python indexing)

		if not all(item < nc for item in component):
			raise ValueError('integer values in component must not exceed the number of components in the model')

		# Set up tooltip values
		cVectPC1 = pcaModel.loadings[component[0], :]
		cVectPC2 = pcaModel.loadings[component[1], :]
		PC1_id = [component[0] + 1] * cVectPC1.shape[0]
		PC2_id = [component[1] + 1] * cVectPC2.shape[0]
		WPC1_str = ["%.4f" % i for i in cVectPC1]  # Format text for tooltips first PC
		WPC2_str = ["%.4f" % i for i in cVectPC2]  # Format text for tooltips second PC

	# Set up
	data = []


	# Plot single PC
	if not multiPC:

		# For MS data
		if hasattr(featureMetadata, 'Retention Time'):

			hovertext = ["Feature: %s; W: %s" % i for i in zip(featureMetadata['Feature Name'], W_str)] # Text for tooltips

			# Convert cVect to a value between 0.1 and 1 - to set the alpha of each point relative to loading weight
			alphas = (((abs(cVect) - numpy.min(abs(cVect))) * (1 - 0.2)) / (maxcol - numpy.min(abs(cVect)))) + 0.2

			LOADSplot = go.Scatter(
				x = featureMetadata['Retention Time'],
				y = featureMetadata['m/z'],
				mode = 'markers',
				marker = dict(
					colorscale = 'RdBu',
					cmin = -maxcol,
					cmax = maxcol,
					color = cVect,
					opacity = alphas,
					showscale = True,
					),
				text = hovertext,
				hoverinfo = 'x, y, text',
				showlegend = False
				)

			data.append(LOADSplot)
			xReverse = True
			Xlabel = 'Retention Time'
			Ylabel = 'm/z'


		# For NMR data
		elif hasattr(featureMetadata, 'ppm'):

			Xvals = featureMetadata['ppm']
			hovertext = ["ppm: %.4f; W: %s" % i for i in zip(featureMetadata['ppm'], W_str)] # Text for tooltips

			# Bar starts at minimum spectral intensity
			LOADSmin = go.Bar(
				x = Xvals,
				y = numpy.min(dataMasked.intensityData, axis=0),
	#			y = numpy.percentile(PCAmodel.intensityData, 1, axis=0),
				marker = dict(
					color = 'white'
					),
				hoverinfo = 'skip',
				showlegend = False
				)

			# Bar ends at maximum spectral intensity, bar for each feature coloured by loadings weight
			LOADSmax = go.Bar(
				x = Xvals,
				y = numpy.max(dataMasked.intensityData, axis=0),
	#			y = numpy.percentile(PCAmodel.intensityData, 99, axis=0),
				marker = dict(
					colorscale = 'RdBu',
					cmin = -maxcol,
					cmax = maxcol,
					color = cVect,
					showscale = True,
					),
				text = hovertext,
				hoverinfo = 'text',
				showlegend = False
				)

			# Add line for median spectral intensity
			LOADSline = go.Scatter(
				x = Xvals,
				y = numpy.median(dataMasked.intensityData, axis=0),
				mode = 'lines',
				line = dict(
					color = 'black',
					width = 1
					),
				hoverinfo = 'skip',
				showlegend = False
			)

			data.append(LOADSmin)
			data.append(LOADSmax)
			data.append(LOADSline)
			xReverse = 'reversed'
			Xlabel = chr(948)+ '1H'
			Ylabel = 'Intensity'


		# Other data
		else:
			# X axis is PC loading, Y axis is ordered features
			sortOrder = numpy.argsort(pcaModel.loadings[component, :])
			Yvals     = list(range(pcaModel.loadings.shape[1], 0, -1))
			W_str = numpy.array(W_str)
			W_str = W_str[sortOrder]
			
			hovertext = ["Feature: %s; W: %s" % i for i in zip(featureMetadata['Feature Name'][sortOrder], W_str)]  # Text for tooltips

			LOADSplot = go.Scatter(
				x=pcaModel.loadings[component, sortOrder],
				y=Yvals,
				mode='markers',
				text=hovertext,
				hoverinfo='text',
				showlegend=False
			)

			data.append(LOADSplot)
			xReverse = True
			Xlabel = 'Principal Component ' + str(component + 1)
			Ylabel = 'Feature'


		# Add annotation
		layout = {
			'xaxis': dict(
				title=Xlabel,
				autorange=xReverse
			),
			'yaxis': dict(
				title=Ylabel
			),
			'title': 'Loadings for PC ' + str(component + 1),
			'hovermode': 'closest',
			'bargap': 0,
			'barmode': 'stack'
		}


	# Plot multi PC
	else:

		# NMR doesn't have Feature Name
		if hasattr(featureMetadata, 'ppm'):
			featureMetadata['Feature Name'] = ["%.4f" % i for i in featureMetadata['ppm']]

		hovertext = ["Feature: %s; W PC%s: %s; W PC%s: %s" % i for i in zip(featureMetadata['Feature Name'], PC1_id, WPC1_str, PC2_id, WPC2_str)]  # Text for tooltips

		LOADSplot = go.Scatter(
			x=pcaModel.loadings[component[0], :],
			y=pcaModel.loadings[component[1], :],
			mode='markers',
			text=hovertext,
			hoverinfo='text',
			showlegend=False
		)

		data.append(LOADSplot)
		Xlabel = 'Principal Component ' + str(component[0] + 1)
		Ylabel = 'Principal Component ' + str(component[1] + 1)

		# Add annotation
		layout = {
			'xaxis': dict(
				title=Xlabel
			),
			'yaxis': dict(
				title=Ylabel
			),
			'title': 'Loadings for PC ' + str(component[0] + 1) + ' and ' + str(component[1] + 1),
			'hovermode': 'closest',
			'bargap': 0,
			'barmode': 'stack'
		}


	figure = go.Figure(data=data, layout=layout)

	return figure


def plotMetadataDistribution(sampleMetadata, valueType, figures=None, savePath=None, figureFormat='png', dpi=72, figureSize=(11,7)):
	"""
	Plot the distribution of a set of data, e.g., sampleMetadata fields. Plots a bar chart for categorical data, or a histogram for continuous data.

	:param sampleMetadata: Set of measurements/groupings associated with each sample, note can contain multiple columns, but they must be of one **valueType**
	:type sampleMetadata: dataset.sampleMetadata
	:param str valueType: Type of data contained in **sampleMetadata**, one of ``continuous``, ``categorical`` or ``date``
	:param dict figures: If not ``None``, saves location of each figure for output in html report (see multivariateReport.py)
	"""
	import math

	# Check inputs
	if not isinstance(valueType, str) & (valueType in {'categorical', 'continuous', 'date'}):
		raise ValueError('valueType must be == ' + str({'categorical', 'continuous', 'date'}))

	# Set up for plotting in subplot figures 1x2
	nax = 3 # number of axis per figure
	nv = sampleMetadata.shape[1]
	nf = math.ceil(nv/nax)
	plotNo = 0
	field = sampleMetadata.columns

	# Plot
	for figNo in range(nf):

		fig, axIXs = plt.subplots(1, nax, figsize=(figureSize[0], figureSize[1]/nax), dpi=dpi)

		for axNo in range(len(axIXs)):

			if plotNo >= nv:
				axIXs[axNo].axis('off')

			# Continuous plot histogram
			elif valueType == 'continuous':

				pandas.DataFrame.hist(sampleMetadata, column=field[plotNo], ax=axIXs[axNo])
				axIXs[axNo].set_ylabel('Count')
				axIXs[axNo].set_title(field[plotNo])

			# Categorical plot bar
			elif valueType == 'categorical':

				# Define colors (gray for NaNs, cycle through tab10 otherwise)
				classes = sampleMetadata[field[plotNo]].copy()
				nans = [i for i, x in enumerate(classes) if x in {'nan', 'NaN', 'NaT', '', 'NA'}]
				nonans = [i for i, x in enumerate(classes) if x not in {'nan', 'NaN', 'NaT', '', 'NA'}]
				colors = []
				labels = []
				counts = []
				if nans:
					classes[nans] = 'NA'
					colors.append('#D3D3D3')
					labels.append('NA')
					counts.append(len(nans))
				temp = classes[nonans].value_counts()
				temp.sort_index(inplace=True)

				ix=0
				cmap = plt.cm.get_cmap('tab10')

				for i in temp.index:
					colors.append(rgb2hex(cmap(ix)[:3]))
					labels.append(i)
					counts.append(temp[i])
					ix += 1
					if(ix%10==0):
						ix=0

				# If 4 or less classes plot as pie chart
				if len(counts) <= 4:
					axIXs[axNo].pie(counts, labels=labels, colors=colors, labeldistance=1.05)
					x0,x1 = axIXs[axNo].get_xlim()
					y0,y1 = axIXs[axNo].get_ylim()
					axIXs[axNo].set_aspect(abs(x1-x0)/abs(y1-y0))
					axIXs[axNo].set_ylabel('')

				# Else plot bar chart
				else:
					axIXs[axNo].bar(numpy.arange(len(counts)), counts, align='center', color=colors, tick_label=labels)
					axIXs[axNo].set_xticklabels(axIXs[axNo].xaxis.get_majorticklabels(), rotation=90)
					axIXs[axNo].set_ylabel('Count')

				axIXs[axNo].set_title(field[plotNo])

			# Date
			elif valueType == 'date':
				
				try:
					xtime = mdates.date2num(sampleMetadata[field[plotNo]].values)
					axIXs[axNo].hist(xtime, bins=20)
					locator = mdates.AutoDateLocator()
					axIXs[axNo].xaxis.set_major_locator(locator)
					axIXs[axNo].xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%y %H:%M'))#AutoDateFormatter(locator))
					axIXs[axNo].xaxis.set_tick_params(rotation=90)
					axIXs[axNo].grid()
	
					axIXs[axNo].set_ylabel('Count')
					axIXs[axNo].set_title(field[plotNo])					
				except:
					pass

			# Advance plotNo
			plotNo = plotNo+1


		if savePath:
			if figures is not None:
				figures['metadataDistribution_' + valueType + str(figNo)] = savePath + 'metadataDistribution_' + valueType + str(figNo) + '.' + figureFormat

			plt.savefig(savePath + 'metadataDistribution_' + valueType + str(figNo) + '.' + figureFormat, bbox_inches='tight', format=figureFormat, dpi=dpi)
			plt.close()
		else:
			plt.show()

	if figures is not None:
		return figures


def _shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
	'''
	From Paul H at Stack Overflow
	http://stackoverflow.com/questions/7404116/defining-the-midpoint-of-a-colormap-in-matplotlib
	Function to offset the "center" of a colormap. Useful for
	data with a negative min and positive max and you want the
	middle of the colormap's dynamic range to be at zero

	Input
	-----
	  cmap : The matplotlib colormap to be altered
	  start : Offset from lowest point in the colormap's range.
		  Defaults to 0.0 (no lower ofset). Should be between
		  0.0 and `midpoint`.
	  midpoint : The new center of the colormap. Defaults to
		  0.5 (no shift). Should be between 0.0 and 1.0. In
		  general, this should be  1 - vmax/(vmax + abs(vmin))
		  For example if your data range from -15.0 to +5.0 and
		  you want the center of the colormap at 0.0, `midpoint`
		  should be set to  1 - 5/(5 + 15)) or 0.75
	  stop : Offset from highets point in the colormap's range.
		  Defaults to 1.0 (no upper ofset). Should be between
		  `midpoint` and 1.0.
	'''
	cdict = {
		'red': [],
		'green': [],
		'blue': [],
		'alpha': []
	}

	# regular index to compute the colors
	reg_index = numpy.linspace(start, stop, 257)

	# shifted index to match the data
	shift_index = numpy.hstack([
		numpy.linspace(0.0, midpoint, 128, endpoint=False),
		numpy.linspace(midpoint, 1.0, 129, endpoint=True)
	])

	for ri, si in zip(reg_index, shift_index):
		r, g, b, a = cmap(ri)

		cdict['red'].append((si, r, r))
		cdict['green'].append((si, g, g))
		cdict['blue'].append((si, b, b))
		cdict['alpha'].append((si, a, a))

	newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
	plt.register_cmap(cmap=newcmap)

	return newcmap