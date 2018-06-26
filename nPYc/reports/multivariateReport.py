# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 13:33:59 2016

@author: cs401
"""
import os
import numpy
import pandas
from collections import OrderedDict
import matplotlib.pyplot as plt
import seaborn as sns
import copy
from .._toolboxPath import toolboxPath
from ..objects import Dataset
from pyChemometrics.ChemometricsPCA import ChemometricsPCA
from ..multivariate.multivariateUtilities import pcaSignificance, metadataTypeGrouping
from ..plotting._multivariatePlotting import plotMetadataDistribution, plotScree, plotScores, plotLoadings, plotOutliers
from ..utilities._internal import _copyBackingFiles as copyBackingFiles
from ..enumerations import AssayRole, SampleType
import re
import numbers
import shutil
from IPython.display import display

from ..__init__ import __version__ as version

def multivariateQCreport(dataTrue, pcaModel, reportType='all', withExclusions=False, biologicalMeasurements=None, dModX_criticalVal=None, dModX_criticalVal_type=None, scores_criticalVal=None, kw_threshold=0.05, r_threshold=0.3, hotellings_alpha=0.05, excludeFields=None, destinationPath=None):
	"""
	PCA based analysis of a dataset. A PCA model is generated for the data object, then potential associations between the scores and any sample metadata determined by correlation (continuous data) or a Kruskal-Wallis test (categorical data).

	* **'analytical'** Reports on analytical qualities of the data only (as defined in the relevant SOP).
	* **'biological'** Reports on biological qualities of the data only (all columns in *sampleMetadata* except those defined as analytical or skipped in the SOP).
	* **'all'** Reports on all qualities of the data (all columns in *sampleMetadata* except those defined as skipped in the SOP).

	:param Dataset dataTrue: Dataset to report on
	:param ChemometricsPCA pcaModel: PCA model object (scikit-learn based)
	:param str reportType: Type of sample metadata to report on, one of ``analytical``, ``biological`` or ``all``
	:param bool withExclusions: If ``True``, only report on features and samples not masked by the sample and feature masks
	:param dict biologicalMeasurements: Dictionary of type of data contained in each biological sampleMetadata field. Keys are sampleMetadata column names, and values one of 'categorical', 'continuous', 'date'
	:param dModX_criticalVal: Samples with a value in DModX space exceeding this critical value are listed as potential outliers
	:type dModX_criticalVal: None or float
	:param dModX_criticalVal_type: Type of critical value in DModX, one of ``Fcrit`` or ``Percentile``
	:type dModX_criticalVal_type: None or str
	:param scores_criticalVal: Samples with a value in scores space exceeding this critical value are listed as potential outliers
	:type scores_criticalVal: None or float
	:param kw_threshold: Fields with a Kruskal-Willis p-value greater than this are not deemed to have a significant association with the PCA score
	:type kw_threshold: None or float
	:param r_threshold: Fields with a (absolute) correlation coefficient value less than this are not deemed to have a significant association with the PCA score
	:type r_threshold: None or float
	:param float hotellings_alpha: Alpha value for plotting the Hotelling's ellipse in scores plots (default = 0.05)
	:param excludeFields: If not None, list of sample metadata fields to be additionally excluded from analysis
	:type excludeFields: None or list
	:param destinationPath: If ``None`` plot interactively, otherwise save report to the path specified
	:type destinationPath: None or str
	"""

	# Check inputs
	if not isinstance(dataTrue, Dataset):
		raise TypeError('dataTrue must be an instance of nPYc.Dataset')

	if not isinstance(pcaModel, ChemometricsPCA):
		raise TypeError('PCA model must be an instance of pyChemometrics.ChemometricsPCA')

	if not isinstance(reportType, str) & (reportType in {'all', 'analytical', 'biological'}):
		raise ValueError('reportType must be == ' + str({'all', 'analytical', 'biological'}))

	if not isinstance(withExclusions, bool):
		raise TypeError('withExclusions must be a bool')

	if biologicalMeasurements is not None:
		if not isinstance(biologicalMeasurements, dict):
			raise TypeError('biologicalMeasurements must be a dictionary')
		temp = list(biologicalMeasurements.values())
		if any(val not in {'categorical','continuous','date'} for val in temp):
			raise ValueError('biologicalMeasurements values must be == ' + str({'categorical', 'continuous', 'date'}))

	if dModX_criticalVal is not None:
		if not isinstance(dModX_criticalVal, numbers.Number) & ((dModX_criticalVal < 1) & (dModX_criticalVal > 0)):
			raise ValueError('dModX_criticalVal must be a number in the range 0 to 1')			
		if dModX_criticalVal_type is None:
			raise ValueError('If dModX_criticalVal is specfied, specify dModX_criticalVal_type (must be == ' + str({'Fcrit', 'Percentile'}) + ')')
			
	if dModX_criticalVal_type is not None:
		if not isinstance(dModX_criticalVal_type, str) & (dModX_criticalVal_type in {'Fcrit', 'Percentile'}):
			raise ValueError('dModX_criticalVal_type must be == ' + str({'Fcrit', 'Percentile'}))

	if scores_criticalVal is not None:
		if not isinstance(scores_criticalVal, numbers.Number) & ((scores_criticalVal < 1) & (scores_criticalVal > 0)):
			raise ValueError('scores_criticalVal must be a number in the range 0 to 1')

	if kw_threshold is not None:
		if not isinstance(kw_threshold, numbers.Number) or kw_threshold < 0:
			raise ValueError('kw_threshold must be a positive number')

	if r_threshold is not None:
		if not isinstance(r_threshold, numbers.Number) or r_threshold < 0:
			raise ValueError('r_threshold must be a positive number')
			
	if not isinstance(hotellings_alpha, numbers.Number) & ((hotellings_alpha < 1) & (hotellings_alpha > 0)):
		 raise ValueError('hotellings_alpha must be a number in the range 0 to 1')

	if excludeFields is not None:
		if not isinstance(excludeFields, list):
			raise TypeError('excludeFields must be a list of column headers from data.sampleMetadata')

	if destinationPath is not None:
		if not isinstance(destinationPath, str):
			raise TypeError('destinationPath must be a string')


	# Create directory to save destinationPath
	if destinationPath:

		saveDir = os.path.join(destinationPath, 'graphics', 'report_multivariate' + reportType.capitalize())

		# If directory exists delete directory and contents
		if os.path.exists(saveDir):
			shutil.rmtree(saveDir)

		# Create directory to save destinationPath
		os.makedirs(saveDir)

	else:
		saveAs = None

	# Filter dataset if required
	data = copy.deepcopy(dataTrue)

	if withExclusions:
		data.applyMasks()

	if hasattr(pcaModel, '_npyc_dataset_shape'):
		if pcaModel._npyc_dataset_shape['NumberSamples'] != data.intensityData.shape[0] \
				or pcaModel._npyc_dataset_shape['NumberFeatures'] != data.intensityData.shape[1]:
			raise ValueError('Data dimension mismatch: Number of samples and features in the nPYc Dataset do not match'
							 'the numbers present when PCA was fitted. Verify if withExclusions argument is matching.')
	else:
		raise ValueError('Fit a PCA model beforehand using exploratoryAnalysisPCA.')

	# Set up template item and save required info
	figuresQCscores = OrderedDict()
	figuresLoadings = OrderedDict()
	figuresCORscores = OrderedDict()
	figuresKWscores = OrderedDict()
	figuresOTHERscores = OrderedDict()
	item = dict()
	item['ReportType'] = reportType.title()
	item['Name'] = data.name
	ns, nv = data.intensityData.shape
	item['Nfeatures'] = str(nv)
	item['Nsamples'] = str(ns)
	SPmask = (data.sampleMetadata['SampleType'] == SampleType.StudyPool) & (data.sampleMetadata['AssayRole'] == AssayRole.PrecisionReference)
	item['SPcount'] = str(sum(SPmask))
	SSmask = (data.sampleMetadata['SampleType'] == SampleType.StudySample) & (data.sampleMetadata['AssayRole'] == AssayRole.Assay)
	item['SScount'] = str(sum(SSmask))
	ERmask = (data.sampleMetadata['SampleType'] == SampleType.ExternalReference) & (data.sampleMetadata['AssayRole'] == AssayRole.PrecisionReference)
	item['ERcount'] = str(sum(ERmask))
	item['OTHERcount'] = str(ns - sum(SSmask) - sum(SPmask) - sum(ERmask))
	data.sampleMetadata.loc[~SSmask & ~SPmask & ~ERmask, 'Plot Sample Type'] = 'Sample'
	data.sampleMetadata.loc[SSmask, 'Plot Sample Type'] = 'Study Sample'
	data.sampleMetadata.loc[SPmask, 'Plot Sample Type'] = 'Study Pool'
	data.sampleMetadata.loc[ERmask, 'Plot Sample Type'] = 'External Reference'

	item['Normalisation'] = str(dataTrue.Normalisation)

	special_scaling = dict([(0, 'mc'), (1, 'uv'), (0.5, 'par')])
	if pcaModel.scaler.scale_power in special_scaling:
		item['Scaling'] = special_scaling[pcaModel.scaler.scale_power]
	else:
		item['Scaling'] = str(pcaModel.scaler.scale_power)

	# Fields to plot
	includeForPlotting = {}

	if reportType in {'analytical', 'all'}:
		includeForPlotting.update(data.Attributes['analyticalMeasurements'])

	if reportType in {'biological', 'all'}:

		if biologicalMeasurements is not None:
			includeForPlotting.update(biologicalMeasurements)

		else:
			temp = [val for val in data.sampleMetadata.columns if val not in data.Attributes['analyticalMeasurements']]

			# Create dictionary with key and type (categorical/continuous etc) for each biological parameter field
			for plotdata in temp:
				out = metadataTypeGrouping(data.sampleMetadata[plotdata], sampleGroups=data.sampleMetadata['Plot Sample Type'])
				includeForPlotting[plotdata] = out

	# Fields not to plot
	excludeFromPlotting = data.Attributes['excludeFromPlotting']
	if excludeFields != None:
		excludeFromPlotting.append(excludeFields)

	# Remove fields either marked not to plot or not present in sampleMetadata
	includeForPlotting = {i:includeForPlotting[i] for i in includeForPlotting if ((i in data.sampleMetadata.columns) and (i not in excludeFromPlotting))}
 
	# Generate DataFrame of only data for plotting
	dataForPlotting = copy.deepcopy(data.sampleMetadata[list(includeForPlotting.keys())])

	# Check for data integrity
	for plotdata in includeForPlotting.keys():

		# Check all values in column have the same type
		myset = set(list(type(data.sampleMetadata[plotdata][i]) for i in range(ns)))

		if len(myset) == 1:
			pass

		# elif all((my == pandas._libs.tslib.NaTType or my == pandas._libs.tslib.Timestamp) for my in myset):
		# 	pass

		else:
			print(plotdata)
			print(myset)
			print('Ensure datatype of all entries in ' + plotdata + ' is consistent')
			return

		# Change type if uniform, uniformBySampleType or unique (and categorical) - do not plot these
		out = metadataTypeGrouping(data.sampleMetadata[plotdata], sampleGroups=data.sampleMetadata['Plot Sample Type'])
		if out in {'uniform', 'uniformBySampleType', 'unique'}:
			includeForPlotting[plotdata] = out

		# Remove unwanted characters from column titles
		dataForPlotting.rename(columns={plotdata: plotdata.translate({ord(c): " " for c in "!@#$%^&*()[]{};:,./<>?\|`~-=+_"}).strip()}, inplace=True)
		# Correct duplicate column names
		cols = pandas.Series(dataForPlotting.columns)
		for dup in dataForPlotting.columns[dataForPlotting.columns.duplicated()].unique():
			cols.loc[dataForPlotting.columns.get_loc(dup)] = [dup + '.' + str(d_idx) if d_idx != 0 else dup for d_idx in range(dataForPlotting.columns.get_loc(dup).sum())]
		dataForPlotting.columns = cols

	nc = pcaModel.ncomps
	
	item['Ncomponents'] = str(nc)	
	if dModX_criticalVal is not None:
		if dModX_criticalVal_type == 'Fcrit':
			item['dModX_criticalVal'] = dModX_criticalVal_type + ' (' + str(dModX_criticalVal) + ')'
		else:
			item['dModX_criticalVal'] = 'Q' + str(100-dModX_criticalVal*100)
	else:
		item['dModX_criticalVal'] = 'None'
		
	if scores_criticalVal is not None:
		item['scores_criticalVal'] = 'Q' + str(100-scores_criticalVal*100)	
	else:
		item['scores_criticalVal'] = 'None'
		
	# Add check for if 2nd component added for plotting purposes only
	if nc==2:
		if ( (pcaModel.cvParameters['Q2X_Scree'][1] - pcaModel.cvParameters['Q2X_Scree'][0])/pcaModel.cvParameters['Q2X_Scree'][0] < pcaModel.cvParameters['stopping_condition'] ):
			item['Ncomponents_optimal'] = '1'

	# Datast summary
	if destinationPath is None:
		print('\033[1m' + 'Dataset' + '\033[0m')
		print('\nOriginal data consists of ' + item['Nsamples'] + ' samples and ' + item['Nfeatures'] + ' features')
		print('\t' + item['SScount'] + ' Study Samples')
		print('\t' + item['SPcount'] + ' Study Pool Samples')
		print('\t' + item['ERcount'] + ' External Reference Samples')
		print('\t' + item['OTHERcount'] + ' Other Samples')
		
		print('\033[1m' + '\nPCA Analysis' + '\033[0m')
		print('\nPCA Model Parameters')
		print('\tNormalisation method: ' + item['Normalisation'])
		print('\tScaling: ' + item['Scaling'])
		print('\tNumber of components: ' + item['Ncomponents'])
		
		if 'Ncomponents_optimal' in item:
			print('\t' + '\033[1m' + 'IMPORTANT NOTE: Optimal number of components: 1 (second component added for plotting purposes)' + '\033[0m')
		
		print('\tCritical value for flagging outliers in DmodX space: ' + item['dModX_criticalVal'])
		print('\tCritical value for flagging outliers in scores space: ' + item['scores_criticalVal'])
		
		print('\033[1m' + '\nPCA QC Outputs' + '\033[0m')


	# Scree plot
	if destinationPath:
		item['PCA_screePlot'] = os.path.join(saveDir, item['Name'] + '_PCAscreePlot.' + data.Attributes['figureFormat'])
		saveAs = item['PCA_screePlot']
		item['PCA_var_exp'] = pcaModel.modelParameters['VarExpRatio']
	else:
		print('\nFigure 1: PCA scree plot of variance explained by each component (cumulative)')

	plotScree(pcaModel.cvParameters['R2X_Scree'],
		Q2=pcaModel.cvParameters['Q2X_Scree'],
		xlabel='Component',
		ylabel='Percentage variance (cumulative)',
		savePath=saveAs,
		figureFormat=data.Attributes['figureFormat'],
		dpi=data.Attributes['dpi'],
		figureSize=data.Attributes['figureSize'])


	# Scores plot (coloured by sample type)
	temp = dict()
	if destinationPath:
		temp['PCA_scoresPlot'] = os.path.join(saveDir, item['Name'] + '_PCAscoresPlot_')
		saveAs = temp['PCA_scoresPlot']
	else:
		print('\n\nFigure 2: PCA scores plots coloured by sample type.')

	figuresQCscores = plotScores(pcaModel,
		classes=data.sampleMetadata['Plot Sample Type'],
		classType='Plot Sample Type',
		title='Sample Type',
		figures=figuresQCscores,
		alpha=hotellings_alpha,
		savePath=saveAs,
		figureFormat=data.Attributes['figureFormat'],
		dpi=data.Attributes['dpi'],
		figureSize=data.Attributes['figureSize'])

	for key in figuresQCscores:
		if os.path.join(destinationPath, 'graphics') in str(figuresQCscores[key]):
			figuresQCscores[key] = re.sub('.*graphics', 'graphics', figuresQCscores[key])

	item['QCscores'] = figuresQCscores


	# Calculate sum of scores across all PCs for each sample
	sumT = numpy.sum(numpy.absolute(pcaModel.scores), axis=1)

	# Scatter plot of summed scores distance from origin (strong outliers in PCA)
	if destinationPath:
		item['PCA_strongOutliersPlot'] = os.path.join(saveDir, item['Name'] + '_strongOutliersPlot.' + data.Attributes['figureFormat'])
		saveAs = item['PCA_strongOutliersPlot']
	else:
		print('\n\nFigure 3: Distribution in total distance from origin (scores space) by sample type.')

	if not 'Run Order' in data.sampleMetadata.columns:
		data.sampleMetadata['Run Order'] = data.sampleMetadata.index.values

	# Flag potential strong outliers (exceed outliers_criticalVal)
	if scores_criticalVal is not None:
		PcritPercentile = 100 - scores_criticalVal*100
		quantilesVals = numpy.percentile(sumT, [100 - scores_criticalVal*100])
		which_scores_outlier = (sumT >= quantilesVals)
		item['Noutliers_strong'] = str(sum(which_scores_outlier))

	else:
		PcritPercentile = None
		which_scores_outlier = numpy.zeros(sumT.shape, dtype=bool)


	plotOutliers(sumT,
		data.sampleMetadata['Run Order'],
		sampleType=data.sampleMetadata['Plot Sample Type'],
		addViolin=True,
		ylabel='Summed distance from origin (all PCs)',
		PcritPercentile=PcritPercentile,
		savePath=saveAs,
		figureFormat=data.Attributes['figureFormat'],
		dpi=data.Attributes['dpi'],
		figureSize=data.Attributes['figureSize'])
	
	if (scores_criticalVal is not None) & (destinationPath is None):
		print('\nExcluding samples with total distance from origin exceeding the ' + item['scores_criticalVal'] + ' limit would result in ' + item['Noutliers_strong'] + ' exclusions.')
		

	# Scatter plot of DmodX (moderate outliers in PCA)
	if destinationPath:
		item['PCA_modOutliersPlot'] = os.path.join(saveDir, item['Name'] + '_modOutliersPlot.' + data.Attributes['figureFormat'])
		saveAs = item['PCA_modOutliersPlot']
	else:
		print('\n\nFigure 4: Distribution in distance from model (DmodX) by sample type.')


	sample_dmodx_values = pcaModel.dmodx(data.intensityData)
	
	# Define defaults for plotting if no critical values specified by user
	PcritPercentile = None
	Fcrit = pcaModel._dmodx_fcrit(data.intensityData, alpha = 0.05)
	FcritAlpha = 0.05 
	which_dmodx_outlier = numpy.zeros(sample_dmodx_values.shape, dtype=bool)
	
	# Flag potential moderate outliers (exceed critical value) 	
	if dModX_criticalVal is not None:
		if dModX_criticalVal_type == 'Fcrit':
			dModX_threshold = pcaModel._dmodx_fcrit(data.intensityData, alpha = dModX_criticalVal)
			Fcrit = dModX_threshold
			FcritAlpha = dModX_criticalVal

		else:
			dModX_threshold = numpy.percentile(sample_dmodx_values, [100 - dModX_criticalVal*100])
			PcritPercentile = 100 - dModX_criticalVal*100

		which_dmodx_outlier = (sample_dmodx_values >= dModX_threshold)
		
		item['Noutliers_moderate'] = str(sum(which_dmodx_outlier))

	
	plotOutliers(sample_dmodx_values,
		data.sampleMetadata['Run Order'],
		sampleType=data.sampleMetadata['Plot Sample Type'],
		addViolin=True,
		Fcrit=Fcrit,
		FcritAlpha=FcritAlpha,
		PcritPercentile=PcritPercentile,
		ylabel='DmodX',
		savePath=saveAs,
		figureFormat=data.Attributes['figureFormat'],
		dpi=data.Attributes['dpi'],
		figureSize=data.Attributes['figureSize'])

	if (dModX_criticalVal is not None) & (destinationPath is None):
		print('\nExcluding samples with DmodX exceeding the ' + item['dModX_criticalVal'] + ' limit would result in ' + item['Noutliers_moderate'] + ' exclusions.')

	# Total number of outliers		
	if sum(which_scores_outlier | which_dmodx_outlier) > 0:
		outliers = (which_scores_outlier | which_dmodx_outlier)
		
		item['Noutliers_total'] = str(sum(outliers))
		item['Outliers_total_details'] =  data.sampleMetadata[['Sample File Name']][outliers]
		item['Outliers_total_details']['DModX Outlier'] = which_dmodx_outlier[outliers]
		item['Outliers_total_details']['Scores Outlier'] = which_scores_outlier[outliers]
		
		if destinationPath is None:
			print('\nExcluding outliers (as specified) would result in ' + item['Noutliers_total'] + ' exclusions.')
			display(item['Outliers_total_details'])
			print('\n')

	# Loadings plot
	if destinationPath:
		temp['PCA_loadingsPlot'] = os.path.join(saveDir, item['Name'] + '_PCAloadingsPlot_')
		saveAs = temp['PCA_loadingsPlot']
	else:
		print('\n\nFigure 5: PCA loadings plots.')

	figuresLoadings = plotLoadings(pcaModel,
		data,
		title='',
		figures=figuresLoadings,
		savePath=saveAs,
		figureFormat=data.Attributes['figureFormat'],
		dpi=data.Attributes['dpi'],
		figureSize=data.Attributes['figureSize'])

	for key in figuresLoadings:
		if os.path.join(destinationPath, 'graphics') in str(figuresLoadings[key]):
			figuresLoadings[key] = re.sub('.*graphics', 'graphics', figuresLoadings[key])

	item['loadings'] = figuresLoadings


	# Plot metadata and assess potential association with PCA scores

	# Set up:
	if destinationPath:
		temp['metadataPlot'] = os.path.join(saveDir, item['Name'] + '_metadataPlot_')
		saveAs = temp['metadataPlot']
	else:
		print('\033[1m' + '\nDistribution of Values in each Metadata Field\n'+ '\033[0m')
		print('Figure 6: Distribution of values in each metadata field (plotted for fields with non-uniform values only).\n')

	# Plot distribution for each field and calculate measure of association to PCA scores (categorical/continuous only)
	valueType = list(includeForPlotting.values())
	allTypes = set(valueType)
	signif = numpy.full([nc,len(includeForPlotting)], numpy.nan)
	countKW = 0
	fieldsKW = []
	countKWfail = 0
	fieldsKWfail = []

	for eachType in allTypes:

		if eachType in {'continuous', 'categorical', 'date'}:

			figuresMetadataDist = OrderedDict()

			if destinationPath is None:
				print(eachType.title() + ' data.')

			# Find indices of instances of this type
			indices = [i for i, x in enumerate(valueType) if x == eachType]

			# Plot
			figuresMetadataDist = plotMetadataDistribution(dataForPlotting.iloc[:, indices],
				  eachType,
				  figures=figuresMetadataDist,
				  savePath=saveAs,
				  figureFormat=data.Attributes['figureFormat'],
				  dpi=data.Attributes['dpi'],
				  figureSize=data.Attributes['figureSize'])

			for key in figuresMetadataDist:
				if os.path.join(destinationPath, 'graphics') in str(figuresMetadataDist[key]):
					figuresMetadataDist[key] = re.sub('.*graphics', 'graphics', figuresMetadataDist[key])

			if eachType == 'continuous':
				item['metadataDistContinuous'] = figuresMetadataDist
			elif eachType == 'categorical':
				item['metadataDistCategorical'] = figuresMetadataDist
			else:
				item['metadataDistDate'] = figuresMetadataDist

			# Calculate metric of association between metadata and PCA score
			if eachType in {'continuous', 'categorical'}:
				
				for field in dataForPlotting.columns[indices]:
					out = pcaSignificance(pcaModel.scores, dataForPlotting[field], eachType)

					if out is not None:
						index = dataForPlotting.columns.get_loc(field)
						signif[:,index] = out

						if eachType == 'categorical':
							countKW = countKW+1
							fieldsKW.append(field)

					# Count the number of classes where KW cannot be calculated
					else:
						if eachType == 'categorical':
							countKWfail = countKWfail+1
							fieldsKWfail.append(field)


	fieldNames = dataForPlotting.columns

	item['Nmetadata'] = str(len(includeForPlotting))
	item['Ncorr'] = str(valueType.count('continuous'))
	item['Nkw'] = str(countKW)
	item['Ndate'] = str(valueType.count('date'))
	item['Ninsuf'] = str(countKWfail)
	item['Nuniform'] = str(valueType.count('uniform'))
	item['NuniformByType'] = str(valueType.count('uniformBySampleType'))
	item['Nunique'] = str(valueType.count('unique'))
	item['Nex'] = str(valueType.count('excluded'))
	item['r_threshold'] = str(r_threshold)
	item['kw_threshold'] = str(kw_threshold)

	if destinationPath is None:

		# Summarise results
		print('\033[1m' + '\n\nAssociation of PCA Scores with Metadata' + '\033[0m')
		print('\nCalculations Performed')
		print('\nTotal number of metadata fields: ' + item['Nmetadata'])
		print('\tNumber of fields where correlation to PCA scores calculated: ' + item['Ncorr'])
		print('\tNumber of fields where Kruskal-Wallis test between groups in PCA scores calculated: ' + item['Nkw'])
		print('\tNumber of fields with date values: ' + item['Ndate'])
		print('\tNumber of fields where insufficent sample numbers to estimate significance: ' + item['Ninsuf'])
		print('\tNumber of fields with uniform class for all samples: ' + item['Nuniform'])
		print('\tNumber of fields with uniform class for all samples with same sample type: ' + item['NuniformByType'])
		print('\tNumber of fields with unique non-numeric values for all samples in class: ' + item['Nunique'])
		print('\tNumber of fields excluded from calculations: ' + item['Nex'])

		print('\n\tCorrelation threshold for plotting: ' + item['r_threshold'])
		print('\tKruskal-Willis p-value threshold for plotting: ' + item['kw_threshold'])


	# Heatmap of results - correlation
	if destinationPath is None:
		print('\n\nFigure 7: Heatmap of correlation to PCA scores for suitable metadata fields.')
	if valueType.count('continuous') > 0:
		sigCor = numpy.full([nc*valueType.count('continuous'), 3], numpy.nan)
		index = [i for i, j in enumerate(valueType) if j == 'continuous']
		i=0
		for IX in index:
			sigCor[i:i+nc,0] = IX
			sigCor[i:i+nc,1] = numpy.arange(1,nc+1)
			sigCor[i:i+nc,2] = signif[:, IX]
			i=i+nc
		sigCor = pandas.DataFrame(sigCor, columns=['Field', 'PC', 'Correlation'])
		sigCor['Field'] = fieldNames[sigCor['Field'].values.astype('int')]

		sigCor = sigCor.pivot('Field','PC','Correlation')

		# plot heatmap
		with sns.axes_style("white"):
			plt.figure(figsize=data.Attributes['figureSize'], dpi=data.Attributes['dpi'])
			sns.heatmap(sigCor, annot=True, fmt='.3g', vmin=-1, vmax=1, cmap='RdBu_r')
			if destinationPath:
				item['sigCorHeatmap'] = os.path.join(saveDir, item['Name'] + '_sigCorHeatmap.' + data.Attributes['figureFormat'])
				plt.savefig(item['sigCorHeatmap'], bbox_inches='tight', format=data.Attributes['figureFormat'], dpi=data.Attributes['dpi'])
				plt.close()
			else:
				plt.show()
	else:
		if destinationPath is None:
			print('\n' + str(valueType.count('correlation')) + ' fields where correlation to PCA scores calculated.')

	# Heatmap of results - Kruskal-Wallis
	if destinationPath is None:
		print('\n\nFigure 8: Heatmap of Kruskal-Wallis Test against PCA scores for suitable metadata fields.')
	if countKW > 0:
		sigKru = numpy.full([nc*countKW, 3], numpy.nan)
		index = [dataForPlotting.columns.get_loc(field) for field in fieldsKW]
		i=0
		for IX in index:
			sigKru[i:i+nc,0] = IX
			sigKru[i:i+nc,1] = numpy.arange(1,nc+1)
			sigKru[i:i+nc,2] = signif[:, IX]
			i=i+nc
		sigKru = pandas.DataFrame(sigKru, columns=['Field', 'PC', 'Kruskal-Wallis p-value'])
		sigKru['Field'] = fieldNames[sigKru['Field'].values.astype('int')]

		sigKru = sigKru.pivot('Field','PC','Kruskal-Wallis p-value')

		# plot heatmap
		with sns.axes_style("white"):
			plt.figure(figsize=data.Attributes['figureSize'], dpi=data.Attributes['dpi'])
			sns.heatmap(sigKru, annot=True, fmt='.3g', vmin=0, vmax=1, cmap='OrRd_r')
			if destinationPath:
				item['sigKruHeatmap'] = os.path.join(saveDir, item['Name'] + '_sigKruHeatmap.' + data.Attributes['figureFormat'])
				plt.savefig(item['sigKruHeatmap'], bbox_inches='tight', format=data.Attributes['figureFormat'], dpi=data.Attributes['dpi'])
				plt.close()
			else:
				plt.show()
	else:
		if destinationPath is None:
			print('\n'+ str(valueType.count('KW')) + ' fields where Kruskal-Wallis test between groups in PCA scores calculated.')

	# Scores plots coloured by each available metadata, above thresholds if required and sorted by significance
	if destinationPath:
		saveAs = saveDir

	# Plots for continuous data fields (passing correlation threshold)
	item['Ncorr_passing'] = '0'
	if destinationPath is None:
		print('\n\nFigure 9: PCA scores plots coloured by metadata (significance by correlation).')
		
	if valueType.count('continuous') > 0:
		if r_threshold == 'None':
			r_threshold = numpy.min(abs(sigCor.values))
		item['Ncorr_passing'] = str(sum((abs(sigCor.values) >= r_threshold).any(axis=1)==True))
		if destinationPath is None:
			print('\n' + item['Ncorr_passing'] + ' fields where correlation coefficient to PCA scores exceeded threshold of ' + str(r_threshold))
		if (abs(sigCor.values) >= r_threshold).any():
			fields = sigCor.index[(abs(sigCor.values) >= r_threshold).any(axis=1)]
			figuresCORscores = _plotScoresLocal(dataForPlotting,
				   fields,
				   pcaModel,
				   'continuous',
				   data.name,
				   alpha=hotellings_alpha,
				   plotAssociation=sigCor,
				   r_threshold=r_threshold,
				   saveDir=saveAs,
				   figures=figuresCORscores,
				   figureFormat=data.Attributes['figureFormat'],
				   dpi=data.Attributes['dpi'],
				   figureSize=data.Attributes['figureSize'])

			if destinationPath is not None:
				for key in figuresCORscores:
					if os.path.join(destinationPath, 'graphics') in str(figuresCORscores[key]):
						figuresCORscores[key] = re.sub('.*graphics', 'graphics', figuresCORscores[key])
			item['CORscores'] = figuresCORscores
	else:
		if destinationPath is None:
			print('\n' + item['Ncorr_passing'] + ' fields where correlation coefficient to PCA scores exceeded threshold of ' + str(r_threshold))

	# Plots for catagorical data fields (passing Kruskal-Wallis threshold)
	item['Nkw_passing'] = '0'
	if destinationPath is None:
		print('\n\nFigure 10: PCA scores plots coloured by metadata (significance by Kruskal-Wallis).')
		
	if countKW > 0:
		if kw_threshold == 'None':
			kw_threshold = numpy.max(abs(sigKru.values))
		item['Nkw_passing'] = str(sum((sigKru.values <= kw_threshold).any(axis=1)==True))
		if destinationPath is None:
			print('\n' + item['Nkw_passing'] + ' fields where Kruskal-Wallis p-value against PCA scores exceeded threshold of ' + str(kw_threshold))
		if (sigKru.values <= kw_threshold).any():
			fields = sigKru.index[(sigKru.values <= kw_threshold).any(axis=1)]
			figuresKWscores = _plotScoresLocal(dataForPlotting,
				  fields,
				  pcaModel,
				  'categorical',
				  data.name,
				  alpha=hotellings_alpha,
				  plotAssociation=sigKru,
				  kw_threshold=kw_threshold,
				  saveDir=saveAs,
				  figures=figuresKWscores,
				  figureFormat=data.Attributes['figureFormat'],
				  dpi=data.Attributes['dpi'],
				  figureSize=data.Attributes['figureSize'])

			if destinationPath is not None:
				for key in figuresKWscores:
					if os.path.join(destinationPath, 'graphics') in str(figuresKWscores[key]):
						figuresKWscores[key] = re.sub('.*graphics', 'graphics', figuresKWscores[key])
			item['KWscores'] = figuresKWscores
	else:
		if destinationPath is None:
			print('\n' + item['Nkw_passing'] + ' fields where Kruskal-Wallis p-value against PCA scores exceeded threshold of ' + str(kw_threshold))

	# Plots for catagorical data fields (with insufficient numbers to test significance)
	if destinationPath is None:
		print('\n\nFigure 11: PCA scores plots coloured by metadata (insufficent sample numbers to estimate significance).')
		print('\n' + item['Ninsuf'] + ' fields where insufficent sample numbers to estimate significance.')
		
	if countKWfail > 0:
		
		# Create a dataframe with null significance values
		sigNone = numpy.full([nc*countKWfail, 3], numpy.nan)
		index = [dataForPlotting.columns.get_loc(field) for field in fieldsKWfail]	
		i=0
		for IX in index:
			sigNone[i:i+nc,0] = IX
			sigNone[i:i+nc,1] = numpy.arange(1,nc+1)
			i=i+nc
		sigNone = pandas.DataFrame(sigNone, columns=['Field', 'PC', 'Kruskal-Wallis p-value'])
		sigNone['Field'] = fieldNames[sigNone['Field'].values.astype('int')]
		sigNone = sigNone.pivot('Field','PC','Kruskal-Wallis p-value')
		
		figuresOTHERscores = _plotScoresLocal(dataForPlotting,
			fieldsKWfail,
			pcaModel,
			'categorical',
			data.name,
			alpha=hotellings_alpha,
			plotAssociation=sigNone,
			saveDir=saveAs,
			figures=figuresOTHERscores,
			figureFormat=data.Attributes['figureFormat'],
			dpi=data.Attributes['dpi'],
			figureSize=data.Attributes['figureSize'])

		if destinationPath is not None:
			for key in figuresOTHERscores:
				if os.path.join(destinationPath, 'graphics') in str(figuresOTHERscores[key]):
					figuresOTHERscores[key] = re.sub('.*graphics', 'graphics', figuresOTHERscores[key])
		item['OTHERscores'] = figuresOTHERscores

	# Generate html report
	if destinationPath: 

		# Make paths for graphics local not absolute for use in the HTML.
		for key in item:
			if os.path.join(destinationPath, 'graphics') in str(item[key]):
				item[key] = re.sub('.*graphics', 'graphics', item[key])

		# Generate report
		from jinja2 import Environment, FileSystemLoader

		env = Environment(loader=FileSystemLoader(os.path.join(toolboxPath(), 'Templates')))

		template = env.get_template('NPC_MultivariateReport.html')
		filename = os.path.join(destinationPath, data.name + '_report_multivariate' + reportType.capitalize() + '.html')
		f = open(filename,'w')
		f.write(template.render(item=item, version=version, graphicsPath='/report_multivariate' + reportType.capitalize()))
		f.close()

		copyBackingFiles(toolboxPath(), os.path.join(destinationPath, 'graphics'))

	return None


def _plotScoresLocal(data, metadata, pcaModel, classType, name, alpha=0.05, plotAssociation=None, r_threshold=None, kw_threshold=None, saveDir=None, figures=None, figureFormat='png', dpi=72, figureSize=(11, 7)):
	"""
	Local function to plot scores for each metadata field
	"""

	temp = dict()
	nc = pcaModel.scores.shape[1]

	if saveDir:
		temp['PCA_scoresPlot'] = os.path.join(saveDir, name + '_PCAscoresPlot_')
		saveAs = temp['PCA_scoresPlot']
	else:
		saveAs = None

	fieldNames = data.columns

	for plotdata in metadata:

		if saveDir is None:
			print('\n' + plotdata)

		# Find components with significance exceeding threshold
		if plotAssociation is None:
			sigLocal = None
		else:
			sigLocal = plotAssociation.loc[plotdata].values

		if r_threshold is not None:
			components = abs(sigLocal) >= r_threshold
		elif kw_threshold is not None:
			components = sigLocal <= kw_threshold
		else:
			components = numpy.ones([nc]).astype(bool)

		index = fieldNames.str.match(plotdata+'$')

		if index.any():
			plotScores(pcaModel,
					   classes=data.iloc[:,index].squeeze(),
					   classType=classType,
					   components=components,
					   alpha=alpha,
					   plotAssociation=sigLocal,
					   title=plotdata,
					   figures=figures,
					   savePath=saveAs,
					   figureFormat=figureFormat,
					   dpi=dpi,
					   figureSize=figureSize)
		else:
			print(plotdata + ' not present in sampleMetadata - check this!')

	if figures:
		return figures