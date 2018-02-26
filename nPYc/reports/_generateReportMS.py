import sys
import os
import numpy
import pandas
from collections import OrderedDict
import matplotlib.pyplot as plt
import seaborn as sns
import copy
from IPython.display import display 
import warnings
import re
import shutil
from matplotlib import gridspec

from .._toolboxPath import toolboxPath
from ..objects import MSDataset
from pyChemometrics.ChemometricsPCA import ChemometricsPCA
from ..plotting import plotTIC, histogram, plotLRTIC, jointplotRSDvCorrelation, plotRSDs, plotIonMap, plotBatchAndROCorrection, plotScores, plotLoadings
from ._generateSampleReport import _generateSampleReport
from ..utilities import generateLRmask, rsd
from ..utilities._internal import _vcorrcoef
from ..utilities._internal import _copyBackingFiles as copyBackingFiles
from ..enumerations import AssayRole, SampleType
from ._generateBasicPCAReport import generateBasicPCAReport

from ..__init__ import __version__ as version

def _generateReportMS(msDataTrue, reportType, withExclusions=False, withArtifactualFiltering=None, output=None, msDataCorrected=None, pcaModel=None, batch_correction_window=11):
	"""
	Summarise different aspects of an MS dataset

	Generate reports for ``feature summary``, ``correlation to dilution``, ``batch correction assessment``, ``batch correction summary``, ``feature selection``, or ``final report``
	
	* **'feature summary'** Generates feature summary report, plots figures including those for feature abundance, sample TIC and acquisition structure, correlation to dilution, RSD and an ion map.
	* **'correlation to dilution'** Generates a more detailed report on correlation to dilution, broken down by batch subset with TIC, detector voltage, a summary, and heatmap indicating potential saturation or other issues.
	* **'batch correction assessment'** Generates a report before batch correction showing TIC overall and intensity and batch correction fit for a subset of features, to aid specification of batch start and end points.
	* **'batch correction summary'** Generates a report post batch correction with pertinant figures (TIC, RSD etc.) before and after.
	* **'feature selection'** Generates a summary of the number of features passing feature selection (with current settings as definite in the SOP), and a heatmap showing how this number would be affected by changes to RSD and correlation to dilution thresholds.
	* **'final report'** Generates a summary of the final dataset, lists sample numbers present, a selection of figures summarising dataset quality, and a final list of samples missing from acquisition. 

	:param MSDataset msDataTrue: MSDataset to report on
	:param str reportType: Type of report to generate, one of ``feature summary``, ``correlation to dilution``, ``batch correction``, ``feature selection``, or ``final report``
	:param bool withExclusions: If ``True``, only report on features and samples not masked by the sample and feature masks
	:param None or bool withArtifactualFiltering: If ``None`` use the value from ``Attributes['artifactualFilter']``. If ``True`` apply artifactual filtering to the ``feature selection`` report and ``final report``
	:param output: If ``None`` plot interactively, otherwise save report to the path specified
	:type output: None or str
	:param MSDataset msDataCorrected: Only if ``batch correction``, if msDataCorrected included will generate report post correction
	:param PCAmodel pcaModel: Only if ``final report``, if PCAmodel object is available PCA scores plots coloured by sample type will be added to report
	"""
	
	# Check inputs
	if not isinstance(msDataTrue, MSDataset):
		raise TypeError('msDataTrue must be an instance of nPYc.MSDataset')

	acceptAllOptions = {'feature summary', 'correlation to dilution', 'batch correction assessment', 'batch correction summary', 'feature selection', 'final report'}
	if not isinstance(reportType, str) & (reportType in acceptAllOptions):
		raise ValueError('reportType must be == ' + str(acceptAllOptions))

	if not isinstance(withExclusions, bool):		
		raise TypeError('withExclusions must be a bool')	

	if withArtifactualFiltering is not None:
		if not isinstance(withArtifactualFiltering, bool):
			raise TypeError('withArtifactualFiltering must be a bool')
	if withArtifactualFiltering is None:
		withArtifactualFiltering = msDataTrue.Attributes['artifactualFilter']
	# if self.Attributes['artifactualFilter'] is False, can't/shouldn't apply it. However if self.Attributes['artifactualFilter'] is True, the user can have the choice to not apply it (withArtifactualFilering=False).
	if (withArtifactualFiltering is True) & (msDataTrue.Attributes['artifactualFilter'] is False):
		warnings.warn("Warning: Attributes['artifactualFilter'] set to \'False\', artifactual filtering cannot be applied.")
		withArtifactualFiltering = False

	if output is not None:
		if not isinstance(output, str):
			raise TypeError('output must be a string')

	if msDataCorrected is not None:
		if not isinstance(msDataCorrected, MSDataset):
			raise TypeError('msDataCorrected must be an instance of nPYc.MSDataset')

	if pcaModel is not None:
		if not isinstance(pcaModel, ChemometricsPCA):
			raise TypeError('pcaModel must be a ChemometricsPCA object')

	sns.set_style("whitegrid")

	# Create directory to save output		
	if output:
		
		reportTypeCase = reportType.title().replace(" ","")
		reportTypeCase = reportTypeCase[0].lower() + reportTypeCase[1:]
		saveDir = os.path.join(output, 'graphics', 'report_' + reportTypeCase)
		
		# If directory exists delete directory and contents
		if os.path.exists(saveDir):
			shutil.rmtree(saveDir)
		
		# Create directory to save output
		os.makedirs(saveDir)
		
	else:
		saveAs = None

	# Apply sample/feature masks if exclusions to be applied
	msData = copy.deepcopy(msDataTrue)
	if withExclusions:
		msData.applyMasks()

	# Define sample masks
	SSmask = (msData.sampleMetadata['SampleType'].values == SampleType.StudySample) & (msData.sampleMetadata['AssayRole'].values == AssayRole.Assay)
	SPmask = (msData.sampleMetadata['SampleType'].values == SampleType.StudyPool) & (msData.sampleMetadata['AssayRole'].values == AssayRole.PrecisionReference)
	ERmask = (msData.sampleMetadata['SampleType'].values == SampleType.ExternalReference) & (msData.sampleMetadata['AssayRole'].values == AssayRole.PrecisionReference)
	LRmask = (msData.sampleMetadata['SampleType'].values == SampleType.StudyPool) & (msData.sampleMetadata['AssayRole'].values == AssayRole.LinearityReference)
	[ns, nv] = msData.intensityData.shape

	# Set up template item and save required info
	item = dict() 
	item['Name'] = msData.name
	item['ReportType'] = reportType
	item['Nfeatures'] = str(nv)
	item['Nsamples'] = str(ns)
	item['SScount'] = str(sum(SSmask))
	item['SPcount'] = str(sum(SPmask))
	item['ERcount'] = str(sum(ERmask))
	item['corrMethod'] = msData.Attributes['corrMethod']

	# Feature summary report
	if reportType == 'feature summary':
		"""
		Generates feature summary report, plots figures including those for feature abundance, sample TIC and acquisition structure, correlation to dilution, RSD and an ion map.
		"""

		# Mean intensities of Study Pool samples (for future plotting segmented by intensity)
		meanIntensitiesSP = numpy.log(numpy.nanmean(msData.intensityData[SPmask,:], axis=0))
		meanIntensitiesSP[numpy.mean(msData.intensityData[SPmask,:], axis=0) == 0] = numpy.nan
		meanIntensitiesSP[numpy.isinf(meanIntensitiesSP)] = numpy.nan


		# Figure 1: Histogram of log mean abundance by sample type        
		if output:
			item['FeatureIntensityFigure'] = os.path.join(saveDir, item['Name'] + '_meanIntesityFeature.' + msData.Attributes['figureFormat'])
			saveAs = item['FeatureIntensityFigure']
		else:
			print('Figure 1: Feature intensity histogram for all samples and all features in dataset (by sample type).')  

		_plotAbundanceBySampleType(msData.intensityData, SSmask, SPmask, ERmask, saveAs, msData)    


		# Figure 2: Sample intensity TIC and distribution by sample type
		if output:
			item['SampleIntensityFigure'] = os.path.join(saveDir, item['Name'] + '_meanIntesitySample.' + msData.Attributes['figureFormat'])
			saveAs = item['SampleIntensityFigure']
		else:
			print('Figure 2: Sample Total Ion Count (TIC) and distribtion (coloured by sample type).')

		# TIC all samples
		plotTIC(msData, 
			addViolin=True,
			savePath=saveAs, 
			title='',
			figureFormat=msData.Attributes['figureFormat'],
			dpi=msData.Attributes['dpi'],
			figureSize=msData.Attributes['figureSize'])

		# Figure 3: Acquisition structure and detector voltage
		if output:
			item['AcquisitionStructureFigure'] = os.path.join(saveDir, item['Name'] + '_acquisitionStructure.' + msData.Attributes['figureFormat'])
			saveAs = item['AcquisitionStructureFigure']
		else:
			print('Figure 3: Acquisition structure (coloured by detector voltage).')

		# TIC all samples
		plotTIC(msData, 
			addViolin=False,
			addBatchShading=True,
			addLineAtGaps=True,
			colourByDetectorVoltage=True,
			savePath=saveAs, 
			title='',
			figureFormat=msData.Attributes['figureFormat'],
			dpi=msData.Attributes['dpi'],
			figureSize=msData.Attributes['figureSize'])

		# Correlation to dilution figures:
		if sum(LRmask) != 0:

			# Figure 4: Histogram of correlation to dilution by abundance percentiles
			if output:
				item['CorrelationByPercFigure'] = os.path.join(saveDir, item['Name'] + '_correlationByPerc.' + msData.Attributes['figureFormat'])
				saveAs = item['CorrelationByPercFigure']
			else:
				print('Figure 4: Histogram of ' + item['corrMethod'] + ' correlation of features to serial dilution, segmented by percentile.')

			histogram(msData.correlationToDilution, 
				xlabel='Correlation to Dilution', 
				histBins=msData.Attributes['histBins'],
				quantiles=msData.Attributes['quantiles'],
				inclusionVector=numpy.exp(meanIntensitiesSP),
				savePath=saveAs, 
				figureFormat=msData.Attributes['figureFormat'],
				dpi=msData.Attributes['dpi'],
				figureSize=msData.Attributes['figureSize'])	

			# Figure 5: TIC of linearity reference samples
			if output:
				item['TICinLRfigure'] = os.path.join(saveDir, item['Name'] + '_TICinLR.' + msData.Attributes['figureFormat'])
				saveAs = item['TICinLRfigure']
			else:
				print('Figure 5: TIC of linearity reference (LR) samples coloured by sample dilution.')

			plotLRTIC(msData, 
				sampleMask=LRmask, 
				savePath=saveAs,
				figureFormat=msData.Attributes['figureFormat'],
				dpi=msData.Attributes['dpi'],
				figureSize=msData.Attributes['figureSize'])

		else:
			if not output:
				print('Figure 4: Histogram of ' + item['corrMethod'] + ' correlation of features to serial dilution, segmented by percentile.')
				print('Unable to calculate (no linearity reference samples present in dataset).\n')
				
				print('Figure 5: TIC of linearity reference (LR) samples coloured by sample dilution')
				print('Unable to calculate (no linearity reference samples present in dataset).\n')


		# Figure 6: Histogram of RSD in SP samples by abundance percentiles
		if output:
			item['RsdByPercFigure'] = os.path.join(saveDir, item['Name'] + '_rsdByPerc.' + msData.Attributes['figureFormat'])
			saveAs = item['RsdByPercFigure']
		else:
			print('Figure 6: Histogram of Residual Standard Deviation (RSD) in study pool (SP) samples, segmented by abundance percentiles.')
			
		histogram(msData.rsdSP, 
			xlabel='RSD',
			histBins=msData.Attributes['histBins'],
			quantiles=msData.Attributes['quantiles'],
			inclusionVector=numpy.exp(meanIntensitiesSP),
			logx=False,
			xlim=(0, 100),
			savePath=saveAs, 
			figureFormat=msData.Attributes['figureFormat'],
			dpi=msData.Attributes['dpi'],
			figureSize=msData.Attributes['figureSize'])


		# Figure 7: Scatterplot of RSD vs correlation to dilution
		if sum(LRmask) !=0:
			if output:
				item['RsdVsCorrelationFigure'] = os.path.join(saveDir, item['Name'] + '_rsdVsCorrelation.' + msData.Attributes['figureFormat'])
				saveAs = item['RsdVsCorrelationFigure']
			else:
				print('Figure 7: Scatterplot of RSD vs correlation to dilution.')

			jointplotRSDvCorrelation(msData.rsdSP, 
					msData.correlationToDilution,
					savePath=saveAs,
					figureFormat=msData.Attributes['figureFormat'],
					dpi=msData.Attributes['dpi'],
					figureSize=msData.Attributes['figureSize'])

		else:
			if not output:
				print('Figure 7: Scatterplot of RSD vs correlation to dilution.')
				print('Unable to calculate (no serial dilution samples present in dataset).\n')

		if 'Peak Width' in msData.featureMetadata.columns:
			# Figure 8: Histogram of chromatographic peak width
			if output:
				item['PeakWidthFigure'] = os.path.join(saveDir, item['Name'] + '_peakWidth.' + msData.Attributes['figureFormat'])
				saveAs = item['PeakWidthFigure']
			else:
				print('Figure 8: Histogram of chromatographic peak width.')

			histogram(msData.featureMetadata['Peak Width'], 
				xlabel='Peak Width (minutes)', 
				histBins=msData.Attributes['histBins'],
				savePath=saveAs, 
				figureFormat=msData.Attributes['figureFormat'],
				dpi=msData.Attributes['dpi'],
				figureSize=msData.Attributes['figureSize'])
		else:
			if not output:
				print('\x1b[31;1m No peak width data to plot')
				print('Figure 8: Histogram of chromatographic peak width.')

		# Figure 9: Residual Standard Deviation (RSD) distribution for all samples and all features in dataset (by sample type)
		if output:
			item['RSDdistributionFigure'] = os.path.join(saveDir, item['Name'] + '_RSDdistributionFigure.' + msData.Attributes['figureFormat'])
			saveAs = item['RSDdistributionFigure']
		else:
			print('Figure 9: RSD distribution for all samples and all features in dataset (by sample type).')

		plotRSDs(msData,
			ratio=False,
			logx=True,
			color='matchReport',
			savePath=saveAs,
			figureFormat=msData.Attributes['figureFormat'],
			dpi=msData.Attributes['dpi'],
			figureSize=msData.Attributes['figureSize'])


		# Figure 10: Ion map
		if output:
			item['IonMap'] = os.path.join(saveDir, item['Name'] + '_ionMap.' + msData.Attributes['figureFormat'])
			saveAs = item['IonMap']
		else:
			print('Figure 10: Ion map of all features (coloured by log median intensity).')

		plotIonMap(msData, 
			savePath=saveAs, 
			figureFormat=msData.Attributes['figureFormat'],
			dpi=msData.Attributes['dpi'],
			figureSize=msData.Attributes['figureSize'])	
    

	# Correlation to dilution report
	if reportType == 'correlation to dilution':
		"""
		Generates a more detailed report on correlation to dilution, broken down by batch subset with TIC, detector voltage, a summary, and heatmap indicating potential saturation or other issues.
		"""
		
		# Check inputs
		if not hasattr(msData.sampleMetadata, 'Correction Batch'):
			raise ValueError("Correction Batch information missing, run addSampleInfo(descriptionFormat=\'Batches\')")

		# Generate correlation to dilution for each batch subset - plot TIC and histogram of correlation to dilution

		# generate LRmask
		LRmask = generateLRmask(msData)

		# instantiate dictionarys
		corLRbyBatch = {} # to save correlations
		corLRsummary = {} # summary of number of features with correlation above threshold
		corLRsummary['TotalOriginal'] = len(msData.featureMask)
		
		if output:
			saveAs = saveDir
			figuresCorLRbyBatch = OrderedDict() # To save figures
		else:
			figuresCorLRbyBatch = None

		for key in sorted(LRmask):
			corLRbyBatch[key] = _vcorrcoef(msData.intensityData, msData.sampleMetadata['Dilution'].values, method=msData.Attributes['corrMethod'], sampleMask = LRmask[key])
			corLRsummary[key] = sum(corLRbyBatch[key] >= msData.Attributes['corrThreshold'])
			figuresCorLRbyBatch = _localLRPlots(msData, 
				LRmask[key], 
				corLRbyBatch[key], 
				key,
				figures=figuresCorLRbyBatch,
				savePath=saveAs)

		# Calculate average (mean) correlation across all batch subsets
		corALL = numpy.zeros([len(corLRbyBatch),len(msData.featureMask)])
		n = 0
		for key in corLRbyBatch:
			corALL[n,:] = corLRbyBatch[key]
			n=n+1

		corLRbyBatch['MeanAllSubsets'] = numpy.mean(corALL, axis=0)
		corLRsummary['MeanAllSubsets'] = sum(corLRbyBatch['MeanAllSubsets'] >= msData.Attributes['corrThreshold'])
		figuresCorLRbyBatch = _localLRPlots(msData,
			(msData.sampleMetadata['SampleType'].values == SampleType.StudyPool) & (msData.sampleMetadata['AssayRole'].values == AssayRole.LinearityReference),
			corLRbyBatch['MeanAllSubsets'],
			'MeanAllSubsets',
			figures=figuresCorLRbyBatch,
			savePath=saveAs)
		
		if figuresCorLRbyBatch is not None:			
			# Make paths for graphics local not absolute for use in the HTML.
			for key in figuresCorLRbyBatch:
				if os.path.join(output, 'graphics') in str(figuresCorLRbyBatch[key]):
					figuresCorLRbyBatch[key] = re.sub('.*graphics', 'graphics', figuresCorLRbyBatch[key])
			# Save to item
			item['figuresCorLRbyBatch'] = figuresCorLRbyBatch

		# Summary table of number of features passing threshold with each subset
		temp = pandas.DataFrame(corLRsummary, index=range(1))
		temp['CurrentSettings'] = sum(msData.correlationToDilution>=msData.Attributes['corrThreshold'])
		temp = temp.T
		temp.rename(columns = {0 :'N Features'}, inplace = True)

		item['NfeaturesSummary'] = temp
		item['corrThreshold'] = str(msData.Attributes['corrThreshold'])
		item['corrMethod'] = msData.Attributes['corrMethod']
		if sum(msData.corrExclusions) != msData.noSamples:
			item['corrExclusions'] = str(msData.sampleMetadata.loc[msData.corrExclusions == False, 'Sample File Name'].values)
		else:
			item['corrExclusions'] = 'none'

		if not output:

			print('Number of features exceeding correlation to dilution threshold (' + str(item['corrThreshold']) + ') for each LR sample subset/correlation to dilution method')
			display(temp)

			print('\nCurrent correlation settings:' + 
				'\nCorrelation method: ' + item['corrMethod'] +
				'\nCorrelation exclusions: ' + item['corrExclusions'] +
				'\nCorrelation threshold: ' + item['corrThreshold'])
	

		# Assessment of potential saturation

		# Heatmap showing the proportion of features (across different intensities) where median
		# intensity at lower dilution factor >= that at higher dilution factor 

		# calculate median feature intensity quantiles and feature masks
		medI = numpy.nanmedian(msData.intensityData, axis=0)
		quantiles = numpy.percentile(medI, [25,75])
		nf = msData.intensityData.shape[1]
		lowImask = medI <= quantiles[0]
		midImask = (medI > quantiles[0]) & (medI <= quantiles[1])
		highImask = medI >= quantiles[1]

		# dilution factors present
		dilutions = (numpy.unique(msData.sampleMetadata['Dilution'].values[~numpy.isnan(msData.sampleMetadata['Dilution'].values)])).astype(int)
		dilutions.sort()

		# LR batch subsets
		LRbatchmask = generateLRmask(msData)

		# median feature intensities for different dilution samples
		medItable = numpy.full([nf, len(dilutions)*len(LRbatchmask)], numpy.nan)
		i = 0
		for key in LRbatchmask:
			for d in dilutions:
				mask = (msData.sampleMetadata['Dilution'].values == d) & (LRbatchmask[key])
				medItable[:,i] = numpy.nanmedian(msData.intensityData[mask,:], axis=0)
				i = i+1
		
		# dataframe for proportion of features with median intensity at lower dilution factor >= that at higher dilution factor
		i = 0
		for key in sorted(LRbatchmask):
			for d in numpy.arange(0, len(dilutions)-1):
				if 'sat' not in locals():
					sat = pandas.DataFrame({'Average feature intensity': ['1. low ' + key], 'LR' : [str(d+1) + '. ' + str(dilutions[d+1]) + '<=' + str(dilutions[d])], 'Proportion of features' : [sum(medItable[lowImask, i+1] <= medItable[lowImask, i]) / sum(lowImask) * 100]})
					sat = sat.append({'Average feature intensity': '2. medium ' + key, 'LR' : str(d+1) + '. ' +str(dilutions[d+1]) + '<=' + str(dilutions[d]), 'Proportion of features' : sum(medItable[midImask, i+1] <= medItable[midImask, i]) / sum(midImask) * 100}, ignore_index=True)
					sat = sat.append({'Average feature intensity': '3. high ' + key, 'LR' : str(d+1) + '. ' +str(dilutions[d+1]) + '<=' + str(dilutions[d]), 'Proportion of features' : sum(medItable[highImask, i+1] <= medItable[highImask, i]) / sum(highImask) * 100}, ignore_index=True)
				else:
					sat = sat.append({'Average feature intensity': '1. low ' + key, 'LR' : str(d+1) + '. ' +str(dilutions[d+1]) + '<=' + str(dilutions[d]), 'Proportion of features' : sum(medItable[lowImask, i+1] <= medItable[lowImask, i]) / sum(lowImask) * 100}, ignore_index=True)
					sat = sat.append({'Average feature intensity': '2. medium ' + key, 'LR' : str(d+1) + '. ' +str(dilutions[d+1]) + '<=' + str(dilutions[d]), 'Proportion of features' : sum(medItable[midImask, i+1] <= medItable[midImask, i]) / sum(midImask) * 100}, ignore_index=True)
					sat = sat.append({'Average feature intensity': '3. high ' + key, 'LR' : str(d+1) + '. ' +str(dilutions[d+1]) + '<=' + str(dilutions[d]), 'Proportion of features' : sum(medItable[highImask, i+1] <= medItable[highImask, i]) / sum(highImask) * 100}, ignore_index=True)
				i = i+1
			i = i+1

		satHeatmap = sat.pivot('Average feature intensity', 'LR', 'Proportion of features')
		satLineplot = sat.pivot('LR', 'Average feature intensity', 'Proportion of features')

		# plot heatmap
		with sns.axes_style("white"):
			fig = plt.figure(figsize=msData.Attributes['figureSize'], dpi=msData.Attributes['dpi'])
			gs = gridspec.GridSpec(1, 11)
			ax1 = plt.subplot(gs[0,:5])
			ax2 = plt.subplot(gs[0, -5:])
			ax1 = sns.heatmap(satHeatmap, ax=ax1, annot=True, fmt='.3g', vmin=0, vmax=100, cmap='Reds', cbar=False)
			ax2 = satLineplot.plot(kind='line', ax=ax2, ylim=[0,100], colormap='jet')
			if output:
				item['SatFeaturesHeatmap'] = os.path.join(saveDir, item['Name'] + '_satFeaturesHeatmap.' + msData.Attributes['figureFormat'])
				plt.savefig(item['SatFeaturesHeatmap'], bbox_inches='tight', format=msData.Attributes['figureFormat'], dpi=msData.Attributes['dpi'])
				plt.close()
			else:
				print('\n\nAssessment of potential saturation')
				print('\nHeatmap/lineplot showing the proportion of features (in different intensity quantiles, low:0-25, medium:25-75, and high:75-100%) where the median intensity at lower dilution factors >= that at higher dilution factors')
				plt.show()
			

	# Batch correction assessment report
	if reportType == 'batch correction assessment':
		"""
		Generates a report before batch correction showing TIC overall and intensity and batch correction fit for a subset of features, to aid specification of batch start and end points.
		"""

		# Pre-correction report (report is example of results when batch correction applied)

		# Check inputs
		if not hasattr(msData.sampleMetadata, 'Correction Batch'):
			raise ValueError("Correction Batch information missing, run addSampleInfo(descriptionFormat=\'Batches\')")

		# Figure 1: TIC for all samples by sample type and detector voltage change
		if output:
			item['TICdetectorBatches'] = os.path.join(saveDir, item['Name'] + '_TICdetectorBatches.' + msData.Attributes['figureFormat'])
			saveAs = item['TICdetectorBatches']
		else:
			print('Overall Total Ion Count (TIC) for all samples and features, coloured by batch.')

		plotTIC(msData,
			addViolin=True,
			addBatchShading=True,
			savePath=saveAs,
			figureFormat=msData.Attributes['figureFormat'],
			dpi=msData.Attributes['dpi'],
			figureSize=msData.Attributes['figureSize'])

		# Remaining figures: Sample of fits for selection of features 
		(preData, postData, maskNum) = batchCorrectionTest(msData, nFeatures=10, window=batch_correction_window)
		item['NoBatchPlotFeatures'] = len(maskNum)
		
		if output:
			figuresCorrectionExamples = OrderedDict() # To save figures
		else:
			print('Example batch correction plots for a subset of features, results of batch correction with specified batches.')
			figuresCorrectionExamples = None
			
		for feature in range(len(maskNum)):
			
			featureName = str(numpy.squeeze(preData.featureMetadata.loc[feature, 'Feature Name'])).replace('/', '-')
			if output:
				figuresCorrectionExamples['Feature ' + featureName] = os.path.join(saveDir, item['Name'] + '_batchPlotFeature_' + featureName + '.' + msData.Attributes['figureFormat'])
				saveAs = figuresCorrectionExamples['Feature ' + featureName]
			else:
				print('Feature ' + featureName)

			plotBatchAndROCorrection(preData, 
				postData,
				feature,
				logy=True,
				savePath=saveAs,
				figureFormat=msData.Attributes['figureFormat'],
				dpi=msData.Attributes['dpi'],
				figureSize=msData.Attributes['figureSize'])
	
		if figuresCorrectionExamples is not None:			
			# Make paths for graphics local not absolute for use in the HTML.
			for key in figuresCorrectionExamples:
				if os.path.join(output, 'graphics') in str(figuresCorrectionExamples[key]):
					figuresCorrectionExamples[key] = re.sub('.*graphics', 'graphics', figuresCorrectionExamples[key])					
			# Save to item
			item['figuresCorrectionExamples'] = figuresCorrectionExamples
				
		
	# Post-correction report (TIC pre and post batch correction)
	if reportType == 'batch correction summary':
		"""
		Generates a report post batch correction with pertinant figures (TIC, RSD etc.) before and after.
		"""
		
		# Define sample masks
		SSmask = (msData.sampleMetadata['SampleType'].values == SampleType.StudySample) & (msData.sampleMetadata['AssayRole'].values == AssayRole.Assay)
		SPmask = (msData.sampleMetadata['SampleType'].values == SampleType.StudyPool) & (msData.sampleMetadata['AssayRole'].values == AssayRole.PrecisionReference)
		ERmask = (msData.sampleMetadata['SampleType'].values == SampleType.ExternalReference) & (msData.sampleMetadata['AssayRole'].values == AssayRole.PrecisionReference)
		
		# Mean intensities of Study Pool samples (for future plotting segmented by intensity)
		meanIntensitiesSP = numpy.log(numpy.nanmean(msData.intensityData[SPmask,:], axis=0))
		meanIntensitiesSP[numpy.mean(msData.intensityData[SPmask,:], axis=0) == 0] = numpy.nan
		meanIntensitiesSP[numpy.isinf(meanIntensitiesSP)] = numpy.nan
				
		# Figure 1: Feature intensity histogram for all samples and all features in dataset (by sample type).

		# Pre-correction
		if output:
			item['FeatureIntensityFigurePRE'] = os.path.join(saveDir, item['Name'] + '_BCS1_meanIntesityFeaturePRE.' + msData.Attributes['figureFormat'])
			saveAs = item['FeatureIntensityFigurePRE']
		else:
			print('Figure 1: Feature intensity histogram for all samples and all features in dataset (by sample type).')
			print('Pre-correction.')

		_plotAbundanceBySampleType(msData.intensityData, SSmask, SPmask, ERmask, saveAs, msData)	
		
		# Post-correction
		if output:
			item['FeatureIntensityFigurePOST'] = os.path.join(saveDir, item['Name'] + '_BCS1_meanIntesityFeaturePOST.' + msData.Attributes['figureFormat'])
			saveAs = item['FeatureIntensityFigurePOST']
		else:
			print('Post-correction.')  

		_plotAbundanceBySampleType(msDataCorrected.intensityData, SSmask, SPmask, ERmask, saveAs, msDataCorrected)

			
		# Figure 2: TIC for all samples and features.
					
		# Pre-correction
		if output:
			item['TicPRE'] = os.path.join(saveDir, item['Name'] + '_BCS2_TicPRE.' + msData.Attributes['figureFormat'])
			saveAs = item['TicPRE']
		else:
			print('Sample Total Ion Count (TIC) and distribtion (coloured by sample type).')
			print('Pre-correction.')
		
		plotTIC(msData,
			addViolin=True,
			title='TIC Pre Batch-Correction',
			savePath=saveAs,
			figureFormat=msData.Attributes['figureFormat'],
			dpi=msData.Attributes['dpi'],
			figureSize=msData.Attributes['figureSize'])

		# Post-correction
		if output:
			item['TicPOST'] = os.path.join(saveDir, item['Name'] + '_BCS2_TicPOST.' + msData.Attributes['figureFormat'])
			saveAs = item['TicPOST']
		else:
			print('Post-correction.')
		
		plotTIC(msDataCorrected,
			addViolin=True,
			title='TIC Post Batch-Correction',
			savePath=saveAs,
			figureFormat=msData.Attributes['figureFormat'],
			dpi=msData.Attributes['dpi'],
			figureSize=msData.Attributes['figureSize'])
		
		
		# Figure 3: Histogram of RSD in study pool (SP) samples, segmented by abundance percentiles.

		# Pre-correction			
		if output:
			item['RsdByPercFigurePRE'] = os.path.join(saveDir, item['Name'] + '_BCS3_rsdByPercPRE.' + msData.Attributes['figureFormat'])
			saveAs = item['RsdByPercFigurePRE']
		else:
			print('Figure 3: Histogram of Residual Standard Deviation (RSD) in study pool (SP) samples, segmented by abundance percentiles.')
			print('Pre-correction.')
			
		histogram(msData.rsdSP, 
			xlabel='RSD',
			histBins=msData.Attributes['histBins'],
			quantiles=msData.Attributes['quantiles'],
			inclusionVector=numpy.exp(meanIntensitiesSP),
			logx=False,
			xlim=(0, 100),
			savePath=saveAs, 
			figureFormat=msData.Attributes['figureFormat'],
			dpi=msData.Attributes['dpi'],
			figureSize=msData.Attributes['figureSize'])

		# Post-correction			
		if output:
			item['RsdByPercFigurePOST'] = os.path.join(saveDir, item['Name'] + '_BCS3_rsdByPercPOST.' + msData.Attributes['figureFormat'])
			saveAs = item['RsdByPercFigurePOST']
		else:
			print('Post-correction.')
			
		histogram(msDataCorrected.rsdSP, 
			xlabel='RSD',
			histBins=msData.Attributes['histBins'],
			quantiles=msData.Attributes['quantiles'],
			inclusionVector=numpy.exp(meanIntensitiesSP),
			logx=False,
			xlim=(0, 100),
			savePath=saveAs, 
			figureFormat=msData.Attributes['figureFormat'],
			dpi=msData.Attributes['dpi'],
			figureSize=msData.Attributes['figureSize'])
		
		
		# Figure 4: Residual Standard Deviation (RSD) distribution for all samples and all features in dataset (by sample type).
		
		# Pre-correction
		if output:
			item['RSDdistributionFigurePRE'] = os.path.join(saveDir, item['Name'] + '_BCS4_RSDdistributionFigurePRE.' + msData.Attributes['figureFormat'])
			saveAs = item['RSDdistributionFigurePRE']
		else:
			print('Figure 4: RSD distribution for all samples and all features in dataset (by sample type).')
			print('Pre-correction.')

		plotRSDs(msData,
			ratio=False,
			logx=True,
			color='matchReport',
			savePath=saveAs,
			figureFormat=msData.Attributes['figureFormat'],
			dpi=msData.Attributes['dpi'],
			figureSize=msData.Attributes['figureSize'])

		# Post-correction
		if output:
			item['RSDdistributionFigurePOST'] = os.path.join(saveDir, item['Name'] + '_BCS4_RSDdistributionFigurePOST.' + msData.Attributes['figureFormat'])
			saveAs = item['RSDdistributionFigurePOST']
		else:
			print('Post-correction.')

		plotRSDs(msDataCorrected,
			ratio=False,
			logx=True,
			color='matchReport',
			savePath=saveAs,
			figureFormat=msData.Attributes['figureFormat'],
			dpi=msData.Attributes['dpi'],
			figureSize=msData.Attributes['figureSize'])
			
	
	# Feature selection report
	if reportType == 'feature selection':
		"""
		Generates a summary of the number of features passing feature selection (with current settings as definite in the SOP), and a heatmap showing how this number would be affected by changes to RSD and correlation to dilution thresholds.
		"""

		# Feature selection parameters and numbers passing

		# rsdSP <= rsdSS
		rsdSS = rsd(msData.intensityData[SSmask,:])
		item['rsdSPvsSSvarianceRatio'] = str(msData.Attributes['varianceRatio'])
		item['rsdSPvsSSPassed'] = sum((msData.rsdSP * msData.Attributes['varianceRatio']) <= rsdSS)
		
		# Correlation to dilution
		item['corrMethod'] = msData.Attributes['corrMethod']
		item['corrThreshold'] = msData.Attributes['corrThreshold']
		if sum(msData.corrExclusions) != msData.noSamples:
			item['corrExclusions'] = str(msData.sampleMetadata.loc[msData.corrExclusions == False, 'Sample File Name'].values)
		else:
			item['corrExclusions'] = 'none'
		item['corrPassed'] = sum(msData.correlationToDilution >= msData.Attributes['corrThreshold'])

		# rsdSP
		item['rsdThreshold'] = msData.Attributes['rsdThreshold']
		item['rsdPassed'] = sum(msData.rsdSP <= msData.Attributes['rsdThreshold'])

		# Artifactual filtering
		passMask = (msData.correlationToDilution >= msData.Attributes['corrThreshold']) & (msData.rsdSP <= msData.Attributes['rsdThreshold']) & ((msData.rsdSP * msData.Attributes['varianceRatio']) <= rsdSS) & (msData.featureMask == True)
		if withArtifactualFiltering:
			passMask = msData.artifactualFilter(featMask=passMask)

		if 'blankThreshold' in msData.Attributes.keys():
			from ..utilities._filters import blankFilter

			blankThreshold = msData.Attributes['blankThreshold']

			blankMask = blankFilter(msData)

			passMask = numpy.logical_and(passMask, blankMask)

			item['BlankPassed'] = sum(blankMask)

		if withArtifactualFiltering:
			item['artifactualPassed'] = sum(passMask)
		item['featuresPassed'] = sum(passMask)
		
		# Heatmap of the number of features passing selection with different RSD and correlation to dilution thresholds
		rsdVals = numpy.arange(5,55,5)
		rVals = numpy.arange(0.5,1.01,0.05)
		rValsRep = numpy.tile(numpy.arange(0.5,1.01,0.05),[1, len(rsdVals)])
		rsdValsRep = numpy.reshape(numpy.tile(numpy.arange(5,55,5), [len(rVals),1]), rValsRep.shape, order='F')
		featureNos = numpy.zeros(rValsRep.shape, dtype=numpy.int)
		if withArtifactualFiltering:
			# with blankThreshold in heatmap
			if 'blankThreshold' in msData.Attributes.keys():
				for rsdNo in range(rValsRep.shape[1]):
					featureNos[0, rsdNo] = sum(msData.artifactualFilter(featMask=((msData.correlationToDilution >= rValsRep[0, rsdNo]) & (msData.rsdSP <= rsdValsRep[0, rsdNo]) & ((msData.rsdSP * msData.Attributes['varianceRatio']) <= rsdSS) & (msData.featureMask == True) & (blankMask == True))))
			# without blankThreshold
			else:
				for rsdNo in range(rValsRep.shape[1]):
					featureNos[0, rsdNo] = sum(msData.artifactualFilter(featMask=((msData.correlationToDilution >= rValsRep[0, rsdNo]) & (msData.rsdSP <= rsdValsRep[0, rsdNo]) & ((msData.rsdSP * msData.Attributes['varianceRatio']) <= rsdSS) & (msData.featureMask==True))))
		else:
			# with blankThreshold in heatmap
			if 'blankThreshold' in msData.Attributes.keys():
				for rsdNo in range(rValsRep.shape[1]):
					featureNos[0, rsdNo] = sum((msData.correlationToDilution >= rValsRep[0, rsdNo]) & (msData.rsdSP <= rsdValsRep[0, rsdNo]) & ((msData.rsdSP * msData.Attributes['varianceRatio']) <= rsdSS) & (msData.featureMask == True) & (blankMask == True))
			# without blankThreshold
			else:
				for rsdNo in range(rValsRep.shape[1]):
					featureNos[0, rsdNo] = sum((msData.correlationToDilution >= rValsRep[0, rsdNo]) & (msData.rsdSP <= rsdValsRep[0, rsdNo]) & ((msData.rsdSP * msData.Attributes['varianceRatio']) <= rsdSS) & (msData.featureMask == True))
		test = pandas.DataFrame(data=numpy.transpose(numpy.concatenate([rValsRep,rsdValsRep,featureNos])), columns=['Correlation to dilution','RSD','nFeatures'])
		test = test.pivot('Correlation to dilution','RSD','nFeatures')	
 	
		fig, ax = plt.subplots(1, figsize=msData.Attributes['figureSize'], dpi=msData.Attributes['dpi'])
		sns.heatmap(test, annot=True, fmt='g', cbar=False)
		plt.tight_layout()
		
		if output:
			item['NoFeaturesHeatmap'] = os.path.join(saveDir, item['Name'] + '_noFeatures.' + msData.Attributes['figureFormat'])
			plt.savefig(item['NoFeaturesHeatmap'], format=msData.Attributes['figureFormat'], dpi=msData.Attributes['dpi'])
			plt.close()
		
		else:
			print('Heatmap of the number of features passing selection with different Residual Standard Deviation (RSD) and correlation to dilution thresholds')
			plt.show()
			
			print('Summary of current feature filtering parameters and number of features passing at each stage\n')
			print('Number of features in original dataset: ' + str(item['Nfeatures']) + '\n\n' +
				'Features filtered on:\n' + 
				'Correlation (' + item['corrMethod'] + ', exclusions: ' + item['corrExclusions'] + ') to dilution greater than ' + str(item['corrThreshold']) + ': ' + str(item['corrPassed']) + ' passed selection\n' +
				'Relative Standard Deviation (RSD) in study pool (SP) samples below ' + str(item['rsdThreshold']) + ': ' + str(item['rsdPassed']) + ' passed selection\n' +
				'RSD in study samples (SS) * ' + item['rsdSPvsSSvarianceRatio'] + ' >= RSD in SP samples: ' + str(item['rsdSPvsSSPassed']) + ' passed selection')
			if blankThreshold:
				print('%i features above blank threshold.' % (item['BlankPassed']))
			if withArtifactualFiltering:
				print('Artifactual features filtering: ' + str(item['artifactualPassed']) + ' passed selection')
			print('\nTotal number of features after filtering: ' + str(item['featuresPassed']))
	

	# Final summary report
	if reportType == 'final report':
		"""
		Generates a summary of the final dataset, lists sample numbers present, a selection of figures summarising dataset quality, and a final list of samples missing from acquisition.
		"""

		# Table 1: Sample summary
										
		# Generate sample summary
		sampleSummary = _generateSampleReport(msData, withExclusions=True, output=None, returnOutput=True)
		
		# Extract summary table for samples acquired
		sampleSummaryTable = copy.deepcopy(sampleSummary['Acquired'])
		
		# Drop unwanted columns
		sampleSummaryTable.drop(['Marked for Exclusion'], axis=1, inplace=True)
		if 'LIMS marked as missing' in sampleSummaryTable.columns:
			sampleSummaryTable.drop(['LIMS marked as missing', 'Missing from LIMS'], axis=1, inplace=True) 
		if 'Missing Subject Information' in sampleSummaryTable.columns:
			sampleSummaryTable.drop(['Missing Subject Information'], axis=1, inplace=True) 
		
		# Rename 'already excluded'
		sampleSummaryTable.rename(columns={'Already Excluded': 'Excluded'}, inplace=True)
		
		# Add 'unavailable' column
		if 'NotAcquired' in sampleSummary:
			sampleSummaryTable = sampleSummaryTable.join(pandas.DataFrame(data=sampleSummary['NotAcquired']['Marked as Sample'] - sampleSummary['NotAcquired']['Already Excluded'], columns=['Unavailable']), how='left', sort=False)
		else:
			sampleSummaryTable['Unavailable'] = 0
	
		# Update 'All', 'Unavailable' to only reflect sample types present in data
		sampleSummaryTable.loc['All', 'Unavailable'] = sum(sampleSummaryTable['Unavailable'][1:])
		
		# Save to item
		item['SampleSummaryTable'] = sampleSummaryTable
		
		# Save details of study samples missing from dataset
		if sampleSummaryTable['Unavailable']['Study Sample'] != 0:
			item['SamplesMissingInfo'] = sampleSummary['NotAcquired Details'].loc[sampleSummary['NotAcquired Details']['Sampling ID'].isnull()==False,:]
			item['SamplesMissingInfo'] = item['SamplesMissingInfo'].drop(['LIMS Marked Missing'], axis=1)
			item['SamplesMissingNo'] = str(sampleSummaryTable['Unavailable']['Study Sample'])
		
		# Save details of study samples excluded from dataset
		if hasattr(sampleSummaryTable, 'Excluded'):
			if sampleSummaryTable['Excluded']['Study Sample'] != 0:
				item['SamplesExcludedInfo'] = sampleSummary['Excluded Details'].loc[(sampleSummary['Excluded Details']['SampleType'] == SampleType.StudySample) & (sampleSummary['Excluded Details']['AssayRole'] == AssayRole.Assay),:]
				item['SamplesExcludedInfo'] = item['SamplesExcludedInfo'].drop(['Sample Base Name', 'SampleType', 'AssayRole'], axis=1)
				item['SamplesExcludedNo'] = str(sampleSummaryTable['Excluded']['Study Sample'])
		
		if not output:
			print('Final Dataset for: ' + item['Name'])
			print('\n\t' + item['Nsamples'] + ' samples\n\t' + item['Nfeatures'] + ' features\n')
			
			print('Sample Summary')
			display(item['SampleSummaryTable'])
			print('\n')

	
		# Figure 1: Acquisition Structure, TIC by sample and batch
		nBatchCollect = len((numpy.unique(msData.sampleMetadata['Batch'].values[~numpy.isnan(msData.sampleMetadata['Batch'].values)])).astype(int))	
		if nBatchCollect == 1:
			item['nBatchesCollect'] = '1 batch'
		else:
			item['nBatchesCollect'] = str(nBatchCollect) + ' batches'		
			
		nBatchCorrect = len((numpy.unique(msData.sampleMetadata['Correction Batch'].values[~numpy.isnan(msData.sampleMetadata['Correction Batch'].values)])).astype(int))
		if nBatchCorrect == 1:
			item['nBatchesCorrect'] = '1 batch'
		else:
			item['nBatchesCorrect'] = str(nBatchCorrect) + ' batches'
	
		start = pandas.to_datetime(str(msData.sampleMetadata['Acquired Time'].loc[msData.sampleMetadata['Run Order'] == min(msData.sampleMetadata['Run Order'][msData.sampleMask])].values[0]))
		end = pandas.to_datetime(str(msData.sampleMetadata['Acquired Time'].loc[msData.sampleMetadata['Run Order'] == max(msData.sampleMetadata['Run Order'][msData.sampleMask])].values[0]))
		item['start'] = start.strftime('%d/%m/%y')
		item['end'] = end.strftime('%d/%m/%y')
	
		if output:
			item['finalTICbatches'] = os.path.join(saveDir, item['Name'] + '_finalTICbatches.' + msData.Attributes['figureFormat'])
			saveAs = item['finalTICbatches']
		else:
			print('Acquisition Structure')
			print('\n\tSamples acquired in ' +  item['nBatchesCollect'] + ' between ' + item['start'] + ' and ' + item['end'])
			print('\n\tBatch correction applied (LOESS regression fitted to SP samples in ' +  item['nBatchesCorrect'] + ') for run-order correction and batch alignment\n')
			print('Figure 1: Acquisition Structure')
	
		plotTIC(msData, 
				savePath=saveAs,
				addBatchShading=True,
				figureFormat=msData.Attributes['figureFormat'],
				dpi=msData.Attributes['dpi'],
				figureSize=msData.Attributes['figureSize'])


		# Table 2: Feature Selection parameters
		FeatureSelectionTable = pandas.DataFrame(data = ['yes', msData.Attributes['corrMethod'], msData.Attributes['corrThreshold']],
			index = ['Correlation to Dilution','Correlation to Dilution: Method', 'Correlation to Dilution: Threshold'],
			columns = ['Applied'])
		if sum(msData.corrExclusions) != msData.noSamples:
			temp = ', '.join(msData.sampleMetadata.loc[msData.corrExclusions == False, 'Sample File Name'].values)
			FeatureSelectionTable = FeatureSelectionTable.append(pandas.DataFrame(data=temp, index=['Correlation to Dilution: Sample Exclusions'], columns=['Applied']))
		else:
			FeatureSelectionTable = FeatureSelectionTable.append(pandas.DataFrame(data = ['none'], index = ['Correlation To Dilution: Sample Exclusions'], columns = ['Applied']))
		FeatureSelectionTable = FeatureSelectionTable.append(pandas.DataFrame(data = ['yes', msData.Attributes['rsdThreshold'], 'yes'], index = ['Relative Standard Devation (RSD)', 'RSD of SP Samples: Threshold', 'RSD of SS Samples > RSD of SP Samples'], columns = ['Applied']))
		if withArtifactualFiltering:
			FeatureSelectionTable = FeatureSelectionTable.append(pandas.DataFrame(data = ['yes', msData.Attributes['deltaMzArtifactual'], msData.Attributes['overlapThresholdArtifactual'], msData.Attributes['corrThresholdArtifactual']],
			index = ['Artifactual Filtering', 'Artifactual Filtering: Delta m/z', 'Artifactual Filtering: Overlap Threshold', 'Artifactual Filtering: Correlation Threshold'], columns = ['Applied']))

		item['FeatureSelectionTable'] = FeatureSelectionTable
	
		if not output:
			print('Feature Selection Summary')
			print('Features selected based on:')
			display(item['FeatureSelectionTable'])
			print('\n')		
			
		# Figure 2: Final TIC
		if output:
			item['finalTIC'] = os.path.join(saveDir, item['Name'] + '_finalTIC.' + msData.Attributes['figureFormat'])
			saveAs = item['finalTIC']
		else:
			print('Figure 2: Total Ion Count (TIC) for all samples and all features in final dataset.')	
		
		plotTIC(msData, 
			addViolin=True,
			title='',
			savePath=saveAs,
			figureFormat=msData.Attributes['figureFormat'],
			dpi=msData.Attributes['dpi'],
			figureSize=msData.Attributes['figureSize'])


		# Figure 3: Histogram of log mean abundance by sample type
		if output:
			item['finalFeatureIntensityHist'] = os.path.join(saveDir, item['Name'] + '_finalFeatureIntensityHist.' + msData.Attributes['figureFormat'])
			saveAs = item['finalFeatureIntensityHist']
		else:
			print('Figure 3: Feature intensity histogram for all samples and all features in final dataset (by sample type)')

		_plotAbundanceBySampleType(msData.intensityData, SSmask, SPmask, ERmask, saveAs, msData) 


		# Figure 4: Histogram of RSDs in SP and SS
		if output:
			item['finalRSDdistributionFigure'] = os.path.join(saveDir, item['Name'] + '_finalRSDdistributionFigure.' + msData.Attributes['figureFormat'])
			saveAs = item['finalRSDdistributionFigure']
		else:
			print('Figure 4: Residual Standard Deviation (RSD) distribution for all samples and all features in final dataset (by sample type)')

		plotRSDs(msData,
			ratio=False,
			logx=True,
			color='matchReport',
			savePath=saveAs,
			figureFormat=msData.Attributes['figureFormat'],
			dpi=msData.Attributes['dpi'],
			figureSize=msData.Attributes['figureSize'])

		# Figure 5: Ion map
		if output:
			item['finalIonMap'] = os.path.join(saveDir, item['Name'] + '_finalIonMap.' + msData.Attributes['figureFormat'])
			saveAs = item['finalIonMap']
		else:
			print('Figure 5: Ion map of all features (coloured by log median intensity).')

		plotIonMap(msData, 
			savePath=saveAs, 
			figureFormat=msData.Attributes['figureFormat'],
			dpi=msData.Attributes['dpi'],
			figureSize=msData.Attributes['figureSize'])	

		# Figures 6 and 7: (if available) PCA scores and loadings plots by sample type
		##
		# PCA plots
		##
		if pcaModel:
			if output:
				pcaPath = saveDir
			else:
				pcaPath = None
			pcaModel = generateBasicPCAReport(pcaModel, msData, figureCounter=6, output=pcaPath, fileNamePrefix='')

		# Add final tables of excluded/missing study samples
		if not output:

			if (('SamplesMissingInfo' in item) | ('SamplesExcludedInfo' in item)):

				print('Samples Missing from Acquisition\n')

				if 'SamplesMissingInfo' in item:
					print('Samples unavailable for acquisition (' + item['SamplesMissingNo'] + ')')
					display(item['SamplesMissingInfo'])
					print('\n')			

				if 'SamplesExcludedInfo' in item:
					print('Samples excluded on analytical criteria (' + item['SamplesExcludedNo'] + ')')
					display(item['SamplesExcludedInfo'])
					print('\n')

	# Generate HTML report	
	if output:

		# Make paths for graphics local not absolute for use in the HTML.
		for key in item:
			if os.path.join(output, 'graphics') in str(item[key]):
				item[key] = re.sub('.*graphics', 'graphics', item[key])

		# Generate report
		from jinja2 import Environment, FileSystemLoader

		env = Environment(loader=FileSystemLoader(os.path.join(toolboxPath(), 'Templates')))
		template = env.get_template('generateReportMS.html')
		filename = os.path.join(output, msData.name + '_report_' + reportTypeCase + '.html')

		f = open(filename,'w')
		f.write(template.render(item=item,
								version=version,
								graphicsPath='/report_' + reportTypeCase,
								pcaPlots=pcaModel))
		f.close() 

		copyBackingFiles(toolboxPath(), saveDir)


def _plotAbundanceBySampleType(intensityData, SSmask, SPmask, ERmask, saveAs, msData):

	# Load toolbox wide color scheme
	if 'sampleTypeColours' in msData.Attributes.keys():
		sTypeColourDict = copy.deepcopy(msData.Attributes['sampleTypeColours'])
		for stype in SampleType:
			if stype.name in sTypeColourDict.keys():
				sTypeColourDict[stype] = sTypeColourDict.pop(stype.name)
	else:
		sTypeColourDict = {SampleType.StudySample: 'b', SampleType.StudyPool: 'g', SampleType.ExternalReference: 'r',
							SampleType.MethodReference: 'm', SampleType.ProceduralBlank: 'c', 'Other': 'grey'}

	meanIntensities = OrderedDict()
	temp = numpy.nanmean(intensityData[SSmask,:], axis=0)
	temp[numpy.isinf(temp)] = numpy.nan
	meanIntensities['Study Sample'] = temp
	colour = [sTypeColourDict[SampleType.StudySample]]
	if sum(SPmask) != 0: 
		temp = numpy.nanmean(intensityData[SPmask,:], axis=0)
		temp[numpy.isinf(temp)] = numpy.nan
		meanIntensities['Study Pool'] = temp
		colour.append(sTypeColourDict[SampleType.StudyPool])
	if sum(ERmask) != 0: 
		temp = numpy.nanmean(intensityData[ERmask,:], axis=0)
		temp[numpy.isinf(temp)] = numpy.nan
		meanIntensities['External Reference'] = temp
		colour.append(sTypeColourDict[SampleType.ExternalReference])

	histogram(meanIntensities, 
		xlabel='Mean Feature Intensity',
		color=colour,
		title='',
		histBins=msData.Attributes['histBins'],
		logx=True,
		savePath=saveAs, 
		figureFormat=msData.Attributes['figureFormat'],
		dpi=msData.Attributes['dpi'],
		figureSize=msData.Attributes['figureSize'])

def _localLRPlots(MSData, LRmask, corToLR, saveName, figures=None, savePath=None):

	# Plot TIC
	if savePath:
		saveTemp = saveName + ' LR Sample TIC (coloured by dilution)'			
		figures[saveTemp] = os.path.join(savePath, saveTemp + '.' + MSData.Attributes['figureFormat'])
		saveAs = figures[saveTemp]
	else:		
		print(saveName + ' LR Sample TIC (coloured by dilution)')
		saveAs = None;
		
	plotLRTIC(MSData, 
		sampleMask=LRmask,
		savePath=saveAs,
		figureFormat=MSData.Attributes['figureFormat'],
		dpi=MSData.Attributes['dpi'],
		figureSize=MSData.Attributes['figureSize'])


	# Plot TIC detector voltage change
	if savePath:
		saveTemp = saveName + ' LR Sample TIC (coloured by change in detector voltage)'			
		figures[saveTemp] = os.path.join(savePath, saveTemp + '.' + MSData.Attributes['figureFormat'])
		saveAs = figures[saveTemp]
	else:
		print(saveName + ' LR Sample TIC (coloured by change in detector voltage)')
		saveAs = None;

	plotLRTIC(MSData,
		sampleMask=LRmask,
		colourByDetectorVoltage=True,
		savePath=saveAs,
		figureFormat=MSData.Attributes['figureFormat'],
		dpi=MSData.Attributes['dpi'],
		figureSize=MSData.Attributes['figureSize'])


	# Plot histogram of correlation to dilution
	if savePath:
		saveTemp = saveName + ' Histogram of Correlation To Dilution'			
		figures[saveTemp] = os.path.join(savePath, saveTemp + '.' + MSData.Attributes['figureFormat'])
		saveAs = figures[saveTemp]
	else:
		print(saveName + ' Histogram of Correlation To Dilution')
		saveAs = None
		
	histogram(corToLR, 
		xlabel='Correlation to Dilution', 
		histBins=MSData.Attributes['histBins'],
		savePath=saveAs, 
		figureFormat=MSData.Attributes['figureFormat'],
		dpi=MSData.Attributes['dpi'],
		figureSize=MSData.Attributes['figureSize'])
		
	if figures is not None:
		return figures


def batchCorrectionTest(msData, nFeatures=10, window=11):

	import copy
	import numpy
	import random
	from ..batchAndROCorrection._batchAndROCorrection import _batchCorrection
	
	
	# Samplemask
	SSmask = (msData.sampleMetadata['SampleType'].values == SampleType.StudySample) & (msData.sampleMetadata['AssayRole'].values == AssayRole.Assay)
	SPmask = (msData.sampleMetadata['SampleType'].values == SampleType.StudyPool) & (msData.sampleMetadata['AssayRole'].values == AssayRole.PrecisionReference)
	ERmask = (msData.sampleMetadata['SampleType'].values == SampleType.ExternalReference) & (msData.sampleMetadata['AssayRole'].values == AssayRole.PrecisionReference)
	LRmask = (msData.sampleMetadata['SampleType'].values == SampleType.ExternalReference) & (msData.sampleMetadata['AssayRole'].values == AssayRole.LinearityReference)
	sampleMask = (SSmask | SPmask | ERmask | LRmask) & (msData.sampleMask==True).astype(bool)


	# Select subset of features (passing on correlation to dilution)
	
	# Correlation to dilution
	if not hasattr(msData, 'corrMethod'):
		msData.correlationToDilution

	# Exclude features failing correlation to dilution
	passMask = msData.correlationToDilution >= msData.Attributes['corrThreshold']
	
	# Exclude features with zero values
	zeroMask = sum(msData.intensityData[sampleMask,:]==0)
	zeroMask = zeroMask == 0
	
	passMask = passMask & zeroMask
	
	# Select subset of features on which to perform batch correction	
	maskNum = [i for i, x in enumerate(passMask) if x]
	random.shuffle(maskNum)
	
	# Do batch correction
	featureList = []
	correctedData = numpy.zeros([msData.intensityData.shape[0], nFeatures])
	fits = numpy.zeros([msData.intensityData.shape[0], nFeatures])
	featureIX = 0
	parameters = dict()
	parameters['window'] = window
	parameters['method'] = 'LOWESS'
	parameters['align'] = 'median'
	
	for feature in maskNum:
		correctedP = _batchCorrection(msData.intensityData[:,feature], 
						msData.sampleMetadata['Run Order'].values,
						SPmask,
						msData.sampleMetadata['Correction Batch'].values,
						range(0, 1), # All features
						parameters,
						0)
		
		if sum(numpy.isfinite(correctedP[0][1])) == msData.intensityData.shape[0]:
			correctedData[:,featureIX] = correctedP[0][1]
			fits[:,featureIX] = correctedP[0][2]
			featureList.append(feature)
			featureIX = featureIX + 1
		
		if featureIX == nFeatures:
			break

	# Create copy of msData and trim
	preData = copy.deepcopy(msData)	 
	preData.intensityData = msData.intensityData[:,featureList]
	preData.featureMetadata = msData.featureMetadata.loc[featureList,:]
	preData.featureMetadata.reset_index(drop=True, inplace=True)

	# Run batch correction
	postData = copy.deepcopy(preData)
	postData.intensityData = correctedData
	postData.fit = fits

	# Return results
	return preData, postData, featureList
