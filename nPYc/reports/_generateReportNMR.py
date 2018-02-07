# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 10:53:12 2017

@author: aahmed1
"""
from .._toolboxPath import toolboxPath
from ..utilities._internal import _copyBackingFiles as copyBackingFiles
from ..utilities._nmr import _qcCheckBaseline, _qcCheckWaterPeak
from ._generateSampleReport import _generateSampleReport
from pyChemometrics.ChemometricsPCA import ChemometricsPCA
from collections import OrderedDict
from ..plotting import plotScores, plotLoadings, plotWaterResonance, plotWaterResonanceInteractive, plotBaseline, plotBaselineInteractive, plotCalibration, plotCalibrationInteractive, plotLineWidthInteractive, histogram
from ..enumerations import AssayRole, SampleType
#copied from qc.py-may not need all
import os
import numpy
import pandas
import nPYc
from ..objects import NMRDataset
import seaborn as sns
import copy
import pandas as pd
from plotly.offline import iplot
from ..utilities._nmr import cutSec

from IPython.display import display
import re
import warnings
import shutil

from ..__init__ import __version__ as version

def _generateReportNMR(nmrDataTrue, reportType, withExclusions=False, output=None,  pcaModel=None):
	"""
	Summarise different aspects of an NMR dataset

	Generate reports for ``feature summary``,  ``feature selection``, or ``final report``
	
	* **'feature summary'** Generates feature summary report/ QC summary report, plots figures including those for feature calibration check against glucose or TSP, linewidth box plot and baseline/water peak plots.
	* **'final report'** Generates a summary of the final dataset, lists sample numbers present, a selection of figures summarising dataset quality, and a final list of samples missing from acquisition. 

	:param NMRDataset nmrDataTrue: NMRDataset to report on
	:param str reportType: Type of report to generate, one of ``feature summary``,  or ``final report``
	:param bool withExclusions: If ``True``, only report on features and samples not masked by the sample and feature masks
	:param output: If ``None`` plot interactively, otherwise save report to the path specified
	:type output: None or str
	"""
	# Check inputs
	if not isinstance(nmrDataTrue, NMRDataset):
		raise TypeError('nmrDataTrue must be an instance of NMRDataset')

	acceptAllOptions = {'feature summary', 'final report'}
	if not isinstance(reportType, str) & (reportType in acceptAllOptions):
		raise ValueError('reportType must be == ' + str(acceptAllOptions))

	if not isinstance(withExclusions, bool):		
		raise TypeError('withExclusions must be a bool')	

	if output is not None:
		if not isinstance(output, str):
			raise TypeError('output must be a string')

	if pcaModel is not None:
		if not isinstance(pcaModel, ChemometricsPCA):
			raise TypeError('pcaModel must be a ChemometricsPCA object')

	sns.set_style("whitegrid")

	# Create directory to save output 
	if output:
		if not os.path.exists(output):
			os.makedirs(output)
		if not os.path.exists(os.path.join(output, 'graphics')):
			os.makedirs(os.path.join(output, 'graphics'))
	else:
		saveDir = None

	# Apply sample/feature masks if exclusions to be applied
	nmrData = copy.deepcopy(nmrDataTrue)
	if withExclusions:
		nmrData.applyMasks()

	# Define sample masks
	SSmask = (nmrData.sampleMetadata['SampleType'].values == SampleType.StudySample) & (nmrData.sampleMetadata['AssayRole'].values == AssayRole.Assay)
	SPmask = (nmrData.sampleMetadata['SampleType'].values == SampleType.StudyPool) & (nmrData.sampleMetadata['AssayRole'].values == AssayRole.PrecisionReference)
	ERmask = (nmrData.sampleMetadata['SampleType'].values == SampleType.ExternalReference) & (nmrData.sampleMetadata['AssayRole'].values == AssayRole.PrecisionReference)
	[ns, nv] = nmrData.intensityData.shape

	# Set up template item and save required info
	item = dict() 
	item['Name'] = nmrData.name
	item['ReportType'] = reportType
	item['Nfeatures'] = str(nv)
	item['Nsamples'] = str(ns)
	item['SScount'] = str(sum(SSmask))
	item['SRcount'] = str(sum(SPmask))
	item['LTRcount'] = str(sum(ERmask))

	# Feature summary report
	if reportType == 'feature summary':
		"""
		Generates feature summary report, plots figures including those for calibration, peak width and baseline calculations.

		Generate NMR QC summary report.
	
		   :params: output directory to save report in
	
		   :returns: all graphs that are present in final report
	   
		"""	
			
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
			saveDir = None

		sampleSummary=[] #set t o empty for now, just to not include the above line ie tables from samplesummaryreport

		# Chemical shift calibration Check
		bounds = numpy.std(nmrData.sampleMetadata['Delta PPM']) * 3
		meanVal = numpy.mean(nmrData.sampleMetadata['Delta PPM'])
		# QC metrics - keep the simple one here but we can remove for latter to feature summary
		nmrData.sampleMetadata['calibrationPass'] = numpy.logical_or((nmrData.sampleMetadata['Delta PPM'] > meanVal - bounds),
																  (nmrData.sampleMetadata['Delta PPM'] < meanVal + bounds))

		# LineWidth quality check
		nmrData.sampleMetadata['LineWidthPass'] = nmrData.sampleMetadata['Line Width (Hz)'] <= nmrData.Attributes[
			'PWFailThreshold']

		# Check baseline
		baselineLowIndex = nmrData.Attributes['baselineCheckRegion'][0]
		baselineHighIndex = nmrData.Attributes['baselineCheckRegion'][1]
		specsLowBaselineRegion = nmrData.intensityData[nmrData.sampleMask, nmrData.featureMask & nmrData]
		specsHighBaselineRegion = nmrData.intensityData[nmrData.sampleMask, nmrData.featureMask & nmrData]

		isOutlierBaselineLow = _qcCheckBaseline(specsLowBaselineRegion)
		isOutlierBaselineHigh = _qcCheckBaseline(specsHighBaselineRegion)

		# Check water peaks
		ppmWaterLowIndex = nmrData.Attributes['waterPeakCheckRegion'][0]
		ppmWaterHighIndex = nmrData.Attributes['waterPeakCheckrRegion'][1]
		specsLowWaterRegion = nmrData.intensityData[nmrData.sampleMask, nmrData.featureMask & nmrData]
		specsHighWaterRegion = nmrData.intensityData[nmrData.sampleMask, nmrData.featureMask & nmrData]

		isOutlierWaterPeakLow = _qcCheckWaterPeak(specsLowWaterRegion)
		isOutlierWaterPeakHigh = _qcCheckWaterPeak(specsHighWaterRegion)

		graphsAndPlots(nmrData,saveDir, item, reportType, SSmask, SPmask, ERmask, pcaModel) #do not actually need SR,SS and LTR for this report

		#convert each mask whihc are lists to one dataframe with column headings
		tempDf = pd.DataFrame({'Sample_File_Name':(nmrData.sampleMetadata['Sample File Name'])})

		#reorder columns
		tempDf = tempDf[['Sample_File_Name','Import_Fail', 'PW_threshold_Fail', 'BL_low_outliersFailArea','BL_low_outliersFailNeg','BL_high_outliersFailArea','BL_high_outliersFailNeg','WP_low_outliersFailArea','WP_low_outliersFailNeg', 'WP_high_outliersFailArea', 'WP_high_outliersFailNeg','Calibration_Fail']]
		#convert to 0s and 1s rather than true false
		tempDf[['Import_Fail', 'PW_threshold_Fail', 'BL_low_outliersFailArea','BL_low_outliersFailNeg','BL_high_outliersFailArea','BL_high_outliersFailNeg','WP_low_outliersFailArea','WP_low_outliersFailNeg', 'WP_high_outliersFailArea', 'WP_high_outliersFailNeg', 'Calibration_Fail']] = tempDf[['Import_Fail', 'PW_threshold_Fail', 'BL_low_outliersFailArea','BL_low_outliersFailNeg','BL_high_outliersFailArea','BL_high_outliersFailNeg','WP_low_outliersFailArea','WP_low_outliersFailNeg', 'WP_high_outliersFailArea', 'WP_high_outliersFailNeg', 'Calibration_Fail']].astype(int)
		item['failSummary'] = tempDf.query('BL_low_outliersFailArea == True | BL_low_outliersFailNeg== True | BL_high_outliersFailArea== True | BL_high_outliersFailNeg== True | WP_low_outliersFailArea== True |WP_low_outliersFailNeg== True | WP_high_outliersFailArea== True | WP_high_outliersFailNeg== True | PW_threshold_Fail == True | Import_Fail == True | Calibration_Fail == True')
		#Import_Fail', 'PW_threshold_Fail', 'CritVal_BLWP_Exceed', 'Calibration_Fail
		del tempDf
		if not output:#we dont want to dosplay if we saving output
			print('Table 1: All samples that failed')
			display(item['failSummary'])

		if output:
	
			# Make paths for graphics local not absolute for use in the HTML.
			for key in item:
				if os.path.join(output, 'graphics') in str(item[key]):
					item[key] = re.sub('.*graphics', 'graphics', item[key])
	
			# Generate report
			from jinja2 import Environment, FileSystemLoader
	
			env = Environment(loader=FileSystemLoader(os.path.join(toolboxPath(), 'Templates')))
			template = env.get_template('NMR_QCSummaryReport.html')
	
			filename = os.path.join(output, nmrData.name + '_report_' + reportTypeCase + '.html')
	
			f = open(filename,'w')
			f.write(template.render(item=item, sampleSummary=sampleSummary,  version=version, graphicsPath='/report_' + reportTypeCase))
			f.close() 
	
			copyBackingFiles(toolboxPath(), saveDir)

		# Final summary report

	if reportType == 'final report':
		"""
		Generates a summary of the final dataset, lists sample numbers present, a selection of figures summarising dataset quality, and a final list of samples missing from acquisition.
		"""   
		
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
			saveDir = None
		
		#generate the final dataset   
		generateNMRFinalDataset(nmrData, output,sampleTypeOutput=False)
		
		# Table 1: Sample summary
										
		# Generate sample summary
		sampleSummary = _generateSampleReport(nmrData, withExclusions=True, output=None, returnOutput=True)
		
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
		#check if there is anything in the samplesummary table if there isnt report wont work as lims file didnt match any data so throw warning
		if numpy.isnan(sampleSummaryTable.Total.values[0]):
			warnings.warn("Warning:.The following reports will not generate possibly due to incorrect or incomplete LIMS file loaded/matched; check LIMS file and re-run")
			# Save details of study samples missing from dataset
		else:	
			if sampleSummaryTable['Unavailable']['Study Sample']  != 0:
				item['SamplesMissingInfo'] = sampleSummary['NotAcquired Details'].loc[sampleSummary['NotAcquired Details']['Sampling ID'].isnull()==False,:]
				item['SamplesMissingInfo'].drop(['LIMS Marked Missing'], axis=1, inplace=True)
				item['SamplesMissingNo'] = str(sampleSummaryTable['Unavailable']['Study Sample'] )
			
			 #Save details of study samples excluded from dataset
			if hasattr(sampleSummaryTable, 'Excluded'):
				if sampleSummaryTable['Excluded']['Study Sample']  != 0:
					item['SamplesExcludedInfo'] = sampleSummary['Excluded Details'].loc[(sampleSummary['Excluded Details']['SampleType'] == SampleType.StudySample) & (sampleSummary['Excluded Details']['AssayRole'] == AssayRole.Assay),:]
					item['SamplesExcludedInfo'] = item['SamplesExcludedInfo'].drop(['Sample Base Name', 'SampleType', 'AssayRole'], axis=1)
					item['SamplesExcludedNo'] = str(sampleSummaryTable['Excluded']['Study Sample'] )
				
			
	
			#item['Nsamples'] = nmrData.noSamples
			item['baselineLow_regionTo']=nmrData.Attributes['baselineLow_regionTo']
			item['baselineHigh_regionFrom']=nmrData.Attributes['baselineHigh_regionFrom']
			item['baseline_alpha']=nmrData.Attributes['baseline_alpha']
			item['baseline_threshold']=nmrData.Attributes['baseline_threshold']
			item['PWFailThreshold']=nmrData.Attributes['PWFailThreshold']
			item['points']=nmrData.intensityData.shape[1]
			item['alignTo']=nmrData.Attributes['alignTo']
			item['waterPeakCutRegionA']=nmrData.Attributes['waterPeakCutRegionA']
			item['waterPeakCutRegionB']=nmrData.Attributes['waterPeakCutRegionB']
			item['LWpeakRange']=nmrData.Attributes['LWpeakRange']
			#datetime code
			
			dt = min(nmrData.sampleMetadata['Acquired Time'])
			dd = dt.day, dt.month, dt.year
			dd=str(dd).replace(", ", "/")
			item['toA_from']=dd
		
			dt = max(nmrData.sampleMetadata['Acquired Time'])
			dd = dt.day, dt.month, dt.year
			dd=str(dd).replace(", ", "/")
			item['toA_to']=dd

			sampleSummary = _generateSampleReport(nmrData, withExclusions=True, output=None, returnOutput=True)
		
			item = graphsAndPlots(nmrData,saveDir, item, reportType, SSmask, SPmask, ERmask, pcaModel)#do not actually need SR,SS and LTR for this report

			if output:

				# Make paths for graphics local not absolute for use in the HTML.
				for key in item:
					if os.path.join(output, 'graphics') in str(item[key]):
						item[key] = re.sub('.*graphics', 'graphics', item[key])

				# Generate report
				from jinja2 import Environment, FileSystemLoader
		
				env = Environment(loader=FileSystemLoader(os.path.join(toolboxPath(), 'Templates')))
				template = env.get_template('NMR_FinalSummaryReport.html')
				filename = os.path.join(output, nmrData.name + '_report_' + reportTypeCase + '.html')
		
				f = open(filename,'w')
				f.write(template.render(item=item, sampleSummary=sampleSummary, version=version, graphicsPath='/report_' + reportTypeCase))
				f.close() 
		
				copyBackingFiles(toolboxPath(), saveDir)


def graphsAndPlots(nmrData,output, item, reportType, SSmask, SPmask, ERmask, PCAmodel):
	"""graphsAndPlots

	   :params: output directory to save report in nmrData, item and reportType

	   :returns: all graphs that are present for purpose of nmr report -- this is a separate function as to avoid repeating the code in final report and qcsummary report
	   
	"""	


		#figure 1 calibration check plot glucose(doublet) TSP singlet---------------------------------------------------------------------------------------------
	if output and reportType != 'final report':
		item['calibrationCheck'] = os.path.join(output, nmrData.name + '_calibrationCheck.' + nmrData.Attributes['figureFormat'])#.replace("\\","/")
		saveAs = item['calibrationCheck']
		
	else:
		if reportType != 'final report':
			print('Figure 1: Calibration Check Plot, aligned to:', nmrData.Attributes['alignTo'])
			saveAs = None
	
	if reportType != 'final report':
		if output:
			plotCalibration(nmrData,
				savePath=saveAs,
				figureFormat=nmrData.Attributes['figureFormat'],
				dpi=nmrData.Attributes['dpi'],
				figureSize=nmrData.Attributes['figureSize'])
		else:
			figure = plotCalibrationInteractive(nmrData)
			iplot(figure)

		#figure 2 peakwidth boxplot---------------------------------------------------------------------------------------------
	if output:
		
		item['peakWidthBoxplot'] = os.path.join(output, item['Name'] + '_peakWidthBoxplot.' + nmrData.Attributes['figureFormat'])#.replace("\\","/")
		saveAs = item['peakWidthBoxplot']
	else:
		if reportType == 'final report':
			print('Figure 1: Peak Width Boxplot (Hz)')
		else:
			print('Figure 2: Peak Width Boxplot (Hz)')
		saveAs = None
		
	nPYc.plotting.plotPW(nmrData,
			savePath=saveAs,
			figureFormat=nmrData.Attributes['figureFormat'],
			dpi=nmrData.Attributes['dpi'],
			figureSize=nmrData.Attributes['figureSize'])
	
	if not output:
		figure = plotLineWidthInteractive(nmrData)
		iplot(figure)

	#summary text ie how many failed so can place below plot output
	if reportType != 'final report':#why? because we have excluded all the failures by the time of final report so the final should not have any failures in 
		tempNo=sum(nmrData.sampleMetadata['Line Width (Hz)']>nmrData.Attributes['PWFailThreshold'])
		item['fig1SummaryText'] = str(tempNo)+' sample(s) failed (shown as red dots) based on peakwidth: >'+ str(nmrData.Attributes['PWFailThreshold'])+'Hz'
		del tempNo
	else:
		item['fig1SummaryText'] =''
	if not output:
		print(item['fig1SummaryText'])
		#figure 3
	if output:
#		item['finalFeatureBLWPplots1'] = os.path.join(output, 'graphics', item['Name'] + '_finalFeatureBLWPplots1.' + nmrData.Attributes['figureFormat']).replace("\\","/")
		item['finalFeatureBLWPplots1'] = os.path.join(output, item['Name'] + '_finalFeatureBLWPplots1.' + nmrData.Attributes['figureFormat'])#.replace("\\","/")
		saveAs = item['finalFeatureBLWPplots1']
	else:
		if reportType == 'final report':
			print('Figure 2: Baseline Low and High')
		else:
			print('Figure 3: Baseline Low and High')

		saveAs = None	
	areaToPlot='BL'
	
	if output:
		plotBaseline(nmrData,
					savePath=saveAs,
					figureFormat=nmrData.Attributes['figureFormat'],
					dpi=nmrData.Attributes['dpi'],
					figureSize=nmrData.Attributes['figureSize'])

	else:
		figure = plotBaselineInteractive(nmrData)
		iplot(figure)

	
		#figure 5 waterpeak low---------------------------------------------------------------------------------------------
	if output:
		item['finalFeatureBLWPplots3'] = os.path.join(output, item['Name'] + '_finalFeatureBLWPplots3.' + nmrData.Attributes['figureFormat'])#.replace("\\","/")
		saveAs = item['finalFeatureBLWPplots3']
		
	else:
		if reportType == 'final report':
			print('Figure 3: Waterpeak Low and High')
		else:
			print('Figure 4: Waterpeak Low and High')
		saveAs = None
	if output:
		plotWaterResonance(nmrData, margin=1,
									savePath=saveAs,
									figureFormat=nmrData.Attributes['figureFormat'],
									dpi=nmrData.Attributes['dpi'],
									figureSize=nmrData.Attributes['figureSize'])

	else:
		figure = plotWaterResonanceInteractive(nmrData)
		iplot(figure)

	#print failed amounts under plots again only if its not the final report as we excluded this data in final anyway
	if reportType != 'final report':
		failAreaLowCountBL=sum(nmrData.sampleMetadata.BL_low_outliersFailArea==True)
		failNegLowCountBL=sum(nmrData.sampleMetadata.BL_low_outliersFailNeg==True)
		failAreaHighCountBL=sum(nmrData.sampleMetadata.BL_high_outliersFailArea==True)
		failNegHighCountBL=sum(nmrData.sampleMetadata.BL_high_outliersFailNeg==True)
		failAreaLowCountWP=sum(nmrData.sampleMetadata.WP_low_outliersFailArea==True)
		failNegLowCountWP=sum(nmrData.sampleMetadata.WP_low_outliersFailNeg==True)
		failAreaHighCountWP=sum(nmrData.sampleMetadata.WP_high_outliersFailArea==True)
		failNegHighCountWP=sum(nmrData.sampleMetadata.WP_high_outliersFailNeg==True)

		item['fig2to3SummaryText']=(''+str(failAreaLowCountBL)+' sample(s) failed on BL low area'+'\n'+
										''+str(failNegLowCountBL)+' sample(s) failed on BL low negative'+'\n'+
										''+str(failAreaHighCountBL)+' sample(s) failed on BL high area'+'\n'+
										''+str(failNegHighCountBL)+' sample(s) failed on BL high negative'+'\n'+
										''+str(failAreaLowCountWP)+' sample(s) failed on WP low area'+'\n'+
										''+str(failNegLowCountWP)+' sample(s) failed on WP low negative'+'\n'+
										''+str(failAreaHighCountWP)+' sample(s) failed on WP high area'+'\n'+
										''+str(failNegHighCountWP)+' sample(s) failed on WP high negative'+'\n')
		
	else:
		item['fig2to3SummaryText']=''
	if not output:
		print(item['fig2to3SummaryText'])
		

	# Final histogram now only sum was previously of log mean abundance (by sample type)
	intensities = {}
	temp =numpy.nansum(nmrData.intensityData[SSmask,:], axis=1)

	# Load toolbox wide color scheme
	if 'sampleTypeColours' in nmrData.Attributes.keys():
		sTypeColourDict = copy.deepcopy(nmrData.Attributes['sampleTypeColours'])
		for stype in SampleType:
			if stype.name in sTypeColourDict.keys():
				sTypeColourDict[stype] = sTypeColourDict.pop(stype.name)
	else:
		sTypeColourDict = {SampleType.StudySample: 'b', SampleType.StudyPool: 'g', SampleType.ExternalReference: 'r',
						   SampleType.MethodReference: 'm', SampleType.ProceduralBlank: 'c', 'Other': 'grey'}

#	NEED TO CHECK OUTPUT WITH JAKE: changed all axis from 0 to 1
	intensities['Study Samples'] = temp
	colour = [sTypeColourDict[SampleType.StudySample]]

	if sum(SPmask) != 0: 
		temp = numpy.sum(nmrData.intensityData[SPmask,:], axis=1)
		temp[numpy.isinf(temp)] = numpy.nan
		intensities['Study Pool'] = temp
		colour.append(sTypeColourDict[SampleType.StudyPool])
	if sum(ERmask) != 0: 
		temp = numpy.sum(nmrData.intensityData[ERmask,:], axis=1)
		temp[numpy.isinf(temp)] = numpy.nan
		intensities['External Reference'] = temp
		colour.append(sTypeColourDict[SampleType.ExternalReference])
	if output:
		item['finalFeatureIntensityHist'] = os.path.join(output, item['Name'] + '_finalFeatureIntensityHist.' + nmrData.Attributes['figureFormat'])#.replace("\\","/")
		saveAs = item['finalFeatureIntensityHist']
	else:
		if reportType == 'final report':
			print('Figure 4: Feature Intensity Histogram for all samples and all features in final dataset (by sample type)')
		else:
			print('Figure 5: Feature Intensity Histogram for all samples and all features in final dataset (by sample type)')
	histogram(intensities, 
		xlabel='Feature Intensity',
		title='',
		color=colour,
		histBins=nmrData.Attributes['histBins'],
		savePath=saveAs, 
		figureFormat=nmrData.Attributes['figureFormat'],
		dpi=nmrData.Attributes['dpi'],
		figureSize=nmrData.Attributes['figureSize'])	

	#### pCA plot#####
# Figure 6: (if available) PCA scores plot by sample type
	if PCAmodel is not None:

		if not 'Plot Sample Type' in nmrData.sampleMetadata.columns:
			nmrData.sampleMetadata.loc[~SSmask & ~SPmask & ~ERmask, 'Plot Sample Type'] = 'Sample'
			nmrData.sampleMetadata.loc[SSmask, 'Plot Sample Type'] = 'Study Sample'
			nmrData.sampleMetadata.loc[SPmask, 'Plot Sample Type'] = 'Study Pool'
			nmrData.sampleMetadata.loc[ERmask, 'Plot Sample Type'] = 'External Reference'

		figuresQCscores = OrderedDict()
		temp = dict()
		if output:
			temp['PCA_scoresPlotFinal'] =os.path.join(output, item['Name'] + '_PCAscoresPlotFinal_.' + nmrData.Attributes['figureFormat'])#.replace("\\","/")
			saveAs = temp['PCA_scoresPlotFinal']
		else:
			print('Figure 6: PCA scores plots coloured by sample type.')

		figuresQCscores = plotScores(PCAmodel,
			classes=nmrData.sampleMetadata['Plot Sample Type'],
			classType = 'Plot Sample Type',
			title='Sample Type',
			savePath=saveAs,
			figures=figuresQCscores,
			figureFormat=nmrData.Attributes['figureFormat'],
			dpi=nmrData.Attributes['dpi'],
			figureSize=nmrData.Attributes['figureSize'])

		for key in figuresQCscores:
			if os.path.join('graphics', 'report_') in str(figuresQCscores[key]):
				figuresQCscores[key] = re.sub('.*graphics', 'graphics', figuresQCscores[key])
				
		item['QCscores'] = figuresQCscores
		
		figuresQCloadings = OrderedDict()			
		temp = dict()
		if output:
			temp['PCA_loadingsPlotFinal'] =os.path.join(output, item['Name'] + '_PCAloadingsPlotFinal_.' + nmrData.Attributes['figureFormat'])#.replace("\\","/")
			saveAs = temp['PCA_loadingsPlotFinal']
		else:
			print('Figure 7: PCA loadings plots.')
				
		figuresQCloadings = plotLoadings(PCAmodel,
			nmrData,
			savePath=saveAs,
			figures=figuresQCloadings,
			figureFormat=nmrData.Attributes['figureFormat'],
			dpi=nmrData.Attributes['dpi'],
			figureSize=nmrData.Attributes['figureSize'])

		for key in figuresQCloadings:
			if os.path.join('graphics', 'report_') in str(figuresQCloadings[key]):
				figuresQCloadings[key] = re.sub('.*graphics', 'graphics', figuresQCloadings[key])
				
		item['QCloadings'] = figuresQCloadings

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
					
	item['fig2to3SummaryText']=item['fig2to3SummaryText'].replace('\n', '<br />')#make it compatible with html
	return item

				
def generateNMRFinalDataset(nmrData, output,sampleTypeOutput=False):
	"""
	Produces the final dataset and encompasses the generatenmrfinalsummaryreport code, Export data corresponding to SS and SR and LTR samples only (to output all remove sampleTypeOutput argument) amd gemerate final report,
	params:
	input: 
	nmrData object		 
	directory to be saved in or set to None if want to display to screen in interactive mode
	sampletypeoutput ie SS,SR,LTR
	returns:
	output:
	final report as html saved file or on screen if in interactive mode
	final dataset object	
	"""
		# Sample Exclusions
	if sampleTypeOutput is not False:
		assert isinstance(sampleTypeOutput, list)
		
		sampleMask = numpy.zeros(nmrData.sampleMask.shape).astype(bool)
		
		if 'SS' in sampleTypeOutput:
			sampleMask[nmrData.sampleMetadata['Study Sample'].values==True] = True
		if 'SR' in sampleTypeOutput:
			sampleMask[nmrData.sampleMetadata['Study Reference'].values==True] = True			
		if 'LTR' in sampleTypeOutput:
			sampleMask[nmrData.sampleMetadata['Long-Term Reference'].values==True] = True				
			
	else:
		sampleMask = numpy.ones(nmrData.sampleMask.shape).astype(bool)

	nmrData.sampleMask = numpy.asarray((sampleMask & nmrData.sampleMask==True), 'bool')
