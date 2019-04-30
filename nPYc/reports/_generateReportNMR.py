import os
import numpy
import pandas
import nPYc
import copy
import pandas
import re
import warnings
import shutil
import seaborn as sns
from plotly.offline import iplot
from IPython.display import display, HTML


from ..objects import NMRDataset
from .._toolboxPath import toolboxPath
from ..utilities._internal import _copyBackingFiles as copyBackingFiles
from ..utilities._nmr import qcCheckBaseline, qcCheckSolventPeak
from ._generateSampleReport import _generateSampleReport
from ..plotting import plotSolventResonance, plotSolventResonanceInteractive, plotBaseline, plotBaselineInteractive, plotCalibration, plotCalibrationInteractive, plotLineWidthInteractive, histogram
from ._generateBasicPCAReport import generateBasicPCAReport
from ..enumerations import AssayRole, SampleType

from ..__init__ import __version__ as version

def _generateReportNMR(nmrData, reportType, withExclusions=True, destinationPath=None, pcaModel=None):
	"""
	Generate reports on NMRdataset objects, possible options are: ``feature summary`` or ``final report``
	
	* **'feature summary'** Generates feature summary report/ QC summary report, plots figures including those for feature calibration check against glucose or TSP, linewidth box plot and baseline/water peak plots.
	* **'final report'** Generates a summary of the final dataset, lists sample numbers present, a selection of figures summarising dataset quality, and a final list of samples missing from acquisition. 

	:param NMRDataset nmrData: NMRDataset to report on
	:param str reportType: Type of report to generate, one of ``feature summary``,  or ``final report``
	:param bool withExclusions: If ``True``, only report on features and samples not masked by the sample and feature masks
	:param destinationPath: If ``None`` plot interactively, otherwise save report to the path specified
	:type destinationPath: None or str
	"""
	acceptableOptions = {'feature summary', 'final report'}

	# Check inputs
	if not isinstance(nmrData, NMRDataset):
		raise TypeError('nmrData must be an instance of NMRDataset')

	if not isinstance(reportType, str) & (reportType.lower() in acceptableOptions):
		raise ValueError('reportType must be one of: ' + str(acceptableOptions))

	if not isinstance(withExclusions, bool):
		raise TypeError('withExclusions must be a bool')	

	if destinationPath is not None:
		if not isinstance(destinationPath, str):
			raise TypeError('destinationPath must be a string')

	sns.set_style("whitegrid")

	# Create directory to save destinationPath 
	if destinationPath:
		if not os.path.exists(destinationPath):
			os.makedirs(destinationPath)
		if not os.path.exists(os.path.join(destinationPath, 'graphics')):
			os.makedirs(os.path.join(destinationPath, 'graphics'))

	# Apply sample/feature masks if exclusions to be applied
	nmrData = copy.deepcopy(nmrData)
	if withExclusions:
		nmrData.applyMasks()

	# Define sample masks
	SSmask = (nmrData.sampleMetadata['SampleType'].values == SampleType.StudySample) & (nmrData.sampleMetadata['AssayRole'].values == AssayRole.Assay)
	SPmask = (nmrData.sampleMetadata['SampleType'].values == SampleType.StudyPool) & (nmrData.sampleMetadata['AssayRole'].values == AssayRole.PrecisionReference)
	ERmask = (nmrData.sampleMetadata['SampleType'].values == SampleType.ExternalReference) & (nmrData.sampleMetadata['AssayRole'].values == AssayRole.PrecisionReference)

	if not 'Plot Sample Type' in nmrData.sampleMetadata.columns:
		nmrData.sampleMetadata.loc[~SSmask & ~SPmask & ~ERmask, 'Plot Sample Type'] = 'Sample'
		nmrData.sampleMetadata.loc[SSmask, 'Plot Sample Type'] = 'Study Sample'
		nmrData.sampleMetadata.loc[SPmask, 'Plot Sample Type'] = 'Study Reference'
		nmrData.sampleMetadata.loc[ERmask, 'Plot Sample Type'] = 'Long-Term Reference'

	if reportType.lower() == 'feature summary':
		_featureReport(nmrData, destinationPath=destinationPath)
	elif reportType.lower() == 'final report':
		_finalReport(nmrData, destinationPath=destinationPath, pcaModel=pcaModel)


def _featureReport(dataset, destinationPath=None):
	"""
	Report on feature quality
	"""
	item = dict()
	item['Name'] = dataset.name
	item['Nsamples'] = dataset.noSamples

	item['toA_from'] = dataset.sampleMetadata['Acquired Time'].min().strftime('%b %d %Y')
	item['toA_to'] = dataset.sampleMetadata['Acquired Time'].max().strftime('%b %d %Y')
	
	##
	# Report stats
	##
	if destinationPath is not None:
		graphicsPath = os.path.join(destinationPath, 'graphics', 'report_featureSummary')
		if not os.path.exists(graphicsPath):
			os.makedirs(graphicsPath)

	##
	# Chemical shift registration plot
	##
	if destinationPath:
		item['calibrationCheck'] = os.path.join(graphicsPath, dataset.name + '_calibrationCheck.' + dataset.Attributes['figureFormat'])
		saveAs = item['calibrationCheck']
	else:
		print('Figure 1: Calibration Check Plot, aligned to:', dataset.Attributes['alignTo'])

	
	if destinationPath:
		plotCalibration(dataset,
			savePath=saveAs,
			figureFormat=dataset.Attributes['figureFormat'],
			dpi=dataset.Attributes['dpi'],
			figureSize=dataset.Attributes['figureSize'])
	else:
		figure = plotCalibrationInteractive(dataset)
		iplot(figure)
	##
	# LW box plot
	##
	if destinationPath:
		item['peakWidthBoxplot'] = os.path.join(graphicsPath,
					item['Name'] + '_peakWidthBoxplot.' + dataset.Attributes['figureFormat'])
		saveAs = item['peakWidthBoxplot']
	else:
		print('Figure 2: Peak Width Boxplot (Hz)')
		saveAs = None
		
	nPYc.plotting.plotPW(dataset,
			savePath=saveAs,
			figureFormat=dataset.Attributes['figureFormat'],
			dpi=dataset.Attributes['dpi'],
			figureSize=dataset.Attributes['figureSize'])

	##
	# LW shape plot
	##
	if not destinationPath:
		print('Figure 2a: Peak Width Modeling')
		figure = plotLineWidthInteractive(dataset)
		iplot(figure)

	##
	# Baseline plot
	##
	if destinationPath:
		item['finalFeatureBLWPplots1'] = os.path.join(graphicsPath,
					item['Name'] + '_finalFeatureBLWPplots1.' + dataset.Attributes['figureFormat'])
		saveAs = item['finalFeatureBLWPplots1']
	else:
		print('Figure 3: Baseline Low and High')

		saveAs = None

	if destinationPath:
		plotBaseline(dataset,
					savePath=saveAs,
					figureFormat=dataset.Attributes['figureFormat'],
					dpi=dataset.Attributes['dpi'],
					figureSize=dataset.Attributes['figureSize'])

	else:
		figure = plotBaselineInteractive(dataset)
		iplot(figure)

	##
	# Solvent Peak plot
	##
	if destinationPath:
		item['finalFeatureBLWPplots3'] = os.path.join(graphicsPath,
						item['Name'] + '_finalFeatureBLWPplots3.' + dataset.Attributes['figureFormat'])
		saveAs = item['finalFeatureBLWPplots3']
		
	else:
		print('Figure 4: Solvent peak Low and High')
		saveAs = None

	if destinationPath:
		plotSolventResonance(dataset,	savePath=saveAs,
									figureFormat=dataset.Attributes['figureFormat'],
									dpi=dataset.Attributes['dpi'],
									figureSize=dataset.Attributes['figureSize'])
	else:
		figure = plotSolventResonanceInteractive(dataset)
		iplot(figure)
	##
	# exclusion summary
	##
	dataset._nmrQCChecks()

	fail_summary = dataset.sampleMetadata.loc[:, ['Sample File Name', 'LineWidthFail',
												  'CalibrationFail', 'BaselineFail', 'SolventPeakFail']]
	fail_summary = fail_summary[(fail_summary.iloc[:, 1::] == 1).any(axis=1, bool_only=True)]

	if not destinationPath:
		print('Table 1: Summary of samples considered for exclusion')
		display(fail_summary)
	##
	# Write HTML if saving
	##
	if destinationPath:
		# Make paths for graphics local not absolute for use in the HTML.
		for key in item:
			if os.path.join(destinationPath, 'graphics') in str(item[key]):
				item[key] = re.sub('.*graphics', 'graphics', item[key])

		# Generate report
		from jinja2 import Environment, FileSystemLoader

		env = Environment(loader=FileSystemLoader(os.path.join(toolboxPath(), 'Templates')))
		template = env.get_template('NMR_QCSummaryReport.html')
		filename = os.path.join(destinationPath, dataset.name + '_report_featureSummary.html')

		f = open(filename,'w')
		f.write(template.render(item=item,
								attributes=dataset.Attributes,
								version=version,
								failSummary=fail_summary,
								graphicsPath=graphicsPath))
		f.close()

		copyBackingFiles(toolboxPath(), os.path.join(destinationPath, 'graphics'))


def _finalReport(dataset, destinationPath=None, pcaModel=None):
	"""
	Report on final dataset
	"""
    
    # Create save directory if required
	if destinationPath is not None:
		graphicsPath = os.path.join(destinationPath, 'graphics', 'report_finalSummary')
		if not os.path.exists(graphicsPath):
			os.makedirs(graphicsPath)
	else:
		saveAs = None
    
    
	item = dict()
	item['Name'] = dataset.name
	item['Nsamples'] = dataset.noSamples
	item['Nfeatures'] = dataset.noFeatures

	item['start'] = dataset.sampleMetadata['Acquired Time'].min().strftime('%b %d %Y')
	item['end'] = dataset.sampleMetadata['Acquired Time'].max().strftime('%b %d %Y')

    
    # Table 1: Sample summary

    # Generate sample summary
    
	sampleSummary = _generateSampleReport(dataset, withExclusions=True, destinationPath=None, returnOutput=True)
    
    # Tidy table for final report format
	sampleSummary['Acquired'].drop('Marked for Exclusion', inplace=True, axis=1)
    
	if hasattr(sampleSummary['Acquired'], 'Already Excluded'):
		sampleSummary['Acquired'].rename(columns={'Already Excluded': 'Excluded'}, inplace=True)

	sampleSummary['isFinalReport'] = True
	if 'StudySamples Exclusion Details' in sampleSummary:
		sampleSummary['studySamplesExcluded'] = True
	else:
		sampleSummary['studySamplesExcluded'] = False
	item['sampleSummary'] = sampleSummary

	if not destinationPath:
		print('Final Dataset')
		print('\n' + str(item['Nsamples']) + ' samples')        
		print(str(item['Nfeatures']) + ' features')     
		print('\nSample Summary')      
		print('\nTable 1: Summary of samples present.')
		display(sampleSummary['Acquired'])
		print('\nDetails of any missing/excluded study samples given at the end of the report\n')


	# Table 2: data processed with these parameters
	dataParametersTable = pandas.DataFrame(
			data = [dataset.Attributes['calibrateTo'], dataset.Attributes['variableSize'], dataset.Attributes['baselineCheckRegion'],
				   dataset.Attributes['solventPeakCheckRegion'], dataset.Attributes['LWpeakRange'], dataset.Attributes['LWpeakMultiplicity'],
				   dataset.Attributes['PWFailThreshold'], dataset.Attributes['exclusionRegions']],
			index = ['Referenced to (ppm)', 'Spectral Resolution (points)', 'Baseline Region Checked (ppm)', 
					'Solvent Suppresion Region Checked (ppm)', 'Line Width: Calculated on', 'Line Width: Peak Multiplicity',
					'Line Width: Threshold (Hz)', 'Spectral Regions Automatically Removed (ppm)'],
			columns = ['Value Applied']
			)
	item['DataParametersTable'] = dataParametersTable
	
	if not destinationPath:
		print('Spectral Data')
		print('\nTable 2: Data processed with the following criteria:')
		display(dataParametersTable)
		print('\nSamples acquired between ' + item['start'] + ' and ' + item['end'] + '\n')	

	##
	# LW box plot
	##
	if destinationPath:
		item['linewidthBoxplot'] = os.path.join(graphicsPath, item['Name'] + '_linewidthBoxplot.' + dataset.Attributes['figureFormat'])
		saveAs = item['linewidthBoxplot']
	else:
		print('Figure 1: Boxplot of line width distributions (by sample type).')
		
		
	nPYc.plotting.plotPW(dataset,
			savePath=saveAs,
			title='',
			figureFormat=dataset.Attributes['figureFormat'],
			dpi=dataset.Attributes['dpi'],
			figureSize=dataset.Attributes['figureSize'])

	##
	# Solvent Peak plot
	##
	if destinationPath:
		item['spectraSolventPeakRegion'] = os.path.join(graphicsPath, item['Name'] + '_spectraSolventPeakRegion.' + dataset.Attributes['figureFormat'])
		saveAs = item['spectraSolventPeakRegion']
		
		plotSolventResonance(dataset,
						 savePath=saveAs,
						 figureFormat=dataset.Attributes['figureFormat'],
						 dpi=dataset.Attributes['dpi'],
						 figureSize=dataset.Attributes['figureSize'])
				
	else:
		print('Figure 2: Distribution in intensity of spectral data around the removed solvent peak region.')
		figure = plotSolventResonanceInteractive(dataset, title='')
		iplot(figure)

	##
	# PCA plots
	##
	if pcaModel:
		if destinationPath:
			pcaPath = destinationPath
		else:
			pcaPath = None
		pcaModel = generateBasicPCAReport(pcaModel, dataset, figureCounter=3, destinationPath=pcaPath, fileNamePrefix='')

	##
	#  Table 3: Summary of samples excluded
	##
	if not destinationPath:
		if 'StudySamples Exclusion Details' in sampleSummary:
			print('Missing/Excluded Study Samples')
			print('\nTable 3: Details of missing/excluded study samples')
			display(sampleSummary['StudySamples Exclusion Details'])
			
	##
	# Write HTML if saving
	##
	if destinationPath:
		# Make paths for graphics local not absolute for use in the HTML.
		for key in item:
			if os.path.join(destinationPath, 'graphics') in str(item[key]):
				item[key] = re.sub('.*graphics', 'graphics', item[key])

		# Generate report
		from jinja2 import Environment, FileSystemLoader

		env = Environment(loader=FileSystemLoader(os.path.join(toolboxPath(), 'Templates')))
		template = env.get_template('NMR_FinalSummaryReport.html')
		filename = os.path.join(destinationPath, dataset.name + '_report_finalSummary.html')

		f = open(filename,'w')
		f.write(template.render(item=item,
								version=version,
								sampleSummary=sampleSummary,
								graphicsPath=graphicsPath,
								pcaPlots=pcaModel)
								)
		f.close()

		copyBackingFiles(toolboxPath(),os.path.join(destinationPath, 'graphics'))
