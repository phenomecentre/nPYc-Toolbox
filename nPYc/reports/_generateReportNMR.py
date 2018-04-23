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
from ..utilities._nmr import qcCheckBaseline, qcCheckWaterPeak
from ._generateSampleReport import _generateSampleReport
from ..plotting import plotWaterResonance, plotWaterResonanceInteractive, plotBaseline, plotBaselineInteractive, plotCalibration, plotCalibrationInteractive, plotLineWidthInteractive, histogram
from ._generateBasicPCAReport import generateBasicPCAReport
from ..enumerations import AssayRole, SampleType

from ..__init__ import __version__ as version

def _generateReportNMR(nmrData, reportType, withExclusions=True, output=None, pcaModel=None):
	"""
	Generate reports on NMRdataset objects, possible options are: ``feature summary`` or ``final report``
	
	* **'feature summary'** Generates feature summary report/ QC summary report, plots figures including those for feature calibration check against glucose or TSP, linewidth box plot and baseline/water peak plots.
	* **'final report'** Generates a summary of the final dataset, lists sample numbers present, a selection of figures summarising dataset quality, and a final list of samples missing from acquisition. 

	:param NMRDataset nmrData: NMRDataset to report on
	:param str reportType: Type of report to generate, one of ``feature summary``,  or ``final report``
	:param bool withExclusions: If ``True``, only report on features and samples not masked by the sample and feature masks
	:param output: If ``None`` plot interactively, otherwise save report to the path specified
	:type output: None or str
	"""
	acceptableOptions = {'feature summary', 'final report'}

	# Check inputs
	if not isinstance(nmrData, NMRDataset):
		raise TypeError('nmrData must be an instance of NMRDataset')

	if not isinstance(reportType, str) & (reportType.lower() in acceptableOptions):
		raise ValueError('reportType must be one of: ' + str(acceptableOptions))

	if not isinstance(withExclusions, bool):
		raise TypeError('withExclusions must be a bool')	

	if output is not None:
		if not isinstance(output, str):
			raise TypeError('output must be a string')

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
		nmrData.sampleMetadata.loc[SPmask, 'Plot Sample Type'] = 'Study Pool'
		nmrData.sampleMetadata.loc[ERmask, 'Plot Sample Type'] = 'External Reference'

	if reportType.lower() == 'feature summary':
		_featureReport(nmrData, output)
	elif reportType.lower() == 'final report':
		_finalReport(nmrData, output, pcaModel)


def _featureReport(dataset, output=None):
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
	if output is not None:
		if not os.path.exists(output):
			os.makedirs(output)
		if not os.path.exists(os.path.join(output, 'graphics')):
			os.makedirs(os.path.join(output, 'graphics'))
		graphicsPath = os.path.join(output, 'graphics', 'report_featureSummary')
		if not os.path.exists(graphicsPath):
			os.makedirs(graphicsPath)

	##
	# Chemical shift registration plot
	##
	if output:
		item['calibrationCheck'] = os.path.join(graphicsPath, dataset.name + '_calibrationCheck.' + dataset.Attributes['figureFormat'])
		saveAs = item['calibrationCheck']
	else:
		print('Figure 1: Calibration Check Plot, aligned to:', dataset.Attributes['alignTo'])
		saveAs = None
	
	if output:
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
	if output:
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
	if not output:
		print('Figure 2a: Peak Width Modeling')
		figure = plotLineWidthInteractive(dataset)
		iplot(figure)

	##
	# Baseline plot
	##
	if output:
		item['finalFeatureBLWPplots1'] = os.path.join(graphicsPath,
					item['Name'] + '_finalFeatureBLWPplots1.' + dataset.Attributes['figureFormat'])
		saveAs = item['finalFeatureBLWPplots1']
	else:
		print('Figure 3: Baseline Low and High')

		saveAs = None

	if output:
		plotBaseline(dataset,
					savePath=saveAs,
					figureFormat=dataset.Attributes['figureFormat'],
					dpi=dataset.Attributes['dpi'],
					figureSize=dataset.Attributes['figureSize'])

	else:
		figure = plotBaselineInteractive(dataset)
		iplot(figure)

	##
	# Water Peak plot
	##
	if output:
		item['finalFeatureBLWPplots3'] = os.path.join(graphicsPath,
						item['Name'] + '_finalFeatureBLWPplots3.' + dataset.Attributes['figureFormat'])
		saveAs = item['finalFeatureBLWPplots3']
		
	else:
		print('Figure 4: Waterpeak Low and High')
		saveAs = None

	if output:
		plotWaterResonance(dataset,	savePath=saveAs,
									figureFormat=dataset.Attributes['figureFormat'],
									dpi=dataset.Attributes['dpi'],
									figureSize=dataset.Attributes['figureSize'])
	else:
		figure = plotWaterResonanceInteractive(dataset)
		iplot(figure)
	##
	# exclusion summary
	##
	fail_summary = dataset.sampleMetadata.loc[:, ['Sample File Name', 'LineWidthFail',
												  'CalibrationFail', 'BaselineFail', 'WaterPeakFail']]
	fail_summary = fail_summary[(fail_summary.iloc[:, 1::] == 1).any(axis=1, bool_only=True)]

	if not output:
		print('Table 1: Summary of samples considered for exclusion')
		display(fail_summary)
	##
	# Write HTML if saving
	##
	if output:
		# Make paths for graphics local not absolute for use in the HTML.
		for key in item:
			if os.path.join(output, 'graphics') in str(item[key]):
				item[key] = re.sub('.*graphics', 'graphics', item[key])

		# Generate report
		from jinja2 import Environment, FileSystemLoader

		env = Environment(loader=FileSystemLoader(os.path.join(toolboxPath(), 'Templates')))
		template = env.get_template('NMR_QCSummaryReport.html')
		filename = os.path.join(output, dataset.name + '_report_featureSummary.html')

		f = open(filename,'w')
		f.write(template.render(item=item,
								attributes=dataset.Attributes,
								version=version,
								failSummary=fail_summary,
								graphicsPath=graphicsPath))
		f.close()

		copyBackingFiles(toolboxPath(), os.path.join(output, 'graphics'))


def _finalReport(dataset, output=None, pcaModel=None):
	"""
	Report on final dataset
	"""
	item = dict()
	item['Name'] = dataset.name
	item['Nsamples'] = dataset.noSamples

	item['toA_from'] = dataset.sampleMetadata['Acquired Time'].min().strftime('%b %d %Y')
	item['toA_to'] = dataset.sampleMetadata['Acquired Time'].max().strftime('%b %d %Y')

	# Generate sample Summary
	sampleSummary = _generateSampleReport(dataset, withExclusions=True, output=None, returnOutput=True)
	sampleSummary['isFinalReport'] = True
	item['sampleSummary'] = sampleSummary

	##
	# Report stats
	##
	if output is not None:
		if not os.path.exists(output):
			os.makedirs(output)
		if not os.path.exists(os.path.join(output, 'graphics')):
			os.makedirs(os.path.join(output, 'graphics'))
		graphicsPath = os.path.join(output, 'graphics', 'report_finalSummary')
		if not os.path.exists(graphicsPath):
			os.makedirs(graphicsPath)

	##
	# LW box plot
	##
	if output:
		item['peakWidthBoxplot'] = os.path.join(graphicsPath,
					item['Name'] + '_peakWidthBoxplot.' + dataset.Attributes['figureFormat'])
		saveAs = item['peakWidthBoxplot']
	else:
		print('Figure 1: Peak Width Boxplot (Hz)')
		saveAs = None
		
	nPYc.plotting.plotPW(dataset,
			savePath=saveAs,
			figureFormat=dataset.Attributes['figureFormat'],
			dpi=dataset.Attributes['dpi'],
			figureSize=dataset.Attributes['figureSize'])

	##
	# LW shape plot
	##
	if not output:
		print('Figure 1a: Peak Width Modeling')
		figure = plotLineWidthInteractive(dataset)
		iplot(figure)

	##
	# Baseline plot
	##
	if output:
		item['finalFeatureBLWPplots1'] = os.path.join(graphicsPath,
					item['Name'] + '_finalFeatureBLWPplots1.' + dataset.Attributes['figureFormat'])
		saveAs = item['finalFeatureBLWPplots1']
	else:
		print('Figure 2: Baseline Low and High')

		saveAs = None

	if output:
		plotBaseline(dataset,
					savePath=saveAs,
					figureFormat=dataset.Attributes['figureFormat'],
					dpi=dataset.Attributes['dpi'],
					figureSize=dataset.Attributes['figureSize'])

	else:
		figure = plotBaselineInteractive(dataset)
		iplot(figure)

	##
	# Water Peak plot
	##
	if output:
		item['finalFeatureBLWPplots3'] = os.path.join(graphicsPath,
						item['Name'] + '_finalFeatureBLWPplots3.' + dataset.Attributes['figureFormat'])
		saveAs = item['finalFeatureBLWPplots3']
		
	else:
		print('Figure 3: Waterpeak Low and High')
		saveAs = None

	if output:
		plotWaterResonance(dataset,	savePath=saveAs,
									figureFormat=dataset.Attributes['figureFormat'],
									dpi=dataset.Attributes['dpi'],
									figureSize=dataset.Attributes['figureSize'])
	else:
		figure = plotWaterResonanceInteractive(dataset)
		iplot(figure)

	##
	# PCA plots
	##
	if pcaModel:
		if output:
			pcaPath = output
		else:
			pcaPath = None
		pcaModel = generateBasicPCAReport(pcaModel, dataset, figureCounter=6, output=pcaPath, fileNamePrefix='')

	##
	# Sample summary
	##
	if not output:
		print('Table 1: Summary of samples present')
		display(sampleSummary['Acquired'])
		if 'StudySamples Exclusion Details' in sampleSummary:
			print('Table 2: Summary of samples excluded')
			display(sampleSummary['StudySamples Exclusion Details'])

	##
	# Write HTML if saving
	##
	if output:
		# Make paths for graphics local not absolute for use in the HTML.
		for key in item:
			if os.path.join(output, 'graphics') in str(item[key]):
				item[key] = re.sub('.*graphics', 'graphics', item[key])

		# Generate report
		from jinja2 import Environment, FileSystemLoader

		env = Environment(loader=FileSystemLoader(os.path.join(toolboxPath(), 'Templates')))
		template = env.get_template('NMR_FinalSummaryReport.html')
		filename = os.path.join(output, dataset.name + '_report_finalSummary.html')

		f = open(filename,'w')
		f.write(template.render(item=item,
								attributes=dataset.Attributes,
								version=version,
								sampleSummary=sampleSummary,
								graphicsPath=graphicsPath,
								pcaPlots=pcaModel)
								)
		f.close()

		copyBackingFiles(toolboxPath(),os.path.join(output, 'graphics'))
