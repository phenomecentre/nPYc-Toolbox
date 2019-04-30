import scipy
import pandas
import numpy
import sys
import unittest
import unittest.mock
import tempfile
import os
import io
import copy
import warnings
import random
import string
import plotly
from datetime import datetime, timedelta

sys.path.append("..")
import nPYc
from nPYc.enumerations import VariableType, AssayRole, SampleType, QuantificationType, CalibrationMethod
from generateTestDataset import generateTestDataset


class test_plotting(unittest.TestCase):

	def setUp(self):
		# Feature1 has lowest LLOQ and ULOQ in batch1, feature2 has lowest LLOQ and ULOQ in batch2
		# On feature1 and feature2, Sample1 will be <LLOQ, Sample2 >ULOQ, Sample3 same as input
		self.targetedDataset = nPYc.TargetedDataset('', fileType='empty')
		self.targetedDataset.name = 'unittest'
		self.targetedDataset.sampleMetadata = pandas.DataFrame({'Sample File Name': ['UnitTest_targeted_file_001',
																					 'UnitTest_targeted_file_002',
																					 'UnitTest_targeted_file_003'],
																'Sample Name': ['Sample1-B1', 'Sample2-B2',
																				'Sample3-B2'],
																'Sample Type': ['Analyte', 'Analyte', 'Analyte'],
																'Acqu Date': ['10-Sep-16', '10-Sep-16', '10-Sep-16'],
																'Acqu Time': ['03:23:02', '04:52:35', '05:46:40'],
																'Vial': ['1:A,2', '1:A,3', '1:A,4'],
																'Instrument': ['XEVO-TQS#UnitTest', 'XEVO-TQS#UnitTest',
																			   'XEVO-TQS#UnitTest'],
																'Acquired Time': [datetime(2016, 9, 10, 3, 23, 2),
																				  datetime(2016, 9, 10, 4, 52, 35),
																				  datetime(2016, 9, 10, 5, 46, 40)],
																'Run Order': [0, 1, 2], 'Batch': [1, 2, 2],
																'AssayRole': [AssayRole.Assay, AssayRole.Assay,
																			  AssayRole.Assay],
																'SampleType': [SampleType.StudySample,
																			   SampleType.StudySample,
																			   SampleType.StudySample],
																'Dilution': [numpy.nan, numpy.nan, numpy.nan],
																'Correction Batch': [numpy.nan, numpy.nan, numpy.nan],
																'Subject ID': ['', '', ''], 'Sampling ID': ['', '', ''],
																'Sample Base Name': ['', '', ''],
																'Exclusion Details': ['', '', '']})
		self.targetedDataset.sampleMetadata['Acquired Time'] = self.targetedDataset.sampleMetadata['Acquired Time'].dt.to_pydatetime()
		self.targetedDataset.featureMetadata = pandas.DataFrame({'Feature Name': ['Feature1', 'Feature2'], 'TargetLynx Feature ID': [1, 2],'calibrationMethod': [CalibrationMethod.backcalculatedIS,CalibrationMethod.backcalculatedIS],'quantificationType': [QuantificationType.QuantAltLabeledAnalogue,QuantificationType.QuantAltLabeledAnalogue],'unitCorrectionFactor': [1., 1],'Unit': ['a Unit', 'pg/uL'],'Cpd Info': ['info cpd1', 'info cpd2'],'LLOQ_batch1': [5., 20.],'LLOQ_batch2': [20., 5.],'ULOQ_batch1': [80., 100.],'ULOQ_batch2': [100., 80.],'LLOQ': [20., 20.], 'ULOQ': [80., 80.],'extID1': ['F1', 'F2'],'extID2': ['ID1', 'ID2']})
		self.targetedDataset._intensityData = numpy.array([[10., 10.], [90., 90.], [25., 70.]])
		self.targetedDataset.expectedConcentration = pandas.DataFrame(numpy.array([[40., 60.], [40., 60.], [40., 60.]]), columns=self.targetedDataset.featureMetadata['Feature Name'].values.tolist())
		self.targetedDataset.sampleMetadataExcluded = []
		self.targetedDataset.featureMetadataExcluded = []
		self.targetedDataset.intensityDataExcluded = []
		self.targetedDataset.expectedConcentrationExcluded = []
		self.targetedDataset.excludedFlag = []
		self.targetedDataset.calibration = dict()
		self.targetedDataset.calibration['calibIntensityData'] = numpy.ndarray((0, 2))
		self.targetedDataset.calibration['calibSampleMetadata'] = pandas.DataFrame()
		self.targetedDataset.calibration['calibFeatureMetadata'] = pandas.DataFrame(index=['Feature1', 'Feature2'], columns=['Feature Name'])
		self.targetedDataset.calibration['calibExpectedConcentration'] = pandas.DataFrame(columns=['Feature1', 'Feature2'])
		self.targetedDataset.Attributes['methodName'] = 'unittest'
		self.targetedDataset.Attributes['externalID'] = ['extID1', 'extID2']
		self.targetedDataset.VariableType = VariableType.Continuum
		self.targetedDataset.initialiseMasks()


	def test_plotspectralvariance(self):

		noSamp = numpy.random.randint(50, high=100, size=None)
		noFeat = numpy.random.randint(200, high=400, size=None)

		dataset = generateTestDataset(noSamp, noFeat, dtype='NMRDataset', variableType=nPYc.enumerations.VariableType.Spectral)

		with tempfile.TemporaryDirectory() as tmpdirname:
			##
			# Basic output
			##
			outputPath = os.path.join(tmpdirname, 'default')
			nPYc.plotting.plotSpectralVariance(dataset, savePath=outputPath)

			self.assertTrue(os.path.exists(outputPath))

			##
			# Classes
			##
			outputPath = os.path.join(tmpdirname, 'classesAndXlim')
			nPYc.plotting.plotSpectralVariance(dataset, classes='Classes', xlim=(1,9), average='mean', savePath=outputPath)

			self.assertTrue(os.path.exists(outputPath))

			##
			# Loq Y + title + NMR
			##
			outputPath = os.path.join(tmpdirname, 'asNMRDataset')
			dataset.__class__ = nPYc.objects._nmrDataset.NMRDataset
			nPYc.plotting.plotSpectralVariance(dataset, logy=True, title='Figure Name', quantiles=None, savePath=outputPath)

			self.assertTrue(os.path.exists(outputPath))


	def test_plotspectralvariance_raises(self):

		with self.subTest(msg='Wrong Type'):
			self.assertRaises(TypeError, nPYc.plotting.plotSpectralVariance, 1)

		with self.subTest(msg='Discrete Data'):
			dataset = nPYc.Dataset()
			dataset.VariableType = nPYc.enumerations.VariableType.Discrete
			self.assertRaises(ValueError, nPYc.plotting.plotSpectralVariance, dataset)

		with self.subTest(msg='Wrong length quantiles'):
			dataset = nPYc.Dataset()
			dataset.VariableType = nPYc.enumerations.VariableType.Continuum
			self.assertRaises(ValueError, nPYc.plotting.plotSpectralVariance, dataset, quantiles=(1,2,3))

		with self.subTest(msg='Classes not found'):
			dataset = nPYc.Dataset()
			dataset.VariableType = nPYc.enumerations.VariableType.Continuum
			dataset.sampleMetadata = pandas.DataFrame(0, index=numpy.arange(5), columns=['Classes'])
			self.assertRaises(ValueError, nPYc.plotting.plotSpectralVariance, dataset, classes='Not present')


	def test_plotrsds(self):

		noSamp = numpy.random.randint(50, high=100, size=None)
		noFeat = numpy.random.randint(200, high=400, size=None)

		with tempfile.TemporaryDirectory() as tmpdirname:
			##
			# Test without LTR
			##
			msData = generateTestDataset(noSamp, noFeat, dtype='MSDataset')

			msData.sampleMetadata['SampleType'] = nPYc.enumerations.SampleType.StudySample
			msData.sampleMetadata['AssayRole'] = nPYc.enumerations.AssayRole.Assay

			msData.sampleMetadata.iloc[::10, 1] = nPYc.enumerations.SampleType.StudyPool
			msData.sampleMetadata.iloc[::10, 2] = nPYc.enumerations.AssayRole.PrecisionReference

			outputPath = os.path.join(tmpdirname, 'withoutLTR')
			nPYc.plotting.plotRSDs(msData, savePath=outputPath)

			self.assertTrue(os.path.exists(outputPath))

			##
			# Too few LTR
			##
			msData = generateTestDataset(noSamp, noFeat, dtype='MSDataset')

			msData.sampleMetadata['SampleType'] = nPYc.enumerations.SampleType.StudySample
			msData.sampleMetadata['AssayRole'] = nPYc.enumerations.AssayRole.Assay

			msData.sampleMetadata.iloc[::10, 1] = nPYc.enumerations.SampleType.StudyPool
			msData.sampleMetadata.iloc[::10, 2] = nPYc.enumerations.AssayRole.PrecisionReference
			msData.sampleMetadata.iloc[1:2, 1] = nPYc.enumerations.SampleType.ExternalReference
			msData.sampleMetadata.iloc[1:2, 2] = nPYc.enumerations.AssayRole.PrecisionReference

			outputPath = os.path.join(tmpdirname, 'tooFewLTR')
			nPYc.plotting.plotRSDs(msData, savePath=outputPath)

			self.assertTrue(os.path.exists(outputPath))

			##
			# Test with LTR
			##
			msData = generateTestDataset(noSamp, noFeat, dtype='MSDataset')

			outputPath = os.path.join(tmpdirname, 'withLTR')
			nPYc.plotting.plotRSDs(msData, savePath=outputPath)

			self.assertTrue(os.path.exists(outputPath))

			##
			# Test without xlog
			##
			outputPath = os.path.join(tmpdirname, 'withoutXlog')
			nPYc.plotting.plotRSDs(msData, logx=False, savePath=outputPath)

			self.assertTrue(os.path.exists(outputPath))

			##
			# By ratio
			##
			outputPath = os.path.join(tmpdirname, 'byRatio')
			nPYc.plotting.plotRSDs(msData, ratio=True, savePath=outputPath)

			self.assertTrue(os.path.exists(outputPath))

			##
			# Matching report colours
			##
			outputPath = os.path.join(tmpdirname, 'matchReport')
			nPYc.plotting.plotRSDs(msData, color='matchReport', savePath=outputPath)

			self.assertTrue(os.path.exists(outputPath))

			##
			# set xlim
			##
			outputPath = os.path.join(tmpdirname, 'xlim')
			nPYc.plotting.plotRSDs(msData, xlim=(min(msData.rsdSP) - 10, max(msData.rsdSP) + 10), savePath=outputPath)

			self.assertTrue(os.path.exists(outputPath))


	def test_plotionmap_raises(self):

		with self.subTest(msg='Not an MS dataset'):
			self.assertRaises(TypeError, nPYc.plotting.plotIonMap, 1)

		with self.subTest(msg='No m/z column'):
			dataset = generateTestDataset(10, 10, dtype='MSDataset')
			dataset.featureMetadata.drop('m/z', 1, inplace=True)

			self.assertRaises(KeyError, nPYc.plotting.plotIonMap, dataset)


	def test_plot_histogram(self):

		noFeat = numpy.random.randint(900, high=1400, size=None)

		values = numpy.random.randn(1, noFeat)
		inclusionVector = numpy.random.randn(1, noFeat)

		with tempfile.TemporaryDirectory() as tmpdirname:
			outputPath = os.path.join(tmpdirname, 'basic')
			nPYc.plotting.histogram(values, quantiles=[10, 40, 60, 90], savePath=outputPath)
			self.assertTrue(os.path.exists(outputPath))

			outputPath = os.path.join(tmpdirname, 'withIncV')
			nPYc.plotting.histogram(values, inclusionVector=inclusionVector, savePath=outputPath)
			self.assertTrue(os.path.exists(outputPath))

			outputPath = os.path.join(tmpdirname, 'logxBroken')
			nPYc.plotting.histogram(values, logy=True, logx=True, savePath=outputPath)
			self.assertTrue(os.path.exists(outputPath))

			outputPath = os.path.join(tmpdirname, 'logx')
			values = (values - numpy.min(values)) + 1
			nPYc.plotting.histogram(values, logx=True, xlim=(2, 5), savePath=outputPath)
			self.assertTrue(os.path.exists(outputPath))

			outputPath = os.path.join(tmpdirname, 'logxWithZero')
			values = numpy.append(0, values)
			nPYc.plotting.histogram(values, inclusionVector=inclusionVector, logx=True, savePath=outputPath)
			self.assertTrue(os.path.exists(outputPath))


	def test_plot_histogram_raises(self):

		self.assertRaises(ValueError, nPYc.plotting.histogram, dict(), [1, 2])


	def test_plot_plotionmap(self):

		noSamp = numpy.random.randint(50, high=100, size=None)
		noFeat = numpy.random.randint(200, high=400, size=None)
		msData = generateTestDataset(noSamp, noFeat, dtype='MSDataset')

		with tempfile.TemporaryDirectory() as tmpdirname:

			outputPath = os.path.join(tmpdirname, 'basic')
			nPYc.plotting.plotIonMap(msData, savePath=outputPath)
			self.assertTrue(os.path.exists(outputPath))

			outputPath = os.path.join(tmpdirname, 'withLimits')
			nPYc.plotting.plotIonMap(msData, xlim=(80, 500), ylim=(80, 500), logy=True, logx=True, savePath=outputPath)
			self.assertTrue(os.path.exists(outputPath))

			outputPath = os.path.join(tmpdirname, 'centroids')
			nPYc.plotting.plotIonMap(msData, useRetention=False, savePath=outputPath)
			self.assertTrue(os.path.exists(outputPath))

			outputPath = os.path.join(tmpdirname, 'centroidsWithLimits')
			nPYc.plotting.plotIonMap(msData, useRetention=False, xlim=(80, 500), title='test', savePath=outputPath)
			self.assertTrue(os.path.exists(outputPath))

			msData.VariableType = nPYc.enumerations.VariableType.Spectral
			msData.featureMetadata.drop('Retention Time', axis=1, inplace=True)

			outputPath = os.path.join(tmpdirname, 'spectral')
			nPYc.plotting.plotIonMap(msData, xlim=(80, 500), savePath=outputPath)
			self.assertTrue(os.path.exists(outputPath))


	def test_plottic(self):
		noSamp = numpy.random.randint(50, high=100, size=None)
		noFeat = numpy.random.randint(200, high=400, size=None)
		msData = generateTestDataset(noSamp, noFeat)

		with tempfile.TemporaryDirectory() as tmpdirname:
			outputPath = os.path.join(tmpdirname, 'basic')
			nPYc.plotting.plotTIC(msData, savePath=outputPath)
			self.assertTrue(os.path.exists(outputPath))

			outputPath = os.path.join(tmpdirname, 'noViolinByDetector')
			nPYc.plotting.plotTIC(msData, addViolin=False, addBatchShading=True, colourByDetectorVoltage=True, savePath=outputPath)
			self.assertTrue(os.path.exists(outputPath))

			outputPath = os.path.join(tmpdirname, 'withBatches')
			nPYc.plotting.plotTIC(msData, addBatchShading=True, savePath=outputPath)
			self.assertTrue(os.path.exists(outputPath))

			outputPath = os.path.join(tmpdirname, 'logY')
			nPYc.plotting.plotTIC(msData, logy= True, addViolin=False, savePath=outputPath)
			self.assertTrue(os.path.exists(outputPath))


	def test_plottic_raises(self):

		noSamp = numpy.random.randint(50, high=100, size=None)
		noFeat = numpy.random.randint(200, high=400, size=None)
		msData = generateTestDataset(noSamp, noFeat)

		self.assertRaises(ValueError, nPYc.plotting.plotTIC, msData, addViolin=True, colourByDetectorVoltage=True)


	def test_jointplotRSDvCorrelation(self):

		noFeat = numpy.random.randint(200, high=400, size=None)

		with tempfile.TemporaryDirectory() as tmpdirname:

			rsd = numpy.random.lognormal(size=noFeat)
			corr = numpy.random.randn(noFeat)

			outputPath = os.path.join(tmpdirname, 'basic')
			nPYc.plotting.jointplotRSDvCorrelation(rsd, corr, savePath=outputPath)
			self.assertTrue(os.path.exists(outputPath))

			outputPath = os.path.join(tmpdirname, 'fewerBins')
			nPYc.plotting.jointplotRSDvCorrelation(rsd, corr, histBins=10, savePath=outputPath)
			self.assertTrue(os.path.exists(outputPath))

			rsd = numpy.random.randn(noFeat)
			outputPath = os.path.join(tmpdirname, 'neagativeRSDs')
			nPYc.plotting.jointplotRSDvCorrelation(rsd, corr, savePath=outputPath)
			self.assertTrue(os.path.exists(outputPath))


	def test_jointplotRSDvCorrelation_raises(self):

		noFeat = numpy.random.randint(200, high=400, size=None)

		rsd = numpy.random.randn(noFeat)
		corr = numpy.random.randn(noFeat + 1)

		self.assertRaises(ValueError, nPYc.plotting.jointplotRSDvCorrelation, rsd, corr)


	def test_plotDiscreteLoadings_raises(self):

		from nPYc.multivariate.multivariateUtilities import pcaSignificance
		from nPYc.multivariate.exploratoryAnalysisPCA import exploratoryAnalysisPCA
		noSamp = numpy.random.randint(50, high=100, size=None)
		noFeat = numpy.random.randint(10, high=120, size=None)

		data = generateTestDataset(noSamp, noFeat)

		pcaModel = exploratoryAnalysisPCA(data, maxComponents=5)

		self.assertRaises(TypeError, nPYc.plotting.plotDiscreteLoadings, 'Not a Dataset object', pcaModel)
		self.assertRaises(TypeError, nPYc.plotting.plotDiscreteLoadings, data, 'Not a PCA object')
		self.assertRaises(ValueError, nPYc.plotting.plotDiscreteLoadings, data, pcaModel, firstComponent=6)
		self.assertRaises(ValueError, nPYc.plotting.plotDiscreteLoadings, data, pcaModel, firstComponent=0)


	def test_plotDiscreteLoadings(self):

		from nPYc.multivariate import exploratoryAnalysisPCA, pcaSignificance

		noSamp = numpy.random.randint(50, high=100, size=None)
		noFeat = numpy.random.randint(10, high=120, size=None)

		data = generateTestDataset(noSamp, noFeat)

		pcaModel = exploratoryAnalysisPCA(data, maxComponents=5)

		with tempfile.TemporaryDirectory() as tmpdirname:

			outputPath = os.path.join(tmpdirname, 'basic')
			nPYc.plotting.plotDiscreteLoadings(data, pcaModel, savePath=outputPath)
			self.assertTrue(os.path.exists(outputPath))

			outputPath = os.path.join(tmpdirname, 'unsorted_diff_labels')
			nPYc.plotting.plotDiscreteLoadings(data, pcaModel, sort=False, metadataColumn='m/z', savePath=outputPath)
			self.assertTrue(os.path.exists(outputPath))

			outputPath = os.path.join(tmpdirname, 'runningOverTotalComponents')
			nPYc.plotting.plotDiscreteLoadings(data, pcaModel, firstComponent=pcaModel.ncomps-1, savePath=outputPath)
			self.assertTrue(os.path.exists(outputPath))


	@unittest.mock.patch('sys.stdout', new_callable=io.StringIO)
	def test_plotFeatureRanges(self, mock_stdout):

		with tempfile.TemporaryDirectory() as tmpdirname:

			with warnings.catch_warnings():
				# Cause all warnings to always be triggered.
				warnings.simplefilter("ignore")
				datapath = os.path.join("..", "..", "npc-standard-project", "Raw_Data", "nmr", "UnitTest3")
				testData = nPYc.TargetedDataset(datapath, fileType='Bruker Quantification', sop='BrukerBI-LISA', fileNamePattern='.*?results\.xml$')

			testData.addSampleInfo(descriptionFormat='NPC LIMS', filePath=os.path.join('..','..','npc-standard-project','Derived_Worklists','UnitTest3_NMR_serum_PCSOP.012.csv'))

			outputPath = os.path.join(tmpdirname, 'noRanges')
			nPYc.plotting.plotFeatureRanges(testData,
										   ['L1PL', 'L4PL'],
										   savePath=outputPath)
			self.assertTrue(os.path.exists(outputPath))

			testData.addFeatureInfo(descriptionFormat='Reference Ranges', filePath=os.path.join(nPYc._toolboxPath.toolboxPath(), 'StudyDesigns', 'BI-LISA_reference_ranges.json'))

			outputPath = os.path.join(tmpdirname, 'basic')
			nPYc.plotting.plotFeatureRanges(testData,
										   ['L1PL', 'L2PL', 'L3PL', 'L4PL', 'L5PL', 'L6PL'],
										   savePath=outputPath)
			self.assertTrue(os.path.exists(outputPath))

			outputPath = os.path.join(tmpdirname, 'singleFeature')
			nPYc.plotting.plotFeatureRanges(testData,
										   ['L1PL'],
										   savePath=outputPath)
			self.assertTrue(os.path.exists(outputPath))


	@unittest.mock.patch('sys.stdout', new_callable=io.StringIO)
	def test_plotFeatureRanges_logscale(self, mock_stdout):

		with tempfile.TemporaryDirectory() as tmpdirname:

			datapath = os.path.join("..", "..", "npc-standard-project", "Raw_Data", "nmr", "UnitTest1")
			testData = nPYc.TargetedDataset(datapath, fileType='Bruker Quantification', sop='BrukerQuant-UR', fileNamePattern='.*?urine_quant_report_b\.xml$', unit='mmol/L')
			testData.addSampleInfo(descriptionFormat='NPC LIMS', filePath=os.path.join('..','..','npc-standard-project','Derived_Worklists','UnitTest1_NMR_urine_PCSOP.011.csv'))

			outputPath = os.path.join(tmpdirname, 'basic')
			nPYc.plotting.plotFeatureRanges(testData,
										   ['Creatinine', 'Dimethylamine', 'Trimethylamine', 'Valine', 'Myo-Inositol'],
										   logx=True,
										   savePath=outputPath)
			self.assertTrue(os.path.exists(outputPath))

			outputPath = os.path.join(tmpdirname, 'singleFeature')
			nPYc.plotting.plotFeatureRanges(testData,
										   ['Creatinine'],
										   logx=True,
										   savePath=outputPath)
			self.assertTrue(os.path.exists(outputPath))


	def test_plotFeatureRanges_raises(self):

		self.assertRaises(TypeError, nPYc.plotting.plotFeatureRanges, 'not a dataset', dict())


	def test_plotLOQRunOrder(self):
		inputData = copy.deepcopy(self.targetedDataset)
		# Default values
		# plotLOQRunOrder(targetedData, addCalibration=True, compareBatch=True, title='', savePath=None, figureFormat='png', dpi=72, figureSize=(11, 7))

		with tempfile.TemporaryDirectory() as tmpdirname:
			outputPath = os.path.join(tmpdirname, 'basic')
			nPYc.plotting.plotLOQRunOrder(inputData, savePath=outputPath)
			self.assertTrue(os.path.exists(outputPath))

			outputPath = os.path.join(tmpdirname, 'noAddCalibration')
			nPYc.plotting.plotLOQRunOrder(inputData, addCalibration=False, savePath=outputPath)
			self.assertTrue(os.path.exists(outputPath))

			outputPath = os.path.join(tmpdirname, 'noCompareBatch')
			nPYc.plotting.plotLOQRunOrder(inputData, compareBatch=False, savePath=outputPath)
			self.assertTrue(os.path.exists(outputPath))


	def test_plotLOQRunOrder_raises(self):
		brokenData = copy.deepcopy(self.targetedDataset)
		delattr(brokenData, 'calibration')
		self.assertRaises(ValueError, nPYc.plotting.plotLOQRunOrder, brokenData)


	def test_plotFeatureLOQ(self):
		inputData = copy.deepcopy(self.targetedDataset)
		# Default values
		#plotFeatureLOQ(tData, splitByBatch=True, plotBatchLOQ=False, zoomLOQ=False, logY=False, tightYLim=True, nbPlotPerRow=3, savePath=None, figureFormat='png', dpi=72, figureSize=(11, 7)):

		with tempfile.TemporaryDirectory() as tmpdirname:
			outputPath = os.path.join(tmpdirname, 'basic')
			paths = nPYc.plotting.plotFeatureLOQ(inputData, savePath=outputPath)
			for path in paths:
				self.assertTrue(os.path.exists(path))

			outputPath = os.path.join(tmpdirname, 'noSplitByBatch')
			paths = nPYc.plotting.plotFeatureLOQ(inputData, splitByBatch=False, savePath=outputPath)
			for path in paths:
				self.assertTrue(os.path.exists(path))

			outputPath = os.path.join(tmpdirname, 'withPlotBatchLOQ')
			paths = nPYc.plotting.plotFeatureLOQ(inputData, plotBatchLOQ=True, savePath=outputPath)
			for path in paths:
				self.assertTrue(os.path.exists(path))

			outputPath = os.path.join(tmpdirname, 'withZoomLOQ')
			paths = nPYc.plotting.plotFeatureLOQ(inputData, zoomLOQ=True, savePath=outputPath)
			for path in paths:
				self.assertTrue(os.path.exists(path))

			outputPath = os.path.join(tmpdirname, 'withLogY')
			nPYc.plotting.plotFeatureLOQ(inputData, logY=True, savePath=outputPath)
			for path in paths:
				self.assertTrue(os.path.exists(path))

			outputPath = os.path.join(tmpdirname, 'noTightYLim')
			paths = nPYc.plotting.plotFeatureLOQ(inputData, tightYLim=False, savePath=outputPath)
			for path in paths:
				self.assertTrue(os.path.exists(path))

			outputPath = os.path.join(tmpdirname, 'otherNbPlotPerRow')
			nPYc.plotting.plotFeatureLOQ(inputData, nbPlotPerRow=5, savePath=outputPath)
			for path in paths:
				self.assertTrue(os.path.exists(path))


	def test_plotFeatureLOQ_raises(self):
		brokenData = copy.deepcopy(self.targetedDataset)
		delattr(brokenData, 'calibration')
		self.assertRaises(ValueError, nPYc.plotting.plotFeatureLOQ, brokenData)


	def test_plotLOQFeatureViolin_raises(self):
		from nPYc.plotting._plotLOQFeatureViolin import _featureLOQViolinPlotHelper
		self.assertRaises(ValueError, _featureLOQViolinPlotHelper,'','','', subplot='wrong subplot')


	def test_plotVariableScatter(self):
		inputData = pandas.DataFrame()
		inputData[SampleType.StudySample]		= [0., 1., 2., 3., 4., 5.]
		inputData[SampleType.StudyPool]			= [0., 1., 2., 3., 4., 5.]
		inputData[SampleType.ExternalReference] = [0., 1., 2., 3., 4., 5.]
		inputData['yName']						= ['0', '1', '2', '3', '4', '5']
		# Default values
		#plotVariableScatter(inputTable, logX=False, xLim=None, xLabel='', yLabel='', sampletypeColor=False, hLines=None, hLineStyle='-', vLines=None, vLineStyle=':', savePath=None, figureFormat='png', dpi=72, figureSize=(11, 7))

		with tempfile.TemporaryDirectory() as tmpdirname:
			outputPath = os.path.join(tmpdirname, 'basic')
			nPYc.plotting.plotVariableScatter(inputData, savePath=outputPath)
			self.assertTrue(os.path.exists(outputPath))

			outputPath = os.path.join(tmpdirname, 'logX')
			nPYc.plotting.plotVariableScatter(inputData, logX=True, savePath=outputPath)
			self.assertTrue(os.path.exists(outputPath))

			# Add another column that won't be read with sampletypeColor=True
			inputData['another column'] = ['a', 'b', 'c', 'd', 'e', 'f']
			outputPath = os.path.join(tmpdirname, 'withSampletypeColor')
			nPYc.plotting.plotVariableScatter(inputData, sampletypeColor=True, savePath=outputPath)
			self.assertTrue(os.path.exists(outputPath))


	def test_plotVariableScatter_raises(self):

		self.assertRaises(TypeError, nPYc.plotting.plotVariableScatter, inputTable='', xLim=[])
		self.assertRaises(TypeError, nPYc.plotting.plotVariableScatter, inputTable='', xLabel=5.)
		self.assertRaises(TypeError, nPYc.plotting.plotVariableScatter, inputTable='', yLabel=5.)
		self.assertRaises(TypeError, nPYc.plotting.plotVariableScatter, inputTable='', hLines=5.)
		self.assertRaises(TypeError, nPYc.plotting.plotVariableScatter, inputTable='', vLines=5.)
		self.assertRaises(ValueError, nPYc.plotting.plotVariableScatter, inputTable='', hLineStyle='a')
		self.assertRaises(ValueError, nPYc.plotting.plotVariableScatter, inputTable='', vLineStyle='a')
		self.assertRaises(TypeError, nPYc.plotting.plotVariableScatter, inputTable='', hBox='a')
		self.assertRaises(TypeError, nPYc.plotting.plotVariableScatter, inputTable='', vBox='a')
		self.assertRaises(TypeError, nPYc.plotting.plotVariableScatter, inputTable='', hBox=['a'])
		self.assertRaises(TypeError, nPYc.plotting.plotVariableScatter, inputTable='', vBox=['a'])


	def test_plotAccuracyPrecision(self):
		inputData = copy.deepcopy(self.targetedDataset)
		inputData.sampleMetadata['AssayRole']  = [AssayRole.PrecisionReference, AssayRole.PrecisionReference, AssayRole.PrecisionReference]
		inputData.sampleMetadata['SampleType'] = [SampleType.StudyPool, SampleType.StudyPool, SampleType.StudyPool]
		# Default values
		# plotAccuracyPrecision(tData, accuracy=True, savePath=None, figureFormat='png', dpi=72, figureSize=(11, 7))

		with tempfile.TemporaryDirectory() as tmpdirname:
			outputPath = os.path.join(tmpdirname, 'Accuracy')
			nPYc.plotting.plotAccuracyPrecision(inputData, accuracy=True, savePath=outputPath)
			self.assertTrue(os.path.exists(outputPath))

			outputPath = os.path.join(tmpdirname, 'Precision')
			nPYc.plotting.plotAccuracyPrecision(inputData, accuracy=False, savePath=outputPath)
			self.assertTrue(os.path.exists(outputPath))


	def test_plotAccuracyPrecision_raises(self):
		## Error
		# Not satisfy targeted
		brokenData = copy.deepcopy(self.targetedDataset)
		delattr(brokenData, 'calibration')
		self.assertRaises(ValueError, nPYc.plotting.plotAccuracyPrecision, brokenData)

		# wrong percentBox
		inputData = copy.deepcopy(self.targetedDataset)
		self.assertRaises(ValueError, nPYc.plotting.plotAccuracyPrecision, inputData, percentRange='not float')


		## Warning
		inputWarn = copy.deepcopy(self.targetedDataset)
		inputWarn.expectedConcentration = pandas.DataFrame(numpy.array([[4., 6.], [40., 60.], [400., 600.]]), columns=inputWarn.featureMetadata['Feature Name'].values.tolist())

		# No samples for Accuracy
		with warnings.catch_warnings(record=True) as w:
			# Cause all warnings to always be triggered.
			warnings.simplefilter("always")
			# warning
			nPYc.plotting.plotAccuracyPrecision(inputWarn, accuracy=True)
			# check each warning
			self.assertEqual(len(w), 1)
			assert issubclass(w[0].category, UserWarning)
			assert "Warning: no Accuracy values to plot." in str(w[0].message)

		# No samples for Precision
		with warnings.catch_warnings(record=True) as w:
			# Cause all warnings to always be triggered.
			warnings.simplefilter("always")
			# warning
			nPYc.plotting.plotAccuracyPrecision(inputWarn, accuracy=False)
			# check each warning
			self.assertEqual(len(w), 1)
			assert issubclass(w[0].category, UserWarning)
			assert "Warning: no Precision values to plot." in str(w[0].message)


	def test_plotCorrelationToLRbyFeature(self):

		noSamp = numpy.random.randint(100, high=300, size=None)
		noFeat = numpy.random.randint(10, high=100, size=None)
		noPlots = numpy.random.randint(1, high=10, size=None)

		dataset = generateTestDataset(noSamp, noFeat,
									  dtype='MSDataset',
									  variableType=nPYc.enumerations.VariableType.Discrete,
									  sop='GenericMS')

		dataset.sampleMetadata.loc[0:10, 'AssayRole'] = nPYc.enumerations.AssayRole.LinearityReference
		dataset.sampleMetadata.loc[0:10, 'SampleType'] = nPYc.enumerations.SampleType.StudyPool
		dataset.sampleMetadata.loc[0:10, 'Well'] = 1
		dataset.corrExclusions = dataset.sampleMask

		with tempfile.TemporaryDirectory() as tmpdirname:

			nPYc.plotting.plotCorrelationToLRbyFeature(dataset, featureMask=None, title='test', maxNo=noPlots, savePath=tmpdirname)

			onlyfiles = [f for f in os.listdir(tmpdirname) if os.path.isfile(os.path.join(tmpdirname, f))]

			self.assertEqual(len(onlyfiles), noPlots)


	def test_plotSolventResonace(self):
		dataset = generateTestDataset(10, 1000, dtype='NMRDataset',
												variableType=VariableType.Continuum,
												sop='GenericNMRurine')

		high_area = numpy.array([True, False, False, False, False, False, False, False, False, True], dtype=bool)
		high_neg = numpy.array([False, True, False, False, False, False, False, False, False, True], dtype=bool)

		low_area = numpy.array([False, False, True, False, False, False, False, False, False, True], dtype=bool)
		low_neg = numpy.array([False, False, False, True, False, False, False, False, False, True], dtype=bool)

		dataset.sampleMetadata['SolventPeakFail'] = low_area | low_neg | high_area | high_neg
		with tempfile.TemporaryDirectory() as tmpdirname:
			outputPath = os.path.join(tmpdirname, 'plot')

			nPYc.plotting.plotSolventResonance(dataset, savePath=outputPath)

			self.assertTrue(os.path.exists(outputPath))


	def test_plotCalibrationI(self):
		dataset = generateTestDataset(10, 1000, dtype='NMRDataset',
												variableType=VariableType.Continuum,
												sop='GenericNMRurine')

		with tempfile.TemporaryDirectory() as tmpdirname:
			outputPath = os.path.join(tmpdirname, 'plot')

			nPYc.plotting.plotCalibration(dataset, savePath=outputPath)

			self.assertTrue(os.path.exists(outputPath))


	def test_plotBaseline(self):
		dataset = generateTestDataset(10, 1000, dtype='NMRDataset',
												variableType=VariableType.Continuum,
												sop='GenericNMRurine')

		high_area = numpy.array([True, False, False, False, False, False, False, False, False, True], dtype=bool)
		high_neg = numpy.array([False, True, False, False, False, False, False, False, False, True], dtype=bool)

		low_area = numpy.array([False, False, True, False, False, False, False, False, False, True], dtype=bool)
		low_neg = numpy.array([False, False, False, True, False, False, False, False, False, True], dtype=bool)

		dataset.sampleMetadata['BaselineFail'] = high_area | high_neg | low_area | low_neg

		with tempfile.TemporaryDirectory() as tmpdirname:
			outputPath = os.path.join(tmpdirname, 'plot')

			nPYc.plotting.plotBaseline(dataset, savePath=outputPath)

			self.assertTrue(os.path.exists(outputPath))


	def test_plotLineWidth(self):
		dataset = generateTestDataset(10, 1000, dtype='NMRDataset',
												variableType=VariableType.Continuum,
												sop='GenericNMRurine')

		dataset.sampleMetadata['Line Width (Hz)'] =	 [numpy.nan, 1, 2, 0.9, 0, 1, 1, 1, 1, 1]

		with tempfile.TemporaryDirectory() as tmpdirname:
			outputPath = os.path.join(tmpdirname, 'plot')

			nPYc.plotting.plotLineWidth(dataset, savePath=outputPath)

			self.assertTrue(os.path.exists(outputPath))


	def test_plotLoadings(self):

		with self.subTest(msg='Testing MSDataset'):
			noSamp = numpy.random.randint(100, high=500, size=None)
			noFeat = numpy.random.randint(200, high=400, size=None)
			dataset = generateTestDataset(noSamp, noFeat, dtype='MSDataset', variableType=VariableType.Discrete, sop='Generic')

			pcaModel = nPYc.multivariate.exploratoryAnalysisPCA(dataset)

			with tempfile.TemporaryDirectory() as tmpdirname:
				outputPath = os.path.join(tmpdirname, 'plot')

				figureLocs = nPYc.plotting.plotLoadings(pcaModel, dataset, figures=dict(), savePath=outputPath)

				for fig in figureLocs.keys():
					self.assertTrue(os.path.exists(figureLocs[fig]))

		with self.subTest(msg='Testing NMRDataset'):
			noSamp = numpy.random.randint(100, high=500, size=None)
			noFeat = numpy.random.randint(200, high=400, size=None)
			dataset = generateTestDataset(noSamp, noFeat, dtype='NMRDataset', variableType=VariableType.Continuum, sop='Generic')

			pcaModel = nPYc.multivariate.exploratoryAnalysisPCA(dataset)

			with tempfile.TemporaryDirectory() as tmpdirname:
				outputPath = os.path.join(tmpdirname, 'plot')

				figureLocs = nPYc.plotting.plotLoadings(pcaModel, dataset, figures=dict(), savePath=outputPath)
				for fig in figureLocs.keys():
					self.assertTrue(os.path.exists(figureLocs[fig]))


	def test_plotOutliers(self):

		with self.subTest(msg='Testing MSDataset and scores'):
			noSamp = numpy.random.randint(100, high=500, size=None)
			noFeat = numpy.random.randint(200, high=400, size=None)
			dataset = generateTestDataset(noSamp, noFeat, dtype='MSDataset', variableType=VariableType.Discrete, sop='Generic')
			dataset.sampleMetadata.loc[:, 'Plot Sample Type'] = 'Sample'
			dataset.sampleMetadata.loc[dataset.sampleMetadata['SampleType'] == SampleType.StudySample, 'Plot Sample Type'] = 'Study Sample'
			dataset.sampleMetadata.loc[dataset.sampleMetadata['SampleType'] == SampleType.StudyPool, 'Plot Sample Type'] = 'Study Pool'
			dataset.sampleMetadata.loc[dataset.sampleMetadata['SampleType'] == SampleType.ExternalReference, 'Plot Sample Type'] = 'External Reference'

			pcaModel = nPYc.multivariate.exploratoryAnalysisPCA(dataset)
			sumT = numpy.sum(numpy.absolute(pcaModel.scores), axis=1)

			with tempfile.TemporaryDirectory() as tmpdirname:
				outputPath = os.path.join(tmpdirname, 'plot')

				nPYc.plotting.plotOutliers(sumT, dataset.sampleMetadata['Run Order'], sampleType=dataset.sampleMetadata['Plot Sample Type'], addViolin=True, PcritPercentile=95, savePath=outputPath)

				self.assertTrue(os.path.exists(outputPath))

		with self.subTest(msg='Testing NMRDataset and dmodx'):
			noSamp = numpy.random.randint(100, high=500, size=None)
			noFeat = numpy.random.randint(200, high=400, size=None)
			dataset = generateTestDataset(noSamp, noFeat, dtype='NMRDataset', variableType=VariableType.Continuum, sop='Generic')

			pcaModel = nPYc.multivariate.exploratoryAnalysisPCA(dataset)
			sample_dmodx_values = pcaModel.dmodx(dataset.intensityData)
			Fcrit = pcaModel._dmodx_fcrit(dataset.intensityData, alpha = 0.05)
			
			with tempfile.TemporaryDirectory() as tmpdirname:
				outputPath = os.path.join(tmpdirname, 'plot')

				nPYc.plotting.plotOutliers(sample_dmodx_values, dataset.sampleMetadata['Run Order'], Fcrit=Fcrit, FcritAlpha=0.05, savePath=outputPath)

				self.assertTrue(os.path.exists(outputPath))


	def test_plotScores(self):

		noSamp = numpy.random.randint(50, high=100, size=None)
		noFeat = numpy.random.randint(200, high=400, size=None)
		dataset = generateTestDataset(noSamp, noFeat, dtype='MSDataset')

		pcaModel = nPYc.multivariate.exploratoryAnalysisPCA(dataset)

		with tempfile.TemporaryDirectory() as tmpdirname:

			with self.subTest(msg='Basic'):
				outputPath = os.path.join(tmpdirname, 'basic')

				nPYc.plotting.plotScores(pcaModel, figureFormat='png', savePath=outputPath)

				self.assertTrue(os.path.exists(outputPath + '.png'))

			with self.subTest(msg='Continuous varaible'):
				outputPath = os.path.join(tmpdirname, 'continuous')

				nPYc.plotting.plotScores(pcaModel,classes=dataset.sampleMetadata['Detector'], classType='continuous', figureFormat='png', savePath=outputPath)

				self.assertTrue(os.path.exists(outputPath + '.png'))

			with self.subTest(msg='Discrete variables'):
				outputPath = os.path.join(tmpdirname, 'categorical')

				nPYc.plotting.plotScores(pcaModel,classes=dataset.sampleMetadata['AssayRole'], classType='categorical', figureFormat='png', savePath=outputPath)

				self.assertTrue(os.path.exists(outputPath + '.png'))

			with self.subTest(msg='Association plot'):
				outputPath = os.path.join(tmpdirname, 'categoricalAssociation')

				nPYc.plotting.plotScores(pcaModel,classes=dataset.sampleMetadata['AssayRole'], classType='categorical', plotAssociation=[0,1,2], figureFormat='png', savePath=outputPath)

				self.assertTrue(os.path.exists(outputPath + '.png'))

			with self.subTest(msg='Continuous Association'):
				outputPath = os.path.join(tmpdirname, 'continuousAssociation')

				nPYc.plotting.plotScores(pcaModel,classes=dataset.sampleMetadata['Detector'], classType='continuous', plotAssociation=[0,1,2], figureFormat='png', savePath=outputPath)

				self.assertTrue(os.path.exists(outputPath + '.png'))


	def test_plotScores_raises(self):

		self.assertRaises(TypeError, nPYc.plotting.plotScores, 'not a pca model')

		noSamp = 10
		noFeat = 10
		dataset = generateTestDataset(noSamp, noFeat, dtype='MSDataset')
		pcaModel = nPYc.multivariate.exploratoryAnalysisPCA(dataset)

		self.assertRaises(ValueError, nPYc.plotting.plotScores, pcaModel, classes=dataset.sampleMetadata['Detector'], classType=None)


	def test_plotPW(self):
		noSamp = numpy.random.randint(100, high=500, size=None)
		noFeat = numpy.random.randint(2000, high=10000, size=None)
		dataset = generateTestDataset(noSamp, noFeat, dtype='NMRDataset', variableType=VariableType.Continuum, sop='GenericNMRurine')

		with tempfile.TemporaryDirectory() as tmpdirname:
			outputPath = os.path.join(tmpdirname, 'plot')

			nPYc.plotting.plotPW(dataset, savePath=outputPath)

			self.assertTrue(os.path.exists(outputPath))


class test_plotting_interactive(unittest.TestCase):

	def setUp(self):
		noSamp = numpy.random.randint(100, high=500, size=None)
		noFeat = numpy.random.randint(200, high=400, size=None)
		self.dataset = generateTestDataset(noSamp, noFeat, dtype='MSDataset', variableType=VariableType.Discrete, sop='Generic')

	def test_plotTICinteractive(self):

		##
		# Not checking output for correctness, just that we don't crash
		##
		with self.subTest(msg='Default'):
			data = nPYc.plotting.plotTICinteractive(self.dataset)
			self.assertIsNotNone(data)

		with self.subTest(msg='Linearity Reference Plot'):
			data = nPYc.plotting.plotTICinteractive(self.dataset, plottype='Linearity Reference')
			self.assertIsNotNone(data)


	def test_plotTICinteractive_raises(self):


		with self.subTest(msg='Not a string'):
			self.assertRaises(ValueError, nPYc.plotting.plotTICinteractive, self.dataset, plottype=1)

		with self.subTest(msg='Not a valid string'):
			self.assertRaises(ValueError, nPYc.plotting.plotTICinteractive, self.dataset, plottype='1')


	def test_plotSolventResonaceInteractive(self):
		dataset = generateTestDataset(10, 1000, dtype='NMRDataset',
												variableType=VariableType.Continuum,
												sop='GenericNMRurine')

		high_area = numpy.array([True, False, False, False, False, False, False, False, False, True], dtype=bool)
		high_neg = numpy.array([False, True, False, False, False, False, False, False, False, True], dtype=bool)

		low_area = numpy.array([False, False, True, False, False, False, False, False, False, True], dtype=bool)
		low_neg = numpy.array([False, False, False, True, False, False, False, False, False, True], dtype=bool)

		dataset.sampleMetadata['SolventPeakFail'] = low_area | low_neg | high_area | high_neg

		figure = nPYc.plotting.plotSolventResonanceInteractive(dataset)
		self.assertIsInstance(figure, plotly.graph_objs.Figure)


	def test_plotCalibrationInteractive(self):
		dataset = generateTestDataset(10, 1000, dtype='NMRDataset',
												variableType=VariableType.Continuum,
												sop='GenericNMRurine')

		figure = nPYc.plotting.plotCalibrationInteractive(dataset)
		self.assertIsInstance(figure, plotly.graph_objs.Figure)


	def test_plotBaselineInteractive(self):
		dataset = generateTestDataset(10, 1000, dtype='NMRDataset',
												variableType=VariableType.Continuum,
												sop='GenericNMRurine')

		high_area = numpy.array([True, False, False, False, False, False, False, False, False, True], dtype=bool)
		high_neg = numpy.array([False, True, False, False, False, False, False, False, False, True], dtype=bool)

		low_area = numpy.array([False, False, True, False, False, False, False, False, False, True], dtype=bool)
		low_neg = numpy.array([False, False, False, True, False, False, False, False, False, True], dtype=bool)

		dataset.sampleMetadata['BaselineFail'] = high_area | high_neg | low_area | low_neg

		figure = nPYc.plotting.plotBaselineInteractive(dataset)
		self.assertIsInstance(figure, plotly.graph_objs.Figure)


	def test_plotLineWidthInteractive(self):
		dataset = generateTestDataset(10, 1000, dtype='NMRDataset',
												variableType=VariableType.Continuum,
												sop='GenericNMRurine')

		dataset.sampleMetadata['Line Width (Hz)'] =	 [numpy.nan, 1, 2, 0.9, 0, 1, 1, 1, 1, 1]

		figure = nPYc.plotting.plotLineWidthInteractive(dataset)
		self.assertIsInstance(figure, plotly.graph_objs.Figure)


	def test_plotLoadingsInteractive(self):

		with self.subTest(msg='Testing MSDataset, 1 component'):

			pcaModel = nPYc.multivariate.exploratoryAnalysisPCA(self.dataset)

			figure = nPYc.plotting.plotLoadingsInteractive(self.dataset, pcaModel)
			self.assertIsInstance(figure, plotly.graph_objs.Figure)

		with self.subTest(msg='Testing MSDataset, 2 components'):

			pcaModel = nPYc.multivariate.exploratoryAnalysisPCA(self.dataset)

			figure = nPYc.plotting.plotLoadingsInteractive(self.dataset, pcaModel, component=[1, 2])
			self.assertIsInstance(figure, plotly.graph_objs.Figure)

		with self.subTest(msg='Testing NMRDataset, 1 component'):
			noSamp = numpy.random.randint(100, high=500, size=None)
			noFeat = numpy.random.randint(200, high=400, size=None)
			dataset = generateTestDataset(noSamp, noFeat, dtype='NMRDataset', variableType=VariableType.Continuum, sop='Generic')

			pcaModel = nPYc.multivariate.exploratoryAnalysisPCA(dataset)

			figure = nPYc.plotting.plotLoadingsInteractive(dataset, pcaModel)
			self.assertIsInstance(figure, plotly.graph_objs.Figure)

		with self.subTest(msg='Testing NMRDataset, 2 components'):
			noSamp = numpy.random.randint(100, high=500, size=None)
			noFeat = numpy.random.randint(200, high=400, size=None)
			dataset = generateTestDataset(noSamp, noFeat, dtype='NMRDataset', variableType=VariableType.Continuum, sop='Generic')

			pcaModel = nPYc.multivariate.exploratoryAnalysisPCA(dataset)

			figure = nPYc.plotting.plotLoadingsInteractive(dataset, pcaModel, component=[1, 2])
			self.assertIsInstance(figure, plotly.graph_objs.Figure)

		with self.subTest(msg='Testing other dataset, 1 component'):
			noSamp = numpy.random.randint(100, high=500, size=None)
			noFeat = numpy.random.randint(200, high=400, size=None)
			dataset = generateTestDataset(noSamp, noFeat, dtype='Dataset', variableType=VariableType.Discrete, sop='Generic')

			pcaModel = nPYc.multivariate.exploratoryAnalysisPCA(dataset)

			figure = nPYc.plotting.plotLoadingsInteractive(dataset, pcaModel)
			self.assertIsInstance(figure, plotly.graph_objs.Figure)

		with self.subTest(msg='Testing other dataset, 2 components'):
			noSamp = numpy.random.randint(100, high=500, size=None)
			noFeat = numpy.random.randint(200, high=400, size=None)
			dataset = generateTestDataset(noSamp, noFeat, dtype='Dataset', variableType=VariableType.Discrete, sop='Generic')

			pcaModel = nPYc.multivariate.exploratoryAnalysisPCA(dataset)

			figure = nPYc.plotting.plotLoadingsInteractive(dataset, pcaModel, component=[1, 2])
			self.assertIsInstance(figure, plotly.graph_objs.Figure)


		with self.subTest(msg='Testing all components'):
			noSamp = numpy.random.randint(100, high=500, size=None)
			noFeat = numpy.random.randint(200, high=400, size=None)
			dataset = generateTestDataset(noSamp, noFeat, dtype='NMRDataset', variableType=VariableType.Continuum, sop='Generic')

			pcaModel = nPYc.multivariate.exploratoryAnalysisPCA(dataset)

			for i in range(pcaModel.ncomps):

				figure = nPYc.plotting.plotLoadingsInteractive(dataset, pcaModel, component=i)
				self.assertIsInstance(figure, plotly.graph_objs.Figure)


	def test_plotLoadingsInteractive_raises(self):

		self.assertRaises(TypeError, nPYc.plotting.plotLoadingsInteractive, 'not a PCA model')

		pcaModel = nPYc.multivariate.exploratoryAnalysisPCA(self.dataset)

		self.assertRaises(TypeError, nPYc.plotting.plotLoadingsInteractive, 'not a dataset', pcaModel, component=1)

		self.assertRaises(TypeError, nPYc.plotting.plotLoadingsInteractive, self.dataset, pcaModel, component='not an int')

		self.assertRaises(ValueError, nPYc.plotting.plotLoadingsInteractive, self.dataset, pcaModel, component=pcaModel.ncomps + 1)

		self.assertRaises(TypeError, nPYc.plotting.plotLoadingsInteractive, self.dataset, pcaModel, component=['not an int', pcaModel.ncomps])

		self.assertRaises(TypeError, nPYc.plotting.plotLoadingsInteractive, self.dataset, pcaModel, component=[pcaModel.ncomps, pcaModel.ncomps, pcaModel.ncomps])

		self.assertRaises(ValueError, nPYc.plotting.plotLoadingsInteractive, self.dataset, pcaModel, component=[pcaModel.ncomps, pcaModel.ncomps + 1])


	def test_plotScoresInteractive(self):

		pcaModel = nPYc.multivariate.exploratoryAnalysisPCA(self.dataset)

		with self.subTest(msg='Continuous varaible'):
			figure = nPYc.plotting.plotScoresInteractive(self.dataset, pcaModel, 'Detector')
			self.assertIsInstance(figure, plotly.graph_objs.Figure)

		with self.subTest(msg='Discrete variables'):
			figure = nPYc.plotting.plotScoresInteractive(self.dataset, pcaModel, 'Classes')
			self.assertIsInstance(figure, plotly.graph_objs.Figure)


	def test_plotScoresInteractive_raises(self):

		pcaModel = nPYc.multivariate.exploratoryAnalysisPCA(self.dataset)

		self.assertRaises(TypeError, nPYc.plotting.plotScoresInteractive, 'not a Dataset', pcaModel, 'Classes')

		self.assertRaises(TypeError, nPYc.plotting.plotScoresInteractive, self.dataset, 'not a pca model', 'Classes')

		self.assertRaises(TypeError, nPYc.plotting.plotScoresInteractive, self.dataset, pcaModel, 'Classes', components=['a', 1.2])

		self.assertRaises(ValueError, nPYc.plotting.plotScoresInteractive, self.dataset, pcaModel, 'Classes', components=[1, pcaModel.ncomps + 1])


	def test_plotRSDsInteractive(self):

		with self.subTest(msg='Defalt'):
			figure = nPYc.plotting.plotRSDsInteractive(self.dataset)
			self.assertIsInstance(figure, plotly.graph_objs.Figure)

		with self.subTest(msg='Ratio and no xlog'):
			figure = nPYc.plotting.plotRSDsInteractive(self.dataset, ratio=True, logx=False)
			self.assertIsInstance(figure, plotly.graph_objs.Figure)

		with self.subTest(msg='No external reference'):
			self.dataset.sampleMetadata.loc[self.dataset.sampleMetadata['SampleType'] == SampleType.ExternalReference, 'SampleType'] = SampleType.StudySample
			figure = nPYc.plotting.plotRSDsInteractive(self.dataset)
			self.assertIsInstance(figure, plotly.graph_objs.Figure)


	def test_plotIonMapInteractive(self):

		with self.subTest(msg='Default'):
			figure = nPYc.plotting.plotIonMapInteractive(self.dataset)
			self.assertIsInstance(figure, plotly.graph_objs.Figure)

		with self.subTest(msg='With title and log scales'):
			figure = nPYc.plotting.plotIonMapInteractive(self.dataset, title='Title!', logx=True, logy=True)
			self.assertIsInstance(figure, plotly.graph_objs.Figure)

		with self.subTest(msg='With axis limits'):
			figure = nPYc.plotting.plotIonMapInteractive(self.dataset, xlim=[100, 500], ylim=[100, 500])
			self.assertIsInstance(figure, plotly.graph_objs.Figure)


	def test_plotIonMapInteractive_raises(self):

		self.assertRaises(ValueError, nPYc.plotting.plotIonMapInteractive, self.dataset, featureName='not in the featuremetadata')


	def test_plotSpectraInteractive(self):

		noSamp = numpy.random.randint(50, high=100, size=None)
		noFeat = numpy.random.randint(200, high=400, size=None)
		dataset = generateTestDataset(noSamp, noFeat, dtype='NMRDataset', variableType=nPYc.enumerations.VariableType.Spectral)

		with self.subTest(msg='Default'):
			figure = nPYc.plotting.plotSpectraInteractive(dataset)
			self.assertIsInstance(figure, plotly.graph_objs.Figure)

		with self.subTest(msg='With xlims'):
			figure = nPYc.plotting.plotSpectraInteractive(dataset, samples=None, xlim=[3,7])
			self.assertIsInstance(figure, plotly.graph_objs.Figure)

		with self.subTest(msg='Specify samples as int'):
			figure = nPYc.plotting.plotSpectraInteractive(dataset, samples=1)
			self.assertIsInstance(figure, plotly.graph_objs.Figure)

		with self.subTest(msg='Specify samples as list'):
			figure = nPYc.plotting.plotSpectraInteractive(dataset, samples=[1, 0, 3, 5])
			self.assertIsInstance(figure, plotly.graph_objs.Figure)

		with self.subTest(msg='Specify samples as mask'):
			mask = dataset.sampleMask
			mask[0] = False
			figure = nPYc.plotting.plotSpectraInteractive(dataset, samples=mask)
			self.assertIsInstance(figure, plotly.graph_objs.Figure)

		with self.subTest(msg='Specify label text'):
			figure = nPYc.plotting.plotSpectraInteractive(dataset, sampleLabels='SampleType')
			self.assertIsInstance(figure, plotly.graph_objs.Figure)

		with self.subTest(msg='Specify feature names'):
			dataset.featureMetadata['A new column'] = dataset.featureMetadata['ppm']
			figure = nPYc.plotting.plotSpectraInteractive(dataset, featureNames='A new column')
			self.assertIsInstance(figure, plotly.graph_objs.Figure)


	def test_plotSpectraInteractive_raises(self):

		noSamp = numpy.random.randint(50, high=100, size=None)
		noFeat = numpy.random.randint(200, high=400, size=None)
		dataset = generateTestDataset(noSamp, noFeat, dtype='NMRDataset', variableType=nPYc.enumerations.VariableType.Spectral)

		self.assertRaises(TypeError, nPYc.plotting.plotSpectraInteractive, self.dataset)

		self.assertRaises(KeyError, nPYc.plotting.plotSpectraInteractive, dataset, featureNames='not present')

		self.assertRaises(KeyError, nPYc.plotting.plotSpectraInteractive, dataset, sampleLabels='not present')


	def test_plotspectralvarianceInteractive(self):

		noSamp = numpy.random.randint(50, high=100, size=None)
		noFeat = numpy.random.randint(200, high=400, size=None)

		dataset = generateTestDataset(noSamp, noFeat, dtype='NMRDataset', variableType=nPYc.enumerations.VariableType.Spectral)

		##
		# Basic output
		##
		figure = nPYc.plotting.plotSpectralVarianceInteractive(dataset)

		self.assertIsInstance(figure, plotly.graph_objs.Figure)
		
		##
		# Classes
		##
		figure = nPYc.plotting.plotSpectralVarianceInteractive(dataset, classes='Classes', xlim=(1,9), average='mean')

		self.assertIsInstance(figure, plotly.graph_objs.Figure)

		##
		# Loq Y + title + NMR
		##
		figure = nPYc.plotting.plotSpectralVarianceInteractive(dataset, title='Figure Name')

		self.assertIsInstance(figure, plotly.graph_objs.Figure)


	def test_plotspectralvariance_raises(self):

		with self.subTest(msg='Wrong Type'):
			self.assertRaises(TypeError, nPYc.plotting.plotSpectralVarianceInteractive, 1)

		with self.subTest(msg='Discrete Data'):
			dataset = nPYc.Dataset()
			dataset.VariableType = nPYc.enumerations.VariableType.Discrete
			self.assertRaises(ValueError, nPYc.plotting.plotSpectralVarianceInteractive, dataset)

		with self.subTest(msg='Wrong length quantiles'):
			dataset = nPYc.Dataset()
			dataset.VariableType = nPYc.enumerations.VariableType.Continuum
			self.assertRaises(ValueError, nPYc.plotting.plotSpectralVarianceInteractive, dataset, quantiles=(1,2,3))

		with self.subTest(msg='Classes not found'):
			dataset = nPYc.Dataset()
			dataset.VariableType = nPYc.enumerations.VariableType.Continuum
			dataset.sampleMetadata = pandas.DataFrame(0, index=numpy.arange(5), columns=['Classes'])
			self.assertRaises(ValueError, nPYc.plotting.plotSpectralVarianceInteractive, dataset, classes='Not present')


	def test_correlationSpectroscopyInteractive(self):

		with self.subTest(msg='SHY'):
			noSamp = numpy.random.randint(50, high=100, size=None)
			noFeat = numpy.random.randint(200, high=400, size=None)

			dataset = generateTestDataset(noSamp, noFeat, dtype='NMRDataset', variableType=nPYc.enumerations.VariableType.Spectral)

			figure = nPYc.plotting.correlationSpectroscopyInteractive(dataset, dataset.intensityData[:, 10], mode='SHY')

			self.assertIsInstance(figure, plotly.graph_objs.Figure)


class test_plotting_helpers(unittest.TestCase):

	def test_plotRSDsHelper_raises(self):

		with self.subTest(msg='Spectral variables'):
			dataset = nPYc.Dataset()
			dataset.VariableType = nPYc.enumerations.VariableType.Spectral
			self.assertRaises(ValueError, nPYc.plotting._plotRSDs._plotRSDsHelper, dataset)

		with self.subTest(msg='No study samples'):
			dataset = nPYc.Dataset()
			dataset.VariableType = nPYc.enumerations.VariableType.Discrete
			dataset.sampleMetadata = pandas.DataFrame([nPYc.enumerations.SampleType.ProceduralBlank, nPYc.enumerations.SampleType.StudyPool, nPYc.enumerations.SampleType.ExternalReference], columns=['SampleType'])
			dataset.sampleMask = numpy.ones(3, dtype=bool)

			self.assertRaises(ValueError, nPYc.plotting._plotRSDs._plotRSDsHelper, dataset)


	def test_plotRSDsHelper(self):
		from nPYc.plotting._plotRSDs import _plotRSDsHelper

		noSamp = numpy.random.randint(100, high=500, size=None)
		noFeat = numpy.random.randint(200, high=400, size=None)
		dataset = generateTestDataset(noSamp, noFeat, dtype='MSDataset', variableType=VariableType.Discrete, sop='Generic')

		with self.subTest(msg='Check sorting'):
			rsdTable = _plotRSDsHelper(dataset)
			numpy.testing.assert_array_equal(numpy.sort(dataset.rsdSP)[::-1],rsdTable[SampleType.StudyPool].values)

		with self.subTest(msg='Check columns'):
			for sampleType in dataset.sampleMetadata['SampleType'].unique():
				self.assertTrue(sampleType in rsdTable.columns)

		with self.subTest(msg='With a feature mask'):

			dataset.featureMask[0:2:noFeat] = False
			rsdTable = _plotRSDsHelper(dataset)

			self.assertEqual(len(rsdTable), sum(dataset.featureMask))


	def test_rangeFrameLocator(self):

		from nPYc.plotting._rangeFrameLocator import rangeFrameLocator

		obtained = rangeFrameLocator([1,2,3,4,5,6,7,8,9,10], (2,9))
		expected = [2, 3, 4, 5, 6, 7, 8, 9]

		self.assertEqual(obtained, expected)

		obtained = rangeFrameLocator([1,4,6,8,10,12,14,16], (2,9))
		expected = [2, 4, 6, 9]

		self.assertEqual(obtained, expected)

		obtained = rangeFrameLocator([2,4,6,8,10,12,14,16], (1.1,13.5))
		expected = [1.1, 4, 6, 8, 10, 12, 13.5]

		self.assertEqual(obtained, expected)
