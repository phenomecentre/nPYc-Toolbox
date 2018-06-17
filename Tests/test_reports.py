# -*- coding: utf-8 -*-
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

sys.path.append("..")
import nPYc
from nPYc.enumerations import VariableType, AssayRole, SampleType, QuantificationType, CalibrationMethod
from nPYc.reports._generateReportTargeted import _postMergeLOQDataset, _prePostMergeLOQSummaryTable, _getAccuracyPrecisionTable
from pyChemometrics import ChemometricsPCA
from datetime import datetime, timedelta

from generateTestDataset import generateTestDataset, randomword


def datetime_range(start, count, delta):
	current = start
	for i in range(0, count):
		yield current
		current += delta


"""
Test report generation functions - at their most basic, simply check destinationPath is generated
"""
class test_reports_ms_feature_id(unittest.TestCase):

	def setUp(self):

		self.msData = nPYc.MSDataset(os.path.join('..', '..', 'npc-standard-project', 'Derived_Data', 'UnitTest1_PCSOP.069_QI.csv'), fileType='QI')
		self.msData.addSampleInfo(descriptionFormat='Filenames')

		self.msData.sampleMetadata['Correction Batch'] = 1
		self.msData.sampleMetadata['Run Order'] = [i for i in range(1, self.msData.noSamples + 1)]

		self.msData.sampleMetadata['Acquired Time'] = [d for d in datetime_range(datetime.now(), self.msData.noSamples, timedelta(minutes=15))]

		self.msData.fit = self.msData.intensityData


	def test_reports_generateMSIDrequests(self):
	
		with tempfile.TemporaryDirectory() as tmpdirname:
			with warnings.catch_warnings():
				warnings.simplefilter('ignore', UserWarning)
				nPYc.reports.generateMSIDrequests(self.msData,['3.17_262.0378m/z'], outputDir=tmpdirname)

			expectedPath = os.path.join(tmpdirname, 'ID Request_3.17_262.0378m-z.html')
			self.assertTrue(os.path.exists(expectedPath))

			expectedPath = os.path.join(tmpdirname, 'graphics', 'feature_3.17_262.0378m-z_intensity.png')
			self.assertTrue(os.path.exists(expectedPath))

			expectedPath = os.path.join(tmpdirname, 'graphics', 'feature_3.17_262.0378m-z_coelutants.png')
			self.assertTrue(os.path.exists(expectedPath))

			expectedPath = os.path.join(tmpdirname, 'graphics', 'feature_3.17_262.0378m-z_related.png')
			self.assertTrue(os.path.exists(expectedPath))

			expectedPath = os.path.join(tmpdirname, 'graphics', 'feature_3.17_262.0378m-z_abundance.png')
			self.assertTrue(os.path.exists(expectedPath))


class test_reports_generateSamplereport(unittest.TestCase):

	def setUp(self):
		
		self.data = nPYc.MSDataset(os.path.join('..', '..', 'npc-standard-project', 'Derived_Data', 'UnitTest1_PCSOP.069_QI.csv'), fileType='QI')
		self.data.addSampleInfo(descriptionFormat='Filenames')
		self.data.addSampleInfo(descriptionFormat='Raw Data', filePath=os.path.join('..', '..', 'npc-standard-project', 'Raw_Data', 'ms', 'parameters_data'))

		self.data.addSampleInfo(descriptionFormat='NPC LIMS', filePath=os.path.join('..', '..', 'npc-standard-project', 'Derived_Worklists', 'UnitTest1_MS_serum_PCSOP.069.csv'))
		self.data.addSampleInfo(descriptionFormat='NPC Subject Info', filePath=os.path.join('..', '..', 'npc-standard-project', 'Project_Description', 'UnitTest1_metadata_PCDOC.014.xlsx'))


	def test_report_samplesummary(self):
	
		# Generate destinationPath from sampleSummary
		sampleSummary = nPYc.reports._generateSampleReport(self.data, destinationPath=None, returnOutput=True)
	
		# Check returns against expected

		# Acquired - Totals
		assert sampleSummary['Acquired'].loc['All', 'Total'] == 115
		assert sampleSummary['Acquired'].loc['Study Sample', 'Total'] == 8
		assert sampleSummary['Acquired'].loc['Study Pool', 'Total'] == 11
		assert sampleSummary['Acquired'].loc['External Reference', 'Total'] == 1
		assert sampleSummary['Acquired'].loc['Serial Dilution', 'Total'] == 92
		assert sampleSummary['Acquired'].loc['Blank Sample', 'Total'] == 2
		assert sampleSummary['Acquired'].loc['Unspecified Sample Type or Assay Role', 'Total'] == 1

		# Acquired - Marked for exclusion
		assert sampleSummary['Acquired'].loc['All', 'Marked for Exclusion'] == 1
		assert sampleSummary['Acquired'].loc['Study Sample', 'Marked for Exclusion'] == 1
		assert sampleSummary['Acquired'].loc['Study Pool', 'Marked for Exclusion'] == 0
		assert sampleSummary['Acquired'].loc['External Reference', 'Marked for Exclusion'] == 0
		assert sampleSummary['Acquired'].loc['Serial Dilution', 'Marked for Exclusion'] == 0
		assert sampleSummary['Acquired'].loc['Blank Sample', 'Marked for Exclusion'] == 0
		assert sampleSummary['Acquired'].loc['Unspecified Sample Type or Assay Role', 'Marked for Exclusion'] == 0

		# Check details tables
		assert sampleSummary['MarkedToExclude Details'].shape == (1, 2)
		assert sampleSummary['UnknownType Details'].shape == (1,1)


	def test_report_samplesummary_postexclusion(self):	
		
		# Remove samples marked for exclusion (_x) or of unknown type
		self.data.excludeSamples(self.data.sampleMetadata.iloc[self.data.sampleMetadata['Skipped'].values==True]['Sample File Name'], on='Sample File Name', message='Skipped (_x)')
		self.data.excludeSamples(self.data.sampleMetadata[pandas.isnull(self.data.sampleMetadata['Sample Base Name'])]['Sample File Name'], on='Sample File Name', message='Unknown type')		
		self.data.applyMasks()
	
		# Generate destinationPath from sampleSummary
		sampleSummary = nPYc.reports._generateSampleReport(self.data, destinationPath=None, returnOutput=True)
	
		# Check returns against expected
	
		# Acquired - Totals
		assert sampleSummary['Acquired'].loc['All', 'Total'] == 113
		assert sampleSummary['Acquired'].loc['Study Sample', 'Total'] == 7
		assert sampleSummary['Acquired'].loc['Study Pool', 'Total'] == 11
		assert sampleSummary['Acquired'].loc['External Reference', 'Total'] == 1
		assert sampleSummary['Acquired'].loc['Serial Dilution', 'Total'] == 92
		assert sampleSummary['Acquired'].loc['Blank Sample', 'Total'] == 2
		assert 'Unspecified Sample Type or Assay Role' not in sampleSummary['Acquired'].index

		# Acquired - Marked for exclusion
		assert sampleSummary['Acquired'].loc['All', 'Marked for Exclusion'] == 0
		assert sampleSummary['Acquired'].loc['Study Sample', 'Marked for Exclusion'] == 0
		assert sampleSummary['Acquired'].loc['Study Pool', 'Marked for Exclusion'] == 0
		assert sampleSummary['Acquired'].loc['External Reference', 'Marked for Exclusion'] == 0
		assert sampleSummary['Acquired'].loc['Serial Dilution', 'Marked for Exclusion'] == 0
		assert sampleSummary['Acquired'].loc['Blank Sample', 'Marked for Exclusion'] == 0
		# Acquired - Already Excluded
		assert sampleSummary['Acquired'].loc['All', 'Already Excluded'] == 1
		assert sampleSummary['Acquired'].loc['Study Sample', 'Already Excluded'] == 1
		assert sampleSummary['Acquired'].loc['Study Pool', 'Already Excluded'] == 0
		assert sampleSummary['Acquired'].loc['External Reference', 'Already Excluded'] == 0
		assert sampleSummary['Acquired'].loc['Serial Dilution', 'Already Excluded'] == 0
		assert sampleSummary['Acquired'].loc['Blank Sample', 'Already Excluded'] == 0

class test_reports_nmr_generatereport(unittest.TestCase):

	def setUp(self):

		self.noSamp = numpy.random.randint(100, high=500, size=None)
		self.noFeat = numpy.random.randint(2000, high=10000, size=None)
		self.dataset = generateTestDataset(self.noSamp, self.noFeat, dtype='NMRDataset', variableType=VariableType.Continuum, sop='GenericNMRurine')


	def test_report_nmr_features(self):

		with tempfile.TemporaryDirectory() as tmpdirname:
			self.dataset.name = 'TestData'
			self.dataset._nmrQCChecks()

			nPYc.reports.generateReport(self.dataset, 'Feature Summary', destinationPath=tmpdirname)

			expectedPath = os.path.join(tmpdirname, 'TestData_report_featureSummary.html')
			self.assertTrue(os.path.exists(expectedPath))

			expectedPath = os.path.join(tmpdirname, 'graphics', 'report_featureSummary', 'TestData_calibrationCheck.png')
			self.assertTrue(os.path.exists(expectedPath))

			expectedPath = os.path.join(tmpdirname, 'graphics', 'report_featureSummary', 'TestData_finalFeatureBLWPplots1.png')
			self.assertTrue(os.path.exists(expectedPath))

			expectedPath = os.path.join(tmpdirname, 'graphics', 'report_featureSummary', 'TestData_finalFeatureBLWPplots3.png')
			self.assertTrue(os.path.exists(expectedPath))

			expectedPath = os.path.join(tmpdirname, 'graphics', 'report_featureSummary', 'TestData_peakWidthBoxplot.png')
			self.assertTrue(os.path.exists(expectedPath))


	def test_report_nmr_final(self):

		with tempfile.TemporaryDirectory() as tmpdirname:
			self.dataset.name = 'TestData'
			self.dataset.sampleMetadata['Sample Base Name'] = self.dataset.sampleMetadata['Sample File Name']
			self.dataset.sampleMetadata['BaselineFail'] = False
			self.dataset.sampleMetadata['WaterPeakFail'] = False
			self.dataset.sampleMetadata['Metadata Available'] = True
			nPYc.reports.generateReport(self.dataset, 'Final Report', destinationPath=tmpdirname)

			expectedPath = os.path.join(tmpdirname, 'TestData_report_finalSummary.html')
			self.assertTrue(os.path.exists(expectedPath))

			expectedPath = os.path.join(tmpdirname, 'graphics', 'report_finalSummary', 'TestData_finalFeatureBLWPplots1.png')
			self.assertTrue(os.path.exists(expectedPath))

			expectedPath = os.path.join(tmpdirname, 'graphics', 'report_finalSummary', 'TestData_finalFeatureBLWPplots3.png')
			self.assertTrue(os.path.exists(expectedPath))

			expectedPath = os.path.join(tmpdirname, 'graphics', 'report_finalSummary', 'TestData_peakWidthBoxplot.png')
			self.assertTrue(os.path.exists(expectedPath))


	def test_report_nmr_raises(self):

		from nPYc.reports._generateReportNMR import _generateReportNMR

		self.assertRaises(TypeError, _generateReportNMR, 'not a NMRDataset object', None)

		noSamp = numpy.random.randint(100, high=500, size=None)
		noFeat = numpy.random.randint(3000, high=10000, size=None)

		data = generateTestDataset(noSamp, noFeat, dtype='NMRDataset', sop='GenericNMRurine')

		self.assertRaises(ValueError, _generateReportNMR, data, 'Not a vaild plot type')
		self.assertRaises(TypeError, _generateReportNMR, data, 'feature summary', withExclusions='Not a bool')
		self.assertRaises(TypeError, _generateReportNMR, data, 'feature summary', destinationPath=True)


class test_reports_ms_generatereport(unittest.TestCase):

	def test_reports_ms_featuresummary(self):

		noSamp = numpy.random.randint(100, high=500, size=None)
		noFeat = numpy.random.randint(200, high=400, size=None)

		data = generateTestDataset(noSamp, noFeat, dtype='MSDataset', sop='GenericMS')
		data.name = 'test'
		data.featureMetadata['Peak Width'] = (0.01 - 20) * numpy.random.rand(data.noFeatures)
		data.sampleMetadata['Well'] = 1

		data.sampleMetadata.loc[:10, 'SampleType'] = SampleType.StudyPool
		data.sampleMetadata.loc[:10, 'AssayRole'] = AssayRole.LinearityReference
		data.sampleMetadata.loc[:10, 'Dilution'] = [0,1,2,3,4,5,6,7,8,9,10]

		with tempfile.TemporaryDirectory() as tmpdirname:

			nPYc.reports.generateReport(data, 'feature summary', destinationPath=tmpdirname)

			expectedPath = os.path.join(tmpdirname, 'test_report_featureSummary.html')
			self.assertTrue(os.path.exists(expectedPath))

			testFiles = ['test_RSDdistributionFigure.png', 'test_TICinLR.png', 'test_acquisitionStructure.png',
						 'test_correlationByPerc.png', 'test_ionMap.png', 'test_meanIntensityFeature.png', 'test_meanIntensitySample.png',
						 'test_peakWidth.png', 'test_rsdByPerc.png', 'test_rsdVsCorrelation.png']

			for testFile in testFiles:
				expectedPath = os.path.join(tmpdirname, 'graphics', 'report_featureSummary', testFile)
				self.assertTrue(os.path.exists(expectedPath))


	def test_reports_ms_correlationtodilution(self):

		data = nPYc.MSDataset(os.path.join('..', '..', 'npc-standard-project', 'Derived_Data', 'UnitTest1_PCSOP.069_QI.csv'), fileType='QI')
		data.addSampleInfo(descriptionFormat='Filenames')
		data.addSampleInfo(descriptionFormat='Raw Data', filePath=os.path.join('..', '..', 'npc-standard-project', 'Raw_Data', 'ms', 'parameters_data'))
		data.addSampleInfo(descriptionFormat='Batches')
		data.sampleMetadata['Correction Batch'] = data.sampleMetadata['Batch']
		data.corrExclusions = data.sampleMask

		with tempfile.TemporaryDirectory() as tmpdirname:

			nPYc.reports.generateReport(data, 'correlation to dilution', destinationPath=tmpdirname)

			expectedPath = os.path.join(tmpdirname, 'UnitTest1_PCSOP.069_QI_report_correlationToDilutionSummary.html')
			self.assertTrue(os.path.exists(expectedPath))

			testFiles = ['Batch 1.0, series 1.0 Histogram of Correlation To Dilution.png', 'Batch 1.0, series 1.0 LR Sample TIC (coloured by change in detector voltage).png',
						 'Batch 1.0, series 1.0 LR Sample TIC (coloured by dilution).png', 'Batch 1.0, series 2.0 Histogram of Correlation To Dilution.png',
						 'Batch 1.0, series 2.0 LR Sample TIC (coloured by change in detector voltage).png', 'Batch 1.0, series 2.0 LR Sample TIC (coloured by dilution).png',
						 'MeanAllSubsets Histogram of Correlation To Dilution.png', 'MeanAllSubsets LR Sample TIC (coloured by change in detector voltage).png',
						 'MeanAllSubsets LR Sample TIC (coloured by dilution).png', 'UnitTest1_PCSOP.069_QI_satFeaturesHeatmap.png']

			for testFile in testFiles:
				expectedPath = os.path.join(tmpdirname, 'graphics', 'report_correlationToDilutionSummary', testFile)
				self.assertTrue(os.path.exists(expectedPath))


	def test_reports_ms_batchcorrectiontest(self):

		data = nPYc.MSDataset(os.path.join('..', '..', 'npc-standard-project', 'Derived_Data', 'UnitTest1_PCSOP.069_QI.csv'), fileType='QI')
		data.addSampleInfo(descriptionFormat='Filenames')
		data.addSampleInfo(descriptionFormat='Raw Data', filePath=os.path.join('..', '..', 'npc-standard-project', 'Raw_Data', 'ms', 'parameters_data'))
		data.sampleMetadata['Correction Batch'] = data.sampleMetadata['Batch']

		with tempfile.TemporaryDirectory() as tmpdirname:

			nPYc.reports.generateReport(data, 'batch correction assessment', destinationPath=tmpdirname)

			expectedPath = os.path.join(tmpdirname, 'UnitTest1_PCSOP.069_QI_report_batchCorrectionAssessment.html')
			self.assertTrue(os.path.exists(expectedPath))

			testFiles = ['UnitTest1_PCSOP.069_QI_batchPlotFeature_3.17_145.0686m-z.png',
						'UnitTest1_PCSOP.069_QI_batchPlotFeature_3.17_262.0378m-z.png', 'UnitTest1_PCSOP.069_QI_TICdetectorBatches.png']

			for testFile in testFiles:
				expectedPath = os.path.join(tmpdirname, 'graphics', 'report_batchCorrectionAssessment', testFile)
				self.assertTrue(os.path.exists(expectedPath))


	def test_reports_ms_batchcorrectionresults(self):

		data = nPYc.MSDataset(os.path.join('..', '..', 'npc-standard-project', 'Derived_Data', 'UnitTest1_PCSOP.069_QI.csv'), fileType='QI')
		data.addSampleInfo(descriptionFormat='Filenames')
		data.addSampleInfo(descriptionFormat='Raw Data', filePath=os.path.join('..', '..', 'npc-standard-project', 'Raw_Data', 'ms', 'parameters_data'))
		data.sampleMetadata['Correction Batch'] = data.sampleMetadata['Batch']
		datacorrected = nPYc.batchAndROCorrection.correctMSdataset(data, parallelise=True)
		
		with tempfile.TemporaryDirectory() as tmpdirname:

			nPYc.reports.generateReport(data, 'batch correction summary', msDataCorrected=datacorrected, destinationPath=tmpdirname)
			
			expectedPath = os.path.join(tmpdirname, 'UnitTest1_PCSOP.069_QI_report_batchCorrectionSummary.html')
			self.assertTrue(os.path.exists(expectedPath))

			testFiles = ['npc-main.css', 'toolbox_logo.png']
			testFiles = ['UnitTest1_PCSOP.069_QI_BCS1_meanIntesityFeaturePRE.png', 'UnitTest1_PCSOP.069_QI_BCS1_meanIntesityFeaturePOST.png',
						'UnitTest1_PCSOP.069_QI_BCS2_TicPRE.png', 'UnitTest1_PCSOP.069_QI_BCS2_TicPOST.png',
						'UnitTest1_PCSOP.069_QI_BCS3_rsdByPercPRE.png', 'UnitTest1_PCSOP.069_QI_BCS3_rsdByPercPOST.png',
						'UnitTest1_PCSOP.069_QI_BCS4_RSDdistributionFigurePRE.png', 'UnitTest1_PCSOP.069_QI_BCS4_RSDdistributionFigurePOST.png']

			for testFile in testFiles:
				expectedPath = os.path.join(tmpdirname, 'graphics', 'report_batchCorrectionSummary', testFile)
				self.assertTrue(os.path.exists(expectedPath))


	def test_reports_ms_featureselection(self):

		data = nPYc.MSDataset(os.path.join('..', '..', 'npc-standard-project', 'Derived_Data', 'UnitTest1_PCSOP.069_QI.csv'), fileType='QI')
		data.addSampleInfo(descriptionFormat='Filenames')
		data.addSampleInfo(descriptionFormat='Raw Data', filePath=os.path.join('..', '..', 'npc-standard-project', 'Raw_Data', 'ms', 'parameters_data'))
		data.sampleMetadata['Correction Batch'] = data.sampleMetadata['Batch']

		with tempfile.TemporaryDirectory() as tmpdirname:

			nPYc.reports.generateReport(data, 'feature selection', destinationPath=tmpdirname)
			
			expectedPath = os.path.join(tmpdirname, 'UnitTest1_PCSOP.069_QI_report_featureSelectionSummary.html')
			self.assertTrue(os.path.exists(expectedPath))

			testFiles = ['UnitTest1_PCSOP.069_QI_noFeatures.png']

			for testFile in testFiles:
				expectedPath = os.path.join(tmpdirname, 'graphics', 'report_featureSelectionSummary', testFile)
				self.assertTrue(os.path.exists(expectedPath))


	def test_reports_ms_finalreport(self):

		data = nPYc.MSDataset(os.path.join('..', '..', 'npc-standard-project', 'Derived_Data', 'UnitTest1_PCSOP.069_QI.csv'), fileType='QI')
		data.addSampleInfo(descriptionFormat='Filenames')
		data.addSampleInfo(descriptionFormat='Raw Data', filePath=os.path.join('..', '..', 'npc-standard-project', 'Raw_Data', 'ms', 'parameters_data'))
		data.sampleMetadata['Correction Batch'] = data.sampleMetadata['Batch']
		data.sampleMetadata['Metadata Available'] = True
		with tempfile.TemporaryDirectory() as tmpdirname:

			nPYc.reports.generateReport(data, 'final report', destinationPath=tmpdirname)

			expectedPath = os.path.join(tmpdirname, 'UnitTest1_PCSOP.069_QI_report_finalSummary.html')
			self.assertTrue(os.path.exists(expectedPath))

			testFiles = ['UnitTest1_PCSOP.069_QI_finalFeatureIntensityHist.png', 'UnitTest1_PCSOP.069_QI_finalIonMap.png',
						 'UnitTest1_PCSOP.069_QI_finalRSDdistributionFigure.png', 'UnitTest1_PCSOP.069_QI_finalTIC.png', 'UnitTest1_PCSOP.069_QI_finalTICbatches.png']

			for testFile in testFiles:
				expectedPath = os.path.join(tmpdirname, 'graphics', 'report_finalSummary', testFile)
				self.assertTrue(os.path.exists(expectedPath))


	def test_reports_ms_raises(self):

		noSamp = numpy.random.randint(100, high=500, size=None)
		noFeat = numpy.random.randint(200, high=400, size=None)

		data = generateTestDataset(noSamp, noFeat, dtype='MSDataset', sop='GenericMS')

		self.assertRaises(TypeError, nPYc.reports.generateReport, None, 'feature summary')
		self.assertRaises(ValueError, nPYc.reports.generateReport, data, 'Unknown type')
		self.assertRaises(TypeError, nPYc.reports.generateReport, data, 'feature summary', withExclusions='Not a bool')
		self.assertRaises(TypeError, nPYc.reports.generateReport, data, 'feature summary', destinationPath=1)
		self.assertRaises(TypeError, nPYc.reports.generateReport, data, 'batch correction summary', msDataCorrected='Not an object')
		self.assertRaises(TypeError, nPYc.reports.generateReport, data, 'final report', pcaModel='Not a PCA model')


class test_reports_targeted_generatereport(unittest.TestCase):
	"""
	Test reporting of targetedDataset
	"""
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
		self.targetedDataset.sampleMetadata['Acquired Time'] = self.targetedDataset.sampleMetadata['Acquired Time'].astype(datetime)
		self.targetedDataset.featureMetadata = pandas.DataFrame({'Feature Name': ['Feature1', 'Feature2'],
																 'TargetLynx Feature ID': [1, 2],
																 'calibrationMethod': [
																	 CalibrationMethod.backcalculatedIS,
																	 CalibrationMethod.backcalculatedIS],
																 'quantificationType': [
																	 QuantificationType.QuantAltLabeledAnalogue,
																	 QuantificationType.QuantAltLabeledAnalogue],
																 'unitCorrectionFactor': [1., 1],
																 'Unit': ['a Unit', 'pg/uL'],
																 'Cpd Info': ['info cpd1', 'info cpd2'],
																 'LLOQ_batch1': [5., 20.],
																 'LLOQ_batch2': [20., 5.],
																 'ULOQ_batch1': [80., 100.],
																 'ULOQ_batch2': [100., 80.],
																 'extID1': ['F1', 'F2'],
																 'extID2': ['ID1', 'ID2']})
		self.targetedDataset._intensityData = numpy.array([[10., 10.], [90., 90.], [25., 70.]])
		self.targetedDataset.expectedConcentration = pandas.DataFrame(numpy.array([[1., 2.], [4., 5.], [7., 8.]]), columns=self.targetedDataset.featureMetadata['Feature Name'].values.tolist())
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
		self.targetedDataset.VariableType = VariableType.Discrete
		self.targetedDataset.initialiseMasks()

		# The prePostMergeLOQDataset
		self.prePostMerged = copy.deepcopy(self.targetedDataset)
		self.prePostMerged.featureMetadata['LLOQ'] = [20., 20.]
		self.prePostMerged.featureMetadata['ULOQ'] = [80., 80.]

		# prePostSummaryTable
		# LOQTable
		LOQTable = pandas.DataFrame(index=['Feature1', 'Feature2'])
		LOQTable['LLOQ']        = [20., 20.]
		LOQTable['LLOQ_batch1'] = [5., 20.]
		LOQTable['LLOQ_batch2'] = [20., 5.]
		LOQTable['LLOQ Diff.']  = ['X', 'X']
		LOQTable['ULOQ']        = [80., 80.]
		LOQTable['ULOQ_batch1'] = [80., 100.]
		LOQTable['ULOQ_batch2'] = [100., 80.]
		LOQTable['ULOQ Diff.']  = ['X', 'X']
		# LLOQTable
		LLOQTable = pandas.DataFrame(index=['Feature1', 'Feature2'])
		LLOQTable['Batch 1']         = [0, 1]
		LLOQTable['Batch 2']         = [0, 0]
		LLOQTable['Prev. Total']     = [0, 1]
		LLOQTable['Prev. Total (%)'] = [0., 33.3]
		LLOQTable['New Total']       = [1, 1]
		LLOQTable['New Total (%)']   = [33.3, 33.3]
		LLOQTable['Diff.']           = ['X', '']
		LLOQTable.columns.name       = '<LLOQ'
		# ULOQTable
		ULOQTable = pandas.DataFrame(index=['Feature1', 'Feature2'])
		ULOQTable['Batch 1']         = [0, 0]
		ULOQTable['Batch 2']         = [0, 1]
		ULOQTable['Prev. Total']     = [0, 1]
		ULOQTable['Prev. Total (%)'] = [0., 33.3]
		ULOQTable['New Total']       = [1, 1]
		ULOQTable['New Total (%)']   = [33.3, 33.3]
		ULOQTable['Diff.']           = ['X', '']
		ULOQTable.columns.name       = '>ULOQ'
		self.prePostSummaryTable = {'LOQTable': LOQTable, 'LLOQTable': LLOQTable, 'ULOQTable': ULOQTable}

		# getAccuracyPrecision input table
		self.tDataAccPrec = nPYc.TargetedDataset('', fileType='empty')
		self.tDataAccPrec.name = 'unittest'
		self.tDataAccPrec.sampleMetadata = pandas.DataFrame({'AssayRole': [AssayRole.PrecisionReference, AssayRole.PrecisionReference, AssayRole.PrecisionReference, AssayRole.PrecisionReference, AssayRole.PrecisionReference, AssayRole.PrecisionReference, AssayRole.Assay, AssayRole.Assay],
															 'SampleType': [SampleType.StudyPool, SampleType.StudyPool, SampleType.StudyPool, SampleType.ExternalReference, SampleType.ExternalReference, SampleType.ExternalReference, SampleType.StudySample, SampleType.StudySample]})
		self.tDataAccPrec.sampleMetadata['Sample File Name'] = ['UnitTest_targeted_file_001', 'UnitTest_targeted_file_002', 'UnitTest_targeted_file_003', 'UnitTest_targeted_file_004', 'UnitTest_targeted_file_005', 'UnitTest_targeted_file_006', 'UnitTest_targeted_file_007', 'UnitTest_targeted_file_008']
		self.tDataAccPrec.sampleMetadata['Sample Name'] = ['Sample1', 'Sample2', 'Sample3', 'Sample4', 'Sample5', 'Sample6', 'Sample7', 'Sample8']
		self.tDataAccPrec.sampleMetadata['Acqu Date'] = ['10-Sep-16', '10-Sep-16', '10-Sep-16', '10-Sep-16', '10-Sep-16', '10-Sep-16', '10-Sep-16', '10-Sep-16']
		self.tDataAccPrec.sampleMetadata['Acqu Time'] = ['03:23:02', '04:52:35', '05:46:40', '06:10:43', '07:30:06', '08:50:29', '09:05:42', '09:20:58']
		self.tDataAccPrec.sampleMetadata['Instrument'] = ['XEVO-TQS#UnitTest', 'XEVO-TQS#UnitTest', 'XEVO-TQS#UnitTest', 'XEVO-TQS#UnitTest', 'XEVO-TQS#UnitTest', 'XEVO-TQS#UnitTest', 'XEVO-TQS#UnitTest', 'XEVO-TQS#UnitTest']
		self.tDataAccPrec.sampleMetadata['Acquired Time'] = [datetime(2016, 9, 10, 3, 23, 2), datetime(2016, 9, 10, 4, 52, 35), datetime(2016, 9, 10, 5, 46, 40), datetime(2016, 9, 10, 6, 10, 43), datetime(2016, 9, 10, 7, 30, 6), datetime(2016, 9, 10, 8, 50, 29), datetime(2016, 9, 10, 9, 5, 42), datetime(2016, 9, 10, 9, 20, 58)]
		self.tDataAccPrec.sampleMetadata['Run Order'] = [0, 1, 2, 3, 4, 5, 6, 7]
		self.tDataAccPrec.sampleMetadata['Batch'] = [1, 1, 1, 1, 1, 1, 1, 1]
		self.tDataAccPrec.sampleMetadata['Dilution'] = [numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan]
		self.tDataAccPrec.sampleMetadata['Correction Batch'] = [numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan]
		self.tDataAccPrec.sampleMetadata['Subject ID'] = ['', '', '', '', '', '', '', '']
		self.tDataAccPrec.sampleMetadata['Sampling ID'] = ['', '', '', '', '', '', '', '']
		self.tDataAccPrec.sampleMetadata['Sample Base Name'] = ['', '', '', '', '', '', '', '']
		self.tDataAccPrec.sampleMetadata['Exclusion Details'] = ['', '', '', '', '', '', '', '']
		self.tDataAccPrec.featureMetadata = pandas.DataFrame({'Feature Name': ['Feature1', 'Feature2','Feature3']})
		self.tDataAccPrec.featureMetadata['calibrationMethod'] = [CalibrationMethod.backcalculatedIS, CalibrationMethod.backcalculatedIS, CalibrationMethod.backcalculatedIS]
		self.tDataAccPrec.featureMetadata['quantificationType'] = [QuantificationType.QuantOwnLabeledAnalogue, QuantificationType.QuantAltLabeledAnalogue, QuantificationType.QuantAltLabeledAnalogue]
		self.tDataAccPrec.featureMetadata['Unit'] = ['a Unit', 'pg/uL', 'a Unit']
		self.tDataAccPrec.featureMetadata['LLOQ'] = [0., 0., 0.]
		self.tDataAccPrec.featureMetadata['ULOQ'] = [1000., numpy.nan, 1000.]
		self.tDataAccPrec.featureMetadata['extID1'] = ['ID1', 'ID2', 'ID3']
		self.tDataAccPrec._intensityData = numpy.array([[10., 10., 10.], [5., 50., 5.], [15., 90., 15.], [95., 30., 95.], [100., 30., 100], [105., 50., 105.], [100., 101., 100], [95., 105., 95]])
		self.tDataAccPrec.expectedConcentration = pandas.DataFrame(numpy.array([[10., 50., 10.], [10., 50., 10.], [10., 50., 10.], [100., 50., 100.], [100., 50., 100.], [100., 50., 100.], [numpy.nan, numpy.nan, numpy.nan], [numpy.nan, numpy.nan, numpy.nan]]), columns=self.tDataAccPrec.featureMetadata['Feature Name'].values.tolist())
		self.tDataAccPrec.sampleMetadataExcluded = []
		self.tDataAccPrec.featureMetadataExcluded = []
		self.tDataAccPrec.intensityDataExcluded = []
		self.tDataAccPrec.expectedConcentrationExcluded = []
		self.tDataAccPrec.excludedFlag = []
		self.tDataAccPrec.calibration = dict()
		self.tDataAccPrec.calibration['calibIntensityData'] = numpy.ndarray((0, 3))
		self.tDataAccPrec.calibration['calibSampleMetadata'] = pandas.DataFrame()
		self.tDataAccPrec.calibration['calibFeatureMetadata'] = pandas.DataFrame({'Feature Name': ['Feature1', 'Feature2', 'Feature3']})
		self.tDataAccPrec.calibration['calibExpectedConcentration'] = pandas.DataFrame(columns=['Feature1', 'Feature2', 'Feature3'])
		self.tDataAccPrec.Attributes['methodName'] = 'unittest'
		self.tDataAccPrec.Attributes['externalID'] = ['extID1']
		self.tDataAccPrec.VariableType = VariableType.Discrete
		self.tDataAccPrec.initialiseMasks()

		# getAccuracyPrecision result
		self.resAcc = pandas.DataFrame({10.0: [100.0, numpy.nan, 100.0, numpy.nan, numpy.nan, numpy.nan, 100.0, numpy.nan, 100.0], 50.0: [numpy.nan, numpy.nan, numpy.nan, 100.0, 73.3333, 86.6667, numpy.nan, numpy.nan, numpy.nan], 100.0: [numpy.nan, 100.0, 100.0, numpy.nan, numpy.nan, numpy.nan, numpy.nan, 100.0, 100.0], 'Feature': ['Feature1', 'Feature1', 'Feature1', 'Feature2', 'Feature2', 'Feature2', 'Feature3', 'Feature3','Feature3'], 'Sample Type': ['Study Pool', 'External Reference', 'All Samples', 'Study Pool', 'External Reference', 'All Samples', 'Study Pool', 'External Reference', 'All Samples']})
		self.resAcc = self.resAcc.set_index(['Feature', 'Sample Type'])
		self.resAcc.fillna(value='', inplace=True)
		self.resAcc.columns.name = 'Accuracy'

		self.resPre = pandas.DataFrame({10.0: [40.8248, numpy.nan, 40.8248, numpy.nan, numpy.nan, numpy.nan, 40.8248, numpy.nan, 40.8248], 50.0: [numpy.nan, numpy.nan, numpy.nan, 65.3197, 25.7129, 57.5639, numpy.nan, numpy.nan, numpy.nan], 100.0: ['', 4.08248, 4.08248, numpy.nan, numpy.nan, numpy.nan, '', 4.08248, 4.08248], 'Feature': ['Feature1', 'Feature1', 'Feature1', 'Feature2', 'Feature2', 'Feature2', 'Feature3', 'Feature3', 'Feature3'], 'Sample Type': ['Study Pool', 'External Reference', 'All Samples', 'Study Pool', 'External Reference', 'All Samples', 'Study Pool', 'External Reference', 'All Samples']})
		self.resPre = self.resPre.set_index(['Feature', 'Sample Type'])
		self.resPre.fillna(value='', inplace=True)
		self.resPre.columns.name = 'Precision'

		self.resAccPre = pandas.DataFrame({'1': [100.0, numpy.nan, 100.0, numpy.nan, numpy.nan, numpy.nan, 100.0, numpy.nan, 100.0], '2': [40.8248, numpy.nan, 40.8248, numpy.nan, numpy.nan, numpy.nan, 40.8248, numpy.nan, 40.8248], '3': [numpy.nan, numpy.nan, numpy.nan, 100, 73.3333, 86.6667, numpy.nan, numpy.nan, numpy.nan], '4': [numpy.nan, numpy.nan, numpy.nan, 65.3197, 25.7129, 57.5639, numpy.nan, numpy.nan, numpy.nan], '5': ['', 100.0, 100.0, numpy.nan, numpy.nan, numpy.nan, '', 100.0, 100.0], '6': [numpy.nan, 4.08248, 4.08248, numpy.nan, numpy.nan, numpy.nan, numpy.nan, 4.08248, 4.08248]})
		self.resAccPre = self.resAccPre.transpose()
		tuplesCol = [(10.0, 'Acc.'), (10.0, 'Prec.'), (50.0, 'Acc.'), (50.0, 'Prec.'), (100.0, 'Acc.'), (100.0, 'Prec.')]
		colIdx = pandas.MultiIndex.from_tuples(tuplesCol, names=['Concentration', 'Measure'])
		self.resAccPre.index = colIdx
		self.resAccPre = self.resAccPre.transpose()
		tuplesRow = [('Feature1', 'Study Pool'), ('Feature1', 'External Reference'), ('Feature1', 'All Samples'), ('Feature2', 'Study Pool'), ('Feature2', 'External Reference'), ('Feature2', 'All Samples'), ('Feature3', 'Study Pool'), ('Feature3', 'External Reference'), ('Feature3', 'All Samples')]
		rowIdx = pandas.MultiIndex.from_tuples(tuplesRow, names=['Feature', 'Sample Type'])
		self.resAccPre.index = rowIdx
		self.resAccPre.fillna(value='', inplace=True)


	def test_report_postMergeLOQDataset(self):
		# Expected
		expected = copy.deepcopy(self.prePostMerged)
		# Result
		inputDataset = copy.deepcopy(self.targetedDataset)
		result = _postMergeLOQDataset(inputDataset)

		# Class
		self.assertEqual(type(result), type(expected))
		# sampleMetadata
		pandas.util.testing.assert_frame_equal(result.sampleMetadata, expected.sampleMetadata)
		# featureMetadata
		pandas.util.testing.assert_frame_equal(result.featureMetadata, expected.featureMetadata)
		# intensityData
		numpy.testing.assert_array_equal(result._intensityData, expected._intensityData)
		# expectedConcentration
		pandas.util.testing.assert_frame_equal(result.expectedConcentration, expected.expectedConcentration)
		# Calibration
		numpy.testing.assert_array_equal(result.calibration['calibIntensityData'], expected.calibration['calibIntensityData'])
		pandas.util.testing.assert_frame_equal(result.calibration['calibSampleMetadata'], expected.calibration['calibSampleMetadata'])
		pandas.util.testing.assert_frame_equal(result.calibration['calibFeatureMetadata'], expected.calibration['calibFeatureMetadata'])
		pandas.util.testing.assert_frame_equal(result.calibration['calibExpectedConcentration'], expected.calibration['calibExpectedConcentration'])


	def test_report_prePostMergeLOQSummaryTable(self):
		# Expected
		expected = copy.deepcopy(self.prePostSummaryTable)
		# Result
		inputDataset = copy.deepcopy(self.prePostMerged)
		result = _prePostMergeLOQSummaryTable(inputDataset)

		# LOQTable
		pandas.util.testing.assert_frame_equal(result['LOQTable'],  expected['LOQTable'])
		# LLOQTable
		pandas.util.testing.assert_frame_equal(result['LLOQTable'], expected['LLOQTable'])
		# ULOQTable
		pandas.util.testing.assert_frame_equal(result['ULOQTable'], expected['ULOQTable'])


	def test_report_getaccuracyprecisiontable(self):
		# Expected
		expectedAcc    = copy.deepcopy(self.resAcc)
		expectedPre    = copy.deepcopy(self.resPre)
		expectedAccPre = copy.deepcopy(self.resAccPre)
		# Result
		inputDataset = copy.deepcopy(self.tDataAccPrec)
		resultAcc    = _getAccuracyPrecisionTable(inputDataset, table='accuracy')
		resultPre    = _getAccuracyPrecisionTable(inputDataset, table='precision')
		resultAccPre = _getAccuracyPrecisionTable(inputDataset, table='both')

		# Multiindex order might not be identical on all platforms
		expectedAcc.sort_index(level=[0,1], inplace=True)
		expectedPre.sort_index(level=[0,1], inplace=True)
		expectedAccPre.sort_index(level=[0,1], inplace=True)
		expectedAcc.sort_index(axis=1, inplace=True)
		expectedPre.sort_index(axis=1, inplace=True)
		expectedAccPre.sort_index(axis=1, inplace=True)
		# Result
		resultAcc.sort_index(level=[0,1], inplace=True)
		resultPre.sort_index(level=[0,1], inplace=True)
		resultAccPre.sort_index(level=[0,1], inplace=True)
		resultAcc.sort_index(axis=1, inplace=True)
		resultPre.sort_index(axis=1, inplace=True)
		resultAccPre.sort_index(axis=1, inplace=True)
		# Accuracy Table
		
		pandas.util.testing.assert_frame_equal(resultAcc, expectedAcc, check_dtype=False, check_index_type=False, check_column_type=False, check_like=False)
		# Precision Table
		pandas.util.testing.assert_frame_equal(resultPre, expectedPre, check_dtype=False, check_index_type=False, check_column_type=False, check_like=False)
		# Both Tables
		pandas.util.testing.assert_frame_equal(resultAccPre, expectedAccPre, check_dtype=False, check_index_type=False, check_column_type=False, check_like=False)


	def test_reports_targeted_featuresummary(self):
		inputDataset = copy.deepcopy(self.tDataAccPrec)

		with tempfile.TemporaryDirectory() as tmpdirname:
			with warnings.catch_warnings():
				warnings.simplefilter('ignore', UserWarning)
				nPYc.reports.generateReport(inputDataset, 'feature summary', destinationPath=tmpdirname)

			expectedPath = os.path.join(tmpdirname, 'unittest_report_featureSummary.html')
			self.assertTrue(os.path.exists(expectedPath))

			testFiles = ['unittest_AcquisitionStructure.png', 'unittest_FeatureAccuracy-A.png', 'unittest_FeatureAccuracy-B.png', 'unittest_FeatureConcentrationDistribution-A.png', 'unittest_FeatureConcentrationDistribution-B.png', 'unittest_FeaturePrecision-A.png', 'unittest_FeaturePrecision-B.png']

			for testFile in testFiles:
				expectedPath = os.path.join(tmpdirname, 'graphics', 'report_featureSummary', testFile)
				self.assertTrue(os.path.exists(expectedPath))


	def test_reports_targeted_mergeloqassessment(self):
		inputDataset = copy.deepcopy(self.targetedDataset)

		with tempfile.TemporaryDirectory() as tmpdirname:
			nPYc.reports.generateReport(inputDataset, 'merge LOQ assessment', destinationPath=tmpdirname)

			expectedPath = os.path.join(tmpdirname, 'unittest_report_mergeLoqAssessment.html')
			self.assertTrue(os.path.exists(expectedPath))

			testFiles = ['unittest_ConcentrationPrePostMergeLOQ.png']

			for testFile in testFiles:
				expectedPath = os.path.join(tmpdirname, 'graphics', 'report_mergeLoqAssessment', testFile)
				self.assertTrue(os.path.exists(expectedPath))


	def test_reports_targeted_finalreport(self):

		inputDataset = copy.deepcopy(self.tDataAccPrec)

		with tempfile.TemporaryDirectory() as tmpdirPCA:

			pcaModel = nPYc.multivariate.exploratoryAnalysisPCA(inputDataset, withExclusions=True, scaling=1.0, maxComponents=2)


		with tempfile.TemporaryDirectory() as tmpdirname:

			nPYc.reports.multivariateReport.multivariateQCreport(inputDataset, pcaModel=pcaModel, reportType='analytical', withExclusions=True, destinationPath=tmpdirPCA)
			inputDataset.sampleMetadata['Metadata Available'] = True
			with warnings.catch_warnings():
				warnings.simplefilter('ignore', UserWarning)
				warnings.simplefilter('ignore', RuntimeWarning)
				nPYc.reports.generateReport(inputDataset, 'final report', destinationPath=tmpdirname, pcaModel=pcaModel)

			expectedPath = os.path.join(tmpdirname, 'unittest_report_finalSummary.html')
			self.assertTrue(os.path.exists(expectedPath))

			testFiles = ['unittest_AcquisitionStructure.png',
						 'unittest_FeatureAccuracy-A.png',
						 'unittest_FeatureAccuracy-B.png',
						 'unittest_FeatureConcentrationDistribution-A.png',
						 'unittest_FeatureConcentrationDistribution-B.png',
						 'unittest_FeaturePrecision-A.png',
						 'unittest_FeaturePrecision-B.png']

			multivariateTestFiles = ['unittest_PCAloadingsPlot_PCAloadings.png', 'unittest_PCAscoresPlot_SampleTypePC1vsPC2.png']

			for testFile in testFiles:
				expectedPath = os.path.join(tmpdirname, 'graphics', 'report_finalSummary', testFile)
				self.assertTrue(os.path.exists(expectedPath))
			# Separete here in case a future refactor splits different folders for PCA as _basicPCAReport is only being used in
			# final summary reports
			for testFile in multivariateTestFiles:
				expectedPath = os.path.join(tmpdirname, 'graphics', 'report_finalSummary', testFile)
				self.assertTrue(os.path.exists(expectedPath))


	def test_reports_targeted_raises(self):
		inputDataset    = copy.deepcopy(self.targetedDataset)
		notValidDataset = copy.deepcopy(self.targetedDataset)
		delattr(notValidDataset, 'expectedConcentration')

		# no input dataset
		self.assertRaises(TypeError, nPYc.reports.generateReport, None, 'feature summary')
		# not a BasicTargetedDataset
		self.assertRaises(ValueError, nPYc.reports.generateReport, notValidDataset, 'feature summary')
		# not a known reportType
		self.assertRaises(ValueError, nPYc.reports.generateReport, inputDataset, 'Unknown type')
		# withExclusions is not a bool
		self.assertRaises(TypeError, nPYc.reports.generateReport, inputDataset, 'feature summary', withExclusions='Not a bool')
		# destinationPath is not str or None
		self.assertRaises(TypeError, nPYc.reports.generateReport, inputDataset, 'feature summary', destinationPath=1)
		# numberPlotPerRowLOQ is not an int
		self.assertRaises(TypeError, nPYc.reports.generateReport, inputDataset, 'feature summary', numberPlotPerRowLOQ='Not an int')
		# numberPlotPerRowFeature is not an int
		self.assertRaises(TypeError, nPYc.reports.generateReport, inputDataset, 'feature summary', numberPlotPerRowFeature='Not an int')


class test_reports_modules(unittest.TestCase):

	def test_reports_generatefeaturereport(self):

		from nPYc.reports._generateFeatureDistributionReport import generateFeatureDistributionReport

		# Synthetic data
		noSamp = numpy.random.randint(100, high=500, size=None)
		noFeat = numpy.random.randint(200, high=400, size=None)

		dataset = generateTestDataset(noSamp, noFeat)

		# Create random feature groupings
		reportingGroups = dict()
		noGroups = numpy.random.randint(1, high=3, size=None)
		for i in range(noGroups):
			groupName = randomword(10)
			reportingGroups[groupName] = dict()

			noPlots = numpy.random.randint(1, high=4, size=None)
			for j in range(noPlots):
				featureIndexes = numpy.random.randint(0, high=noFeat-1, size=numpy.random.randint(1,5, size=None))

				reportingGroups[groupName][randomword(10)] = dataset.featureMetadata['Feature Name'].iloc[featureIndexes].values

		with tempfile.TemporaryDirectory() as tmpdirname:

			# Build reports
			report = generateFeatureDistributionReport(dataset, reportingGroups, destinationPath=tmpdirname)

			for groupName in report.keys():
				for plotName in report[groupName].keys():
					self.assertTrue(os.path.exists(report[groupName][plotName]))


	def test_reports_generatefeaturereport_filterUnits(self):

		from nPYc.reports._generateFeatureDistributionReport import generateFeatureDistributionReport

		# Synthetic data
		noSamp = numpy.random.randint(100, high=500, size=None)
		noFeat = numpy.random.randint(200, high=400, size=None)

		dataset = generateTestDataset(noSamp, noFeat)

		dataset.featureMetadata['Unit'] = 'Unit A'

		newUnits = numpy.arange(noFeat)
		numpy.random.shuffle(newUnits)

		newUnits = round(noFeat / 2)

		dataset.featureMetadata.iloc[newUnits, dataset.featureMetadata.columns.get_loc('Unit')] = 'Unit B'

		# Create random feature groupings
		reportingGroups = dict()
		noGroups = numpy.random.randint(1, high=3, size=None)
		for i in range(noGroups):
			groupName = randomword(10)
			reportingGroups[groupName] = dict()

			noPlots = numpy.random.randint(1, high=4, size=None)
			for j in range(noPlots):
				featureIndexes = numpy.random.randint(0, high=noFeat-1, size=numpy.random.randint(1,5, size=None))

				reportingGroups[groupName][randomword(10)] = dataset.featureMetadata['Feature Name'].iloc[featureIndexes].values

		with tempfile.TemporaryDirectory() as tmpdirname:

			# Build reports
			report = generateFeatureDistributionReport(dataset, reportingGroups, filterUnits='Unit B', destinationPath=tmpdirname)

			for groupName in report.keys():
				for plotName in report[groupName].keys():
					self.assertTrue(os.path.exists(report[groupName][plotName]))


	@unittest.mock.patch('sys.stdout', new_callable=io.StringIO)
	def test_reports_generateBasicPCAReport(self, mock_stdout):

		from nPYc.reports._generateBasicPCAReport import generateBasicPCAReport
		from nPYc.multivariate import exploratoryAnalysisPCA
		from nPYc.enumerations import AssayRole, SampleType

		# Synthetic data
		noSamp = numpy.random.randint(100, high=500, size=None)
		noFeat = numpy.random.randint(200, high=400, size=None)

		dataset = generateTestDataset(noSamp, noFeat)

		SSmask = (dataset.sampleMetadata['SampleType'].values == SampleType.StudySample) & (dataset.sampleMetadata['AssayRole'].values == AssayRole.Assay)
		SPmask = (dataset.sampleMetadata['SampleType'].values == SampleType.StudyPool) & (dataset.sampleMetadata['AssayRole'].values == AssayRole.PrecisionReference)
		ERmask = (dataset.sampleMetadata['SampleType'].values == SampleType.ExternalReference) & (dataset.sampleMetadata['AssayRole'].values == AssayRole.PrecisionReference)

		dataset.sampleMetadata.loc[~SSmask & ~SPmask & ~ERmask, 'Plot Sample Type'] = 'Sample'
		dataset.sampleMetadata.loc[SSmask, 'Plot Sample Type'] = 'Study Sample'
		dataset.sampleMetadata.loc[SPmask, 'Plot Sample Type'] = 'Study Pool'
		dataset.sampleMetadata.loc[ERmask, 'Plot Sample Type'] = 'External Reference'

		pcaModel = exploratoryAnalysisPCA(dataset)

		with tempfile.TemporaryDirectory() as tmpdirname:

			if not os.path.exists(os.path.join(tmpdirname, 'graphics')):
				os.makedirs(os.path.join(tmpdirname, 'graphics'))

			report = generateBasicPCAReport(pcaModel, dataset, destinationPath=tmpdirname)

			for groupName in report['QCscores'].keys():
				path = os.path.join(tmpdirname, report['QCscores'][groupName])
				self.assertTrue(os.path.exists(path))

			for groupName in report['loadings'].keys():
				path = os.path.join(tmpdirname, report['loadings'][groupName])
				self.assertTrue(os.path.exists(path))

		# with self.subTest(msg='ploting interactivly'):
		#
		# 	report = generateBasicPCAReport(pcaModel, dataset, destinationPath=None)
		# 	self.assertIsNone(report)


	def test_reports_generateBasicPCAReport_raises(self):

		from nPYc.reports._generateBasicPCAReport import generateBasicPCAReport

		self.assertRaises(TypeError, generateBasicPCAReport, 'not a ChemometricsPCA object', None)


class test_reports_multivariate(unittest.TestCase):

	def setUp(self):

		noSamp = numpy.random.randint(30, high=500, size=None)
		noFeat = numpy.random.randint(200, high=400, size=None)

		self.dataset = generateTestDataset(noSamp, noFeat)
		self.dataset.Attributes['excludeFromPlotting'] = []
		self.dataset.Attributes['analyticalMeasurements'] = {}


	def test_multivariateqcreport_all(self):

		with tempfile.TemporaryDirectory() as tmpdirname:
			# PCA model generation
			pcaModel = nPYc.multivariate.exploratoryAnalysisPCA(self.dataset)

			with self.subTest(msg='PCA object generation'):
				self.assertIsInstance(pcaModel, ChemometricsPCA)

				numpy.testing.assert_equal(pcaModel._npyc_dataset_shape['NumberSamples'], self.dataset.intensityData.shape[0])
				numpy.testing.assert_equal(pcaModel._npyc_dataset_shape['NumberFeatures'], self.dataset.intensityData.shape[1])

			# Report generation
			with self.subTest(msg='Report generation'):
				nPYc.reports.multivariateQCreport(self.dataset, pcaModel=pcaModel, reportType='all', destinationPath=tmpdirname)

				expectedPath = os.path.join(tmpdirname, 'Dataset_report_multivariateAll.html')
				self.assertTrue(os.path.exists(expectedPath))

				testFiles = ['Dataset_PCAloadingsPlot_PCAloadingsPC1.png', 'Dataset_PCAloadingsPlot_PCAloadingsPC2.png', 'Dataset_PCAscoresPlot_SampleTypePC1vsPC2.png',
							 'Dataset_PCAscreePlot.png', 'Dataset_metadataPlot_metadataDistribution_categorical0.png', 'Dataset_metadataPlot_metadataDistribution_continuous0.png',
							 'Dataset_metadataPlot_metadataDistribution_date0.png', 'Dataset_modOutliersPlot.png', 'Dataset_sigCorHeatmap.png', 'Dataset_sigKruHeatmap.png',
							 'Dataset_strongOutliersPlot.png', 'npc-main.css', 'toolbox_logo.png']

				for testFile in testFiles:
					expectedPath = os.path.join(tmpdirname, 'graphics', 'report_multivariateAll', testFile)
					self.assertTrue(os.path.exists(expectedPath))


	def test_multivariateqcreport_analytical(self):

		with tempfile.TemporaryDirectory() as tmpdirname:

			self.dataset.Attributes['analyticalMeasurements'] = {'SampleType': 'categorical', 'AssayRole' : 'categorical', 'Acquired Time' : 'date', 'Run Order' : 'continuous', 'Dilution' : 'continuous', 'Detector' : 'continuous', 'Correction Batch' : 'categorical', 'Batch' : 'categorical'}

			pcaModel = nPYc.multivariate.exploratoryAnalysisPCA(self.dataset)

			with self.subTest(msg='PCA object generation'):
				self.assertIsInstance(pcaModel, ChemometricsPCA)

				numpy.testing.assert_equal(pcaModel._npyc_dataset_shape['NumberSamples'], self.dataset.intensityData.shape[0])
				numpy.testing.assert_equal(pcaModel._npyc_dataset_shape['NumberFeatures'], self.dataset.intensityData.shape[1])


			with self.subTest(msg='Report generation'):

				nPYc.reports.multivariateQCreport(self.dataset, pcaModel=pcaModel, reportType='analytical',
												  destinationPath=tmpdirname)

				expectedPath = os.path.join(tmpdirname, 'Dataset_report_multivariateAnalytical.html')
				self.assertTrue(os.path.exists(expectedPath))

				testFiles = ['Dataset_PCAloadingsPlot_PCAloadingsPC1.png', 'Dataset_PCAloadingsPlot_PCAloadingsPC2.png', 'Dataset_PCAscoresPlot_SampleTypePC1vsPC2.png',
							 'Dataset_PCAscreePlot.png', 'Dataset_metadataPlot_metadataDistribution_categorical0.png', 'Dataset_metadataPlot_metadataDistribution_continuous0.png',
							 'Dataset_metadataPlot_metadataDistribution_date0.png', 'Dataset_modOutliersPlot.png', 'Dataset_sigCorHeatmap.png', 'Dataset_sigKruHeatmap.png',
							 'Dataset_strongOutliersPlot.png', 'npc-main.css', 'toolbox_logo.png']

				for testFile in testFiles:
					expectedPath = os.path.join(tmpdirname, 'graphics', 'report_multivariateAnalytical', testFile)
					self.assertTrue(os.path.exists(expectedPath))


	def test_multivariateqcreport_raises(self):

		pcaModel = nPYc.multivariate.exploratoryAnalysisPCA(self.dataset)

		self.assertRaises(TypeError, nPYc.reports.multivariateQCreport, 'not a dataset', pcaModel)

		self.assertRaises(TypeError, nPYc.reports.multivariateQCreport, self.dataset, 'not a pca model' )

		self.assertRaises(ValueError, nPYc.reports.multivariateQCreport, self.dataset, pcaModel, reportType=1)

		self.assertRaises(TypeError, nPYc.reports.multivariateQCreport, self.dataset, pcaModel, withExclusions='not a bool')

		self.assertRaises(TypeError, nPYc.reports.multivariateQCreport, self.dataset, pcaModel, biologicalMeasurements='not a dict')

		self.assertRaises(ValueError, nPYc.reports.multivariateQCreport, self.dataset, pcaModel, reportType=1)

		self.assertRaises(ValueError, nPYc.reports.multivariateQCreport, self.dataset, pcaModel, dModX_criticalVal=0.05, dModX_criticalVal_type=None)

		self.assertRaises(ValueError, nPYc.reports.multivariateQCreport, self.dataset, pcaModel, dModX_criticalVal=-1, dModX_criticalVal_type='Fcrit')

		self.assertRaises(ValueError, nPYc.reports.multivariateQCreport, self.dataset, pcaModel, dModX_criticalVal=101, dModX_criticalVal_type='Fcrit')

		self.assertRaises(ValueError, nPYc.reports.multivariateQCreport, self.dataset, pcaModel, dModX_criticalVal_type = -1)

		self.assertRaises(ValueError, nPYc.reports.multivariateQCreport, self.dataset, pcaModel, dModX_criticalVal_type = 'wrong threshold')

		self.assertRaises(ValueError, nPYc.reports.multivariateQCreport, self.dataset, pcaModel, scores_criticalVal=-1)

		self.assertRaises(ValueError, nPYc.reports.multivariateQCreport, self.dataset, pcaModel, scores_criticalVal=101)

		self.assertRaises(ValueError, nPYc.reports.multivariateQCreport, self.dataset, pcaModel, kw_threshold=-1)

		self.assertRaises(ValueError, nPYc.reports.multivariateQCreport, self.dataset, pcaModel, r_threshold=-1.5)

		self.assertRaises(ValueError, nPYc.reports.multivariateQCreport, self.dataset, pcaModel, hotellings_alpha=-1.5)

		self.assertRaises(TypeError, nPYc.reports.multivariateQCreport, self.dataset, pcaModel, excludeFields='not a list')

		self.assertRaises(TypeError, nPYc.reports.multivariateQCreport, self.dataset, pcaModel, destinationPath=1)

		self.assertRaises(ValueError, nPYc.reports.multivariateQCreport, self.dataset, pcaModel, dModX_criticalVal=0.05)


if __name__ == '__main__':
	unittest.main()
