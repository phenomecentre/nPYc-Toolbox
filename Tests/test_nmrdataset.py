import pandas
import numpy
import sys
import unittest
import os
import copy
import warnings
import tempfile


sys.path.append("..")
import nPYc
from nPYc.enumerations import AssayRole, SampleType
from nPYc.utilities._nmr import qcCheckBaseline

from generateTestDataset import generateTestDataset

class test_nmrdataset_synthetic(unittest.TestCase):

	def setUp(self):

		self.noSamp = numpy.random.randint(50, high=100, size=None)
		self.noFeat = numpy.random.randint(200, high=400, size=None)

		self.dataset = generateTestDataset(self.noSamp, self.noFeat, dtype='NMRDataset',
																	 variableType=nPYc.enumerations.VariableType.Spectral,
																	 sop='GenericNMRurine')


	def test_getsamplemetadatafromfilename(self):
		"""
		Test we are parsing NPC MS filenames correctly (PCSOP.081).
		"""

		# Create an empty object with simple filenames
		dataset = nPYc.NMRDataset('', fileType='empty')

		dataset.sampleMetadata['Sample File Name'] = ['Test1_serum_Rack1_SLT_090114/101',
													  'Test_serum_Rack10_SLR_090114/10',
													  'Test2_serum_Rack100_DLT_090114/102',
													  'Test2_urine_Rack103_MR_090114/20',
													  'Test2_serum_Rack010_JTP_090114/80',
													  'Test1_water_Rack10_TMP_090114/90']

		dataset._getSampleMetadataFromFilename(dataset.Attributes['filenameSpec'])

		rack = pandas.Series([1, 10, 100, 103, 10, 10],
							name='Rack',
							dtype=int)

		pandas.util.testing.assert_series_equal(dataset.sampleMetadata['Rack'], rack)

		study = pandas.Series(['Test1', 'Test', 'Test2', 'Test2', 'Test2', 'Test1'],
							name='Study',
							dtype=str)

		pandas.util.testing.assert_series_equal(dataset.sampleMetadata['Study'], study)


	def test_nmrdataset_raises(self):

		self.assertRaises(NotImplementedError, nPYc.NMRDataset, '', fileType='Unknown import type')
		self.assertRaises(TypeError, nPYc.NMRDataset, '', fileType='Bruker', bounds='not a list')
		self.assertRaises(TypeError, nPYc.NMRDataset, '', fileType='Bruker', calibrateTo='not a number')
		self.assertRaises(TypeError, nPYc.NMRDataset, '', fileType='Bruker', variableSize=0.1)


	def test_load_npc_lims_masking_reruns(self):

		limspath = os.path.join('..', '..', 'npc-standard-project', 'Derived_Worklists', 'UnitTest1_NMR_urine_PCSOP.011.csv')

		dataset = nPYc.NMRDataset('', 'empty')

		dataset.sampleMetadata = pandas.DataFrame([], columns=['Sample File Name'])

		dataset.sampleMetadata['Sample File Name'] = ['UnitTest1_Urine_Rack1_SLL_270814/10', 'UnitTest1_Urine_Rack1_SLL_270814/12', 'UnitTest1_Urine_Rack1_SLL_270814/20', 'UnitTest1_Urine_Rack1_SLL_270814/30', 'UnitTest1_Urine_Rack1_SLL_270814/40','UnitTest1_Urine_Rack1_SLL_270814/51', 'UnitTest1_Urine_Rack1_SLL_270814/52', 'UnitTest1_Urine_Rack1_SLL_270814/50', 'UnitTest1_Urine_Rack1_SLL_270814/60', 'UnitTest1_Urine_Rack1_SLL_270814/70', 'UnitTest1_Urine_Rack1_SLL_270814/80', 'UnitTest1_Urine_Rack1_SLL_270814/81', 'UnitTest1_Urine_Rack1_SLL_270814/90']
		dataset.intensityData = numpy.zeros((13, 2))
		dataset.intensityData[:, 0] = numpy.arange(1, 14, 1)
		dataset.initialiseMasks()

		with warnings.catch_warnings(record=True) as w:
			# Cause all warnings to always be triggered.
			warnings.simplefilter("always")
			# warning
			dataset.addSampleInfo(descriptionFormat='NPC LIMS', filePath=limspath)
			# check
			assert issubclass(w[0].category, UserWarning)
			assert "previous acquisitions masked, latest is kept" in str(w[0].message)


		with self.subTest(msg='Masking of reruns'):
			expectedMask = numpy.array([False, True, True, True, True, False, True, False, True, True, False, True,  True], dtype=bool)

			numpy.testing.assert_array_equal(dataset.sampleMask, expectedMask)


	def test_updateMasks_samples(self):

		from nPYc.enumerations import VariableType, DatasetLevel, AssayRole, SampleType

		dataset = generateTestDataset(18, 5, dtype='NMRDataset',
											  variableType=nPYc.enumerations.VariableType.Spectral,
											  sop='GenericNMRurine')

		dataset.Attributes.pop('PWFailThreshold', None)
		dataset.Attributes.pop('baselineCheckRegion', None)
		dataset.Attributes.pop('waterPeakCheckRegion', None)

		dataset.sampleMetadata['AssayRole'] = pandas.Series([AssayRole.LinearityReference,
								AssayRole.LinearityReference,
								AssayRole.LinearityReference,
								AssayRole.LinearityReference,
								AssayRole.LinearityReference,
								AssayRole.PrecisionReference,
								AssayRole.PrecisionReference,
								AssayRole.PrecisionReference,
								AssayRole.PrecisionReference,
								AssayRole.PrecisionReference,
								AssayRole.PrecisionReference,
								AssayRole.Assay,
								AssayRole.Assay,
								AssayRole.Assay,
								AssayRole.Assay,
								AssayRole.Assay,
								AssayRole.PrecisionReference,
								AssayRole.PrecisionReference],
								name='AssayRole',
								dtype=object)

		dataset.sampleMetadata['SampleType'] = pandas.Series([SampleType.StudyPool,
								SampleType.StudyPool,
								SampleType.StudyPool,
								SampleType.StudyPool,
								SampleType.StudyPool,
								SampleType.StudyPool,
								SampleType.StudyPool,
								SampleType.StudyPool,
								SampleType.StudyPool,
								SampleType.StudyPool,
								SampleType.StudyPool,
								SampleType.StudySample,
								SampleType.StudySample,
								SampleType.StudySample,
								SampleType.StudySample,
								SampleType.StudySample,
								SampleType.ExternalReference,
								SampleType.MethodReference],
								name='SampleType',
								dtype=object)

		with self.subTest(msg='Default Parameters'):
			expectedSampleMask = numpy.array([False, False, False, False, False,  True,  True,  True,  True, True,  True,  True,  True,  True,  True,  True, False, False], dtype=bool)

			dataset.initialiseMasks()
			dataset.updateMasks(filterFeatures=False)

			numpy.testing.assert_array_equal(expectedSampleMask, dataset.sampleMask)

		with self.subTest(msg='Export SP and ER'):
			expectedSampleMask = numpy.array([False, False, False, False, False,  True,  True,  True,  True, True,  True, False, False, False, False, False,  True, False], dtype=bool)

			dataset.initialiseMasks()
			dataset.updateMasks(filterFeatures=False,
								sampleTypes=[SampleType.StudyPool, SampleType.ExternalReference], 
								assayRoles=[AssayRole.PrecisionReference])

			numpy.testing.assert_array_equal(expectedSampleMask, dataset.sampleMask)

		with self.subTest(msg='Export Dilution Samples only'):
			expectedSampleMask = numpy.array([True,  True,  True,  True,  True, False, False, False, False, False, False, False, False, False, False, False, False, False], dtype=bool)

			dataset.initialiseMasks()
			dataset.updateMasks(filterFeatures=False,
								sampleTypes=[SampleType.StudyPool], 
								assayRoles=[AssayRole.LinearityReference])

			numpy.testing.assert_array_equal(expectedSampleMask, dataset.sampleMask)


	def test_updateMasks_features(self):

		noSamp = 10
		noFeat = numpy.random.randint(1000, high=10000, size=None)

		dataset = generateTestDataset(noSamp, noFeat, dtype='NMRDataset',
													   variableType=nPYc.enumerations.VariableType.Spectral,
													   sop='GenericNMRurine')

		dataset.Attributes.pop('PWFailThreshold', None)
		dataset.Attributes.pop('baselineCheckRegion', None)
		dataset.Attributes.pop('waterPeakCheckRegion', None)

		ppm = numpy.linspace(-10, 10, noFeat)
		dataset.featureMetadata = pandas.DataFrame(ppm, columns=['ppm'])

		with self.subTest(msg='Single range'):
			ranges = (-1.1, 1.2)

			dataset.initialiseMasks()
			dataset.updateMasks(filterFeatures=True,
								filterSamples=False,
								exclusionRegions=ranges)

			expectedFeatureMask = numpy.logical_or(ppm < ranges[0],
												   ppm > ranges[1])

			numpy.testing.assert_array_equal(expectedFeatureMask, dataset.featureMask)

		with self.subTest(msg='Reversed range'):
			ranges = (7.1, 1.92)

			dataset.initialiseMasks()
			dataset.updateMasks(filterFeatures=True,
								filterSamples=False,
								exclusionRegions=ranges)

			expectedFeatureMask = numpy.logical_or(ppm < ranges[1],
													ppm > ranges[0])

			numpy.testing.assert_array_equal(expectedFeatureMask, dataset.featureMask)

		with self.subTest(msg='list of ranges'):
			ranges = [(-5,-1), (1,5)]

			dataset.initialiseMasks()
			dataset.updateMasks(filterFeatures=True,
								filterSamples=False,
								exclusionRegions=ranges)

			expectedFeatureMask1 = numpy.logical_or(ppm < ranges[0][0],
													 ppm > ranges[0][1])
			expectedFeatureMask2 = numpy.logical_or(ppm < ranges[1][0],
													 ppm > ranges[1][1])
			expectedFeatureMask = numpy.logical_and(expectedFeatureMask1,
													expectedFeatureMask2)

			numpy.testing.assert_array_equal(expectedFeatureMask, dataset.featureMask)


	def test_updateMasks_raises(self):

		with self.subTest(msg='No Ranges'):
			self.dataset.Attributes['exclusionRegions'] = None
			self.assertRaises(ValueError, self.dataset.updateMasks, filterFeatures=True, filterSamples=False, exclusionRegions=None)


	def test_updateMasks_warns(self):

		with self.subTest(msg='Range low == high'):
			self.dataset.Attributes['exclusionRegions'] = None
			self.assertWarnsRegex(UserWarning, 'Low \(1\.10\) and high \(1\.10\) bounds are identical, skipping region', self.dataset.updateMasks, filterFeatures=True, filterSamples=False, exclusionRegions=(1.1,1.1))


	def test_nmrQCchecks(self):

		self.dataset.Attributes.pop('PWFailThreshold', None)
		self.dataset.Attributes.pop('baselineCheckRegion', None)
		self.dataset.Attributes.pop('waterPeakCheckRegion', None)

		with self.subTest('Calibration'):
			bounds = numpy.std(self.dataset.sampleMetadata['Delta PPM']) * 3
			self.dataset.sampleMetadata.loc[0::30, 'Delta PPM'] = bounds * 15

			self.dataset._nmrQCChecks()

			# Check mask
			expected = numpy.zeros_like(self.dataset.sampleMask, dtype=bool)
			expected[0::30] = True
			numpy.testing.assert_array_equal(expected, self.dataset.sampleMetadata['CalibrationFail'].values)

			# Check other tests have not happened
			#for skipedCheck in ['LineWidthFail', 'BaselineFail', 'WaterPeakFail']:
			#	self.assertFalse(skipedCheck in self.dataset.sampleMetadata.columns)

		with self.subTest('Line Width'):
			self.dataset.Attributes['PWFailThreshold'] = 2

			self.dataset.sampleMetadata['Line Width (Hz)'] = 1.5
			self.dataset.sampleMetadata.loc[0::5, 'Line Width (Hz)'] = 3

			self.dataset._nmrQCChecks()

			expected = numpy.zeros_like(self.dataset.sampleMask, dtype=bool)
			expected[0::5] = True

			numpy.testing.assert_array_equal(expected, self.dataset.sampleMetadata['LineWidthFail'].values)

			# Check other tests have not happend
			#for skipedCheck in ['BaselineFail', 'WaterPeakFail']:
		#		self.assertFalse(skipedCheck in self.dataset.sampleMetadata.columns)

		with self.subTest('Baseline'):
			self.dataset.Attributes['baselineCheckRegion'] = [(-2, -0.5), (9.5, 12.5)]

			self.dataset.intensityData[0,:] = 100
			self.dataset.intensityData[2,:] = -100

			self.dataset._nmrQCChecks()

			expected = numpy.zeros_like(self.dataset.sampleMask, dtype=bool)
			expected[0] = True
			expected[2] = True

			numpy.testing.assert_array_equal(expected, self.dataset.sampleMetadata['BaselineFail'].values)

			# Check other tests have not happend
			#self.assertFalse('WaterPeakFail' in self.dataset.sampleMetadata.columns)

		with self.subTest('Water Peak'):
			self.dataset.Attributes['waterPeakCheckRegion'] = [(-2, -0.5), (9.5, 12.5)]

			self.dataset._nmrQCChecks()

			expected = numpy.zeros_like(self.dataset.sampleMask, dtype=bool)
			expected[0] = True
			# expected[2] = True

			numpy.testing.assert_array_equal(expected, self.dataset.sampleMetadata['WaterPeakFail'].values)


	def test_baselineAreaAndNeg(self):
		"""
		Validate baseline/WP code, creates random spectra and values that should always fail ie <0 and high extreme and diagonal.
		"""

		variableSize = 20000

		X = numpy.random.rand(86, variableSize)*1000

		X = numpy.r_[X, numpy.full((1, variableSize), -10000)] # add a minus  val row r_ shortcut notation for vstack
		X = numpy.r_[X, numpy.full((1, variableSize), 200000)] # add a minus  val row r_ shortcut notation for vstack

		a1 = numpy.arange(0,variableSize,1)[numpy.newaxis] #diagonal ie another known fail

		X = numpy.concatenate((X, a1), axis=0)#concatenate into X

		X = numpy.r_[X, numpy.random.rand(2, variableSize)* 10000]
		#add more fails random but more variablility than the average 86 above

		#create ppm
		ppm = numpy.linspace(-1,10, variableSize) #
		ppm_high = numpy.where(ppm >= 9.5)[0]
		ppm_low = numpy.where(ppm <= -0.5)[0]

		high_baseline = qcCheckBaseline(X[:, ppm_high], 0.05)
		low_baseline = qcCheckBaseline(X[:, ppm_low], 0.05)

		baseline_fail_calculated = high_baseline | low_baseline
		baseline_fail_expected = numpy.zeros(91, dtype=bool)
		baseline_fail_expected[86:89] = True

		numpy.testing.assert_array_equal(baseline_fail_expected, baseline_fail_calculated)


##
#unit test for Bruker data
##
from nPYc.utilities._nmr import interpolateSpectrum
from nPYc.reports._generateReportNMR import _generateReportNMR
from math import ceil

class test_nmrdataset_bruker(unittest.TestCase):

	def setUp(self):
		"""
		setup the pulseprogram and path for purpose of testing NMR bruker data functions
		"""
		self.pulseProgram = 'noesygppr1d'
		self.path=os.path.join('..','..','npc-standard-project','unitTest_Data','nmr')#where path is location of test files


	def test_addSampleInfo_npclims(self):

		with self.subTest(msg='Urine dataset (UnitTest1).'):
			dataPath = os.path.join('..', '..', 'npc-standard-project', 'Raw_Data', 'nmr', 'UnitTest1')
			limsFilePath = os.path.join('..', '..', 'npc-standard-project', 'Derived_Worklists', 'UnitTest1_NMR_urine_PCSOP.011.csv')

			with warnings.catch_warnings():
				warnings.simplefilter("ignore")
				dataset = nPYc.NMRDataset(dataPath, pulseProgram='noesygppr1d', sop='GenericNMRurine')

			dataset.sampleMetadata.sort_values('Sample File Name', inplace=True)

			sortIndex = dataset.sampleMetadata.index.values
			dataset.intensityData = dataset.intensityData[sortIndex, :]

			dataset.sampleMetadata = dataset.sampleMetadata.reset_index(drop=True)

			expected = copy.deepcopy(dataset.sampleMetadata)

			dataset.addSampleInfo(descriptionFormat='NPC LIMS', filePath=limsFilePath)

			testSeries = ['Sampling ID', 'Status', 'AssayRole', 'SampleType']

			expected['Sampling ID'] = ['UT1_S2_u1', 'UT1_S3_u1', 'UT1_S4_u1', 'UT1_S4_u2', 'UT1_S4_u3',
									   'UT1_S4_u4', 'External Reference Sample', 'Study Pool Sample']

			expected['Status'] = ['Sample', 'Sample', 'Sample', 'Sample', 'Sample', 'Sample', 'Long Term Reference', 'Study Reference']

			expected['AssayRole'] = [AssayRole.Assay, AssayRole.Assay, AssayRole.Assay, AssayRole.Assay,
									 AssayRole.Assay, AssayRole.Assay, AssayRole.PrecisionReference, AssayRole.PrecisionReference]

			expected['SampleType'] = [SampleType.StudySample, SampleType.StudySample, SampleType.StudySample, SampleType.StudySample,
									  SampleType.StudySample, SampleType.StudySample, SampleType.ExternalReference, SampleType.StudyPool]

			for series in testSeries:
				with self.subTest(msg='Testing %s' % series):
					pandas.util.testing.assert_series_equal(dataset.sampleMetadata[series], expected[series])

		with self.subTest(msg='Serum dataset (UnitTest3).'):
			dataPath = os.path.join('..', '..', 'npc-standard-project', 'Raw_Data', 'nmr', 'UnitTest3')
			limsFilePath = os.path.join('..', '..', 'npc-standard-project', 'Derived_Worklists', 'UnitTest3_NMR_serum_PCSOP.012.csv')

			with warnings.catch_warnings():
				warnings.simplefilter("ignore")
				dataset = nPYc.NMRDataset(dataPath, pulseProgram='cpmgpr1d', sop='GenericNMRurine') # Use blood sop to avoid calibration  of empty spectra

			dataset.sampleMetadata.sort_values('Sample File Name', inplace=True)

			sortIndex = dataset.sampleMetadata.index.values
			dataset.intensityData = dataset.intensityData[sortIndex, :]

			dataset.sampleMetadata = dataset.sampleMetadata.reset_index(drop=True)

			expected = copy.deepcopy(dataset.sampleMetadata)

			dataset.addSampleInfo(descriptionFormat='NPC LIMS', filePath=limsFilePath)

			testSeries = ['Sampling ID', 'Status', 'AssayRole', 'SampleType']

			expected['Sampling ID'] = ['UT3_S7', 'UT3_S8', 'UT3_S6', 'UT3_S5', 'UT3_S4', 'UT3_S3', 'UT3_S2', 'External Reference Sample', 'Study Pool Sample', 'UT3_S1']

			expected['Status'] = ['Sample', 'Sample', 'Sample', 'Sample', 'Sample', 'Sample', 'Sample', 'Long Term Reference', 'Study Reference', 'nan']

			expected['AssayRole'] = [AssayRole.Assay, AssayRole.Assay, AssayRole.Assay, AssayRole.Assay, AssayRole.Assay, AssayRole.Assay,
									 AssayRole.Assay, AssayRole.PrecisionReference, AssayRole.PrecisionReference, AssayRole.Assay]

			expected['SampleType'] = [SampleType.StudySample, SampleType.StudySample, SampleType.StudySample, SampleType.StudySample, SampleType.StudySample,
									  SampleType.StudySample, SampleType.StudySample, SampleType.ExternalReference, SampleType.StudyPool, SampleType.StudySample]

			for series in testSeries:
				with self.subTest(msg='Testing %s' % series):
					pandas.util.testing.assert_series_equal(dataset.sampleMetadata[series], expected[series])


class test_nmrdataset_ISATAB(unittest.TestCase):

	def test_exportISATAB(self):

		nmrData = nPYc.NMRDataset('', fileType='empty')
		raw_data = {
			'Acquired Time': ['09/08/2016  01:36:23', '09/08/2016  01:56:23', '09/08/2016  02:16:23',
							  '09/08/2016  02:36:23', '09/08/2016  02:56:23'],
			'AssayRole': ['AssayRole.LinearityReference', 'AssayRole.LinearityReference',
						  'AssayRole.LinearityReference', 'AssayRole.Assay', 'AssayRole.Assay'],
			'SampleType': ['SampleType.StudyPool', 'SampleType.StudyPool', 'SampleType.StudyPool',
						   'SampleType.StudySample', 'SampleType.StudySample'],
			'Subject ID': ['', '', '', 'SCANS-120', 'SCANS-130'],
			'Sampling ID': ['', '', '', 'T0-7-S', 'T0-9-S'],
			'Study': ['TestStudy', 'TestStudy', 'TestStudy', 'TestStudy', 'TestStudy'],
			'Gender': ['', '', '', 'Female', 'Male'],
			'Age': ['', '', '', '55', '66'],
			'Sampling Date': ['', '', '', '27/02/2006', '28/02/2006'],
			'Sample batch': ['', '', '', 'SB 1', 'SB 2'],
			'Acquisition batch': ['1', '2', '3', '4', '5'],
			'Run Order': ['0', '1', '2', '3', '4'],
			'Instrument': ['QTOF 2', 'QTOF 2', 'QTOF 2', 'QTOF 2', 'QTOF 2'],
			'Assay data name': ['', '', '', 'SS_LNEG_ToF02_S1W4', 'SS_LNEG_ToF02_S1W5']
		}
		nmrData.sampleMetadata = pandas.DataFrame(raw_data,
												  columns=['Acquired Time', 'AssayRole', 'SampleType', 'Subject ID',
														   'Sampling ID', 'Study', 'Gender', 'Age', 'Sampling Date',
														   'Sample batch', 'Acquisition batch',
														   'Run Order', 'Instrument', 'Assay data name'])

		with tempfile.TemporaryDirectory() as tmpdirname:
			nmrData.exportDataset(destinationPath=tmpdirname, saveFormat='ISATAB', withExclusions=False)
			a = os.path.join(tmpdirname, 'NMRDataset', 'a_npc-test-study_metabolite_profiling_NMR_spectroscopy.txt')
			self.assertTrue(os.path.exists(a))


if __name__ == '__main__':
	unittest.main()
