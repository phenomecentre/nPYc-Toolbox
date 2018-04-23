# -*- coding: utf-8 -*-
import pandas
import numpy
import sys
import unittest
from datetime import datetime
from pandas.util.testing import assert_frame_equal
import os
import copy
sys.path.append("..")
import warnings
import nPYc
from nPYc.enumerations import SampleType
from nPYc.enumerations import AssayRole
from nPYc.enumerations import VariableType
from generateTestDataset import generateTestDataset


class test_msdataset_synthetic(unittest.TestCase):
	"""
	Test MSDataset object functions with synthetic data
	"""

	def setUp(self):
		self.msData = nPYc.MSDataset('', fileType='empty')
		self.msData.sampleMetadata = pandas.DataFrame(
			{'Sample File Name': ['Unittest_file_001', 'Unittest_file_002', 'Unittest_file_003'],
			 'Sample Base Name': ['Unittest_file_001', 'Unittest_file_002', 'Unittest_file_003'],
			 'AssayRole': [AssayRole.Assay, AssayRole.PrecisionReference, AssayRole.PrecisionReference],
			 'SampleType': [SampleType.StudySample, SampleType.StudyPool, SampleType.ExternalReference],
			 'Sample Name': ['Sample1', 'Sample2', 'Sample3'], 'Acqu Date': ['26-May-17', '26-May-17', '26-May-17'],
			 'Acqu Time': ['16:42:57', '16:58:49', '17:14:41'], 'Vial': ['1:A,1', '1:A,2', '1:A,3'],
			 'Instrument': ['XEVO-TOF#UnitTest', 'XEVO-TOF#UnitTest', 'XEVO-TOF#UnitTest'],
			 'Acquired Time': [datetime(2017, 5, 26, 16, 42, 57), datetime(2017, 5, 26, 16, 58, 49),
							   datetime(2017, 5, 26, 17, 14, 41)], 'Run Order': [0, 1, 2], 'Batch': [1, 1, 2],
			 'Correction Batch': [numpy.nan, 1, 2], 'Matrix': ['U', 'U', 'U'],
			 'Subject ID': ['subject1', 'subject1', 'subject2'], 'Sampling ID': ['sample1', 'sample2', 'sample3'],
			 'Dilution': [numpy.nan, '60.0', '100.0'],'Exclusion Details': ['','','']})
		self.msData.featureMetadata = pandas.DataFrame(
			{'Feature Name': ['Feature1', 'Feature2', 'Feature3'], 'Retention Time': [6.2449, 2.7565, 5.0564],
			 'm/z': [249.124281, 381.433191, 471.132083]})
		self.msData._intensityData = numpy.array([[10.2, 20.95, 30.37], [10.1, 20.03, 30.74], [3.065, 15.83, 30.16]])
		# Attributes
		self.msData.Attributes['FeatureExtractionSoftware'] = 'UnitTestSoftware'
		# excluded data
		self.msData.sampleMetadataExcluded = []
		self.msData.intensityDataExcluded = []
		self.msData.featureMetadataExcluded = []
		self.msData.excludedFlag = []
		self.msData.sampleMetadataExcluded.append(self.msData.sampleMetadata[[True, False, False]])
		self.msData.intensityDataExcluded.append(self.msData._intensityData[0, :])
		self.msData.featureMetadataExcluded.append(self.msData.featureMetadata)
		self.msData.excludedFlag.append('Samples')
		self.msData.featureMetadataExcluded.append(self.msData.featureMetadata[[True, False, False]])
		self.msData.intensityDataExcluded.append(self.msData._intensityData[:, 0])
		self.msData.sampleMetadataExcluded.append(self.msData.sampleMetadata)
		self.msData.excludedFlag.append('Features')
		# finish
		self.msData.VariableType = VariableType.Discrete
		self.msData.initialiseMasks()


	def test_rsd_raises(self):

		msData = nPYc.MSDataset('', fileType='empty')

		with self.subTest(msg='No reference samples'):
			msData.sampleMetadata = pandas.DataFrame(None)

			with self.assertRaises(ValueError):
				msData.rsdSP

		with self.subTest(msg='Only one reference sample'):
			msData.sampleMetadata = pandas.DataFrame([[nPYc.enumerations.AssayRole.PrecisionReference, nPYc.enumerations.SampleType.StudyPool]], columns=['AssayRole', 'SampleType'])

			with self.assertRaises(ValueError):
				msData.rsdSP


	def test_getsamplemetadatafromfilename(self):
		"""
		Test we are parsing NPC MS filenames correctly (PCSOP.081).
		"""

		# Create an empty object with simple filenames
		msData = nPYc.MSDataset('', fileType='empty')

		msData.sampleMetadata['Sample File Name'] = ['Test1_HPOS_ToF01_P1W02',
													 'Test2_RPOS_ToF02_U2W03',
													 'Test3_RNEG_ToF03_S3W04',
													 'Test4_LPOS_ToF04_P4W05_LTR',
													 'Test5_LNEG_ToF05_U5W06_SR',
													 'Test6_HPOS_ToF06_S4W05_MR',
													 'Test1_HPOS_ToF01_P1W02_x',
													 'Test2_RPOS_ToF02_U2W03_b',
													 'Test3_RNEG_ToF03_S3W04_2',
													 'Test4_RPOS_ToF04_B1S1_SR_q',
													 'Test5_LPOS_ToF05_B2E2_SR',
													 'Test6_LNEG_ToF06_B3SRD01_9',
													 'Test1_HPOS_ToF06_Blank01',
													 'Test1_HPOS_ToF06_IC02',
													 'Test1_HPOS_ToF06_EIC21']

		msData._getSampleMetadataFromFilename(msData.Attributes['filenameSpec'])

		##
		# Check basename
		##
		basename = pandas.Series(['Test1_HPOS_ToF01_P1W02',
								  'Test2_RPOS_ToF02_U2W03',
								  'Test3_RNEG_ToF03_S3W04',
								  'Test4_LPOS_ToF04_P4W05_LTR',
								  'Test5_LNEG_ToF05_U5W06_SR',
								  'Test6_HPOS_ToF06_S4W05_MR',
								  'Test1_HPOS_ToF01_P1W02',
								  'Test2_RPOS_ToF02_U2W03',
								  'Test3_RNEG_ToF03_S3W04',
								  'Test4_RPOS_ToF04_B1S1_SR',
								  'Test5_LPOS_ToF05_B2E2_SR',
								  'Test6_LNEG_ToF06_B3SRD01',
								  'Test1_HPOS_ToF06_Blank01',
								  'Test1_HPOS_ToF06_IC02',
								  'Test1_HPOS_ToF06_EIC21'],
								  name='Sample Base Name',
								  dtype='str')

		pandas.util.testing.assert_series_equal(msData.sampleMetadata['Sample Base Name'], basename)

		##
		# Check Study
		##
		study = pandas.Series(['Test1',
							   'Test2',
							   'Test3',
							   'Test4',
							   'Test5',
							   'Test6',
							   'Test1',
							   'Test2',
							   'Test3',
							   'Test4',
							   'Test5',
							   'Test6',
							   'Test1',
							   'Test1',
							   'Test1'],
							   name='Study',
							   dtype='str')

		pandas.util.testing.assert_series_equal(msData.sampleMetadata['Study'], study)

		##
		# 
		##
		chromatography = pandas.Series(['H',
										'R',
										'R',
										'L',
										'L',
										'H',
										'H',
										'R',
										'R',
										'R',
										'L',
										'L',
										'H',
										'H',
										'H'],
										name='Chromatography',
										dtype='str')

		pandas.util.testing.assert_series_equal(msData.sampleMetadata['Chromatography'], chromatography)

		##
		#
		##
		ionisation = pandas.Series(['POS',
									'POS',
									'NEG',
									'POS',
									'NEG',
									'POS',
									'POS',
									'POS',
									'NEG',
									'POS',
									'POS',
									'NEG',
									'POS',
									'POS',
									'POS'],
									name='Ionisation',
									dtype='str')

		pandas.util.testing.assert_series_equal(msData.sampleMetadata['Ionisation'], ionisation)

		##
		#
		##
		instrument = pandas.Series(['ToF01',
									'ToF02',
									'ToF03',
									'ToF04',
									'ToF05',
									'ToF06',
									'ToF01',
									'ToF02',
									'ToF03',
									'ToF04',
									'ToF05',
									'ToF06',
									'ToF06',
									'ToF06',
									'ToF06'],
									name='Instrument',
									dtype='str')

		pandas.util.testing.assert_series_equal(msData.sampleMetadata['Instrument'], instrument)

		##
		#
		##
		reRun = pandas.Series(['',
							'',
							'',
							'',
							'',
							'',
							'',
							'b',
							'',
							'q',
							'',
							'',
							'',
							'',
							''],
							name='Re-Run',
							dtype='str')

		pandas.util.testing.assert_series_equal(msData.sampleMetadata['Re-Run'], reRun)

		##
		#
		##
		suplemental = pandas.Series(['',
									'',
									'',
									'',
									'',
									'',
									'',
									'',
									'2',
									'',
									'',
									'9',
									'',
									'',
									''],
									name='Suplemental Injections',
									dtype='str')

		pandas.util.testing.assert_series_equal(msData.sampleMetadata['Suplemental Injections'], suplemental)

		##
		#
		##
		skipped = pandas.Series([False,
								False,
								False,
								False,
								False,
								False,
								True,
								False,
								False,
								False,
								False,
								False,
								False,
								False,
								False],
								name='Skipped',
								dtype='bool')

		pandas.util.testing.assert_series_equal(msData.sampleMetadata['Skipped'], skipped)

		##
		#
		##
		matrix = pandas.Series(['P',
								'U',
								'S',
								'P',
								'U',
								'S',
								'P',
								'U',
								'S',
								'',
								'',
								'',
								'',
								'',
								''],
								name='Matrix',
								dtype='str')

		pandas.util.testing.assert_series_equal(msData.sampleMetadata['Matrix'], matrix)

		##
		#
		##
		well = pandas.Series([2,
							3,
							4,
							5,
							6,
							5,
							2,
							3,
							4,
							1,
							2,
							1,
							-1,
							-1,
							-1],
							name='Well',
							dtype='int')

		pandas.util.testing.assert_series_equal(msData.sampleMetadata['Well'], well, check_dtype=False)
		self.assertEqual(msData.sampleMetadata['Well'].dtype.kind, well.dtype.kind)

		##
		#
		##
		plate = pandas.Series([1,
								2,
								3,
								4,
								5,
								4,
								1,
								2,
								3,
								1,
								2,
								3,
								1,
								2,
								21],
								name='Plate',
								dtype='int')

		pandas.util.testing.assert_series_equal(msData.sampleMetadata['Plate'], plate, check_dtype=False)
		self.assertEqual(msData.sampleMetadata['Plate'].dtype.kind, well.dtype.kind)

		##
		#
		##
		batch = pandas.Series([numpy.nan,
							numpy.nan,
							numpy.nan,
							numpy.nan,
							numpy.nan,
							numpy.nan,
							numpy.nan,
							numpy.nan,
							numpy.nan,
							1.0,
							2.0,
							3.0,
							numpy.nan,
							numpy.nan,
							numpy.nan],
							name='Batch',
							dtype='float')

		pandas.util.testing.assert_series_equal(msData.sampleMetadata['Batch'], batch)

		##
		#
		##
		dilution = pandas.Series([numpy.nan,
								numpy.nan,
								numpy.nan,
								numpy.nan,
								numpy.nan,
								numpy.nan,
								numpy.nan,
								numpy.nan,
								numpy.nan,
								numpy.nan,
								numpy.nan,
								1.0,
								numpy.nan,
								numpy.nan,
								numpy.nan],
								name='Dilution',
								dtype='float')

		pandas.util.testing.assert_series_equal(msData.sampleMetadata['Dilution'], dilution)

		##
		#
		##
		assayRole = pandas.Series([AssayRole.Assay,
								AssayRole.Assay,
								AssayRole.Assay,
								AssayRole.PrecisionReference,
								AssayRole.PrecisionReference,
								AssayRole.PrecisionReference,
								AssayRole.Assay,
								AssayRole.Assay,
								AssayRole.Assay,
								AssayRole.PrecisionReference,
								AssayRole.PrecisionReference,
								AssayRole.LinearityReference,
								AssayRole.LinearityReference,
								AssayRole.Assay,
								AssayRole.Assay],
								name='AssayRole',
								dtype=object)

		pandas.util.testing.assert_series_equal(msData.sampleMetadata['AssayRole'], assayRole)

		##
		#
		##
		sampleType = pandas.Series([SampleType.StudySample,
								SampleType.StudySample,
								SampleType.StudySample,
								SampleType.ExternalReference,
								SampleType.StudyPool,
								SampleType.MethodReference,
								SampleType.StudySample,
								SampleType.StudySample,
								SampleType.StudySample,
								SampleType.StudyPool,
								SampleType.StudyPool,
								SampleType.StudyPool,
								SampleType.ProceduralBlank,
								SampleType.StudyPool,
								SampleType.StudyPool],
								name='SampleType',
								dtype=object)

		pandas.util.testing.assert_series_equal(msData.sampleMetadata['SampleType'], sampleType)


	def test_updateMasks_features(self):

		msData = nPYc.MSDataset('', fileType='empty')
		msData.Attributes['artifactualFilter'] = True

		##
		# Variables:
		# Good Corr, Good RSD
		# Poor Corr, Good RSD
		# Good Corr, Poor RSD
		# Poor Corr, Poor RSD
		# Good Corr, Good RSD, below blank
		##
		msData.intensityData = numpy.array([[100, 23, 99, 51, 100],
											[90, 54, 91, 88, 91],
											[50, 34, 48, 77, 49],
											[10, 66, 11, 56, 11],
											[1, 12, 2, 81, 2],
											[50, 51, 2, 12, 49],
											[51, 47, 1, 100, 50],
											[47, 50, 70, 21, 48],
											[51, 49, 77, 91, 50],
											[48, 49, 12, 2, 49],
											[50, 48, 81, 2, 51],
											[54, 53, 121, 52, 53],
											[57, 49, 15, 51, 56],
											[140, 41, 97, 47, 137],
											[52, 60, 42, 60, 48],
											[12, 48, 8, 56, 12],
											[1, 2, 1, 1.21, 51],
											[2, 1, 1.3, 1.3, 63]],
											dtype=float)

		msData.sampleMetadata = pandas.DataFrame(data=[[100, 1, 1, 1, AssayRole.LinearityReference, SampleType.StudyPool],
														[90, 1, 1, 2, AssayRole.LinearityReference, SampleType.StudyPool],
														[50, 1, 1, 3, AssayRole.LinearityReference, SampleType.StudyPool],
														[10, 1, 1, 4, AssayRole.LinearityReference, SampleType.StudyPool],
														[1, 1, 1, 5, AssayRole.LinearityReference, SampleType.StudyPool],
														[numpy.nan, 1, 1, 1, AssayRole.PrecisionReference, SampleType.StudyPool],
														[numpy.nan, 1, 1, 1, AssayRole.PrecisionReference, SampleType.StudyPool],
														[numpy.nan, 1, 1, 1, AssayRole.PrecisionReference, SampleType.StudyPool],
														[numpy.nan, 1, 1, 1, AssayRole.PrecisionReference, SampleType.StudyPool],
														[numpy.nan, 1, 1, 1, AssayRole.PrecisionReference, SampleType.StudyPool],
														[numpy.nan, 1, 1, 1, AssayRole.PrecisionReference, SampleType.StudyPool],
														[numpy.nan, 1, 1, 1, AssayRole.Assay, SampleType.StudySample],
														[numpy.nan, 1, 1, 1, AssayRole.Assay, SampleType.StudySample],
														[numpy.nan, 1, 1, 1, AssayRole.Assay, SampleType.StudySample],
														[numpy.nan, 1, 1, 1, AssayRole.Assay, SampleType.StudySample],
														[numpy.nan, 1, 1, 1, AssayRole.Assay, SampleType.StudySample],
														[0, 1, 1, 1, AssayRole.Assay, SampleType.ProceduralBlank],
														[0, 1, 1, 1, AssayRole.Assay, SampleType.ProceduralBlank]],
														columns=['Dilution', 'Batch', 'Correction Batch', 'Well', 'AssayRole', 'SampleType'])

		msData.featureMetadata = pandas.DataFrame(data=[['Feature_1', 0.5, 100., 0.3],
														['Feature_2', 0.55, 100.04, 0.3],
														['Feature_3', 0.75, 200., 0.1],
														['Feature_4', 0.9, 300., 0.1],
														['Feature_5', 0.95, 300.08, 0.1]],
												  		columns=['Feature Name','Retention Time','m/z','Peak Width'])

		msData.initialiseMasks()

		with self.subTest(msg='Default Parameters'):
			expectedFeatureMask = numpy.array([True, False, False, False, False], dtype=bool)

			msData.updateMasks(withArtifactualFiltering=False)

			numpy.testing.assert_array_equal(expectedFeatureMask, msData.featureMask)

		with self.subTest(msg='Lax RSD threshold'):
			expectedFeatureMask = numpy.array([True, False, True, False, False], dtype=bool)

			msData.initialiseMasks()
			msData.updateMasks(withArtifactualFiltering=False, rsdThreshold=90, varianceRatio=0.1, correlationThreshold=0.7)

			numpy.testing.assert_array_equal(expectedFeatureMask, msData.featureMask)

		with self.subTest(msg='Lax correlation threshold'):
			expectedFeatureMask = numpy.array([True, True, False, False, False], dtype=bool)

			msData.initialiseMasks()
			msData.updateMasks(withArtifactualFiltering=False, rsdThreshold=30, varianceRatio=1.1, correlationThreshold=0)

			numpy.testing.assert_array_equal(expectedFeatureMask, msData.featureMask)

		with self.subTest(msg='High variance ratio'):
			expectedFeatureMask = numpy.array([False, False, False, False, False], dtype=bool)

			msData.initialiseMasks()
			msData.updateMasks(withArtifactualFiltering=False, rsdThreshold=30, varianceRatio=100, correlationThreshold=0.7)

			numpy.testing.assert_array_equal(expectedFeatureMask, msData.featureMask)

		with self.subTest(msg='Lax blank filter'):
			expectedFeatureMask = numpy.array([True, False, False, False, True], dtype=bool)

			msData.initialiseMasks()
			msData.updateMasks(withArtifactualFiltering=False, blankThreshold=0.5)

			numpy.testing.assert_array_equal(expectedFeatureMask, msData.featureMask)

		with self.subTest(msg='No blank filter'):
			expectedFeatureMask = numpy.array([True, False, False, False, True], dtype=bool)

			msData.initialiseMasks()
			msData.updateMasks(withArtifactualFiltering=False, blankThreshold=False)

			numpy.testing.assert_array_equal(expectedFeatureMask, msData.featureMask)

		with self.subTest(msg='Default withArtifactualFiltering'):
			expectedTempArtifactualLinkageMatrix = pandas.DataFrame(data=[[0,1],[3,4]],columns=['node1','node2'])

			msData.initialiseMasks()
			msData.updateMasks(withArtifactualFiltering=True, blankThreshold=False)

			pandas.util.testing.assert_frame_equal(expectedTempArtifactualLinkageMatrix, msData._tempArtifactualLinkageMatrix)

		with self.subTest(msg='Altered withArtifactualFiltering parameters'):
			expectedArtifactualLinkageMatrix = pandas.DataFrame(data=[[0,1]],columns=['node1','node2'])

			msData.updateMasks(withArtifactualFiltering=True, deltaMzArtifactual=300, overlapThresholdArtifactual=0.1, corrThresholdArtifactual=0.2, blankThreshold=False)

			self.assertEqual(msData.Attributes['deltaMzArtifactual'], 300)
			self.assertEqual(msData.Attributes['overlapThresholdArtifactual'], 0.1)
			self.assertEqual(msData.Attributes['corrThresholdArtifactual'], 0.2)
			pandas.util.testing.assert_frame_equal(expectedArtifactualLinkageMatrix, msData._artifactualLinkageMatrix)

		with self.subTest(msg='withArtifactualFiltering=None, Attribute[artifactualFilter]=False'):
			msData2 = copy.deepcopy(msData)
			msData2.Attributes['artifactualFilter'] = False
			expectedFeatureMask = numpy.array([True, False, False, False, False], dtype=bool)

			msData2.initialiseMasks()
			msData2.updateMasks(withArtifactualFiltering=None)

			numpy.testing.assert_array_equal(expectedFeatureMask, msData2.featureMask)

		with self.subTest(msg='withArtifactualFiltering=None, Attribute[artifactualFilter]=True'):
			msData2 = copy.deepcopy(msData)
			msData2.Attributes['artifactualFilter'] = True
			expectedTempArtifactualLinkageMatrix = pandas.DataFrame(data=[[0, 1], [3, 4]], columns=['node1', 'node2'])

			msData2.initialiseMasks()
			msData2.updateMasks(withArtifactualFiltering=None)

			pandas.util.testing.assert_frame_equal(expectedTempArtifactualLinkageMatrix, msData2._tempArtifactualLinkageMatrix)


	def test_updateMasks_samples(self):

		from nPYc.enumerations import VariableType, DatasetLevel, AssayRole, SampleType

		msData = nPYc.MSDataset('', fileType='empty')

		msData.intensityData = numpy.zeros([18, 5],dtype=float)
		
		
		msData.sampleMetadata['AssayRole'] = pandas.Series([AssayRole.LinearityReference,
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

		msData.sampleMetadata['SampleType'] = pandas.Series([SampleType.StudyPool,
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

			msData.initialiseMasks()
			msData.updateMasks(withArtifactualFiltering=False, filterFeatures=False)

			numpy.testing.assert_array_equal(expectedSampleMask, msData.sampleMask)

		with self.subTest(msg='Export SP and ER'):
			expectedSampleMask = numpy.array([False, False, False, False, False,  True,  True,  True,  True, True,  True, False, False, False, False, False,  True, False], dtype=bool)

			msData.initialiseMasks()
			msData.updateMasks(withArtifactualFiltering=False, filterFeatures=False,
								sampleTypes=[SampleType.StudyPool, SampleType.ExternalReference], 
								assayRoles=[AssayRole.PrecisionReference])

			numpy.testing.assert_array_equal(expectedSampleMask, msData.sampleMask)

		with self.subTest(msg='Export Dilution Samples only'):
			expectedSampleMask = numpy.array([True,  True,  True,  True,  True, False, False, False, False, False, False, False, False, False, False, False, False, False], dtype=bool)

			msData.initialiseMasks()
			msData.updateMasks(withArtifactualFiltering=False, filterFeatures=False,
								sampleTypes=[SampleType.StudyPool], 
								assayRoles=[AssayRole.LinearityReference])

			numpy.testing.assert_array_equal(expectedSampleMask, msData.sampleMask)


	def test_updateMasks_raises(self):

		msData = nPYc.MSDataset('', fileType='empty')

		with self.subTest(msg='Correlation'):
			self.assertRaises(ValueError, msData.updateMasks, correlationThreshold=-1.01)
			self.assertRaises(ValueError, msData.updateMasks, correlationThreshold=1.01)
			self.assertRaises(TypeError, msData.updateMasks, correlationThreshold='0.7')

		with self.subTest(msg='RSD'):
			self.assertRaises(ValueError, msData.updateMasks, rsdThreshold=-1.01)
			self.assertRaises(TypeError, msData.updateMasks, rsdThreshold='30')

		with self.subTest(msg='Blanks'):
			self.assertRaises(TypeError, msData.updateMasks, blankThreshold='A string')

		with self.subTest(msg='RSD'):
			self.assertRaises(ValueError, msData.updateMasks, rsdThreshold=-1.01)
			self.assertRaises(TypeError, msData.updateMasks, rsdThreshold='30')

		with self.subTest(msg='Variance Ratio'):
			self.assertRaises(TypeError, msData.updateMasks, varianceRatio='1.1')

		with self.subTest(msg='ArtifactualParameters'):
			self.assertRaises(TypeError, msData.updateMasks, withArtifactualFiltering='A string', blankThreshold=False)
			self.assertRaises(ValueError, msData.updateMasks, corrThresholdArtifactual=1.01, blankThreshold=False)
			self.assertRaises(ValueError, msData.updateMasks, corrThresholdArtifactual=-0.01, blankThreshold=False)
			self.assertRaises(TypeError, msData.updateMasks, corrThresholdArtifactual='0.7', blankThreshold=False)
			self.assertRaises(TypeError, msData.updateMasks, deltaMzArtifactual='100', blankThreshold=False)
			self.assertRaises(TypeError, msData.updateMasks, overlapThresholdArtifactual='0.5', blankThreshold=False)


	def test_correlationToDilution(self):

		from nPYc.utilities._internal import _vcorrcoef

		noSamp = numpy.random.randint(30, high=500, size=None)
		noFeat = numpy.random.randint(200, high=400, size=None)

		dataset = generateTestDataset(noSamp, noFeat, dtype='MSDataset', sop='GenericMS')

		dataset.sampleMetadata['SampleType'] = nPYc.enumerations.SampleType.StudyPool
		dataset.sampleMetadata['AssayRole'] = nPYc.enumerations.AssayRole.LinearityReference
		dataset.sampleMetadata['Well'] = 1
		dataset.sampleMetadata['Dilution'] = numpy.linspace(1,noSamp, num=noSamp)

		correlations = dataset.correlationToDilution

		numpy.testing.assert_array_almost_equal(correlations, _vcorrcoef(dataset.intensityData, dataset.sampleMetadata['Dilution'].values))


	def test_correlateToDilution_raises(self):

		noSamp = numpy.random.randint(30, high=500, size=None)
		noFeat = numpy.random.randint(200, high=400, size=None)

		dataset = generateTestDataset(noSamp, noFeat, dtype='MSDataset')

		with self.subTest(msg='Unknown correlation type'):
			self.assertRaises(ValueError, dataset._MSDataset__correlateToDilution, method='unknown')

		with self.subTest(msg='No LR samples'):
			dataset.sampleMetadata['AssayRole'] = AssayRole.Assay
			self.assertRaises(ValueError, dataset._MSDataset__correlateToDilution)


		with self.subTest(msg='No Dilution field'):
			dataset.sampleMetadata.drop(['Dilution'], axis=1, inplace=True)
			self.assertRaises(KeyError, dataset._MSDataset__correlateToDilution)



	def test_validateObject(self):
		with self.subTest(msg='validateObject successful on correct dataset'):
			goodDataset = copy.deepcopy(self.msData)
			self.assertEqual(goodDataset.validateObject(verbose=False, raiseError=False, raiseWarning=True), {'Dataset': True, 'BasicMSDataset':True ,'QC':True, 'sampleMetadata':True})

		with self.subTest(msg='BasicMSDataset fails on empty MSDataset'):
			badDataset = nPYc.MSDataset('', fileType='empty')
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicMSDataset':False ,'QC':False, 'sampleMetadata':False})

		with self.subTest(msg='check raise no warnings with raiseWarning=False'):
			badDataset = copy.deepcopy(self.msData)
			del badDataset.Attributes['rtWindow']
			with warnings.catch_warnings(record=True) as w:
				# Cause all warnings to always be triggered.
				warnings.simplefilter("always")
				# warning
				result = badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False)
				# check it generally worked
				self.assertEqual(result, {'Dataset': True, 'BasicMSDataset': False, 'QC': False, 'sampleMetadata': False})
				# check each warning
				self.assertEqual(len(w), 0)

		with self.subTest(msg='check fail and raise warnings on bad Dataset'):
			badDataset = copy.deepcopy(self.msData)
			delattr(badDataset, 'featureMetadata')
			with warnings.catch_warnings(record=True) as w:
				# Cause all warnings to always be triggered.
				warnings.simplefilter("always")
				# warning
				result = badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=True)
				# check it generally worked
				self.assertEqual(result, {'Dataset': False, 'BasicMSDataset': False, 'QC': False, 'sampleMetadata': False})
				# check each warning
				self.assertEqual(len(w), 5)
				assert issubclass(w[0].category, UserWarning)
				assert "Failure, no attribute 'self.featureMetadata'" in str(w[0].message)
				assert issubclass(w[1].category, UserWarning)
				assert "Does not conform to Dataset:" in str(w[1].message)
				assert issubclass(w[2].category, UserWarning)
				assert "Does not conform to basic MSDataset" in str(w[2].message)
				assert issubclass(w[3].category, UserWarning)
				assert "Does not have QC parameters" in str(w[3].message)
				assert issubclass(w[4].category, UserWarning)
				assert "Does not have sample metadata information" in str(w[4].message)

		with self.subTest(msg='check raise warnings BasicMSDataset'):
			badDataset = copy.deepcopy(self.msData)
			del badDataset.Attributes['rtWindow']
			with warnings.catch_warnings(record=True) as w:
				# Cause all warnings to always be triggered.
				warnings.simplefilter("always")
				# warning
				result = badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=True)
				# check it generally worked
				self.assertEqual(result, {'Dataset': True, 'BasicMSDataset': False, 'QC': False, 'sampleMetadata': False})
				# check each warning
				self.assertEqual(len(w), 4)
				assert issubclass(w[0].category, UserWarning)
				assert "Failure, no attribute 'self.Attributes['rtWindow']" in str(w[0].message)
				assert issubclass(w[1].category, UserWarning)
				assert "Does not conform to basic MSDataset:" in str(w[1].message)
				assert issubclass(w[2].category, UserWarning)
				assert "Does not have QC parameters" in str(w[2].message)
				assert issubclass(w[3].category, UserWarning)
				assert "Does not have sample metadata information" in str(w[3].message)

		with self.subTest(msg='check raise warnings QC parameters'):
			badDataset = copy.deepcopy(self.msData)
			badDataset.sampleMetadata['Batch'] = 'not an int or float'
			with warnings.catch_warnings(record=True) as w:
				# Cause all warnings to always be triggered.
				warnings.simplefilter("always")
				# warning
				result = badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=True)
				# check it generally worked
				self.assertEqual(result, {'Dataset': True, 'BasicMSDataset': True, 'QC': False, 'sampleMetadata': False})
				# check each warning
				self.assertEqual(len(w), 3)
				assert issubclass(w[0].category, UserWarning)
				assert "Failure, 'self.sampleMetadata['Batch']' is <class 'str'>" in str(w[0].message)
				assert issubclass(w[1].category, UserWarning)
				assert "Does not have QC parameters:" in str(w[1].message)
				assert issubclass(w[2].category, UserWarning)
				assert "Does not have sample metadata information:" in str(w[2].message)

		with self.subTest(msg='check raise warnings sampleMetadata'):
			badDataset = copy.deepcopy(self.msData)
			badDataset.sampleMetadata.drop(['Subject ID'], axis=1, inplace=True)
			with warnings.catch_warnings(record=True) as w:
				# Cause all warnings to always be triggered.
				warnings.simplefilter("always")
				# warning
				result = badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=True)
				# check it generally worked
				self.assertEqual(result, {'Dataset': True, 'BasicMSDataset': True, 'QC': True, 'sampleMetadata': False})
				# check each warning
				self.assertEqual(len(w), 2)
				assert issubclass(w[0].category, UserWarning)
				assert "Failure, 'self.sampleMetadata' lacks a 'Subject ID' column" in str(w[0].message)
				assert issubclass(w[1].category, UserWarning)
				assert "Does not have sample metadata information:" in str(w[1].message)

		with self.subTest(msg='self.Attributes[\'rtWindow\'] does not exist'):
			badDataset = copy.deepcopy(self.msData)
			del badDataset.Attributes['rtWindow']
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicMSDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(AttributeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.Attributes[\'rtWindow\'] is not an int or float'):
			badDataset = copy.deepcopy(self.msData)
			badDataset.Attributes['rtWindow'] = 'not an int or float'
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicMSDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(TypeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='self.Attributes[\'msPrecision\'] does not exist'):
			badDataset = copy.deepcopy(self.msData)
			del badDataset.Attributes['msPrecision']
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicMSDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(AttributeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.Attributes[\'msPrecision\'] is not an int or float'):
			badDataset = copy.deepcopy(self.msData)
			badDataset.Attributes['msPrecision'] = 'not an int or float'
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicMSDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(TypeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='self.Attributes[\'varianceRatio\'] does not exist'):
			badDataset = copy.deepcopy(self.msData)
			del badDataset.Attributes['varianceRatio']
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicMSDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(AttributeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.Attributes[\'varianceRatio\'] is not an int or float'):
			badDataset = copy.deepcopy(self.msData)
			badDataset.Attributes['varianceRatio'] = 'not an int or float'
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicMSDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(TypeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='self.Attributes[\'blankThreshold\'] does not exist'):
			badDataset = copy.deepcopy(self.msData)
			del badDataset.Attributes['blankThreshold']
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicMSDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(AttributeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.Attributes[\'blankThreshold\'] is not an int or float'):
			badDataset = copy.deepcopy(self.msData)
			badDataset.Attributes['blankThreshold'] = 'not an int or float'
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicMSDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(TypeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='self.Attributes[\'corrMethod\'] does not exist'):
			badDataset = copy.deepcopy(self.msData)
			del badDataset.Attributes['corrMethod']
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicMSDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(AttributeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.Attributes[\'corrMethod\'] is not a str'):
			badDataset = copy.deepcopy(self.msData)
			badDataset.Attributes['corrMethod'] = 5.0
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicMSDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(TypeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='self.Attributes[\'corrThreshold\'] does not exist'):
			badDataset = copy.deepcopy(self.msData)
			del badDataset.Attributes['corrThreshold']
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicMSDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(AttributeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.Attributes[\'corrThreshold\'] is not an int or float'):
			badDataset = copy.deepcopy(self.msData)
			badDataset.Attributes['corrThreshold'] = 'not an int or float'
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicMSDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(TypeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='self.Attributes[\'rsdThreshold\'] does not exist'):
			badDataset = copy.deepcopy(self.msData)
			del badDataset.Attributes['rsdThreshold']
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicMSDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(AttributeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.Attributes[\'rsdThreshold\'] is not an int or float'):
			badDataset = copy.deepcopy(self.msData)
			badDataset.Attributes['rsdThreshold'] = 'not an int or float'
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicMSDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(TypeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='self.Attributes[\'artifactualFilter\'] does not exist'):
			badDataset = copy.deepcopy(self.msData)
			del badDataset.Attributes['artifactualFilter']
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicMSDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(AttributeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.Attributes[\'artifactualFilter\'] is not a bool'):
			badDataset = copy.deepcopy(self.msData)
			badDataset.Attributes['artifactualFilter'] = 'not a bool'
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicMSDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(TypeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='self.Attributes[\'deltaMzArtifactual\'] does not exist'):
			badDataset = copy.deepcopy(self.msData)
			del badDataset.Attributes['deltaMzArtifactual']
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicMSDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(AttributeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.Attributes[\'deltaMzArtifactual\'] is not an int or float'):
			badDataset = copy.deepcopy(self.msData)
			badDataset.Attributes['deltaMzArtifactual'] = 'not an int or float'
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicMSDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(TypeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='self.Attributes[\'overlapThresholdArtifactual\'] does not exist'):
			badDataset = copy.deepcopy(self.msData)
			del badDataset.Attributes['overlapThresholdArtifactual']
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicMSDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(AttributeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.Attributes[\'overlapThresholdArtifactual\'] is not an int or float'):
			badDataset = copy.deepcopy(self.msData)
			badDataset.Attributes['overlapThresholdArtifactual'] = 'not an int or float'
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicMSDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(TypeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='self.Attributes[\'corrThresholdArtifactual\'] does not exist'):
			badDataset = copy.deepcopy(self.msData)
			del badDataset.Attributes['corrThresholdArtifactual']
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicMSDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(AttributeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.Attributes[\'corrThresholdArtifactual\'] is not an int or float'):
			badDataset = copy.deepcopy(self.msData)
			badDataset.Attributes['corrThresholdArtifactual'] = 'not an int or float'
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicMSDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(TypeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='self.Attributes[\'FeatureExtractionSoftware\'] does not exist'):
			badDataset = copy.deepcopy(self.msData)
			del badDataset.Attributes['FeatureExtractionSoftware']
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicMSDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(AttributeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.Attributes[\'FeatureExtractionSoftware\'] is not a str'):
			badDataset = copy.deepcopy(self.msData)
			badDataset.Attributes['FeatureExtractionSoftware'] = 5.0
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicMSDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(TypeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='self.Attributes[\'Raw Data Path\'] does not exist'):
			badDataset = copy.deepcopy(self.msData)
			del badDataset.Attributes['Raw Data Path']
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicMSDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(AttributeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.Attributes[\'Raw Data Path\'] is not a str'):
			badDataset = copy.deepcopy(self.msData)
			badDataset.Attributes['Raw Data Path'] = 5.0
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicMSDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(TypeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='self.Attributes[\'Feature Names\'] does not exist'):
			badDataset = copy.deepcopy(self.msData)
			del badDataset.Attributes['Feature Names']
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicMSDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(AttributeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.Attributes[\'Feature Names\'] is not a str'):
			badDataset = copy.deepcopy(self.msData)
			badDataset.Attributes['Feature Names'] = 5.0
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicMSDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(TypeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.VariableType is not an enum VariableType'):
			badDataset = copy.deepcopy(self.msData)
			badDataset.VariableType = 'not an enum'
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicMSDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(TypeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='self.corrExclusions does not exist'):
			badDataset = copy.deepcopy(self.msData)
			delattr(badDataset, 'corrExclusions')
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicMSDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(AttributeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='self._correlationToDilution does not exist'):
			badDataset = copy.deepcopy(self.msData)
			delattr(badDataset, '_correlationToDilution')
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicMSDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(AttributeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self._correlationToDilution is not a numpy.ndarray'):
			badDataset = copy.deepcopy(self.msData)
			badDataset._correlationToDilution = 'not a numpy.ndarray'
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicMSDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(TypeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='self._artifactualLinkageMatrix does not exist'):
			badDataset = copy.deepcopy(self.msData)
			delattr(badDataset, '_artifactualLinkageMatrix')
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicMSDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(AttributeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self._artifactualLinkageMatrix is not a pandas.DataFrame'):
			badDataset = copy.deepcopy(self.msData)
			badDataset._artifactualLinkageMatrix = 'not a pandas.DataFrame'
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicMSDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(TypeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='self._tempArtifactualLinkageMatrix does not exist'):
			badDataset = copy.deepcopy(self.msData)
			delattr(badDataset, '_tempArtifactualLinkageMatrix')
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicMSDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(AttributeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self._tempArtifactualLinkageMatrix is not a pandas.DataFrame'):
			badDataset = copy.deepcopy(self.msData)
			badDataset._tempArtifactualLinkageMatrix = 'not a pandas.DataFrame'
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicMSDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(TypeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='self.fileName does not exist'):
			badDataset = copy.deepcopy(self.msData)
			delattr(badDataset, 'fileName')
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicMSDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(AttributeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.fileName is not a str'):
			badDataset = copy.deepcopy(self.msData)
			badDataset.fileName = 5.
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicMSDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(TypeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='self.filePath does not exist'):
			badDataset = copy.deepcopy(self.msData)
			delattr(badDataset, 'filePath')
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicMSDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(AttributeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.filePath is not a str'):
			badDataset = copy.deepcopy(self.msData)
			badDataset.filePath = 5.
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicMSDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(TypeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.sampleMetadata does not have the same number of samples as self._intensityData'):
			badDataset = copy.deepcopy(self.msData)
			badDataset.sampleMetadata.drop([0], axis=0, inplace=True)
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicMSDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(ValueError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.sampleMetadata[\'Sample File Name\'] is not str'):
			badDataset = copy.deepcopy(self.msData)
			badDataset.sampleMetadata['Sample File Name'] = 5.
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicMSDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(TypeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.sampleMetadata[\'AssayRole\'] is not an enum \'AssayRole\''):
			badDataset = copy.deepcopy(self.msData)
			badDataset.sampleMetadata['AssayRole'] = 5.
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicMSDataset': True, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(TypeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.sampleMetadata[\'SampleType\'] is not an enum \'SampleType\''):
			badDataset = copy.deepcopy(self.msData)
			badDataset.sampleMetadata['SampleType'] = 5.
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicMSDataset': True, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(TypeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.sampleMetadata[\'Dilution\'] is not an int or float'):
			badDataset = copy.deepcopy(self.msData)
			badDataset.sampleMetadata['Dilution'] = 'not an int or float'
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicMSDataset': True, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(TypeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.sampleMetadata[\'Batch\'] is not an int or float'):
			badDataset = copy.deepcopy(self.msData)
			badDataset.sampleMetadata['Batch'] = 'not an int or float'
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicMSDataset': True, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(TypeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.sampleMetadata[\'Correction Batch\'] is not an int or float'):
			badDataset = copy.deepcopy(self.msData)
			badDataset.sampleMetadata['Correction Batch'] = 'not an int or float'
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicMSDataset': True, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(TypeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.sampleMetadata[\'Run Order\'] is not an int'):
			badDataset = copy.deepcopy(self.msData)
			badDataset.sampleMetadata['Run Order'] = 'not an int'
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicMSDataset': True, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(TypeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.sampleMetadata[\'Acquired Time\'] is not a datetime'):
			badDataset = copy.deepcopy(self.msData)
			badDataset.sampleMetadata['Acquired Time'] = 5.
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicMSDataset': True, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(TypeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.sampleMetadata[\'Sample Base Name\'] is not str'):
			badDataset = copy.deepcopy(self.msData)
			badDataset.sampleMetadata['Sample Base Name'] = 5.
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicMSDataset': True, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(TypeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.sampleMetadata does not have a Matrix column'):
			badDataset = copy.deepcopy(self.msData)
			badDataset.sampleMetadata.drop(['Matrix'], axis=1, inplace=True)
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicMSDataset': True, 'QC': True, 'sampleMetadata': False})
			self.assertRaises(LookupError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.sampleMetadata[\'Matrix\'] is not str'):
			badDataset = copy.deepcopy(self.msData)
			badDataset.sampleMetadata['Matrix'] = 5.
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicMSDataset': True, 'QC': True, 'sampleMetadata': False})
			self.assertRaises(TypeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.sampleMetadata does not have a Subject ID column'):
			badDataset = copy.deepcopy(self.msData)
			badDataset.sampleMetadata.drop(['Subject ID'], axis=1, inplace=True)
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicMSDataset': True, 'QC': True, 'sampleMetadata': False})
			self.assertRaises(LookupError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.sampleMetadata[\'Subject ID\'] is not str'):
			badDataset = copy.deepcopy(self.msData)
			badDataset.sampleMetadata['Subject ID'] = 5.
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicMSDataset': True, 'QC': True, 'sampleMetadata': False})
			self.assertRaises(TypeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.sampleMetadata[\'Sampling ID\'] is not str'):
			badDataset = copy.deepcopy(self.msData)
			badDataset.sampleMetadata['Sampling ID'] = 5.
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicMSDataset': True, 'QC': True, 'sampleMetadata': False})
			self.assertRaises(TypeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.featureMetadata does not have the same number of samples as self._intensityData'):
			badDataset = copy.deepcopy(self.msData)
			badDataset.featureMetadata.drop([0], axis=0, inplace=True)
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicMSDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(ValueError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.featureMetadata[\'Feature Name\'] is not a str'):
			badDataset = copy.deepcopy(self.msData)
			badDataset.featureMetadata['Feature Name'] = 5.
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicMSDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(TypeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.featureMetadata[\'Feature Name\'] is not unique'):
			badDataset = copy.deepcopy(self.msData)
			badDataset.featureMetadata['Feature Name'] = ['Feature1','Feature1','Feature1']
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicMSDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(ValueError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.featureMetadata does not have a m/z column'):
			badDataset = copy.deepcopy(self.msData)
			badDataset.featureMetadata.drop(['m/z'], axis=1, inplace=True)
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicMSDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(LookupError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.featureMetadata[\'m/z\'] is not an int or float'):
			badDataset = copy.deepcopy(self.msData)
			badDataset.featureMetadata['m/z'] = 'not an int or float'
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicMSDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(TypeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.featureMetadata does not have a Retention Time column'):
			badDataset = copy.deepcopy(self.msData)
			badDataset.featureMetadata.drop(['Retention Time'], axis=1, inplace=True)
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicMSDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(LookupError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.featureMetadata[\'Retention Time\'] is not an int or float'):
			badDataset = copy.deepcopy(self.msData)
			badDataset.featureMetadata['Retention Time'] = 'not an int or float'
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicMSDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(TypeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.sampleMask has not been initialised'):
			badDataset = copy.deepcopy(self.msData)
			badDataset.sampleMask = numpy.array(False, dtype=bool)
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicMSDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(ValueError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.sampleMask does not have the same number of samples as self._intensityData'):
			badDataset = copy.deepcopy(self.msData)
			badDataset.sampleMask = numpy.squeeze(numpy.ones([5, 1], dtype=bool), axis=1)
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicMSDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(ValueError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.featureMask has not been initialised'):
			badDataset = copy.deepcopy(self.msData)
			badDataset.featureMask = numpy.array(False, dtype=bool)
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicMSDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(ValueError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.featureMask does not have the same number of samples as self._intensityData'):
			badDataset = copy.deepcopy(self.msData)
			badDataset.featureMask = numpy.squeeze(numpy.ones([5, 1], dtype=bool), axis=1)
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicMSDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(ValueError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)


class test_msdataset_batch_inference(unittest.TestCase):
	"""
	Check batches are generated and amended correctly
	"""

	def setUp(self):
		self.msData = nPYc.MSDataset('', fileType='empty')

		self.msData.sampleMetadata['Sample File Name'] = ['Test_RPOS_ToF04_B1S1_SR',
														'Test_RPOS_ToF04_B1S2_SR',
														'Test_RPOS_ToF04_B1S3_SR',
														'Test_RPOS_ToF04_B1S4_SR',
														'Test_RPOS_ToF04_B1S5_SR',
														'Test_RPOS_ToF04_P1W01',
														'Test_RPOS_ToF04_P1W02_SR',
														'Test_RPOS_ToF04_P1W03',
														'Test_RPOS_ToF04_B1E1_SR',
														'Test_RPOS_ToF04_B1E2_SR',
														'Test_RPOS_ToF04_B1E3_SR',
														'Test_RPOS_ToF04_B1E4_SR',
														'Test_RPOS_ToF04_B1E5_SR',
														'Test_RPOS_ToF04_B2S1_SR',
														'Test_RPOS_ToF04_B2S2_SR',
														'Test_RPOS_ToF04_B2S3_SR',
														'Test_RPOS_ToF04_B2S4_SR',
														'Test_RPOS_ToF04_B2S5_SR',
														'Test_RPOS_ToF04_P2W01',
														'Test_RPOS_ToF04_P2W02_SR',
														'Test_RPOS_ToF04_P3W03',
														'Test_RPOS_ToF04_B2S1_SR_2',
														'Test_RPOS_ToF04_B2S2_SR_2',
														'Test_RPOS_ToF04_B2S3_SR_2',
														'Test_RPOS_ToF04_B2S4_SR_2',
														'Test_RPOS_ToF04_B2S5_SR_2',
														'Test_RPOS_ToF04_P3W03_b',
														'Test_RPOS_ToF04_B2E1_SR',
														'Test_RPOS_ToF04_B2E2_SR',
														'Test_RPOS_ToF04_B2E3_SR',
														'Test_RPOS_ToF04_B2E4_SR',
														'Test_RPOS_ToF04_B2E5_SR',
														'Test_RPOS_ToF04_B2SRD1']

		self.msData.addSampleInfo(descriptionFormat='Filenames')
		self.msData.sampleMetadata['Run Order'] = self.msData.sampleMetadata.index + 1


	def test_fillbatches_correctionbatch(self):

		self.msData._fillBatches()

		correctionBatch = pandas.Series([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
										1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0,
										3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, numpy.nan],
										name='Correction Batch',
										dtype='float')

		pandas.util.testing.assert_series_equal(self.msData.sampleMetadata['Correction Batch'], correctionBatch)


	def test_fillbatches_warns(self):

		self.msData.sampleMetadata.drop('Run Order', axis=1, inplace=True)

		self.assertWarnsRegex(UserWarning, 'Unable to infer batches without run order, skipping\.', self.msData._fillBatches)


	def test_amendbatches(self):
		"""
		
		"""

		self.msData._fillBatches()

		self.msData.amendBatches(20)

		correctionBatch = pandas.Series([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
										1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 4.0,
										4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, numpy.nan],
										name='Correction Batch',
										dtype='float')

		pandas.util.testing.assert_series_equal(self.msData.sampleMetadata['Correction Batch'], correctionBatch)


	def test_msdataset_addsampleinfo_batches(self):

		self.msData.addSampleInfo(descriptionFormat='Batches')

		correctionBatch = pandas.Series([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
										1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0,
										3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, numpy.nan],
										name='Correction Batch',
										dtype='float')

		pandas.util.testing.assert_series_equal(self.msData.sampleMetadata['Correction Batch'], correctionBatch)


class test_msdataset_import_csvexport(unittest.TestCase):
	"""
	Test import from NPC csv files
	"""
	def test_raise_notimplemented(self):
		self.assertRaises(NotImplementedError, nPYc.MSDataset, os.path.join('nopath'), fileType='csv export')


class test_msdataset_import_undefined(unittest.TestCase):
	"""
	Test we raise an error when passing an fileType we don't understand.
	"""
	def test_raise_notimplemented(self):
		self.assertRaises(NotImplementedError, nPYc.MSDataset, os.path.join('nopath'), fileType='Unknown filetype')


class test_msdataset_import_QI(unittest.TestCase):
	"""
	Test import from Bruker Metaboscape
	"""

	def setUp(self):

		self.msData = nPYc.MSDataset(os.path.join('..','..','npc-standard-project','Derived_Data','UnitTest1_PCSOP.069_Metaboscape.xlsx'),
									 fileType='Metaboscape',
									 sheetName='Test Data',
									 noFeatureParams=18)

		self.msData.addSampleInfo(descriptionFormat='Filenames')


	def test_dimensions(self):

		self.assertEqual((self.msData.noSamples, self.msData.noFeatures), (115, 4))


	def test_samples(self):

		samples = pandas.Series(['UnitTest1_LPOS_ToF02_B1SRD01', 'UnitTest1_LPOS_ToF02_B1SRD02',
								'UnitTest1_LPOS_ToF02_B1SRD03', 'UnitTest1_LPOS_ToF02_B1SRD04',
								'UnitTest1_LPOS_ToF02_B1SRD05', 'UnitTest1_LPOS_ToF02_B1SRD06',
								'UnitTest1_LPOS_ToF02_B1SRD07', 'UnitTest1_LPOS_ToF02_B1SRD08',
								'UnitTest1_LPOS_ToF02_B1SRD09', 'UnitTest1_LPOS_ToF02_B1SRD10',
								'UnitTest1_LPOS_ToF02_B1SRD11', 'UnitTest1_LPOS_ToF02_B1SRD12',
								'UnitTest1_LPOS_ToF02_B1SRD13', 'UnitTest1_LPOS_ToF02_B1SRD14',
								'UnitTest1_LPOS_ToF02_B1SRD15', 'UnitTest1_LPOS_ToF02_B1SRD16',
								'UnitTest1_LPOS_ToF02_B1SRD17', 'UnitTest1_LPOS_ToF02_B1SRD18',
								'UnitTest1_LPOS_ToF02_B1SRD19', 'UnitTest1_LPOS_ToF02_B1SRD20',
								'UnitTest1_LPOS_ToF02_B1SRD21', 'UnitTest1_LPOS_ToF02_B1SRD22',
								'UnitTest1_LPOS_ToF02_B1SRD23', 'UnitTest1_LPOS_ToF02_B1SRD24',
								'UnitTest1_LPOS_ToF02_B1SRD25', 'UnitTest1_LPOS_ToF02_B1SRD26',
								'UnitTest1_LPOS_ToF02_B1SRD27', 'UnitTest1_LPOS_ToF02_B1SRD28',
								'UnitTest1_LPOS_ToF02_B1SRD29', 'UnitTest1_LPOS_ToF02_B1SRD30',
								'UnitTest1_LPOS_ToF02_B1SRD31', 'UnitTest1_LPOS_ToF02_B1SRD32',
								'UnitTest1_LPOS_ToF02_B1SRD33', 'UnitTest1_LPOS_ToF02_B1SRD34',
								'UnitTest1_LPOS_ToF02_B1SRD35', 'UnitTest1_LPOS_ToF02_B1SRD36',
								'UnitTest1_LPOS_ToF02_B1SRD37', 'UnitTest1_LPOS_ToF02_B1SRD38',
								'UnitTest1_LPOS_ToF02_B1SRD39', 'UnitTest1_LPOS_ToF02_B1SRD40',
								'UnitTest1_LPOS_ToF02_B1SRD41', 'UnitTest1_LPOS_ToF02_B1SRD42',
								'UnitTest1_LPOS_ToF02_B1SRD43', 'UnitTest1_LPOS_ToF02_B1SRD44',
								'UnitTest1_LPOS_ToF02_B1SRD45', 'UnitTest1_LPOS_ToF02_B1SRD46',
								'UnitTest1_LPOS_ToF02_B1SRD47', 'UnitTest1_LPOS_ToF02_B1SRD48',
								'UnitTest1_LPOS_ToF02_B1SRD49', 'UnitTest1_LPOS_ToF02_B1SRD50',
								'UnitTest1_LPOS_ToF02_B1SRD51', 'UnitTest1_LPOS_ToF02_B1SRD52',
								'UnitTest1_LPOS_ToF02_B1SRD53', 'UnitTest1_LPOS_ToF02_B1SRD54',
								'UnitTest1_LPOS_ToF02_B1SRD55', 'UnitTest1_LPOS_ToF02_B1SRD56',
								'UnitTest1_LPOS_ToF02_B1SRD57', 'UnitTest1_LPOS_ToF02_B1SRD58',
								'UnitTest1_LPOS_ToF02_B1SRD59', 'UnitTest1_LPOS_ToF02_B1SRD60',
								'UnitTest1_LPOS_ToF02_B1SRD61', 'UnitTest1_LPOS_ToF02_B1SRD62',
								'UnitTest1_LPOS_ToF02_B1SRD63', 'UnitTest1_LPOS_ToF02_B1SRD64',
								'UnitTest1_LPOS_ToF02_B1SRD65', 'UnitTest1_LPOS_ToF02_B1SRD66',
								'UnitTest1_LPOS_ToF02_B1SRD67', 'UnitTest1_LPOS_ToF02_B1SRD68',
								'UnitTest1_LPOS_ToF02_B1SRD69', 'UnitTest1_LPOS_ToF02_B1SRD70',
								'UnitTest1_LPOS_ToF02_B1SRD71', 'UnitTest1_LPOS_ToF02_B1SRD72',
								'UnitTest1_LPOS_ToF02_B1SRD73', 'UnitTest1_LPOS_ToF02_B1SRD74',
								'UnitTest1_LPOS_ToF02_B1SRD75', 'UnitTest1_LPOS_ToF02_B1SRD76',
								'UnitTest1_LPOS_ToF02_B1SRD77', 'UnitTest1_LPOS_ToF02_B1SRD78',
								'UnitTest1_LPOS_ToF02_B1SRD79', 'UnitTest1_LPOS_ToF02_B1SRD80',
								'UnitTest1_LPOS_ToF02_B1SRD81', 'UnitTest1_LPOS_ToF02_B1SRD82',
								'UnitTest1_LPOS_ToF02_B1SRD83', 'UnitTest1_LPOS_ToF02_B1SRD84',
								'UnitTest1_LPOS_ToF02_B1SRD85', 'UnitTest1_LPOS_ToF02_B1SRD86',
								'UnitTest1_LPOS_ToF02_B1SRD87', 'UnitTest1_LPOS_ToF02_B1SRD88',
								'UnitTest1_LPOS_ToF02_B1SRD89', 'UnitTest1_LPOS_ToF02_B1SRD90',
								'UnitTest1_LPOS_ToF02_B1SRD91', 'UnitTest1_LPOS_ToF02_B1SRD92',
								'UnitTest1_LPOS_ToF02_Blank01', 'UnitTest1_LPOS_ToF02_Blank02',
								'UnitTest1_LPOS_ToF02_B1E1_SR', 'UnitTest1_LPOS_ToF02_B1E2_SR',
								'UnitTest1_LPOS_ToF02_B1E3_SR', 'UnitTest1_LPOS_ToF02_B1E4_SR',
								'UnitTest1_LPOS_ToF02_B1E5_SR', 'UnitTest1_LPOS_ToF02_B1S1_SR',
								'UnitTest1_LPOS_ToF02_B1S2_SR', 'UnitTest1_LPOS_ToF02_B1S3_SR',
								'UnitTest1_LPOS_ToF02_B1S4_SR', 'UnitTest1_LPOS_ToF02_B1S5_SR',
								'UnitTest1_LPOS_ToF02_S1W01', 'UnitTest1_LPOS_ToF02_S1W02',
								'UnitTest1_LPOS_ToF02_S1W03', 'UnitTest1_LPOS_ToF02_S1W04',
								'UnitTest1_LPOS_ToF02_S1W05', 'UnitTest1_LPOS_ToF02_S1W06',
								'UnitTest1_LPOS_ToF02_S1W07', 'UnitTest1_LPOS_ToF02_S1W08_x',
								'UnitTest1_LPOS_ToF02_S1W11_LTR', 'UnitTest1_LPOS_ToF02_S1W12_SR',
								'UnitTest1_LPOS_ToF02_ERROR'],
								name='Sample File Name',
								dtype=str)

		pandas.util.testing.assert_series_equal(self.msData.sampleMetadata['Sample File Name'], samples)


	def test_featuremetadata_import(self):

		with self.subTest(msg='Checking Feature Names'):
			features = pandas.Series(['3.17_262.0378m/z',
									'3.17_293.1812m/z',
									'3.17_145.0686m/z',
									'3.17_258.1033m/z'],
									name='Feature Name',
									dtype='str')

			pandas.util.testing.assert_series_equal(self.msData.featureMetadata['Feature Name'], features)

		with self.subTest(msg='M meas.'):
			peakWidth = pandas.Series([263.0378339,
									   294.1811941,
									   146.0686347,
									   259.1033447],
									name='M meas.',
									dtype='float')

			pandas.util.testing.assert_series_equal(self.msData.featureMetadata['M meas.'], peakWidth)

		with self.subTest(msg='Checking m/z'):
			mz = pandas.Series([262.0378339,
							293.1811941,
							145.0686347,
							258.1033447],
							name='m/z',
							dtype='float')

			pandas.util.testing.assert_series_equal(self.msData.featureMetadata['m/z'], mz)

		with self.subTest(msg='Checking Retention Time'):
			rt = pandas.Series([3.17485,
							3.17485,
							3.17485,
							3.17485],
							name='Retention Time',
							dtype='float')

			pandas.util.testing.assert_series_equal(self.msData.featureMetadata['Retention Time'], rt)

		with self.subTest(msg='Checking ΔRT'):
			isotope = pandas.Series([0,
									0.1,
									0,
									-0.1],
							name='Isotope Distribution',
							dtype='float')

			pandas.util.testing.assert_series_equal(self.msData.featureMetadata['ΔRT'], isotope)


	def test_variabletype(self):

		self.assertEqual(self.msData.VariableType, nPYc.enumerations.VariableType.Discrete)


class test_msdataset_import_QI(unittest.TestCase):
	"""
	Test import from QI csv files
	"""

	def setUp(self):

		self.msData = nPYc.MSDataset(os.path.join('..','..','npc-standard-project','Derived_Data','UnitTest1_PCSOP.069_QI.csv'), fileType='QI')

		self.msData.addSampleInfo(descriptionFormat='Filenames')


	def test_dimensions(self):

		self.assertEqual((self.msData.noSamples, self.msData.noFeatures), (115, 4))


	def test_samples(self):

		samples = pandas.Series(['UnitTest1_LPOS_ToF02_B1SRD01', 'UnitTest1_LPOS_ToF02_B1SRD02',
								'UnitTest1_LPOS_ToF02_B1SRD03', 'UnitTest1_LPOS_ToF02_B1SRD04',
								'UnitTest1_LPOS_ToF02_B1SRD05', 'UnitTest1_LPOS_ToF02_B1SRD06',
								'UnitTest1_LPOS_ToF02_B1SRD07', 'UnitTest1_LPOS_ToF02_B1SRD08',
								'UnitTest1_LPOS_ToF02_B1SRD09', 'UnitTest1_LPOS_ToF02_B1SRD10',
								'UnitTest1_LPOS_ToF02_B1SRD11', 'UnitTest1_LPOS_ToF02_B1SRD12',
								'UnitTest1_LPOS_ToF02_B1SRD13', 'UnitTest1_LPOS_ToF02_B1SRD14',
								'UnitTest1_LPOS_ToF02_B1SRD15', 'UnitTest1_LPOS_ToF02_B1SRD16',
								'UnitTest1_LPOS_ToF02_B1SRD17', 'UnitTest1_LPOS_ToF02_B1SRD18',
								'UnitTest1_LPOS_ToF02_B1SRD19', 'UnitTest1_LPOS_ToF02_B1SRD20',
								'UnitTest1_LPOS_ToF02_B1SRD21', 'UnitTest1_LPOS_ToF02_B1SRD22',
								'UnitTest1_LPOS_ToF02_B1SRD23', 'UnitTest1_LPOS_ToF02_B1SRD24',
								'UnitTest1_LPOS_ToF02_B1SRD25', 'UnitTest1_LPOS_ToF02_B1SRD26',
								'UnitTest1_LPOS_ToF02_B1SRD27', 'UnitTest1_LPOS_ToF02_B1SRD28',
								'UnitTest1_LPOS_ToF02_B1SRD29', 'UnitTest1_LPOS_ToF02_B1SRD30',
								'UnitTest1_LPOS_ToF02_B1SRD31', 'UnitTest1_LPOS_ToF02_B1SRD32',
								'UnitTest1_LPOS_ToF02_B1SRD33', 'UnitTest1_LPOS_ToF02_B1SRD34',
								'UnitTest1_LPOS_ToF02_B1SRD35', 'UnitTest1_LPOS_ToF02_B1SRD36',
								'UnitTest1_LPOS_ToF02_B1SRD37', 'UnitTest1_LPOS_ToF02_B1SRD38',
								'UnitTest1_LPOS_ToF02_B1SRD39', 'UnitTest1_LPOS_ToF02_B1SRD40',
								'UnitTest1_LPOS_ToF02_B1SRD41', 'UnitTest1_LPOS_ToF02_B1SRD42',
								'UnitTest1_LPOS_ToF02_B1SRD43', 'UnitTest1_LPOS_ToF02_B1SRD44',
								'UnitTest1_LPOS_ToF02_B1SRD45', 'UnitTest1_LPOS_ToF02_B1SRD46',
								'UnitTest1_LPOS_ToF02_B1SRD47', 'UnitTest1_LPOS_ToF02_B1SRD48',
								'UnitTest1_LPOS_ToF02_B1SRD49', 'UnitTest1_LPOS_ToF02_B1SRD50',
								'UnitTest1_LPOS_ToF02_B1SRD51', 'UnitTest1_LPOS_ToF02_B1SRD52',
								'UnitTest1_LPOS_ToF02_B1SRD53', 'UnitTest1_LPOS_ToF02_B1SRD54',
								'UnitTest1_LPOS_ToF02_B1SRD55', 'UnitTest1_LPOS_ToF02_B1SRD56',
								'UnitTest1_LPOS_ToF02_B1SRD57', 'UnitTest1_LPOS_ToF02_B1SRD58',
								'UnitTest1_LPOS_ToF02_B1SRD59', 'UnitTest1_LPOS_ToF02_B1SRD60',
								'UnitTest1_LPOS_ToF02_B1SRD61', 'UnitTest1_LPOS_ToF02_B1SRD62',
								'UnitTest1_LPOS_ToF02_B1SRD63', 'UnitTest1_LPOS_ToF02_B1SRD64',
								'UnitTest1_LPOS_ToF02_B1SRD65', 'UnitTest1_LPOS_ToF02_B1SRD66',
								'UnitTest1_LPOS_ToF02_B1SRD67', 'UnitTest1_LPOS_ToF02_B1SRD68',
								'UnitTest1_LPOS_ToF02_B1SRD69', 'UnitTest1_LPOS_ToF02_B1SRD70',
								'UnitTest1_LPOS_ToF02_B1SRD71', 'UnitTest1_LPOS_ToF02_B1SRD72',
								'UnitTest1_LPOS_ToF02_B1SRD73', 'UnitTest1_LPOS_ToF02_B1SRD74',
								'UnitTest1_LPOS_ToF02_B1SRD75', 'UnitTest1_LPOS_ToF02_B1SRD76',
								'UnitTest1_LPOS_ToF02_B1SRD77', 'UnitTest1_LPOS_ToF02_B1SRD78',
								'UnitTest1_LPOS_ToF02_B1SRD79', 'UnitTest1_LPOS_ToF02_B1SRD80',
								'UnitTest1_LPOS_ToF02_B1SRD81', 'UnitTest1_LPOS_ToF02_B1SRD82',
								'UnitTest1_LPOS_ToF02_B1SRD83', 'UnitTest1_LPOS_ToF02_B1SRD84',
								'UnitTest1_LPOS_ToF02_B1SRD85', 'UnitTest1_LPOS_ToF02_B1SRD86',
								'UnitTest1_LPOS_ToF02_B1SRD87', 'UnitTest1_LPOS_ToF02_B1SRD88',
								'UnitTest1_LPOS_ToF02_B1SRD89', 'UnitTest1_LPOS_ToF02_B1SRD90',
								'UnitTest1_LPOS_ToF02_B1SRD91', 'UnitTest1_LPOS_ToF02_B1SRD92',
								'UnitTest1_LPOS_ToF02_Blank01', 'UnitTest1_LPOS_ToF02_Blank02',
								'UnitTest1_LPOS_ToF02_B1E1_SR', 'UnitTest1_LPOS_ToF02_B1E2_SR',
								'UnitTest1_LPOS_ToF02_B1E3_SR', 'UnitTest1_LPOS_ToF02_B1E4_SR',
								'UnitTest1_LPOS_ToF02_B1E5_SR', 'UnitTest1_LPOS_ToF02_B1S1_SR',
								'UnitTest1_LPOS_ToF02_B1S2_SR', 'UnitTest1_LPOS_ToF02_B1S3_SR',
								'UnitTest1_LPOS_ToF02_B1S4_SR', 'UnitTest1_LPOS_ToF02_B1S5_SR',
								'UnitTest1_LPOS_ToF02_S1W01', 'UnitTest1_LPOS_ToF02_S1W02',
								'UnitTest1_LPOS_ToF02_S1W03', 'UnitTest1_LPOS_ToF02_S1W04',
								'UnitTest1_LPOS_ToF02_S1W05', 'UnitTest1_LPOS_ToF02_S1W06',
								'UnitTest1_LPOS_ToF02_S1W07', 'UnitTest1_LPOS_ToF02_S1W08_x',
								'UnitTest1_LPOS_ToF02_S1W11_LTR', 'UnitTest1_LPOS_ToF02_S1W12_SR',
								'UnitTest1_LPOS_ToF02_ERROR'],
								name='Sample File Name',
								dtype=str)

		pandas.util.testing.assert_series_equal(self.msData.sampleMetadata['Sample File Name'], samples)


	def test_featuremetadata_import(self):

		with self.subTest(msg='Checking Feature Names'):
			features = pandas.Series(['3.17_262.0378m/z',
									'3.17_293.1812m/z',
									'3.17_145.0686m/z',
									'3.17_258.1033m/z'],
									name='Feature Name',
									dtype='str')

			pandas.util.testing.assert_series_equal(self.msData.featureMetadata['Feature Name'], features)

		with self.subTest(msg='Checking Peak Widths'):
			peakWidth = pandas.Series([0.03931667,
									0.01403333,
									0.01683333,
									0.01683333],
									name='Peak Width',
									dtype='float')

			pandas.util.testing.assert_series_equal(self.msData.featureMetadata['Peak Width'], peakWidth)

		with self.subTest(msg='Checking m/z'):
			mz = pandas.Series([262.0378339,
							293.1811941,
							145.0686347,
							258.1033447],
							name='m/z',
							dtype='float')

			pandas.util.testing.assert_series_equal(self.msData.featureMetadata['m/z'], mz)

		with self.subTest(msg='Checking Retention Time'):
			rt = pandas.Series([3.17485,
							3.17485,
							3.17485,
							3.17485],
							name='Retention Time',
							dtype='float')

			pandas.util.testing.assert_series_equal(self.msData.featureMetadata['Retention Time'], rt)

		with self.subTest(msg='Checking Isotope Distribution'):
			isotope = pandas.Series(['100 - 36.9',
									'100 - 11.9',
									'100 - 8.69',
									'100 - 73.4'],
							name='Isotope Distribution',
							dtype='str')

			pandas.util.testing.assert_series_equal(self.msData.featureMetadata['Isotope Distribution'], isotope)


	def test_dilutionlevels(self):

		dilution = pandas.Series([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 20., 20., 20., 20., 20.,
								40., 40., 40., 60., 60., 60., 80., 80., 80., 80., 80., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100.,
								1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 20., 20., 20., 20., 20.,
								40., 40., 40., 60., 60., 60., 80., 80., 80., 80., 80., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100.,
								numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan,
								numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan], 
								name='Dilution',
								dtype='float')

		pandas.util.testing.assert_series_equal(self.msData.sampleMetadata['Dilution'], dilution)


	def test_feature_correlation(self):

		self.msData.addSampleInfo(descriptionFormat='Raw Data', filePath=os.path.join('..','..','npc-standard-project','Raw_Data'))
		self.msData.addSampleInfo(descriptionFormat='Batches')

		with self.subTest(msg='Testing Pearson correlations'):
			correlations = numpy.array([0.99999997, 0.32017508, 1., -0.0693418])

			numpy.testing.assert_array_almost_equal(self.msData.correlationToDilution, correlations)

		with self.subTest(msg='Testing Spearman correlations'):
			correlations = numpy.array([0.9992837, 0.34708745, 1., -0.038844])

			self.msData.Attributes['corrMethod'] = 'spearman'

			numpy.testing.assert_array_almost_equal(self.msData.correlationToDilution, correlations)


	def test_variabletype(self):

		self.assertEqual(self.msData.VariableType, nPYc.enumerations.VariableType.Discrete)


class test_msdataset_import_xcms(unittest.TestCase):
	"""
	Test import from XCMS csv files
	"""

	def setUp(self):

		self.msData = nPYc.MSDataset(os.path.join('..','..','npc-standard-project','Derived_Data','UnitTest1_PCSOP.069_xcms.csv'), fileType='XCMS', noFeatureParams=9)
		self.msData_PeakTable = nPYc.MSDataset(os.path.join('..','..','npc-standard-project','Derived_Data','UnitTest1_PCSOP.069_xcms_peakTable.csv'), fileType='XCMS', noFeatureParams=8)

		self.msData.addSampleInfo(descriptionFormat='Filenames')
		self.msData_PeakTable.addSampleInfo(descriptionFormat='Filenames')

	def test_dimensions(self):

		self.assertEqual((self.msData.noSamples, self.msData.noFeatures), (111, 4))
		self.assertEqual((self.msData_PeakTable.noSamples, self.msData_PeakTable.noFeatures), (111, 4))

	def test_samples(self):

		samples = pandas.Series(['UnitTest1_LPOS_ToF02_B1SRD01', 'UnitTest1_LPOS_ToF02_B1SRD02',
								'UnitTest1_LPOS_ToF02_B1SRD03', 'UnitTest1_LPOS_ToF02_B1SRD04',
								'UnitTest1_LPOS_ToF02_B1SRD05', 'UnitTest1_LPOS_ToF02_B1SRD06',
								'UnitTest1_LPOS_ToF02_B1SRD07', 'UnitTest1_LPOS_ToF02_B1SRD08',
								'UnitTest1_LPOS_ToF02_B1SRD09', 'UnitTest1_LPOS_ToF02_B1SRD10',
								'UnitTest1_LPOS_ToF02_B1SRD11', 'UnitTest1_LPOS_ToF02_B1SRD12',
								'UnitTest1_LPOS_ToF02_B1SRD13', 'UnitTest1_LPOS_ToF02_B1SRD14',
								'UnitTest1_LPOS_ToF02_B1SRD15', 'UnitTest1_LPOS_ToF02_B1SRD16',
								'UnitTest1_LPOS_ToF02_B1SRD17', 'UnitTest1_LPOS_ToF02_B1SRD18',
								'UnitTest1_LPOS_ToF02_B1SRD19', 'UnitTest1_LPOS_ToF02_B1SRD20',
								'UnitTest1_LPOS_ToF02_B1SRD21', 'UnitTest1_LPOS_ToF02_B1SRD22',
								'UnitTest1_LPOS_ToF02_B1SRD23', 'UnitTest1_LPOS_ToF02_B1SRD24',
								'UnitTest1_LPOS_ToF02_B1SRD25', 'UnitTest1_LPOS_ToF02_B1SRD26',
								'UnitTest1_LPOS_ToF02_B1SRD27', 'UnitTest1_LPOS_ToF02_B1SRD28',
								'UnitTest1_LPOS_ToF02_B1SRD29', 'UnitTest1_LPOS_ToF02_B1SRD30',
								'UnitTest1_LPOS_ToF02_B1SRD31', 'UnitTest1_LPOS_ToF02_B1SRD32',
								'UnitTest1_LPOS_ToF02_B1SRD33', 'UnitTest1_LPOS_ToF02_B1SRD34',
								'UnitTest1_LPOS_ToF02_B1SRD35', 'UnitTest1_LPOS_ToF02_B1SRD36',
								'UnitTest1_LPOS_ToF02_B1SRD37', 'UnitTest1_LPOS_ToF02_B1SRD38',
								'UnitTest1_LPOS_ToF02_B1SRD39', 'UnitTest1_LPOS_ToF02_B1SRD40',
								'UnitTest1_LPOS_ToF02_B1SRD41', 'UnitTest1_LPOS_ToF02_B1SRD42',
								'UnitTest1_LPOS_ToF02_B1SRD43', 'UnitTest1_LPOS_ToF02_B1SRD44',
								'UnitTest1_LPOS_ToF02_B1SRD45', 'UnitTest1_LPOS_ToF02_B1SRD46',
								'UnitTest1_LPOS_ToF02_B1SRD47', 'UnitTest1_LPOS_ToF02_B1SRD48',
								'UnitTest1_LPOS_ToF02_B1SRD49', 'UnitTest1_LPOS_ToF02_B1SRD50',
								'UnitTest1_LPOS_ToF02_B1SRD51', 'UnitTest1_LPOS_ToF02_B1SRD52',
								'UnitTest1_LPOS_ToF02_B1SRD53', 'UnitTest1_LPOS_ToF02_B1SRD54',
								'UnitTest1_LPOS_ToF02_B1SRD55', 'UnitTest1_LPOS_ToF02_B1SRD56',
								'UnitTest1_LPOS_ToF02_B1SRD57', 'UnitTest1_LPOS_ToF02_B1SRD58',
								'UnitTest1_LPOS_ToF02_B1SRD59', 'UnitTest1_LPOS_ToF02_B1SRD60',
								'UnitTest1_LPOS_ToF02_B1SRD61', 'UnitTest1_LPOS_ToF02_B1SRD62',
								'UnitTest1_LPOS_ToF02_B1SRD63', 'UnitTest1_LPOS_ToF02_B1SRD64',
								'UnitTest1_LPOS_ToF02_B1SRD65', 'UnitTest1_LPOS_ToF02_B1SRD66',
								'UnitTest1_LPOS_ToF02_B1SRD67', 'UnitTest1_LPOS_ToF02_B1SRD68',
								'UnitTest1_LPOS_ToF02_B1SRD69', 'UnitTest1_LPOS_ToF02_B1SRD70',
								'UnitTest1_LPOS_ToF02_B1SRD71', 'UnitTest1_LPOS_ToF02_B1SRD72',
								'UnitTest1_LPOS_ToF02_B1SRD73', 'UnitTest1_LPOS_ToF02_B1SRD74',
								'UnitTest1_LPOS_ToF02_B1SRD75', 'UnitTest1_LPOS_ToF02_B1SRD76',
								'UnitTest1_LPOS_ToF02_B1SRD77', 'UnitTest1_LPOS_ToF02_B1SRD78',
								'UnitTest1_LPOS_ToF02_B1SRD79', 'UnitTest1_LPOS_ToF02_B1SRD80',
								'UnitTest1_LPOS_ToF02_B1SRD81', 'UnitTest1_LPOS_ToF02_B1SRD82',
								'UnitTest1_LPOS_ToF02_B1SRD83', 'UnitTest1_LPOS_ToF02_B1SRD84',
								'UnitTest1_LPOS_ToF02_B1SRD85', 'UnitTest1_LPOS_ToF02_B1SRD86',
								'UnitTest1_LPOS_ToF02_B1SRD87', 'UnitTest1_LPOS_ToF02_B1SRD88',
								'UnitTest1_LPOS_ToF02_B1SRD89', 'UnitTest1_LPOS_ToF02_B1SRD90',
								'UnitTest1_LPOS_ToF02_B1SRD91', 'UnitTest1_LPOS_ToF02_B1SRD92',
								'UnitTest1_LPOS_ToF02_B1E1_SR', 'UnitTest1_LPOS_ToF02_B1E2_SR',
								'UnitTest1_LPOS_ToF02_B1E3_SR', 'UnitTest1_LPOS_ToF02_B1E4_SR',
								'UnitTest1_LPOS_ToF02_B1E5_SR', 'UnitTest1_LPOS_ToF02_B1S1_SR',
								'UnitTest1_LPOS_ToF02_B1S2_SR', 'UnitTest1_LPOS_ToF02_B1S3_SR',
								'UnitTest1_LPOS_ToF02_B1S4_SR', 'UnitTest1_LPOS_ToF02_B1S5_SR',
								'UnitTest1_LPOS_ToF02_S1W01', 'UnitTest1_LPOS_ToF02_S1W02',
								'UnitTest1_LPOS_ToF02_S1W03', 'UnitTest1_LPOS_ToF02_S1W04',
								'UnitTest1_LPOS_ToF02_S1W05', 'UnitTest1_LPOS_ToF02_S1W06',
								'UnitTest1_LPOS_ToF02_S1W07', 'UnitTest1_LPOS_ToF02_S1W11_LTR',
								'UnitTest1_LPOS_ToF02_S1W12_SR'],
								name='Sample File Name',
								dtype=str)

		pandas.util.testing.assert_series_equal(self.msData.sampleMetadata['Sample File Name'], samples)
		pandas.util.testing.assert_series_equal(self.msData_PeakTable.sampleMetadata['Sample File Name'], samples)

	def test_featuremetadata_import(self):

		with self.subTest(msg='Checking Feature Names'):
			features = pandas.Series(['3.17_262.0378m/z',
									'3.17_293.1812m/z',
									'3.17_145.0686m/z',
									'3.17_258.1033m/z'],
									name='Feature Name',
									dtype='str')

			pandas.util.testing.assert_series_equal(self.msData.featureMetadata['Feature Name'], features)
			pandas.util.testing.assert_series_equal(self.msData_PeakTable.featureMetadata['Feature Name'], features)

		with self.subTest(msg='Checking m/z'):
			mz = pandas.Series([262.0378339,
							293.1811941,
							145.0686347,
							258.1033447],
							name='m/z',
							dtype='float')

			pandas.util.testing.assert_series_equal(self.msData.featureMetadata['m/z'], mz)
			pandas.util.testing.assert_series_equal(self.msData_PeakTable.featureMetadata['m/z'], mz)


		with self.subTest(msg='Checking Retention Time'):
			rt = pandas.Series([3.17485 / 60.0,
								3.17485 / 60.0,
								3.17485 / 60.0,
								3.17485 / 60.0],
								name='Retention Time',
								dtype='float')

			pandas.util.testing.assert_series_equal(self.msData.featureMetadata['Retention Time'], rt)
			pandas.util.testing.assert_series_equal(self.msData_PeakTable.featureMetadata['Retention Time'], rt)


	def test_dilutionlevels(self):

		dilution = pandas.Series([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 20., 20., 20., 20., 20.,
								40., 40., 40., 60., 60., 60., 80., 80., 80., 80., 80., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100.,
								1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 20., 20., 20., 20., 20.,
								40., 40., 40., 60., 60., 60., 80., 80., 80., 80., 80., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100.,
								numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan,
								numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan], 
								name='Dilution',
								dtype='float')

		pandas.util.testing.assert_series_equal(self.msData.sampleMetadata['Dilution'], dilution)
		pandas.util.testing.assert_series_equal(self.msData_PeakTable.sampleMetadata['Dilution'], dilution)


	def test_feature_correlation(self):

		self.msData.addSampleInfo(descriptionFormat='Raw Data', filePath=os.path.join('..','..','npc-standard-project','Raw_Data'))
		self.msData.addSampleInfo(descriptionFormat='Batches')

		self.msData_PeakTable.addSampleInfo(descriptionFormat='Raw Data', filePath=os.path.join('..','..','npc-standard-project','Raw_Data'))
		self.msData_PeakTable.addSampleInfo(descriptionFormat='Batches')

		with self.subTest(msg='Testing Pearson correlations'):
			correlations = numpy.array([0.99999997, 0.32017508, 1., -0.0693418])

			numpy.testing.assert_array_almost_equal(self.msData.correlationToDilution, correlations)
			numpy.testing.assert_array_almost_equal(self.msData_PeakTable.correlationToDilution, correlations)

		with self.subTest(msg='Testing Spearman correlations'):
			correlations = numpy.array([0.9992837, 0.34708745, 1., -0.038844])

			self.msData.Attributes['corrMethod'] = 'spearman'
			self.msData_PeakTable.Attributes['corrMethod'] = 'spearman'

			numpy.testing.assert_array_almost_equal(self.msData.correlationToDilution, correlations)
			numpy.testing.assert_array_almost_equal(self.msData_PeakTable.correlationToDilution, correlations)


	def test_variabletype(self):

		self.assertEqual(self.msData.VariableType, nPYc.enumerations.VariableType.Discrete)
		self.assertEqual(self.msData_PeakTable.VariableType, nPYc.enumerations.VariableType.Discrete)


class test_msdataset_import_biocrates(unittest.TestCase):
	"""
	Test import of Biocrate sheets
	"""

	def setUp(self):

		self.msData = nPYc.MSDataset(os.path.join('..','..','npc-standard-project','Derived_Data','UnitTest1_PCSOP.069_Biocrates.xlsx'), fileType='Biocrates', sheetName='Master Samples')


	def test_dimensions(self):

		self.assertEqual((self.msData.noSamples, self.msData.noFeatures), (9, 144))


	def test_samples(self):

		with self.subTest(msg='Checking Sampling IDs'):
			samples = pandas.Series(['UnitTest1_LPOS_ToF02_S1W01', 'UnitTest1_LPOS_ToF02_S1W02',
									'UnitTest1_LPOS_ToF02_S1W03', 'UnitTest1_LPOS_ToF02_S1W04',
									'UnitTest1_LPOS_ToF02_S1W05', 'UnitTest1_LPOS_ToF02_S1W06',
									'UnitTest1_LPOS_ToF02_S1W07', 'UnitTest1_LPOS_ToF02_S1W11_LTR',
									'UnitTest1_LPOS_ToF02_S1W12_SR'],
									name='Sampling ID',
									dtype=str)

			pandas.util.testing.assert_series_equal(self.msData.sampleMetadata['Sampling ID'], samples)

		with self.subTest(msg='Checking Sample Bar Code'):
			samples = pandas.Series([1010751983, 1010751983, 1010751983, 1010751983, 1010751983, 1010751998, 1010751998, 1010751998, 1010751998],
									name='Sample Bar Code',
									dtype=int)

			pandas.util.testing.assert_series_equal(self.msData.sampleMetadata['Sample Bar Code'], samples)


	def test_featuremetadata_import(self):

		with self.subTest(msg='Checking Feature Names'):
			features = pandas.Series(['C0', 'C10', 'C10:1', 'C10:2', 'C12', 'C12-DC', 'C12:1', 'C14', 'C14:1', 'C14:1-OH', 'C14:2', 'C14:2-OH', 'C16', 'C16-OH',
									'C16:1', 'C16:1-OH', 'C16:2', 'C16:2-OH', 'C18', 'C18:1', 'C18:1-OH', 'C18:2', 'C2', 'C3', 'C3-DC (C4-OH)', 'C3-OH', 'C3:1',
									'C4', 'C4:1', 'C6 (C4:1-DC)', 'C5', 'C5-M-DC', 'C5-OH (C3-DC-M)', 'C5:1', 'C5:1-DC', 'C5-DC (C6-OH)', 'C6:1', 'C7-DC', 'C8',
									'C9', 'lysoPC a C14:0', 'lysoPC a C16:0', 'lysoPC a C16:1', 'lysoPC a C17:0', 'lysoPC a C18:0', 'lysoPC a C18:1', 'lysoPC a C18:2',
									'lysoPC a C20:3', 'lysoPC a C20:4', 'lysoPC a C24:0', 'lysoPC a C26:0', 'lysoPC a C26:1', 'lysoPC a C28:0', 'lysoPC a C28:1',
									'PC aa C24:0', 'PC aa C26:0', 'PC aa C28:1', 'PC aa C30:0', 'PC aa C32:0', 'PC aa C32:1', 'PC aa C32:2', 'PC aa C32:3', 'PC aa C34:1',
									'PC aa C34:2', 'PC aa C34:3', 'PC aa C34:4', 'PC aa C36:0', 'PC aa C36:1', 'PC aa C36:2', 'PC aa C36:3', 'PC aa C36:4', 'PC aa C36:5',
									'PC aa C36:6', 'PC aa C38:0', 'PC aa C38:3', 'PC aa C38:4', 'PC aa C38:5', 'PC aa C38:6', 'PC aa C40:1', 'PC aa C40:2', 'PC aa C40:3',
									'PC aa C40:4', 'PC aa C40:5', 'PC aa C40:6', 'PC aa C42:0', 'PC aa C42:1', 'PC aa C42:2', 'PC aa C42:4', 'PC aa C42:5', 'PC aa C42:6',
									'PC ae C30:0', 'PC ae C30:1', 'PC ae C30:2', 'PC ae C32:1', 'PC ae C32:2', 'PC ae C34:0', 'PC ae C34:1', 'PC ae C34:2', 'PC ae C34:3',
									'PC ae C36:0', 'PC ae C36:1', 'PC ae C36:2', 'PC ae C36:3', 'PC ae C36:4', 'PC ae C36:5', 'PC ae C38:0', 'PC ae C38:1', 'PC ae C38:2',
									'PC ae C38:3', 'PC ae C38:4', 'PC ae C38:5', 'PC ae C38:6', 'PC ae C40:1', 'PC ae C40:2', 'PC ae C40:3', 'PC ae C40:4', 'PC ae C40:5',
									'PC ae C40:6', 'PC ae C42:0', 'PC ae C42:1', 'PC ae C42:2', 'PC ae C42:3', 'PC ae C42:4', 'PC ae C42:5', 'PC ae C44:3', 'PC ae C44:4',
									'PC ae C44:5', 'PC ae C44:6', 'SM (OH) C14:1', 'SM (OH) C16:1', 'SM (OH) C22:1', 'SM (OH) C22:2', 'SM (OH) C24:1', 'SM C16:0',
									'SM C16:1', 'SM C18:0', 'SM C18:1', 'SM C20:2', 'SM C24:0', 'SM C24:1', 'SM C26:0', 'SM C26:1', 'H1', 'H1.1'],
									name='Feature Name',
									dtype=str)

			pandas.util.testing.assert_series_equal(self.msData.featureMetadata['Feature Name'], features)

		with self.subTest(msg='Class'):
			classField = pandas.Series(['acylcarnitines', 'acylcarnitines', 'acylcarnitines', 'acylcarnitines', 'acylcarnitines', 'acylcarnitines',
									'acylcarnitines', 'acylcarnitines', 'acylcarnitines', 'acylcarnitines', 'acylcarnitines', 'acylcarnitines',
									'acylcarnitines', 'acylcarnitines', 'acylcarnitines', 'acylcarnitines', 'acylcarnitines', 'acylcarnitines',
									'acylcarnitines', 'acylcarnitines', 'acylcarnitines', 'acylcarnitines', 'acylcarnitines', 'acylcarnitines',
									'acylcarnitines', 'acylcarnitines', 'acylcarnitines', 'acylcarnitines', 'acylcarnitines', 'acylcarnitines',
									'acylcarnitines', 'acylcarnitines', 'acylcarnitines', 'acylcarnitines', 'acylcarnitines', 'acylcarnitines',
									'acylcarnitines', 'acylcarnitines', 'acylcarnitines', 'acylcarnitines', 'glycerophospholipids', 'glycerophospholipids',
									'glycerophospholipids', 'glycerophospholipids', 'glycerophospholipids', 'glycerophospholipids',
									'glycerophospholipids', 'glycerophospholipids', 'glycerophospholipids', 'glycerophospholipids',
									'glycerophospholipids', 'glycerophospholipids', 'glycerophospholipids', 'glycerophospholipids',
									'glycerophospholipids', 'glycerophospholipids', 'glycerophospholipids', 'glycerophospholipids',
									'glycerophospholipids', 'glycerophospholipids', 'glycerophospholipids', 'glycerophospholipids',
									'glycerophospholipids', 'glycerophospholipids', 'glycerophospholipids', 'glycerophospholipids',
									'glycerophospholipids', 'glycerophospholipids', 'glycerophospholipids', 'glycerophospholipids',
									'glycerophospholipids', 'glycerophospholipids', 'glycerophospholipids', 'glycerophospholipids',
									'glycerophospholipids', 'glycerophospholipids', 'glycerophospholipids', 'glycerophospholipids',
									'glycerophospholipids', 'glycerophospholipids', 'glycerophospholipids', 'glycerophospholipids',
									'glycerophospholipids', 'glycerophospholipids', 'glycerophospholipids', 'glycerophospholipids',
									'glycerophospholipids', 'glycerophospholipids', 'glycerophospholipids', 'glycerophospholipids',
									'glycerophospholipids', 'glycerophospholipids', 'glycerophospholipids', 'glycerophospholipids',
									'glycerophospholipids', 'glycerophospholipids', 'glycerophospholipids', 'glycerophospholipids',
									'glycerophospholipids', 'glycerophospholipids', 'glycerophospholipids', 'glycerophospholipids',
									'glycerophospholipids', 'glycerophospholipids', 'glycerophospholipids', 'glycerophospholipids',
									'glycerophospholipids', 'glycerophospholipids', 'glycerophospholipids', 'glycerophospholipids',
									'glycerophospholipids', 'glycerophospholipids', 'glycerophospholipids', 'glycerophospholipids',
									'glycerophospholipids', 'glycerophospholipids', 'glycerophospholipids', 'glycerophospholipids',
									'glycerophospholipids', 'glycerophospholipids', 'glycerophospholipids', 'glycerophospholipids',
									'glycerophospholipids', 'glycerophospholipids', 'glycerophospholipids', 'glycerophospholipids',
									'glycerophospholipids', 'glycerophospholipids', 'sphingolipids', 'sphingolipids', 'sphingolipids', 'sphingolipids', 'sphingolipids',
									'sphingolipids', 'sphingolipids', 'sphingolipids', 'sphingolipids', 'sphingolipids', 'sphingolipids', 'sphingolipids', 'sphingolipids',
									'sphingolipids', 'sugars', 'sugars'],
								name='Class',
								dtype=str)

			pandas.util.testing.assert_series_equal(self.msData.featureMetadata['Class'], classField)

		with self.subTest(msg='Checking LOD'):
			lod = pandas.Series([2.1, 0.08, 1.08, 0.156, 0.064, 0.151, 0.857, 0.023, 0.009, 0.015, 0.049, 0.019, 0.018, 0.009, 0.017, 0.029, 0.023, 0.035,
								0.013, 0.029, 0.017, 0.01, 0.063, 0.011, 0.046, 0.02, 0., 0.027, 0.021, 0.02, 0.035, 0.05, 0.037, 0.072, 0.015, 0.014, 0.036,
								0.018, 0.1, 0.017, 5.32, 0.068, 0.064, 0.035, 0.181, 0.023, 0.02, 0.088, 0., 0.038, 0.034, 0.015, 0.105, 0.007, 0.061, 1.1,
								0.079, 0.139, 0.02, 0.006, 0.006, 0., 0.03, 0.015, 0.001, 0.004, 0.203, 0.012, 0.022, 0.004, 0.009, 0.004, 0.002, 0.035, 0.01,
								0.008, 0.005, 0.002, 0.394, 0.058, 0.003, 0.017, 0., 0.188, 0.065, 0.019, 0.058, 0.011, 0.037, 0.248, 0.155, 0.005, 0.01,
								0.002, 0.001, 0.011, 0.014, 0.004, 0.01, 0.059, 0.061, 0.029, 0., 0.084, 0.014, 0.076, 0.031, 0.012, 0.005, 0.009, 0.002,
								0.003, 0.019, 0.006, 0.013, 0.08, 0.003, 0.007, 1.32, 0.119, 0.017, 0.007, 0., 0.843, 0.048, 0.116, 0.072, 0.043, 0., 0.004,
								0.006, 0.001, 0., 0.032, 0.005, 0.003, 0.004, 0.013, 0.006, 0.003, 0.01, 0.003, 912., 912.],
							name='LOD (μM)',
							dtype=float)

			pandas.util.testing.assert_series_equal(self.msData.featureMetadata['LOD (μM)'], lod)


	def test_variabletype(self):

		self.assertEqual(self.msData.VariableType, nPYc.enumerations.VariableType.Discrete)


class test_msdataset_addsampleinfo(unittest.TestCase):
	"""
	Test import from QI csv files
	"""

	def setUp(self):

		self.msData = nPYc.MSDataset(os.path.join('..','..','npc-standard-project','Derived_Data','UnitTest1_PCSOP.069_QI.csv'), fileType='QI')
		self.msData.addSampleInfo(descriptionFormat='Filenames')


	def test_msdataset_load_npc_lims(self):
		"""
		Test we are matching samples IDs in the LIMS correctly
		"""

		samplingIDs = pandas.Series(['Study Pool Sample', 'Study Pool Sample', 'Study Pool Sample', 'Study Pool Sample', 'Study Pool Sample',
									 'Study Pool Sample', 'Study Pool Sample', 'Study Pool Sample', 'Study Pool Sample', 'Study Pool Sample',
									 'Study Pool Sample', 'Study Pool Sample', 'Study Pool Sample', 'Study Pool Sample', 'Study Pool Sample',
									 'Study Pool Sample', 'Study Pool Sample', 'Study Pool Sample', 'Study Pool Sample', 'Study Pool Sample',
									 'Study Pool Sample', 'Study Pool Sample', 'Study Pool Sample', 'Study Pool Sample', 'Study Pool Sample',
									 'Study Pool Sample', 'Study Pool Sample', 'Study Pool Sample', 'Study Pool Sample', 'Study Pool Sample',
									 'Study Pool Sample', 'Study Pool Sample', 'Study Pool Sample', 'Study Pool Sample', 'Study Pool Sample',
									 'Study Pool Sample', 'Study Pool Sample', 'Study Pool Sample', 'Study Pool Sample', 'Study Pool Sample',
									 'Study Pool Sample', 'Study Pool Sample', 'Study Pool Sample', 'Study Pool Sample', 'Study Pool Sample',
									 'Study Pool Sample', 'Study Pool Sample', 'Study Pool Sample', 'Study Pool Sample', 'Study Pool Sample',
									 'Study Pool Sample', 'Study Pool Sample', 'Study Pool Sample', 'Study Pool Sample', 'Study Pool Sample',
									 'Study Pool Sample', 'Study Pool Sample', 'Study Pool Sample', 'Study Pool Sample', 'Study Pool Sample',
									 'Study Pool Sample', 'Study Pool Sample', 'Study Pool Sample', 'Study Pool Sample', 'Study Pool Sample',
									 'Study Pool Sample', 'Study Pool Sample', 'Study Pool Sample', 'Study Pool Sample', 'Study Pool Sample',
									 'Study Pool Sample', 'Study Pool Sample', 'Study Pool Sample', 'Study Pool Sample', 'Study Pool Sample',
									 'Study Pool Sample', 'Study Pool Sample', 'Study Pool Sample', 'Study Pool Sample', 'Study Pool Sample',
									 'Study Pool Sample', 'Study Pool Sample', 'Study Pool Sample', 'Study Pool Sample', 'Study Pool Sample',
									 'Study Pool Sample', 'Study Pool Sample', 'Study Pool Sample', 'Study Pool Sample', 'Study Pool Sample',
									 'Study Pool Sample', 'Study Pool Sample', 'Procedural Blank Sample', 'Procedural Blank Sample',
									 'Study Pool Sample','Study Pool Sample', 'Study Pool Sample', 'Study Pool Sample', 'Study Pool Sample',
									 'Study Pool Sample','Study Pool Sample', 'Study Pool Sample', 'Study Pool Sample', 'Study Pool Sample',
									 'UT1_S1_s1', 'UT1_S2_s1', 'UT1_S3_s1', 'Not specified', 'UT1_S4_s2', 'UT1_S4_s3', 'UT1_S4_s4', 'UT1_S4_s5',
									 'External Reference Sample', 'Study Pool Sample', 'Not specified'], name='Sampling ID', dtype='str')
		samplingIDs = samplingIDs.astype(str)

		self.msData.addSampleInfo(descriptionFormat='NPC LIMS', filePath=os.path.join('..','..','npc-standard-project','Derived_Worklists','UnitTest1_MS_serum_PCSOP.069.csv'))
		pandas.util.testing.assert_series_equal(self.msData.sampleMetadata['Sampling ID'], samplingIDs)


	def test_msdataset_load_watersraw_metadata(self):
		"""
		Test we read raw data from Waters .raw and concatenate it correctly - currently focusing on parameters of importance to the workflow.
		
		TODO: Test all paramaters
		"""
		
		# 

		# Expected data starts with the same samples 
		expected = copy.deepcopy(self.msData.sampleMetadata)

		##
		# Define a test subset of columns with one unique value
		##
		testSeries = ['Sampling Cone', 'Scan Time (sec)', 'Source Offset', 'Source Temperature (°C)', 'Start Mass', 'End Mass', 'Column Serial Number:', 'ColumnType:']
		expected['Sampling Cone'] = 20.0
		expected['Scan Time (sec)']  = 0.15
		expected['Source Offset'] = 80.0
		expected['Source Temperature (°C)'] = 120.0
		expected['Start Mass'] = 50.0
		expected['End Mass'] = 1200.0
		expected['Column Serial Number:'] = 1573413615729.
		expected['ColumnType:'] = 'ACQUITY UPLC® HSS T3 1.8µm'

		##
		# And a subset with multiple values
		##
		testSeries.append('Detector')
		expected['Detector'] = [3161., 3161., 3166., 3166., 3166., 3166., 3171., 3171., 3171., 3171., 3171., 3171., 3171., 3171., 3179.,
								3179., 3179., 3179., 3179., 3179., 3184., 3184., 3184., 3188., 3188., 3188., 3188., 3188., 3188., 3193.,
								3193., 3193., 3193., 3193., 3197., 3197., 3197., 3197., 3197., 3203., 3203., 3203., 3203., 3203., 3208.,
								3208., 3407., 3407., 3407., 3407., 3407., 3407., 3407., 3407., 3407., 3407., 3407., 3407., 3407., 3407.,
								3407., 3407., 3407., 3407., 3407., 3407., 3407., 3407., 3399., 3399., 3399., 3399., 3399., 3399., 3399.,
								3399., 3399., 3399., 3399., 3399., 3399., 3399., 3399., 3399., 3399., 3399., 3399., 3399., 3399., 3399.,
								3399., 3399., 3399., 3399., 3399., 3399., 3399., 3399., 3399., 3212., 3212., 3217., 3217., 3217., 3293.,
								3293., 3293., 3299., 3299., 3299., 3299., 3299., 3293., 3299., 3299.]

		testSeries.append('Measurement Date')
		expected['Measurement Date'] = ['25-Nov-2014', '25-Nov-2014', '25-Nov-2014', '25-Nov-2014', '25-Nov-2014', '25-Nov-2014', '25-Nov-2014', '25-Nov-2014', '25-Nov-2014',
						'25-Nov-2014', '25-Nov-2014', '25-Nov-2014', '25-Nov-2014', '25-Nov-2014', '25-Nov-2014', '25-Nov-2014', '25-Nov-2014', '25-Nov-2014',
						'25-Nov-2014', '25-Nov-2014', '25-Nov-2014', '25-Nov-2014', '25-Nov-2014', '25-Nov-2014', '25-Nov-2014', '25-Nov-2014', '25-Nov-2014',
						'25-Nov-2014', '25-Nov-2014', '25-Nov-2014', '25-Nov-2014', '25-Nov-2014', '25-Nov-2014', '25-Nov-2014', '25-Nov-2014', '25-Nov-2014',
						'25-Nov-2014', '25-Nov-2014', '25-Nov-2014', '26-Nov-2014', '26-Nov-2014', '26-Nov-2014', '26-Nov-2014', '26-Nov-2014', '26-Nov-2014',
						'26-Nov-2014', '30-Nov-2014', '30-Nov-2014', '30-Nov-2014', '30-Nov-2014', '30-Nov-2014', '30-Nov-2014', '30-Nov-2014', '30-Nov-2014',
						'30-Nov-2014', '30-Nov-2014', '30-Nov-2014', '30-Nov-2014', '30-Nov-2014', '30-Nov-2014', '30-Nov-2014', '30-Nov-2014', '30-Nov-2014',
						'30-Nov-2014', '30-Nov-2014', '30-Nov-2014', '30-Nov-2014', '30-Nov-2014', '30-Nov-2014', '30-Nov-2014', '30-Nov-2014', '30-Nov-2014',
						'30-Nov-2014', '30-Nov-2014', '30-Nov-2014', '30-Nov-2014', '30-Nov-2014', '30-Nov-2014', '30-Nov-2014', '30-Nov-2014', '30-Nov-2014',
						'30-Nov-2014', '30-Nov-2014', '30-Nov-2014', '30-Nov-2014', '30-Nov-2014', '30-Nov-2014', '30-Nov-2014', '30-Nov-2014', '30-Nov-2014',
						'30-Nov-2014', '30-Nov-2014', '24-Nov-2014', '24-Nov-2014', '30-Nov-2014', '30-Nov-2014', '30-Nov-2014', '30-Nov-2014', '30-Nov-2014',
						'26-Nov-2014', '26-Nov-2014', '26-Nov-2014', '26-Nov-2014', '26-Nov-2014', '27-Nov-2014', '27-Nov-2014', '27-Nov-2014', '27-Nov-2014',
						'27-Nov-2014', '27-Nov-2014', '27-Nov-2014', '27-Nov-2014', '27-Nov-2014', '27-Nov-2014', '27-Nov-2014']

		testSeries.append('Measurement Time')
		expected['Measurement Time'] = ['13:43:57', '13:59:44', '14:15:39', '14:31:26', '14:47:21', '15:03:07', '15:19:00', '15:34:46', '15:50:40', '16:06:26',
										 '16:22:12', '16:38:06', '16:54:01', '17:09:56', '17:25:44', '17:41:30', '17:57:16', '18:13:02', '18:28:47', '18:44:35',
										 '19:00:22', '19:16:10', '19:31:56', '19:47:51', '20:03:46', '20:19:33', '20:35:19', '20:51:13', '21:07:09', '21:22:55',
										 '21:38:50', '21:54:43', '22:10:28', '22:26:15', '22:42:09', '22:57:56', '23:13:41', '23:29:35', '23:45:29', '00:01:23',
										 '00:17:10', '00:32:56', '00:48:49', '01:04:33', '01:20:20', '01:36:06', '19:53:55', '19:38:18', '19:22:41', '19:07:03',
										 '18:51:23', '18:35:46', '18:20:06', '18:04:29', '17:48:57', '17:33:20', '17:17:42', '17:02:05', '16:46:27', '16:30:57',
										 '16:15:18', '15:59:40', '15:44:03', '15:28:24', '15:12:48', '14:57:10', '14:41:33', '14:25:55', '14:10:24', '13:54:46',
										 '13:39:08', '13:23:38', '13:08:08', '12:52:30', '12:36:50', '12:21:13', '12:05:41', '11:50:03', '11:34:25', '11:18:55',
										 '11:03:25', '10:47:55', '10:32:18', '10:16:40', '10:01:10', '09:45:32', '09:30:01', '09:14:25', '08:58:53', '08:43:23',
										 '08:27:47', '08:12:10', '08:12:47', '08:25:10', '06:52:08', '07:09:38', '07:25:16', '07:40:52', '07:56:32', '02:39:17',
										 '02:55:03', '03:10:49', '03:26:43', '03:42:35', '12:11:04', '12:26:51', '12:42:35', '12:58:13', '13:14:01', '13:45:26',
										 '14:01:05', '14:16:51', '11:53:27', '13:29:48', '13:46:48']

		self.msData.addSampleInfo(descriptionFormat='Raw Data', filePath=os.path.join('..', '..', 'npc-standard-project', 'Raw_Data', 'ms', 'parameters_data'))

		for series in testSeries:
			pandas.util.testing.assert_series_equal(self.msData.sampleMetadata[series], expected[series])


	def test_msdataset__getSampleMetadataFromRawData_invalidpath(self):

		self.assertRaises(ValueError, self.msData.addSampleInfo, descriptionFormat='Raw Data', filePath='./NOT_A_REAL_PATH')


class test_msdataset_artifactual_filtering(unittest.TestCase):
	"""
	Test calculation and update of artifactual filtering results
	
	Simulated data design:
		14 features, 10 samples
		normal threshold | 0.1mz, 0.9corr, 50% ov
		advanced threshold | 0.05mz, 0.95 corr, 70% ov
		
		0 and 1, Nothing (corr=0, 10. mz, ov 0%)
		1 and 2, corr>0.9, 10. mz, ov 80% | fail default param
		3 and 4, corr<0.3, 0.005mz, ov 60% | fail default corr, present in ._temp not in ._arti
		5 and 6, corr 0.91, 0.01mz, ov 90% | ok default, advance fails corr (present in ._temp, not in ._arti) (5 is higher intensity)
		7 and 8, corr 0.97, 0.09mz, ov 90% | ok default, then fail mz (7 is higher intensity)
		9 and 10, corr 0.97, 0.01mz, ov 55% | ok default, then fail overlap (9 is higher intensity)
		11 and 12, corr 0.97, 0.01mz, ov 90% | ok default, then still okay (11 is higher intensity)
		13, 14, 15, 16, all ov > 92.5%, close mz, corr | ok default, then still okay (13 is higher intensity)
	
	Test:
		1) compare output .artifactualLinkageMatrix, ._artifactualLinkageMatrix, ._tempArtifactualLinkageMatrix
			test .artifactualFilter() results with and without a given featureMask
		2) ensure that deepcopy resets ._arti, ._temp
		3) change deltaMzArtifactual, overlapThresholdArtifactual, corrThresholdArtifactual (changes in .artifactualLinkageMatrix, ._artifactualLinkageMatrix and ._tempArtifactualLinkageMatrix)
		4) applyMask: remove samples -> only corr, remove features -> all
	"""

	def setUp(self):
		self.msData = nPYc.MSDataset(
			os.path.join('..', '..', 'npc-standard-project', 'Derived_Data', 'UnitTest2_artifactualFiltering.csv'),
			fileType='QI')
		self.msData.Attributes['artifactualFilter'] = True
		self.msData.addSampleInfo(descriptionFormat='Filenames')


	def test_artifactualFilter_raise(self):
		with self.subTest(msg='Attributes artifactualFilter is False'):
			partialMsData = copy.deepcopy(self.msData)
			partialMsData.Attributes['artifactualFilter'] = False
			self.assertRaises(ValueError,partialMsData._MSDataset__generateArtifactualLinkageMatrix)
		with self.subTest(msg='Missing Feature Name'):
			partialMsData1 = copy.deepcopy(self.msData)
			partialMsData1.featureMetadata.drop('Feature Name', axis=1, inplace=True)
			self.assertRaises(LookupError,partialMsData1._MSDataset__generateArtifactualLinkageMatrix)
		with self.subTest(msg='Missing Retention Time'):
			partialMsData2 = copy.deepcopy(self.msData)
			partialMsData2.featureMetadata.drop('Retention Time', axis=1, inplace=True)
			self.assertRaises(LookupError,partialMsData2._MSDataset__generateArtifactualLinkageMatrix)
		with self.subTest(msg='Missing m/z'):
			partialMsData3 = copy.deepcopy(self.msData)
			partialMsData3.featureMetadata.drop('m/z', axis=1, inplace=True)
			self.assertRaises(LookupError,partialMsData3._MSDataset__generateArtifactualLinkageMatrix)
		with self.subTest(msg='Missing Peak Width'):
			partialMsData4 = copy.deepcopy(self.msData)
			partialMsData4.featureMetadata.drop('Peak Width', axis=1, inplace=True)
			self.assertRaises(LookupError,partialMsData4._MSDataset__generateArtifactualLinkageMatrix)


	def test_artifactualFilter(self):
		"""
		Test artifactualFilter() and corresponding functions and variables.
		"""
		##
		# _artifactualLinkageMatrix, artifactualLinkageMatrix and __generateArtifactualLinkageMatrix()
		##
		result_artifactualLinkageMatrix = pandas.DataFrame(
			[[5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [13, 15], [13, 16], [14, 15], [14, 16], [15, 16]],
			index=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], columns=['node1', 'node2'])
		assert_frame_equal(self.msData.artifactualLinkageMatrix, result_artifactualLinkageMatrix)
		assert_frame_equal(self.msData._artifactualLinkageMatrix, result_artifactualLinkageMatrix)
		
		## _tempArtifactualLinkageMatrix (not filtered by correlation)
		result_tempArtifactualLinkageMatrix = pandas.DataFrame(
			[[3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [13, 15], [13, 16], [14, 15], [14, 16], [15, 16]],
			index=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], columns=['node1', 'node2'])
		assert_frame_equal(self.msData._tempArtifactualLinkageMatrix, result_tempArtifactualLinkageMatrix)
		
		## artifactualFilter()
		# default msData.featureMask
		result_artifactualFilter = numpy.array(
			[True, True, True, True, True, True, False, True, False, True, False, True, False, True, False, False,
			 False], dtype=bool)
		numpy.testing.assert_equal(self.msData.artifactualFilter(), result_artifactualFilter)
		
		# with featureMask excluding feature 0, 1 and 2
		result_artifactualFilter_featMask = numpy.array(
			[False, False, False, True, True, True, False, True, False, True, False, True, False, True, False, False,
			 False], dtype=bool)
		numpy.testing.assert_equal(self.msData.artifactualFilter(numpy.array(
			[False, False, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True],
			dtype=bool)), result_artifactualFilter_featMask)


	def test_deepcopy_artifactualFilter(self):
		"""
		Ensure variables necessary to artifactual filtering are reset when a deepcopy is employed
		"""
		##
		# deepcopy
		##
		self.msData2 = copy.deepcopy(self.msData)
		self.assertTrue(self.msData2._tempArtifactualLinkageMatrix.empty,
						msg='_tempArtifactualLinkageMatrix hasnt been reset by deepcopy')
		self.assertTrue(self.msData2._artifactualLinkageMatrix.empty,
						msg='_artifactualLinkageMatrix hasnt been reset by deepcopy')


	def test_deleter_artifactualFilter(self):
		"""
		Ensure variables necessary to artifactual filtering are reset when a deepcopy is employed
		"""

		del self.msData.artifactualLinkageMatrix
		self.assertTrue(self.msData._tempArtifactualLinkageMatrix.empty,
						msg='_tempArtifactualLinkageMatrix hasnt been reset by delete')
		self.assertTrue(self.msData._artifactualLinkageMatrix.empty,
						msg='_artifactualLinkageMatrix hasnt been reset by delete')


	def test_artifactualFilter_parameterChange(self):
		"""
		Test artifactual filtering parameter alteration.
		"""
		##
		# Change artifactualThresholds & update
		##
		self.msData3 = copy.deepcopy(self.msData)
		self.msData3.Attributes['deltaMzArtifactual'] = 0.05
		self.msData3.Attributes['overlapThresholdArtifactual'] = 70
		self.msData3.Attributes['corrThresholdArtifactual'] = 0.95
		self.msData3.updateArtifactualLinkageMatrix()
		
		## _artifactualLinkageMatrix, artifactualLinkageMatrix
		updatedRes_artifactualLinkageMatrix = pandas.DataFrame(
			[[11, 12], [13, 14], [13, 15], [13, 16], [14, 15], [14, 16], [15, 16]], index=[1, 2, 3, 4, 5, 6, 7],
			columns=['node1', 'node2'])
		assert_frame_equal(self.msData3.artifactualLinkageMatrix, updatedRes_artifactualLinkageMatrix)
		assert_frame_equal(self.msData3._artifactualLinkageMatrix, updatedRes_artifactualLinkageMatrix)
		
		## _tempArtifactualLinkageMatrix (not filtered by correlation)
		updatedRes_tempArtifactualLinkageMatrix = pandas.DataFrame(
			[[5, 6], [11, 12], [13, 14], [13, 15], [13, 16], [14, 15], [14, 16], [15, 16]],
			index=[0, 1, 2, 3, 4, 5, 6, 7], columns=['node1', 'node2'])
		assert_frame_equal(self.msData3._tempArtifactualLinkageMatrix, updatedRes_tempArtifactualLinkageMatrix)
		
		## artifactualFilter()
		# default self.featureMask
		updatedRes_artifactualFilter = numpy.array(
			[True, True, True, True, True, True, True, True, True, True, True, True, False, True, False, False, False],
			dtype=bool)
		numpy.testing.assert_equal(self.msData3.artifactualFilter(), updatedRes_artifactualFilter)
		
		# with featureMask excluding feature 0, 1 and 2
		updatedRes_artifactualFilter_featMask = numpy.array(
			[False, False, False, True, True, True, True, True, True, True, True, True, False, True, False, False,
			 False], dtype=bool)
		numpy.testing.assert_equal(self.msData3.artifactualFilter(numpy.array(
			[False, False, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True],
			dtype=bool)), updatedRes_artifactualFilter_featMask)


	def test_applyMasks_artifactualFilter(self):
		"""
		Ensure artifactualLinkageMatrix is updated when sample or feature masks are modified
		"""
		##
		# sampleMask, remove sample 1 2 3 4 5
		##
		self.msData4 = copy.deepcopy(self.msData)
		self.msData4.artifactualLinkageMatrix
		self.msData4.sampleMask = numpy.array([False, False, False, False, False, True, True, True, True, True],
											  dtype=bool)
		self.msData4.applyMasks()
		
		# _artifactualLinkageMatrix, artifactualLinkageMatrix are modified
		sampleMaskRes_artifactualLinkageMatrix = pandas.DataFrame([[14, 15], [14, 16], [15, 16]], index=[8, 9, 10],
																  columns=['node1', 'node2'])
		assert_frame_equal(self.msData4.artifactualLinkageMatrix, sampleMaskRes_artifactualLinkageMatrix)
		assert_frame_equal(self.msData4._artifactualLinkageMatrix, sampleMaskRes_artifactualLinkageMatrix)
		
		# _tempArtifactualLinkageMatrix (not filtered by correlation) is not changed
		sampleMaskRes_tempArtifactualLinkageMatrix = pandas.DataFrame(
			[[3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [13, 15], [13, 16], [14, 15], [14, 16], [15, 16]],
			index=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], columns=['node1', 'node2'])
		assert_frame_equal(self.msData4._tempArtifactualLinkageMatrix, sampleMaskRes_tempArtifactualLinkageMatrix)
		
		##
		# featureMask, remove sample 12 15 16
		##
		self.msData5 = copy.deepcopy(self.msData)
		self.msData5.artifactualLinkageMatrix
		self.msData5.featureMask = numpy.array([True,True,True,True,True,True,True,True,True,True,True,True,False,True,True,False,False],dtype=bool)
		self.msData5.applyMasks()
		
		# _artifactualLinkageMatrix, artifactualLinkageMatrix are modified
		featMaskRes_artifactualLinkageMatrix = pandas.DataFrame([[5, 6], [7, 8], [9, 10], [12, 13]], index=[1, 2, 3, 4],
																columns=['node1', 'node2'])
		assert_frame_equal(self.msData5.artifactualLinkageMatrix, featMaskRes_artifactualLinkageMatrix)
		assert_frame_equal(self.msData5._artifactualLinkageMatrix, featMaskRes_artifactualLinkageMatrix)
		
		# _tempArtifactualLinkageMatrix (not filtered by correlation) is changed
		featMaskRes_tempArtifactualLinkageMatrix = pandas.DataFrame([[3, 4], [5, 6], [7, 8], [9, 10], [12, 13]],
																	index=[0, 1, 2, 3, 4], columns=['node1', 'node2'])
		assert_frame_equal(self.msData5._tempArtifactualLinkageMatrix, featMaskRes_tempArtifactualLinkageMatrix)


class test_msdataset_ISATAB(unittest.TestCase):

	def test_exportISATAB(self):
		import tempfile

		msData = nPYc.MSDataset('', fileType='empty')
		raw_data = {
			'Acquired Time': ['09/08/2016  01:36:23', '09/08/2016  01:56:23', '09/08/2016  02:16:23', '09/08/2016  02:36:23', '09/08/2016  02:56:23'],
			'AssayRole': ['AssayRole.LinearityReference', 'AssayRole.LinearityReference', 'AssayRole.LinearityReference', 'AssayRole.Assay', 'AssayRole.Assay'],
			'SampleType': ['SampleType.StudyPool', 'SampleType.StudyPool', 'SampleType.StudyPool', 'SampleType.StudySample', 'SampleType.StudySample'],
			'Subject ID': ['', '', '', 'SCANS-120', 'SCANS-130'],
			'Sampling ID': ['', '', '', 'T0-7-S', 'T0-9-S'],
			'Dilution': ['1', '10', '20', '', ''],
			'Study': ['TestStudy', 'TestStudy', 'TestStudy', 'TestStudy', 'TestStudy'],
			'Gender': ['', '', '', 'Female', 'Male'],
			'Age': ['', '', '', '55', '66'],
			'Sampling Date': ['', '', '', '27/02/2006', '28/02/2006'],
			'Detector': ['2780', '2780', '2780', '2780', '2780'],
			'Sample batch': ['', '', '', 'SB 1', 'SB 2'],
			'Well': ['1', '2', '3', '4', '5'],
			'Plate': ['1', '1', '1', '1', '1'],
			'Batch': ['2', '2', '3', '4', '5'],
			'Correction Batch': ['', '', '1', '1', '1'],
			'Run Order': ['0', '1', '2', '3', '4'],
			'Instrument': ['QTOF 2', 'QTOF 2', 'QTOF 2', 'QTOF 2', 'QTOF 2'],
			'Chromatography': ['L', 'L', 'L', 'L', 'L'],
			'Ionisation': ['NEG', 'NEG', 'NEG', 'NEG', 'NEG'],

			'Assay data name': ['', '', '', 'SS_LNEG_ToF02_S1W4', 'SS_LNEG_ToF02_S1W5']
		}

		msData.sampleMetadata = pandas.DataFrame(raw_data, columns = ['Acquired Time', 'AssayRole', 'SampleType','Subject ID','Sampling ID','Dilution','Study',
																'Gender','Age','Sampling Date','Detector','Sample batch','Well',
																'Plate','Batch','Correction Batch','Run Order','Instrument','Chromatography','Ionisation','Assay data name'])

		with tempfile.TemporaryDirectory() as tmpdirname:
			msData.exportDataset(destinationPath=tmpdirname, saveFormat='ISATAB',withExclusions=False,filterMetadata=False)

			a = os.path.join(tmpdirname,'a_npc-test-study_metabolite_profiling_mass_spectrometry.txt')
			self.assertTrue(os.path.exists(a))


if __name__ == '__main__':
	unittest.main()
