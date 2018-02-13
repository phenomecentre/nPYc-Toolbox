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

class test_nmrdataset_synthetic(unittest.TestCase):

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

		dataset = nPYc.NMRDataset('', fileType='empty')

		dataset.intensityData = numpy.zeros([18, 5],dtype=float)

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

		dataset = nPYc.NMRDataset('', fileType='empty')

		dataset.intensityData = numpy.zeros([10, noFeat],dtype=float)
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

		dataset = nPYc.NMRDataset('', fileType='empty')

		with self.subTest(msg='No Ranges'):
			dataset.Attributes['exclusionRegions'] = None
			self.assertRaises(ValueError, dataset.updateMasks, filterFeatures=True, filterSamples=False, exclusionRegions=None)

	def test_updateMasks_warns(self):

		dataset = nPYc.NMRDataset('', fileType='empty')

		with self.subTest(msg='Range low == high'):
			dataset.Attributes['exclusionRegions'] = None
			self.assertWarnsRegex(UserWarning, 'Low \(1\.10\) and high \(1\.10\) bounds are identical, skipping region', dataset.updateMasks, filterFeatures=True, filterSamples=False, exclusionRegions=(1.1,1.1))
\

class test_nmrdataset_bilisa(unittest.TestCase):

	def setUp(self):
		datapath = os.path.join("..", "..", "npc-standard-project", "Derived_Data", "UnitTest3_BI-LISA.xls")
		sheetname = 'ICLONDON_UNITTEST3'
		self.testData = nPYc.NMRDataset(datapath, fileType='BI-LISA', pulseProgram=sheetname)

		# Hardcoded data size
		self.noSamp = 10
		self.noFeat = 105


	def test_dimensions(self):

		self.assertEqual((self.testData.noSamples, self.testData.noFeatures), (self.noSamp, self.noFeat))


	def test_samplenames(self):
		"""
		Check loaded samples names against hard coded values.
		"""

		sampleName = pandas.Series(['UnitTest3_Serum_Rack01_RCM_190116/10',
									'UnitTest3_Serum_Rack01_RCM_190116/100',
									'UnitTest3_Serum_Rack01_RCM_190116/110',
									'UnitTest3_Serum_Rack01_RCM_190116/120',
									'UnitTest3_Serum_Rack01_RCM_190116/130',
									'UnitTest3_Serum_Rack01_RCM_190116/140',
									'UnitTest3_Serum_Rack01_RCM_190116/150',
									'UnitTest3_Serum_Rack01_RCM_190116/160',
									'UnitTest3_Serum_Rack01_RCM_190116/170',
									'UnitTest3_Serum_Rack01_RCM_190116/180'],
									name='Sample File Name',
									dtype='str')

		pandas.util.testing.assert_series_equal(self.testData.sampleMetadata['Sample File Name'], sampleName)


	def test_featurenames(self):
		"""
		Test import of feature metadata.
		"""

		with self.subTest(msg="Testing component"):
			testData = pandas.Series(['Total Plasma', 'Total Plasma', 'Total Plasma', 'Total Plasma', 'Total Plasma', 'Total Plasma',
										'VLDL', 'VLDL', 'VLDL', 'VLDL', 'VLDL', 'IDL','IDL','IDL', 'IDL', 'IDL', 'LDL', 'LDL', 'LDL',
										'LDL', 'LDL', 'HDL', 'HDL', 'HDL', 'HDL', 'HDL', 'HDL', 'VLDL-1', 'VLDL-1', 'VLDL-1', 'VLDL-1',
										'VLDL-2', 'VLDL-2', 'VLDL-2', 'VLDL-2', 'VLDL-3', 'VLDL-3', 'VLDL-3', 'VLDL-3', 'VLDL-4', 'VLDL-4',
										'VLDL-4', 'VLDL-4', 'VLDL-5', 'VLDL-5', 'VLDL-5', 'VLDL-5', 'VLDL-6', 'VLDL-6', 'VLDL-6', 'VLDL-6',
										'LDL-1', 'LDL-1', 'LDL-1', 'LDL-1', 'LDL-1', 'LDL-2', 'LDL-2', 'LDL-2', 'LDL-2', 'LDL-2', 'LDL-3',
										'LDL-3', 'LDL-3', 'LDL-3', 'LDL-3', 'LDL-4', 'LDL-4', 'LDL-4', 'LDL-4', 'LDL-4', 'LDL-5', 'LDL-5',
										'LDL-5', 'LDL-5', 'LDL-5', 'LDL-6', 'LDL-6', 'LDL-6', 'LDL-6', 'LDL-6', 'HDL-1', 'HDL-1', 'HDL-1',
										'HDL-1', 'HDL-1', 'HDL-1', 'HDL-2', 'HDL-2', 'HDL-2', 'HDL-2', 'HDL-2', 'HDL-2', 'HDL-3', 'HDL-3',
										'HDL-3', 'HDL-3', 'HDL-3', 'HDL-3', 'HDL-4', 'HDL-4', 'HDL-4', 'HDL-4', 'HDL-4','HDL-4'],
										name='Component',
										dtype='str')

			pandas.util.testing.assert_series_equal(self.testData.featureMetadata['Component'], testData)

		with self.subTest(msg="Testing analyte"):
			testData = pandas.Series(['Triglycerides', 'Cholesterol', 'Free Cholesterol', 'Apo-A1', 'Apo-A2', 'Apo-B', 'Triglycerides',
										'Cholesterol', 'Free Cholesterol', 'Phospholipids', 'Apo-B', 'Triglycerides', 'Cholesterol',
										'Free Cholesterol', 'Phospholipids', 'Apo-B', 'Triglycerides', 'Cholesterol', 'Free Cholesterol',
										'Phospholipids', 'Apo-B', 'Triglycerides', 'Cholesterol', 'Free Cholesterol', 'Phospholipids', 'Apo-A1',
										'Apo-A2', 'Triglycerides', 'Cholesterol', 'Free Cholesterol', 'Phospholipids', 'Triglycerides',
										'Cholesterol', 'Free Cholesterol', 'Phospholipids', 'Triglycerides', 'Cholesterol', 'Free Cholesterol',
										'Phospholipids', 'Triglycerides', 'Cholesterol', 'Free Cholesterol', 'Phospholipids', 'Triglycerides',
										'Cholesterol', 'Free Cholesterol', 'Phospholipids', 'Triglycerides', 'Cholesterol', 'Free Cholesterol',
										'Phospholipids', 'Triglycerides', 'Cholesterol', 'Free Cholesterol', 'Phospholipids', 'Apo-B', 'Triglycerides',
										'Cholesterol', 'Free Cholesterol', 'Phospholipids', 'Apo-B', 'Triglycerides', 'Cholesterol', 'Free Cholesterol',
										'Phospholipids', 'Apo-B', 'Triglycerides', 'Cholesterol', 'Free Cholesterol', 'Phospholipids', 'Apo-B',
										'Triglycerides', 'Cholesterol', 'Free Cholesterol', 'Phospholipids', 'Apo-B', 'Triglycerides', 'Cholesterol',
										'Free Cholesterol', 'Phospholipids', 'Apo-B', 'Triglycerides', 'Cholesterol', 'Free Cholesterol', 'Phospholipids',
										'Apo-A1', 'Apo-A2', 'Triglycerides', 'Cholesterol', 'Free Cholesterol', 'Phospholipids', 'Apo-A1', 'Apo-A2',
										'Triglycerides', 'Cholesterol', 'Free Cholesterol','Phospholipids', 'Apo-A1', 'Apo-A2', 'Triglycerides',
										'Cholesterol', 'Free Cholesterol', 'Phospholipids', 'Apo-A1', 'Apo-A2'],
										name='Analyte',
										dtype='str')

			pandas.util.testing.assert_series_equal(self.testData.featureMetadata['Analyte'], testData)

		with self.subTest(msg="Testing name"):
			testData = pandas.Series(['TPTG', 'TPCH', 'TPFC', 'TPA1', 'TPA2', 'TPAB', 'VLTG', 'VLCH', 'VLFC', 'VLPL', 'VLAB', 'IDTG', 'IDCH', 'IDFC',
								'IDPL', 'IDAB', 'LDTG', 'LDCH', 'LDFC', 'LDPL', 'LDAB', 'HDTG', 'HDCH', 'HDFC', 'HDPL', 'HDA1', 'HDA2', 'V1TG',
								'V1CH', 'V1FC', 'V1PL', 'V2TG', 'V2CH', 'V2FC', 'V2PL', 'V3TG', 'V3CH', 'V3FC', 'V3PL', 'V4TG', 'V4CH', 'V4FC',
								'V4PL', 'V5TG', 'V5CH', 'V5FC', 'V5PL', 'V6TG', 'V6CH', 'V6FC', 'V6PL', 'L1TG', 'L1CH', 'L1FC', 'L1PL', 'L1AB',
								'L2TG', 'L2CH', 'L2FC', 'L2PL', 'L2AB', 'L3TG', 'L3CH', 'L3FC', 'L3PL', 'L3AB', 'L4TG', 'L4CH', 'L4FC', 'L4PL',
								'L4AB', 'L5TG', 'L5CH', 'L5FC', 'L5PL', 'L5AB', 'L6TG', 'L6CH', 'L6FC', 'L6PL', 'L6AB', 'H1TG', 'H1CH', 'H1FC',
								'H1PL', 'H1A1', 'H1A2', 'H2TG', 'H2CH', 'H2FC', 'H2PL', 'H2A1', 'H2A2', 'H3TG', 'H3CH', 'H3FC', 'H3PL', 'H3A1',
								'H3A2', 'H4TG', 'H4CH', 'H4FC', 'H4PL', 'H4A1', 'H4A2'],
								name='Feature Name',
								dtype='str')

			pandas.util.testing.assert_series_equal(self.testData.featureMetadata['Feature Name'], testData)

		with self.subTest(msg="Testing unit"):
			testData = pandas.Series(['mg/dL'] * self.noFeat,
									name='Unit',
									dtype='str')

			pandas.util.testing.assert_series_equal(self.testData.featureMetadata['Unit'], testData)


	def test_intensities(self):
		"""
		Validate import by calculating feature means and compareing to expected values.
		"""

		expected = numpy.array([9.467411849100071208340523298830e+01, 1.871559382280019576683116611093e+02, 5.255211769108241526282654376701e+01,
								1.352772870802980662574555026367e+02, 3.004238770303824779261958610732e+01, 7.687791262350995680208143312484e+01,
								4.761030950357590540988894645125e+01, 1.963214387073095679170364746824e+01, 7.813238946324133848975179716945e+00,
								1.610578365299804559640506340656e+01, 7.243266737237779473446153133409e+00, 4.263658916078100880042711651186e+00,
								1.550990191264557793715539446566e+01, 4.165030770402514903594237694051e+00, 5.316024295291354562209562573116e+00,
								6.939189234726555355337040964514e+00, 2.109875987604672786801529582590e+01, 8.961467976324010464850289281458e+01,
								2.801409104788903547955669637304e+01, 5.419881120213104708227547234856e+01, 6.022582906767079435894629568793e+01,
								1.259475639607611796577657514717e+01, 5.728696168772629704335486167111e+01, 1.367471203600569218394866766175e+01,
								8.031007527912481691600987687707e+01, 1.357467412856030364309845026582e+02, 3.082141956873376642533912672661e+01,
								2.805662654527266752779723901767e+01, 6.649395091317562567212462454336e+00, 1.220470473733668681504127562221e+00,
								4.489431188389593430088098102715e+00, 3.735080536171959231239725340856e+00, 1.881653983192479229202831447765e+00,
								7.673950668037978761759632106987e-01, 1.356332936341682948722109358641e+00, 4.559263671657125982505931460764e+00,
								2.832896792450751988212687138002e+00, 1.067792226367063168623872115859e+00, 2.308340145731800241435394127620e+00,
								6.950597160343905400736730371136e+00, 6.078969219950581148737001058180e+00, 2.683018206054986087849556497531e+00,
								4.924256410292683128204771492165e+00, 3.501114894862889403981398572796e+00, 1.617995612700857011034827337426e+00,
								1.071998852114419253922505959054e+00, 1.965477069575777946397465711925e+00, 3.109840532677314861587092309492e+00,
								1.516583860838282560301593093754e-01, 9.777984563425488884202962935888e-02, 2.491094295337557551484053419699e-01,
								6.276819877492788357642439223127e+00, 2.606251947240883737322292290628e+01, 7.517752797809128750827767362352e+00,
								1.544118477409747036688258958748e+01, 1.448390175825583625623949046712e+01, 2.809936342939324482870233623544e+00,
								1.326438651491017139960604254156e+01, 4.542286116096219572568770672660e+00, 8.229050834657479995826179219875e+00,
								7.673637224713234950002060941188e+00, 3.143206995820660321072637088946e+00, 1.047549604441575787916463013971e+01,
								3.706889019987181566051503978088e+00, 6.818705911936374874926514166873e+00, 7.326229025935361960364389233291e+00,
								2.797922460087013174501180401421e+00, 1.024327805934125734665940399282e+01, 3.651799980734138362237217734219e+00,
								6.467986728976216070918781042565e+00, 7.285507839652121120366246032063e+00, 2.714242522717084327155134815257e+00,
								1.265208088558398102918545191642e+01, 3.649127777769757496884039937868e+00, 7.301783055750076911749602004420e+00,
								9.784401590387412639415742887650e+00, 4.421130848224513520960954338079e+00, 1.680077983625232462827625568025e+01,
								3.573286059559366290017123901634e+00, 1.027878641005318272050317318644e+01, 1.388978333387895780504095455399e+01,
								4.806159327947600701236297027208e+00, 2.048202808265125796083339082543e+01, 4.942966784510231192939500033390e+00,
								2.638724178702198486234919982962e+01, 3.175716833119086501824313018005e+01, 3.334195741492399722716299947933e+00,
								2.531047599010542548114699457074e+00, 1.032476581613613930699102638755e+01, 2.341697357244604482673366874224e+00,
								1.640101937270670617863288498484e+01, 2.039736988415744889380221138708e+01, 3.839970659562040200540877776803e+00,
								2.211585264421761731767901437706e+00, 9.845947140219882598444200993981e+00, 1.539979395234864600894297836930e+00,
								1.540329718003984815766216343036e+01, 2.632825178613001781968705472536e+01, 6.236348523776173990995630447287e+00,
								2.896143959715657523190657229861e+00, 1.418651805989118130923998251092e+01, 1.954242770460509870389387288014e+00,
								2.034013379245358166258483834099e+01, 5.747629672949936718850949546322e+01, 1.462931334316217757418598921504e+01])
		observed = numpy.mean(self.testData.intensityData, axis=0)

		numpy.testing.assert_array_almost_equal_nulp(expected, observed)


	def test_load_npc_lims(self):
		"""
		Validate `NMRDataset.addSampleInfo(descriptionFormat='NPC LIMS' ...)` works with BI-LISA datasets.
		"""

		self.testData.addSampleInfo(descriptionFormat='NPC LIMS', filePath=os.path.join('..','..','npc-standard-project','Derived_Worklists','UnitTest3_NMR_serum_PCSOP.012.csv'))

		with self.subTest(msg='Sample ID'):
			expected = pandas.Series(['UT3_S8', 'UT3_S7', 'UT3_S6', 'UT3_S5', 'UT3_S4',
										'UT3_S3', 'UT3_S2', 'External Reference Sample',
										'Study Pool Sample', 'UT3_S1'],
										name='Sampling ID',
										dtype='str')

			pandas.util.testing.assert_series_equal(self.testData.sampleMetadata['Sampling ID'], expected)

		with self.subTest(msg='Sample position'):
			expected = pandas.Series(['A1', 'A2', 'A3', 'A4', 'A5',
									'A6', 'A7', 'A8', 'A9', 'A10'],
										name='Sample position',
										dtype='str')

			pandas.util.testing.assert_series_equal(self.testData.sampleMetadata['Sample position'], expected)

##
#unit test for Bruker data
##
from nPYc.utilities._baseline import baseline
from nPYc.utilities.nmr import interpolateSpectrum, baselinePWcalcs
from nPYc.reports._generateReportNMR import _generateReportNMR
from math import ceil

class test_nmrdataset_bruker(unittest.TestCase):

	def setUp(self):
		"""
		setup the pulseprogram and path for purpose of testing NMR bruker data functions
		"""
		self.pulseProgram = 'noesygppr1d'
		self.path=os.path.join('..','..','npc-standard-project','unitTest_Data','nmr')#where path is location of test files


	def test_baselineAreaAndNeg(self):
		"""
		Validate baseline/WP code, creates random spectra and values that should always fail ie <0 and high extreme and diagonal.
		"""

		filePathList = pandas.Series(['user1\\Desktop\\unitTestData'], name='File Path', dtype='str')
		for i in range(91):
			filePathList.loc[i] = 'unitTestData'+str(i)
		variableSize = 20000#numpy.random.randint(low=1000, high=50000, size=1)#normally coded into sop as 20000
		#create array of random exponetial values
		X=numpy.random.rand(86, variableSize)*1000
		X = numpy.r_[X, numpy.full((1, variableSize), -10000)]# add a minus  val row r_ shortcut notation for vstack
		X = numpy.r_[X, numpy.full((1, variableSize), 200000)]# add a minus  val row r_ shortcut notation for vstack
		a1=numpy.arange(0,variableSize,1)[numpy.newaxis]#diagonal ie another known fail
		X=numpy.concatenate((X, a1), axis=0)#concatenate into X

		X = numpy.r_[X, numpy.random.rand(2, variableSize)* 10000]		#add more fails random but more variablility than the average 86 above

		#create ppm
		ppm=numpy.linspace(-1,10, variableSize)#bounds and variablesize read from SOP
		#scale[::-1]#if need to flip
		ppmrange1=ppm[numpy.random.randint(numpy.max(numpy.where(ppm<1)), high=numpy.min(numpy.where(ppm>4.9)), size=1)[0]]#get random range value so we not testing static values
		ppmrange2=ppm[numpy.random.randint(numpy.max(numpy.where(ppm<5)), high=numpy.min(numpy.where(ppm>9.9)), size=1)[0]]
		df1 = baseline(filePathList, X, ppm, -1.0,-0.5,'BL_low_', 0.05, 90)#leave one harcoded version covering neg range
		df2 = baseline(filePathList, X, ppm, ppmrange1,ppmrange2,'BL_high_', 0.05, 90)#normally defined in sop as 9.5-10
		df3 = baseline(filePathList, X, ppm, ppmrange1,ppmrange2,'BL_high_', 5, 90)#nchanging alpha to higher value should fail more so lets put to 5

		#NOTE: NOT tested for adjusting threshold only testing at 90 !!

		numpy.testing.assert_array_equal(df1['BL_low_outliersFailArea'], [False, False, False, False, False, False, False, False, False, False,False, False, False, False, False, False, False, False, False, False,False, False, False, False, False, False, False, False, False, False,False, False, False, False, False, False, False, False, False, False,False, False, False, False, False, False, False, False, False, False,False, False, False, False, False, False, False, False, False, False,False, False, False, False, False, False, False, False, False, False,False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, True, False, True, True])
		numpy.testing.assert_array_equal(df1['BL_low_outliersFailNeg'], [False, False, False, False, False, False, False, False, False, False,False, False, False, False, False, False, False, False, False, False,False, False, False, False, False, False, False, False, False, False,False, False, False, False, False, False, False, False, False, False,False, False, False, False, False, False, False, False, False, False,False, False, False, False, False, False, False, False, False, False,False, False, False, False, False, False, False, False, False, False,False, False, False, False, False, False, False, False, False, False,False, False, False, False, False, False, True, False, False, False, False])
		numpy.testing.assert_array_equal(df2['BL_high_outliersFailArea'], [False, False, False, False, False, False, False, False, False, False,False, False, False, False, False, False, False, False, False, False,False, False, False, False, False, False, False, False, False, False,False, False, False, False, False, False, False, False, False, False,False, False, False, False, False, False, False, False, False, False,False, False, False, False, False, False, False, False, False, False,False, False, False, False, False, False, False, False, False, False,False, False, False, False, False, False, False, False, False, False,False, False, False, False, False, False, True, True, True, False, False])
		numpy.testing.assert_array_equal(df2['BL_high_outliersFailNeg'], [False, False, False, False, False, False, False, False, False, False,False, False, False, False, False, False, False, False, False, False,False, False, False, False, False, False, False, False, False, False,False, False, False, False, False, False, False, False, False, False,False, False, False, False, False, False, False, False, False, False,False, False, False, False, False, False, False, False, False, False,False, False, False, False, False, False, False, False, False, False,False, False, False, False, False, False, False, False, False, False,False, False, False, False, False, False, True, False, False, False, False])
		numpy.testing.assert_array_equal(df3['BL_high_outliersFailArea'], [True, True, True, True, True, True, True, True, True, True,True, True, True, True, True, True, True, True, True, True,True, True, True, True, True, True, True, True, True, True,True, True, True, True, True, True, True, True, True, True,True, True, True, True, True, True, True, True, True, True,True, True, True, True, True, True, True, True, True, True,True, True, True, True, True, True, True, True, True, True,True, True, True, True, True, True, True, True, True, True,True, True, True, True, True, True, True, True, True, True, True])
		numpy.testing.assert_array_equal(df3['BL_high_outliersFailNeg'], [False, False, False, False, False, False, False, False, False, False,False, False, False, False, False, False, False, False, False, False,False, False, False, False, False, False, False, False, False, False,False, False, False, False, False, False, False, False, False, False,False, False, False, False, False, False, False, False, False, False,False, False, False, False, False, False, False, False, False, False,False, False, False, False, False, False, False, False, False, False,False, False, False, False, False, False, False, False, False, False,False, False, False, False, False, False, True, False, False, False, False])


	def test_reports(self):
		from nPYc.enumerations import AssayRole
		from nPYc.enumerations import SampleType
		from datetime import datetime
		"""
		Validate generate feature summary report
		at the moment all it will test is if the plots and reports are saved, not checking contents
		"""

#		empty object
		testData = nPYc.NMRDataset('', fileType='empty')
		#need to hardcode in attributes for testing purposes only rather than read in from the SOP, some are generated from the code
		testData.Attributes['WP_highRegionTo'] =5.0
		testData.Attributes['WP_lowRegionFrom']=4.6000000000000005
		testData.Attributes['BL_lowRegionFrom']=-1.0
		testData.Attributes['BL_highRegionTo']=10.0

		testData.Attributes['alignTo']= 'xxxx'#as i dont want it to execute plotting code, maybe for future to do code coverage will have to modify and let it use default from SOP
		testData.sampleMetadata['BL_low_outliersFailArea']=[False,
														False,
														False,
														False]
		testData.sampleMetadata['BL_low_outliersFailNeg']=[False,
														False,
														False,
														False]
		testData.sampleMetadata['BL_high_outliersFailArea']=[False,
														False,
														False,
														False]
		testData.sampleMetadata['BL_high_outliersFailNeg']=[False,
														False,
														False,
														False]
		testData.sampleMetadata['WP_low_outliersFailArea']=[False,
														False,
														False,
														False]
		testData.sampleMetadata['WP_low_outliersFailNeg']=[False,
														False,
														False,
														False]
		testData.sampleMetadata['WP_high_outliersFailArea']=[False,
														False,
														False,
														False]
		testData.sampleMetadata['WP_high_outliersFailNeg']=[False,
														False,
														False,
														False]
		testData.sampleMetadata['Rack']=['Rack1',
														'Rack1',
														'Rack1',
														'Rack1']
		testData.sampleMetadata['Study']=['unitTest',
														'unitTest',
														'unitTest',
														'unitTest']

		testData.sampleMetadata['AssayRole'] = pandas.Series([
								AssayRole.Assay,
								AssayRole.Assay,
								AssayRole.Assay,
								AssayRole.Assay,],
								name='AssayRole',
								dtype=object)

		testData.sampleMetadata['SampleType'] = pandas.Series([SampleType.StudySample,
								SampleType.StudySample,
								SampleType.StudySample,
								SampleType.StudySample,],
								name='SampleType',
								dtype=object)

		testData.sampleMetadata['Acquired Time'] = [datetime(2012,12,1),
													datetime(2012,12,2),
													datetime(2012,12,3),
													datetime(2012,12,4)]

		testData.sampleMetadata['Exclusion Details'] = ['None',
														'None',
														'None',
														'None']

		testData.sampleMetadata['ImportFail'] = [False,
												 False,
												 False,
												 False]

		testData.sampleMetadata['exceed90critical'] = [False,
													   False,
													   False,
													   False]

		testData.sampleMetadata['calibrPass'] = [True,
												 True,
												 True,
												 True]

		testData.sampleMetadata['Line Width (Hz)']=[0.818454,
													1.060146,
													0.876968,
													0.876968]

		testData.sampleMetadata['BL_low_failArea']=[0.220022,
													0.000000,
													1.210121,
													1.210121]

		testData.sampleMetadata['BL_low_failNeg']=[0.0,
												   0.0,
												   0.0,
												   0.0]

		testData.sampleMetadata['BL_high_failArea']=[7.929515,
													 6.387665,
													 11.563877,
													 11.563877]

		testData.sampleMetadata['BL_high_failNeg']=[0.0,
													0.0,
													0.0,
													0.0]

		testData.sampleMetadata['WP_low_failArea']=[1.657459,
													0.000000,
													1.210121,
													1.210121]

		testData.sampleMetadata['WP_low_failNeg']=[0.0,
												   0.0,
												   28.176796,
												   28.176796]

		testData.sampleMetadata['WP_high_failArea']=[19.889503,
													 13.812155,
													 53.038674,
													 53.038674]

		testData.sampleMetadata['WP_high_failNeg']=[0.0,
													0.0,
													0.0,
													0.0]

		testData.sampleMetadata['Status'] = ['Sample',
											 'Sample',
											 'Sample',
											 'Sample']

		testData.sampleMetadata['path']=['UNITTEST01_test/UNITTEST01_Plasma_Rack39_RCM_101214/10',
										 'UNITTEST01_test/UNITTEST01_Plasma_Rack39_RCM_101214/20',
										 'UNITTEST01_test/UNITTEST01_Plasma_Rack39_RCM_101214/30',
										 'UNITTEST01_Plasma_Rack39_RCM_101214/40']

		testData.sampleMetadata['Sample File Name']=['UNITTEST01_Plasma_Rack39_RCM_101214/10',
													 'UNITTEST01_Plasma_Rack39_RCM_101214/20',
													 'UNITTEST01_Plasma_Rack39_RCM_101214/30',
													 'UNITTEST01_Plasma_Rack39_RCM_101214/40']

		testData.sampleMetadata['Sample Base Name']=['UNITTEST01_Plasma_Rack39_RCM_101214/10',
													 'UNITTEST01_Plasma_Rack39_RCM_101214/20',
													 'UNITTEST01_Plasma_Rack39_RCM_101214/30',
													 'UNITTEST01_Plasma_Rack39_RCM_101214/40']

		noFeat = 2000

		testData.sampleMask = numpy.array([False, True, False, False], dtype=bool)
		testData.featureMask = numpy.ones(noFeat, dtype=bool)

		testData.intensityData =  numpy.random.randn(4, noFeat)
		testData.featureMetadata = pandas.DataFrame(numpy.linspace(10, -1, noFeat), columns=('ppm',), dtype=float)

		# create a temporary directory using the context manager
		with tempfile.TemporaryDirectory() as tmpdirname:
			_generateReportNMR(testData, 'feature summary', output=tmpdirname)#run the code for feature summary
			assert os.path.exists(os.path.join(tmpdirname,'graphics','report_featureSummary', 'NMRDataset_calibrationCheck.png')) == 1
			assert os.path.exists(os.path.join(tmpdirname,'graphics','report_featureSummary', 'NMRDataset_finalFeatureBLWPplots1.png')) == 1
			assert os.path.exists(os.path.join(tmpdirname,'graphics','report_featureSummary', 'NMRDataset_finalFeatureBLWPplots3.png')) ==1
			assert os.path.exists(os.path.join(tmpdirname,'graphics','report_featureSummary', 'NMRDataset_finalFeatureIntensityHist.png')) ==1
			assert os.path.exists(os.path.join(tmpdirname,'graphics', 'report_featureSummary','NMRDataset_peakWidthBoxplot.png'))==1
			assert os.path.exists(os.path.join(tmpdirname,'graphics', 'report_featureSummary','npc-main.css'))==1
			assert os.path.exists(os.path.join(tmpdirname,'NMRDataset_report_featureSummary.html')) ==1

		#test final report using same data
		with tempfile.TemporaryDirectory() as tmpdirname:
			_generateReportNMR(testData, 'final report', output=tmpdirname, withExclusions=False)#run the code for feature summary
			assert os.path.exists(os.path.join(tmpdirname,'graphics', 'report_finalReport','NMRDataset_finalFeatureBLWPplots1.png')) == 1
			assert os.path.exists(os.path.join(tmpdirname,'graphics', 'report_finalReport','NMRDataset_finalFeatureBLWPplots3.png')) ==1
			assert os.path.exists(os.path.join(tmpdirname,'graphics', 'report_finalReport','NMRDataset_finalFeatureIntensityHist.png')) ==1
			assert os.path.exists(os.path.join(tmpdirname,'graphics', 'report_finalReport','NMRDataset_peakWidthBoxplot.png'))==1
			assert os.path.exists(os.path.join(tmpdirname,'graphics', 'report_finalReport','npc-main.css'))==1
			assert os.path.exists(os.path.join(tmpdirname,'NMRDataset_report_finalReport.html')) ==1


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
			details = {
			    'investigation_identifier' : "i1",
			    'investigation_title' : "Give it a title",
			    'investigation_description' : "Add a description",
			    'investigation_submission_date' : "2016-11-03", #use today if not specified
			    'investigation_public_release_date' : "2016-11-03",
			    'first_name' : "Noureddin",
			    'last_name' : "Sadawi",
			    'affiliation' : "University",
			    'study_filename' : "my_nmr_study",
			    'study_material_type' : "Serum",
			    'study_identifier' : "s1",
			    'study_title' : "Give the study a title",
			    'study_description' : "Add study description",
			    'study_submission_date' : "2016-11-03",
			    'study_public_release_date' : "2016-11-03",
			    'assay_filename' : "my_nmr_assay"
			}
			nmrData.exportDataset(destinationPath=tmpdirname, isaDetailsDict=details, saveFormat='ISATAB')
			a = os.path.join(tmpdirname,'a_my_nmr_assay.txt')
			self.assertTrue(os.path.exists(a))


if __name__ == '__main__':
	unittest.main()
