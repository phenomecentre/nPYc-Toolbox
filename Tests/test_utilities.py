"""

"""

import scipy
import pandas
import numpy
import sys
import unittest
from pandas.util.testing import assert_frame_equal
import os
import tempfile
import inspect
import copy
import warnings

sys.path.append("..")
import nPYc

def L(x, x0, gamma):
	""" Return Lorentzian line shape at x with HWHM gamma """
	return (1 / numpy.pi) * ((gamma/2) / (((x-x0)**2) + ((gamma/2)**2)))

class test_utilities_internal(unittest.TestCase):

	def test_correlation(self):
		"""
		Validate _vcorrcoef by comparing output to scipy's functions.
		"""

		xdim = numpy.random.randint(10,50)
		ydim = numpy.random.randint(70,300)

		X = numpy.random.normal(size=(xdim, ydim))

		spearman = nPYc.utilities._internal._vcorrcoef(X, X[:,0], method='spearman')
		pearson = nPYc.utilities._internal._vcorrcoef(X, X[:,0],  method='pearson')

		spearman_scipy = numpy.zeros_like(spearman)
		pearson_scipy = numpy.zeros_like(pearson)

		for i in range(ydim):
			pearson_scipy[i] = scipy.stats.pearsonr(X[:,i], X[:,0])[0]
			spearman_scipy[i] = scipy.stats.spearmanr(X[:,i], X[:,0])[0]

		with self.subTest(msg='Testing Spearman Correlation'):
			numpy.testing.assert_allclose(spearman, spearman_scipy, err_msg='Spearman Correlation output does not equal scipy.')
		with self.subTest(msg='Testing Pearson Correlation'):
			numpy.testing.assert_allclose(pearson, pearson_scipy, err_msg='Pearson Correlation output does not equal scipy.')


	def test_correlation_masking(self):
		"""
		Validate _vcorrcoef by comparing output to scipy's functions.
		"""

		# Genreate datasets and mask
		xdim = numpy.random.randint(10,50)
		ydim = numpy.random.randint(70,300)

		sampleMask = numpy.ones(xdim, dtype=bool)
		sampleMask[numpy.random.randint(1, xdim, size=numpy.random.randint(2, int(xdim / 2)))] = False
		featureMask = numpy.ones(ydim, dtype=bool)
		featureMask[numpy.random.randint(1, ydim, size=numpy.random.randint(2, int(ydim / 2)))] = False

		X = numpy.random.normal(size=(xdim, ydim))

		# Genreate correlations
		spearman = nPYc.utilities._internal._vcorrcoef(X, X[:,0], method='spearman', sampleMask=sampleMask, featureMask=featureMask)
		pearson = nPYc.utilities._internal._vcorrcoef(X, X[:,0],  method='pearson', sampleMask=sampleMask, featureMask=featureMask)

		spearman_scipy = numpy.zeros_like(spearman)
		pearson_scipy = numpy.zeros_like(pearson)

		# Apply masks to local data.
		X = X[:,featureMask]
		X = X[sampleMask, :]
		for i in range(sum(featureMask)):
			pearson_scipy[i] = scipy.stats.pearsonr(X[:,i], X[:,0])[0]
			spearman_scipy[i] = scipy.stats.spearmanr(X[:,i], X[:,0])[0]

		with self.subTest(msg='Testing Spearman Correlation'):
			numpy.testing.assert_allclose(spearman, spearman_scipy, err_msg='Spearman Correlation output does not equal scipy.')
		with self.subTest(msg='Testing Pearson Correlation'):
			numpy.testing.assert_allclose(pearson, pearson_scipy, err_msg='Pearson Correlation output does not equal scipy.')


	def test_copybackingfiles(self):
		"""
		Check files are copied to the location specified (we trust the shutil.copy call to preserve contents).
		"""

		expectedFiles = ['npc-main.css', 'toolbox_logo.png']

		with tempfile.TemporaryDirectory() as tmpdirname:

			toolboxPath = os.path.abspath(os.path.dirname(inspect.getfile(nPYc)))

			nPYc.utilities._internal._copyBackingFiles(toolboxPath, tmpdirname)

			for expectedFile in expectedFiles:
				testPath = os.path.join(tmpdirname, expectedFile)
				self.assertTrue(os.path.exists(testPath))


	def test_copybackingfiles_withgraphics(self):
		"""
		Check files are copied to the location specified (we trust the shutil.copy call to preserve contents), when graphics alread exists.
		"""

		expectedFiles = ['npc-main.css', 'toolbox_logo.png']

		with tempfile.TemporaryDirectory() as tmpdirname:

			toolboxPath = os.path.abspath(os.path.dirname(inspect.getfile(nPYc)))

			# Call twice to check the presance of files dons't trip us up
			nPYc.utilities._internal._copyBackingFiles(toolboxPath, tmpdirname)
			nPYc.utilities._internal._copyBackingFiles(toolboxPath, tmpdirname)

			for expectedFile in expectedFiles:
				testPath = os.path.join(tmpdirname, expectedFile)
				self.assertTrue(os.path.exists(testPath))


class test_utilities_ms(unittest.TestCase):

	def test_rsd(self):

		testData = numpy.array([[1, 2, 3], [1, 5, 10], [1, -5,-10]])
		testResults = nPYc.utilities.ms.rsd(testData)

		testData = numpy.array([[1, 2, 3], [1, 2, 3], [1, 2, numpy.nan]])
		testResultsWithNaNs = nPYc.utilities.ms.rsd(testData)

		numpy.testing.assert_allclose(testResults, [0., 628.4902545, 828.65352631], err_msg='RSD calculations not correct.')
		numpy.testing.assert_allclose(testResultsWithNaNs, [0, 0, numpy.finfo(numpy.float64).max], err_msg='RSD calculation not handling NaNs correctly.')


	def test_sequentialPrecision(self):
		# Fix the random seed for reproducible results
		numpy.random.seed(seed=200)

		x = numpy.random.randn(200, 3)
		x = numpy.add(x, 10)

		preDrift = nPYc.utilities.ms.sequentialPrecision(x)

		x[:, 0] = numpy.add(x[:, 0], numpy.linspace(1,5, 200))
		x[:, 1] = numpy.multiply(x[:, 1], numpy.linspace(1,5, 200))
		x[:, 2] = numpy.add(x[:, 2], numpy.linspace(1,10, 200))
		x[2,2] = numpy.inf

		postDrift = nPYc.utilities.ms.sequentialPrecision(x)

		numpy.testing.assert_allclose(preDrift, [9.6409173, 10.39057302, 9.61397365], err_msg='SP calculations not correct.')
		numpy.testing.assert_allclose(postDrift, [7.42880240e+000, 1.16717529e+001, numpy.finfo(numpy.float64).max], err_msg='SP calculations not correct.')
		numpy.random.seed()


	def test_generatesrdmask(self):

		# Create an empty object with simple filenames
		msData = nPYc.MSDataset('', fileType='empty')

		msData.sampleMetadata['Sample File Name'] = ['Test1_HPOS_ToF01_B1SRD01', 'Test1_HPOS_ToF01_B1SRD02', 'Test1_HPOS_ToF01_B1SRD43',
													'Test1_HPOS_ToF01_B1SRD44','Test1_HPOS_ToF01_B1SRD45','Test1_HPOS_ToF01_B1SRD46',
													'Test1_HPOS_ToF01_B1SRD47','Test1_HPOS_ToF01_B1SRD48','Test1_HPOS_ToF01_B1SRD49',
													'Test1_HPOS_ToF01_B1SRD50','Test1_HPOS_ToF01_B1SRD51','Test1_HPOS_ToF01_B1SRD92',
													'Test1_HPOS_ToF01_B2SRD01','Test1_HPOS_ToF01_B2SRD02','Test1_HPOS_ToF01_B2SRD43',
													'Test1_HPOS_ToF01_B2SRD44','Test1_HPOS_ToF01_B2SRD45','Test1_HPOS_ToF01_B2SRD46',
													'Test1_HPOS_ToF01_P2W30','Test1_HPOS_ToF01_P2W31_SR','Test1_HPOS_ToF01_P2W32',
													'Test1_HPOS_ToF01_B2SRD47','Test1_HPOS_ToF01_B2SRD48','Test1_HPOS_ToF01_B2SRD49',
													'Test1_HPOS_ToF01_B2SRD50','Test1_HPOS_ToF01_B2SRD51','Test1_HPOS_ToF01_B2SRD92',
													'Test1_HPOS_ToF01_B3SRD01','Test1_HPOS_ToF01_B3SRD02','Test1_HPOS_ToF01_B3SRD43',
													'Test1_HPOS_ToF01_B3SRD44','Test1_HPOS_ToF01_B3SRD45','Test1_HPOS_ToF01_B3SRD46',
													'Test1_HPOS_ToF01_B3SRD47','Test1_HPOS_ToF01_B3SRD48','Test1_HPOS_ToF01_B3SRD49',
													'Test1_HPOS_ToF01_B3SRD50','Test1_HPOS_ToF01_B3SRD51','Test1_HPOS_ToF01_B3SRD92']

		msData.intensityData = numpy.zeros((39,2))
		msData.initialiseMasks()
		msData.sampleMetadata['Run Order'] = msData.sampleMetadata.index + 1
		msData.addSampleInfo(descriptionFormat='Filenames')
		msData.addSampleInfo(descriptionFormat='Batches')
		msData.corrExclusions = msData.sampleMask

		srdMask = nPYc.utilities.ms.generateLRmask(msData)

		cannonicalMask = {'Batch 1.0, series 1.0': numpy.array([True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, False, False, False, False, False, False, 
																False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],
															   dtype=bool),
						  'Batch 2.0, series 1.0': numpy.array([False, False, False, False, False, False, False, False, False, False, False, False, True, True, True, True, True, True, False, False,
						  										False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],
															   dtype=bool),
						  'Batch 2.0, series 2.0': numpy.array([False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False,
						  										False, True, True, True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False],
															   dtype=bool),
						  'Batch 3.0, series 2.0': numpy.array([False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False,
																False, False, False, False, False, False, False, True, True, True, True, True, True, True, True, True, True, True, True],
															   dtype=bool)}

		numpy.testing.assert_equal(srdMask, cannonicalMask)


	def test_generatesrdmask_raises(self):

		dataset = nPYc.MSDataset('', fileType='empty')

		self.assertRaises(ValueError, nPYc.utilities.ms.generateLRmask, dataset)


	def test_rsdsBySampleType(self):

		from generateTestDataset import generateTestDataset

		noSamp = numpy.random.randint(100, high=500, size=None)
		noFeat = numpy.random.randint(200, high=400, size=None)

		data = generateTestDataset(noSamp, noFeat)

		with self.subTest(msg="Default"):
			rsds = nPYc.utilities.rsdsBySampleType(data)
			self.assertTrue('External Reference' in rsds.keys())
			self.assertTrue('Study Pool' in rsds.keys())
			self.assertTrue('Study Sample' not in rsds.keys())
			self.assertTrue(len(rsds.keys()) == 2)

		with self.subTest(msg="With all samples"):
			rsds = nPYc.utilities.rsdsBySampleType(data, onlyPrecisionReferences=False)
			self.assertTrue('External Reference' in rsds.keys())
			self.assertTrue('Study Pool' in rsds.keys())
			self.assertTrue('Study Sample' in rsds.keys())
			self.assertTrue(len(rsds.keys()) == 3)


	def test_rsdsBySampleType_raises(self):

		self.assertRaises(TypeError, nPYc.utilities.rsdsBySampleType, 'Not a Dataset')
		self.assertRaises(KeyError, nPYc.utilities.rsdsBySampleType, nPYc.Dataset(), useColumn='Not There')


class test_utilities_conditionaljoin(unittest.TestCase):

	def test_utilities_conditionaljoin_assertstring(self):

		self.assertRaises(TypeError, nPYc.utilities._conditionalJoin.conditionalJoin, 'a', 'b', separator=1)


	def test_utilities_conditionaljoin_join(self):

		expected = 'a; b'
		actual = nPYc.utilities._conditionalJoin.conditionalJoin('a', 'b')
		self.assertEqual(expected, actual)


	def test_utilities_conditionaljoin_join_first(self):

		expected = 'a'
		with self.subTest(msg='Test None'):
			actual = nPYc.utilities._conditionalJoin.conditionalJoin('a', None)
			self.assertEqual(expected, actual)

		with self.subTest(msg='Test empty string'):
			actual = nPYc.utilities._conditionalJoin.conditionalJoin('a', '')
			self.assertEqual(expected, actual)


	def test_utilities_conditionaljoin_join_second(self):

		expected = 'b'
		with self.subTest(msg='Test None'):
			actual = nPYc.utilities._conditionalJoin.conditionalJoin(None, 'b')
			self.assertEqual(expected, actual)

		with self.subTest(msg='Test empty string'):
			actual = nPYc.utilities._conditionalJoin.conditionalJoin('', 'b')
			self.assertEqual(expected, actual)


	def test_utilities_conditionaljoin_join_empty(self):

		with self.subTest(msg='Test None'):
			actual = nPYc.utilities._conditionalJoin.conditionalJoin(None, None)
			self.assertEqual(None, actual)

		with self.subTest(msg='Test empty string'):
			actual = nPYc.utilities._conditionalJoin.conditionalJoin('', '')
			self.assertEqual('', actual)


	def test_utilities_conditionaljoin_join_seperator(self):

		expected = 'a+b'
		actual = nPYc.utilities._conditionalJoin.conditionalJoin('a', 'b', separator='+')
		self.assertEqual(expected, actual)


from nPYc.utilities._npc_sampleledger import parseRelationships, loadSampleManifest
class test_utilities_sample_ledger(unittest.TestCase):

	def setUp(self):

		self.manifestPath = os.path.join('..','..','npc-standard-project','Project_Description','UnitTest1_metadata_PCDOC.014.xlsx')


	def test_parseRelationships_raises(self):

		samplingTable = pandas.DataFrame([[True, True], [True, True]], columns=['Subject ID', 'Sampling ID'])
		subjectTable = pandas.DataFrame([[True], [True]], columns=['Subject ID'])

		with self.subTest(msg='No subject ID'):

			subjectTable = pandas.DataFrame([True], columns=['No Subject ID'])
			self.assertRaises(LookupError, parseRelationships, subjectTable, samplingTable)

		with self.subTest(msg='samplingTable - No Sampling ID'):

			samplingTable = pandas.DataFrame([True], columns=['Subject ID'])
			self.assertRaises(LookupError, parseRelationships, subjectTable, samplingTable)

		with self.subTest(msg='samplingTable - No Subject ID'):

			samplingTable = pandas.DataFrame([True], columns=['Sampling ID'])
			self.assertRaises(LookupError, parseRelationships, subjectTable, samplingTable)


	def test_parseRelationships_columns(self):

		subjects = pandas.read_excel(self.manifestPath, 'Subject Info', header=0)
		samplings = pandas.read_excel(self.manifestPath, 'Sampling Events', header=0)

		actual = parseRelationships(subjects, samplings)

		columns = ['Subject ID', 'Class', 'Date of Birth', 'Gender', 'Further Subject info?', 'Environmental measures', 'Sampling ID','Subject ID',
					'Sample Type', 'Sampling Date', 'Further Sample info?', 'Person responsible', 'Sampling Protocol', 'Creatinine (mM)', 'Glucose (mM)']

		for column in columns:
			self.subTest(msg='Checking ' + column)
			self.assertIn(column, actual.keys())


	def test_loadSampleManifest(self):

		actual = loadSampleManifest(self.manifestPath)

		expected = pandas.read_csv(os.path.join('..','..','npc-standard-project','Project_Description','UnitTest1_metadata_Unified.csv'), index_col=0)
		expected['Date of Birth'] = expected['Date of Birth'].apply(pandas.to_datetime)

		pandas.util.testing.assert_frame_equal(actual, expected)


class test_utilities_generic(unittest.TestCase):

	def test_removeDuplicateColumns(self):
		data1 = {
		'Subject ID': ['1', '2', '3', '4'],
		'Sample Base Name': ['Test2_RPOS_ToF02_U2W03','Test3_RNEG_ToF03_S3W04', 'Test4_LPOS_ToF04_P4W05_LTR', 'Test5_LNEG_ToF05_U5W06_SR']
		}
		df1 = pandas.DataFrame(data1, columns = ['Subject ID', 'Sample Base Name'])

		data2 = {
		'Subject ID': ['11', '12', '13', '14'],
		'Sample Base Name': ['Test2_RPOS_ToF02_U2W04','Test3_RNEG_ToF03_S3W05', 'Test4_LPOS_ToF04_P4W06_LTR', 'Test5_LNEG_ToF05_U5W07_SR']
		}
		df2 = pandas.DataFrame(data2, columns = ['Subject ID', 'Sample Base Name'])

		df3 = pandas.merge(df1, df2, left_on='Subject ID', right_on='Subject ID')
		df3 = nPYc.utilities.generic.removeDuplicateColumns(df3)
		#after applying the method, no columns in df3 should end with '_x' or '_y'
		self.assertEqual(any((c[-2:] in ['_x','_y']) for c in df3.columns) , False)


	def test_removeTrailingColumnNumbering(self):
		test_list = ['Col','Col.1','Col.2','AnotherCol','AnotherCol.1','YetAnotherCol']
		correct_list = ['Col','Col','Col','AnotherCol','AnotherCol','YetAnotherCol']
		ls = nPYc.utilities.generic.removeTrailingColumnNumbering(test_list)
		self.assertListEqual(ls,correct_list)


	def test_utilities_checkinrange(self):

		from nPYc.utilities._checkInRange import checkInRange

		with self.subTest(msg='No bounds'):
			sampNo = numpy.random.randint(70,500)
			values = numpy.random.randn(sampNo)
			inRange = checkInRange(values, None, None)

			self.assertTrue(inRange)

		with self.subTest(msg='In range'):
			sampNo = numpy.random.randint(70,500)
			values = numpy.random.randn(sampNo)
			inRange = checkInRange(values, (2.5, -3), (97.5, 3))

			self.assertTrue(inRange)

		with self.subTest(msg='Out of range'):
			sampNo = numpy.random.randint(70,500)
			values = numpy.random.randn(sampNo)
			inRange = checkInRange(values, (2.5, 0), (97.5, 1))

			self.assertFalse(inRange)

		with self.subTest(msg='2D data'):
			sampNo = numpy.random.randint(70,500)
			sampNo2 = numpy.random.randint(70,500)

			values = numpy.random.randn(sampNo, sampNo2)
			inRange = checkInRange(values, (2.5, -3), (97.5, 3))

			self.assertTrue(inRange)


class test_utilities_filters(unittest.TestCase):

	def setUp(self):
		from nPYc.enumerations import AssayRole, SampleType

		self.msData = nPYc.MSDataset('', fileType='empty')

		##
		# Variables:
		# Above blank
		# Below blank (default)
		# Below blank * 5
		##
		self.msData.intensityData = numpy.array([[54, 53, 121],
												[57, 49, 15],
												[140, 41, 97],
												[52, 60, 42],
												[12, 48, 8],
												[1, 60, 41],
												[2, 21, 42,]],
												dtype=float)

		self.msData.sampleMetadata = pandas.DataFrame(data=[ [numpy.nan, 1, 1, 1, AssayRole.Assay, SampleType.StudySample],
															[numpy.nan, 1, 1, 1, AssayRole.Assay, SampleType.StudySample],
															[numpy.nan, 1, 1, 1, AssayRole.Assay, SampleType.StudySample],
															[numpy.nan, 1, 1, 1, AssayRole.Assay, SampleType.StudySample],
															[numpy.nan, 1, 1, 1, AssayRole.Assay, SampleType.StudySample],
															[0, 1, 1, 1, AssayRole.Assay, SampleType.ProceduralBlank],
															[0, 1, 1, 1, AssayRole.Assay, SampleType.ProceduralBlank]],
															columns=['Dilution', 'Batch', 'Correction Batch', 'Well', 'AssayRole', 'SampleType'])

		self.msData.featureMetadata = pandas.DataFrame(data=[['Feature_1', 0.5, 100., 0.3],
															['Feature_2', 0.55, 100.04, 0.3],
															['Feature_3', 0.75, 200., 0.1]],
															columns=['Feature Name','Retention Time','m/z','Peak Width'])

		self.msData.initialiseMasks()


	def test_blank_filter(self):

		with self.subTest(msg='Default settings'):
			blankMaskObtained = nPYc.utilities._filters.blankFilter(self.msData)
			expected = numpy.array([True, False, True])
			numpy.testing.assert_array_equal(blankMaskObtained, expected)

		with self.subTest(msg='Custom threshold'):
			blankMaskObtained = nPYc.utilities._filters.blankFilter(self.msData, threshold=2.)
			expected = numpy.array([True, False, False])
			numpy.testing.assert_array_equal(blankMaskObtained, expected)

		with self.subTest(msg='No filter'):
			blankMaskObtained = nPYc.utilities._filters.blankFilter(self.msData, threshold=False)
			expected = numpy.array([True, True, True])
			numpy.testing.assert_array_equal(blankMaskObtained, expected)


	def test_blank_filter_raises(self):
		msData = nPYc.MSDataset('', fileType='empty')

		with self.subTest(msg='Invalid threshold'):
			self.assertRaises(TypeError, nPYc.utilities._filters.blankFilter, msData, threshold='A string')

		with self.subTest(msg='True threshold'):
			self.assertRaises(TypeError, nPYc.utilities._filters.blankFilter, msData, threshold=True)


	def test_blank_filter_warns(self):

		self.msData.sampleMetadata['SampleType'] = nPYc.enumerations.SampleType.StudySample

		self.assertWarnsRegex(UserWarning, 'No Procedural blank samples present, skipping blank filter\.', nPYc.utilities._filters.blankFilter, self.msData)


class  test_utilities_addReferenceRanges(unittest.TestCase):

	def test_utilities_addReferenceRanges(self):

		from nPYc.utilities._addReferenceRanges import addReferenceRanges

		featureMetadata = pandas.DataFrame(['TPTG', 'TPCH', 'TPFC', 'TPA1', 'TPA2', 'TPAB', 'VLTG', 'VLCH', 'VLFC', 'VLPL', 'VLAB', 'IDTG', 'IDCH', 'IDFC', 'IDPL',
											'IDAB', 'LDTG', 'LDCH', 'LDFC', 'LDPL', 'LDAB', 'HDTG', 'HDCH', 'HDFC', 'HDPL', 'HDA1', 'HDA2', 'V1TG', 'V1CH', 'V1FC',
											'V1PL', 'V2TG', 'V2CH', 'V2FC', 'V2PL', 'V3TG', 'V3CH', 'V3FC', 'V3PL', 'V4TG', 'V4CH', 'V4FC', 'V4PL', 'V5TG', 'V5CH',
											'V5FC', 'V5PL', 'V6TG', 'V6CH', 'V6FC', 'V6PL', 'L1TG', 'L1CH', 'L1FC', 'L1PL', 'L1AB', 'L2TG', 'L2CH', 'L2FC', 'L2PL',
											'L2AB', 'L3TG', 'L3CH', 'L3FC', 'L3PL', 'L3AB', 'L4TG', 'L4CH', 'L4FC', 'L4PL', 'L4AB', 'L5TG', 'L5CH', 'L5FC', 'L5PL',
											'L5AB', 'L6TG', 'L6CH', 'L6FC', 'L6PL', 'L6AB', 'H1TG', 'H1CH', 'H1FC', 'H1PL', 'H1A1', 'H1A2', 'H2TG', 'H2CH', 'H2FC',
											'H2PL', 'H2A1', 'H2A2', 'H3TG', 'H3CH', 'H3FC', 'H3PL', 'H3A1', 'H3A2', 'H4TG', 'H4CH', 'H4FC', 'H4PL', 'H4A1', 'H4A2'],
											columns=['Feature Name'])

		expectedFeatureMetadata = copy.deepcopy(featureMetadata)

		referencePath = os.path.join('..', 'nPYc', 'StudyDesigns', 'BI-LISA_reference_ranges.json')

		addReferenceRanges(featureMetadata, referencePath)

		expectedFeatureMetadata['Upper Reference Bound'] = [97.5, 97.5, numpy.nan, 97.5, 97.5, 97.5, 97.5, 97.5, 97.5, 97.5, 97.5, 97.5, 97.5, 97.5, 97.5, 97.5, 97.5, 97.5, 97.5,
															97.5, 97.5, 97.5, 97.5, 97.5, 97.5, 97.5, 97.5, 97.5, 97.5, 97.5, 97.5, 97.5, 97.5, 97.5, 97.5, 97.5, 97.5, 97.5, 97.5,
															97.5, 97.5, 97.5, 97.5, 97.5, 97.5, 97.5, 97.5, numpy.nan, numpy.nan, numpy.nan, numpy.nan, 97.5, 97.5, 97.5, 97.5, 97.5,
															97.5, 97.5, 97.5, 97.5, 97.5, 97.5, 97.5, 97.5, 97.5, 97.5, 97.5, 97.5, 97.5, 97.5, 97.5, 97.5, 97.5, 97.5, 97.5, 97.5,
															97.5, 97.5, 97.5, 97.5, 97.5, 97.5, 97.5, 97.5, 97.5, 97.5, 97.5, 97.5, 97.5, 97.5, 97.5, 97.5, 97.5, 97.5, 97.5, 97.5,
															97.5, 97.5, 97.5, 97.5, 97.5, 97.5, 97.5, 97.5, 97.5]

		expectedFeatureMetadata['Upper Reference Value'] = [490., 341., numpy.nan, 217., 48., 160., 366., 77., 33., 68., 26., 100., 50., 14., 33., 17., 45., 227., 63., 121., 141., 29.,
															96., 27., 136., 222., 48., 212., 35., 13., 32., 67., 15., 7., 15., 49., 16., 8., 14., 28., 15., 7., 13., 7., 4., 2., 5.,
															numpy.nan, numpy.nan, numpy.nan, numpy.nan, 14., 59., 17., 30., 31., 6., 48., 14., 25., 23., 6., 46., 13., 24., 27., 8., 49.,
															12., 25., 32., 9., 49., 13., 25., 34., 13., 54., 12., 28., 45., 12., 46., 12., 57., 75., 8., 5., 16., 5., 27., 36., 8., 5.,
															19., 5., 32., 47., 12., 8., 30., 9., 44., 110., 30.]

		expectedFeatureMetadata['Lower Reference Bound'] = [2.5, 2.5, numpy.nan, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5,
															2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, numpy.nan,
															numpy.nan, numpy.nan, numpy.nan, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5,
															2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5,
															2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5]

		expectedFeatureMetadata['Lower Reference Value'] = [53., 140., numpy.nan, 112., 24., 48., 21., 5., 3., 6., 3., 5., 4., 1., 3., 2., 12., 5., 17., 37., 42., 7., 35., 7., 57., 110.,
															25., 6., 1., 0., 1., 3., 0., 0., 1., 2., 0., 0., 1., 3., 1., 0., 2., 1., 0., 0., 0., numpy.nan, numpy.nan, numpy.nan, numpy.nan,
															3., 8., 2., 6., 5., 1., 2., 1., 2., 3., 1., 3., 1., 2., 3., 1., 4., 1., 3., 4., 1., 5., 2., 4., 5., 1., 6., 2., 4., 5., 1., 6.,
															1., 8., 6., 1., 1., 4., 1., 7., 10., 2., 1., 7., 1., 12., 18., 5., 2., 11., 2., 20., 56., 12.]

		expectedFeatureMetadata['Unit'] = ['mg/dL', 'mg/dL', numpy.nan, 'mg/dL', 'mg/dL', 'mg/dL', 'mg/dL', 'mg/dL', 'mg/dL', 'mg/dL', 'mg/dL', 'mg/dL', 'mg/dL', 'mg/dL', 'mg/dL', 'mg/dL',
										'mg/dL', 'mg/dL', 'mg/dL', 'mg/dL', 'mg/dL', 'mg/dL', 'mg/dL', 'mg/dL', 'mg/dL', 'mg/dL', 'mg/dL', 'mg/dL', 'mg/dL', 'mg/dL', 'mg/dL', 'mg/dL', 'mg/dL',
										'mg/dL', 'mg/dL', 'mg/dL', 'mg/dL', 'mg/dL', 'mg/dL', 'mg/dL', 'mg/dL', 'mg/dL', 'mg/dL', 'mg/dL', 'mg/dL', 'mg/dL', 'mg/dL', numpy.nan, numpy.nan, numpy.nan,
										numpy.nan, 'mg/dL', 'mg/dL', 'mg/dL', 'mg/dL', 'mg/dL', 'mg/dL', 'mg/dL', 'mg/dL', 'mg/dL', 'mg/dL', 'mg/dL', 'mg/dL', 'mg/dL', 'mg/dL', 'mg/dL', 'mg/dL',
										'mg/dL', 'mg/dL', 'mg/dL', 'mg/dL', 'mg/dL', 'mg/dL', 'mg/dL', 'mg/dL', 'mg/dL', 'mg/dL', 'mg/dL', 'mg/dL', 'mg/dL', 'mg/dL', 'mg/dL', 'mg/dL', 'mg/dL',
										'mg/dL', 'mg/dL', 'mg/dL', 'mg/dL', 'mg/dL', 'mg/dL', 'mg/dL', 'mg/dL', 'mg/dL', 'mg/dL', 'mg/dL', 'mg/dL', 'mg/dL', 'mg/dL', 'mg/dL', 'mg/dL', 'mg/dL',
										'mg/dL', 'mg/dL', 'mg/dL', 'mg/dL']

		pandas.util.testing.assert_frame_equal(expectedFeatureMetadata, featureMetadata, check_dtype=False)


class test_utilities_read_bruker_xml(unittest.TestCase):

	def test_utilities_readBrukerXML(self):
		
		from nPYc.utilities._readBrukerXML import readBrukerXML
		
		with self.subTest(msg='BI-LISA type'):
			expected = ('UnitTest6_expno10.100000.11r',
						 '27-Aug-2015 09:59:47',
						 [{'Feature Name': 'TPTG',
						   'Lower Reference Bound': 2.5,
						   'Lower Reference Value': 53.45,
						   'Unit': 'mg/dL',
						   'Upper Reference Bound': 97.5,
						   'Upper Reference Value': 489.81,
						   'comment': 'Main Parameters, Triglycerides, TG',
						   'lod': '-',
						   'loq': '-',
						   'type': 'prediction',
						   'value': '134.65'},
						  {'Feature Name': 'TPCH',
						   'Lower Reference Bound': 2.5,
						   'Lower Reference Value': 140.31,
						   'Unit': 'mg/dL',
						   'Upper Reference Bound': 97.5,
						   'Upper Reference Value': 341.43,
						   'comment': 'Main Parameters, Cholesterol, Chol',
						   'lod': '-',
						   'loq': '-',
						   'type': 'prediction',
						   'value': '183.95'},
						  {'Feature Name': 'LDCH',
						   'Lower Reference Bound': 2.5,
						   'Lower Reference Value': 54.52,
						   'Unit': 'mg/dL',
						   'Upper Reference Bound': 97.5,
						   'Upper Reference Value': 226.6,
						   'comment': 'Main Parameters, LDL Cholesterol, LDL-Chol',
						   'lod': '-',
						   'loq': '-',
						   'type': 'prediction',
						   'value': '94.57'},
						  {'Feature Name': 'HDCH',
						   'Lower Reference Bound': 2.5,
						   'Lower Reference Value': 34.6,
						   'Unit': 'mg/dL',
						   'Upper Reference Bound': 97.5,
						   'Upper Reference Value': 96.25,
						   'comment': 'Main Parameters, HDL Cholesterol, HDL-Chol',
						   'lod': '-',
						   'loq': '-',
						   'type': 'prediction',
						   'value': '52.22'}])
			actual = readBrukerXML(os.path.join('..', '..', 'npc-standard-project', 'Derived_Data','bruker_quant_BILISA.xml'))

			self.assertEqual(expected, actual)
			
		with self.subTest(msg='Urine Quant type'):
			expected = ('UnitTest5_expno840.100000.10r',
						 '15-Aug-2017 13:06:45',
						 [{'Feature Name': 'Creatinine',
						   'Unit': 'mmol/L',
						   'comment': '',
						   'lod': '-',
						   'loq': '-',
						   'type': 'quantification',
						   'value': '4.3'},
						  {'Feature Name': 'Dimethylamine',
						   'Unit': 'mmol/L',
						   'comment': '',
						   'lod': 0.133,
						   'loq': '-',
						   'type': 'quantification',
						   'value': '0.19'},
						  {'Feature Name': 'Dimethylamine',
						   'Lower Reference Bound': 2.5,
						   'Lower Reference Value': '-',
						   'Unit': 'mmol/mol Crea',
						   'Upper Reference Bound': 97.5,
						   'Upper Reference Value': 54,
						   'comment': '',
						   'lod': 31,
						   'loq': '-',
						   'type': 'quantification',
						   'value': '43'},
						  {'Feature Name': 'Trimethylamine',
						   'Unit': 'mmol/L',
						   'comment': '',
						   'lod': 0.009,
						   'loq': '-',
						   'type': 'quantification',
						   'value': '0'},
						  {'Feature Name': 'Trimethylamine',
						   'Lower Reference Bound': 2.5,
						   'Lower Reference Value': '-',
						   'Unit': 'mmol/mol Crea',
						   'Upper Reference Bound': 97.5,
						   'Upper Reference Value': 3,
						   'comment': '',
						   'lod': 2,
						   'loq': '-',
						   'type': 'quantification',
						   'value': '0'},
						  {'Feature Name': '1-Methylhistidine',
						   'Unit': 'mmol/L',
						   'comment': '',
						   'lod': 0.065,
						   'loq': '-',
						   'type': 'quantification',
						   'value': '0'},
						  {'Feature Name': '1-Methylhistidine',
						   'Lower Reference Bound': 2.5,
						   'Lower Reference Value': '-',
						   'Unit': 'mmol/mol Crea',
						   'Upper Reference Bound': 97.5,
						   'Upper Reference Value': 15,
						   'comment': '',
						   'lod': 15,
						   'loq': '-',
						   'type': 'quantification',
						   'value': '0'}])
			actual = readBrukerXML(os.path.join('..', '..', 'npc-standard-project', 'Derived_Data','bruker_quant_urine.xml'))

			self.assertEqual(expected, actual)


	def test_utilities_readBrukerXML_warns(self):

		from nPYc.utilities._readBrukerXML import importBrukerXML

		with tempfile.TemporaryDirectory() as tmpdirname:
			tmpfile = os.path.join(tmpdirname, 'malformedxml.xml')

			with open(tmpfile, 'w') as tmpf:
				tmpf.write('Most definitely not xml <as \n')

			self.assertWarnsRegex(UserWarning, 'Error parsing xml in .+?, skipping', importBrukerXML, [tmpfile])


	def test_utilities_importBrukerXML(self):

		from nPYc.utilities._readBrukerXML import importBrukerXML

		path = os.path.join('..', '..', 'npc-standard-project', 'Derived_Data','bruker_quant_urine.xml')
		
		paths = [path, path, path]
		
		expectedIntesityData = numpy.array([[4.3, 0.19, 43., 0., 0., 0., 0.],
											[4.3, 0.19, 43., 0., 0., 0., 0.],
											[4.3, 0.19, 43., 0., 0., 0., 0.]])

		expectedFeatureMetadata = pandas.DataFrame.from_dict({'Feature Name': {0: 'Creatinine', 1: 'Dimethylamine', 2: 'Dimethylamine', 3: 'Trimethylamine',
																			   4: 'Trimethylamine', 5: '1-Methylhistidine',6: '1-Methylhistidine'},
															  'Lower Reference Bound': {0: numpy.nan, 1: numpy.nan, 2: 2.5, 3: numpy.nan, 4: 2.5, 5: numpy.nan, 6: 2.5},
															  'Lower Reference Value': {0: numpy.nan, 1: numpy.nan, 2: '-', 3: numpy.nan, 4: '-', 5: numpy.nan, 6: '-'},
															  'Unit': {0: 'mmol/L', 1: 'mmol/L', 2: 'mmol/mol Crea', 3: 'mmol/L', 4: 'mmol/mol Crea',
																	   5: 'mmol/L', 6: 'mmol/mol Crea'},
															  'Upper Reference Bound': {0: numpy.nan, 1: numpy.nan, 2: 97.5, 3: numpy.nan, 4: 97.5, 5: numpy.nan, 6: 97.5},
															  'Upper Reference Value': {0: numpy.nan, 1: numpy.nan, 2: 54.0, 3: numpy.nan, 4: 3.0, 5: numpy.nan, 6: 15.0},
															  'comment': {0: '', 1: '', 2: '', 3: '', 4: '', 5: '', 6: ''},
															  'lod': {0: '-', 1: 0.133, 2: 31, 3: 0.009, 4: 2, 5: 0.065, 6: 15},
															  'loq': {0: '-', 1: '-', 2: '-', 3: '-', 4: '-', 5: '-', 6: '-'},
															  'type': {0: 'quantification', 1: 'quantification', 2: 'quantification',
																	   3: 'quantification', 4: 'quantification', 5: 'quantification',
																	   6: 'quantification'}})

		expectedSampleMetadata = pandas.DataFrame([], columns=['Sample File Name', 'Sample Base Name', 'expno', 'Path', 'Acquired Time', 'Run Order'])
		expectedSampleMetadata['Path'] = [path, path, path]
		#expectedSampleMetadata['Sample File Name'] = ['UnitTest5_expno840.100000.10r', 'UnitTest5_expno840.100000.10r', 'UnitTest5_expno840.100000.10r'] # change to Sample File Name matching Sample Base Name
		expectedSampleMetadata['Sample File Name'] = ['UnitTest5/840', 'UnitTest5/840', 'UnitTest5/840']
		expectedSampleMetadata['Sample Base Name'] = ['UnitTest5/840', 'UnitTest5/840', 'UnitTest5/840']
		expectedSampleMetadata['Acquired Time'] = [pandas.Timestamp('2017-08-15 13:06:45'), pandas.Timestamp('2017-08-15 13:06:45'), pandas.Timestamp('2017-08-15 13:06:45')]
		expectedSampleMetadata['Run Order'] = [0, 1, 2]
		expectedSampleMetadata['expno'] = [840, 840, 840]

		(intensityData, sampleMetadata, featureMetadata) = importBrukerXML(paths)

		numpy.testing.assert_array_equal(intensityData, expectedIntesityData)
		pandas.testing.assert_frame_equal(sampleMetadata, expectedSampleMetadata)
		pandas.testing.assert_frame_equal(featureMetadata, expectedFeatureMetadata)


	def test_utilities_importBrukerXML_fails(self):

		from nPYc.utilities._readBrukerXML import importBrukerXML

		path = os.path.join('..', '..', 'npc-standard-project', 'Derived_Data','bruker_quant_urine.xml')

		with tempfile.TemporaryDirectory() as tmpdirname:
			tmpfile = os.path.join(tmpdirname, 'malformedxml.xml')

			with open(tmpfile, 'w') as tmpf:
				tmpf.write('Most definitely not xml <as \n')

			paths = [path, path, tmpfile, path, tmpfile, tmpfile]

			with warnings.catch_warnings():
				warnings.simplefilter("ignore")
				(intensityData, sampleMetadata, featureMetadata) = importBrukerXML(paths)

		expectedSampleMetadata = pandas.DataFrame([], columns=['Sample File Name', 'Sample Base Name', 'expno', 'Path', 'Acquired Time', 'Run Order'])
		expectedSampleMetadata['Path'] = [path, path, path]
		#expectedSampleMetadata['Sample File Name'] = ['UnitTest5_expno840.100000.10r', 'UnitTest5_expno840.100000.10r', 'UnitTest5_expno840.100000.10r'] # change to Sample File Name matching Sample Base Name
		expectedSampleMetadata['Sample File Name'] = ['UnitTest5/840', 'UnitTest5/840', 'UnitTest5/840']
		expectedSampleMetadata['Sample Base Name'] = ['UnitTest5/840', 'UnitTest5/840', 'UnitTest5/840']
		expectedSampleMetadata['Acquired Time'] = [pandas.Timestamp('2017-08-15 13:06:45'), pandas.Timestamp('2017-08-15 13:06:45'), pandas.Timestamp('2017-08-15 13:06:45')]
		expectedSampleMetadata['Run Order'] = [0, 1, 2]
		expectedSampleMetadata['expno'] = [840, 840, 840]

		pandas.testing.assert_frame_equal(sampleMetadata, expectedSampleMetadata)
		self.assertEqual(intensityData.shape[0], 3)


class test_utilities_calibratePPMscale(unittest.TestCase):

	def L(self, x, x0, gamma):
		""" Return Lorentzian line shape at x with HWHM gamma """
		return gamma / numpy.pi / ((x-x0)**2 + gamma**2)


	def test_calibratePPM_raises(self):

		from nPYc.utilities._calibratePPMscale import calibratePPM

		self.assertRaises(NotImplementedError, calibratePPM, None, None, None, [0,1], None, spectrumType='J-RES')

		self.assertRaises(NotImplementedError, calibratePPM, 'singlet', 1, (0.5, 1.5), numpy.array([0,1, 2]), numpy.array([0,1,0]), align='interpolate')

		self.assertRaises(NotImplementedError, calibratePPM, 'Not an understood type', 1, (0.5, 1.5), numpy.array([0,1, 2]), numpy.array([0,1,0]))
		
		self.assertRaises(ValueError, calibratePPM, None, None, None, numpy.array([0,1, 2]), None, spectrumType='Not an understood type')


	def test_calibratePPM(self):

		from nPYc.utilities._calibratePPMscale import calibratePPM, referenceToSinglet, referenceToResolvedMultiplet

		with self.subTest(msg='Singlet'):

			specSize = numpy.random.randint(100, 1000)
			ppm = numpy.linspace(-1, 10, specSize)

			shift = ppm[numpy.random.randint(specSize - 20) + 10]
			targetShift = ppm[numpy.random.randint(specSize - 20) + 10]

			spectrum = L(ppm, shift, 0.001)

			spectrum = spectrum + L(ppm, shift + 3, 0.001) * 2

			(alignedSpectrum, alignedPPM, ppmShift) = calibratePPM('Singlet', targetShift, (shift - 1, shift + 1), ppm, spectrum, spectrumType='1D')

			obtainedPPM = referenceToSinglet(alignedSpectrum, alignedPPM, (targetShift - 1, targetShift + 1))

			self.assertEqual(alignedPPM[obtainedPPM], targetShift)


		with self.subTest(msg='Singlet, reversed'):

			specSize = numpy.random.randint(100, 1000)
			ppm = numpy.linspace(10, -1, specSize)

			shift = ppm[numpy.random.randint(specSize -20) + 10]
			targetShift = ppm[numpy.random.randint(specSize  - 20) + 10]

			spectrum = L(ppm, shift, 0.001)

			spectrum = spectrum + L(ppm, shift + 3, 0.001) * 2

			(alignedSpectrum, alignedPPM, ppmShift) = calibratePPM('Singlet', targetShift, (shift - 1, shift + 1), ppm, spectrum, spectrumType='1D')

			obtainedPPM = referenceToSinglet(alignedSpectrum, alignedPPM, (targetShift - 1, targetShift + 1))

			self.assertEqual(alignedPPM[obtainedPPM], targetShift)


		with self.subTest(msg='Doublet'):

			specSize = numpy.random.randint(500, 1000)
			ppm = numpy.linspace(4.8, 5.7, specSize)

			# Reserve below 5 ppm for decoy peaks
			shift = ppm[numpy.random.randint(numpy.max(numpy.where(ppm<5.1)), high=numpy.min(numpy.where(ppm>5.5)), size=1)[0]]
			targetShift = ppm[numpy.random.randint(numpy.max(numpy.where(ppm<5.1)), high=numpy.min(numpy.where(ppm>5.5)), size=1)[0]]

			baseline = L(ppm, 5.3, .09)

			peakOne = L(ppm, shift, 0.001)
			peakTwo = L(ppm, shift + 0.01, 0.001)

			peakThree = L(ppm, 4.81, 0.001)
			peakFour = L(ppm, 4.83, 0.001)

			spectrum = (baseline * 300) + peakOne + peakTwo + (peakThree + peakFour) * 5

			(alignedSpectrum, alignedPPM, ppmShift) = calibratePPM('Doublet', targetShift, (shift - 0.2, shift + 0.2), ppm, spectrum, spectrumType='1D')

			obtainedPPM = referenceToResolvedMultiplet(alignedSpectrum, alignedPPM, (targetShift - 0.2, targetShift + 0.2), 2)
			obtainedPPM = int(round(numpy.mean(obtainedPPM)))

			# Use all close because midpoint of pleaks may fall between the ppm grid
			numpy.testing.assert_allclose(alignedPPM[obtainedPPM], targetShift, atol=ppm[1]-ppm[0])


	def test_referenceToSinglet(self):

		from nPYc.utilities._calibratePPMscale import referenceToSinglet

		spectrum = numpy.array([0, 5, 6, 9, 6, 5, 0, 0, 0, 7, 8, 6, 0, 0, 0, 0, 0, 0, 0, 0])
		ppm = numpy.linspace(-10, 10, 20)

		with self.subTest(msg='Using whole spectrum'):

			obtainedPPM = referenceToSinglet(spectrum, ppm, (-10, 10))
			expectedDeltaPPM = 3

			self.assertEqual(obtainedPPM, expectedDeltaPPM)

		with self.subTest(msg='Using masked spectrum'):

			obtainedPPM = referenceToSinglet(spectrum, ppm, (-2, 8))
			expectedDeltaPPM = 10

			self.assertEqual(obtainedPPM, expectedDeltaPPM)


	def test_referenceToDoublet(self):

		from nPYc.utilities._calibratePPMscale import referenceToResolvedMultiplet

		ppm = numpy.linspace(4.8, 5.7, 900)

		baseline = L(ppm, 5.3, .09)

		peakOne = L(ppm, 5.225, 0.001)
		peakTwo = L(ppm, 5.235, 0.001)

		peakThree = L(ppm, 4.81, 0.001)
		peakFour = L(ppm, 4.83, 0.001)

		spectrum = (baseline * 300) + peakOne + peakTwo + (peakThree + peakFour) * 5


		with self.subTest(msg='Using whole spectrum'):

			obtainedPPM = referenceToResolvedMultiplet(spectrum, ppm, (0, 50), 2)
			expectedDeltaPPM = [10, 30]

			self.assertEqual(obtainedPPM, expectedDeltaPPM)

		with self.subTest(msg='Using masked spectrum'):

			obtainedPPM = referenceToResolvedMultiplet(spectrum, ppm, (5.1, 5.4), 2)
			expectedDeltaPPM = [425, 435]

			self.assertEqual(obtainedPPM, expectedDeltaPPM)


class test_utilities_nmr(unittest.TestCase):

	def test_interpolateSpectrum(self):

		# Test ripped off from the iterp1d test in scipy

		with self.subTest(msg='Single Spectrum'):

			x = numpy.linspace(-3,3,numpy.random.randint(10, 50))
			x2 = numpy.linspace(-3,3,numpy.random.randint(100, 200))

			# Test scaling sin down
			target = numpy.sin(x)
			interpolationStartPoint = numpy.sin(x2)

			result = nPYc.utilities._nmr.interpolateSpectrum(interpolationStartPoint, x2, x)

			numpy.testing.assert_allclose(target, result, atol=1e-3)

		with self.subTest(msg='Multiple spectra'):

			x = numpy.linspace(-3,3,numpy.random.randint(10, 50))
			x2 = numpy.linspace(-3,3,numpy.random.randint(100, 200))

			noSpectra = numpy.random.randint(5, 50)

			target = numpy.zeros((noSpectra, x.size))
			interpolationStartPoint = numpy.zeros((noSpectra, x2.size))

			for i in range(noSpectra):
				offset = numpy.random.randn()
				target[i, :] = numpy.sin(x + offset)
				interpolationStartPoint[i, :] = numpy.sin(x2 + offset)

			result = nPYc.utilities._nmr.interpolateSpectrum(interpolationStartPoint, x2, x)

			numpy.testing.assert_allclose(target, result, atol=1e-3)


	def test_interpolateSpectrum_raises(self):

		threeD = numpy.empty((3,3,3))

		self.assertRaises(ValueError, nPYc.utilities._nmr.interpolateSpectrum, threeD, None, None)


	def test_generateBaseName(self):
		from nPYc.utilities._nmr import generateBaseName

		sampleMetadata = pandas.DataFrame(['UnitTest3_Serum_Rack01_RCM_190116/10',
										   'UnitTest3_Serum_Rack01_RCM_190116/11',
										   'UnitTest3_Serum_Rack01_RCM_190116/23',
										   'UnitTest3_Serum_Rack01_RCM_190116/39'],
										   columns=['Sample File Name'])

		expectedBN = ['UnitTest3_Serum_Rack01_RCM_190116/10',
					  'UnitTest3_Serum_Rack01_RCM_190116/10',
					  'UnitTest3_Serum_Rack01_RCM_190116/20',
					  'UnitTest3_Serum_Rack01_RCM_190116/30']

		expectedExpno = [10, 11,  23, 39]

		obtainedBN, obtainedExpno = generateBaseName(sampleMetadata)

		numpy.testing.assert_equal(obtainedBN, expectedBN)
		numpy.testing.assert_equal(obtainedExpno, expectedExpno)


	def test_generateBaseName_definedexpno(self):
		from nPYc.utilities._nmr import generateBaseName

		sampleMetadata = pandas.DataFrame(['UnitTest3_Serum_Rack01_RCM_190116/10',
										   'UnitTest3_Serum_Rack01_RCM_190116/11',
										   'UnitTest3_Serum_Rack01_RCM_190116/23',
										   'UnitTest3_Serum_Rack01_RCM_190116/39'],
										   columns=['Sample File Name'])
		sampleMetadata['expno'] = [40, 33, 29, 81]

		expectedBN = ['UnitTest3_Serum_Rack01_RCM_190116/40',
					  'UnitTest3_Serum_Rack01_RCM_190116/30',
					  'UnitTest3_Serum_Rack01_RCM_190116/20',
					  'UnitTest3_Serum_Rack01_RCM_190116/80']

		obtainedBN, obtainedExpno = generateBaseName(sampleMetadata)

		numpy.testing.assert_equal(obtainedBN, expectedBN)


class test_utilities_linewidth(unittest.TestCase):

	def setUp(self):
		self.trueLineWidth = (5 - 0.1) * numpy.random.random_sample() + 0.1

		self.y = numpy.linspace(-10, 10, numpy.random.randint(500, 1000))
		self.x = L(self.y, 0, self.trueLineWidth)

		self.sf = 600


	def test_lineWidth_singlet(self):
		from nPYc.utilities._lineWidth import lineWidth

		calculatedLW = lineWidth(self.x, self.y, 1, [-5, 5], multiplicity='singlet')

		numpy.testing.assert_allclose(calculatedLW, self.trueLineWidth, atol=0.001)


	def test_lineWidth_sf(self):
		from nPYc.utilities._lineWidth import lineWidth

		sf = (1000 - 200) * numpy.random.random_sample() + 200
		trueLineWidth = 2
		trueLineWidth = trueLineWidth / sf

		x = L(self.y, 0, trueLineWidth)

		calculatedLW = lineWidth(x, self.y, sf, [-5, 5], multiplicity='singlet')

		numpy.testing.assert_allclose(calculatedLW, trueLineWidth * sf, atol=0.001)


	def test_lineWidth_doublet(self):
		from nPYc.utilities._lineWidth import lineWidth

		# Aproximates a 0.3 to 3hz range
		trueLineWidth = (3 - 0.3) * numpy.random.random_sample() + 0.3

		points = numpy.random.randint(500, 1000)
		x = numpy.linspace(1.322, 1.38, points)

		y1 = L(x, 1.342, trueLineWidth / self.sf)
		y2 = L(x, 1.354, trueLineWidth / self.sf)
		bl = numpy.linspace(500, 1, points)
		y = ((y1 + y2) * 10) + bl

		calculatedLW = lineWidth(y, x, self.sf, [1.322, 1.38], multiplicity='doublet')

		numpy.testing.assert_allclose(calculatedLW, trueLineWidth, atol=0.001)


	def test_lineWidth_quartet(self):
		from nPYc.utilities._lineWidth import lineWidth

		x = numpy.linspace(4.07, 4.145, 1000)

		# Aproximates a 0.3 to 3hz range
		trueLineWidth = (3 - 0.3) * numpy.random.random_sample() + 0.3

		y1 = L(x, 4.090, trueLineWidth / self.sf)
		y2 = L(x, 4.1013, trueLineWidth / self.sf)
		y3 = L(x, 4.113, trueLineWidth / self.sf)
		y4 = L(x, 4.124, trueLineWidth / self.sf)
		bl = numpy.linspace(10, 1, 1000)
		n = numpy.random.randn(1000)
		y = ((y1 + y4) / 3) + (y2 + y3) + bl

		calculatedLW = lineWidth(y, x, self.sf, [4.07, 4.145], multiplicity='quartet')

		numpy.testing.assert_allclose(calculatedLW, trueLineWidth, atol=0.001)


	def test_lineWidth_peakIntesityFraction(self):
		from nPYc.utilities._lineWidth import lineWidth

		x = numpy.linspace(4.07, 4.145, 1000)

		# Aproximates a 0.3 to 3hz range
		trueLineWidth = (3 - 0.3) * numpy.random.random_sample() + 0.3

		y1 = L(x, 4.090, trueLineWidth / self.sf)
		y2 = L(x, 4.1013, trueLineWidth / self.sf)
		y3 = L(x, 4.113, trueLineWidth / self.sf)
		y4 = L(x, 4.124, trueLineWidth / self.sf)
		bl = numpy.linspace(10, 1, 1000)
		n = numpy.random.randn(1000)
		y = ((y1 + y4) / 3) + (y2 + y3) + bl

		peakBaselineRatio = (y2.sum()/numpy.absolute(bl.sum())) * 100

		with self.subTest(msg='Peak ratio too low'):
			calculatedLW = lineWidth(y, x, self.sf, [4.07, 4.145], multiplicity='quartet', peakIntesityFraction=peakBaselineRatio+1)
			self.assertTrue(numpy.isnan(calculatedLW))


class test_utilities_fitPeak(unittest.TestCase):

	def test_integrateResonance(self):
		from nPYc.utilities._fitPeak import integrateResonance

		modeled = []
		true = []
		for i in range(10):
			lineWidth = (0.01 - 0.001) * numpy.random.random_sample() + 0.001

			magnitude = numpy.random.randn()

			specSize = numpy.random.randint(300, 700)

			baseline = numpy.random.randn(specSize)

			y = numpy.linspace(-.5, .5, specSize)
			peak = L(y, 0, lineWidth)
			x = (peak * magnitude * 100) + baseline

			intergral = integrateResonance(x, y, 0)

			modeled.append(integrateResonance(x, y, 0))
			true.append(sum(peak) * magnitude * 100)

		correlation = numpy.corrcoef(true, modeled)[0,1]
		self.assertGreaterEqual(correlation, 0.999)


	def test_fitPeak_returnType(self):
		from nPYc.utilities._fitPeak import fitPeak
		import lmfit

		trueLineWidth = (5 - 0.1) * numpy.random.random_sample() + 0.1

		y = numpy.linspace(-10, 10, numpy.random.randint(500, 1000))
		x = L(y, 0, trueLineWidth)

		fit = fitPeak(x, y, [-5, 5], 'singlet', maxLW=5, estLW=None)

		self.assertTrue(isinstance(fit, lmfit.model.ModelResult))


	def test_fitPeak_reversedscale(self):
		from nPYc.utilities._fitPeak import fitPeak
		import lmfit

		trueLineWidth = (5 - 0.1) * numpy.random.random_sample() + 0.1

		y = numpy.linspace(10, -10, numpy.random.randint(500, 1000))
		x = L(y, 0, trueLineWidth)

		fit = fitPeak(x, y, [-5, 5], 'singlet', maxLW=5, estLW=None)

		self.assertTrue(isinstance(fit, lmfit.model.ModelResult))


	def test_fitPeak_raises(self):
		from nPYc.utilities._fitPeak import fitPeak

		self.assertRaises(ValueError, fitPeak, numpy.array([0, 1]), numpy.array([0, 1]), [0,1], 'Too many')


if __name__ == '__main__':
	unittest.main()
