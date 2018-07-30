import pandas
import numpy
import sys
import unittest
from pandas.util.testing import assert_frame_equal
import os
import tempfile
import random
import string
import json
import copy
import warnings

sys.path.append("..")
import nPYc
from generateTestDataset import generateTestDataset
from nPYc.enumerations import VariableType

class test_dataset_synthetic(unittest.TestCase):
	"""
	Test Dataset object functions with synthetic data
	"""

	def setUp(self):
		# Load empty object and populate with sythetic data.

		# Empty object
		self.data = nPYc.Dataset()

		validChars = string.ascii_letters + string.digits
		# Function to generate random strings:
		def randomword(length):
			return ''.join(random.choice(validChars) for i in range(length))

		# Randomly sized intensity data
		self.name = randomword(10)
		self.noFeat = numpy.random.randint(3,100)
		self.noSamp = numpy.random.randint(3,100)
		self.data._intensityData = numpy.random.rand(self.noSamp,self.noFeat)

		self.data.sampleMetadata['Sample File Name'] = list(map(str, numpy.linspace(1, self.noSamp, num=self.noSamp, dtype=int)))
		self.data.sampleMetadata['Sample Metadata'] = [randomword(10) for x in range(0, self.noSamp)]
		self.data.featureMetadata['Feature Name'] = list(map(str, numpy.linspace(1, self.noFeat, num=self.noFeat, dtype=int)))
		self.data.featureMetadata['Feature Metadata'] = [randomword(10) for x in range(0, self.noFeat)]

		self.data.VariableType = VariableType.Discrete


	def test_nofeatures(self):

		self.assertEqual(self.data.noFeatures, self.noFeat)


	def test_name(self):

		self.data.name = self.name

		self.assertEqual(self.data.name, self.name)


	def test_name_raises(self):

		with self.assertRaises(TypeError):
			self.data.name = 5


	def test_normalisation(self):

		from nPYc.utilities import normalisation

		with self.subTest(msg='Check initialised with a NullNormaliser'):

			self.assertIsInstance(self.data.Normalisation, normalisation.NullNormaliser)

			numpy.testing.assert_equal(self.data.intensityData, self.data._intensityData)

		with self.subTest(msg='Check swap to TA normaliser'):

			self.data.Normalisation = normalisation.TotalAreaNormaliser()

			taNormaliser = normalisation.TotalAreaNormaliser()

			numpy.testing.assert_array_equal(self.data.intensityData, taNormaliser.normalise(self.data._intensityData))


	def test_normalisation_raises(self):

		with self.assertRaises(TypeError):
			self.data.Normalisation = 'Not a Normaliser'


	def test_nosamples(self):

		self.assertEqual(self.data.noSamples, self.noSamp)


	def test__repr__(self):

		pointer = id(self.data)
		reprString = str(self.data)
		testString = "<%s instance at %s, named %s, with %d samples, %d features>" % (nPYc.Dataset().__class__.__name__, pointer, nPYc.Dataset().__class__.__name__,  self.noSamp, self.noFeat)

		self.assertEqual(reprString, testString)


	def test_initialisemasks(self):

		self.data.initialiseMasks()

		featureMask = numpy.squeeze(numpy.ones([self.noFeat, 1], dtype=bool))
		sampleMask = numpy.squeeze(numpy.ones([self.noSamp, 1], dtype=bool))

		with self.subTest(msg='Checking featureMask.'):
			numpy.testing.assert_equal(self.data.featureMask, featureMask)
		with self.subTest(msg='Checking sampleMask.'):
			numpy.testing.assert_equal(self.data.sampleMask, sampleMask)


	def test_applymasks(self):

		# exclude feature 2, samples 1 and 3
		featureMask = numpy.squeeze(numpy.ones([self.noFeat, 1], dtype=bool))
		featureMask[1] = False
		sampleMask = numpy.squeeze(numpy.ones([self.noSamp, 1], dtype=bool))
		sampleMask[[0, 2]] = False
		expectedDataset = copy.deepcopy(self.data)
		expectedDataset.sampleMetadataExcluded = []
		expectedDataset.intensityDataExcluded = []
		expectedDataset.featureMetadataExcluded = []
		expectedDataset.excludedFlag = []
		expectedDataset.sampleMetadataExcluded.append(expectedDataset.sampleMetadata.loc[~sampleMask, :])
		expectedDataset.intensityDataExcluded.append(expectedDataset.intensityData[~sampleMask, :])
		expectedDataset.featureMetadataExcluded.append(expectedDataset.featureMetadata)
		expectedDataset.excludedFlag.append('Samples')
		expectedDataset.featureMetadataExcluded.append(expectedDataset.featureMetadata.loc[~featureMask, :])
		expectedDataset.intensityDataExcluded.append(expectedDataset.intensityData[sampleMask, :][:, ~featureMask])
		expectedDataset.sampleMetadataExcluded.append(expectedDataset.sampleMetadata.loc[sampleMask, :])
		expectedDataset.sampleMetadataExcluded[1].reset_index(drop=True, inplace=True)
		expectedDataset.excludedFlag.append('Features')
		expectedDataset.intensityData = expectedDataset.intensityData[sampleMask, :][:, featureMask]
		expectedDataset.sampleMetadata = expectedDataset.sampleMetadata.loc[sampleMask, :]
		expectedDataset.sampleMetadata.reset_index(drop=True, inplace=True)
		expectedDataset.featureMetadata = expectedDataset.featureMetadata.loc[featureMask, :]
		expectedDataset.featureMetadata.reset_index(drop=True, inplace=True)
		expectedDataset.initialiseMasks()

		maskedDataset = copy.deepcopy(self.data)
		maskedDataset.initialiseMasks()
		maskedDataset.featureMask[1] = False
		maskedDataset.sampleMask[[0, 2]] = False
		maskedDataset.applyMasks()

		with self.subTest(msg='Checking sampleMetadata'):
			pandas.util.testing.assert_frame_equal(maskedDataset.sampleMetadata, expectedDataset.sampleMetadata)
		with self.subTest(msg='Checking featureMetadata'):
			pandas.util.testing.assert_frame_equal(	maskedDataset.featureMetadata.reindex(sorted(maskedDataset.featureMetadata), axis=1), expectedDataset.featureMetadata.reindex(sorted(expectedDataset.featureMetadata), axis=1))
		with self.subTest(msg='Checking intensityData'):
			numpy.testing.assert_array_equal(maskedDataset.intensityData, expectedDataset.intensityData)
		with self.subTest(msg='Checking sampleMetadataExcluded'):
			pandas.util.testing.assert_frame_equal(expectedDataset.sampleMetadataExcluded[0], maskedDataset.sampleMetadataExcluded[0])
			pandas.util.testing.assert_frame_equal(expectedDataset.sampleMetadataExcluded[1], maskedDataset.sampleMetadataExcluded[1])
		with self.subTest(msg='Checking intensityMetadataExcluded'):
			numpy.testing.assert_array_equal(expectedDataset.intensityDataExcluded[0], maskedDataset.intensityDataExcluded[0])
			numpy.testing.assert_array_equal(expectedDataset.intensityDataExcluded[1], maskedDataset.intensityDataExcluded[1])
		with self.subTest(msg='Checking featureMetadataExcluded'):
			pandas.util.testing.assert_frame_equal(expectedDataset.featureMetadataExcluded[0], maskedDataset.featureMetadataExcluded[0])
			pandas.util.testing.assert_frame_equal(expectedDataset.featureMetadataExcluded[1], maskedDataset.featureMetadataExcluded[1])
		with self.subTest(msg='Checking excludedFlag'):
			self.assertListEqual(expectedDataset.excludedFlag, maskedDataset.excludedFlag)
		with self.subTest(msg='Checking featureMask'):
			numpy.testing.assert_array_equal(expectedDataset.featureMask, maskedDataset.featureMask)
		with self.subTest(msg='Checking sampleMask'):
			numpy.testing.assert_array_equal(expectedDataset.sampleMask, maskedDataset.sampleMask)


	def test_updateMasks_raises(self):

		self.data.initialiseMasks()

		with self.subTest(msg='Features not implemented'):
			self.assertRaises(NotImplementedError, self.data.updateMasks, filterFeatures=True)

		with self.subTest(msg='Sample Types'):
			self.assertRaises(TypeError, self.data.updateMasks, sampleTypes=[1, 2, 4])
			self.assertRaises(TypeError, self.data.updateMasks, sampleTypes='not a list')

		with self.subTest(msg='Assay Roles'):
			self.assertRaises(TypeError, self.data.updateMasks, assayRoles=[1, 2, 4])
			self.assertRaises(TypeError, self.data.updateMasks, assayRoles='not a list')


	def test_updateMasks_samples(self):

		from nPYc.enumerations import VariableType, DatasetLevel, AssayRole, SampleType

		dataset = nPYc.Dataset()

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
			dataset.updateMasks(withArtifactualFiltering=False, filterFeatures=False)

			numpy.testing.assert_array_equal(expectedSampleMask, dataset.sampleMask)

		with self.subTest(msg='Export SP and ER'):
			expectedSampleMask = numpy.array([False, False, False, False, False,  True,  True,  True,  True, True,  True, False, False, False, False, False,  True, False], dtype=bool)

			dataset.initialiseMasks()
			dataset.updateMasks(withArtifactualFiltering=False, filterFeatures=False,
								sampleTypes=[SampleType.StudyPool, SampleType.ExternalReference], 
								assayRoles=[AssayRole.PrecisionReference])

			numpy.testing.assert_array_equal(expectedSampleMask, dataset.sampleMask)

		with self.subTest(msg='Export Dilution Samples only'):
			expectedSampleMask = numpy.array([True,  True,  True,  True,  True, False, False, False, False, False, False, False, False, False, False, False, False, False], dtype=bool)

			dataset.initialiseMasks()
			dataset.updateMasks(withArtifactualFiltering=False, filterFeatures=False,
								sampleTypes=[SampleType.StudyPool], 
								assayRoles=[AssayRole.LinearityReference])

			numpy.testing.assert_array_equal(expectedSampleMask, dataset.sampleMask)

		with self.subTest(msg='No filtering'):
			expectedSampleMask = numpy.array([True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True], dtype=bool)

			dataset.initialiseMasks()
			dataset.updateMasks(withArtifactualFiltering=False, filterFeatures=False, filterSamples=False)

			numpy.testing.assert_array_equal(expectedSampleMask, dataset.sampleMask)


	def test_validateObject(self):
		testDataset = copy.deepcopy(self.data)
		# fake exclusions
		testDataset.sampleMetadataExcluded = []
		testDataset.intensityDataExcluded = []
		testDataset.featureMetadataExcluded = []
		testDataset.excludedFlag = []
		testDataset.sampleMetadataExcluded.append(testDataset.sampleMetadata.loc[[0, 2], :])
		testDataset.intensityDataExcluded.append(testDataset.intensityData[[0, 2], :])
		testDataset.featureMetadataExcluded.append(testDataset.featureMetadata)
		testDataset.excludedFlag.append('Samples')
		testDataset.featureMetadataExcluded.append(testDataset.featureMetadata.loc[3, :])
		testDataset.intensityDataExcluded.append(testDataset.intensityData[:, 3])
		testDataset.sampleMetadataExcluded.append(testDataset.sampleMetadata)
		testDataset.excludedFlag.append('Features')

		with self.subTest(msg='validateObject successful on empty Dataset'):
			goodDataset = nPYc.Dataset()
			self.assertTrue(goodDataset.validateObject(verbose=False, raiseError=False, raiseWarning=True))

		with self.subTest(msg='validateObject successful on basic dataset'):
			goodDataset = copy.deepcopy(testDataset)
			self.assertTrue(goodDataset.validateObject(verbose=False, raiseError=False, raiseWarning=True))

		with self.subTest(msg='check raise warnings'):
			badDataset = copy.deepcopy(testDataset)
			delattr(badDataset, 'featureMetadata')
			with warnings.catch_warnings(record=True) as w:
				# Cause all warnings to always be triggered.
				warnings.simplefilter("always")
				# warning
				result = badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=True)
				# check it generally worked
				self.assertEqual(result, False)
				# check each warning
				self.assertEqual(len(w), 2)
				assert issubclass(w[0].category, UserWarning)
				assert "Failure, no attribute 'self.featureMetadata'" in str(w[0].message)
				assert issubclass(w[1].category, UserWarning)
				assert "Does not conform to Dataset:" in str(w[1].message)

		with self.subTest(msg='check not raise warnings with raiseWarning=False'):
			badDataset = copy.deepcopy(testDataset)
			delattr(badDataset, 'featureMetadata')
			with warnings.catch_warnings(record=True) as w:
				# Cause all warnings to always be triggered.
				warnings.simplefilter("always")
				# warning
				result = badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False)
				# check it generally worked
				self.assertEqual(result, False)
				# check each warning
				self.assertEqual(len(w), 0)

		with self.subTest(msg='if self.Attributes does not exist'):
			badDataset = copy.deepcopy(testDataset)
			delattr(badDataset, 'Attributes')
			self.assertFalse(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False))
			self.assertRaises(AttributeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.Attributes is not a dict'):
			badDataset = copy.deepcopy(testDataset)
			badDataset.Attributes = 'not a dict'
			self.assertFalse(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False))
			self.assertRaises(TypeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='self.Attributes[\'Log\'] does not exist'):
			badDataset = copy.deepcopy(testDataset)
			del badDataset.Attributes['Log']
			self.assertFalse(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False))
			self.assertRaises(AttributeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.Attributes[\'Log\'] is not a list'):
			badDataset = copy.deepcopy(testDataset)
			badDataset.Attributes['Log'] = 'not a list'
			self.assertFalse(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False))
			self.assertRaises(TypeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='self.Attributes[\'dpi\'] does not exist'):
			badDataset = copy.deepcopy(testDataset)
			del badDataset.Attributes['dpi']
			self.assertFalse(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False))
			self.assertRaises(AttributeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.Attributes[\'dpi\'] is not an int'):
			badDataset = copy.deepcopy(testDataset)
			badDataset.Attributes['dpi'] = 'not an int'
			self.assertFalse(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False))
			self.assertRaises(TypeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='self.Attributes[\'figureSize\'] does not exist'):
			badDataset = copy.deepcopy(testDataset)
			del badDataset.Attributes['figureSize']
			self.assertFalse(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False))
			self.assertRaises(AttributeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.Attributes[\'figureSize\'] is not a list'):
			badDataset = copy.deepcopy(testDataset)
			badDataset.Attributes['figureSize'] = 'not a list'
			self.assertFalse(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False))
			self.assertRaises(TypeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.Attributes[\'figureSize\'] is not of length 2'):
			badDataset = copy.deepcopy(testDataset)
			badDataset.Attributes['figureSize'] = ['too short list']
			self.assertFalse(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False))
			self.assertRaises(ValueError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.Attributes[\'figureSize\'][0] is not an int or float'):
			badDataset = copy.deepcopy(testDataset)
			badDataset.Attributes['figureSize'][0] = 'not an int or float'
			self.assertFalse(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False))
			self.assertRaises(TypeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.Attributes[\'figureSize\'][1] is not an int or float'):
			badDataset = copy.deepcopy(testDataset)
			badDataset.Attributes['figureSize'][1] = 'not an int or float'
			self.assertFalse(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False))
			self.assertRaises(TypeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='self.Attributes[\'figureFormat\'] does not exist'):
			badDataset = copy.deepcopy(testDataset)
			del badDataset.Attributes['figureFormat']
			self.assertFalse(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False))
			self.assertRaises(AttributeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.Attributes[\'figureFormat\'] is not a str'):
			badDataset = copy.deepcopy(testDataset)
			badDataset.Attributes['figureFormat'] = 5.0
			self.assertFalse(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False))
			self.assertRaises(TypeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='self.Attributes[\'histBins\'] does not exist'):
			badDataset = copy.deepcopy(testDataset)
			del badDataset.Attributes['histBins']
			self.assertFalse(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False))
			self.assertRaises(AttributeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.Attributes[\'histBins\'] is not an int'):
			badDataset = copy.deepcopy(testDataset)
			badDataset.Attributes['histBins'] = 'not an int'
			self.assertFalse(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False))
			self.assertRaises(TypeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='self.Attributes[\'noFiles\'] does not exist'):
			badDataset = copy.deepcopy(testDataset)
			del badDataset.Attributes['noFiles']
			self.assertFalse(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False))
			self.assertRaises(AttributeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.Attributes[\'noFiles\'] is not an int'):
			badDataset = copy.deepcopy(testDataset)
			badDataset.Attributes['noFiles'] = 'not an int'
			self.assertFalse(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False))
			self.assertRaises(TypeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='self.Attributes[\'quantiles\'] does not exist'):
			badDataset = copy.deepcopy(testDataset)
			del badDataset.Attributes['quantiles']
			self.assertFalse(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False))
			self.assertRaises(AttributeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.Attributes[\'quantiles\'] is not a list'):
			badDataset = copy.deepcopy(testDataset)
			badDataset.Attributes['quantiles'] = 'not a list'
			self.assertFalse(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False))
			self.assertRaises(TypeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.Attributes[\'quantiles\'] is not of length 2'):
			badDataset = copy.deepcopy(testDataset)
			badDataset.Attributes['quantiles'] = ['too short list']
			self.assertFalse(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False))
			self.assertRaises(ValueError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.Attributes[\'quantiles\'][0] is not an int or float'):
			badDataset = copy.deepcopy(testDataset)
			badDataset.Attributes['quantiles'][0] = 'not an int or float'
			self.assertFalse(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False))
			self.assertRaises(TypeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.Attributes[\'quantiles\'][1] is not an int or float'):
			badDataset = copy.deepcopy(testDataset)
			badDataset.Attributes['quantiles'][1] = 'not an int or float'
			self.assertFalse(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False))
			self.assertRaises(TypeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='self.Attributes[\'sampleMetadataNotExported\'] does not exist'):
			badDataset = copy.deepcopy(testDataset)
			del badDataset.Attributes['sampleMetadataNotExported']
			self.assertFalse(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False))
			self.assertRaises(AttributeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.Attributes[\'sampleMetadataNotExported\'] is not a list'):
			badDataset = copy.deepcopy(testDataset)
			badDataset.Attributes['sampleMetadataNotExported'] = 'not a list'
			self.assertFalse(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False))
			self.assertRaises(TypeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='self.Attributes[\'featureMetadataNotExported\'] does not exist'):
			badDataset = copy.deepcopy(testDataset)
			del badDataset.Attributes['featureMetadataNotExported']
			self.assertFalse(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False))
			self.assertRaises(AttributeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.Attributes[\'featureMetadataNotExported\'] is not a list'):
			badDataset = copy.deepcopy(testDataset)
			badDataset.Attributes['featureMetadataNotExported'] = 'not a list'
			self.assertFalse(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False))
			self.assertRaises(TypeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='self.Attributes[\'analyticalMeasurements\'] does not exist'):
			badDataset = copy.deepcopy(testDataset)
			del badDataset.Attributes['analyticalMeasurements']
			self.assertFalse(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False))
			self.assertRaises(AttributeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.Attributes[\'analyticalMeasurements\'] is not a dict'):
			badDataset = copy.deepcopy(testDataset)
			badDataset.Attributes['analyticalMeasurements'] = 'not a dict'
			self.assertFalse(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False))
			self.assertRaises(TypeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='self.Attributes[\'excludeFromPlotting\'] does not exist'):
			badDataset = copy.deepcopy(testDataset)
			del badDataset.Attributes['excludeFromPlotting']
			self.assertFalse(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False))
			self.assertRaises(AttributeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.Attributes[\'excludeFromPlotting\'] is not a list'):
			badDataset = copy.deepcopy(testDataset)
			badDataset.Attributes['excludeFromPlotting'] = 'not a list'
			self.assertFalse(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False))
			self.assertRaises(TypeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='self.VariableType does not exist'):
			badDataset = copy.deepcopy(testDataset)
			delattr(badDataset, 'VariableType')
			self.assertFalse(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False))
			self.assertRaises(AttributeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='self._Normalisation does not exist'):
			badDataset = copy.deepcopy(testDataset)
			delattr(badDataset, '_Normalisation')
			self.assertFalse(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False))
			self.assertRaises(AttributeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='self._Normalisation is not Normaliser ABC'):
			badDataset = copy.deepcopy(testDataset)
			badDataset._Normalisation = 'not Normaliser ABC'
			self.assertFalse(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False))
			self.assertRaises(TypeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='self._name does not exist'):
			badDataset = copy.deepcopy(testDataset)
			delattr(badDataset, '_name')
			self.assertFalse(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False))
			self.assertRaises(AttributeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self._name is not a str'):
			badDataset = copy.deepcopy(testDataset)
			badDataset._name = 5.
			self.assertFalse(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False))
			self.assertRaises(TypeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='self._intensityData does not exist'):
			badDataset = copy.deepcopy(testDataset)
			delattr(badDataset, '_intensityData')
			self.assertFalse(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False))
			self.assertRaises(AttributeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self._intensityData is not a numpy.ndarray'):
			badDataset = copy.deepcopy(testDataset)
			badDataset._intensityData = 5.
			self.assertFalse(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False))
			self.assertRaises(TypeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='self.sampleMetadata does not exist'):
			badDataset = copy.deepcopy(testDataset)
			delattr(badDataset, 'sampleMetadata')
			self.assertFalse(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False))
			self.assertRaises(AttributeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.sampleMetadata is not a pandas.DataFrame'):
			badDataset = copy.deepcopy(testDataset)
			badDataset.sampleMetadata = 5.
			self.assertFalse(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False))
			self.assertRaises(TypeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.sampleMetadata does not have a Sample File Name column'):
			badDataset = copy.deepcopy(testDataset)
			badDataset.sampleMetadata.drop(['Sample File Name'], axis=1, inplace=True)
			self.assertFalse(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False))
			self.assertRaises(LookupError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.sampleMetadata does not have an AssayRole column'):
			badDataset = copy.deepcopy(testDataset)
			badDataset.sampleMetadata.drop(['AssayRole'], axis=1, inplace=True)
			self.assertFalse(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False))
			self.assertRaises(LookupError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.sampleMetadata does not have a SampleType column'):
			badDataset = copy.deepcopy(testDataset)
			badDataset.sampleMetadata.drop(['SampleType'], axis=1, inplace=True)
			self.assertFalse(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False))
			self.assertRaises(LookupError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.sampleMetadata does not have a Dilution column'):
			badDataset = copy.deepcopy(testDataset)
			badDataset.sampleMetadata.drop(['Dilution'], axis=1, inplace=True)
			self.assertFalse(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False))
			self.assertRaises(LookupError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.sampleMetadata does not have a Batch column'):
			badDataset = copy.deepcopy(testDataset)
			badDataset.sampleMetadata.drop(['Batch'], axis=1, inplace=True)
			self.assertFalse(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False))
			self.assertRaises(LookupError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.sampleMetadata does not have a Correction Batch column'):
			badDataset = copy.deepcopy(testDataset)
			badDataset.sampleMetadata.drop(['Correction Batch'], axis=1, inplace=True)
			self.assertFalse(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False))
			self.assertRaises(LookupError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.sampleMetadata does not have a Run Order column'):
			badDataset = copy.deepcopy(testDataset)
			badDataset.sampleMetadata.drop(['Run Order'], axis=1, inplace=True)
			self.assertFalse(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False))
			self.assertRaises(LookupError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.sampleMetadata does not have an Acquired Time column'):
			badDataset = copy.deepcopy(testDataset)
			badDataset.sampleMetadata.drop(['Acquired Time'], axis=1, inplace=True)
			self.assertFalse(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False))
			self.assertRaises(LookupError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.sampleMetadata does not have a Sample Base Name column'):
			badDataset = copy.deepcopy(testDataset)
			badDataset.sampleMetadata.drop(['Sample Base Name'], axis=1, inplace=True)
			self.assertFalse(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False))
			self.assertRaises(LookupError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.sampleMetadata does not have a Sampling ID column'):
			badDataset = copy.deepcopy(testDataset)
			badDataset.sampleMetadata.drop(['Sampling ID'], axis=1, inplace=True)
			self.assertFalse(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False))
			self.assertRaises(LookupError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.sampleMetadata does not have a Sampling ID column'):
			badDataset = copy.deepcopy(testDataset)
			badDataset.sampleMetadata.drop(['Exclusion Details'], axis=1, inplace=True)
			self.assertFalse(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False))
			self.assertRaises(LookupError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='self.featureMetadata does not exist'):
			badDataset = copy.deepcopy(testDataset)
			delattr(badDataset, 'featureMetadata')
			self.assertFalse(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False))
			self.assertRaises(AttributeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.featureMetadata is not a pandas.DataFrame'):
			badDataset = copy.deepcopy(testDataset)
			badDataset.featureMetadata = 5.
			self.assertFalse(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False))
			self.assertRaises(TypeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.featureMetadata does not have a Feature Name column'):
			badDataset = copy.deepcopy(testDataset)
			badDataset.featureMetadata.drop(['Feature Name'], axis=1, inplace=True)
			self.assertFalse(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False))
			self.assertRaises(LookupError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.sampleMask does not exist'):
			badDataset = copy.deepcopy(testDataset)
			delattr(badDataset, 'sampleMask')
			self.assertFalse(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False))
			self.assertRaises(AttributeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.sampleMask is not a numpy.ndarray'):
			badDataset = copy.deepcopy(testDataset)
			badDataset.sampleMask = 'not an array'
			self.assertFalse(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False))
			self.assertRaises(TypeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.sampleMask are not a bool'):
			badDataset = copy.deepcopy(testDataset)
			badDataset.sampleMask = numpy.matrix([[5.]])
			self.assertFalse(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False))
			self.assertRaises(ValueError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.featureMask does not exist'):
			badDataset = copy.deepcopy(testDataset)
			delattr(badDataset, 'featureMask')
			self.assertFalse(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False))
			self.assertRaises(AttributeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.featureMask is not a numpy.ndarray'):
			badDataset = copy.deepcopy(testDataset)
			badDataset.featureMask = 'not an array'
			self.assertFalse(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False))
			self.assertRaises(TypeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.featureMask are not a bool'):
			badDataset = copy.deepcopy(testDataset)
			badDataset.featureMask = numpy.matrix([[5.]])
			self.assertFalse(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False))
			self.assertRaises(ValueError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.sampleMetadataExcluded does not exist'):
			badDataset = copy.deepcopy(testDataset)
			delattr(badDataset, 'sampleMetadataExcluded')
			self.assertFalse(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False))
			self.assertRaises(AttributeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.sampleMetadataExcluded is not a list'):
			badDataset = copy.deepcopy(testDataset)
			badDataset.sampleMetadataExcluded = 'not a list'
			self.assertFalse(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False))
			self.assertRaises(TypeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.intensityDataExcluded does not exist'):
			badDataset = copy.deepcopy(testDataset)
			delattr(badDataset, 'intensityDataExcluded')
			self.assertFalse(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False))
			self.assertRaises(AttributeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.intensityDataExcluded is not a list'):
			badDataset = copy.deepcopy(testDataset)
			badDataset.intensityDataExcluded = 'not a list'
			self.assertFalse(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False))
			self.assertRaises(TypeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.intensityDataExcluded does not have the same number of exclusions as self.sampleMetadataExcluded'):
			badDataset = copy.deepcopy(testDataset)
			badDataset.intensityDataExcluded = [[1], [1], [1]]
			self.assertFalse(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False))
			self.assertRaises(ValueError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.featureMetadataExcluded does not exist'):
			badDataset = copy.deepcopy(testDataset)
			delattr(badDataset, 'featureMetadataExcluded')
			self.assertFalse(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False))
			self.assertRaises(AttributeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.featureMetadataExcluded is not a list'):
			badDataset = copy.deepcopy(testDataset)
			badDataset.featureMetadataExcluded = 'not a list'
			self.assertFalse(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False))
			self.assertRaises(TypeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.featureMetadataExcluded does not have the same number of exclusions as self.sampleMetadataExcluded'):
			badDataset = copy.deepcopy(testDataset)
			badDataset.featureMetadataExcluded = [[1], [1], [1]]
			self.assertFalse(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False))
			self.assertRaises(ValueError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.excludedFlag does not exist'):
			badDataset = copy.deepcopy(testDataset)
			delattr(badDataset, 'excludedFlag')
			self.assertFalse(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False))
			self.assertRaises(AttributeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.excludedFlag is not a list'):
			badDataset = copy.deepcopy(testDataset)
			badDataset.excludedFlag = 'not a list'
			self.assertFalse(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False))
			self.assertRaises(TypeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.excludedFlag does not have the same number of exclusions as self.sampleMetadataExcluded'):
			badDataset = copy.deepcopy(testDataset)
			badDataset.excludedFlag = [[1], [1], [1]]
			self.assertFalse(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False))
			self.assertRaises(ValueError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)


	def test_exportDataset(self):

		with tempfile.TemporaryDirectory() as tmpdirname:
			tmpdirname = os.path.join(tmpdirname, 'testOutput')
			projectName = 'tempFile'
			self.data.name = projectName

			self.data.exportDataset(tmpdirname, saveFormat='CSV', withExclusions=False, filterMetadata=False)

			# Load exported data back in, cast types back to str
			filePath = os.path.join(tmpdirname, projectName + '_intensityData.csv')
			intensityData = numpy.loadtxt(filePath, dtype=float, delimiter=',')

			filePath = os.path.join(tmpdirname, projectName + '_featureMetadata.csv')
			featureMetadata = pandas.read_csv(filePath, index_col=0)
			featureMetadata['Feature Name'] = featureMetadata['Feature Name'].astype(str)

			filePath = os.path.join(tmpdirname, projectName + '_sampleMetadata.csv')
			sampleMetadata = pandas.read_csv(filePath, index_col=0)
			sampleMetadata['Sample File Name'] = sampleMetadata['Sample File Name'].astype(str)

		numpy.testing.assert_array_almost_equal_nulp(self.data.intensityData, intensityData)
		pandas.util.testing.assert_frame_equal(self.data.featureMetadata, featureMetadata, check_dtype=False)
		pandas.util.testing.assert_frame_equal(self.data.sampleMetadata, sampleMetadata, check_dtype=False)


	def test_exportcsv(self):
		"""
		Check that csvs as written match the data tables in memory.
		"""

		with tempfile.TemporaryDirectory() as tmpdirname:
			projectName = os.path.join(tmpdirname, 'tempFile')
			self.data._exportCSV(projectName, escapeDelimiters=False)

			# Load exported data back in, cast types back to str
			filePath = os.path.join(tmpdirname, projectName + '_intensityData.csv')
			intensityData = numpy.loadtxt(filePath, dtype=float, delimiter=',')

			filePath = os.path.join(tmpdirname, projectName + '_featureMetadata.csv')
			featureMetadata = pandas.read_csv(filePath, index_col=0)
			featureMetadata['Feature Name'] = featureMetadata['Feature Name'].astype(str)

			filePath = os.path.join(tmpdirname, projectName + '_sampleMetadata.csv')
			sampleMetadata = pandas.read_csv(filePath, index_col=0)
			sampleMetadata['Sample File Name'] = sampleMetadata['Sample File Name'].astype(str)

		numpy.testing.assert_array_almost_equal_nulp(self.data.intensityData, intensityData)
		pandas.util.testing.assert_frame_equal(self.data.featureMetadata, featureMetadata, check_dtype=False)
		pandas.util.testing.assert_frame_equal(self.data.sampleMetadata, sampleMetadata, check_dtype=False)


	def test_exportunifiedcsv(self):
		"""
		Verify unified csvs are written correctly.
		"""
		with tempfile.TemporaryDirectory() as tmpdirname:
			projectName = os.path.join(tmpdirname, 'tempFile')
			self.data._exportUnifiedCSV(projectName)

			filePath = os.path.join(projectName + '_combinedData.csv')
			savedData = pandas.read_csv(filePath)

			intensityData = savedData.iloc[self.data.featureMetadata.shape[1]:, self.data.sampleMetadata.shape[1]+1:].apply(pandas.to_numeric)

			# Extract feature metadata
			featureMetadata = savedData.iloc[:self.data.featureMetadata.shape[1], self.data.sampleMetadata.shape[1]+1:].T
			featureMetadata.columns = savedData.iloc[:self.data.featureMetadata.shape[1], 0]
			featureMetadata.reset_index(drop=True, inplace=True)
			featureMetadata.columns.name = None

			# Extract sample metadata
			sampleMetadata = savedData.iloc[self.data.featureMetadata.shape[1]:, :self.data.sampleMetadata.shape[1]+1]
			sampleMetadata['Sample File Name'] = sampleMetadata['Sample File Name'].astype(int).astype(str)
			sampleMetadata.drop('Unnamed: 0', axis=1, inplace=True)
			sampleMetadata.reset_index(drop=True, inplace=True)

		numpy.testing.assert_array_almost_equal(self.data.intensityData, intensityData)
		pandas.util.testing.assert_frame_equal(self.data.featureMetadata, featureMetadata, check_dtype=False)
		pandas.util.testing.assert_frame_equal(self.data.sampleMetadata, sampleMetadata, check_dtype=False)


	def test_exportdataset_withexclusions(self):
		"""
		Test that csv files saved with exclusions match the dataset generated after exclusions are applied.
		"""

		self.data.initialiseMasks()

		featureToExclude = numpy.random.randint(1, self.noFeat, size=numpy.random.randint(1, int(self.noFeat / 2) + 1))
		sampleToExclude = numpy.random.randint(1, self.noSamp, size=numpy.random.randint(1, int(self.noSamp / 2) + 1))

		self.data.featureMask[featureToExclude] = False
		self.data.sampleMask[sampleToExclude] = False

		with tempfile.TemporaryDirectory() as tmpdirname:
			projectName = 'tempFile'
			self.data.name = projectName

			self.data.exportDataset(destinationPath=tmpdirname, saveFormat='CSV', withExclusions=True, filterMetadata=False)

			# Load exported data back in, cast types back to str
			filePath = os.path.join(tmpdirname, projectName + '_intensityData.csv')
			intensityData = numpy.loadtxt(filePath, dtype=float, delimiter=',')


			filePath = os.path.join(tmpdirname, projectName + '_featureMetadata.csv')
			featureMetadata = pandas.read_csv(filePath)
			featureMetadata.drop('Unnamed: 0', axis=1, inplace=True)
			featureMetadata['Feature Name'] = featureMetadata['Feature Name'].astype(str)


			filePath = os.path.join(tmpdirname, projectName + '_sampleMetadata.csv')
			sampleMetadata = pandas.read_csv(filePath)
			sampleMetadata.drop('Unnamed: 0', axis=1, inplace=True)
			sampleMetadata['Sample File Name'] = sampleMetadata['Sample File Name'].astype(str)

		# Loaded data and data after _applyMasks() should match
		self.data.applyMasks()

		numpy.testing.assert_equal(self.data.intensityData, intensityData)
		pandas.util.testing.assert_frame_equal(self.data.featureMetadata, featureMetadata, check_dtype=False)
		pandas.util.testing.assert_frame_equal(self.data.sampleMetadata, sampleMetadata, check_dtype=False)


	def test_exportdataset_filterMetadata(self):
		"""
		Test that csv files saved with `filterMetadata` match the dataset with these sampleMetadata and featureMetadata columns removed.
		"""
		featureMetaCols = list(set(self.data.featureMetadata.columns.tolist()) - set(['Feature Name']))
		sampleMetaCols  = list(set(self.data.sampleMetadata.columns.tolist()) - set(['Sample File Name']))
		randomFeatCols = [1] #numpy.random.randint(0, len(featureMetaCols), size=numpy.random.randint(int(len(featureMetaCols) / 2) + 1)).tolist()
		randomSampCols = numpy.random.randint(0, len(sampleMetaCols), size=numpy.random.randint(int(len(sampleMetaCols) / 3) + 1)).tolist()
		self.data.Attributes['featureMetadataNotExported'] = [x for i, x in enumerate(featureMetaCols) if i in randomFeatCols] + ['not an existing featureMeta column']
		self.data.Attributes['sampleMetadataNotExported']  = [x for i, x in enumerate(sampleMetaCols) if i in randomSampCols]  + ['not an existing sampleMeta column']

		with tempfile.TemporaryDirectory() as tmpdirname:
			projectName = 'tempFile'
			self.data.name = projectName

			self.data.exportDataset(destinationPath=tmpdirname, saveFormat='CSV', withExclusions=False, filterMetadata=True)

			# Load exported data back in, cast types back to str
			filePath = os.path.join(tmpdirname, projectName + '_intensityData.csv')
			intensityData = numpy.loadtxt(filePath, dtype=float, delimiter=',')

			filePath = os.path.join(tmpdirname, projectName + '_featureMetadata.csv')
			featureMetadata = pandas.read_csv(filePath)
			featureMetadata.drop('Unnamed: 0', axis=1, inplace=True)
			featureMetadata['Feature Name'] = featureMetadata['Feature Name'].astype(str)

			filePath = os.path.join(tmpdirname, projectName + '_sampleMetadata.csv')
			sampleMetadata = pandas.read_csv(filePath)
			sampleMetadata.drop('Unnamed: 0', axis=1, inplace=True)
			sampleMetadata['Sample File Name'] = sampleMetadata['Sample File Name'].astype(str)

		# Remove the same columns
		self.data.featureMetadata.drop([x for i, x in enumerate(featureMetaCols) if i in randomFeatCols], axis=1, inplace=True)
		self.data.sampleMetadata.drop( [x for i, x in enumerate(sampleMetaCols) if i in randomSampCols],  axis=1, inplace=True)

		numpy.testing.assert_equal(self.data.intensityData, intensityData)
		pandas.util.testing.assert_frame_equal(self.data.featureMetadata, featureMetadata, check_dtype=False)
		pandas.util.testing.assert_frame_equal(self.data.sampleMetadata, sampleMetadata, check_dtype=False)


	def test_exportDataset_raises(self):

		self.assertRaises(TypeError, self.data.exportDataset, destinationPath=1)

		self.assertRaises(ValueError, self.data.exportDataset, saveFormat='Not known', withExclusions=False)

		self.assertRaises(TypeError, self.data.exportDataset, withExclusions='no')

		self.assertRaises(TypeError, self.data.exportDataset, filterMetadata='no')


	def test_print_log(self):
		"""
		Verify logged items are printed correctly.
		"""
		from datetime import datetime

		time1 = datetime.now()
		time2 = datetime(2016, 1, 30, 12, 15)
		time3 = datetime(2016, 10, 10, 13, 30)

		str1 = 'Log entry 1'
		str2 = 'Log entry 2'
		str3 = 'Log entry 3'

		self.data.Attributes['Log'].append([time1, str1])
		self.data.Attributes['Log'].append([time2, str2])
		self.data.Attributes['Log'].append([time3, str3])

		output = self.data.Attributes['Log'][0][0].strftime(self.data._timestampFormat)
		output = output + "\t"
		output = output + self.data.Attributes['Log'][0][1]
		output = output + "\n"

		output = output + time1.strftime(self.data._timestampFormat)
		output = output + "\t"
		output = output + str1
		output = output + "\n"

		output = output + time2.strftime(self.data._timestampFormat)
		output = output + "\t"
		output = output + str2
		output = output + "\n"

		output = output + time3.strftime(self.data._timestampFormat)
		output = output + "\t"
		output = output + str3
		output = output + "\n"

		self.assertEqual(self.data.log, output)


	def test_exclude_samples(self):

		exclusionList = numpy.random.randint(1, self.noSamp, size=numpy.random.randint(1, int(self.noSamp / 2) + 1))
		exclusionList = set(exclusionList)
		exclusionList = list(exclusionList)
		self.data.initialiseMasks()

		exclusionsStr = list(map(str, exclusionList))
		exclusionsStr.append('Not a sample in the list')

		missingSamples = self.data.excludeSamples(exclusionsStr, on='Sample File Name', message='Test Excluded')

		exclusionList = [x - 1 for x in exclusionList]

		expectedSampleMask = numpy.squeeze(numpy.ones([self.noSamp, 1], dtype=bool))
		expectedSampleMask[numpy.ix_(exclusionList)] = False

		numpy.testing.assert_array_equal(self.data.sampleMask, expectedSampleMask)
		self.assertEqual(missingSamples, ['Not a sample in the list'])


	def test_exclude_samples_raises(self):

		exclusionList = numpy.random.randint(1, self.noSamp, size=numpy.random.randint(1, int(self.noSamp / 2) + 1))
		exclusionList = set(exclusionList)
		exclusionList = list(exclusionList)
		self.data.initialiseMasks()

		self.assertRaises(ValueError, self.data.excludeSamples, map(str, exclusionList), on='Not a real key', message='Test Excluded')
		self.assertRaises(TypeError, self.data.excludeSamples, map(str, exclusionList), on='Sample File Name', message=list())


	def test_exclude_features_discrete(self):

		exclusionList = numpy.random.randint(1, self.noFeat, size=numpy.random.randint(1, int(self.noFeat / 2) + 1))
		exclusionList = set(exclusionList)
		exclusionList = list(exclusionList)
		self.data.initialiseMasks()

		exclusionsStr = list(map(str, exclusionList))
		exclusionsStr.append('Not a feature in the list')

		missingFeatures = self.data.excludeFeatures(exclusionsStr, on='Feature Name', message='Test Excluded')

		exclusionList = [x - 1 for x in exclusionList]

		expectedFeatureMask = numpy.squeeze(numpy.ones([self.noFeat, 1], dtype=bool))
		expectedFeatureMask[numpy.ix_(exclusionList)] = False

		numpy.testing.assert_array_equal(self.data.featureMask, expectedFeatureMask)
		self.assertEqual(missingFeatures, ['Not a feature in the list'])


	def test_exclude_features_spectral(self):

		noSamp = numpy.random.randint(5, high=10, size=None)
		noFeat = numpy.random.randint(500, high=1000, size=None)

		dataset = generateTestDataset(noSamp, noFeat, dtype='NMRDataset', variableType=VariableType.Spectral)

		ranges = [(1,2), (8.5, 7)]

		mask = numpy.ones_like(dataset.featureMask, dtype=bool)
		for spRange in ranges:
			localMask = numpy.logical_or(dataset.featureMetadata['ppm'] < min(spRange),
				 						 dataset.featureMetadata['ppm'] > max(spRange))

			mask = numpy.logical_and(mask, localMask)

		dataset.excludeFeatures(ranges, on='ppm')

		numpy.testing.assert_array_equal(mask, dataset.featureMask)


	def test_exclude_features_raises(self):

		exclusionList = numpy.random.randint(1, self.noFeat, size=numpy.random.randint(1, int(self.noFeat / 2) + 1))
		exclusionList = set(exclusionList)
		exclusionList = list(exclusionList)
		self.data.initialiseMasks()

		self.assertRaises(ValueError, self.data.excludeFeatures, map(str, exclusionList), on='Not a real key', message='Test Excluded')
		self.assertRaises(TypeError, self.data.excludeFeatures, map(str, exclusionList), on='Feature Name', message=list())

		self.data.VariableType = 'Not a real type'
		self.assertRaises(ValueError, self.data.excludeFeatures, ['unimportant'], on='Feature Name', message='Test Excluded')


	def test_exclude_features_warns(self):

		self.data.VariableType = VariableType.Spectral
		self.assertWarns(UserWarning, self.data.excludeFeatures, [(1, 1)], on='Feature Name', message='Test Excluded')


	def test_get_features_discrete(self):

		self.data.VariableType = nPYc.enumerations.VariableType.Discrete
		self.data.initialiseMasks()

		with self.subTest(msg='List of features'):
			# Select a random set of features
			featureList = numpy.random.randint(1, self.noFeat, size=numpy.random.randint(1, int(self.noFeat / 2) + 1))
			featureNames = [*self.data.featureMetadata.loc[featureList, 'Feature Name']]

			features, measuments = self.data.getFeatures(featureNames, by='Feature Name')

			numpy.testing.assert_array_equal(self.data.intensityData[:, featureList], measuments)
			pandas.util.testing.assert_frame_equal(self.data.featureMetadata.iloc[featureList], features)

		with self.subTest(msg='Single feature'):
			# Select a random set of features
			featureList = numpy.random.randint(1, self.noFeat)
			featureName = self.data.featureMetadata.loc[featureList, 'Feature Name']

			features, measuments = self.data.getFeatures(featureName, by='Feature Name')

			numpy.testing.assert_array_equal(self.data.intensityData[:, featureList], numpy.squeeze(measuments))
			pandas.util.testing.assert_frame_equal(self.data.featureMetadata.iloc[[featureList]], features)


	def test_get_features_spectral(self):

		spectrumRange = (-5, 5)

		data = nPYc.Dataset()

		validChars = string.ascii_letters + string.digits
		# Function to generate random strings:

		# Randomly sized intensity data
		noFeat = numpy.random.randint(100,1000)
		noSamp = numpy.random.randint(3,100)
		data.intensityData = numpy.random.rand(noSamp,noFeat)

		data.sampleMetadata = pandas.DataFrame(numpy.linspace(1,noSamp, num=noSamp, dtype=int), columns=['Sample File Name']).astype(str)
		data.featureMetadata = pandas.DataFrame(numpy.linspace(spectrumRange[0],spectrumRange[1], num=noFeat, dtype=float), columns=['ppm'])

		data.VariableType = nPYc.enumerations.VariableType.Spectral
		data.Attributes['Feature Names'] = 'ppm'
		data.initialiseMasks()

		with self.subTest(msg='List of features'):
			# Select a random set of features
			# Between two and five ranges
			noRanges = numpy.random.randint(2, 5)
			ppmRange = list()
			rangeMask = numpy.zeros_like(data.featureMask)
			for i in range(0, noRanges):
				startIndex = numpy.random.randint(0, noFeat/2)
				endIndex = startIndex + numpy.random.randint(2, 10)
				if endIndex > data.noFeatures:
					endIndex = data.noFeatures
				rangeMask[startIndex:endIndex+1] = True

				ppmRange.append((data.featureMetadata.loc[startIndex, 'ppm'], data.featureMetadata.loc[endIndex, 'ppm']))

			features, measuments = data.getFeatures(ppmRange, by='ppm')
			pandas.util.testing.assert_frame_equal(data.featureMetadata.loc[rangeMask], features)
			numpy.testing.assert_array_equal(data.intensityData[:, rangeMask], measuments)


		with self.subTest(msg='Single feature'):
			# Select a random set of features
			startIndex = numpy.random.randint(0, noFeat/2)
			endIndex = startIndex + numpy.random.randint(2, noFeat/10)

			ppmRange = (data.featureMetadata.loc[startIndex, 'ppm'], data.featureMetadata.loc[endIndex, 'ppm'])

			features, measuments = data.getFeatures(ppmRange, by='ppm')

			numpy.testing.assert_array_equal(data.intensityData[:, startIndex:endIndex+1], measuments)
			pandas.util.testing.assert_frame_equal(data.featureMetadata.iloc[startIndex:endIndex+1], features)

		with self.subTest(msg='Fliped range feature'):
			# Select a random set of features
			startIndex = numpy.random.randint(0, noFeat/2)
			endIndex = startIndex + numpy.random.randint(2, noFeat/10)

			ppmRange = (data.featureMetadata.loc[endIndex, 'ppm'], data.featureMetadata.loc[startIndex, 'ppm'])

			features, measuments = data.getFeatures(ppmRange, by='ppm')

			numpy.testing.assert_array_equal(data.intensityData[:, startIndex:endIndex+1], measuments)
			pandas.util.testing.assert_frame_equal(data.featureMetadata.iloc[startIndex:endIndex+1], features)


	def test_get_features_autofeaturename(self):
		self.data.initialiseMasks()
		featureNames = ''.join(random.choice(string.ascii_letters + string.digits) for i in range(numpy.random.randint(3,15)))

		self.data.VariableType = nPYc.enumerations.VariableType.Discrete
		self.data.Attributes['Feature Names'] = featureNames

		self.data.featureMetadata.rename(columns={'Feature Name': featureNames}, inplace=True)

		# Select a random set of features
		featureList = numpy.random.randint(1, self.noFeat, size=numpy.random.randint(1, int(self.noFeat / 2) + 1))
		featureNames = [*self.data.featureMetadata.loc[featureList, featureNames]]

		features, measuments = self.data.getFeatures(featureNames)

		numpy.testing.assert_array_equal(self.data.intensityData[:, featureList], measuments)
		pandas.util.testing.assert_frame_equal(self.data.featureMetadata.iloc[featureList], features)


	def test_get_features_raises(self):

		self.data.VariableType = nPYc.enumerations.VariableType.Discrete
		self.assertRaises(KeyError, self.data.getFeatures, 'featureName', by='Banana')

		self.data.VariableType = 'Not an enum'
		self.assertRaises(TypeError, self.data.getFeatures, 'featureName', by='Feature Name')


class test_dataset_loadsop(unittest.TestCase):
	"""
	Test the loading of SOP json
	"""

	def setUp(self):
		# Load empty object and populate with sythetic data.

		# Empty object
		self.data = nPYc.Dataset()


	def test_loadparameters(self):

		with self.subTest(msg='Checking null return for \'Generic\' SOP.'):
			attributes = {'Log': self.data.Attributes['Log'],
						'dpi': 300,
						'figureFormat': 'png',
						'figureSize': [11, 7],
						'histBins': 100,
						'noFiles': 10,
						'quantiles': [25, 75],
						'sampleMetadataNotExported': ["Exclusion Details"],
						'featureMetadataNotExported': [],
						"analyticalMeasurements":{},
						"excludeFromPlotting":[],
						"sampleTypeColours": {"StudySample": "b", "StudyPool": "g", "ExternalReference": "r", "MethodReference": "m", "ProceduralBlank": "c", "Other": "grey"}
						}

			self.assertEqual(self.data.Attributes, attributes)


	def test_overrideparameters(self):

		data = nPYc.Dataset(figureFormat='svg', squids=True)

		attributes = {'Log': data.Attributes['Log'],
					'dpi': 300,
					'figureFormat': 'svg',
					'figureSize': [11, 7],
					'histBins': 100,
					'noFiles': 10,
					'quantiles': [25, 75],
					'squids': True,
					'sampleMetadataNotExported': ["Exclusion Details"],
					'featureMetadataNotExported': [],
					"analyticalMeasurements":{},
					"excludeFromPlotting":[],
					"sampleTypeColours": {"StudySample": "b", "StudyPool": "g", "ExternalReference": "r", "MethodReference": "m", "ProceduralBlank": "c", "Other": "grey"}
					}

		self.assertEqual(data.Attributes, attributes)


	def test_loadParameters_invalidsoppath(self):

		with tempfile.TemporaryDirectory() as tmpdirname:

			fakeSOPpath = os.path.join(tmpdirname, 'foldernotthere')
			self.assertRaises(ValueError, self.data._loadParameters, 'fakeSOP', fakeSOPpath)


	def test_loadParameters_invalidsop(self):

		with tempfile.TemporaryDirectory() as tmpdirname:

			self.assertRaises(ValueError, self.data._loadParameters, 'fakeSOP', None)


	def test_loadParameters_customsoppath(self):

		testSOPcontents = {'testsopkey':'testsopvalue', 'testsopkey2': 2}

		with tempfile.TemporaryDirectory() as tmpdirname:
			# Create temp SOP file
			with open(os.path.join(tmpdirname, 'testSOP.json'), 'w') as outfile:
				json.dump(testSOPcontents, outfile)

			self.data._loadParameters('testSOP', tmpdirname)

			testSOPcontents = {**self.data.Attributes, **testSOPcontents}

			self.assertEqual(self.data.Attributes, testSOPcontents)


class test_dataset_addsampleinfo(unittest.TestCase):
	"""
	Test the loading of study designs
	"""

	def setUp(self):
		self.Data = nPYc.MSDataset(os.path.join('..','..','npc-standard-project','Derived_Data','UnitTest1_PCSOP.069_QI.csv'), fileType='QI')
		self.Data.addSampleInfo(descriptionFormat='Filenames')


	def test_dataset_load_npc_lims(self):
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

		data = copy.deepcopy(self.Data)
		data.addSampleInfo(descriptionFormat='NPC LIMS', filePath=os.path.join('..','..','npc-standard-project','Derived_Worklists','UnitTest1_MS_serum_PCSOP.069.csv'))
		pandas.util.testing.assert_series_equal(data.sampleMetadata['Sampling ID'], samplingIDs)


	def test_dataset_load_npc_subjectinfo_columns(self):

		columns = ['Person responsible', 'Sampling Protocol', 'Creatinine (mM)', 'Glucose (mM)', 'Class', 'Date of Birth', 'Gender', 'Further Subject info?', 'Environmental measures', 'SubjectInfoData']

		data = copy.deepcopy(self.Data)
		data.addSampleInfo(descriptionFormat='NPC LIMS', filePath=os.path.join('..','..','npc-standard-project','Derived_Worklists','UnitTest1_MS_serum_PCSOP.069.csv'))
		data.addSampleInfo(descriptionFormat='NPC Subject Info', filePath=os.path.join('..','..','npc-standard-project','Project_Description','UnitTest1_metadata_PCDOC.014.xlsx'))

		for column in columns:
			self.subTest(msg='Checking ' + column)
			self.assertIn(column, data.sampleMetadata.keys())


	def test_dataset_load_csv(self):

		from nPYc.enumerations import AssayRole, SampleType

		data = nPYc.MSDataset(os.path.join('..','..','npc-standard-project','Derived_Data','UnitTest1_PCSOP.069_QI.csv'), fileType='QI')

		data.addSampleInfo(descriptionFormat='Basic CSV', filePath=os.path.join('..', '..','npc-standard-project','Derived_Worklists', 'UnitTest1_metadata_basic_csv.csv'))

		expectedSampleMask = numpy.array([True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True,
										  True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True,
										  True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True,
										  True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True,
										  True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True,
										  True, True, True, True, True, True, True, False], dtype=bool)

		expectedSampleMetadata = pandas.DataFrame(0, index=numpy.arange(115), columns=['Sample File Name', 'Sample Base Name', 'Batch', 'Correction Batch', 'Acquired Time', 'Run Order',
																					   'Exclusion Details', 'Metadata Available', 'Sampling ID', 'AssayRole', 'SampleType', 'Dilution'])

		expectedSampleMetadata['Sample File Name'] = ['UnitTest1_LPOS_ToF02_B1SRD01', 'UnitTest1_LPOS_ToF02_B1SRD02', 'UnitTest1_LPOS_ToF02_B1SRD03', 'UnitTest1_LPOS_ToF02_B1SRD04',
												  'UnitTest1_LPOS_ToF02_B1SRD05', 'UnitTest1_LPOS_ToF02_B1SRD06', 'UnitTest1_LPOS_ToF02_B1SRD07', 'UnitTest1_LPOS_ToF02_B1SRD08',
												  'UnitTest1_LPOS_ToF02_B1SRD09', 'UnitTest1_LPOS_ToF02_B1SRD10', 'UnitTest1_LPOS_ToF02_B1SRD11', 'UnitTest1_LPOS_ToF02_B1SRD12',
												  'UnitTest1_LPOS_ToF02_B1SRD13', 'UnitTest1_LPOS_ToF02_B1SRD14', 'UnitTest1_LPOS_ToF02_B1SRD15', 'UnitTest1_LPOS_ToF02_B1SRD16',
												  'UnitTest1_LPOS_ToF02_B1SRD17', 'UnitTest1_LPOS_ToF02_B1SRD18', 'UnitTest1_LPOS_ToF02_B1SRD19', 'UnitTest1_LPOS_ToF02_B1SRD20',
												  'UnitTest1_LPOS_ToF02_B1SRD21', 'UnitTest1_LPOS_ToF02_B1SRD22', 'UnitTest1_LPOS_ToF02_B1SRD23', 'UnitTest1_LPOS_ToF02_B1SRD24',
												  'UnitTest1_LPOS_ToF02_B1SRD25', 'UnitTest1_LPOS_ToF02_B1SRD26', 'UnitTest1_LPOS_ToF02_B1SRD27', 'UnitTest1_LPOS_ToF02_B1SRD28',
												  'UnitTest1_LPOS_ToF02_B1SRD29', 'UnitTest1_LPOS_ToF02_B1SRD30', 'UnitTest1_LPOS_ToF02_B1SRD31', 'UnitTest1_LPOS_ToF02_B1SRD32',
												  'UnitTest1_LPOS_ToF02_B1SRD33', 'UnitTest1_LPOS_ToF02_B1SRD34', 'UnitTest1_LPOS_ToF02_B1SRD35', 'UnitTest1_LPOS_ToF02_B1SRD36',
												  'UnitTest1_LPOS_ToF02_B1SRD37', 'UnitTest1_LPOS_ToF02_B1SRD38', 'UnitTest1_LPOS_ToF02_B1SRD39', 'UnitTest1_LPOS_ToF02_B1SRD40',
												  'UnitTest1_LPOS_ToF02_B1SRD41', 'UnitTest1_LPOS_ToF02_B1SRD42', 'UnitTest1_LPOS_ToF02_B1SRD43', 'UnitTest1_LPOS_ToF02_B1SRD44',
												  'UnitTest1_LPOS_ToF02_B1SRD45', 'UnitTest1_LPOS_ToF02_B1SRD46', 'UnitTest1_LPOS_ToF02_B1SRD47', 'UnitTest1_LPOS_ToF02_B1SRD48',
												  'UnitTest1_LPOS_ToF02_B1SRD49', 'UnitTest1_LPOS_ToF02_B1SRD50', 'UnitTest1_LPOS_ToF02_B1SRD51', 'UnitTest1_LPOS_ToF02_B1SRD52',
												  'UnitTest1_LPOS_ToF02_B1SRD53', 'UnitTest1_LPOS_ToF02_B1SRD54', 'UnitTest1_LPOS_ToF02_B1SRD55', 'UnitTest1_LPOS_ToF02_B1SRD56',
												  'UnitTest1_LPOS_ToF02_B1SRD57', 'UnitTest1_LPOS_ToF02_B1SRD58', 'UnitTest1_LPOS_ToF02_B1SRD59', 'UnitTest1_LPOS_ToF02_B1SRD60',
												  'UnitTest1_LPOS_ToF02_B1SRD61', 'UnitTest1_LPOS_ToF02_B1SRD62', 'UnitTest1_LPOS_ToF02_B1SRD63', 'UnitTest1_LPOS_ToF02_B1SRD64',
												  'UnitTest1_LPOS_ToF02_B1SRD65', 'UnitTest1_LPOS_ToF02_B1SRD66', 'UnitTest1_LPOS_ToF02_B1SRD67', 'UnitTest1_LPOS_ToF02_B1SRD68',
												  'UnitTest1_LPOS_ToF02_B1SRD69', 'UnitTest1_LPOS_ToF02_B1SRD70', 'UnitTest1_LPOS_ToF02_B1SRD71', 'UnitTest1_LPOS_ToF02_B1SRD72',
												  'UnitTest1_LPOS_ToF02_B1SRD73', 'UnitTest1_LPOS_ToF02_B1SRD74', 'UnitTest1_LPOS_ToF02_B1SRD75', 'UnitTest1_LPOS_ToF02_B1SRD76',
												  'UnitTest1_LPOS_ToF02_B1SRD77', 'UnitTest1_LPOS_ToF02_B1SRD78', 'UnitTest1_LPOS_ToF02_B1SRD79', 'UnitTest1_LPOS_ToF02_B1SRD80',
												  'UnitTest1_LPOS_ToF02_B1SRD81', 'UnitTest1_LPOS_ToF02_B1SRD82', 'UnitTest1_LPOS_ToF02_B1SRD83', 'UnitTest1_LPOS_ToF02_B1SRD84',
												  'UnitTest1_LPOS_ToF02_B1SRD85', 'UnitTest1_LPOS_ToF02_B1SRD86', 'UnitTest1_LPOS_ToF02_B1SRD87', 'UnitTest1_LPOS_ToF02_B1SRD88',
												  'UnitTest1_LPOS_ToF02_B1SRD89', 'UnitTest1_LPOS_ToF02_B1SRD90', 'UnitTest1_LPOS_ToF02_B1SRD91', 'UnitTest1_LPOS_ToF02_B1SRD92',
												  'UnitTest1_LPOS_ToF02_Blank01', 'UnitTest1_LPOS_ToF02_Blank02', 'UnitTest1_LPOS_ToF02_B1E1_SR', 'UnitTest1_LPOS_ToF02_B1E2_SR',
												  'UnitTest1_LPOS_ToF02_B1E3_SR', 'UnitTest1_LPOS_ToF02_B1E4_SR', 'UnitTest1_LPOS_ToF02_B1E5_SR', 'UnitTest1_LPOS_ToF02_B1S1_SR',
												  'UnitTest1_LPOS_ToF02_B1S2_SR', 'UnitTest1_LPOS_ToF02_B1S3_SR', 'UnitTest1_LPOS_ToF02_B1S4_SR', 'UnitTest1_LPOS_ToF02_B1S5_SR',
												  'UnitTest1_LPOS_ToF02_S1W01', 'UnitTest1_LPOS_ToF02_S1W02', 'UnitTest1_LPOS_ToF02_S1W03', 'UnitTest1_LPOS_ToF02_S1W04',
												  'UnitTest1_LPOS_ToF02_S1W05', 'UnitTest1_LPOS_ToF02_S1W06', 'UnitTest1_LPOS_ToF02_S1W07', 'UnitTest1_LPOS_ToF02_S1W08_x',
												  'UnitTest1_LPOS_ToF02_S1W11_LTR', 'UnitTest1_LPOS_ToF02_S1W12_SR', 'UnitTest1_LPOS_ToF02_ERROR']

		expectedSampleMetadata['Sample Base Name'] = expectedSampleMetadata['Sample File Name']
		expectedSampleMetadata['Metadata Available'] = True
		expectedSampleMetadata['Batch'] = numpy.nan

		expectedSampleMetadata['Correction Batch'] = [numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan,
													  numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan,
													  numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan,
													  numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan,
													  numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan,
													  numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan,
													  numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan,
													  numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan,
													  numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan,
													  numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan,
													  numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan,
													  numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan,
													  numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan,
													  numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan,
													  numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan,
													  numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan,
													  numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan,
													  numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan,
													  numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan,
													  numpy.nan]

		expectedSampleMetadata['Acquired Time'] = [numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan,
												   numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan,
												   numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan,
												   numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan,
												   numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan,
												   numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan,
												   numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan,
												   numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan,
												   numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan,
												   numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan,
												   numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan,
												   numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan,
												   numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan,
												   numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan,
												   numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan,
												   numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan,
												   numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan,
												   numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan,
												   numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan,
												   numpy.nan]

		expectedSampleMetadata['Run Order'] = [numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan,
											   numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan,
											   numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan,
											   numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan,
											   numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan,
											   numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan,
											   numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan,
											   numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan,
											   numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan,
											   numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan,
											   numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan,
											   numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan,
											   numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan,
											   numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan,
											   numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan,
											   numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan,
											   numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan,
											   numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan,
											   numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan,
											   numpy.nan]

		expectedSampleMetadata['Exclusion Details'] = [numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan,
													   numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan,
													   numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan,
													   numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan,
													   numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan,
													   numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan,
													   numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan,
													   numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan,
													   numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan,
													   numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan,
													   numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan,
													   numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan,
													   numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan,
													   numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan,
													   numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan,
													   numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan,
													   numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan,
													   numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan,
													   numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan,
													   numpy.nan]

		expectedSampleMetadata['Sampling ID'] = [numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan,
												 numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan,
												 numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan,
												 numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan,
												 numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan,
												 numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan,
												 numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan,
												 numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan,
												 numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan,
												 numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan,
												 numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan,
												 numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan,
												 numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan,
												 numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan,
												 numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan,
												 numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan,
												 numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan,
												 numpy.nan, numpy.nan, 'UT1_S1_s1', 'UT1_S2_s1', 'UT1_S3_s1',
												 'UT1_S4_s1', 'UT1_S4_s2', 'UT1_S4_s3', 'UT1_S4_s4', 'UT1_S4_s5', 'LTR',
												 'SR', numpy.nan]

		expectedSampleMetadata['AssayRole'] = [AssayRole.LinearityReference, AssayRole.LinearityReference, AssayRole.LinearityReference, AssayRole.LinearityReference,
												AssayRole.LinearityReference, AssayRole.LinearityReference, AssayRole.LinearityReference, AssayRole.LinearityReference,
												AssayRole.LinearityReference, AssayRole.LinearityReference, AssayRole.LinearityReference, AssayRole.LinearityReference,
												AssayRole.LinearityReference, AssayRole.LinearityReference, AssayRole.LinearityReference, AssayRole.LinearityReference,
												AssayRole.LinearityReference, AssayRole.LinearityReference, AssayRole.LinearityReference, AssayRole.LinearityReference,
												AssayRole.LinearityReference, AssayRole.LinearityReference, AssayRole.LinearityReference, AssayRole.LinearityReference,
												AssayRole.LinearityReference, AssayRole.LinearityReference, AssayRole.LinearityReference, AssayRole.LinearityReference,
												AssayRole.LinearityReference, AssayRole.LinearityReference, AssayRole.LinearityReference, AssayRole.LinearityReference,
												AssayRole.LinearityReference, AssayRole.LinearityReference, AssayRole.LinearityReference, AssayRole.LinearityReference,
												AssayRole.LinearityReference, AssayRole.LinearityReference, AssayRole.LinearityReference, AssayRole.LinearityReference,
												AssayRole.LinearityReference, AssayRole.LinearityReference, AssayRole.LinearityReference, AssayRole.LinearityReference,
												AssayRole.LinearityReference, AssayRole.LinearityReference, AssayRole.LinearityReference, AssayRole.LinearityReference,
												AssayRole.LinearityReference, AssayRole.LinearityReference, AssayRole.LinearityReference, AssayRole.LinearityReference,
												AssayRole.LinearityReference, AssayRole.LinearityReference, AssayRole.LinearityReference, AssayRole.LinearityReference,
												AssayRole.LinearityReference, AssayRole.LinearityReference, AssayRole.LinearityReference, AssayRole.LinearityReference,
												AssayRole.LinearityReference, AssayRole.LinearityReference, AssayRole.LinearityReference, AssayRole.LinearityReference,
												AssayRole.LinearityReference, AssayRole.LinearityReference, AssayRole.LinearityReference, AssayRole.LinearityReference,
												AssayRole.LinearityReference, AssayRole.LinearityReference, AssayRole.LinearityReference, AssayRole.LinearityReference,
												AssayRole.LinearityReference, AssayRole.LinearityReference, AssayRole.LinearityReference, AssayRole.LinearityReference,
												AssayRole.LinearityReference, AssayRole.LinearityReference, AssayRole.LinearityReference, AssayRole.LinearityReference,
												AssayRole.LinearityReference, AssayRole.LinearityReference, AssayRole.LinearityReference, AssayRole.LinearityReference,
												AssayRole.LinearityReference, AssayRole.LinearityReference, AssayRole.LinearityReference, AssayRole.LinearityReference,
												AssayRole.LinearityReference, AssayRole.LinearityReference, AssayRole.LinearityReference, AssayRole.LinearityReference,
												AssayRole.Assay, AssayRole.Assay, AssayRole.PrecisionReference, AssayRole.PrecisionReference, AssayRole.PrecisionReference,
												AssayRole.PrecisionReference, AssayRole.PrecisionReference, AssayRole.PrecisionReference, AssayRole.PrecisionReference,
												AssayRole.PrecisionReference, AssayRole.PrecisionReference, AssayRole.PrecisionReference, AssayRole.Assay, AssayRole.Assay,
												AssayRole.Assay, AssayRole.Assay, AssayRole.Assay, AssayRole.Assay, AssayRole.Assay, AssayRole.Assay,
												AssayRole.PrecisionReference, AssayRole.PrecisionReference, numpy.nan]

		expectedSampleMetadata['SampleType'] = [SampleType.StudyPool, SampleType.StudyPool, SampleType.StudyPool, SampleType.StudyPool, SampleType.StudyPool, SampleType.StudyPool,
												SampleType.StudyPool, SampleType.StudyPool, SampleType.StudyPool, SampleType.StudyPool, SampleType.StudyPool, SampleType.StudyPool,
												SampleType.StudyPool, SampleType.StudyPool, SampleType.StudyPool, SampleType.StudyPool, SampleType.StudyPool, SampleType.StudyPool,
												SampleType.StudyPool, SampleType.StudyPool, SampleType.StudyPool, SampleType.StudyPool, SampleType.StudyPool, SampleType.StudyPool,
												SampleType.StudyPool, SampleType.StudyPool, SampleType.StudyPool, SampleType.StudyPool, SampleType.StudyPool, SampleType.StudyPool,
												SampleType.StudyPool, SampleType.StudyPool, SampleType.StudyPool, SampleType.StudyPool, SampleType.StudyPool, SampleType.StudyPool,
												SampleType.StudyPool, SampleType.StudyPool, SampleType.StudyPool, SampleType.StudyPool, SampleType.StudyPool, SampleType.StudyPool,
												SampleType.StudyPool, SampleType.StudyPool, SampleType.StudyPool, SampleType.StudyPool, SampleType.StudyPool, SampleType.StudyPool,
												SampleType.StudyPool, SampleType.StudyPool, SampleType.StudyPool, SampleType.StudyPool, SampleType.StudyPool, SampleType.StudyPool,
												SampleType.StudyPool, SampleType.StudyPool, SampleType.StudyPool, SampleType.StudyPool, SampleType.StudyPool, SampleType.StudyPool,
												SampleType.StudyPool, SampleType.StudyPool, SampleType.StudyPool, SampleType.StudyPool, SampleType.StudyPool, SampleType.StudyPool,
												SampleType.StudyPool, SampleType.StudyPool, SampleType.StudyPool, SampleType.StudyPool, SampleType.StudyPool, SampleType.StudyPool,
												SampleType.StudyPool, SampleType.StudyPool, SampleType.StudyPool, SampleType.StudyPool, SampleType.StudyPool, SampleType.StudyPool,
												SampleType.StudyPool, SampleType.StudyPool, SampleType.StudyPool, SampleType.StudyPool, SampleType.StudyPool, SampleType.StudyPool,
												SampleType.StudyPool, SampleType.StudyPool, SampleType.StudyPool, SampleType.StudyPool, SampleType.StudyPool, SampleType.StudyPool,
												SampleType.StudyPool, SampleType.StudyPool, SampleType.ProceduralBlank, SampleType.ProceduralBlank, SampleType.StudyPool,
												SampleType.StudyPool, SampleType.StudyPool, SampleType.StudyPool, SampleType.StudyPool, SampleType.StudyPool, SampleType.StudyPool,
												SampleType.StudyPool, SampleType.StudyPool, SampleType.StudyPool, SampleType.StudySample, SampleType.StudySample, SampleType.StudySample,
												SampleType.StudySample, SampleType.StudySample, SampleType.StudySample, SampleType.StudySample, SampleType.StudySample,
												SampleType.ExternalReference, SampleType.StudyPool, numpy.nan]

		expectedSampleMetadata['Dilution'] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 20,
											  20, 20, 20, 20, 40, 40, 40, 60, 60, 60, 80, 80, 80, 80, 80,
											  100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 1, 1, 1, 1, 1, 1, 1, 1,
											  1, 1, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 20, 20, 20, 20,
											  20, 40, 40, 40, 60, 60, 60, 80, 80, 80, 80, 80, 100, 100, 100, 100, 100,
											  100, 100, 100, 100, 100, 0, 0, 100, 100, 100, 100, 100, 100,
											  100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]

		pandas.util.testing.assert_frame_equal(expectedSampleMetadata, data.sampleMetadata, check_dtype=False)
		numpy.testing.assert_array_equal(expectedSampleMask, data.sampleMask)


	def test_dataset_load_csv_raises(self):

		with tempfile.TemporaryDirectory() as tmpdirname:
			# Generate a CSV with no 'Sample File Name' column
			testDF = pandas.DataFrame([[1,2,3],[1,2,3]], columns={'a', 'b', 'c'})
			testDF.to_csv(os.path.join(tmpdirname, 'tmp.csv'))

			data = nPYc.Dataset()
			self.assertRaises(KeyError, data.addSampleInfo, descriptionFormat='Basic CSV', filePath=os.path.join(tmpdirname, 'tmp.csv'))


	def test_dataset_load_isatab(self):

		columns = ['Sample File Name', 'Sample Base Name', 'Sample Base Name Normalised',
			  'Sampling ID', 'Assay data name', 'Dilution', 'Run Order',
			  'Acquired Time', 'Instrument', 'Chromatography', 'Ionisation', 'Batch',
			  'Plate', 'Well', 'Correction Batch', 'Detector', 'Subject ID', 'Age',
			  'Gender', 'Status', 'Sample Name', 'Assay data name Normalised',
			  'Exclusion Details', 'Study Sample', 'Long-Term Reference',
			  'Study Reference', 'Method Reference', 'Dilution Series',
			  'LIMS Marked Missing', 'Data Present', 'LIMS Present', 'Skipped',
			  'AssayRole', 'SampleType', 'SubjectInfoData']

		data = copy.deepcopy(self.Data)

		with warnings.catch_warnings():
			warnings.simplefilter('ignore', UserWarning)
			data.addSampleInfo(descriptionFormat='ISATAB',
									  filePath=os.path.join('..', '..', 'npc-standard-project', 'Project_Description', 'ISATAB-Unit-Test'),
									  studyID=1,
									  assay='MS',
									  assayID=1)

		for column in columns:
			self.subTest(msg='Checking ' + column)
			self.assertIn(column, data.sampleMetadata.columns)


	def test_dataset_parsefilename(self):

		data = nPYc.Dataset()
		self.assertRaises(NotImplementedError, data.addSampleInfo, descriptionFormat='Filenames', filenameSpec='')


	def test_dataset_raises(self):

		data = nPYc.Dataset()
		self.assertRaises(NotImplementedError, data.addSampleInfo, descriptionFormat='Not an understood format', filenameSpec='')


class test_dataset_addfeatureinfo(unittest.TestCase):

	def test_dataset_add_reference_ranges(self):
		"""
		Assume the addReferenceRanges function is well tested - just check the expected columns appear
		"""
		data = nPYc.Dataset()
		referencePath = os.path.join('..', 'nPYc', 'StudyDesigns', 'BI-LISA_reference_ranges.json')

		data.featureMetadata = pandas.DataFrame(['TPTG', 'TPCH', 'TPFC', 'TPA1', 'TPA2', 'TPAB', 'VLTG', 'VLCH', 'VLFC', 'VLPL', 'VLAB'],
												columns=['Feature Name'])

		data.addFeatureInfo(descriptionFormat='Reference Ranges', filePath=referencePath)

		columns =  ['Unit', 'Upper Reference Bound', 'Upper Reference Value', 'Lower Reference Bound', 'Lower Reference Value']
		for column in columns:
			self.subTest(msg='Checking ' + column)
			self.assertIn(column, data.featureMetadata.columns)

	def test_dataset_add_feature_info(self):

		data = nPYc.Dataset()
		referencePath = os.path.join('..', 'nPYc', 'StudyDesigns', 'featureMetadataInfo.csv')

		data.featureMetadata = pandas.DataFrame(
			['TPTG', 'TPCH', 'TPFC', 'TPA1', 'TPA2', 'TPAB', 'VLTG', 'VLCH', 'VLFC', 'VLPL', 'VLAB'],
			columns=['Feature Name'])

		data.addFeatureInfo(descriptionFormat='Reference Ranges', filePath=referencePath)

		data = nPYc.Dataset()
		self.assertRaises(NotImplementedError, data.addFeatureInfo, descriptionFormat='Not an understood format',
						  filenameSpec='')


if __name__ == '__main__':
	unittest.main()
