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
import datetime

sys.path.append("..")
import nPYc
from generateTestDataset import generateTestDataset

class test_multivariate_multivariateutilities(unittest.TestCase):

	def test_multivariateutilities_exploratoryAnalysisPCA_raises(self):

		self.assertRaises(TypeError, nPYc.multivariate.exploratoryAnalysisPCA, 'npyc_dataset argument must be one of the nPYc dataset objects')

		dataset = generateTestDataset(10, 20)

		self.assertRaises(TypeError, nPYc.multivariate.exploratoryAnalysisPCA, dataset, scaling='invalid scaling')
		self.assertRaises(TypeError, nPYc.multivariate.exploratoryAnalysisPCA, dataset, scaling=10)

		self.assertRaises(TypeError, nPYc.multivariate.exploratoryAnalysisPCA, dataset, maxComponents='Not an integer')
		self.assertRaises(TypeError, nPYc.multivariate.exploratoryAnalysisPCA, dataset, maxComponents=0)

		self.assertRaises(TypeError, nPYc.multivariate.exploratoryAnalysisPCA, dataset,  minQ2='Not a number')


	def test_multivariateutilities_exploratoryAnalysisPCA(self):

		dataset = generateTestDataset(50, 300)
		numpy.random.seed(seed=200)
		dataset.intensityData = numpy.random.lognormal(size=(50, 300))

		pcaModel = nPYc.multivariate.exploratoryAnalysisPCA(dataset)

		with self.subTest(msg='Number of components'):
			self.assertEqual(pcaModel.ncomps, 2)

		numpy.random.seed()


	def test_metadataTypeGrouping(self):

		with self.subTest(msg='Catagorical Data'):

			testData = pandas.Series([1,1,1,1,1,1,1,1,12,2,2,2,2,2,2,2])

			result = nPYc.multivariate.metadataTypeGrouping(testData)
			self.assertEqual(result, 'categorical')

		with self.subTest(msg='Continuous Data'):

			testData = pandas.Series(numpy.linspace(0,10))

			result = nPYc.multivariate.metadataTypeGrouping(testData)
			self.assertEqual(result, 'continuous')

		with self.subTest(msg='Uniform Data'):

			testData = pandas.Series([1,1,1,1,1,1])

			result = nPYc.multivariate.metadataTypeGrouping(testData)
			self.assertEqual(result, 'uniform')

		with self.subTest(msg='Dates'):

			testData = pandas.Series([datetime.datetime.now(), datetime.datetime.now() + datetime.timedelta(0,3)])

			result = nPYc.multivariate.metadataTypeGrouping(testData)
			self.assertEqual(result, 'date')

		with self.subTest(msg='Unique'):

			testData = pandas.Series(['1','2','3','4','5','6','7'])

			result = nPYc.multivariate.metadataTypeGrouping(testData)
			self.assertEqual(result, 'unique')

		with self.subTest(msg='Uniform By Sample Groups'):

			testData = pandas.Series(['1','1','2','2','3','3','4'])

			result = nPYc.multivariate.metadataTypeGrouping(testData, sampleGroups=testData)
			self.assertEqual(result, 'uniformBySampleType')

		with self.subTest(msg='catVsContRatio'):

			testData = pandas.Series([1,1,1,2,3,4])

			result = nPYc.multivariate.metadataTypeGrouping(testData, catVsContRatio=0.5)
			self.assertEqual(result, 'continuous')

			result = nPYc.multivariate.metadataTypeGrouping(testData, catVsContRatio=0.9)
			self.assertEqual(result, 'categorical')

		with self.subTest(msg='Sample Groups'):

			testData = pandas.Series([1,1,1,1,1,1,1,1,12,2,2,2,2,2,2,2])
			testGroups = pandas.Series([1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2])

			result = nPYc.multivariate.metadataTypeGrouping(testData, sampleGroups=testGroups)
			self.assertEqual(result, 'categorical')


