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
