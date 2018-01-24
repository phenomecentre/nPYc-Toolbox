"""
Test mass spectrum tools sub module.
"""

import scipy
import pandas
import numpy
import sys
import unittest
import os
import tempfile
import inspect
import copy

sys.path.append("..")
import nPYc

from nPYc.utilities._massSpectrumBuilder import massSpectrumBuilder

class test_utilities_massSpectrumBuilder_synthetic(unittest.TestCase):

	def setUp(self):
		##
		# Five features:
		# Three correlated, with two in close RT proximity
		# Two uncorrelated
		##
		self.msData = nPYc.MSDataset('', fileType='empty')
		self.msData.sampleMetadata = pandas.DataFrame(['a','b','c','d'], columns=['Sample Name'])
		self.msData.intensityData = numpy.array([[1,21,10.5,4,5],
												 [2,22,12.5,5,6],
												 [3,23,11.5,6,5],
												 [4,24,12,7,6]],
												 dtype=float)
		self.msData.featureMetadata = pandas.DataFrame([[3.12, 5, '100 - 10 - 1', '3.12_127.1212m/z', numpy.nan, 127.1212],
													   [3.13, 5, '100 - 10 - 1', '3.13_220.1419n', 'M+H, M+Na, M+K, 2M+Na', 219.1419],
													   [3.12, 5, '100 - 20', '3.12_170.2233m/z', numpy.nan, 170.2233], 
													   [5.32, 5, '100', '5.32_89.9812m/z', numpy.nan, 89.9812],
													   [0.56, 5, '90 - 100 - 50', '0.56_214.1245n', 'M+H, M+Na, M+K, 2M+Na', 213.1245]],
													   columns=['Retention Time','Peak Width','Isotope Distribution','Feature Name','Adducts','m/z'])
		self.msData.Attributes['FeatureExtractionSoftware'] = 'Progenesis QI'
		self.msData.initialiseMasks()


	def test_utilities_massSpectrumBuilder(self):
		"""
		Test defaults
		"""

		filteredData = massSpectrumBuilder(self.msData, correlationThreshold=0.95, rtWindow=20, simulatedSpecra=True)

		expectedFeatureMetadata = pandas.DataFrame([[3.13, 5, '100 - 10 - 1', '3.13_220.1419n', 'M+H, M+Na, M+K, 2M+Na', 219.1419, '3.12_127.1212m/z', [(127.1212, 100.0), (128.124555, 10.0), (129.12791, 1.0), (221.14917599999998, 100.0), (222.15253099999998, 10.0), (223.15588599999998, 1.0), (243.131118, 100.0), (244.13447299999999, 10.0), (245.13782799999998, 1.0), (259.105058, 100.0), (260.108413, 10.0), (261.111768, 1.0), (463.273018, 100.0), (464.276373, 10.0), (465.279728, 1.0)]],
													[3.12, 5, '100 - 20', '3.12_170.2233m/z', numpy.nan, 170.2233, '', [(170.2233, 100.0), (171.226655, 20.0)]], 
													[5.32, 5, '100', '5.32_89.9812m/z', numpy.nan, 89.9812, '', [(89.9812, 100)]],
													[0.56, 5, '90 - 100 - 50', '0.56_214.1245n', 'M+H, M+Na, M+K, 2M+Na', 213.1245, '', [(215.131776, 90.0), (216.135131, 100.0), (217.138486, 50.0), (237.113718, 90.0), (238.117073, 100.0), (239.120428, 50.0), (253.087658, 90.0), (254.091013, 100.0), (255.094368, 50.0), (451.238218, 90.0), (452.241573, 100.0), (453.244928, 50.0)]]],
													columns=['Retention Time','Peak Width','Isotope Distribution','Feature Name','Adducts','m/z', 'Correlated Features', 'Mass Spectrum'])

		pandas.util.testing.assert_frame_equal(filteredData.featureMetadata, expectedFeatureMetadata)


	def test_utilities_massSpectrumBuilder_rtwindow(self):
		"""
		Test the RT window filter
		"""

		with self.subTest(msg='All-inclusive window'):
			filteredData = massSpectrumBuilder(self.msData, correlationThreshold=0.95, rtWindow=1000, simulatedSpecra=False)

			expectedFeatureMetadata = pandas.DataFrame([[3.13, 5, '100 - 10 - 1', '3.13_220.1419n', 'M+H, M+Na, M+K, 2M+Na', 219.1419, '3.12_127.1212m/z; 5.32_89.9812m/z'],
													[3.12, 5, '100 - 20', '3.12_170.2233m/z', numpy.nan, 170.2233, ''], 
													[0.56, 5, '90 - 100 - 50', '0.56_214.1245n', 'M+H, M+Na, M+K, 2M+Na', 213.1245, '']],
													columns=['Retention Time','Peak Width','Isotope Distribution','Feature Name','Adducts','m/z', 'Correlated Features'])
			pandas.util.testing.assert_frame_equal(filteredData.featureMetadata, expectedFeatureMetadata)

		with self.subTest(msg='Restrictive window'):
			filteredData = massSpectrumBuilder(self.msData, correlationThreshold=0.95, rtWindow=0.1, simulatedSpecra=False)

			expectedFeatureMetadata = copy.copy(self.msData.featureMetadata)
			expectedFeatureMetadata['Correlated Features'] = ''
			pandas.util.testing.assert_frame_equal(filteredData.featureMetadata, expectedFeatureMetadata)


	def test_utilities_massSpectrumBuilder_correlationThreshold(self):
		"""
		Test corr cutoff
		"""
		filteredData = massSpectrumBuilder(self.msData, correlationThreshold=0.5, rtWindow=20, simulatedSpecra=False)

		expectedFeatureMetadata = pandas.DataFrame([[3.13, 5, '100 - 10 - 1', '3.13_220.1419n', 'M+H, M+Na, M+K, 2M+Na', 219.1419, '3.12_127.1212m/z; 3.12_170.2233m/z'],
													[5.32, 5, '100', '5.32_89.9812m/z', numpy.nan, 89.9812, ''], 
													[0.56, 5, '90 - 100 - 50', '0.56_214.1245n', 'M+H, M+Na, M+K, 2M+Na', 213.1245, '']],
													columns=['Retention Time','Peak Width','Isotope Distribution','Feature Name','Adducts','m/z', 'Correlated Features'])
		pandas.util.testing.assert_frame_equal(filteredData.featureMetadata, expectedFeatureMetadata)


	def test_utilities_massSpectrumBuilder_log(self):
		"""
		Test log attribute
		"""
		filteredData = massSpectrumBuilder(self.msData, correlationThreshold=0.8, rtWindow=50, simulatedSpecra=False)

		self.assertEqual(filteredData.Attributes['Log'][-1][1], "Redundant features removed with rtWindow of: %f seconds and correlationThreshold of: %f." % (50, 0.8))


from nPYc.utilities._buildSpectrumFromQIfeature import buildMassSpectrumFromQIfeature
from nPYc.utilities._buildSpectrumFromQIfeature import _buildSpectrumFromQIisotopes

class test_utilities_buildSpectrumFromQIfeature(unittest.TestCase):

	def setUp(self):

		self.featureMetdata = pandas.DataFrame([
												(3.656683333,0.165616667,'100 - 14.3 - 8.98 - 0.0461','3.66_378.1288n',"M+H, M+Na, M+K",379.1364308),
												(3.423683333,0.053333333,'100 - 22','3.42_165.0697m/z',numpy.nan,165.0696637)
												],
												columns=['Retention Time','Peak Width','Isotope Distribution','Feature Name','Adducts','m/z'])



	def test_buildSpectrumFromQIfeature_buildSpectrumFromQIfeature(self):
		with self.subTest(msg='With adducts'):
			obtained = buildMassSpectrumFromQIfeature(self.featureMetdata.iloc[0].to_dict())

			expected = [(379.136076, 100.0),
						(380.139431, 14.3),
						(381.142786, 8.98),
						(382.146141, 0.0461),
						(401.118018, 100.0),
						(402.121373, 14.3),
						(403.124728, 8.98),
						(404.128083, 0.0461),
						(417.09195800000003, 100.0),
						(418.09531300000003, 14.3),
						(419.09866800000003, 8.98),
						(420.10202300000003, 0.0461)]

			self.assertEqual(expected, obtained)

		with self.subTest(msg='No adducts'):
			obtained = buildMassSpectrumFromQIfeature(self.featureMetdata.iloc[1].to_dict())

			expected = [(165.06966370000001, 100.0), (166.07301870000001, 22.0)]

			self.assertEqual(expected, obtained)


	def test_buildSpectrumFromQIfeature_buildSpectrumFromQIfeature_unknownAdduct(self):
		"""
		Check we raise an error for unknown adducts
		"""
		self.assertRaises(KeyError, buildMassSpectrumFromQIfeature, self.featureMetdata.iloc[0].to_dict(), adducts={'M+U':(235.0439299, 1)})


	def test_buildSpectrumFromQIfeature_buildSpectrumFromQIisotopes(self):
		"""
		Test generation of isotope spectra.
		"""

		with self.subTest(msg='Default delta'):
			result = _buildSpectrumFromQIisotopes(self.featureMetdata['m/z'].iloc[0], self.featureMetdata['Isotope Distribution'].iloc[0])

			expected = [(379.13643080000003, 100.0),
						(380.13978580000003, 14.3),
						(381.14314080000003, 8.98),
						(382.14649580000003, 0.0461)]
			self.assertEqual(result, expected)

		with self.subTest(msg='Custom delta'):
			result = _buildSpectrumFromQIisotopes(self.featureMetdata['m/z'].iloc[0], self.featureMetdata['Isotope Distribution'].iloc[0], delta=15.1)

			expected = [(379.13643080000003, 100.0),
						(394.23643080000005, 14.3),
						(409.33643080000002, 8.98),
						(424.43643080000004, 0.0461)]
			self.assertEqual(result, expected)
