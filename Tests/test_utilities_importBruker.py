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

class test_utilities_importBruker(unittest.TestCase):

	def test_importBrukerSpectrum(self):

		with self.subTest(msg='Little Endian'):
			from nPYc.utilities._importBrukerSpectrum import importBrukerSpectrum
			intensityData, ppm = importBrukerSpectrum(os.path.join('..', '..', 'npc-standard-project',
																	'Raw_Data', 'nmr', 'UnitTest3',
																	'UnitTest3_Serum_Rack01_RCM_190116',
																	'10', 'pdata', '1', '1r'),
													19.70476, 18028.8461538461, -6, 600.450006160573, 131072, 0)
			expectedIntergral = 15782070494
			expectedMinPPM = -10.320568369935412
			expectedMaxPPM = 19.70476
			expectedDeltaPPM = 0.00022907682378203731

			intergral = numpy.sum(intensityData, dtype='i8') # Seeing some overflow issues in certain Win installs
			minPPM = numpy.min(ppm)
			maxPPM = numpy.max(ppm)
			deltaPPM = ppm[0] - ppm[1]

			self.assertEqual(intergral, expectedIntergral)
			numpy.testing.assert_allclose(minPPM, expectedMinPPM)
			numpy.testing.assert_allclose(maxPPM, expectedMaxPPM)
			numpy.testing.assert_allclose(deltaPPM, expectedDeltaPPM)

		with self.subTest(msg='Big Endian'):
			from nPYc.utilities._importBrukerSpectrum import importBrukerSpectrum
			intensityData, ppm = importBrukerSpectrum(os.path.join('..', '..', 'npc-standard-project',
																	'Raw_Data', 'nmr', 'UnitTest1',
																	'UnitTest1_Urine_Rack1_SLL_270814',
																	'10', 'pdata', '1', '1r'),
													14.7869, 12019.2307692308, -4, 600.289936863629, 32768, 1)
			expectedIntergral = 3599726584
			expectedMinPPM = -5.2348648728622784
			expectedMaxPPM = 14.786899999999999
			expectedDeltaPPM = 0.00061103442099863514

			intergral = numpy.sum(intensityData, dtype='i8') # Seeing some overflow issues in certain Win installs
			minPPM = numpy.min(ppm)
			maxPPM = numpy.max(ppm)
			deltaPPM = ppm[0] - ppm[1]

			self.assertEqual(intergral, expectedIntergral)
			numpy.testing.assert_allclose(minPPM, expectedMinPPM)
			numpy.testing.assert_allclose(maxPPM, expectedMaxPPM)
			numpy.testing.assert_allclose(deltaPPM, expectedDeltaPPM)


	def test_importBrukerSpectrum_raises(self):
		from nPYc.utilities._importBrukerSpectrum import importBrukerSpectrum

		self.assertRaises(IOError, importBrukerSpectrum, 'not a valid path to a file', None, None, None, None, None, None)

		self.assertRaises(NotImplementedError, importBrukerSpectrum, 'path to/2rr', None, None, None, None, None, None)


	def test_importBrukerSpectra_raises(self):
		from nPYc.utilities._importBrukerSpectrum import importBrukerSpectra

		with self.subTest(msg='No Bruker format spectra'):

			self.assertRaises(ValueError, importBrukerSpectra, '.', 'noesygppr1d', 1, dict())

		with self.subTest(msg='No matching pulse program'):

			path = os.path.join('..', '..', 'npc-standard-project',
								'Raw_Data', 'nmr', 'UnitTest3', 
								'UnitTest3_Serum_Rack01_RCM_190116')

			self.assertRaises(ValueError, importBrukerSpectra, path, 'notpresent', 1, dict())


	def test_importBrukerSpectra(self):
		from nPYc.utilities._importBrukerSpectrum import importBrukerSpectra

		Attributes = dict()
		Attributes['variableSize'] = numpy.random.randint(10000, high=50000, size=None)
		
		lowBound = (-0.5 - -5) * numpy.random.random_sample() + -5
		highBound = (19 - 10) * numpy.random.random_sample() + 10
		Attributes['bounds'] = [lowBound, highBound]
		
		Attributes['alignTo'] = 'doublet'
		Attributes['calibrateTo'] = (6 - 4.5) * numpy.random.random_sample() + 4.5
		offset = 5.233 - Attributes['calibrateTo']
		Attributes['ppmSearchRange'] = [4.9, 5.733]
		
		Attributes['LWpeakRange'] = [4.08 - offset, 4.14 - offset]
		Attributes['LWpeakMultiplicity'] = 'quartet'
		Attributes['LWpeakIntesityFraction'] = 1e-4

		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			intensityData, ppm, metadata = importBrukerSpectra(os.path.join('..', '..', 'npc-standard-project',
																			'Raw_Data', 'nmr', 'UnitTest3',
																			'UnitTest3_Serum_Rack01_RCM_190116'),
																			'noesygppr1d', 1, Attributes)

		##
		# Sort to account for filesystem ordering
		##
		metadata.sort_values('Sample File Name', inplace=True)
		sortIndex = metadata.index.values
		intensityData = intensityData[sortIndex, :]
		metadata = metadata.reset_index(drop=True)

		warningsCol = pandas.Series(['',
									'Error loading file',
									'Error loading file',
									'Error loading file',
									'Error loading file',
									'Error loading file',
									'Error loading file',
									'Error loading file',
									'Error loading file',
									'Error loading file'],
									name='Warnings',
									dtype='str')
		pandas.util.testing.assert_series_equal(metadata['Warnings'], warningsCol)

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
		pandas.util.testing.assert_series_equal(metadata['Sample File Name'], sampleName)

		sumFailedImports = numpy.sum(intensityData[1:, :])
		numpy.testing.assert_allclose(sumFailedImports, 0)

		self.assertEqual(len(ppm), Attributes['variableSize'])
		numpy.testing.assert_allclose(numpy.min(ppm), Attributes['bounds'][0])
		numpy.testing.assert_allclose(numpy.max(ppm), Attributes['bounds'][1])

		expectedLW = 0.97083128379560091
		numpy.testing.assert_allclose(metadata.loc[0, 'Line Width (Hz)'], expectedLW)

		expectedERETIC = 181331824.09952235
		numpy.testing.assert_allclose(metadata.loc[0, 'ERETIC Integral'], expectedERETIC)


	def test_importBrukerSpectra_pdata(self):
		from nPYc.utilities._importBrukerSpectrum import importBrukerSpectra

		Attributes = dict()
		Attributes['variableSize'] = numpy.random.randint(10000, high=50000, size=None)
		
		lowBound = (-0.5 - -2) * numpy.random.random_sample() + -2
		highBound = (14.5 - 10) * numpy.random.random_sample() + 10
		Attributes['bounds'] = [lowBound, highBound]
		
		Attributes['alignTo'] = 'singlet'
		Attributes['calibrateTo'] = (0.5 - -0.5) * numpy.random.random_sample() + -0.5
		offset = Attributes['calibrateTo']
		Attributes['ppmSearchRange'] = [-0.5, -0.5]
		
		Attributes['LWpeakRange'] = [-0.5 + offset, 0.5 + offset]
		Attributes['LWpeakMultiplicity'] = 'singlet'
		Attributes['LWpeakIntesityFraction'] = 1e-4

		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			intensityData, ppm, metadata = importBrukerSpectra(os.path.join('..', '..', 'npc-standard-project',
																			'Raw_Data', 'nmr', 'UnitTest1'),
																			'noesygppr1d', 2, Attributes)
		##
		# Sort to account for filesystem ordering
		##
		metadata.sort_values('Sample File Name', inplace=True)
		sortIndex = metadata.index.values
		intensityData = intensityData[sortIndex, :]
		metadata = metadata.reset_index(drop=True)

		warningsCol = pandas.Series([''],
									name='Warnings',
									dtype='str')
		pandas.util.testing.assert_series_equal(metadata['Warnings'], warningsCol)

		sampleName = pandas.Series(['UnitTest1_Urine_Rack1_SLL_270814/30'],
									name='Sample File Name',
									dtype='str')
		pandas.util.testing.assert_series_equal(metadata['Sample File Name'], sampleName)

		self.assertEqual(len(ppm), Attributes['variableSize'])
		numpy.testing.assert_allclose(numpy.min(ppm), Attributes['bounds'][0])
		numpy.testing.assert_allclose(numpy.max(ppm), Attributes['bounds'][1])

		expectedLW = 0.7083017156078231
		numpy.testing.assert_allclose(metadata.loc[0, 'Line Width (Hz)'], expectedLW, atol=0.01)

		expectedERETIC = 152106691.42761227
		numpy.testing.assert_allclose(metadata.loc[0, 'ERETIC Integral'], expectedERETIC)


	def test_importBrukerSpectra_malformederetic(self):
		from nPYc.utilities._importBrukerSpectrum import importBrukerSpectra

		Attributes = dict()
		Attributes['variableSize'] = numpy.random.randint(10000, high=50000, size=None)

		Attributes['bounds'] = [-0.5, 10]

		Attributes['alignTo'] = 'singlet'
		Attributes['calibrateTo'] = 0
		Attributes['ppmSearchRange'] = [-0.3, 0.3]

		Attributes['LWpeakRange'] = [-0.3, 0.3]
		Attributes['LWpeakMultiplicity'] = 'singlet'
		Attributes['LWpeakIntesityFraction'] = 1e-4

		with self.assertWarnsRegex(UserWarning, 'Error parsing `QuantFactorSample`'):
			intensityData, ppm, metadata = importBrukerSpectra(os.path.join('..', '..', 'npc-standard-project',
																			'Raw_Data', 'nmr', 'UnitTest1', 
																			'UnitTest1_Urine_Rack1_SLL_270814', '10'),
																			'noesypr1d', 1, Attributes)

		expectedWarningText = 'Error calculating ERETIC integral'
		self.assertEqual(metadata.loc[0, 'Warnings'], expectedWarningText)
		
		expectedERETIC = numpy.nan
		numpy.testing.assert_allclose(metadata.loc[0, 'ERETIC Integral'], expectedERETIC)


	def test_parseQuantFactorSample(self):
		from nPYc.utilities._importBrukerSpectrum import parseQuantFactorSample

		with self.subTest(msg='Urine'):
			path = os.path.join('..', '..', 'npc-standard-project',
								'Raw_Data', 'nmr', 'UnitTest1',
								'UnitTest1_Urine_Rack1_SLL_270814', '20',
								'QuantFactorSample.xml')

			obtained = parseQuantFactorSample(path)

			self.assertEqual(obtained, (12.0, 0.75, 10.0))

		with self.subTest(msg='Serum'):
			path = os.path.join('..', '..', 'npc-standard-project',
								'Raw_Data', 'nmr', 'UnitTest3',
								'UnitTest3_Serum_Rack01_RCM_190116', '10',
								'QuantFactorSample.xml')

			obtained = parseQuantFactorSample(path)

			self.assertEqual(obtained, (15.0, 0.75, 10.0))
