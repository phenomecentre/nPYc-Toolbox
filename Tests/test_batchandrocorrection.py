"""
Test that batch and run-order correction behaves sensibly with a combination of synthetic and model datasets.
"""

import numpy
import sys
import unittest

sys.path.append("..")
import nPYc
from generateTestDataset import generateTestDataset
from nPYc.enumerations import SampleType, AssayRole


class test_rocorrection(unittest.TestCase):
	"""
	Use stored and synthetic datasets to validate run-order correction.
	"""

	def setUp(self):
		self.noSamp = numpy.random.randint(500, high=2000, size=None)
		self.noFeat = numpy.random.randint(50, high=100, size=None)

		self.msData = generateTestDataset(self.noSamp, self.noFeat, dtype='MSDataset', sop='GenericMS')

		self.msDataERCorrection = generateTestDataset(20, 2, dtype='MSDataset', sop='GenericMS')
		self.msDataERCorrection._intensityData[self.msDataERCorrection.sampleMetadata['SampleType'] == SampleType.StudyPool, :] = 0.5
		self.msDataERCorrection._intensityData[self.msDataERCorrection.sampleMetadata['SampleType'] == SampleType.StudySample, :] = 2
		self.msDataERCorrection._intensityData[self.msDataERCorrection.sampleMetadata['SampleType'] == SampleType.ExternalReference, :] = 1

		self.msDataLRCorrection = generateTestDataset(20, 2, dtype='MSDataset', sop='GenericMS')
		self.msDataLRCorrection.sampleMetadata['SampleType'] = SampleType.StudyPool
		self.msDataLRCorrection.sampleMetadata.loc[self.msDataLRCorrection.sampleMetadata['AssayRole'] != AssayRole.PrecisionReference, 'AssayRole'] = AssayRole.LinearityReference

	def test_correctMSDataset(self):
		with self.subTest(msg='Test correctMSdataset parallelisation'):
			"""
			Check that parallel and single threaded code paths return the same result.
			"""

			correctedDataP = nPYc.batchAndROCorrection.correctMSdataset(self.msData, parallelise=True)
			correctedDataS = nPYc.batchAndROCorrection.correctMSdataset(self.msData, parallelise=False)

			numpy.testing.assert_array_almost_equal(correctedDataP.fit, correctedDataS.fit, err_msg="Serial and parallel fits not equal.")

			numpy.testing.assert_array_almost_equal(correctedDataP.intensityData, correctedDataS.intensityData, err_msg="Serial and parallel corrected data not equal.")

		with self.subTest(msg='Test correctMSdataset correction sample selection'):
			"""
			Check that parallel and single threaded code paths return the same result.
			"""
			expectedCorrectedERDataFit = numpy.ones((20, 2))

			expectedCorrectedERDataIntensity = numpy.ones((20, 2))
			expectedCorrectedERDataIntensity[self.msDataERCorrection.sampleMetadata['SampleType'] == SampleType.StudyPool, :] = 0.5
			expectedCorrectedERDataIntensity[self.msDataERCorrection.sampleMetadata['SampleType'] == SampleType.StudySample, :] = 2

			correctedDataER = nPYc.batchAndROCorrection.correctMSdataset(self.msDataERCorrection, parallelise=False, correctionSampleType=SampleType.ExternalReference)

			numpy.testing.assert_array_almost_equal(correctedDataER.fit, expectedCorrectedERDataFit, err_msg="Correction trendlines are not equal.")

			numpy.testing.assert_array_almost_equal(correctedDataER.intensityData, expectedCorrectedERDataIntensity, err_msg="Corrected intensities are not equal")

		with self.subTest(msg='Test linearity reference samples are not corrected'):
			"""
			Check that linearity samples (by default not corrected, see 'GenericMS.py') are not corrected
			"""

			correctedDataLR = nPYc.batchAndROCorrection.correctMSdataset(self.msDataLRCorrection,
																		 correctionSampleType=SampleType.StudyPool)

			LRmask = (correctedDataLR.sampleMetadata['SampleType'] == SampleType.StudyPool) & \
					 (correctedDataLR.sampleMetadata['AssayRole'] == AssayRole.LinearityReference)

			numpy.testing.assert_array_almost_equal(correctedDataLR.intensityData[LRmask, :],
													self.msDataLRCorrection.intensityData[LRmask, :],
													err_msg="By default, linearity reference samples should not be corrected")


class test_rocorrection_synthetic(unittest.TestCase):

	def setUp(self):

		# Generate synthetic data
		#Use a random number of samples
		noSamples = numpy.random.randint(100, high=500, size=None)

		# Genreate monotonically increasing data
		self.testD = numpy.linspace(1,10, num=noSamples)

		# Generate run order
		self.testRO = numpy.linspace(1,noSamples, num=noSamples, dtype=int)

		# Build SR mask, and make sure first and last samples are references
		self.testSRmask = numpy.zeros_like(self.testD, dtype=bool)
		self.testSRmask[0:noSamples:7] = True
		self.testSRmask[:2] = True
		self.testSRmask[-2:] = True

	def test_runOrderCompensation_synthetic(self):
		"""
		Testing unpacking of parameters for RO correction.
		(only testing LOWESS at present)
		"""

		corrected, fit = nPYc.batchAndROCorrection._batchAndROCorrection.runOrderCompensation(self.testD,
																							self.testRO,
																							self.testSRmask,
																							{'window': 11,
																							 'method': 'LOWESS',
																							 'align': 'median'})

		numpy.testing.assert_array_almost_equal(self.testD, fit)
		numpy.testing.assert_array_almost_equal(numpy.std(corrected), 0.)

	def test_doLOESScorrection_synthetic(self):
		"""
		Correction of a monotonically increasing trend should be essentially perfect.
		"""

		corrected, fit = nPYc.batchAndROCorrection._batchAndROCorrection.doLOESScorrection(self.testD[self.testSRmask],
																							self.testRO[self.testSRmask],
																							self.testD,
																							self.testRO,
																							window=11)

		numpy.testing.assert_array_almost_equal(self.testD, fit)
		numpy.testing.assert_array_almost_equal(numpy.std(corrected), 0.)

	def test_batchCorrection_synthetic(self):

		# We need at least two features - pick which randomly
		testD2 = numpy.array([self.testD, self.testD]).T
		featureNo = numpy.random.randint(0, high=1, size=None)

		output = nPYc.batchAndROCorrection._batchAndROCorrection._batchCorrection(testD2,
																				self.testRO,
																				self.testSRmask,
																				numpy.ones_like(self.testRO),
																				[featureNo],
																				{'align': 'mean', 'window': 11, 'method': 'LOWESS'},
																				0)

		featureNo_out = output[featureNo][0]
		corrected = output[featureNo][1]
		fit = output[featureNo][2]

		self.assertEqual(featureNo, featureNo_out)

		numpy.testing.assert_array_almost_equal(self.testD, fit)
		numpy.testing.assert_array_almost_equal(numpy.std(corrected), 0.)


class test_batchcorrection(unittest.TestCase):
	"""
	Test alignment of batch offsets
	"""

	def setUp(self):
		##
		# Generate synthetic data
		##
		#Use a random number of samples
		noSamples = numpy.random.randint(100, high=500, size=None)

		# Generate run order
		self.testRO = numpy.linspace(1,noSamples, num=noSamples, dtype=int)
		
		# Generate batches
		self.batch = numpy.ones(noSamples)
		splitPoint = numpy.random.randint(noSamples / 4., noSamples / 2., size=None)
		self.batch[splitPoint:] = self.batch[splitPoint:] + 1

		# Build SR mask, and make sure first and last samples are references
		self.testSRmask = numpy.zeros(noSamples, dtype=bool)
		self.testSRmask[0:noSamples:7] = True
		self.testSRmask[:2] = True
		self.testSRmask[-2:] = True

		# Generate normally distributed separately for each batch
		self.testD = numpy.zeros(noSamples)
		batches = list(set(self.batch))
		for batch in batches:
			batchMask = numpy.squeeze(numpy.asarray(self.batch == batch, 'bool'))
			noSamples = sum(batchMask)
	
			batchMean = numpy.random.randn(1) * numpy.random.randint(1, 1000, size=None)
			sigma = numpy.random.randn(1)
			self.testD[batchMask] = sigma * numpy.random.randn(noSamples) + batchMean

	def test_batchCorrection_synthetic(self):
		"""
		Check we can correct the offset in averages of two normal distributions
		"""
		# We need at least two features - pick which randomly
		testD2 = numpy.array([self.testD, self.testD]).T
		featureNo = numpy.random.randint(0, high=1, size=None)

		with self.subTest(msg='Checking alignment to mean'):
			overallMean = numpy.mean(self.testD[self.testSRmask])

			output = nPYc.batchAndROCorrection._batchAndROCorrection._batchCorrection(testD2,
																					self.testRO,
																					self.testSRmask,
																					self.batch,
																					[featureNo],
																					{'align': 'mean', 'window': 11, 'method': None},
																					0)

			means = list()
			batches = list(set(self.batch))
			for batch in batches:
				feature = output[featureNo][1]
				means.append(numpy.mean(feature[(self.batch == batch) & self.testSRmask]))

			numpy.testing.assert_allclose(means, overallMean)

		with self.subTest(msg='Checking alignment to median'):
			overallMedian = numpy.median(self.testD[self.testSRmask])

			output = nPYc.batchAndROCorrection._batchAndROCorrection._batchCorrection(testD2,
																					self.testRO,
																					self.testSRmask,
																					self.batch,
																					[featureNo],
																					{'align': 'median', 'window': 11, 'method': None},
																					0)

			medians = list()
			batches = list(set(self.batch))
			for batch in batches:
				feature = output[featureNo][1]
				medians.append(numpy.median(feature[(self.batch == batch) & self.testSRmask]))

			numpy.testing.assert_allclose(medians, overallMedian)

		with self.subTest(msg='Checking alignment=\'none\', with method == None'):

			output = nPYc.batchAndROCorrection._batchAndROCorrection._batchCorrection(testD2,
																					self.testRO,
																					self.testSRmask,
																					self.batch,
																					[featureNo],
																					{'align': 'no', 'window': 11, 'method': None},
																					0)
			# Check if means are unchanged before and after correction (with method == None)
			means = list()
			batchWiseMeans = list()
			batches = list(set(self.batch))
			for batch in batches:
				feature = output[featureNo][1]
				batchWiseMeans.append(numpy.mean(self.testD[(self.batch == batch) & self.testSRmask]))
				means.append(numpy.mean(feature[(self.batch == batch) & self.testSRmask]))

			numpy.testing.assert_allclose(means, batchWiseMeans)

			with self.subTest(msg='Checking alignment=\'none\', with method == LOWESS'):

				output = nPYc.batchAndROCorrection._batchAndROCorrection._batchCorrection(
					testD2,
					self.testRO,
					self.testSRmask,
					self.batch,
					[featureNo],
					{'align': 'no', 'window': 11, 'method': 'LOWESS'},
					0)

				# Check if means are unchanged before and after correction (with method == 'LOWESS')
				# Without alignment, the mean of corrected data is expected to be close to 1 if batch
				# contains positive values and -np.inf if batch contains negative values
				batches = list(set(self.batch))
				expectedMeans = numpy.array([1 if numpy.mean(self.testD[(self.batch == x) & self.testSRmask]) > 0 else -numpy.inf for x in batches])
				means = list()
				for batch in batches:
					feature = output[featureNo][1]
					means.append(numpy.mean(
						feature[(self.batch == batch) & self.testSRmask]))
				#print("expected means = %s" % expectedMeans)
				#print("means = %s" % means)

				numpy.testing.assert_allclose(means, expectedMeans,  rtol=1.5e-02)
				#print("tol = %s" % 1.5e-02)

		def test_correctMSdataset_raises(self):

			with self.subTest(msg='Object type'):
				self.assertRaises(TypeError, nPYc.batchAndROCorrection.correctMSdataset, 's')

			with self.subTest(msg='Parallelise type'):
				dataset = nPYc.MSDataset('')
				self.assertRaises(TypeError, nPYc.batchAndROCorrection.correctMSdataset, dataset, parallelise=1)


if __name__ == '__main__':
	unittest.main()
