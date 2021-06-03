"""
Test core functionality of normaliser objects
"""

import numpy
import sys
import unittest

sys.path.append("..")

from nPYc.utilities.normalisation._nullNormaliser import NullNormaliser
from nPYc.utilities.normalisation._totalAreaNormaliser import TotalAreaNormaliser
from nPYc.utilities.normalisation._probabilisticQuotientNormaliser import ProbabilisticQuotientNormaliser

class test_utilities_normalisation(unittest.TestCase):
	"""

	Test class for the normalisation objects. Contains tests covering the basic functionality of individual objects
	and their interaction and usage inside the nPYc Dataset objects.

	"""

	def setUp(self):
		# Simulate some data
		self.noSamp = numpy.random.randint(5, high=50, size=None)
		self.noFeat = numpy.random.randint(60, high=200, size=None)

		self.X = numpy.random.randn(self.noSamp, self.noFeat)


	# Object test
	def test_nullNormaliser(self):
		"""
		Check that the NullNormaliser works
		"""
		# Check if output data = input data (its not supposed to do anything)
		numpy.testing.assert_array_equal(self.X, NullNormaliser().normalise(self.X), err_msg="Null Normaliser not working as expected")
		self.assertEqual(1, NullNormaliser().normalisation_coefficients)


	def test_nullNormaliser_eq_(self):
		"""
		Check that the NullNormaliser equality testing works
		"""
		with self.subTest():
			norm = NullNormaliser()
			norm2 = NullNormaliser()
			
			self.assertEqual(norm, norm2)

		pqn = ProbabilisticQuotientNormaliser()
		tanorm = TotalAreaNormaliser(keepMagnitude=False)
		tanorm2 = TotalAreaNormaliser(keepMagnitude=True)

		notEqualList = [1, True, 'str', 1.1, list(), dict(), tanorm, tanorm2, pqn]
		norm = NullNormaliser()
		for comparison in notEqualList:
			with self.subTest(msg=comparison):
				self.assertNotEqual(norm, comparison)

class test_utilities_totalAreaNormaliser(unittest.TestCase):

	def setUp(self):
		# Simulate some data
		self.noSamp = numpy.random.randint(5, high=50, size=None)
		self.noFeat = numpy.random.randint(60, high=200, size=None)

		self.X = numpy.random.randn(self.noSamp, self.noFeat)

	# Object test
	def test_totalAreaNormaliser(self):
		"""
		Check that the TotalAreaNormaliser works
		"""
		# Check if algorithm is being performed correctly
		tanorm = TotalAreaNormaliser(keepMagnitude=False)
		X = numpy.copy(self.X)
		x_normed = X/X.sum(axis=1)[:, None]

		numpy.testing.assert_array_almost_equal(x_normed, tanorm.normalise(X), err_msg="Total Area normaliser not working correctly")
		numpy.testing.assert_array_equal(X.sum(axis=1), tanorm.normalisation_coefficients)


	def test_eq_(self):
		"""
		Check that the TotalAreaNormaliser equality testing works
		"""
		with self.subTest(msg='Test keepMagnitude is respected'):
			tanorm = TotalAreaNormaliser(keepMagnitude=False)
			tanorm2 = TotalAreaNormaliser(keepMagnitude=False)
			tanorm3 = TotalAreaNormaliser(keepMagnitude=True)
			tanorm4 = TotalAreaNormaliser(keepMagnitude=True)

			self.assertEqual(tanorm, tanorm2)
			self.assertEqual(tanorm3, tanorm3)
			self.assertNotEqual(tanorm, tanorm3)

		pqn = ProbabilisticQuotientNormaliser()
		notEqualList = [1, True, 'str', 1.1, list(), dict(), NullNormaliser(), pqn]
		tanorm = TotalAreaNormaliser()
		for comparison in notEqualList:
			with self.subTest(msg=comparison):
				self.assertNotEqual(tanorm, comparison)


	def test_raises(self):

		tanorm = TotalAreaNormaliser(keepMagnitude=False)

		with self.subTest(msg='Not two dimensions'):

			X = numpy.random.randn(2,2,2)
			self.assertRaises(ValueError, tanorm.normalise, X)

			X = numpy.random.randn(2)
			self.assertRaises(ValueError, tanorm.normalise, X)


	def test_repr(self):

		with self.subTest(msg='Preserving magnitude'):

			tanorm = TotalAreaNormaliser(keepMagnitude=False)
			strform = str(tanorm)
			self.assertEqual(strform, 'Normalised to unit area.')

		with self.subTest(msg='Preserving magnitude'):

			tanorm = TotalAreaNormaliser(keepMagnitude=True)
			strform = str(tanorm)
			self.assertEqual(strform, 'Normalised to constant area, preserving magnitude.')


class test_utilities_probabilisticQuotientNormaliser(unittest.TestCase):

	def setUp(self):
		# Simulate some data
		self.noSamp = numpy.random.randint(5, high=50, size=None)
		self.noFeat = numpy.random.randint(60, high=200, size=None)

		self.X = numpy.random.randn(self.noSamp, self.noFeat)


	# Object test
	def test_probabilisticQuotientNormaliser(self):
		"""
		Check that the ProbabilisticQuotientNormaliser 
		"""

		X = numpy.copy(self.X)
		reference = numpy.nanmedian(X, axis=0)
		pqn_norm = ProbabilisticQuotientNormaliser()

		fold_change_matrix = X / reference
		pqn_norm_coefs = numpy.absolute(numpy.median(fold_change_matrix, axis=1))
		pqn_normed = X / pqn_norm_coefs[:, None]

		numpy.testing.assert_array_almost_equal(pqn_normed, pqn_norm.normalise(self.X), err_msg="PQN normaliser not working correctly - mismatching normalised data")
		# Run twice to pick up the hashed coefficients
		numpy.testing.assert_array_almost_equal(pqn_normed, pqn_norm.normalise(self.X), err_msg="PQN normaliser not working correctly - mismatching normalised data")
		numpy.testing.assert_array_almost_equal(pqn_norm_coefs, pqn_norm.normalisation_coefficients, err_msg="PQN normaliser not working correctly - non-matching PQN coefficients")
		numpy.testing.assert_array_equal(reference, pqn_norm.reference)

	def test_nans(self):

		self.X[0, 0] = numpy.nan
		pqn_norm = ProbabilisticQuotientNormaliser()

		pqn_norm.normalise(self.X)

	def test_repr(self):

		with self.subTest(msg='Default reference profile'):
			pqn_norm = ProbabilisticQuotientNormaliser()
			strform = str(pqn_norm)
			self.assertEqual(strform, 'Normalised to median fold-change, reference profile was the median profile.')

	def test_delete_reference(self):

		pqn_norm = ProbabilisticQuotientNormaliser()
		pqn_norm.normalise(self.X)

		del pqn_norm.reference

		self.assertIsNone(pqn_norm.normalisation_coefficients)

	def test_pass_reference(self):

		X = numpy.copy(self.X)
		reference = numpy.abs(numpy.random.randn(self.noFeat))
		pqn_norm = ProbabilisticQuotientNormaliser(reference=reference)
		pqn_norm.normalise(X)

		featureMask = numpy.logical_and(numpy.isfinite(reference), reference != 0)
		fold_change_matrix = X[:, featureMask] / reference[featureMask]
		fold_change_matrix[fold_change_matrix == 0] = numpy.nan
		pqn_norm_coefs = numpy.absolute(numpy.median(fold_change_matrix, axis=1))
		# Set 0 cofficients to 1
		pqn_norm_coefs[pqn_norm_coefs == 0] = 1

		numpy.testing.assert_array_almost_equal(pqn_norm_coefs,
												pqn_norm.normalisation_coefficients,
												err_msg="Change of reference does not work")

	def test_raises(self):

		pqn_norm = ProbabilisticQuotientNormaliser()
		with self.subTest(msg='1D X matrix'):
			X = numpy.array([2])
			self.assertRaises(ValueError, pqn_norm.normalise, X)

		with self.subTest(msg='3D X matrix'):
			X = numpy.array([2,2,2])
			self.assertRaises(ValueError, pqn_norm.normalise, X)

		with self.subTest(msg='Reference wrong size'):
			X = numpy.array([5,5])
			pqn_norm.reference = numpy.array([4])
			self.assertRaises(ValueError, pqn_norm.normalise, X)


if __name__ == '__main__':
	unittest.main()
