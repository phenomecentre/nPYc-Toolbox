import numpy
from copy import deepcopy

from ._normaliserABC import Normaliser

class TotalAreaNormaliser(Normaliser):
	"""
	Normalisation object which performs Total Area normalisation. Each row in the matrix provided will be scaled to sum to the same value.

	:param bool keepMagnitude: If ``True`` scales **X** such that the mean area of **X** remains constant for the dataset as a whole.
	"""

	def __init__(self, keepMagnitude=True):
		self._normalisationcoefficients = None
		self._keepMagnitude = keepMagnitude


	def _reset(self):
		"""
		Resets :py:attr:`normalisation_coefficients` causing them to be calculated again next time :py:meth:`normalise` is called.
		"""
		self._normalisationcoefficients = None


	@property
	def normalisation_coefficients(self):
		return self._normalisationcoefficients


	def normalise(self, X):
		"""
		Apply Total Area normalisation to the dataset.

		:param X: Data intensity matrix
		:type X: numpy.ndarray, shape [n_samples, n_features]
		:return: A read-only, normalised view of **X**
		:rtype: numpy.ndarray, shape [n_samples, n_features]
		:raises ValueError: If X is not a numpy 2-d array representing a data matrix
		"""
		try:

			if X.ndim != 2:
				raise ValueError('X is not a valid data intensity matrix')

			areas = numpy.nansum(X, axis=1)

			if self._keepMagnitude:
				scaleFactor = numpy.mean(areas[numpy.isfinite(areas)])
			else:
				scaleFactor = 1

			self._normalisationcoefficients = numpy.divide(areas, scaleFactor)

			X = X / self._normalisationcoefficients[:, None]

			X.setflags(write=False)

			return X

		except ValueError as verr:
			raise verr


	def __eq__(self, other):
		if isinstance(other, TotalAreaNormaliser):
			return self._keepMagnitude == other._keepMagnitude
		else:
			return False


	def __str__(self):
		if self._keepMagnitude:
			return 'Normalised to constant area, preserving magnitude.'
		else:
			return 'Normalised to unit area.'
