import numpy

from ._normaliserABC import Normaliser

class NullNormaliser(Normaliser):
	"""
	Null normalisation object which performs no normalisation, returning the provided matrix unchanged when :py:meth:`~nPYc.utilities.normalisation.NullNormaliser.normalise` is called.
	"""

	def __init__(self):
		pass

	def _reset(self):
		"""
		Does nothing as coefficients in the :py:class:`NullNormaliser` are always 1.
		"""
		pass


	@property
	def normalisation_coefficients(self):
		"""
		Returns normalisation coefficients.
		:return: 1
		"""
		return 1


	def normalise(self, X):
		"""
		Returns **X** unchanged.

		:param X: Data intensity matrix
		:type X: numpy.ndarray, shape [n_samples, n_features]
		:return: The original X matrix without any modification
		:rtype: numpy.ndarray, shape [n_samples, n_features]
		"""
		return X


	def __eq__(self, other):
		"""
		All :py:class:`~nPYc.utilities.normalisation.NullNormaliser` are equal, so just check if the classes are the same,
		:param other: Object to compare to
		:return: If **other** is also a NullNormaliser
		:rtype: Bool
		"""
		return isinstance(other, NullNormaliser)


	def __str__(self):
		return 'No normalisation applied.'
