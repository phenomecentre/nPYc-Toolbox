from abc import ABCMeta, abstractmethod
from copy import deepcopy

class Normaliser(metaclass=ABCMeta):
	"""
	Normaliser classes must implement the :py:meth:`~nPYc.utilities.normalisation._normaliserABC.Normaliser.normalise`, :py:meth:`~nPYc.utilities.normalisation._normaliserABC.Normaliser.__eq__` and :py:meth:`~nPYc.utilities.normalisation._normaliserABC.Normaliser.__str__` methods and :py:attr:`~nPYc.utilities.normalisation._normaliserABC.Normaliser.normalisation_coefficients` attribute.
	"""

	@abstractmethod
	def __init__(self):
		pass


	@abstractmethod
	def _reset(self):
		"""
		Resets the objects including any :py:attr:`~nPYc.utilities.normalisation._normaliserABC.Normaliser.normalisation_coefficients`, causing them to be calculated again next time :py:meth:`~nPYc.utilities.normalisation._normaliserABC.Normaliser.normalise` is called.

		Should additionally reset any configuration applied after initialisation.
		"""
		pass


	@property
	@abstractmethod
	def normalisation_coefficients(self):
		"""
		Returns the last set of normalisation coefficients calculated.

		:return: Normalisation coefficients or ``None`` if they have not been generated yet
		:raises AttributeError: Setting the normalisation coefficients directly is not allowed and raises an error
		"""
		pass

	@normalisation_coefficients.setter
	def normalisation_coefficients(self, value):
		"""

		"""
		return AttributeError('Cannot modify coefficients')

	@normalisation_coefficients.deleter
	def normalisation_coefficients(self):
		# At its most basic this is just a reset
		self._reset()


	@abstractmethod
	def normalise(self, X):
		"""
		Apply normalisation to the data in matrix **X** and return a view to the normalised matrix.

		Where relevant the method must ensure that it is not possible to write to the returned normalised **X** where this cannot be meaningfully reflected in the raw **X**.

		:param X: Data intensity matrix
		:type X: numpy.ndarray, shape [n_samples, n_features]
		:return: The normalised **X** matrix
		:rtype: numpy.ndarray, shape [n_samples, n_features]
		:raises ValueError: If **X** is not a numpy 2-d array representing a data matrix
		"""
		pass


	@abstractmethod
	def __eq__(self, other):
		"""
		Compares two :py:class:`~nPYc.utilities.normalisation._normaliserABC.Normaliser` objects. Should return ``True`` when either object would return identical values for identical **X**\ s.

		:param other: Object to compare for equality
		:return: ``True`` if both objects perform equivalent normalisations
		:rtype: bool
		"""
		pass


	@abstractmethod
	def __str__(self):
		"""
		:return: A human readable description of the normalisation the object applies.
		:rtype: str
		"""
		pass


	def __deepcopy__(self, memo):
		cls = self.__class__
		result = cls.__new__(cls)
		memo[id(self)] = result
		for k, v in self.__dict__.items():
			setattr(result, k, deepcopy(v, memo))
		return result
