import numpy
from hashlib import sha1

from ._normaliserABC import Normaliser


class ProbabilisticQuotientNormaliser(Normaliser):
	"""
	Normalisation object which performs Probabilistic Quotient normalisation (Dieterle *et al* Analytical Chemistry, 78(13):4281 – 90, 2006)

	:param reference: Source of the reference profile. If ``None``, use the median of **X**, if an int treat as the index of a spectrum in **X** to use as the reference, if an array with same width as X, treat as the reference profile.
	:type reference: str, int, or numpy.ndarray
	:param referenceDescription: A textual description of the reference provided
	:type referenceDescription: None, or str
	:param bool keepMagnitude: If ``True`` scales **X** such that the mean area of **X** remains constant for the dataset as a whole.

	"""

	def __init__(self, reference=None, referenceDescription=None):

		self._normalisationcoefficients = None
		self._reference = reference
		self._norm_hash = None
		self._referenceDescription = referenceDescription

	def _reset(self):
		"""
		Resets :py:attr:`normalisation_coefficients` causing them to be calculated again next time :py:meth`normalise` is called.
		"""
		self._normalisationcoefficients = None
		self._norm_hash = None
		self._reference = None
		self._referenceDescription = None

	@property
	def normalisation_coefficients(self):
		return self._normalisationcoefficients

	@property
	def reference(self):
		"""
		Allows the reference profile used to calculated fold-changes to be queried or set.

		:return: The reference profile used to calculate normalisation coefficients
		"""
		return self._reference

	@reference.setter
	def reference(self, value):
		self._reference = value

	@reference.deleter
	def reference(self):
		# The whole object has to be reset
		self._reset()

	def normalise(self, X):
		"""
		Apply Probabilistic Quotient normalisation to a dataset.

		:param X: Data intensity matrix
		:type X: numpy.ndarray, shape [n_samples, n_features]
		:param reference: Spectrum to use as the normalisation reference
		:type reference: numpy.ndarray, shape [n_features]
		:return: A read-only, normalised view of **X**
		:rtype: numpy.ndarray, shape [n_samples, n_features]
		:raises ValueError: if X is not a numpy 2-d array representing a data matrix
		"""
		try:

			if X.ndim != 2:
				raise ValueError('X is not a valid data intensity matrix')
			# Assume reference = nanmedian if None is passed
			if self._reference is None:
				self._reference = numpy.nanmedian(X, axis=0)
			else:
				if self._reference.shape[0] != X.shape[1]:
					raise ValueError('The dimensions of X and the reference provided do not match')

			# Do not repeat coefficient calculation if unnecessary
			currentNormHash = sha1(numpy.ascontiguousarray(self._reference)).hexdigest() + \
							sha1(numpy.ascontiguousarray(X)).hexdigest()

			if self._norm_hash == currentNormHash:
				X = X / self._normalisationcoefficients[:, None]

			else:
				##
				# Mask out features that are not finite or 0 
				##
				featureMask = numpy.logical_and(numpy.isfinite(self._reference),
												self._reference != 0)

				fold_change_matrix = X[:, featureMask] / self._reference[featureMask]

				# Change all 0's to nan so they are ignored
				fold_change_matrix[fold_change_matrix == 0] = numpy.nan

				self._normalisationcoefficients = numpy.absolute(numpy.nanmedian(fold_change_matrix, axis=1))

				# Set 0 cofficients to 1
				self._normalisationcoefficients[self._normalisationcoefficients == 0] = 1

				X = X / self._normalisationcoefficients[:, None]

				self._norm_hash = currentNormHash

			# Prevent writing back to the normalised array
			X.setflags(write=False)

			return X

		except ValueError as valerr:
			raise valerr

	def __eq__(self, other):

		if isinstance(other, ProbabilisticQuotientNormaliser):
			return self.reference == other.reference
		else:
			return False

	def __str__(self):
		if self._reference is None:
			string = 'Normalised to median fold-change, reference profile was the median profile.'
		else:
			string = 'Normalised to median fold-change, reference profile was %s.' % (self._referenceDescription)

		return string