"""
The :py:mod:`~nPYc.utilities` module implements several Normaliser objects, that perform intensity normalisation on the provided numpy matrix.

All normaliser objects must implement the :py:class:`~nPYc.utilities.normalisation._normaliserABC.Normaliser` abstract base class.

Normalisers may be configured as required upon initialisation, then a normalised view of a matrix obtained by passing the data to be normalised to the :py:meth:`~Normaliser.normalise` method.

Once :py:meth:`~Normaliser.normalise` has been called, the normalisation coefficients last used can be obtained from :py:attr:`~Normaliser.normalisation_coefficients`.
"""
from ._normaliserABC import Normaliser
from ._nullNormaliser import NullNormaliser
from ._probabilisticQuotientNormaliser import ProbabilisticQuotientNormaliser
from ._totalAreaNormaliser import TotalAreaNormaliser

__all__ = ['NullNormaliser', 'ProbabilisticQuotientNormaliser', 'TotalAreaNormaliser']
