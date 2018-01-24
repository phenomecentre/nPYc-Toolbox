"""
The :py:mod:`~nPYc.multivariate` module provides tools to conduct multivariate analysis of :py:class:`~nPYc.objects.Dataset` objects.

The module implements\:

* Principal Components Analysis (PCA) [#]_

.. [#]  Pearson, K., "On Lines and Planes of Closest Fit to Systems of Points in Space", Philosophical Magazine. 2 (11):559–572., 1901 doi:10.1080/14786440109462720.
"""
from .multivariateUtilities import pcaSignificance, metadataTypeGrouping
from .exploratoryAnalysisPCA import exploratoryAnalysisPCA

__all__ = ['pcaSignificance', 'metadataTypeGrouping', 'exploratoryAnalysisPCA']
