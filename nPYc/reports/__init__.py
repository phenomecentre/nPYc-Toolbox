"""
The :py:mod:`~nPYc.reports` submodule provides functions to generate a variety of automated reports on :py:class:`~nPYc.objects.Dataset` objects. Most reports can be displayed inline (i.e. in a Jupyter notebook), or saved to disk as an HTML file with images.
"""
from .featureID import generateMSIDrequests
from .generateReport import generateReport
from ._generateSampleReport import _generateSampleReport
from .multivariateReport import multivariateQCreport

__all__ = ['generateMSIDrequests', 'generateReport', '_generateSampleReport', 'multivariateQCreport']
