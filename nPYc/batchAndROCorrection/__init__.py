"""
The :py:mod:`~nPYc.batchAndROCorrection` module provides tools to detect and correct for per-feature run-order and batch effects in datasets, by characterising the effect in reference samples and interpolating a correction factor to the intermediate samples.

.. figure:: _static/plotBatchAndROCorrection.svg
	:figwidth: 70%
	:alt: Samples with run-order and batch correction applied
	
	Samples pre and post run-order and batch correction, plus the fit applied to a dataset.
"""
from ._batchAndROCorrection import correctMSdataset

__all__ = ['correctMSdataset']
