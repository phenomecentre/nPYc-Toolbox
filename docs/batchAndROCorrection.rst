Batch *&* Run-Order Correction
------------------------------

The :py:mod:`~nPYc.batchAndROCorrection` module provides tools to detect and correct for per-feature run-order and batch effects in datasets, by characterising the effect in reference samples and interpolating a correction factor to the intermediate samples.

.. figure:: _static/plotBatchAndROCorrection.svg
	:figwidth: 70%
	:alt: Samples with run-order and batch correction applied
	
	Samples pre and post run-order and batch correction, plus the fit applied to a dataset.

Run-order and batch correction may be applied following an adapted version of the LOWESS approach proposed by Dunn *et al* [#]_.

In brief, for each MS feature, a LOWESS estimator is fitted on the series of consecutive :term:`Study Reference` samples for each analytical batch (which can be defined by the user, see :doc:`Adding Sample Metadata<../samplemetadata>`). The value for that feature in each sample is corrected by dividing the original intensity value by the interpolated value of the LOWESS curve at its position in the run order (final intensity units are a ratio to intensity in the ”mean” *Study Reference* sample expressed by the LOWESS curve). The window of the LOWESS smoother can be set by the user, care should be taken not to over-fit the run-order correction (see Figure 4).
Batch diverences are corrected by aligning median feature intensities in the *Study Reference* samples between batches.


.. automodule:: nPYc.batchAndROCorrection
   :members:


.. [#] Warwick B Dunn, David Broadhurst, Paul Begley, Eva Zelena, Sue Francis-McIntyre, Nadine Anderson, Marie Brown, Joshau D Knowles, Antony Halsall, John N Haselden, Andrew W Nicholls, Ian D Wilson, Douglas B Kell, Royston Goodacre, and The Human Serum Metabolome (HUSERMET) Consortium. Procedures for large-scale metabolic profiling of serum and plasma using gas chromatography and liquid chromatography coupled to mass spectrometry. Nature Protocols, 6:1060 EP –, 06 2011. URL: http://dx.doi.org/10.1038/nprot.2011.335.