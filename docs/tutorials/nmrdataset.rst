Preprocesssing *&* Quality-control of NMR profiling datasets
------------------------------------------------------------

:py:class:`~nPYc.objects.NMRDataset` objects containing spectral data, may have their per-sample analytical quality assessed on the criteria laid out in Dona *et al.* [#]_, being judged on:

* Line-width
	By default, line-widths below 1.4\ Hz, are considered acceptable
* Even baseline
	The noise in the baseline regions flanking the spectrum are expected to have equal variance across the dataset, and not be predominantly below zero
* Adequate water-suppression
	The residual water signal should not affect the spectrum outside of the 4.9 to 4.5\ ppm region

Before finalising the dataset, typically the wings of the spectrum will be trimmed, and the residual water signal and references resonance removed. Where necessary the chemical shift scale can also referenced to a specified resonance.

.. [#] Precision High-Throughput Proton NMR Spectroscopy of Human Urine, Serum, and Plasma for Large-Scale Metabolic Phenotyping, Anthony C. Dona *et al.* **Anal. Chem.**, 2014, 86 (19), pp 9887–9894
