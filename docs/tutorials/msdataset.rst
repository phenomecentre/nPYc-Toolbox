Preprocesssing *&* Quality-control of UPLC-MS profiling datasets
----------------------------------------------------------------

By default the nPYc toolbox assumes an :py:class:`~nPYc.objects.MSDataset` instance contains untargeted peak-picked UPLC-MS data, and defines two primary quality control criteria for the features detected, as outlined in Lewis *et al.* [#]_.

* Precision of measurement
	A Relative Standard Deviation (RSD) threshold ensures that only features measured with a precision above this level are propagated on to further data analysis. This can be defined both in absolute terms, as measured on reference samples, but also by removing features where analytical variance is not sufficiently lower than biological variation.
	In order to characterise RSDs, the dataset must include a sufficient number of precision reference samples, ideally a study reference pool to allow calculation of RSDs for all detected features.
* Linearity of response
	By filtering features based on the linearity of their measurement *vs* concentration in the matrix, we ensure that only features that can be meaningfully related to the study design are propagated into the analysis.
	To asses linearity, features must be assayed across a range of concentrations, again in untargeted assays, using the pooled study reference will ensure all relevant features are represented.

Beyond feature QC, the toolbox also allows for the detection and reduction of analytical run-order and batch effects.

.. [#] Development and Application of Ultra-Performance Liquid Chromatography-TOF MS for Precision Large Scale Urinary Metabolic Phenotyping, Lewis MR, *et al.*, **Anal. Chem.**, 2016, 88, pp 9004-9013