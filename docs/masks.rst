Sample and Feature Masks
------------------------

Dataset objects contain two internal `mask` vectors, the :py:attr:`~nPYc.objects.Dataset.sampleMask` and the :py:attr:`~nPYc.objects.Dataset.featureMask`. They store whether a sample or feature, respectively, should be used when calculating QC metrics, in the visualisations in the report functions and when exporting the dataset.

There are several functions that modify these internal masks:

- :py:meth:`~nPYc.objects.Dataset.updateMasks` is a method to automatically mask certain samples and/or features, for example, by type or based on dataset specific quality control checks
- :py:meth:`~nPYc.objects.Dataset.excludeSamples` and :py:meth:`~nPYc.objects.Dataset.excludeFeatures` are methods to directly exclude specific samples or features respectively. Masked samples and features will remain in the dataset, but will be hidden, and thus ignored when calling the reporting functions, fitting PCA models, and exporting the pre-processed datasets.
- :py:meth:`~nPYc.objects.Dataset.initialiseMasks` is a method to reset the masks to include all samples/features.
- :py:meth:`~nPYc.objects.Dataset.applyMasks` is a method to permanently exclude from the dataset all samples and features which have been previously masked. After calling this command the excluded features are deleted and the masks are re-initialised so that all remaining samples and features are unmasked. This method should be used only when it is absolutely certain that the masked features and samples are to be removed, as it is permanent so if samples/features were again required, the full dataset would have to be re-imported.

Further details of each of these methods are given below, and full worked examples of how these are used during the import and preprocessing of specific datasets are provided in :doc:`Tutorials<tutorial>`.

In brief, however, the following describes a step-wise example utilising the above functions to generate a feature filtered LC-MS dataset containing only :term:`Study Samples<Study Sample>`, with a specific sample ('PipelineTesting_RPOS_ToF10_U1W04') excluded::

	# To automatically mask features not passing quality control criteria
	msData.updateMasks(filterFeatures=True, filterSamples=False)
	
	# To reset (initialise) the masks, and run with an updated RSD threshold
	msData.initialiseMasks()
	dataset.Attributes['rsdThreshold'] = 20
	msData.updateMasks(filterFeatures=True, filterSamples=False)
	
	# To automatically mask all sample types except for study samples
	msData.updateMasks(filterSamples=True, sampleTypes=[SampleType.StudySample], assayRoles=[AssayRole.Assay], filterFeatures=False)
	
	# To exclude a specific sample, 'PipelineTesting_RPOS_ToF10_U1W04', by 'Sample File Name'
	msData.excludeSamples(['PipelineTesting_RPOS_ToF10_U1W04'], on='Sample File Name', message='User excluded')
	
	# To finally apply the masks and permanently exclude the masked samples and features
	msData.applyMasks()
	

Using updateMasks
=================

:py:meth:`~nPYc.objects.Dataset.updateMasks` is a method to automatically mask certain samples and/or features, for example, by type or based on dataset specific quality control checks. There are a number of different ways in which the 'updateMasks' function can be used, some of which are dataset type specific, common usage includes:


**Using updateMasks to mask samples based on sample type (all dataset types)**

By default, all samples are included in the datasets. However, it is often the case that some sample types (see :doc:`studydesign`) would not be required in the final dataset, or when running various quality control checks.

By setting preferences with the "sampleTypes" and "assayRoles" arguments, samples which are not required can also be masked (see :doc:enumerations for possible options). For example, the dataset would be masked to contain only study samples ('SampleType.StudySample, AssayRole.Assay') and study reference samples ('SampleType.StudyPool, AssayRole.PrecisionReference') by running the following::

	dataset.updateMasks(filterSamples=True, sampleTypes=[SampleType.StudySample, SampleType.StudyPool], assayRoles=[AssayRole.Assay, AssayRole.PrecisionReference], filterFeatures=False)


**Using updateMasks to mask samples failing quality control checks (NMR datasets only)**

For NMR datasets, there are a number of quality control criteria (Dona *et al* [#]_) which are automatically checked (see :doc:`Feature Summary Report: NMR Datasets<reports>` for full details):

- Chemical shift calibration
- Line width
- Baseline consistency
- Quality of solvent suppression

For each of the above, default acceptable values are given in :doc:`configuration/builtinSOPs`, samples not meeting these criteria can be automatically masked by applying "sampleQCChecks" when applying updateMasks, for example::

	# To mask all samples failing on any quality control parameter, "sampleQCChecks" would be set to:
	nmrData.updateMasks(filterSamples=True, sampleQCChecks=['CalibrationFail', 'LineWidthFail', 'BaselineFail', 'SolventPeakFail'], filterFeatures=False)
	
	# If only samples failing on Line Width criteria were required to be masked, "sampleQCChecks" would be set to:
	nmrData.updateMasks(filterSamples=True, sampleQCChecks=['LineWidthFail'], filterFeatures=False)


**Using updateMasks to mask feature failing quality control checks (MS datasets only)**

For LC-MS datasets, features should be filtered based on their individual precision and accuracy (Lewis *et al* [#]_) in the nPYc-Toolbox the default parameters for feature filtering are as follows:

.. table:: LC-MS Feature Filtering Criteria
   :widths: auto
   
   ========================================== ================================================ =================== =====================
   Criteria                                   In                                               Default Value       Assesses
   ========================================== ================================================ =================== =====================
   Correlation to dilution                    :term:`Serial Dilution Sample`                   > 0.7               Intensity responds to changes in abundance (accuracy)
   :term:`Relative Standard Deviation` (RSD)  :term:`Study Reference`                          < 30                Analytical stability (precision)
   RSD in SS * *default value* > RSD in SR    :term:`Study Sample` and :term:`Study Reference` 1.1                 Variation in SS should always be greater than variation in SR
   ========================================== ================================================ =================== =====================
   
The distribution of correlation to dilution, and RSD can be visualised in the *Feature Summary Report* (see :doc:`reports` for more details).

A report summarising number of features passing selection with different criteria can also be produced using::

	nPYc.reports.generateReport(dataset, 'feature selection')
	
This generates a list of the number of features passing each filtering criteria, alongside a heatmap showing the number of features resulting from applying different RSD and correlation to dilution thresholds.

.. figure:: _static/featureSelection_heatmap.svg
	:figwidth: 70%
	:alt: Heatmap of the number of features passing selection with different Residual Standard Deviation (RSD) and correlation to dilution thresholds
	
	Heatmap of the number of features passing selection with different Residual Standard Deviation (RSD) and correlation to dilution thresholds

Criteria can be modified if required, for example for the RSD threshold using::

	dataset.Attributes['rsdThreshold'] = 20
	
Features failing selection can be automatically flagged for removal using::

	dataset.updateMasks(filterSamples=False, filterFeatures=True)


**Using updateMasks to mask unwanted/uninformative features (NMR datasets only)**

For NMR datasets, feature filtering typically takes the form of removing one or more sections of the spectra known to contain unwanted or un-informative signals.

The regions typically removed are pre-defined in the :doc:`Configuration Files<configuration/builtinSOPs>`, and can be automatically flagged for removal::

	nmrData.updateMasks(filterSamples=False, filterFeatures=True)
	
Additional regions can also be masked by using 'updateMasks' with the additional "exclusionRegions" parameter. For example, to also mask the region between 8.4 and 8.5 ppm the following would be run::

    nmrData.updateMasks(filterSamples=False, filterFeatures=True, exclusionRegions=[(8.4, 8.5)])
	

Using excludeSamples and excludeFeatures
========================================

The 'updateMasks' function works to mask samples or features not meeting specific criteria, in addition to this, the nPYc-Toolbox also contains two additional methods to mask specific samples or features directly, :py:meth:`~nPYc.objects.Dataset.excludeSamples` and :py:meth:`~nPYc.objects.Dataset.excludeFeatures` respectively.

These functions both take three input arguments:

1. A list of sample or feature identifiers
2. "on": the name of the column in 'sampleMetadata' (for 'excludeSamples') or 'featureMetadata' (for 'excludeFeatures') where these identifiers can be found
3. "message": an optional message as to why these samples or features have been flagged for masking

Depending on the dataset type, and the sample and feature metadata available, the value of "on" could differ, but some examples include::

	# To exclude a sample with 'Sample File Name' = 'DEVSET U 1D NMR raw data files/930'
	nmrData.excludeSamples(['DEVSET U 1D NMR raw data files/930'], on='Sample File Name', message='Unknown type')

	# To exclude all features with 'ppm' > 8
	nmrData.excludeFeatures([nmrData.featureMetadata['ppm'][nmrData.featureMetadata['ppm'] > 8].values], on='ppm', message='ppm > 8')
	
	# To exclude a sample with 'Run Order' = 93:
	msDatacorrected.excludeSamples([93], on='Run Order', message='outlying TIC')
	
	
Using applyMasks and initialiseMasks
====================================

Once satisfied with the sample and feature masks, exclusions can be applied (permanently removed from the dataset) using the :py:meth:`~nPYc.objects.Dataset.applyMasks` function::

	msDatacorrected.applyMasks() 

This method should be used only when it is absolutely certain that the masked features and samples are to be removed, as the full dataset would otherwise have to be re-imported.

Before masks have been applied, however, feature/sample masking can be reset to include all samples/features using :py:meth:`~nPYc.objects.Dataset.initialiseMasks`::

	msDatacorrected.initialiseMasks() 


.. [#] Anthony C Dona, Beatriz Jiménez, Hartmut Schäfer, Eberhard Humpfer, Manfred Spraul, Matthew R Lewis, Jake TM Pearce, Elaine Holmes, John C Lindon and Jeremy K Nicholson. Precision High-Throughput Proton NMR Spectroscopy of Human Urine, Serum, and Plasma for Large-Scale Metabolic Phenotyping. Analytical Chemistry, 86(19):9887-9894, 2014. URL: http://dx.doi.org/10.1021/ac5025039
	
.. [#] Matthew R Lewis, Jake TM Pearce, Konstantina Spagou, Martin Green, Anthony C Dona, Ada HY Yuen, Mark David, David J Berry, Katie Chappell, Verena Horneffer-van der Sluis, Rachel Shaw, Simon Lovestone, Paul Elliott, John Shockcor, John C Lindon, Olivier Cloarec, Zoltan Takats, Elaine Holmes and Jeremy K Nicholson. Development and Application of Ultra-Performance Liquid Chromatography-TOF MS for Precision Large Scale Urinary Metabolic Phenotyping. Analytical Chemistry, 88(18):9004-9013, 2016. URL: http://dx.doi.org/10.1021/acs.analchem.6b01481