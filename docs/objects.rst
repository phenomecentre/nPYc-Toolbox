Datasets
--------

The nPYc toolbox is built around a core :py:class:`~nPYc.objects.Dataset` class, that represents a collection of measurements, with biological and analytical metadata associated with each sample, and analytical and chemical metadata associated with each feature.

A Dataset class can be set up from a number of common data types, including certain raw data formats, common interchange formats, and the outputs of popular data-processing tools. There are three main Dataset derived subclasses, each specific for a certain data type:
 
* :py:class:`~nPYc.objects.MSDataset` for LC-MS profiling data
* :py:class:`~nPYc.objects.NMRDataset` for NMR profiling data
* :py:class:`~nPYc.objects.TargetedDataset` for targeted datasets

These can be created using (e.g. for LC-MS data)::
	
	msData = nPYc.MSDataset('path to data')

When setting up the Dataset classes, default parameters are loaded from their associated :doc:`Configuration Files<configuration/configuration>`, and subsequently saved in the :py:attr:`~nPYc.objects.Dataset.Attributes` dictionary. 

For NMR data, the nPYc-Toolbox contains two default configuration files, 'GenericNMRUrine' and 'GenericNMRBlood' for urine and blood datasets respecively, therefore, to import NMR spectra from urine samples the *sop* parameter would be::

	nmrData = nPYc.NMRDataset('path to data', sop='GenericNMRurine')
	
A full list of the parameters for each dataset type is given in the :doc:`configuration/builtinSOPs`. If different values are required, these can be modified directly in the appropriate *SOP* file, or alternatively they can be set by the user by modifying the required 'Attribute', either at import, or by subsequent direct modification in the pipeline. For example, to set the line width threshold (*LWFailThreshold*) to subsequently flag NMR spectra with line widths not meeting this value::
	
	# EITHER, set the required value (here 0.8) at import
	nmrData = nPYc.NMRDataset(rawDataPath, pulseProgram='noesygppr1d', LWFailThreshold=0.8)
	
	# OR, set the *Attribute* directly (after importing nmrData)
	nmrData.Attributes['LWFailThreshold'] = 0.8
	
Dataset classes have several other key atttributes, including:

* :py:attr:`~Dataset.sampleMetadata`: A :math:`n` × :math:`p` pandas dataframe of sample identifiers and sample associated metadata (each row here corresponds to a row in the intensityData file)
* :py:attr:`~Dataset.featureMetadata`: A :math:`m` × :math:`q`  pandas dataframe of feature identifiers and feature associated metadata (each row here corresponds to a column in the intensityData file)
* :py:attr:`~Dataset.intensityData`: A :math:`n` × :math:`m` numpy matrix of measurements, where each row and column respectively correspond to a the measured intensity of a specific sample feature
* :py:attr:`~Dataset.sampleMask`: A :math:`n` numpy boolean vector where `True` and `False` flag samples for inclusion or exclusion respectively
* :py:attr:`~Dataset.featureMask`: A :math:`m` numpy boolean vector where `True` and `False` flag features for inclusion or exclusion respectively

.. figure:: _static/Dataset_structure.svg
	:alt: Structure of the key attributes of a dataset
	
	Structure of the key attributes of a :py:class:`~nPYc.objects.Dataset` object. Of note, rows in the :py:attr:`~nPYc.objects.Dataset.featureMetadata` Dataframe correspond to columns in the :py:attr:`~nPYc.objects.Dataset.intensityData` matrix.
	
Once created, you can query the number of features or samples it contains::

	dataset.noFeatures
	dataset.noSamples

Or directly inspect the sample or feature metadata, and the raw measurements::

	dataset.sampleMetadata
	dataset.featureMetadata
	dataset.intensityData


It is possible to add additional study design parameters or sample metadata into the Dataset using the :py:meth:`~nPYc.objects.Dataset.addSampleInfo` method (see :doc:`samplemetadata` for details). 

For full method specific details see :doc:`tutorial`.


Sample and Feature Masks
========================

Dataset classes also contains two internal `mask` vectors, the :py:attr:`~nPYc.objects.Dataset.sampleMask` and the :py:attr:`~nPYc.objects.Dataset.featureMask`. They store whether a sample or feature, respectively, should be used when calculating QC metrics, in the visualizations in the report functions and when exporting the dataset.

There are several functions which modify these internal masks:

- :py:meth:`~nPYc.objects.Dataset.updateMasks` is a method to automatically mask certain specific sample types, or enforce the quality control checks
- :py:meth:`~nPYc.objects.Dataset.excludeSamples` and :py:meth:`~nPYc.objects.Dataset.excludeFeatures` are methods to directly directly exclude specific samples or features respectively. Masked samples and features will remain in the dataset, but will be hidden, and thus ignored when calling the reporting functions, fitting PCA models, and exporting the pre-processed datasets.
- :py:meth:`~nPYc.objects.Dataset.initialiseMasks` resets the masks to include all samples/features.
- :py:meth:`~nPYc.objects.Dataset.applyMasks` completely excludes from the dataset all samples and features which have been previously masked. After calling this command the excluded features are and the masks are re-initialized so that all reaming samples and features are unmasked. This method should be used only when it is absolutely certain that the masked features and samples are to be removed, as the excluded data will have to be re-imported.

For examples of how these masks are used during the import and preprocessing of specific datasets see :doc:`tutorial`.

In brief, however, the following describes an example interatively utilising the above functions to generate a feature filtered (see :doc:`featurefiltering`) LC-MS dataset containing only :term:`Study Sample` samples, with a specific sample excluded::

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


Dataset Specific Syntax and Parameters
======================================

Note, the Dataset object serves as a common parent to :py:class:`~nPYc.objects.MSDataset`, :py:class:`~nPYc.objects.TargetedDataset`, and :py:class:`~nPYc.objects.NMRDataset`, and should not typically be instantiated independently.

.. autoclass:: nPYc.objects.Dataset
  :members:

.. autoclass:: nPYc.objects.MSDataset
  :members:

.. autoclass:: nPYc.objects.NMRDataset
  :members:

.. autoclass:: nPYc.objects.TargetedDataset
  :members:
	
