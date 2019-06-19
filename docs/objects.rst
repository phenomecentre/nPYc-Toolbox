Datasets
--------

The nPYc-Toolbox is built around creating an object for each imported dataset. This object contains the metabolic profiling data itself, alongside all associated sample and feature metadata; various methods for generating, reporting and plotting important quality control parameters; and methods for pre-processing such as filtering poor quality features or correcting trends in batch and run-order.

The first step in creating an nPYc-Toolbox object is to import the acquired data, creating a :py:class:`~nPYc.objects.Dataset` specific for the data type:
 
* :py:class:`~nPYc.objects.MSDataset` for LC-MS profiling data
* :py:class:`~nPYc.objects.NMRDataset` for NMR profiling data
* :py:class:`~nPYc.objects.TargetedDataset` for targeted datasets

For example, to import LC-MS data into a MSDataset object::

	msData = nPYc.MSDataset('path to data')

Depending on the data type, the Dataset can be set up directly from the raw data, from common interchange formats, or from the outputs of popular data-processing tools. The supported data types are described in more detail in the data specific sections below.

When importing the data, default parameters, for example, specific parameters such as the number of points to interpolate NMR data into, or more generally the format to save figures as, are loaded from the :doc:`Configuration Files<configuration/configuration>`. These parameters are subsequently saved in the :py:attr:`~nPYc.objects.Dataset.Attributes` dictionary and used throughout subsequent implementation of the pipeline.

For example, for NMR data, the nPYc-Toolbox contains two default configuration files, 'GenericNMRUrine' and 'GenericNMRBlood' for urine and blood datasets respectively, therefore, to import NMR spectra from urine samples the *sop* parameter would be::

	nmrData = nPYc.NMRDataset('path to data', sop='GenericNMRurine')
	
A full list of the parameters for each dataset type is given in the :doc:`configuration/builtinSOPs`. If different values are required, these can be modified directly in the appropriate *SOP* file, or alternatively they can be set by the user by modifying the required 'Attribute', either at import, or by subsequent direct modification in the pipeline. For example, to set the line width threshold (*LWFailThreshold*) to subsequently flag NMR spectra with line widths not meeting this value::
	
	# EITHER, set the required value (here 0.8) at import
	nmrData = nPYc.NMRDataset(rawDataPath, pulseProgram='noesygppr1d', LWFailThreshold=0.8)
	
	# OR, set the *Attribute* directly (after importing nmrData)
	nmrData.Attributes['LWFailThreshold'] = 0.8
	
Dataset objects have several key attributes, including:

* :py:attr:`~Dataset.sampleMetadata`: A :math:`n` × :math:`p` pandas dataframe of sample identifiers and sample associated metadata (each row here corresponds to a row in the intensityData file)
* :py:attr:`~Dataset.featureMetadata`: A :math:`m` × :math:`q`  pandas dataframe of feature identifiers and feature associated metadata (each row here corresponds to a column in the intensityData file)
* :py:attr:`~Dataset.intensityData`: A :math:`n` × :math:`m` numpy matrix of measurements, where each row and column respectively correspond to a the measured intensity of a specific sample feature
* :py:attr:`~Dataset.sampleMask`: A :math:`n` numpy boolean vector where `True` and `False` flag samples for inclusion or exclusion respectively
* :py:attr:`~Dataset.featureMask`: A :math:`m` numpy boolean vector where `True` and `False` flag features for inclusion or exclusion respectively

.. figure:: _static/Dataset_structure.svg
	:alt: Structure of the key attributes of a dataset
	
	Structure of the key attributes of a :py:class:`~nPYc.objects.Dataset` object. Of note, rows in the :py:attr:`~nPYc.objects.Dataset.featureMetadata` Dataframe correspond to columns in the :py:attr:`~nPYc.objects.Dataset.intensityData` matrix.
	
Once created, you can query the number of features or samples it contains by running::

	dataset.noFeatures
	dataset.noSamples

Or directly inspect the sample or feature metadata, and the raw measurements::

	dataset.sampleMetadata
	dataset.featureMetadata
	dataset.intensityData
	
For more details on using the sample and feature masks see :doc:`masks`.

It is possible to add additional study design parameters or sample metadata into the Dataset using the :py:meth:`~nPYc.objects.Dataset.addSampleInfo` method (see :doc:`samplemetadata` for details). 

For full method specific details see :doc:`tutorial`.


LC-MS Datasets
==============

The toolbox is designed to be agnostic to the source of peak-picked profiling datasets, currently supporting the outputs of `XCMS <https://bioconductor.org/packages/release/bioc/html/xcms.html>`_ (Tautenhahn *et al* [#]_), `Bruker Metaboscape <https://www.bruker.com/products/mass-spectrometry-and-separations/ms-software/metaboscape/overview.html>`_, and `Progenesis QI <http://www.nonlinear.com/progenesis/qi/>`_, but simply expandable to data from other peak-pickers. Current best-practices in quality control of profiling LC-MS (Want *et al* [#]_, Dunn *et al* [#]_, Lewis *et al* [#]_) data are applied, including utilising repeated injections of :term:`Study Reference` samples in order to calculate analytical precision for the measurement of each feature (:term:`Relative Standard Deviation`), and a serial dilution of the reference sample to asses the linearity of response (:term:`Correlation to Dilution`), for full details see :doc:`Feature Summary Report: LC-MS Datasets<reports>`. 

Study Reference samples are also used (in conjunction with :term:`Long-Term Reference` samples if available) to assess and correct trends in batch and run-order (:doc:`batchAndROCorrection`). Additionally, both RSD and correlation to dilution are used to filter features to retain only those measured with a high precision and accuracy (:doc:`masks`).


NMR Datasets
============

The nPYc-Toolbox supports input of processed Bruker GmbH format 1D experiments. Upon import, each spectrum's chemical shift axis is calibrated to a reference peak (Pearce *et al* [#]_), and all spectra interpolated onto a common scale, with full parameters as per the :doc:`NMRDataset Objects configuration SOPs<configuration/builtinSOPs>`. The toolbox supports automated calculation of the quality control metrics described previously (Dona *et al* [#]_), including assessments of line-width, water suppression quality, and baseline stability, for full details see :doc:`Feature Summary Report: NMR Datasets<reports>`.


Targeted Datasets
=================

The TargetedDataset represents quantitative datasets where compounds are already identified, the exactitude of the quantification can be established, units are known and calibration curve or internal standards are employed (Lee *et al* [#]_). It implements a set of reports and data consistency checks to assist analysts in assessing the presence of batch effects, applying limits of quantification (LOQ), standardizing the linearity range over multiple batches, and determining and visualising the accuracy and precision of each measurement, for more details see :doc:`Feature Summary Report: NMR Targeted Datasets<reports>`.

The nPYc-Toolbox supports input of both MS-derived targeted datasets (tutorial and further documentation in progress), and two Bruker proprietary human biofluid quantification platforms (IVDr algorithms) that generate targeted outputs from the NMR profiling data, `BI-LISA <https://www.bruker.com/products/mr/nmr-preclinical-screening/lipoprotein-subclass-analysis.html>`_ for quantification of Lipoproteins (blood samples only) and `BIQUANT-PS <https://www.bruker.com/products/mr/nmr-preclinical-screening/biquant-ps.html>`_  and `BIQUANT-UR <https://www.bruker.com/products/mr/nmr-preclinical-screening/biquant-ur.html>`_ for small molecule metabolites (for blood and urine respectively).


Dataset Specific Syntax and Parameters
======================================

The main function parameters (which may be of interest to advanced users) are as follows:

Note, the Dataset object serves as a common parent to :py:class:`~nPYc.objects.MSDataset`, :py:class:`~nPYc.objects.TargetedDataset`, and :py:class:`~nPYc.objects.NMRDataset`, and should not typically be instantiated independently.

.. autoclass:: nPYc.objects.Dataset
  :members:

.. autoclass:: nPYc.objects.MSDataset
  :members:

.. autoclass:: nPYc.objects.NMRDataset
  :members:

.. autoclass:: nPYc.objects.TargetedDataset
  :members:
	
.. [#] Ralf Tautenhahn, Christoph Bottcher and Steffen Neumann. Highly sensitive feature detection for high resolution LC/MS. BMC Bioinformatics, 9:504, 2008. URL: https://doi.org/10.1186/1471-2105-9-504

.. [#] Elizabeth J Want, Ian D Wilson, Helen Gika, Georgios Theodoridis, Robert S Plumb, John Shockcor, Elaine Holmes and Jeremy K Nicholson. Global metabolic profiling procedures for urine using UPLC-MS. Nature Protocols, 5(6):1005-18, 2010. URL: http://dx.doi.org/10.1038/nprot.2010.50

.. [#] Warwick B Dunn, David Broadhurst, Paul Begley, Eva Zelena, Sue Francis-McIntyre, Nadine Anderson, Marie Brown, Joshau D Knowles, Antony Halsall, John N Haselden, Andrew W Nicholls, Ian D Wilson, Douglas B Kell, Royston Goodacre and The Human Serum Metabolome (HUSERMET) Consortium. Procedures for large-scale metabolic profiling of serum and plasma using gas chromatography and liquid chromatography coupled to mass spectrometry. Nature Protocols, 6(7):1060-83, 2011. URL: http://dx.doi.org/10.1038/nprot.2011.335

.. [#] Matthew R Lewis, Jake TM Pearce, Konstantina Spagou, Martin Green, Anthony C Dona, Ada HY Yuen, Mark David, David J Berry, Katie Chappell, Verena Horneffer-van der Sluis, Rachel Shaw, Simon Lovestone, Paul Elliott, John Shockcor, John C Lindon, Olivier Cloarec, Zoltan Takats, Elaine Holmes and Jeremy K Nicholson. Development and Application of Ultra-Performance Liquid Chromatography-TOF MS for Precision Large Scale Urinary Metabolic Phenotyping. Analytical Chemistry, 88(18):9004-9013, 2016. URL: http://dx.doi.org/10.1021/acs.analchem.6b01481

.. [#] Jake TM Pearce, Toby J Athersuch, Timothy MD Ebbels, John C Lindon, Jeremy K Nicholson and Hector C Keun. Robust Algorithms for Automated Chemical Shift Calibration of 1D 1H NMR Spectra of Blood Serum. Analytical Chemistry, 80(18):7158-62, 2008. URL: http://dx.doi.org/10.1021/ac8011494

.. [#] Anthony C Dona, Beatriz Jiménez, Hartmut Schäfer, Eberhard Humpfer, Manfred Spraul, Matthew R Lewis, Jake TM Pearce, Elaine Holmes, John C Lindon and Jeremy K Nicholson. Precision High-Throughput Proton NMR Spectroscopy of Human Urine, Serum, and Plasma for Large-Scale Metabolic Phenotyping. Analytical Chemistry, 86(19):9887-9894, 2014. URL: http://dx.doi.org/10.1021/ac5025039

.. [#] Jean W Lee, Viswanath Devanarayan, Yu Chen Barrett, Russell Weiner, John Allinson, Scott Fountain, Stephen Keller, Ira Weinryb, Marie Green, Larry Duan, James A Rogers, Robert Millham, Peter J O'Brien, Jeff Sailstad, Masood Khan, Chad Ray and John A Wagner. Fit-for-purpose method development and validation for successful biomarker measurement. Pharmaceutical Research, 23(2):312-28, 2006. URL: http://dx.doi.org/10.1007/s11095-005-9045-3
