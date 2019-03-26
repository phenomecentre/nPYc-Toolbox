Using the nPYc toolbox
----------------------

The nPYc-Toolbox is a general Python 3 implementation of the MRC-NIHR National Phenome Centre toolchain for the import, quality-control, and preprocessing of metabolic profiling datasets.


Getting started
===============

Ensure that the Anaconda with Python 3.5 or above, and the nPYc-Toolbox are installed. The nPYc-Toolbox can be installed by using the Anaconda Prompt and typing ‘pip install nPYc’. This will install the toolbox alongside any required dependency, and make it available as a general python package.

The folder NMR contains all the raw NMR data for this tutorial. The tutorial dataset can also be downloaded from the MetaboLights database (www.ebi.ac.uk/metabolights), study ID: MTBLS694.

After installing all the necessary requirements (Anaconda with Python 3.x), follow these instructions to start a Jupyter Notebook session:

On Windows:
Option 1: Graphical User Interface: UI
•	Open the Anaconda explorer software and change the current directory to the directory containing the tutorials code and dataset. Launch a Jupyter Notebook session.
 
Option 2: Command line	
•	Alternatively, open the Anaconda prompt console, navigate to the folder where you want to initialize the notebook.
•	Then type “jupyter notebook” on the console. This will open the Jupyter Notebook session in the browser.

In Linux/Mac:
•	Open a terminal/console and navigate to the folder containing the tutorial code and dataset and type ‘jupyter notebook’ in the console.


Preprocessing and quality control of NMR data with the nPYc-Toolbox
===================================================================

link to this

Preprocessing and quality control of LC-MS data with the nPYc-Toolbox
=====================================================================

link to this

Preprocessing and quality control of NMR targeted data with the nPYc-Toolbox
============================================================================

link to this


Importing datasets *&* mapping metadata
=======================================



Datasets in the nPYc toolbox are represented by instances of sub-classes of the :py:class:`~nPYc.objects.Dataset` class. Each instance of a :py:class:`~nPYc.objects.Dataset` represents the measured abundances of features, plus sample and feature metadata, in a metabolic profiling experiment.

Depending on the data type, objects may be initialised from a single document containing several samples, or by integrating a directory of raw data files::

	msDataset = nPYc.MSDataset('~/path to file.csv', fileType='xcms')

Or::

	nmrDataset = nPYc.NMRDataset('~/path to Topspin experiment directory/', fileType='Bruker')

Once created, you can query the number of features or samples it contains::

	dataset.noFeatures
	dataset.noSamples

Or directly inspect the sample or feature metadata, and the raw measurements::

	dataset.sampleMetadata
	dataset.featureMetadata
	
	dataset.intensityData

Additional study design parameters or sample metadata may be mapped into the Dataset using the :py:meth:`~nPYc.objects.Dataset.addSampleInfo` method. For the purpose of standardising QC filtering procedures, the nPYc toolbox defines a small set of terms for describing reference sample types and design elements, as listed in :doc:`nomenclature<../nomenclature>`.


Assessing Analytical Quality
============================

The nPYc toolbox incorporates the concept of analytical quality directly into the subclasses of :py:class:`~nPYc.objects.Dataset`. Depending on the analytical platform and protocol, quality metrics may be judged on the basis of sample-by-sample or feature-by-feature comparisons, or both.

To generate reports of analytical quality, call the :py:func:`~nPYc.reports.generateReport` function, with the dataset object as an argument::

	nPYc.reports.generateReport(datasetObject, 'feature summary')

Parameters for the quality control procedure can be specified in a :doc:`SOP JSON<../configuration/configurationSOPs>` file as the Dataset object is created, and amended after creation by modifying the relevant entry of the :py:attr:`~nPYc.objects.Dataset.Attributes` dictionary.

Each object type supports its own QC tests, see :doc:`nmrdataset`, :doc:`msdataset`, and :doc:`targeteddataset` for specific details.

Filtering of samples *&* variables
**********************************

Filtering of features by the generic procedures defined for each type of dataset, using the thresholds loaded from the :doc:`SOP <../configuration/configurationSOPs>` and defined in :py:attr:`~nPYc.objects.Dataset.Attributes` is accomplished with the :py:meth:`~nPYc.objects.Dataset.updateMasks` method. When called, the elements in the :py:attr:`~nPYc.objects.Dataset.featureMask` are set to ``False`` where the feature does not meet quality criteria, and elements in :py:attr:`~nPYc.objects.Dataset.sampleMask` are set to ``False`` for samples that do not pass quality criteria, or sample types and roles not specified.

The defaults arguments to :py:meth:`~nPYc.objects.Dataset.updateMasks` will filter the dataset to contain only study and study reference samples and only those features meeting quality criteria::

	dataset.updateMasks(filterSamples=True, filterFeatures=True, sampleTypes=[SampleType.StudySample, SampleType.StudyPool], assayRoles=[AssayRole.Assay, AssayRole.PrecisionReference])

Specific samples or features may be excluded based on their ID or other associated metadata with the :py:meth:`~nPYc.objects.Dataset.excludeFeatures` and :py:meth:`~nPYc.objects.Dataset.excludeSamples` methods.

These methods operate by setting the relevant entries in the :py:attr:`~nPYc.objects.Dataset.featureMask` and :py:attr:`~nPYc.objects.Dataset.sampleMask` vectors to ``False``, which has the effect of hiding the sample or feature from further analysis. Elements masked from the dataset may then be permanently removed by calling the :py:meth:`~nPYc.objects.Dataset.applyMasks` method.


Normalisation
=============

Dilution effects on global sample intensity can be normalised by attaching one of the classes in the :py:mod:`~nPYc.utilities.normalisation` sub-module to the :py:attr:`~nPYc.objects.Dataset.Normalisation` attribute of a :py:class:`~nPYc.objects.Dataset`. 

By default new :py:class:`~nPYc.objects.Dataset` objects have a :py:class:`~nPYc.utilities.normalisation.NullNormaliser` attached, which carries out no normalisation. By assigning an instance of a :py:class:`~nPYc.utilities.normalisation._normaliserABC.Normaliser` class to this attribute::

	totalAreaNormaliser = nPYc.utilities.normalisation.TotalAreaNormaliser
	dataset.Normalisation = totalAreaNormaliser

will cause all calls to :py:attr:`~nPYc.objects.Dataset.intensityData` to return values transformed by the normaliser.


Basic Multivariate Visualisation
================================

Simple PCA models of a :py:class:`~nPYc.objects.Dataset` can be visualised by the :py:func:`~nPYc.reports.multivariateQCreport` function. This report will take a pyChemomentrics PCA model of the dataset, and visualise the scores and loadings of the model, optionally highlighting the scores by the supplied sample metadata.

Scores and loadings of the models generated by :py:func:`~nPYc.reports.multivariateQCreport` can be explored interactively with the :py:func:`~nPYc.plotting.plotScoresInteractive` and :py:func:`~nPYc.plotting.plotLoadingsInteractive` functions.


Exporting Datasets
==================

:py:class:`~nPYc.objects.Dataset` objects can be exported in a variety of formats for import into other analytical pacakages, additionally automated reports generated by the toolbox can be saved as html documents with embedded figures.

Saving reports
**************

Report generated interactively by the :py:mod:`~nPYc.reports` module can be saved as html documents with static images by supplying a path in which to save the report and figures to the *output=* parameter of the :py:func:`~nPYc.reports.generateReport` function.

Exporting data
**************

Datasets can be exported in a variety of formats with the :py:meth:`~nPYc.objects.Dataset.exportDataset` method. '*UnifiedCSV*' provides a good default output, exporting the :-:`~nPYc.objects.Dataset.sampleMetadata`, :py:attr:`~nPYc.objects.Dataset.featureMetadata`, and :py:attr:`~nPYc.objects.Dataset.intensityData` concatenated as a single coma-separated text file, with samples in rows, and features in columns. Where the number of features in a dataset might result in a file with too many columns to be opened by certain software packages, the '*CSV*' option allows the :py:attr:`~nPYc.objects.Dataset.sampleMetadata`, :py:attr:`~nPYc.objects.Dataset.featureMetadata`, and :py:attr:`~nPYc.objects.Dataset.intensityData` to each be saved to a separate CSV file. The nPYc toolbox currently also supports exporting metadata in ISATAB format.


.. toctree::
   :maxdepth: 2
   :hidden:

   SampleMetadata
   nmrdataset
   msdataset
   targeteddataset
