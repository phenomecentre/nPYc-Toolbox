Sample Metadata
---------------

Using the nPYc-Toolbox, additional study design parameters or sample metadata may be mapped into the Dataset using the :py:meth:`~nPYc.objects.Dataset.addSampleInfo` method. This additional sample information may be added from a number of different sources, for example, from an associated CSV file, or from the raw data (see below), 'addSampleInfo' extracts the appropriate information, matches it to the acquired samples, and adds it into the sampleMetadata attribute of the dataset (a pandas dataframe of sample identifiers and sample associated metadata (see :doc:`objects` for details).


CSV Template for Metadata Import
================================

The ‘Basic CSV’ format specifies a simple method for matching analytical data to metadata using::

	dataset.addSampleInfo(descriptionFormat='Basic CSV', filePath='path to basicCSV.csv')

Although optional, it is recommend to generate such a CSV file containing basic metadata about each of the imported spectra. 

The nPYc-Toolbox options contains a default syntax for adding sample metadata in a predefined CSV format.

In brief, this CSV file format expects information to be provided for 6 pre-defined column names, ‘Sample File Name’, ‘Sample ID’, ‘SampleType’, ‘AssayRole’, ‘Dilution’, ‘Include Sample’. Any extra metadata (such as patient characteristics or clinical metadata) can be placed in this file, as long as the column names are not in the list of expected fields.

- 'Sample ID': Unique identifier for each sample
- 'Sample File Name': the 'Basic CSV' file matches based on the entries in the 'Sample File Name' column to the 'Sample File Name' in the :py:attr:`~nPYc.objects.Dataset.sampleMetadata` table
- 'AssayRole': :term:`assay role<assay role>` as described above
- 'SampleType': :term:`sample type<sample type>` as described above
- 'Dilution': Relative dilution factor for each sample
- 'Include Sample': where 'Include Sample' is ``False``, the :py:attr:`~nPYc.objects.Dataset.sampleMask` for that sample will be set to ``False`` and the corresponding sample marked for exclusion from the dataset

.. table:: Minimal structure of a basic csv file
   :widths: auto
   
   =========== ============================== =================== ================== ======== ==============
   Sample ID   Sample File Name               AssayRole           SampleType         Dilution Include Sample
   =========== ============================== =================== ================== ======== ==============
   Dilution 1  UnitTest1_LPOS_ToF02_B1SRD01   Linearity Reference Study Pool         1        TRUE
   Dilution 2  UnitTest1_LPOS_ToF02_B1SRD02   Linearity Reference Study Pool         50       TRUE
   Sample 1    UnitTest1_LPOS_ToF02_S1W07     Assay               Study Sample       100      TRUE
   Sample 2    UnitTest1_LPOS_ToF02_S1W08_x   Assay               Study Sample       100      TRUE
   LTR         UnitTest1_LPOS_ToF02_S1W11_LTR Precision Reference External Reference 100      TRUE
   SR          UnitTest1_LPOS_ToF02_S1W12_SR  Precision Reference Study Pool         100      TRUE
   Sample 3    UnitTest1_LPOS_ToF02_S1W09_x   Assay               Study Sample       100      FALSE
   Blank 1     UnitTest1_LPOS_ToF02_Blank01   Assay               Procedural Blank   0        TRUE
   =========== ============================== =================== ================== ======== ==============

Any additional columns in the basic csv file will be appended to the :py:attr:`~nPYc.objects.Dataset.sampleMetadata` table as additional sample metadata.


Analytical Parameter Extraction
===============================

With the nPYc-Toolbox it is also possible to extract parameters directly from raw data files (currently for Bruker and Waters .RAW data only) using::

	dataset.addSampleInfo(descriptionFormat='Raw Data', filePath='path to raw data')
	
This links to the underlying :py:meth:`~nPYc.utilities.extractParams` method.

.. automodule:: nPYc.utilities.extractParams
   :members:
   :exclude-members: buildFileList, extractWatersRAWParams, extractBrukerparams