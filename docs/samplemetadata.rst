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
- 'AssayRole': :term:`assay role<assay role>` as described in :doc:`studydesign`
- 'SampleType': :term:`sample type<sample type>` as described in :doc:`studydesign`
- 'Dilution': Relative dilution factor for each sample
- 'Include Sample': where 'Include Sample' is ``False``, the :py:attr:`~nPYc.objects.Dataset.sampleMask` for that sample will be set to ``False`` and the corresponding sample marked for exclusion from the dataset (see :doc:`masks` for details)

.. table:: Minimal structure of a basic csv file
   :widths: auto
   
   =========== ============================== =================== ================== ======== ==============
   Sample ID   Sample File Name               AssayRole           SampleType         Dilution Include Sample
   =========== ============================== =================== ================== ======== ==============
   Dilution 1  UnitTest1_LPOS_ToF02_B1SRD01   Linearity Reference Study Pool         1        TRUE
   Dilution 2  UnitTest1_LPOS_ToF02_B1SRD02   Linearity Reference Study Pool         50       TRUE
   Sample 1    UnitTest1_LPOS_ToF02_S1W07     Assay               Study Sample       100      TRUE
   Sample 2    UnitTest1_LPOS_ToF02_S1W08     Assay               Study Sample       100      TRUE
   LTR         UnitTest1_LPOS_ToF02_S1W11_LTR Precision Reference External Reference 100      TRUE
   SR          UnitTest1_LPOS_ToF02_S1W12_SR  Precision Reference Study Pool         100      TRUE
   Sample 3    UnitTest1_LPOS_ToF02_S1W09_x   Assay               Study Sample       100      FALSE
   Blank 1     UnitTest1_LPOS_ToF02_Blank01   Assay               Procedural Blank   0        TRUE
   =========== ============================== =================== ================== ======== ==============

Any additional columns in the basic csv file will be appended to the :py:attr:`~nPYc.objects.Dataset.sampleMetadata` table as additional sample metadata.

**Important Note for LC-MS Datasets**

For full nPYc-Toolbox functionality for LC-MS data, there are three additional columns which should be specified, these are:

- 'Acquired Time': time of sample acquisition (date/time format)
- 'Run Order': the order in which the samples were acquired (an integer value from 1 to the number of samples in 'Acquisition Time' order)
- 'Correction Batch': the :term:`Analytical Batch` in which each sample was acquired (integer value)

'Acquired Time' would routinely be extracted from the raw data (see *Analytical Parameter Extraction* below) and 'Run Order' subsequently automatically inferred from this, however, in cases where this is not possible they can be added manually as columns to the 'Basic CSV' file. 

Similarly, 'Correction Batch' can be defined in the 'Basic CSV' file, or, if all samples were acquired in the same batch, a column can be added to the sampleMetadata attribute of the LC-MS dataset object after importing into the pipeline by running::

	msData.sampleMetadata['Correction Batch'] = 1
	
While inclusion of 'Run Order' and 'Correction Batch' is critical for functionality (namely :doc:`batchAndROCorrection` and :doc:`multivariate`), inclusion of 'Acquired Time' does not affect functionality but does enable key plots relying on this data to be generated. 

For a full example csv file with these columns included see 'DEVSET U RPOS Basic CSV.csv' (:doc:`tutorial`), but in brief, if adding manually into the 'Basic CSV' file the structure of the extra columns might look something like this:

.. table:: Additional columns required for full funtionality with LC-MS datasets (note these can be added)
   :widths: auto
   
   =========== ============================== =================== ================== ======== ============== ==================== ========= ================
   Sample ID   Sample File Name               AssayRole           SampleType         Dilution Include Sample Acquired Time        Run Order Correction Batch
   =========== ============================== =================== ================== ======== ============== ==================== ========= ================
   Dilution 1  UnitTest1_LPOS_ToF02_B1SRD01   Linearity Reference Study Pool         1        TRUE           18/01/2018  02:25:00 1         1
   Dilution 2  UnitTest1_LPOS_ToF02_B1SRD02   Linearity Reference Study Pool         50       TRUE           18/01/2018  02:40:00 2         1
   Sample 1    UnitTest1_LPOS_ToF02_S1W07     Assay               Study Sample       100      TRUE           18/01/2018  02:55:00 3         1
   Sample 2    UnitTest1_LPOS_ToF02_S1W08     Assay               Study Sample       100      TRUE           18/01/2018  03:10:00 4         1
   LTR         UnitTest1_LPOS_ToF02_S1W11_LTR Precision Reference External Reference 100      TRUE           18/01/2018  03:25:00 5         1
   SR          UnitTest1_LPOS_ToF02_S1W12_SR  Precision Reference Study Pool         100      TRUE           18/01/2018  03:40:00 6         1
   Sample 3    UnitTest1_LPOS_ToF02_S1W09_x   Assay               Study Sample       100      FALSE          18/01/2018  03:56:00 7         1
   Blank 1     UnitTest1_LPOS_ToF02_Blank01   Assay               Procedural Blank   0        TRUE           18/01/2018  04:11:00 8         1
   =========== ============================== =================== ================== ======== ============== ==================== ========= ================


Analytical Parameter Extraction
===============================

With the nPYc-Toolbox it is also possible to extract parameters directly from raw data files (currently for Bruker and Waters .RAW data only) and match to the imported dataset using::

	dataset.addSampleInfo(descriptionFormat='Raw Data', filePath='path to raw data')
	
This links to the underlying :py:meth:`~nPYc.utilities.extractParams` method.

.. automodule:: nPYc.utilities.extractParams
   :members:
   :exclude-members: buildFileList, extractWatersRAWParams, extractBrukerparams