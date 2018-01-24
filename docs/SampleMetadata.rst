Basic CSV Sample Metadata Import
--------------------------------

A bare-bones CSV description file includes six columns:

================ ========
Column           Contents
================ ========
Sampling ID      :term:`Sampling ID`
Sample File Name :term:`Sample File Name`
AssayRole        :term:`Assay Role`, if the entry can be mapped to an :py:class:`~nPYc.enumerations.AssayRole` enum, this will be used, otherwise the entry is preserved as is.
SampleType       :term:`Sample Type`, if the entry can be mapped to an :py:class:`~nPYc.enumerations.SampleType` enum, this will be used, otherwise the entry is preserved as is.
Dilution         Expected dilution of a :term:`Linearity Reference` sample
Include Sample   Either ``TRUE``, or ``FALSE`` for samples that should be masked from analysis. 
================ ========

Rows in the CSV description are mapped to rows in the :py:attr:`~nPYc.objects.Dataset.sampleMetadata` table by matching on the 'Sample File Name' column. Where columns are present in both the csv and sampleMetadata table, the column in sampleMetadata is overwritten by that in csv.

Additional columns may be included to define further sample parameters, including both columns defined for the :py:attr:`~nPYc.objects.Dataset.sampleMetadata` table, plus arbitrary extra columns.

.. table:: Example CSV study description with one additional column ('Class')
   :widths: auto

   ============= ============================== ================== ================= ======== ============== =======
   Sampling ID   Sample File Name               AssayRole          SampleType        Dilution Include Sample Class
   ============= ============================== ================== ================= ======== ============== =======
   UT1_S4_s4     UnitTest1_LPOS_ToF02_S1W07     Assay              StudySample       100      TRUE           Case
   UT1_S4_s5     UnitTest1_LPOS_ToF02_S1W08_x   Assay              StudySample       100      TRUE           Control
   LTR           UnitTest1_LPOS_ToF02_S1W11_LTR PrecisionReference ExternalReference 100      TRUE
   SR            UnitTest1_LPOS_ToF02_S1W12_SR  PrecisionReference StudyPool         100      TRUE
   Failed Sample UnitTest1_LPOS_ToF02_ERROR                        New Type          100      FALSE
   Blank1        UnitTest1_LPOS_ToF02_Blank01   Assay              ProceduralBlank   0        TRUE
   Blank2        UnitTest1_LPOS_ToF02_Blank02   Assay              ProceduralBlank   0        TRUE
   ============= ============================== ================== ================= ======== ============== =======
