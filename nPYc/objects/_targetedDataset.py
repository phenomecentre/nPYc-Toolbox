"""
Module for the import and manipulation of quantified targeted MS data sets.
"""

import copy
import os
import re
from datetime import datetime
import numpy
import pandas
import collections
import warnings
from .._toolboxPath import toolboxPath
from ._dataset import Dataset
from ..utilities import normalisation, rsd
from ..enumerations import VariableType, AssayRole, SampleType, QuantificationType, CalibrationMethod, AnalyticalPlatform


class TargetedDataset(Dataset):
    """
    TargetedDataset(dataPath, fileType='TargetLynx', sop='Generic', \*\*kwargs)

    :py:class:`~TargetedDataset` extends :py:class:`Dataset` to represent quantitative datasets, where compounds are already identified, the exactitude of the quantification can be established, units are known and calibration curve or internal standards are employed.
    The :py:class:`~TargetedDataset` class include methods to apply limits of quantification (LLOQ and ULOQ), merge multiple analytical batch, and report accuracy and precision of each measurements.

    In addition to the structure of :py:class:`~Dataset`, :py:class:`~TargetedDataset` requires the following attributes:

    * :py:attr:`~TargetedDataset.expectedConcentration`:
        A :math:`n` × :math:`m` pandas dataframe of expected concentrations (matching the :py:attr:`~Dataset.intensityData` dimension), with column names matching :py:attr:`~TargetedDataset.featureMetadata[‘Feature Name’]`

    * :py:attr:`~TargetedDataset.calibration`:
        A dictionary containing pandas dataframe describing calibration samples:

        * :py:attr:`~TargetedDataset.calibration['calibIntensityData']`:
            A :math:`r` x :math:`m` numpy matrix of measurements. Features must match features in :py:attr:`~TargetedDataset.intensityData`

        * :py:attr:`~TargetedDataset.calibration['calibSampleMetadata']`:
            A :math:`r` x :math:`m` pandas dataframe of calibration sample identifiers and metadata

        * :py:attr:`~TargetedDataset.calibration['calibFeatureMetadata']`:
            A :math:`m` × :math:`q` pandas dataframe of feature identifiers and metadata

        * :py:attr:`~TargetedDataset.calibration['calibExpectedConcentration']`:
            A :math:`r` × :math:`m` pandas dataframe of calibration samples expected concentrations

    * :py:attr:`~TargetedDataset.Attributes` must contain the following (can be loaded from a method specific JSON on import):

        * ``methodName``:
            A (str) name of the method

        * ``externalID``:
            A list of external ID, each external ID must also be present in *Attributes* as a list of identifier (for that external ID) for each feature. For example, if ``externalID=['PubChem ID']``, ``Attributes['PubChem ID']=['ID1','ID2','','ID75']``

    * :py:attr:`~TargetedDataset.featureMetadata` expects the following columns:
        * ``quantificationType``:
            A :py:class:`~nPYc.enumerations.QuantificationType` enum specifying the exactitude of the quantification procedure employed.
        * ``calibrationMethod``:
            A :py:class:`~nPYc.enumerations.CalibrationMethod` enum specifying the calibration method employed.
        * ``Unit``:
            A (str) unit corresponding the the feature measurement value.
        * ``LLOQ``:
            The lowest limit of quantification, used to filter concentrations < LLOQ
        * ``ULOQ``:
            The upper limit of quantification, used to filter concentrations > ULOQ
        * externalID:
            All externalIDs listed in :py:attr:`~TargetedDataset.Attributes['externalID']` must be present as their own column


    Currently targeted assay results processed using **TargetLynx** or **Bruker quantification results** can be imported.
    To create an import for any other form of semi-quantitative or quantitative results, the procedure is as follow:

        * Create a new ``fileType == 'myMethod'`` entry in :py:meth:`~TargetedDataset.__init__`
        * Define functions to populate all expected dataframes (using file readers, JSON,...)
        * Separate calibration samples from study samples (store in :py:attr:`~TargetedDataset.calibration`). *If none exist, intialise empty dataframes with the correct number of columns and column names.*
        * Execute pre-processing steps if required (note: all feature values should be expressed in the unit listed in :py:attr:`~TargetedDataset.featureMetadata['Unit']`)
        * Apply limits of quantification using :py:meth:`~TargetedDataset._applyLimitsOfQuantification`. (This function does not apply limits of quantification to features marked as :py:class:`~nPYc.enumerations.QuantificationType` == QuantificationType.Monitored for compounds monitored for relative information.)

    The resulting :py:class:`~TargetedDatset` created must satisfy to the criteria for *BasicTargetedDataset*, which can be checked with :py:meth:`~TargetedDataset.validatedObject` (list the minimum requirements for all class methods).


    * ``fileType == 'TargetLynx'`` to import data processed using **TargetLynx**

        TargetLynx import operates on ``xml`` files exported *via* the 'File -> Export -> XML' TargetLynx menu option. Import requires a ``calibration_report.csv`` providing lower and upper limits of quantification (LLOQ, ULOQ) with the ``calibrationReportPath`` keyword argument.

        Targeted data measurements as well as calibration report information are read and mapped with pre-defined SOPs. All measurments are converted to pre-defined units and measurements inferior to the lowest limits of quantification or superior to the upper limits of quantification are replaced. Once the import is finished, only analysed samples are returned (no calibration samples) and only features mapped onto the pre-defined SOP and sufficiently described.

        Instructions to created new ``TargetLynx`` SOP can be found on the :doc:`generation of targeted SOPs <configuration/targetedSOPs>` page.

        Example: ``TargetedDataset(datapath, fileType='TargetLynx', sop='OxylipinMS', calibrationReportPath=calibrationReportPath, sampleTypeToProcess=['Study Sample','QC'], noiseFilled=False, onlyLLOQ=False, responseReference=None)``

        * ``sop``
            Currently implemented are `'OxylipinMS'` and `'AminoAcidMS'`

            `AminoAcidMS`: Gray N. `et al`. Human Plasma and Serum via Precolumn Derivatization with 6‑Aminoquinolyl‑N‑hydroxysuccinimidyl Carbamate: Application to Acetaminophen-Induced Liver Failure. `Analytical Chemistry`, 2017, 89, 2478−87.

            `OxylipinMS`: Wolfer AM. `et al.` Development and Validation of a High-Throughput Ultrahigh-Performance Liquid Chromatography-Mass Spectrometry Approach for Screening of Oxylipins and Their Precursors. `Analytical Chemistry`, 2015, 87 (23),11721–31

        * ``calibrationReportPath``
            Path to the calibration report `csv` following the provided report template.

            The following columns are required (leave an empty value to reject a compound):

                * Compound
                    The compound name, identical to the one employed in the SOP `json` file.

                * TargetLynx ID
                    The compound TargetLynx ID, identical to the one employed in the SOP `json` file.

                * LLOQ
                    Lowest limit of quantification concentration, in the same unit as indicated in TargetLynx.

                * ULOQ
                    Upper limit of quantification concentration, in the same unit as indicated in TargetLynx.

            The following columns are expected by :py:meth:`~TargetedDataset._targetLynxApplyLimitsOfQuantificationNoiseFilled`:

                * Noise (area)
                    Area integrated in a blank sample at the same retention time as the compound of interest (if left empty noise concentration calculation cannot take place).

                * a
                    :math:`a` coefficient in the calibration equation (if left empty noise concentration calculation cannot take place).

                * b
                    :math:`b` coefficient in the calibration equation (if left empty noise concentration calculation cannot take place).

            The following columns are recommended but not expected:

                * Cpd Info
                    Additional information relating to the compound (can be left empty).

                * r
                    :math:`r` goodness of fit measure for the calibration equation (can be left empty).

                * r2
                    :math:`r^2` goodness of fit measure for the calibration equation (can be left empty).

        
        * ``sampleTypeToProcess``
            List of *['Study Sample','Blank','QC','Other']* for the sample types to process as defined in MassLynx. Only samples in 'sampleTypeToProcess' are returned. Calibrants should not be processed and are not returned. Most uses should only require `'Study Sample'` as quality controls are identified based on sample names by subsequent functions. `Default value is '['Study Sample','QC']'`.
        
        * ``noiseFilled``
            If True values <LLOQ will be replaced by a concentration equivalent to the noise level in a blank. If False <LLOQ is replaced by :math:`-inf`. `Default value is 'False'`
        
        * ``onlyLLOQ``
            If True only correct <LLOQ, if False correct <LLOQ and >ULOQ. `Default value is 'False'`.
        
        * ``responseReference``
            If noiseFilled=True the noise concentration needs to be calculated. Provide the 'Sample File Name' of a reference sample to use in order to establish the response to use, or list of samples to use (one per feature). If None, the middle of the calibration will be employed. `Default value is 'None'`.

        * ``keepPeakInfo``
            If keepPeakInfo=True (default `False`) adds the :py:attr:`peakInfo` dictionary to the :py:class:`~TargetedDataset.calibration`. :py:attr:`peakInfo` contains the `peakResponse`, `peakArea`, `peakConcentrationDeviation`, `peakIntegrationFlag` and `peakRT`.

        * ``keepExcluded``
            If keepExcluded=True (default `False`), import exclusions (:py:attr:`excludedImportSampleMetadata`, :py:attr:`excludedImportFeatureMetadata`, :py:attr:`excludedImportIntensityData` and :py:attr:`excludedImportExpectedConcentration`) are kept in the object.

        * ``keepIS``
            If keepIS=True (default `False`), features marked as Internal Standards (IS) are retained.


    * ``fileType = 'Bruker Quantification'`` to import Bruker quantification results

        * ``nmrRawDataPath``
            Path to the parent folder where all result files are stored. All subfolders will be parsed and the ``.xml`` results files matching the ``fileNamePattern`` imported.

        * ``fileNamePattern``
            Regex to recognise the result data xml files

        * ``pdata``
            To select the right pdata folders (default 1)

        Two form of Bruker quantification results are supported and selected using the ``sop`` option: *BrukerQuant-UR* and *Bruker BI-LISA*

        * ``sop = 'BrukerQuant-UR'``

            Example: ``TargetedDataset(nmrRawDataPath, fileType='Bruker Quantification', sop='BrukerQuant-UR', fileNamePattern='.*?urine_quant_report_b\.xml$', unit='mmol/mol Crea')``

            * ``unit``
                If features are duplicated with different units, ``unit`` limits the import to features matching said unit. (In case of duplication and no ``unit``, all available units will be listed)

        * ``sop = ''BrukerBI-LISA'``
            Example: ``TargetedDataset(nmrRawDataPath, fileType='Bruker Quantification', sop='BrukerBI-LISA', fileNamePattern='.*?results\.xml$')``

    """

    def __init__(self, datapath, fileType='TargetLynx', sop='Generic', **kwargs):
        """
        Initialisation and pre-processing of input data (load files and match data and calibration and SOP, apply limits of quantification).
        """

        super().__init__(sop=sop, **kwargs)
        self.filePath, fileName = os.path.split(datapath)
        self.fileName, fileExtension = os.path.splitext(fileName)

        self.name = self.fileName

        # Load files and match data, calibration report and SOP, then Apply the limits of quantification
        if fileType == 'TargetLynx':
            # Read files, filter calibration samples, filter IS, applyLLOQ, clean object
            self._loadTargetLynxDataset(datapath, **kwargs)
            # Finalise object
            self.VariableType = VariableType.Discrete
            self.AnalyticalPlatform = AnalyticalPlatform.MS
            self.initialiseMasks()
        elif fileType == 'Bruker Quantification':
            # Read files, clean object
            self._loadBrukerXMLDataset(datapath, **kwargs)
            # Finalise object
            self.VariableType = VariableType.Discrete
            self.AnalyticalPlatform = AnalyticalPlatform.NMR
            self.initialiseMasks()
        elif fileType == 'empty':
            # Build empty object for testing
            pass
        else:
            raise NotImplementedError

        # Check the final object is valid and log
        if fileType != 'empty':
            validDataset = self.validateObject(verbose=False, raiseError=False, raiseWarning=False)
            if not validDataset['BasicTargetedDataset']:
                raise ValueError('Import Error: The imported dataset does not satisfy to the Basic TargetedDataset definition')
        self.Attributes['Log'].append([datetime.now(),
                                       '%s instance initiated, with %d samples, %d features, from %s'
                                       % (self.__class__.__name__, self.noSamples, self.noFeatures, datapath)])
        # Check later
        if 'Metadata Available' not in self.sampleMetadata:
            self.sampleMetadata['Metadata Available'] = False

    @property
    def rsdSP(self):
        """
        Returns percentage :term:`relative standard deviations<RSD>` for each feature in the dataset, calculated on samples with the Assay Role :py:attr:`~nPYc.enumerations.AssayRole.PrecisionReference` and Sample Type :py:attr:`~nPYc.enumerations.SampleType.StudyPool` in :py:attr:`~Dataset.sampleMetadata`.
        Implemented as a back-up to :py:Meth:`accuracyPrecision` when no expected concentrations are known

        :return: Vector of feature RSDs
        :rtype: numpy.ndarray
        """
        # Check we have Study Reference samples defined
        if not ('AssayRole' in self.sampleMetadata.keys() or 'SampleType' in self.sampleMetadata.keys()):
            raise ValueError('Assay Roles and Sample Types must be defined to calculate RSDs.')
        if not sum(self.sampleMetadata['AssayRole'].values == AssayRole.PrecisionReference) > 1:
            raise ValueError('More than one precision reference is required to calculate RSDs.')

        mask = numpy.logical_and(self.sampleMetadata['AssayRole'].values == AssayRole.PrecisionReference,
                                 self.sampleMetadata['SampleType'].values == SampleType.StudyPool)

        return rsd(self._intensityData[mask & self.sampleMask])

    @property
    def rsdSS(self):
        """
        Returns percentage :term:`relative standard deviations<RSD>` for each feature in the dataset, calculated on samples with the Assay Role :py:attr:`~nPYc.enumerations.AssayRole.Assay` and Sample Type :py:attr:`~nPYc.enumerations.SampleType.StudySample` in :py:attr:`~Dataset.sampleMetadata`.

        :return: Vector of feature RSDs
        :rtype: numpy.ndarray
        """
        # Check we have Study Reference samples defined
        if not ('AssayRole' in self.sampleMetadata.keys() or 'SampleType' in self.sampleMetadata.keys()):
            raise ValueError('Assay Roles and Sample Types must be defined to calculate RSDs.')
        if not sum(self.sampleMetadata['AssayRole'].values == AssayRole.Assay) > 1:
            raise ValueError('More than one assay sample is required to calculate RSDs.')

        mask = numpy.logical_and(self.sampleMetadata['AssayRole'].values == AssayRole.Assay,
                                 self.sampleMetadata['SampleType'].values == SampleType.StudySample)

        return rsd(self._intensityData[mask & self.sampleMask])

    def _loadTargetLynxDataset(self, datapath, calibrationReportPath, keepIS=False, noiseFilled=False, keepPeakInfo=False, keepExcluded=False, **kwargs):
        """
        Initialise object from peak-picked and calibrated TargetLynx data. Filter calibration samples, filter IS.

        Targeted data measurements as well as calibration report information are read and mapped with pre-defined SOPs. All units are converted to pre-defined units and measurements inferior to the lowest limits of quantification or superior to the upper limits of quantification are replaced. Once the import is finished, only analysed samples are returned (no calibration samples) and only features mapped onto the pre-defined SOP and sufficiently described.

        * TargetLynx
            TargetLynx import operates on xml files exported *via* the 'File -> Export -> XML' menu option. Import requires a calibration_report.csv providing lower and upper limits of quantification (LLOQ, ULOQ) with the ``calibrationReportPath`` keyword argument.

            Example: ``TargetedDataset(datapath, fileType='TargetLynx', sop='OxylipinMS', calibrationReportPath=calibrationReportPath, sampleTypeToProcess=['Study Sample','QC'], noiseFilled=False, onlyLLOQ=False, responseReference=None)``

            * ``datapath``
                Path to the TargetLynx exported `xml` file.

            * ``calibrationReportPath``
                Path to the calibration report `csv` following the provided report template (leave an empty value in the predefined columns to reject a compound).

            * ``sampleTypeToProcess``
                List of ['Study Sample','Blank','QC','Other'] for the sample types to process as defined in MassLynx. Only samples in 'sampleTypeToProcess' are returned. Calibrants should not be processed and are not returned. Most uses should only require `'Study Sample'` as quality controls are identified based on sample names by subsequent functions. `Default value is '['Study Sample','QC']'`.

            * ``noiseFilled``
                If True values <LLOQ will be replaced by a concentration equivalent to the noise level in a blank. If False <LLOQ is replaced by :math:`-inf`. `Default value is 'False'`

            * ``onlyLLOQ``
                If True only correct <LLOQ, if False correct <LLOQ and >ULOQ. `Default value is 'False'`.

            * ``responseReference``
                If noiseFilled=True the noise concentration needs to be calculated. Provide the 'Sample File Name' of a reference sample to use in order to establish the response to use, or list of samples to use (one per feature). If None, the middle of the calibration will be employed. `Default value is 'None'`.

            * ``keepIS
                If keepIS=True (default `False`), features marked as Internal Standards (IS) are retained.

            * ``keepPeakInfo``
                If keepPeakInfo=True (default `False`) adds the :py:attr:`peakInfo` dictionary to the :py:class:`TargetedDataset` and py:attr:`calibration`. :py:attr:`peakInfo` contains the `peakResponse`, `peakArea`, `peakConcentrationDeviation`, `peakIntegrationFlag` and `peakRT`.

            * ``keepExcluded``
                If keepExcluded=True (default `False`), import exclusions (:py:attr:`excludedImportSampleMetadata`, :py:attr:`excludedImportFeatureMetadata`, :py:attr:`excludedImportIntensityData` and :py:attr:`excludedImportExpectedConcentration`) are kept in the object.

        :param datapath: Path to the TargetLynx exported xml file
        :type datapath: str
        :param calibrationReportPath: Path to the calibration report csv file
        :type calibrationReportPath: str
        :param keepIS: If keepIS=True (default `False`), features marked as Internal Standards (IS) are retained.
        :type keepIS: bool
        :param noiseFilled: If noiseFilled=True (default `False`), values <LLOQ are replaced by the noise concentration
        :type noiseFilled: bool
        :param peakInfo: If keepExcluded=True (default `False`), import exclusions (:py:attr:`excludedImportSampleMetadata`, :py:attr:`excludedImportFeatureMetadata`, :py:attr:`excludedImportIntensityData` and :py:attr:`excludedImportExpectedConcentration`) are kept in the object.
        :type peakInfo: bool
        :param keepExcluded: If keepExcluded=True (default `False`), import exclusions (:py:attr:`excludedImportSampleMetadata`, :py:attr:`excludedImportFeatureMetadata`, :py:attr:`excludedImportIntensityData` and :py:attr:`excludedImportExpectedConcentration`) are kept in the object.
        :type keepExcluded: bool
        :param kwargs: Additional parameters such as `sampleTypeToProcess`, `onlyLLOQ` or `reponseReference`
        :return: None
        """

        # Load TargetLynx output file
        self._readTargetLynxDataset(datapath, calibrationReportPath, **kwargs)

        # Filter calibration samples
        self._filterTargetLynxSamples(**kwargs)

        # Filter IS features (default remove them)
        if keepIS:
            print('IS features are kept for processing:', sum(self.featureMetadata['IS'].values), 'IS features,', sum(~self.featureMetadata['IS'].values), 'other features.')
            print('-----')
            self.Attributes['Log'].append([datetime.now(), 'IS features kept for processing (%d samples). %d IS, %d other features.' % (self.noSamples, sum(self.featureMetadata['IS'].values), sum(~self.featureMetadata['IS'].values))])
        else:
            self._filterTargetLynxIS(**kwargs)

        # Apply limits of quantification
        if noiseFilled:
            self._targetLynxApplyLimitsOfQuantificationNoiseFilled(**kwargs)
        else:
            self._applyLimitsOfQuantification(**kwargs)

        # Remove peakInfo (default remove)
        if keepPeakInfo:
            self.Attributes['Log'].append([datetime.now(), 'TargetLynx peakInfo kept.'])
        else:
            delattr(self, 'peakInfo')
            del self.calibration['calibPeakInfo']

        # Remove import exclusions as they are not useful after import
        if keepExcluded:
            self.Attributes['Log'].append([datetime.now(), 'Features and Samples excluded during import have been kept.'])
        else:
            delattr(self, 'sampleMetadataExcluded')
            delattr(self, 'featureMetadataExcluded')
            delattr(self, 'intensityDataExcluded')
            delattr(self, 'expectedConcentrationExcluded')
            delattr(self, 'excludedFlag')

        # clear **kwargs that have been copied to Attributes
        for i in list(kwargs.keys()):
            try:
                del self.Attributes[i]
            except:
                pass
        for j in ['keepIS','noiseFilled','keepPeakInfo','keepExcluded']:
            try:
                del self.Attributes[j]
            except:
                pass


    def _readTargetLynxDataset(self, datapath, calibrationReportPath, **kwargs):
        """
        Parse a TargetLynx output file (`xml`; sample metadata, feature metadata, intensity, peak area and peak response) and the matching calibration report (`csv`; limits of quantification, noise area, calibration equation parameters), then check their agreement before returning a sufficiently described dataset.
        
        Sets :py:attr:`sampleMetadata`, :py:attr:`featureMetadata`, :py:attr:`intensityData`, :py:attr:`expectedConcentration`, :py:attr:`excludedImportSampleMetadata`, :py:attr:`excludedImportFeatureMetadata`, :py:attr:`excludedImportIntensityData` and :py:attr:`peakInfo`
        
        :param datapath: Path to the TargetLynx export xml file
        :type datapath: str
        :param calibrationReportPath: Path to the calibration report csv file
        :type calibrationReportPath: str
        :return: None
        """

        # Read XML (dumb, no checks, no metadata alteration)
        sampleMetadata, featureMetadata, intensityData, expectedConcentration, peakResponse, peakArea, peakConcentrationDeviation, peakIntegrationFlag, peakRT = self.__getDatasetFromXML(datapath)
        # Read calibration information from .csv (dumb, no metadata alteration, only checks for required columns)
        calibReport = self.__getCalibrationFromReport(calibrationReportPath)
        # Match XML, Calibration Report & SOP
        sampleMetadata, featureMetadata, intensityData, expectedConcentration, excludedImportSampleMetadata, excludedImportFeatureMetadata, excludedImportIntensityData, excludedImportExpectedConcentration, excludedImportFlag, peakResponse, peakArea, peakConcentrationDeviation, peakIntegrationFlag, peakRT = self.__matchDatasetToCalibrationReport(sampleMetadata, featureMetadata, intensityData, expectedConcentration, peakResponse, peakArea, peakConcentrationDeviation, peakIntegrationFlag, peakRT, calibReport)

        self.sampleMetadata        = sampleMetadata
        self.featureMetadata       = featureMetadata
        self._intensityData        = intensityData
        self.expectedConcentration = expectedConcentration
        self.sampleMetadataExcluded        = excludedImportSampleMetadata
        self.featureMetadataExcluded       = excludedImportFeatureMetadata
        self.intensityDataExcluded         = excludedImportIntensityData
        self.expectedConcentrationExcluded = excludedImportExpectedConcentration
        self.excludedFlag                  = excludedImportFlag
        self.peakInfo = {'peakResponse': peakResponse, 'peakArea': peakArea, 'peakConcentrationDeviation': peakConcentrationDeviation, 'peakIntegrationFlag': peakIntegrationFlag, 'peakRT': peakRT}

        # add Dataset mandatory columns
        self.sampleMetadata['AssayRole']         = numpy.nan
        self.sampleMetadata['SampleType']        = numpy.nan
        self.sampleMetadata['Dilution']          = numpy.nan
        self.sampleMetadata['Correction Batch']  = numpy.nan
        self.sampleMetadata['Sample ID']       = numpy.nan
        self.sampleMetadata['Exclusion Details'] = numpy.nan
        #self.sampleMetadata['Batch']             = numpy.nan #already created

        # clear SOP parameters not needed after __matchDatasetToCalibrationReport
        AttributesToRemove = ['compoundID', 'compoundName', 'IS', 'unitFinal', 'unitCorrectionFactor', 'calibrationMethod', 'calibrationEquation', 'quantificationType']
        AttributesToRemove.extend(self.Attributes['externalID'])
        for k in AttributesToRemove:
            del self.Attributes[k]

        self.Attributes['Log'].append([datetime.now(),'TargetLynx data file with %d samples, %d features, loaded from \%s, calibration report read from \%s\'' % (self.noSamples, self.noFeatures, datapath, calibrationReportPath)])


    def __getDatasetFromXML(self, path):
        """
        Parse information for :py:attr:`sampleMetadata`, :py:attr:`featureMetadata`, :py:attr:`intensityData`, :py:attr:`expectedConcentration`, :py:attr:`peakResponse`, :py:attr:`peakArea`, :py:attr:`peakConcentrationDeviation`, :py:attr:`peakIntegrationFlag` and :py:attr:`peakRT` from a xml export file produced by TargetLynx (using the 'File -> Export -> XML' menu option)
        
        :param path: Path to the TargetLynx export xml file
        :type path: str
        :return sampleMetadata: dataframe of sample identifiers and metadata.
        :rtype: pandas.DataFrame, :math:`n` × :math:`p`
        :return featureMetadata: pandas dataframe of feature identifiers and metadata.
        :rtype: pandas.DataFrame, :math:`m` × :math:`q`
        :return intensityData: numpy matrix of intensity measurements.
        :rtype: numpy.ndarray, :math:`n` × :math:`m`
        :return expectedConcentration: pandas dataframe of expected concentration for each sample/feature
        :rtype: pandas.DataFrame, :math:`n` × :math:`m`
        :return peakResponse: pandas dataframe of analytical peak response.
        :rtype: pandas.DataFrame, :math:`n` × :math:`m`
        :return peakArea: pandas dataframe of analytical peak area.
        :rtype: pandas.DataFrame, :math:`n` × :math:`m`
        :return peakConcentrationDeviation: pandas dataframe of %deviation between expected and measured concentration for each sample/feature
        :rtype: pandas.DataFrame, :math:`n` × :math:`m`
        :return peakIntegrationFlag: pandas dataframe of integration flag
        :rtype: pandas.DataFrame, :math:`n` × :math:`m`
        :return peakRT: pandas dataframe of analytical peak Retention time.
        :rtype: pandas.DataFrame, :math:`n` × :math:`m`
        """
        import xml.etree.ElementTree

        inputData = xml.etree.ElementTree.ElementTree(file=path).getroot()[2][0]
        nSamples = int(inputData[1].attrib['count'])
        nFeatures = int(inputData[2].attrib['count'])

        ## Initialise
        # sample metadata
        sample_file_name = list()
        sample_id = list()
        sample_number = list()
        sample_text = list()
        sample_type = list()
        sample_date = list()
        sample_time = list()
        sample_vial = list()
        sample_instrument = list()

        # feature metadata
        compound_name = list()
        compound_id = list()
        compound_IS_id = list()

        # intensity data
        peak_conc = numpy.full([nSamples, nFeatures], numpy.nan)

        # expected concentration
        peak_expconc = numpy.full([nSamples, nFeatures], numpy.nan)

        # Bonus peak info
        peak_concdev = numpy.full([nSamples, nFeatures], numpy.nan)
        peak_area = numpy.full([nSamples, nFeatures], numpy.nan)
        peak_response = numpy.full([nSamples, nFeatures], numpy.nan)
        peak_RT = numpy.full([nSamples, nFeatures], numpy.nan)
        peak_integrationFlag = pandas.DataFrame(index=range(1, nSamples + 1), columns=range(1, nFeatures + 1), dtype='str')

        ## Read data
        # sample metadata & intensity data
        # iterate over samples
        for i_spl in range(0, nSamples):
            spl = inputData[1][i_spl]

            # sample metadata
            sample_file_name.append(spl.attrib['name'])
            sample_id.append(int(spl.attrib['id']))
            sample_number.append(int(spl.attrib['samplenumber']))
            sample_text.append(spl.attrib['sampleid'])
            sample_type.append(spl.attrib['type'])
            sample_date.append(spl.attrib['createdate'])
            sample_time.append(spl.attrib['createtime'])
            sample_vial.append(spl.attrib['vial'])
            sample_instrument.append(spl.attrib['instrument'])

            # iterate over compounds
            for i_cpd in range(0, nFeatures):
                cpdData = spl[i_cpd][0]

                # intensity data
                # for whatever reason, TargetLynx sometimes report no peak by '0.0000' and sometimes by ''
                try:
                    peak_conc[i_spl, i_cpd] = float(cpdData.attrib['analconc'])
                except ValueError:
                    peak_conc[i_spl, i_cpd] = 0.0
                # more peak info
                peak_area[i_spl, i_cpd] = float(cpdData.attrib['area'])
                peak_expconc[i_spl, i_cpd] = float(spl[i_cpd].attrib['stdconc'])
                peak_concdev[i_spl, i_cpd] = float(cpdData.attrib['conccalc'])
                peak_response[i_spl, i_cpd] = float(cpdData.attrib['response'])
                peak_RT[i_spl, i_cpd] = float(cpdData.attrib['foundrt'])
                peak_integrationFlag.iloc[i_spl, i_cpd] = cpdData.attrib['pkflags']

        # feature metadata
        for j_cpd in range(0, nFeatures):
            cpd_calib = inputData[2][j_cpd]
            compound_name.append(cpd_calib.attrib['name'])
            compound_id.append(int(cpd_calib.attrib['id']))
            compound_IS_id.append(cpd_calib[0].attrib['ref'])  # not int() as some IS have ref=''

        ## Output Dataframe
        # sampleMetadata
        sampleMetadata = dict()
        sampleMetadata['Sample File Name'] = sample_file_name
        sampleMetadata['Sample Base Name'] = sample_file_name
        sampleMetadata['TargetLynx Sample ID'] = sample_id
        sampleMetadata['MassLynx Row ID'] = sample_number
        sampleMetadata['Sample Name'] = sample_text
        sampleMetadata['Sample Type'] = sample_type
        sampleMetadata['Acqu Date'] = sample_date
        sampleMetadata['Acqu Time'] = sample_time
        sampleMetadata['Vial'] = sample_vial
        sampleMetadata['Instrument'] = sample_instrument

        # featureMetadata
        featureMetadata = dict()
        featureMetadata['Feature Name'] = compound_name
        featureMetadata['TargetLynx Feature ID'] = compound_id
        featureMetadata['TargetLynx IS ID'] = compound_IS_id

        # intensityData
        intensityData = peak_conc

        # expectedConcentration
        peak_expconc[peak_expconc == 0] = numpy.nan  # remove 0 and replace them by nan
        expectedConcentration = pandas.DataFrame(peak_expconc)

        # Other peak info
        peakResponse = pandas.DataFrame(peak_response)
        peakArea = pandas.DataFrame(peak_area)
        peakConcentrationDeviation = pandas.DataFrame(peak_concdev)
        peakIntegrationFlag = peak_integrationFlag # already dataframe
        peakIntegrationFlag.reset_index(drop=True, inplace=True)
        peakRT = pandas.DataFrame(peak_RT)

        # Convert to DataFrames
        featureMetadata = pandas.concat([pandas.DataFrame(featureMetadata[c], columns=[c]) for c in featureMetadata.keys()], axis=1, sort=False)
        sampleMetadata = pandas.concat([pandas.DataFrame(sampleMetadata[c], columns=[c]) for c in sampleMetadata.keys()], axis=1, sort=False)
        expectedConcentration.columns      = featureMetadata['Feature Name'].values.tolist()
        peakIntegrationFlag.columns        = featureMetadata['Feature Name'].values.tolist()
        peakResponse.columns               = featureMetadata['Feature Name'].values.tolist()
        peakArea.columns                   = featureMetadata['Feature Name'].values.tolist()
        peakConcentrationDeviation.columns = featureMetadata['Feature Name'].values.tolist()
        peakRT.columns                     = featureMetadata['Feature Name'].values.tolist()
        sampleMetadata['Metadata Available'] = False

        return sampleMetadata, featureMetadata, intensityData, expectedConcentration, peakResponse, peakArea, peakConcentrationDeviation, peakIntegrationFlag, peakRT


    def __getCalibrationFromReport(self, path):
        """
        Read the calibration information from a calibration report `csv` following the provided report template.
        
        The following columns are required (leave an empty value to reject a compound):
        
        * Compound
            The compound name, identical to the one employed in the SOP `json` file.
        
        * TargetLynx ID
            The compound TargetLynx ID, identical to the one employed in the SOP `json` file.

        * LLOQ
            Lowest limit of quantification concentration, in the same unit as indicated in TargetLynx.
        
        * ULOQ
            Upper limit of quantification concentration, in the same unit as indicated in TargetLynx.

        The following columns are expected by :py:meth:`~TargetedDataset._targetLynxApplyLimitsOfQuantificationNoiseFilled`:

        * Noise (area)
            Area integrated in a blank sample at the same retention time as the compound of interest (if left empty noise concentration calculation cannot take place).
        
        * a
            :math:`a` coefficient in the calibration equation (if left empty noise concentration calculation cannot take place).
        
        * b
            :math:`b` coefficient in the calibration equation (if left empty noise concentration calculation cannot take place).

        The following columns are recommended:

        * Cpd Info
            Additional information relating to the compound (can be left empty).
        
        * r
            :math:`r` goodness of fit measure for the calibration equation (can be left empty).
        
        * r2
            :math:`r^2` goodness of fit measure for the calibration equation (can be left empty).
        
        :param path: Path to the calibration report csv file.
        :type path: str
        :return calibReport: pandas dataframe of feature identifiers and calibration information.
        :rtype: pandas.DataFrame, :math:`m` × :math:`r`
        :raises LookupError: if the expected columns are absent from the csv file.
        """

        calibReport = pandas.read_csv(path)

        # check minimum number of columns
        expectedCol = ['Compound', 'TargetLynx ID', 'LLOQ', 'ULOQ']
        foundCol = calibReport.columns.values.tolist()

        # if the set is not empty, some columns are missing from the csv
        if set(expectedCol) - set(foundCol) != set():
            raise LookupError('Calibration report (' + os.path.split(path)[1] + ') does not contain the following expected column: ' + str(list(set(expectedCol) - set(foundCol))))

        return calibReport


    def __matchDatasetToCalibrationReport(self, sampleMetadata, featureMetadata, intensityData, expectedConcentration, peakResponse, peakArea, peakConcentrationDeviation, peakIntegrationFlag, peakRT, calibReport):
        """
        Check the agreement of Feature IDs and Feature Names across all inputs (TargetLynx export `xml`, calibration report `csv` and SOP `json`).
        
        First map the calibration report and SOP information, which raise errors in case of disagreement.
        
        This block is then mapped to the TargetLynx `featureMetadata` (on compound ID) and overrides the TargetLynx information (raise warnings).
        
        Features not matched are appended to an `excludedSampleMetadata`, `excludedFeatureMetadata` and `excludedIntensityData` (excluded `peakResponse`, `peakArea`, `peakConcentrationDeviation`, `peakIntegrationFlag` and `peakRT` are discarded).
        
        Additional information is added to the `sampleMetadata` (chromatography, ionisation, acquired time, run order).
        
        Apply the unitCorrectionFactor to the `intensityData`, `LLOQ` and `ULOQ` concentrations and `expectedConcentration`.
        
        :param sampleMetadata: dataframe of sample identifiers and metadata.
        :type sampleMetadata: pandas.DataFrame, :math:`n` × :math:`p`
        :param featureMetadata: pandas dataframe of feature identifiers and metadata.
        :type featureMetadata: pandas.DataFrame, :math:`m` × :math:`q`
        :param intensityData: numpy matrix of intensity measurements.
        :type intensityData: numpy.ndarray, :math:`n` × :math:`m`
        :param expectedConcentration: pandas dataframe of analytical peak expected concentrations.
        :type expectedConcentration: pandas.DataFrame, :math:`n` × :math:`m`
        :param peakResponse: pandas dataframe of analytical peak response.
        :type peakResponse: pandas.DataFrame, :math:`n` × :math:`m`
        :param peakArea: pandas dataframe of analytical peak area.
        :type peakArea: pandas.DataFrame, :math:`n` × :math:`m`
        :param peakConcentrationDeviation: pandas dataframe of analytical peak concentration deviation.
        :type peakConcentrationDeviation: pandas.DataFrame, :math:`n` × :math:`m`
        :param peakIntegrationFlag: pandas dataFrame of analytical peak integration flags.
        :type peakIntegrationFlag: pandas.DataFrame, :math:`n` × :math:`m`
        :param peakRT: pandas dataframe of analytical Retention time.
        :type peakRT: pandas.DataFrame, :math:`n` × :math:`m`
        :param calibReport: pandas dataframe of feature identifiers and calibration informations.
        :type calibReport: pandas.DataFrame, :math:`m` × :math:`r`
        
        :return sampleMetadata: dataframe of sample identifiers and metadata.
        :rtype: pandas.DataFrame, :math:`n` × :math:`p`
        :return finalFeatureMetadata: pandas dataframe of feature identifiers and metadata.
        :rtype: pandas.DataFrame, :math:`m` × :math:`q`
        :return finalIntensityData: numpy matrix of intensity measurements.
        :rtype: numpy.ndarray, :math:`n` × :math:`m`
        :return finalExpectedConcentration: pandas dataframe of expected concentration for each sample/feature
        :rtype: pandas.DataFrame, :math:`n` × :math:`m`
        :return excludedSampleMetadata: list of pandas dataframe of excluded sample measurements for excluded features.
        :rtype: list
        :return excludedFeatureMetadata: list of pandas dataframe of excluded feature identifiers and metadata.
        :rtype: list
        :return excludedIntensityData: list of matrix of intensity measurements for excluded features.
        :rtype: list
        :return excludedExpectedConcentration: list of pandas dataframe of excluded expected concentration.
        :rtype: list
        :return excludedFlag: list of str of exclusion type ('Samples' or 'Features').
        :rtype: list
        :return finalPeakResponse: pandas dataframe of analytical peak response.
        :rtype: pandas.DataFrame, :math:`n` × :math:`m`
        :return finalPeakArea: pandas dataframe of analytical peak area.
        :rtype: pandas.DataFrame, :math:`n` × :math:`m`
        :return finalPeakConcentrationDeviation: pandas dataframe of %deviation between expected and measured concentration for each sample/feature
        :rtype: pandas.DataFrame, :math:`n` × :math:`m`
        :return finalPeakIntegrationFlag: pandas dataframe of integration flag
        :rtype: pandas.DataFrame, :math:`n` × :math:`m`
        :return finalPeakRT: pandas dataframe of analytical peak Retention time
        :rtype: pandas.DataFrame, :math:`n` × :math:`m`
        
        :raises ValueError: if the shape of sampleMetadata, featureMetadata or intensityData shape do not match.
        :raises ValueError: if features in the calibration report and in the SOP differ (number of compounds, compound ID or compound names).
        :raises ValueError: if in the SOP 'quantificationType', 'calibrationMethod' or 'IS' are mismatched.
        """

        import warnings
        from datetime import datetime

        ## sampleMetadata, featureMetadata & intensityData should by construction have the same size
        if sampleMetadata.shape[0] != intensityData.shape[0]:
            raise ValueError('sampleMetadata and intensityData number of samples differ')
        if featureMetadata.shape[0] != intensityData.shape[1]:
            raise ValueError('featureMetadata and intensityData number of compounds differ')
        if intensityData.shape != peakResponse.shape:
            raise ValueError('intensityData and peakResponse number of compounds/samples differ')
        if intensityData.shape != peakArea.shape:
            raise ValueError('intensityData and peakArea number of compounds/samples differ')
        if intensityData.shape != expectedConcentration.shape:
            raise ValueError('intensityData and expectedConcentration number of compounds/samples differ')
        if intensityData.shape != peakConcentrationDeviation.shape:
            raise ValueError('intensityData and peakConcentrationDeviation number of compounds/samples differ')
        if intensityData.shape != peakIntegrationFlag.shape:
            raise ValueError('intensityData and peakIntegrationFlag number of compounds/samples differ')
        if intensityData.shape != peakRT.shape:
            raise ValueError('intensityData and peakRT number of compounds/samples differ')

        # initialise excluded import data
        excludedSampleMetadata        = []
        excludedFeatureMetadata       = []
        excludedIntensityData         = []
        excludedExpectedConcentration = []
        excludedFlag                  = []

        ## SOP is used as 'Truth', if calibReport does not match, it's a problem (Error)
        ## Then if featureMetadata does not match SOP/calibReport, use SOP as reference (message conflict)
        ## Match SOP & calibReport
        # Load SOP
        # calibrationMethod is 'backcalculatedIS' (use response), 'noIS' (use area), or 'noCalibration' (no corrections at all)
        # quantificationType is:
        #   'IS' (expects calibrationMethod=noIS)
        #   'QuantOwnLabeledAnalogue' (would expect 'backcalculatedIS' but could use 'noIS')
        #   'QuantAltLabeledAnalogue' (would expect 'backcalculatedIS' but could use 'noIS')
        #   'Monitored' (which expects 'noCalibration')
        SOPColumnsToLoad = ['compoundID', 'compoundName', 'IS', 'unitFinal', 'unitCorrectionFactor', 'calibrationMethod', 'calibrationEquation', 'quantificationType']
        SOPColumnsToLoad.extend(self.Attributes['externalID'])
        SOPFeatureMetadata = pandas.DataFrame.from_dict(dict((k, self.Attributes[k]) for k in SOPColumnsToLoad), orient='columns')
        SOPFeatureMetadata['compoundID'] = pandas.to_numeric(SOPFeatureMetadata['compoundID'])
        SOPFeatureMetadata['unitCorrectionFactor'] = pandas.to_numeric(SOPFeatureMetadata['unitCorrectionFactor'])
        SOPFeatureMetadata['IS'] = SOPFeatureMetadata['IS'].map({'True': True, 'False': False})
        SOPFeatureMetadata['Unit'] = SOPFeatureMetadata['unitFinal']
        SOPFeatureMetadata.drop('unitFinal', inplace=True, axis=1)

        # convert quantificationType from str to enum
        if 'quantificationType' in SOPFeatureMetadata.columns:
            for qType in QuantificationType:
                SOPFeatureMetadata.loc[SOPFeatureMetadata['quantificationType'].values == qType.name, 'quantificationType'] = qType
        # convert calibrationMethod from str to enum
        if 'calibrationMethod' in SOPFeatureMetadata.columns:
            for cMethod in CalibrationMethod:
                SOPFeatureMetadata.loc[SOPFeatureMetadata['calibrationMethod'].values == cMethod.name, 'calibrationMethod'] = cMethod

        # check that all quantificationType='IS' are also flagged as IS
        # (both have same number of feature + intersection has same number of feature as one of them)
        if (sum((SOPFeatureMetadata['quantificationType'] == QuantificationType.IS)) != sum(SOPFeatureMetadata['IS'])) | (sum((SOPFeatureMetadata['quantificationType'] == QuantificationType.IS) & SOPFeatureMetadata['IS']) != sum(SOPFeatureMetadata['IS'])):
            raise ValueError('Check SOP file, features with quantificationType=\'IS\' must have been flagged as IS=\'True\'')

        # check that all quantificationType='Monitored' have a calibrationMethod='noCalibration'
        # (both have same number of feature + intersection has same number of feature as one of them)
        if (sum((SOPFeatureMetadata['quantificationType'] == QuantificationType.Monitored)) != (sum(SOPFeatureMetadata['calibrationMethod'] == CalibrationMethod.noCalibration))) | (sum((SOPFeatureMetadata['quantificationType'] == QuantificationType.Monitored) & (SOPFeatureMetadata['calibrationMethod'] == CalibrationMethod.noCalibration)) != sum(SOPFeatureMetadata['quantificationType'] == QuantificationType.Monitored)):
            raise ValueError('Check SOP file, features with quantificationType=\'Monitored\' must have a calibrationMethod=\'noCalibration\'\n quantificationType are:\n\'IS\' (expects calibrationMethod=noIS)\n\'QuantOwnLabeledAnalogue\' (would expect \'backcalculatedIS\' but could use \'noIS\' or \'otherCalibration\')\n\'QuantAltLabeledAnalogue\' (would expect \'backcalculatedIS\' but could use \'noIS\' or \'otherCalibration\')\n\'QuantOther\' (can take any CalibrationMethod)\n\'Monitored\' (which expects \'noCalibration\')')

        # check number of compounds in SOP & calibReport
        if SOPFeatureMetadata.shape[0] != calibReport.shape[0]:
            raise ValueError('SOP and Calibration Report number of compounds differ')
        featureCalibSOP = pandas.merge(left=SOPFeatureMetadata, right=calibReport, how='inner', left_on='compoundName', right_on='Compound', sort=False)
        featureCalibSOP.drop('TargetLynx ID', inplace=True, axis=1)

        # check we still have the same number of features (inner join)
        if featureCalibSOP.shape[0] != SOPFeatureMetadata.shape[0]:
            raise ValueError('SOP and Calibration Report compounds differ')

        # check compound names match in SOP and calibReport after join
        if sum(featureCalibSOP['compoundName'] != featureCalibSOP['Compound']) != 0:
            raise ValueError('SOP and Calibration Report compounds names differ: ' + str(featureCalibSOP.loc[(featureCalibSOP['compoundName'] != featureCalibSOP['Compound']), ['compoundName', 'Compound']].values.tolist()))
        featureCalibSOP.drop('Compound', inplace=True, axis=1)

        ## Match calibSOP & featureMetadata
        # left join to keep feature order and limit to features in XML
        finalFeatureMetadata = pandas.merge(left=featureMetadata, right=featureCalibSOP, how='left', left_on='TargetLynx Feature ID', right_on='compoundID', sort=False)

        # limit to compounds present in the SOP (no report of SOP compounds not in XML)
        if finalFeatureMetadata['compoundID'].isnull().sum() != 0:
            warnings.warn("Warning: Only " + str(finalFeatureMetadata['compoundID'].notnull().sum()) + " features shared across the SOP/Calibration report (" + str(featureCalibSOP.shape[0]) + " total) and the TargetLynx output file (" + str(featureMetadata.shape[0]) + " total). " + str(finalFeatureMetadata['compoundID'].isnull().sum()) + " features discarded from the TargetLynx output file.")
            # filter out unavailable features
            unavailableFeatVect = finalFeatureMetadata['compoundID'].isnull().values
            excludedSampleMetadata.append(sampleMetadata)
            excludedFeatureMetadata.append(finalFeatureMetadata.iloc[unavailableFeatVect, :])
            excludedIntensityData.append(intensityData[:, unavailableFeatVect])
            excludedExpectedConcentration.append(expectedConcentration.iloc[:, unavailableFeatVect])
            excludedFlag.append('Features')
            finalFeatureMetadata = finalFeatureMetadata.iloc[~unavailableFeatVect, :]
            finalIntensityData = intensityData[:, ~unavailableFeatVect]
            finalExpectedConcentration = expectedConcentration.iloc[:, ~unavailableFeatVect]
            finalPeakResponse = peakResponse.iloc[:, ~unavailableFeatVect]
            finalPeakArea = peakArea.iloc[:, ~unavailableFeatVect]
            finalPeakConcentrationDeviation = peakConcentrationDeviation.iloc[:, ~unavailableFeatVect]
            finalPeakIntegrationFlag = peakIntegrationFlag.iloc[:, ~unavailableFeatVect]
            finalPeakRT = peakRT.iloc[:, ~unavailableFeatVect]
            # remove duplicate col
            finalFeatureMetadata.drop('compoundID', inplace=True, axis=1)
        else:
            finalIntensityData = intensityData
            finalExpectedConcentration = expectedConcentration
            finalPeakResponse = peakResponse
            finalPeakArea = peakArea
            finalPeakConcentrationDeviation = peakConcentrationDeviation
            finalPeakIntegrationFlag = peakIntegrationFlag
            finalPeakRT = peakRT
            # remove duplicate col
            finalFeatureMetadata.drop('compoundID', inplace=True, axis=1)

        # check names, keep SOP value, report differences
        if sum(finalFeatureMetadata['Feature Name'] != finalFeatureMetadata['compoundName']) != 0:
            warnings.warn('TargetLynx feature names & SOP/Calibration Report compounds names differ; SOP names will be used: ' + str(finalFeatureMetadata.loc[(finalFeatureMetadata['Feature Name'] != finalFeatureMetadata['compoundName']), ['Feature Name','compoundName']].values.tolist()))
            finalFeatureMetadata['Feature Name'] = finalFeatureMetadata['compoundName']
            finalExpectedConcentration.columns      = finalFeatureMetadata['Feature Name'].values.tolist()
            finalPeakResponse.columns               = finalFeatureMetadata['Feature Name'].values.tolist()
            finalPeakArea.columns                   = finalFeatureMetadata['Feature Name'].values.tolist()
            finalPeakConcentrationDeviation.columns = finalFeatureMetadata['Feature Name'].values.tolist()
            finalPeakIntegrationFlag.columns        = finalFeatureMetadata['Feature Name'].values.tolist()
            finalPeakRT.columns                     = finalFeatureMetadata['Feature Name'].values.tolist()
        finalFeatureMetadata.drop('compoundName', inplace=True, axis=1)

        ## Add information to the sampleMetada
        finalSampleMetadata = copy.deepcopy(sampleMetadata)
        # Add chromatography
        finalSampleMetadata.join(pandas.DataFrame([self.Attributes['chromatography']] * finalSampleMetadata.shape[0], columns=['Chromatograpy']))
        # Add ionisation
        finalSampleMetadata.join(pandas.DataFrame([self.Attributes['ionisation']] * finalSampleMetadata.shape[0], columns=['Ionisation']))
        # Add batch, default is 1
        finalSampleMetadata.join(pandas.DataFrame([1] * finalSampleMetadata.shape[0], columns=['Batch']))
        # Process Sample Type
        finalSampleMetadata['Calibrant'] = finalSampleMetadata['Sample Type'] == 'Standard'
        finalSampleMetadata['Study Sample'] = finalSampleMetadata['Sample Type'] == 'Analyte'
        finalSampleMetadata['Blank'] = finalSampleMetadata['Sample Type'] == 'Blank'
        finalSampleMetadata['QC'] = finalSampleMetadata['Sample Type'] == 'QC'
        # unused Sample Types
        # sampleMetadata['Solvent'] = sampleMetadata['Sample Type'] == 'Solvent'
        # sampleMetadata['Recovery'] = sampleMetadata['Sample Type'] == 'Recovery'
        # sampleMetadata['Donor'] = sampleMetadata['Sample Type'] == 'Donor'
        # sampleMetadata['Receptor'] = sampleMetadata['Sample Type'] == 'Receptor'
        finalSampleMetadata['Other'] = (~finalSampleMetadata['Calibrant'] & ~finalSampleMetadata['Study Sample'] & ~finalSampleMetadata['Blank'] & ~finalSampleMetadata['QC'])  # & ~sampleMetadata['Solvent'] & ~sampleMetadata['Recovery'] & ~sampleMetadata['Donor'] & ~sampleMetadata['Receptor']
        # Add Acquired Time
        finalSampleMetadata['Acquired Time'] = numpy.nan
        for i in range(finalSampleMetadata.shape[0]):
            try:
                finalSampleMetadata.loc[i, 'Acquired Time'] = datetime.strptime(str(finalSampleMetadata.loc[i, 'Acqu Date']) + " " + str(finalSampleMetadata.loc[i, 'Acqu Time']),'%d-%b-%y %H:%M:%S')
            except ValueError:
                pass
        finalSampleMetadata['Acquired Time'] = pandas.to_datetime(finalSampleMetadata['Acquired Time'])
        # Add Run Order
        finalSampleMetadata['Order'] = finalSampleMetadata.sort_values(by='Acquired Time').index
        finalSampleMetadata['Run Order'] = finalSampleMetadata.sort_values(by='Order').index
        finalSampleMetadata.drop('Order', axis=1, inplace=True)
        # Initialise the Batch to 1
        finalSampleMetadata['Batch'] = [1]*finalSampleMetadata.shape[0]

        ## Apply unitCorrectionFactor
        finalFeatureMetadata['LLOQ'] = finalFeatureMetadata['LLOQ'] * finalFeatureMetadata['unitCorrectionFactor']  # NaN will be kept
        finalFeatureMetadata['ULOQ'] = finalFeatureMetadata['ULOQ'] * finalFeatureMetadata['unitCorrectionFactor']
        finalIntensityData         = finalIntensityData * finalFeatureMetadata['unitCorrectionFactor'].values
        finalExpectedConcentration = finalExpectedConcentration * finalFeatureMetadata['unitCorrectionFactor'].values

        ## Summary
        print('TagetLynx output, Calibration report and SOP information matched:')
        print('Targeted Method: ' + self.Attributes['methodName'])
        print(str(finalSampleMetadata.shape[0]) + ' samples (' + str(sum(finalSampleMetadata['Calibrant'])) + ' calibration points, ' + str(sum(finalSampleMetadata['Study Sample'])) + ' study samples)')
        print(str(finalFeatureMetadata.shape[0]) + ' features (' + str(sum(finalFeatureMetadata['IS'])) + ' IS, ' + str(sum(finalFeatureMetadata['quantificationType'] == QuantificationType.QuantOwnLabeledAnalogue)) + ' quantified and validated with own labeled analogue, ' + str(sum(finalFeatureMetadata['quantificationType'] == QuantificationType.QuantAltLabeledAnalogue)) + ' quantified and validated with alternative labeled analogue, ' + str(sum(finalFeatureMetadata['quantificationType'] == QuantificationType.QuantOther)) + ' other quantification, ' + str(sum(finalFeatureMetadata['quantificationType'] == QuantificationType.Monitored)) + ' monitored for relative information)')
        if len(excludedFeatureMetadata) != 0:
            print(str(excludedFeatureMetadata[0].shape[0]) + ' features excluded as missing from the SOP')
        print('All concentrations converted to final units')
        print('-----')

        return finalSampleMetadata, finalFeatureMetadata, finalIntensityData, finalExpectedConcentration, excludedSampleMetadata, excludedFeatureMetadata, excludedIntensityData, excludedExpectedConcentration, excludedFlag, finalPeakResponse, finalPeakArea, finalPeakConcentrationDeviation, finalPeakIntegrationFlag, finalPeakRT


    def _filterTargetLynxSamples(self, sampleTypeToProcess=['Study Sample', 'QC'], **kwargs):
        """
        Isolate 'Calibrant' samples ('Sample Type' == 'Standard' in MassLynx) and create the :py:attr:`calibration` dictionary, following :py:meth:`~TargetedDataset._readTargetLynxDataset`.

        Exclude samples based on their MassLynx 'Sample Type'. Only the types passed in `sampleTypeToProcess` are kept. Values are 'Study Sample' ('Analyte' in MassLynx), 'Blank', 'QC' or 'Other' (for all other MassLynx entries).

        :param sampleTypeToProcess: list of ['Study Sample','Blank','QC','Other'] for the sample types to keep.
        :type sampleTypeToProcess: list of str
        :return: None
        :raises ValueError: if 'sampleTypeToProcess' is not recognised.
        :raises AttributeError: if the excludedImport lists do not exist.
        """

        # check inputs
        if set(sampleTypeToProcess) - set(['Study Sample', 'Blank', 'QC', 'Other']) != set():
            raise ValueError('sampleTypeToProcess ' + str(
                set(sampleTypeToProcess) - set(['Study Sample', 'Blank', 'QC', 'Other'])) + ' is not recognised')
        # check excluded exist
        if((not hasattr(self,'sampleMetadataExcluded'))|(not hasattr(self,'featureMetadataExcluded'))|(not hasattr(self,'intensityDataExcluded'))|(not hasattr(self,'expectedConcentrationExcluded'))|(not hasattr(self,'excludedFlag'))):
            raise AttributeError('sampleMetadataExcluded, featureMetadataExcluded, intensityDataExcluded, expectedConcentrationExcluded or excludedFlag have not bee previously initialised')

        sampleMetadata        = copy.deepcopy(self.sampleMetadata)
        featureMetadata       = copy.deepcopy(self.featureMetadata)
        intensityData         = copy.deepcopy(self._intensityData)
        expectedConcentration = copy.deepcopy(self.expectedConcentration)
        excludedImportSampleMetadata        = copy.deepcopy(self.sampleMetadataExcluded)
        excludedImportFeatureMetadata       = copy.deepcopy(self.featureMetadataExcluded)
        excludedImportIntensityData         = copy.deepcopy(self.intensityDataExcluded)
        excludedImportExpectedConcentration = copy.deepcopy(self.expectedConcentrationExcluded)
        excludedImportFlag                  = copy.deepcopy(self.excludedFlag)
        peakInfo = copy.deepcopy(self.peakInfo)

        # Calibration information
        calibFeatureMetadata       = featureMetadata
        calibSampleMetadata        = sampleMetadata.loc[sampleMetadata['Calibrant'].values, :]
        calibIntensityData         = intensityData[sampleMetadata['Calibrant'].values, :]
        calibExpectedConcentration = expectedConcentration.loc[sampleMetadata['Calibrant'].values, :]
        calibPeakResponse               = peakInfo['peakResponse'].loc[sampleMetadata['Calibrant'].values, :]
        calibPeakArea                   = peakInfo['peakArea'].loc[sampleMetadata['Calibrant'].values, :]
        calibPeakConcentrationDeviation = peakInfo['peakConcentrationDeviation'].loc[sampleMetadata['Calibrant'].values, :]
        calibPeakIntegrationFlag        = peakInfo['peakIntegrationFlag'].loc[sampleMetadata['Calibrant'].values, :]
        calibPeakRT                     = peakInfo['peakRT'].loc[sampleMetadata['Calibrant'].values, :]
        calibPeakInfo = {'peakResponse': calibPeakResponse, 'peakArea': calibPeakArea, 'peakConcentrationDeviation': calibPeakConcentrationDeviation, 'peakIntegrationFlag': calibPeakIntegrationFlag, 'peakRT': calibPeakRT}
        calibration   = {'calibSampleMetadata': calibSampleMetadata, 'calibFeatureMetadata': calibFeatureMetadata, 'calibIntensityData': calibIntensityData, 'calibExpectedConcentration': calibExpectedConcentration, 'calibPeakInfo': calibPeakInfo}

        # Samples to keep
        samplesToProcess = [False] * sampleMetadata.shape[0]
        for i in sampleTypeToProcess:
            samplesToProcess = (samplesToProcess | sampleMetadata[i]).values
        # Filter
        tmpSampleMetadata        = sampleMetadata.loc[samplesToProcess, :]
        tmpIntensityData         = intensityData[samplesToProcess, :]
        tmpExpectedConcentration = expectedConcentration.loc[samplesToProcess, :]
        tmpPeakResponse               = peakInfo['peakResponse'].loc[samplesToProcess, :]
        tmpPeakArea                   = peakInfo['peakArea'].loc[samplesToProcess, :]
        tmpPeakConcentrationDeviation = peakInfo['peakConcentrationDeviation'].loc[samplesToProcess, :]
        tmpPeakIntegrationFlag        = peakInfo['peakIntegrationFlag'].loc[samplesToProcess, :]
        tmpPeakRT                     = peakInfo['peakRT'].loc[samplesToProcess, :]
        tmpPeakInfo = {'peakResponse': tmpPeakResponse, 'peakArea': tmpPeakArea, 'peakConcentrationDeviation': tmpPeakConcentrationDeviation, 'peakIntegrationFlag': tmpPeakIntegrationFlag, 'peakRT': tmpPeakRT}

        # Samples to exclude
        samplesToExclude = ~samplesToProcess & ~sampleMetadata['Calibrant'].values  # no need to exclude calibrant
        if sum(samplesToExclude) != 0:
            excludedImportSampleMetadata.append(sampleMetadata.loc[samplesToExclude, :])
            excludedImportFeatureMetadata.append(featureMetadata)
            excludedImportIntensityData.append(intensityData[samplesToExclude, :])
            excludedImportExpectedConcentration.append(expectedConcentration.loc[samplesToExclude, :])
            excludedImportFlag.append('Samples')

        # Clean columns
        tmpSampleMetadata.reset_index(drop=True, inplace=True)
        tmpSampleMetadata = tmpSampleMetadata.drop(['Calibrant', 'Study Sample', 'Blank', 'QC', 'Other'], axis=1)
        tmpExpectedConcentration.reset_index(drop=True, inplace=True)

        # Output
        self.sampleMetadata        = tmpSampleMetadata
        self.featureMetadata       = featureMetadata
        self._intensityData        = tmpIntensityData
        self.expectedConcentration = tmpExpectedConcentration
        self.sampleMetadataExcluded        = excludedImportSampleMetadata
        self.featureMetadataExcluded       = excludedImportFeatureMetadata
        self.intensityDataExcluded         = excludedImportIntensityData
        self.expectedConcentrationExcluded = excludedImportExpectedConcentration
        self.excludedFlag                  = excludedImportFlag
        self.peakInfo    = tmpPeakInfo
        self.calibration = calibration

        # log the modifications
        print(sampleTypeToProcess, 'samples are kept for processing')
        print('-----')
        self.Attributes['Log'].append([datetime.now(), '%s samples kept for processing (%d samples, %d features). %d calibration samples filtered. %d samples excluded.' % (str(sampleTypeToProcess), self.noSamples, self.noFeatures, self.calibration['calibSampleMetadata'].shape[0], sum(samplesToExclude))])


    def _filterTargetLynxIS(self, **kwargs):
        """
        Filter out Internal Standard (IS) features and add them to excludedImportSampleMetadata, excludedImportFeatureMetadata, excludedImportIntensityData and excludedImportExpectedConcentration.
        IS filtered from self.calibration are not saved.
        :return: None
        :raises AttributeError: if the excludedImport lists do not exist.
        :raises AttributeError: if the calibration dictionary does not exist.
        """

        # check excludedImport exist (ensures functions are run in the right order)
        if ((not hasattr(self, 'sampleMetadataExcluded')) | (not hasattr(self, 'featureMetadataExcluded')) | (not hasattr(self, 'intensityDataExcluded')) | (not hasattr(self, 'expectedConcentrationExcluded')) | (not hasattr(self, 'excludedFlag'))):
            raise AttributeError('sampleMetadataExcluded, featureMetadataExcluded, intensityDataExcluded, expectedConcentrationExcluded or excludedFlag have not bee previously initialised')
        # check calibration dictionary exist (ensures functions are run in the right order)
        if not hasattr(self, 'calibration'):
            raise AttributeError('calibration dictionary has not been previously initialised')

        sampleMetadata        = copy.deepcopy(self.sampleMetadata)
        featureMetadata       = copy.deepcopy(self.featureMetadata)
        intensityData         = copy.deepcopy(self._intensityData)
        expectedConcentration = copy.deepcopy(self.expectedConcentration)
        excludedImportSampleMetadata        = copy.deepcopy(self.sampleMetadataExcluded)
        excludedImportFeatureMetadata       = copy.deepcopy(self.featureMetadataExcluded)
        excludedImportIntensityData         = copy.deepcopy(self.intensityDataExcluded)
        excludedImportExpectedConcentration = copy.deepcopy(self.expectedConcentrationExcluded)
        excludedImportFlag                  = copy.deepcopy(self.excludedFlag)
        calibration = copy.deepcopy(self.calibration)
        peakInfo    = copy.deepcopy(self.peakInfo)

        # Feature to keep
        keptFeat = ~featureMetadata['IS'].values.astype(bool)
        # Filter
        tmpFeatureMetadata       = featureMetadata.loc[keptFeat, :]
        tmpIntensityData         = intensityData[:, keptFeat]
        tmpExpectedConcentration = expectedConcentration.loc[:, keptFeat]
        tmpCalibFeatureMetadata       = calibration['calibFeatureMetadata'].loc[keptFeat, :]
        tmpCalibIntensityData         = calibration['calibIntensityData'][:, keptFeat]
        tmpCalibExpectedConcentration = calibration['calibExpectedConcentration'].loc[:, keptFeat]
        tmpCalibPeakResponse               = calibration['calibPeakInfo']['peakResponse'].loc[:, keptFeat]
        tmpCalibPeakArea                   = calibration['calibPeakInfo']['peakArea'].loc[:, keptFeat]
        tmpCalibPeakConcentrationDeviation = calibration['calibPeakInfo']['peakConcentrationDeviation'].loc[:, keptFeat]
        tmpCalibPeakIntegrationFlag        = calibration['calibPeakInfo']['peakIntegrationFlag'].loc[:, keptFeat]
        tmpCalibPeakRT                     = calibration['calibPeakInfo']['peakRT'].loc[:, keptFeat]
        tmpCalibPeakInfo = {'peakResponse': tmpCalibPeakResponse, 'peakArea': tmpCalibPeakArea, 'peakConcentrationDeviation': tmpCalibPeakConcentrationDeviation, 'peakIntegrationFlag': tmpCalibPeakIntegrationFlag, 'peakRT': tmpCalibPeakRT}
        tmpCalibration   = {'calibSampleMetadata': calibration['calibSampleMetadata'], 'calibFeatureMetadata': tmpCalibFeatureMetadata, 'calibIntensityData': tmpCalibIntensityData, 'calibExpectedConcentration': tmpCalibExpectedConcentration, 'calibPeakInfo': tmpCalibPeakInfo}
        tmpPeakResponse               = peakInfo['peakResponse'].loc[:, keptFeat]
        tmpPeakArea                   = peakInfo['peakArea'].loc[:, keptFeat]
        tmpPeakConcentrationDeviation = peakInfo['peakConcentrationDeviation'].loc[:, keptFeat]
        tmpPeakIntegrationFlag        = peakInfo['peakIntegrationFlag'].loc[:, keptFeat]
        tmpPeakRT                     = peakInfo['peakRT'].loc[:, keptFeat]
        tmpPeakInfo = {'peakResponse': tmpPeakResponse, 'peakArea': tmpPeakArea, 'peakConcentrationDeviation': tmpPeakConcentrationDeviation, 'peakIntegrationFlag': tmpPeakIntegrationFlag, 'peakRT': tmpPeakRT}

        # Features to exclude
        ISFeat = ~keptFeat
        if sum(ISFeat) != 0:
            excludedImportSampleMetadata.append(sampleMetadata)
            excludedImportFeatureMetadata.append(featureMetadata.loc[ISFeat, :])
            excludedImportIntensityData.append(intensityData[:, ISFeat])
            excludedImportExpectedConcentration.append(expectedConcentration.loc[:, ISFeat])
            excludedImportFlag.append('Features')

        # Clean columns
        tmpFeatureMetadata.reset_index(drop=True, inplace=True)
        tmpCalibration['calibFeatureMetadata'].reset_index(drop=True, inplace=True)
        tmpFeatureMetadata = tmpFeatureMetadata.drop(['IS', 'TargetLynx IS ID'], axis=1)

        # Output
        self.featureMetadata       = tmpFeatureMetadata
        self._intensityData        = tmpIntensityData
        self.expectedConcentration = tmpExpectedConcentration
        self.sampleMetadataExcluded        = excludedImportSampleMetadata
        self.featureMetadataExcluded       = excludedImportFeatureMetadata
        self.intensityDataExcluded         = excludedImportIntensityData
        self.expectedConcentrationExcluded = excludedImportExpectedConcentration
        self.excludedFlag                  = excludedImportFlag
        self.calibration = tmpCalibration
        self.peakInfo    = tmpPeakInfo

        # log the modifications
        print(sum(keptFeat), 'feature are kept for processing,',sum(ISFeat),'IS removed')
        print('-----')
        self.Attributes['Log'].append([datetime.now(), '%d features kept for processing (%d samples). %d IS features filtered.' % (sum(keptFeat), self.noSamples, sum(ISFeat))])


    def _loadBrukerXMLDataset(self, datapath, fileNamePattern=None, pdata=1, unit=None, **kwargs):
        """
        Initialise object from Bruker XML files. Read files and prepare a valid TargetedDataset.

        Targeted data measurements are read and mapped to pre-defined SOPs. Once the import is finished, only properly read samples are returned and only features mapped onto the pre-defined SOP and sufficiently described. Only the first instance of a duplicated feature is kept.

        :param str datapath: Path to the folder containing all `xml` files, all directories below :file:`datapath` will be scanned for valid `xml` files.
        :param str fileNamePattern: Regex pattern to identify the `xml` files in `datapath` folder
        :param int pdata: pdata files to parse (default 1)
        :param unit: if features are present more than once, only keep the features with the unit passed as input.
        :type unit: None or str
        :raises TypeError: if `fileNamePattern` is not a string
        :raises TypeError: if `pdata` is not an integer
        :raises TypeError: if `unit` is not 'None' or a string
        :raises ValueError: if `unit` is not one of the unit in the input data
        :return: None
        """
        from ..utilities._readBrukerXML import importBrukerXML
        from ..utilities.extractParams import buildFileList

        if fileNamePattern is None:
            fileNamePattern = self.Attributes['fileNamePattern']

        # Check inputs
        if not isinstance(fileNamePattern, str):
            raise TypeError('\'fileNamePattern\' must be a string')
        if not isinstance(pdata, int):
            raise TypeError('\'pdata\' must be an integer')
        if unit is not None:
            if not isinstance(unit, str):
                raise TypeError('\'unit\' must be a string')

        ## Build a list of xml files matching the pdata in the right folder
        pattern = re.compile(fileNamePattern)
        filelist = buildFileList(datapath, pattern)
        pdataPattern = re.compile('.*?pdata.*?%i' % (pdata))
        filelist = [x for x in filelist if pdataPattern.match(x)]

        ## Load intensity, sampleMetadata and featureMetadata. Files that cannot be opened raise warnings, and are filtered from the returned matrices.
        (self.intensityData, self.sampleMetadata, self.featureMetadata) = importBrukerXML(filelist)

        ## Filter unit if required
        avUnit = self.featureMetadata['Unit'].unique().tolist()
        if unit is not None:
            if unit not in self.featureMetadata['Unit'].unique().tolist():
                raise ValueError('The unit \'' + str(unit) + '\' is not present in the input data, available units: ' + str(avUnit))
            keepMask = (self.featureMetadata['Unit'] == unit).values
            self.featureMetadata = self.featureMetadata.loc[keepMask, :]
            self.featureMetadata.reset_index(drop=True, inplace=True)
            self.intensityData = self.intensityData[:, keepMask]

        ## Check all features are unique, and
        u_ids, u_counts = numpy.unique(self.featureMetadata['Feature Name'], return_counts=True)
        if not all(u_counts == 1):
            dupFeat = u_ids[u_counts != 1].tolist()
            warnings.warn('The following features are present more than once, only the first occurence will be kept: ' + str(dupFeat) + '. For further filtering, available units are: ' + str(avUnit))
            # only keep the first of duplicated features
            keepMask = ~self.featureMetadata['Feature Name'].isin(dupFeat).values
            keepFirstVal = [(self.featureMetadata['Feature Name'] == Feat).idxmax() for Feat in dupFeat]
            keepMask[keepFirstVal] = True
            self.featureMetadata = self.featureMetadata.loc[keepMask, :]
            self.featureMetadata.reset_index(drop=True, inplace=True)
            self.intensityData = self.intensityData[:, keepMask]

        ## Reformat featureMetadata
        # quantificationType
        self.featureMetadata['quantificationType'] = numpy.nan
        self.featureMetadata.loc[self.featureMetadata['type'] == 'quantification', 'quantificationType'] = QuantificationType.QuantOther
        self.featureMetadata.loc[self.featureMetadata['type'] != 'quantification', 'quantificationType'] = QuantificationType.Monitored
        self.featureMetadata.drop('type', inplace=True, axis=1)
        # calibrationMethod
        self.featureMetadata['calibrationMethod'] = numpy.nan
        self.featureMetadata.loc[self.featureMetadata['quantificationType'] == QuantificationType.QuantOther, 'calibrationMethod'] = CalibrationMethod.otherCalibration
        self.featureMetadata.loc[self.featureMetadata['quantificationType'] == QuantificationType.Monitored, 'calibrationMethod'] = CalibrationMethod.noCalibration
        # rename columns
        self.featureMetadata.rename(columns={'loq': 'LLOQ', 'lod': 'LOD', 'Lower Reference Bound': 'Lower Reference Percentile', 'Upper Reference Bound': 'Upper Reference Percentile'}, inplace=True)
        # replace '-' with nan
        self.featureMetadata['LLOQ'].replace('-', numpy.nan, inplace=True)
        self.featureMetadata['LLOQ'] = [float(x) for x in self.featureMetadata['LLOQ'].tolist()]
        self.featureMetadata['LOD'].replace('-', numpy.nan, inplace=True)
        self.featureMetadata['LOD'] = [float(x) for x in self.featureMetadata['LOD'].tolist()]
        # ULOQ
        self.featureMetadata['ULOQ'] = numpy.nan

        ## Initialise sampleMetadata
        self.sampleMetadata['AssayRole'] = numpy.nan
        self.sampleMetadata['SampleType'] = numpy.nan
        self.sampleMetadata['Dilution'] = 100
        self.sampleMetadata['Correction Batch'] = numpy.nan
        self.sampleMetadata['Sample ID'] = numpy.nan
        self.sampleMetadata['Exclusion Details'] = None
        # add Run Order
        self.sampleMetadata['Order'] = self.sampleMetadata.sort_values(by='Acquired Time').index
        self.sampleMetadata['Run Order'] = self.sampleMetadata.sort_values(by='Order').index
        self.sampleMetadata.drop('Order', axis=1, inplace=True)
        # initialise the Batch to 1
        self.sampleMetadata['Batch'] = [1] * self.sampleMetadata.shape[0]
        self.sampleMetadata['Metadata Available'] = False

        ## Initialise expectedConcentration
        self.expectedConcentration = pandas.DataFrame(None, index=list(self.sampleMetadata.index), columns=self.featureMetadata['Feature Name'].tolist())

        ## Initialise empty Calibration info
        self.calibration = dict()
        self.calibration['calibIntensityData'] = numpy.ndarray((0, self.featureMetadata.shape[0]))
        self.calibration['calibSampleMetadata'] = pandas.DataFrame(None, columns=self.sampleMetadata.columns)
        self.calibration['calibSampleMetadata']['Metadata Available'] = False
        self.calibration['calibFeatureMetadata'] = pandas.DataFrame({'Feature Name': self.featureMetadata['Feature Name'].tolist()})
        self.calibration['calibExpectedConcentration'] = pandas.DataFrame(None, columns=self.featureMetadata['Feature Name'].tolist())

        ## Summary
        print('Targeted Method: ' + self.Attributes['methodName'])
        print(str(self.sampleMetadata.shape[0]) + ' study samples')
        print(str(self.featureMetadata.shape[0]) + ' features (' + str(sum(self.featureMetadata['quantificationType'] == QuantificationType.IS)) + ' IS, ' + str(sum(self.featureMetadata['quantificationType'] == QuantificationType.QuantOwnLabeledAnalogue)) + ' quantified and validated with own labeled analogue, ' + str(sum(self.featureMetadata['quantificationType'] == QuantificationType.QuantAltLabeledAnalogue)) + ' quantified and validated with alternative labeled analogue, ' + str(sum(self.featureMetadata['quantificationType'] == QuantificationType.QuantOther)) + ' other quantification, ' + str(sum(self.featureMetadata['quantificationType'] == QuantificationType.Monitored)) + ' monitored for relative information)')
        print('-----')

        ## Apply limit of quantification?
        self._applyLimitsOfQuantification(**kwargs)

        ## clear **kwargs that have been copied to Attributes
        for i in list(kwargs.keys()):
            try:
                del self.Attributes[i]
            except:
                pass
        for j in ['fileNamePattern', 'pdata', 'unit']:
            try:
                del self.Attributes[j]
            except:
                pass


    def _applyLimitsOfQuantification(self, onlyLLOQ=False, **kwargs):
        """
        For each feature, replace intensity values inferior to the lowest limit of quantification or superior to the upper limit of quantification, by a fixed value.

        Features missing the minimal required information are excluded from :py:attr:'featureMetadata', :py:attr:'intensityData', :py:attr:'expectedConcentration' and :py:attr:'calibration'. Features `'Monitored for relative information'` (and `'noCalibration'`) are not processed and returned without alterations. Features with `'Other quantification'` are allowed `Nan` in the LLOQ or ULOQ (no replacement takes place).

        Calibration data should not be processed and therefore returned without modification.

        Units in :py:attr:`_intensityData`, :py:attr:`featureMetadata['LLOQ'] and :py:attr:`featureMetadata['ULOQ']` are expected to be identical for a given feature.

        Note: In merged datasets, calibration is a list of dict, with features in each calibration dict potentially different from features in featureMetadata and _intensityData.
        Therefore in merged dataset, features are not filtered in each individual calibration.

        If features are excluded due to the lack of required featureMetadata info, the masks will be reinitialised

        :param onlyLLOQ: if True only correct <LLOQ, if False correct <LLOQ and >ULOQ
        :type onlyLLOQ: bool
        :return: None
        :raises AttributeError: if :py:attr:`featureMetadata['LLOQ']` is missing
        :raises AttributeError: if :py:attr:`featureMetadata['ULOQ']` is missing and onlyLLOQ==False
        """

        sampleMetadata        = copy.deepcopy(self.sampleMetadata)
        featureMetadata       = copy.deepcopy(self.featureMetadata)
        intensityData         = copy.deepcopy(self._intensityData)
        expectedConcentration = copy.deepcopy(self.expectedConcentration)
        calibration           = copy.deepcopy(self.calibration)
        if ((not hasattr(self, 'sampleMetadataExcluded')) | (not hasattr(self, 'featureMetadataExcluded')) | (not hasattr(self, 'intensityDataExcluded')) | (not hasattr(self, 'expectedConcentrationExcluded')) | (not hasattr(self, 'excludedFlag'))):
            sampleMetadataExcluded        = []
            featureMetadataExcluded       = []
            intensityDataExcluded         = []
            expectedConcentrationExcluded = []
            excludedFlag                  = []
        else:
            sampleMetadataExcluded        = copy.deepcopy(self.sampleMetadataExcluded)
            featureMetadataExcluded       = copy.deepcopy(self.featureMetadataExcluded)
            intensityDataExcluded         = copy.deepcopy(self.intensityDataExcluded)
            expectedConcentrationExcluded = copy.deepcopy(self.expectedConcentrationExcluded)
            excludedFlag                  = copy.deepcopy(self.excludedFlag)

        ## Check input columns
        if 'LLOQ' not in featureMetadata.columns:
            raise AttributeError('the featureMetadata[\'LLOQ\'] column is absent')
        if onlyLLOQ==False:
            if 'ULOQ' not in featureMetadata.columns:
                raise AttributeError('featureMetadata[\'ULOQ\'] column is absent')

        ## Features only Monitored are not processed and passed untouched (concatenated back at the end)
        untouched = (featureMetadata['quantificationType'] == QuantificationType.Monitored).values
        if sum(untouched) != 0:
            print('The following features are only monitored and therefore not processed for LOQs: ' + str(featureMetadata.loc[untouched, 'Feature Name'].values.tolist()))
            untouchedFeatureMetadata = featureMetadata.loc[untouched, :]
            featureMetadata = featureMetadata.loc[~untouched, :]
            untouchedIntensityData = intensityData[:, untouched]
            intensityData = intensityData[:, ~untouched]
            untouchedExpectedConcentration = expectedConcentration.loc[:, untouched]
            expectedConcentration = expectedConcentration.loc[:, ~untouched]
            # same reordering of the calibration
            if isinstance(calibration, dict):
                untouchedCalibFeatureMetadata = calibration['calibFeatureMetadata'].loc[untouched, :]
                calibration['calibFeatureMetadata'] = calibration['calibFeatureMetadata'].loc[~untouched, :]
                untouchedCalibIntensityData = calibration['calibIntensityData'][:, untouched]
                calibration['calibIntensityData'] = calibration['calibIntensityData'][:, ~untouched]
                untouchedCalibExpectedConcentration = calibration['calibExpectedConcentration'].loc[:, untouched]
                calibration['calibExpectedConcentration'] = calibration['calibExpectedConcentration'].loc[:, ~untouched]


        ## Exclude features without required information
        unusableFeat = featureMetadata['LLOQ'].isnull().values & (featureMetadata['quantificationType'] != QuantificationType.QuantOther).values
        if not onlyLLOQ:
            unusableFeat = unusableFeat | (featureMetadata['ULOQ'].isnull().values & (featureMetadata['quantificationType'] != QuantificationType.QuantOther).values)
        if sum(unusableFeat) != 0:
            print(str(sum(unusableFeat)) + ' features cannot be pre-processed:')
            print('\t' + str(sum(unusableFeat)) + ' features lack the required information to apply limits of quantification')
            # store
            sampleMetadataExcluded.append(sampleMetadata)
            featureMetadataExcluded.append(featureMetadata.loc[unusableFeat, :])
            intensityDataExcluded.append(intensityData[:, unusableFeat])
            expectedConcentrationExcluded.append(expectedConcentration.loc[:, unusableFeat])
            excludedFlag.append('Features')
            #remove
            featureMetadata = featureMetadata.loc[~unusableFeat, :]
            intensityData = intensityData[:, ~unusableFeat]
            expectedConcentration = expectedConcentration.loc[:, ~unusableFeat]
            if isinstance(calibration, dict):
                calibration['calibFeatureMetadata'] = calibration['calibFeatureMetadata'].loc[~unusableFeat, :]
                calibration['calibIntensityData'] = calibration['calibIntensityData'][:, ~unusableFeat]
                calibration['calibExpectedConcentration'] = calibration['calibExpectedConcentration'].loc[:, ~unusableFeat]


        ## Values replacement (-inf / +inf)
        # iterate over the features
        for i in range(0, featureMetadata.shape[0]):
            # LLOQ
            if not numpy.isnan(featureMetadata['LLOQ'].values[i]):
                toReplaceLLOQ = intensityData[:, i] < featureMetadata['LLOQ'].values[i]
                intensityData[toReplaceLLOQ, i] = -numpy.inf

            # ULOQ
            if not onlyLLOQ:
                if not numpy.isnan(featureMetadata['ULOQ'].values[i]):
                    toReplaceULOQ = intensityData[:, i] > featureMetadata['ULOQ'].values[i]
                    intensityData[toReplaceULOQ, i] = numpy.inf


        ## Add back the untouched monitored features
        if sum(untouched) != 0:
            featureMetadata = pandas.concat([featureMetadata, untouchedFeatureMetadata], axis=0, sort=False)
            intensityData = numpy.concatenate((intensityData, untouchedIntensityData), axis=1)
            expectedConcentration = pandas.concat([expectedConcentration, untouchedExpectedConcentration], axis=1, sort=False)
            # reorder the calib
            if isinstance(calibration, dict):
                calibration['calibFeatureMetadata'] = pandas.concat([calibration['calibFeatureMetadata'], untouchedCalibFeatureMetadata], axis=0, sort=False)
                calibration['calibIntensityData'] = numpy.concatenate((calibration['calibIntensityData'], untouchedCalibIntensityData), axis=1)
                calibration['calibExpectedConcentration'] = pandas.concat([calibration['calibExpectedConcentration'], untouchedCalibExpectedConcentration], axis=1, sort=False)

        # Remove excess info
        featureMetadata.reset_index(drop=True, inplace=True)
        expectedConcentration.reset_index(drop=True, inplace=True)
        if isinstance(calibration, dict):
            calibration['calibFeatureMetadata'].reset_index(drop=True, inplace=True)
            calibration['calibExpectedConcentration'].reset_index(drop=True, inplace=True)

        ## return dataset with limits of quantification applied
        self.featureMetadata       = featureMetadata
        self._intensityData        = intensityData
        self.expectedConcentration = expectedConcentration
        self.calibration           = calibration
        self.sampleMetadataExcluded        = sampleMetadataExcluded
        self.featureMetadataExcluded       = featureMetadataExcluded
        self.intensityDataExcluded         = intensityDataExcluded
        self.expectedConcentrationExcluded = expectedConcentrationExcluded
        self.excludedFlag                  = excludedFlag
        if sum(unusableFeat) != 0:
            # featureMask size will be wrong, requires a reinitialisation
            self.initialiseMasks()

        ## Output and Log
        print('Values <LLOQ replaced by -inf')
        if not onlyLLOQ:
            print('Values >ULOQ replaced by +inf')
        if isinstance(calibration, dict):
            print('\n')

        # log the modifications
        if onlyLLOQ:
            logLimits = 'Limits of quantification applied to LLOQ'
        else:
            logLimits = 'Limits of quantification applied to LLOQ and ULOQ'
        if sum(untouched) != 0:
            logUntouchedFeatures = ' ' + str(sum(untouched)) + ' features only monitored and not processed: ' + str(untouchedFeatureMetadata.loc[:, 'Feature Name'].values.tolist()) + '.'
        else:
            logUntouchedFeatures = ''
        self.Attributes['Log'].append([datetime.now(), '%s (%i samples, %i features). LLOQ are replaced by -inf.%s' % (logLimits, self.noSamples, self.noFeatures, logUntouchedFeatures)])


    def _targetLynxApplyLimitsOfQuantificationNoiseFilled(self, onlyLLOQ=False, responseReference=None, **kwargs):
        """
        For each feature, replace intensity values inferior to the lowest limit of quantification or superior to the upper limit of quantification. Values inferior to the lowest limit of quantification are replaced by the feature noise concentration.

        Features missing the minimal required information are excluded from :py:attr:'featureMetadata', :py:attr:'intensityData', :py:attr:'expectedConcentration' and :py:attr:'calibration'. Features `'Monitored for relative information'` (and `'noCalibration'`) are not processed and returned without alterations.

        Calibration data should not be processed and therefore returned without modification.

        Units in :py:attr:`_intensityData`, :py:attr:`featureMetadata['LLOQ'] and :py:attr:`featureMetadata['ULOQ']` are expected to be identical for a given feature.

        .. Note:: To replace <LLOQ by the concentration equivalent to the noise level, the noise area, as well as the :math:`a` and :math:`b` parameters of the calibration equation must be known. For each feature, the ratio `(IS conc / IS Area)` defined as the responseFactor, is determined in a representative calibration sample. Then the concentration equivalent to the noise area is calculated, before being used to replace values <LLOQ.

        :param onlyLLOQ: if True only correct <LLOQ, if False correct <LLOQ and >ULOQ
        :type onlyLLOQ: bool
        :param responseReference: 'Sample File Name' of reference sample to use in order to establish the response to use, or list of samples to use (one per feature). If None, the middle of the calibration will be employed.
        :type responseReference: None or str or list
        :return: None
        :raises AttributeError: if :py:attr:`featureMetadata['LLOQ']` is missing
        :raises AttributeError: if :py:attr:`featureMetadata['ULOQ']` is missing and onlyLLOQ==False
        :raises AttributeError: if :py:attr:`featureMetadata['calibrationEquation']` is missing
        :raises AttributeError: if :py:attr:`featureMetadata['unitCorrectionFactor']` is missing
        :raises AttributeError: if :py:attr:`featureMetadata['Noise (area)']` is missing
        :raises AttributeError: if :py:attr:`featureMetadata['a']` is missing
        :raises AttributeError: if :py:attr:`featureMetadata['b']` is missing
        :raises AttributeError: if :py:attr:`calibration['calibPeakInfo']` is missing
        :raises ValueError: if :py:attr:`calibration['calibPeakInfo']['peakArea']` number of features or samples do not match the rest of if :py:attr:`calibration`
        :raises ValueError: if :py:attr:`calibration['calibPeakInfo']['peakResponse']` number of features or samples do not match the rest of if :py:attr:`calibration`
        :raises ValueError: if the 'responseReference' sample name is not recognised or the list is of erroneous length.
        :raises ValueError: if calculation using the calibrationEquation fails.
        """

        sampleMetadata        = copy.deepcopy(self.sampleMetadata)
        featureMetadata       = copy.deepcopy(self.featureMetadata)
        intensityData         = copy.deepcopy(self._intensityData)
        expectedConcentration = copy.deepcopy(self.expectedConcentration)
        calibration           = copy.deepcopy(self.calibration)
        if ((not hasattr(self, 'sampleMetadataExcluded')) | (not hasattr(self, 'featureMetadataExcluded')) | (not hasattr(self, 'intensityDataExcluded')) | (not hasattr(self, 'expectedConcentrationExcluded')) | (not hasattr(self, 'excludedFlag'))):
            sampleMetadataExcluded        = []
            featureMetadataExcluded       = []
            intensityDataExcluded         = []
            expectedConcentrationExcluded = []
            excludedFlag                  = []
        else:
            sampleMetadataExcluded        = copy.deepcopy(self.sampleMetadataExcluded)
            featureMetadataExcluded       = copy.deepcopy(self.featureMetadataExcluded)
            intensityDataExcluded         = copy.deepcopy(self.intensityDataExcluded)
            expectedConcentrationExcluded = copy.deepcopy(self.expectedConcentrationExcluded)
            excludedFlag                  = copy.deepcopy(self.excludedFlag)

        ## Check input columns
        if 'LLOQ' not in featureMetadata.columns:
            raise AttributeError('the featureMetadata[\'LLOQ\'] column is absent')
        if onlyLLOQ==False:
            if 'ULOQ' not in featureMetadata.columns:
                raise AttributeError('featureMetadata[\'ULOQ\'] column is absent')
        if 'calibrationEquation' not in featureMetadata.columns:
            raise AttributeError('the featureMetadata[\'calibrationEquation\'] column is absent')
        if 'unitCorrectionFactor' not in featureMetadata.columns:
            raise AttributeError('the featureMetadata[\'unitCorrectionFactor\'] column is absent')
        if 'Noise (area)' not in featureMetadata.columns:
            raise AttributeError('the featureMetadata[\'Noise (area)\'] column is absent')
        if 'a' not in featureMetadata.columns:
            raise AttributeError('the featureMetadata[\'a\'] column is absent')
        if 'b' not in featureMetadata.columns:
            raise AttributeError('the featureMetadata[\'b\'] column is absent')
        if 'calibPeakInfo' not in calibration.keys():
            raise AttributeError('the calibPeakInfo dict is absent from the calibration dict')
        if (not numpy.array_equal(calibration['calibPeakInfo']['peakArea'].index.values, calibration['calibSampleMetadata'].index.values)) | (not numpy.array_equal(calibration['calibPeakInfo']['peakArea'].columns.values, calibration['calibFeatureMetadata']['Feature Name'].values)):
            raise ValueError('calibration[\'calibPeakInfo\'][\'peakArea\'] number of features or samples do not match the rest of \'calibration\'')
        if (not numpy.array_equal(calibration['calibPeakInfo']['peakResponse'].index.values, calibration['calibSampleMetadata'].index.values)) | (not numpy.array_equal(calibration['calibPeakInfo']['peakResponse'].columns.values, calibration['calibFeatureMetadata']['Feature Name'].values)):
            raise ValueError('calibration[\'calibPeakInfo\'][\'peakResponse\'] number of features or samples do not match the rest of \'calibration\'')


        ## Features only Monitored are not processed and passed untouched (concatenated back at the end)
        untouched = (featureMetadata['quantificationType'] == QuantificationType.Monitored).values
        if sum(untouched) != 0:
            print('The following features are only monitored and therefore not processed: ' + str(featureMetadata.loc[untouched, 'Feature Name'].values.tolist()))
            untouchedFeatureMetadata = featureMetadata.loc[untouched, :]
            featureMetadata          = featureMetadata.loc[~untouched, :]
            untouchedIntensityData = intensityData[:, untouched]
            intensityData          = intensityData[:, ~untouched]
            untouchedExpectedConcentration = expectedConcentration.loc[:, untouched]
            expectedConcentration          = expectedConcentration.loc[:, ~untouched]
            # same reordering of the calibration
            untouchedCalibFeatureMetadata       = calibration['calibFeatureMetadata'].loc[untouched, :]
            calibration['calibFeatureMetadata'] = calibration['calibFeatureMetadata'].loc[~untouched, :]
            untouchedCalibIntensityData       = calibration['calibIntensityData'][:, untouched]
            calibration['calibIntensityData'] = calibration['calibIntensityData'][:, ~untouched]
            untouchedCalibExpectedConcentration       = calibration['calibExpectedConcentration'].loc[:, untouched]
            calibration['calibExpectedConcentration'] = calibration['calibExpectedConcentration'].loc[:, ~untouched]
            untouchedCalibPeakArea                   = calibration['calibPeakInfo']['peakArea'].loc[:, untouched]
            calibration['calibPeakInfo']['peakArea'] = calibration['calibPeakInfo']['peakArea'].loc[:, ~untouched]
            untouchedCalibPeakResponse                   = calibration['calibPeakInfo']['peakResponse'].loc[:, untouched]
            calibration['calibPeakInfo']['peakResponse'] = calibration['calibPeakInfo']['peakResponse'].loc[:, ~untouched]
            untouchedCalibPeakConcentrationDeviation                   = calibration['calibPeakInfo']['peakConcentrationDeviation'].loc[:, untouched]
            calibration['calibPeakInfo']['peakConcentrationDeviation'] = calibration['calibPeakInfo']['peakConcentrationDeviation'].loc[:, ~untouched]
            untouchedCalibPeakIntegrationFlag                   = calibration['calibPeakInfo']['peakIntegrationFlag'].loc[:, untouched]
            calibration['calibPeakInfo']['peakIntegrationFlag'] = calibration['calibPeakInfo']['peakIntegrationFlag'].loc[:, ~untouched]
            untouchedCalibPeakRT                   = calibration['calibPeakInfo']['peakRT'].loc[:, untouched]
            calibration['calibPeakInfo']['peakRT'] = calibration['calibPeakInfo']['peakRT'].loc[:, ~untouched]

        ## Exclude features without required information
        unusableFeat = featureMetadata['LLOQ'].isnull().values | featureMetadata['Noise (area)'].isnull() | featureMetadata['a'].isnull() | featureMetadata['b'].isnull()
        if not onlyLLOQ:
            unusableFeat = unusableFeat | featureMetadata['ULOQ'].isnull().values
        unusableFeat = unusableFeat.values
        if sum(unusableFeat) != 0:
            print(str(sum(unusableFeat)) + ' features cannot be pre-processed:')
            print('\t' + str(sum(unusableFeat)) + ' features lack the required information to replace limits of quantification by noise level')
            # store
            sampleMetadataExcluded.append(sampleMetadata)
            featureMetadataExcluded.append(featureMetadata.loc[unusableFeat, :])
            intensityDataExcluded.append(intensityData[:, unusableFeat])
            #return(expectedConcentration, unusableFeat)

            expectedConcentrationExcluded.append(expectedConcentration.loc[:, unusableFeat])
            excludedFlag.append('Features')
            #remove
            featureMetadata = featureMetadata.loc[~unusableFeat, :]
            intensityData = intensityData[:, ~unusableFeat]
            expectedConcentration = expectedConcentration.loc[:, ~unusableFeat]
            calibration['calibFeatureMetadata'] = calibration['calibFeatureMetadata'].loc[~unusableFeat, :]
            calibration['calibIntensityData'] = calibration['calibIntensityData'][:, ~unusableFeat]
            calibration['calibExpectedConcentration'] = calibration['calibExpectedConcentration'].loc[:, ~unusableFeat]
            calibration['calibPeakInfo']['peakResponse'] = calibration['calibPeakInfo']['peakResponse'].loc[:, ~unusableFeat]
            calibration['calibPeakInfo']['peakArea'] = calibration['calibPeakInfo']['peakArea'].loc[:, ~unusableFeat]
            calibration['calibPeakInfo']['peakConcentrationDeviation'] = calibration['calibPeakInfo']['peakConcentrationDeviation'].loc[:, ~unusableFeat]
            calibration['calibPeakInfo']['peakIntegrationFlag'] = calibration['calibPeakInfo']['peakIntegrationFlag'].loc[:, ~unusableFeat]
            calibration['calibPeakInfo']['peakRT'] = calibration['calibPeakInfo']['peakRT'].loc[:, ~unusableFeat]


        ## Calculate each feature's replacement noise concentration
        ##
        ## Approximate the response reference
        ## Needed for calibrationMethod='backcalculatedIS', for 'noIS' responseFactor=1
        # responseReference: None (guessed middle of the curve), 'Sample File Name' to use, or list of 'Sample File Name' (one per feature)
        #
        # ! The calibration curve is plotted in TargetLynx as x-axis concentration, y-axis response
        # The calibration equation obtained is written as: response = a * concentration + b (eq. 1)
        # The response uses the area measured and IS: response = Area * (IS conc / IS Area) (eq. 2) [for 'noIS' response = Area]
        # We can define the responseFactor = (IS conc/IS Area), the ratio of IS Conc/IS Area that can changes from sample to sample.
        # For noise concentration calculation, using eq. 2 and a reference sample,we can approximate responseFactor = response/area [works for both calibrationMethod]

        # make a list of responseReference (one per feature)
        if isinstance(responseReference, str):
            # Check existance of this sample
            if sum(calibration['calibSampleMetadata']['Sample File Name'] == responseReference) == 0:
                raise ValueError('responseReference \'Sample File Name\' unknown: ' + str(responseReference))
            responseReference = [responseReference] * featureMetadata.shape[0]
        elif isinstance(responseReference, list):
            # Check length to match the number of features
            if len(responseReference) != featureMetadata.shape[0]:
                raise ValueError('The number of responseReference \'Sample File Name\' provided does not match the number of features to process:\n' + str(featureMetadata['Feature Name'].values))
            for i in responseReference:
                if sum(calibration['calibSampleMetadata']['Sample File Name'] == i) == 0:
                    raise ValueError('ResponseReference \'Sample File Name\' unknown: ' + str(i))
        elif responseReference is None:
            # Get a compound in the middle of the calibration run, use to your own risks
            responseReference = calibration['calibSampleMetadata'].sort_values(by='Run Order').iloc[int(numpy.ceil(calibration['calibSampleMetadata'].shape[0] / 2)) - 1]['Sample File Name']  # round to the highest value
            warnings.warn('No responseReference provided, sample in the middle of the calibration run employed: ' + str(responseReference))
            responseReference = [responseReference] * featureMetadata.shape[0]
        else:
            raise ValueError('The responseReference provided is not recognised. A \'Sample File Name\', a list of \'Sample File Name\' or None are expected')

        # Get the right Area and Response for each feature
        tmpArea = list()
        tmpResponse = list()
        # iterate over features, get value in responseReference spectra
        for i in range(0, featureMetadata.shape[0]):
            tmpArea.append(calibration['calibPeakInfo']['peakArea'][(calibration['calibSampleMetadata']['Sample File Name'] == responseReference[i]).values].values.flatten()[i])
            tmpResponse.append(calibration['calibPeakInfo']['peakResponse'][(calibration['calibSampleMetadata']['Sample File Name'] == responseReference[i]).values].values.flatten()[i])
        # responseFactor = response/Area
        # Note: responseFactor will be ~equal for all compound sharing the same IS (as ISconc/ISArea will be identical)
        resFact = [resp / area for resp, area in zip(tmpResponse, tmpArea)]
        featureMetadata = featureMetadata.assign(responseFactor=resFact)


        ## Calculate noise concentration equivalent for each feature
        ## Note for equation in .json:
        #   calibration curve in TargetLynx is defined/established as: response = a * concentration + b (eq. 1)
        #   response is defined as: response = Area * (IS conc / IS Area) (eq. 2) [for 'noIS' response = Area]
        #   using eq. 2, we can approximate the ratio IS Conc/IS Area in a representative sample as: responseFactor = response / area (eq. 3)
        #   Therefore: concentration = ((area*responseFactor) - b) / a (eq. 4)
        #
        #   If in TargetLynx 'axis transformation' is set to log ( but still use 'Polynomial Type'=linear and 'Fit Weighting'=None)
        #   eq.1 is changed to: log(response) = a * log(concentration) + b (eq. 5)
        #   and eq. 4 changed to: concentration = 10^( (log(area*responseFactor) - b) / a ) (eq. 5)
        # The equation filled expect the following variables:
        #   area
        #   responseFactor | responseFactor=(IS conc/IS Area)=response/Area, for noIS, responseFactor will be 1.
        #   a
        #   b
        #
        # Examples:
        # '((area * responseFactor)-b)/a'
        # '10**((numpy.log10(area * responseFactor)-b)/a)'
        # 'area/a' | if b not needed, set to 0 in csv [use for linear noIS, area=response, responseFactor=1, and response = a * concentration ]

        tmpNoiseConc = []
        for i in range(0, featureMetadata.shape[0]):
            # set the right values before applying the equation
            calibrationEquation = featureMetadata['calibrationEquation'].values[i]
            area = featureMetadata['Noise (area)'].values[i]
            responseFactor = featureMetadata['responseFactor'].values[i]
            a = featureMetadata['a'].values[i]
            b = featureMetadata['b'].values[i]

            # apply the calibration equation, and the unitCorrectionFactor, as the equations were established with the original area/response/concentrations
            try:
                tmpNoiseConc.append(eval(calibrationEquation) * featureMetadata['unitCorrectionFactor'].values[i])
            except:
                raise ValueError('Verify calibrationEquation: \"' + calibrationEquation + '\", only variables expected are \"area\", \"responseFactor\", \"a\" or \"b\"')
        featureMetadata = featureMetadata.assign(noiseConcentration=tmpNoiseConc)


        ## Values replacement by noise concentration (<LOQ) and +inf for (>ULOQ)
        # iterate over the features
        for i in range(0, featureMetadata.shape[0]):
            # LLOQ
            toReplaceLLOQ = intensityData[:, i] < featureMetadata['LLOQ'].values[i]
            intensityData[toReplaceLLOQ, i] = featureMetadata['noiseConcentration'].values[i]

            # ULOQ
            if not onlyLLOQ:
                toReplaceULOQ = intensityData[:, i] > featureMetadata['ULOQ'].values[i]
                intensityData[toReplaceULOQ, i] = numpy.inf


        ## Add back the untouched monitored features
        if sum(untouched) != 0:
            featureMetadata = pandas.concat([featureMetadata, untouchedFeatureMetadata], axis=0, sort=False)
            intensityData = numpy.concatenate((intensityData, untouchedIntensityData), axis=1)
            expectedConcentration = pandas.concat([expectedConcentration, untouchedExpectedConcentration], axis=1, sort=False)
            # reorder the calib
            calibration['calibFeatureMetadata'] = pandas.concat([calibration['calibFeatureMetadata'], untouchedCalibFeatureMetadata], axis=0, sort=False)
            calibration['calibIntensityData'] = numpy.concatenate((calibration['calibIntensityData'], untouchedCalibIntensityData), axis=1)
            calibration['calibExpectedConcentration'] = pandas.concat([calibration['calibExpectedConcentration'], untouchedCalibExpectedConcentration], axis=1, sort=False)
            calibration['calibPeakInfo']['peakArea'] = pandas.concat([calibration['calibPeakInfo']['peakArea'], untouchedCalibPeakArea], axis=1, sort=False)
            calibration['calibPeakInfo']['peakResponse'] = pandas.concat([calibration['calibPeakInfo']['peakResponse'], untouchedCalibPeakResponse], axis=1, sort=False)
            calibration['calibPeakInfo']['peakConcentrationDeviation'] = pandas.concat([calibration['calibPeakInfo']['peakConcentrationDeviation'], untouchedCalibPeakConcentrationDeviation], axis=1, sort=False)
            calibration['calibPeakInfo']['peakIntegrationFlag'] = pandas.concat([calibration['calibPeakInfo']['peakIntegrationFlag'], untouchedCalibPeakIntegrationFlag], axis=1, sort=False)
            calibration['calibPeakInfo']['peakRT'] = pandas.concat([calibration['calibPeakInfo']['peakRT'], untouchedCalibPeakRT], axis=1, sort=False)

        # Remove excess info
        featureMetadata.reset_index(drop=True, inplace=True)
        calibration['calibFeatureMetadata'].reset_index(drop=True, inplace=True)
        expectedConcentration.reset_index(drop=True, inplace=True)
        calibration['calibExpectedConcentration'].reset_index(drop=True, inplace=True)
        calibration['calibPeakInfo']['peakArea'] .reset_index(drop=True, inplace=True)
        calibration['calibPeakInfo']['peakResponse'].reset_index(drop=True, inplace=True)
        calibration['calibPeakInfo']['peakConcentrationDeviation'].reset_index(drop=True, inplace=True)
        calibration['calibPeakInfo']['peakIntegrationFlag'].reset_index(drop=True, inplace=True)
        calibration['calibPeakInfo']['peakRT'].reset_index(drop=True, inplace=True)

        ## return dataset with limits of quantification applied
        self.featureMetadata       = featureMetadata
        self._intensityData        = intensityData
        self.expectedConcentration = expectedConcentration
        self.calibration           = calibration
        self.sampleMetadataExcluded        = sampleMetadataExcluded
        self.featureMetadataExcluded       = featureMetadataExcluded
        self.intensityDataExcluded         = intensityDataExcluded
        self.expectedConcentrationExcluded = expectedConcentrationExcluded
        self.excludedFlag                  = excludedFlag

        ## Output and Log
        print('Values <LLOQ replaced by the noise concentration')
        if not onlyLLOQ:
            print('Values >ULOQ replaced by +inf')
        print('\n')

        # log the modifications
        if onlyLLOQ:
            logLimits = 'Limits of quantification applied to LLOQ'
        else:
            logLimits = 'Limits of quantification applied to LLOQ and ULOQ'
        if sum(untouched) != 0:
            logUntouchedFeatures = ' ' + str(sum(untouched)) + ' features only monitored and not processed: ' + str(untouchedFeatureMetadata.loc[:, 'Feature Name'].values.tolist()) + '.'
        else:
            logUntouchedFeatures = ''
        self.Attributes['Log'].append([datetime.now(), '%s (%i samples, %i features). LLOQ are replaced by the noise concentration.%s' % (logLimits, self.noSamples, self.noFeatures, logUntouchedFeatures)])


    def __add__(self,other):
        """
        Implements the concatenation of 2 :py:class:`TargetedDataset`
        
        `targetedDataset = targetedDatasetBatch1 + targetedDatasetBatch2`
        
        `targetedDataset = sum([targetedDatasetBatch1, targetedDatasetBatch2`, targetedDatasetBatch3])'
        
        :py:attr:`sampleMetadata` are concatenated, :py:attr:`featureMetadata` are merged and :py:attr:`intensityData` match it.
        In :py:attr:`featureMetadata`, non pre-defined columns names get the suffix '_batchX' appended.
        Excluded features and samples are listed in the same order as the 'Batch'.
        Calibration are listed in the same order as batch. Features are not modified inside the calibration (can have more features in calibFeatureMetadata than in self.featureMetadata)
        FeatureMetadata columns listed in Attribute['additionalQuantParamColumns'] are expected to be identical across all batch (if present), and added to the merge columns.

        :raises ValueError: if the targeted methods employed differ
        :raises ValueError: if an object doesn't pass validation before merge
        :raises ValueError: if the merge object doesn't pass validation
        :raises Warning: to update LOQ using :py:meth:`~TargetedDataset.mergeLimitsOfQuantification`
        """

        import collections
        import warnings
        def flatten(x):
            """ Always provide a single level list, from a list of lists and or str """
            result = []
            for el in x:
                if isinstance(x, collections.Iterable) and not (isinstance(el, str)|isinstance(el, dict)):
                    result.extend(flatten(el))
                else:
                    result.append(el)
            return result

        def reNumber(oldSeries, startNb):
            """ reindex a series of int between the startNB and startNb + number of unique values """
            oldNb = oldSeries.unique().tolist()
            newNb = numpy.arange(startNb, startNb + len(oldNb)).tolist()
            newSeries = pandas.Series(numpy.repeat(numpy.nan, oldSeries.shape[0]))
            for i in range(0, len(oldNb)):
                newSeries[oldSeries == oldNb[i]] = newNb[i]
            changes = dict(zip(oldNb, newNb))
            return newSeries.astype('int64'), changes

        def batchListReNumber(oldList, numberChange, untouchedValues):
            """
            Rename list values; if no '_batchX' suffix present in list value, append one (the lowest key in numberChange).
            Then scan all '_batchX' suffix and update X to all members following numberChange.

            :param list oldList: values to which append and update _batchX
            :param dict numberChanges: dict with old batch number as keys and new batch number as values
            :param list untouchedValues: list of values to leave untouched
            :return: list with appended/updated batch values
            """
            import re

            newList = copy.deepcopy(oldList)

            ## Append'_batchX' with X the smallest original 'Batch' if none already present
            for i in range(len(newList)):
                if (newList[i] not in untouchedValues) & (newList[i].find('_batch') == -1):
                    newList[i] = newList[i] + '_batch' + str(min(numberChange.keys()))

            ## Update X in all '_batchX' column names (look for previous batch numbers and replace)
            for batchNum in numberChange.keys():
                # exact match with end of string ($)
                query = '.+?(?=_batch' + str(batchNum) + '$)'
                for j in range(len(newList)):
                    if newList[j] not in untouchedValues:
                        # string without _batchX
                        searchRes = re.search(query, newList[j])  # if no match returns None
                        if searchRes:
                            newList[j] = searchRes.group() + '_batch' + str(numberChange[batchNum])

            return newList

        def concatenateList(list1, list2):
            """
            Concatenate two lists, always return a list of list
            """
            outputList = []

            ## list1
            # if it's an empty list
            if len(list1) == 0:
                outputList.append(list1)
            # if it's already a list of list (from previous __add__)
            elif isinstance(list1[0], list):
                for i in range(len(list1)):
                    outputList.append(list1[i])
            # first use of __add__, not a list of list
            else:
                outputList.append(list1)

            ## list2
            # if it's an empty list
            if len(list2) == 0:
                outputList.append(list2)
            # if it's already a list of list (from previous __add__)
            elif isinstance(list2[0], list):
                for i in range(len(list2)):
                    outputList.append(list2[i])
            # first use of __add__, not a list of list
            else:
                outputList.append(list2)

            return outputList

        def updatecalibBatch(calib, batchChange):
            """
            change batch number inside each calibration['calibSampleMetadata']

            :param calib: calibration or list of calibration
            :param batchChange: dict of batch number changes
            :return: updated calibration
            """

            if isinstance(calib, list):
                updatedcalib = list()
                for j in range(len(calib)):
                    # all the same
                    updatedcalib.append(calib[j])
                    # modify batch number
                    for batchNum in batchChange.keys():
                        updatedcalib[j]['calibSampleMetadata'].loc[calib[j]['calibSampleMetadata']['Batch'] == batchNum, 'Batch'] = batchChange[batchNum]

            elif isinstance(calib, dict):
                updatedcalib = copy.deepcopy(calib)
                # modify batch number
                for batchNum in batchChange.keys():
                    updatedcalib['calibSampleMetadata'].loc[calib['calibSampleMetadata']['Batch'] == batchNum, 'Batch'] = batchChange[batchNum]

            return updatedcalib


        ## Input checks
        # Run validator (checks for duplicates in featureMetadata['Feature Name']). No check for AssayRole and SampleType as sample info data might not have been imported yet
        validSelfDataset  = self.validateObject(verbose=False, raiseError=False, raiseWarning=False)
        validOtherDataset = other.validateObject(verbose=False, raiseError=False, raiseWarning=False)
        if not validSelfDataset['BasicTargetedDataset']:
            raise ValueError('self does not satisfy to the Basic TargetedDataset definition, check with self.validateObject(verbose=True, raiseError=False)')
        if not validOtherDataset['BasicTargetedDataset']:
            raise ValueError('other does not satisfy to the Basic TargetedDataset definition, check with other.validateObject(verbose=True, raiseError=False)')
        # Warning if duplicate 'Sample File Name' in sampleMetadata
        u_ids, u_counts = numpy.unique(pandas.concat([self.sampleMetadata['Sample File Name'], other.sampleMetadata['Sample File Name']],ignore_index=True, sort=False), return_counts=True)
        if any(u_counts > 1):
            warnings.warn('Warning: The following \'Sample File Name\' are present more than once: ' + str(u_ids[u_counts>1].tolist()))

        if self.AnalyticalPlatform != other.AnalyticalPlatform:
            raise ValueError('Can only add Targeted datasets with the same AnalyticalPlatform Attribute')

        ## Initialise an empty TargetedDataset to overwrite
        targetedData = TargetedDataset(datapath='', fileType='empty')


        ## Attributes
        if self.Attributes['methodName'] != other.Attributes['methodName']:
            raise ValueError('Cannot concatenate different targeted methods: \''+ self.Attributes['methodName'] +'\' and \''+ other.Attributes['methodName'] +'\'')
        # copy from the first (mainly dataset parameters, methodName, chromatography and ionisation)
        targetedData.Attributes = copy.deepcopy(self.Attributes)
        # append both logs
        targetedData.Attributes['Log'] = self.Attributes['Log'] + other.Attributes['Log']

        ## _Normalisation
        targetedData._Normalisation = normalisation.NullNormaliser()

        ## VariableType
        targetedData.VariableType = copy.deepcopy(self.VariableType)

        targetedData.AnalyticalPlatform = copy.deepcopy(self.AnalyticalPlatform)

        ## _name
        targetedData.name = self.name+'-'+other.name

        ## fileName
        targetedData.fileName = flatten([self.fileName, other.fileName])

        ## filePath
        targetedData.filePath = flatten([self.filePath, other.filePath])

        ## sampleMetadata
        tmpSampleMetadata1 = copy.deepcopy(self.sampleMetadata)
        tmpSampleMetadata2 = copy.deepcopy(other.sampleMetadata)
        # reindex the 'Batch' value across both targetedDataset (self starts at 1, other at max(self)+1)
        tmpSampleMetadata1['Batch'], batchChangeSelf  = reNumber(tmpSampleMetadata1['Batch'], 1)
        tmpSampleMetadata2['Batch'], batchChangeOther = reNumber(tmpSampleMetadata2['Batch'], tmpSampleMetadata1['Batch'].values.max()+1)
        # Concatenate samples and reinitialise index
        sampleMetadata = pandas.concat([tmpSampleMetadata1, tmpSampleMetadata2], ignore_index=True, sort=False)
        # Update Run Order
        sampleMetadata['Order'] = sampleMetadata.sort_values(by='Acquired Time').index
        sampleMetadata['Run Order'] = sampleMetadata.sort_values(by='Order').index
        sampleMetadata.drop('Order', axis=1, inplace=True)
        # new sampleMetadata
        targetedData.sampleMetadata = copy.deepcopy(sampleMetadata)


        ## featureMetadata
        ## Merge feature list on the common columns imposed by the targeted SOP employed.
        # All other columns have a '_batchX' suffix amended for traceability. (use the min original 'Batch' for that targetedDataset)
        # From that point onward no variable should exist without a '_batchX'
        # Apply to '_batchX' the batchChangeSelf and batchChangeOther to align it with the 'Batch'
        mergeCol = ['Feature Name', 'calibrationMethod', 'quantificationType', 'Unit']
        mergeCol.extend(self.Attributes['externalID'])
        # additionalQuantParamColumns if present are expected to be identical across batch
        if 'additionalQuantParamColumns' in targetedData.Attributes.keys():
            for col in targetedData.Attributes['additionalQuantParamColumns']:
                if (col in self.featureMetadata.columns) and (col in other.featureMetadata.columns) and (col not in mergeCol):
                    mergeCol.append(col)
        # take each dataset featureMetadata column names, modify them and rename columns
        tmpFeatureMetadata1 = copy.deepcopy(self.featureMetadata)
        updatedCol1 = batchListReNumber(tmpFeatureMetadata1.columns.tolist(), batchChangeSelf, mergeCol)
        tmpFeatureMetadata1.columns = updatedCol1
        tmpFeatureMetadata2 = copy.deepcopy(other.featureMetadata)
        updatedCol2 = batchListReNumber(tmpFeatureMetadata2.columns.tolist(), batchChangeOther, mergeCol)
        tmpFeatureMetadata2.columns = updatedCol2
        # Merge featureMetadata on the mergeCol, no columns with identical name exist
        tmpFeatureMetadata = tmpFeatureMetadata1.merge(tmpFeatureMetadata2, how='outer', on=mergeCol, left_on=None,right_on=None,left_index=False,right_index=False,sort=False,copy=True,indicator=False)
        targetedData.featureMetadata = copy.deepcopy(tmpFeatureMetadata)

        ## featureMetadataNotExported
        # add _batchX to the column names to exclude. The expected columns are 'mergeCol' from featureMetadata. No modification for sampleMetadataNotExported which has been copied with the other Attributes (and is an SOP parameter)
        notExportedSelf  = batchListReNumber(self.Attributes['featureMetadataNotExported'],  batchChangeSelf, mergeCol)
        notExportedOther = batchListReNumber(other.Attributes['featureMetadataNotExported'], batchChangeOther, mergeCol)
        targetedData.Attributes['featureMetadataNotExported'] = list(set().union(notExportedSelf, notExportedOther))


        ## _intensityData
        # samples are simply concatenated, but features are merged. Reproject each dataset on the merge feature list before concatenation.
        # init with nan
        intensityData1 = numpy.full([self._intensityData.shape[0], targetedData.featureMetadata.shape[0]], numpy.nan)
        intensityData2 = numpy.full([other._intensityData.shape[0], targetedData.featureMetadata.shape[0]], numpy.nan)
        # iterate over the merged features
        for i in range(targetedData.featureMetadata.shape[0]):
            featureName = targetedData.featureMetadata.loc[i,'Feature Name']
            featurePosition1 = self.featureMetadata['Feature Name'] == featureName
            featurePosition2 = other.featureMetadata['Feature Name'] == featureName
            if sum(featurePosition1)==1:
                intensityData1[:,i] = self._intensityData[:,featurePosition1].ravel()
            elif sum(featurePosition1)>1:
                raise ValueError('Duplicate feature name in first input: ' + featureName)
            if sum(featurePosition2)==1:
                intensityData2[:, i] = other._intensityData[:, featurePosition2].ravel()
            elif sum(featurePosition2) > 1:
                raise ValueError('Duplicate feature name in second input: ' + featureName)
        intensityData = numpy.concatenate([intensityData1,intensityData2], axis=0)
        targetedData._intensityData = copy.deepcopy(intensityData)


        ## expectedConcentration
        # same approach as _intensityData, samples are concatenated but features are merged. validObject() on input ensures expectedConcentration.columns match featureMetadata['Feature Name']
        expectedConc1 = pandas.DataFrame(numpy.full([self.expectedConcentration.shape[0], targetedData.featureMetadata.shape[0]], numpy.nan),  columns=targetedData.featureMetadata['Feature Name'].tolist())
        expectedConc2 = pandas.DataFrame(numpy.full([other.expectedConcentration.shape[0], targetedData.featureMetadata.shape[0]], numpy.nan), columns=targetedData.featureMetadata['Feature Name'].tolist())
        # iterate over the merged features
        for colname in targetedData.featureMetadata['Feature Name'].tolist():
            if colname in self.expectedConcentration.columns:
                expectedConc1.loc[:,colname] = self.expectedConcentration[colname].ravel()
            if colname in other.expectedConcentration.columns:
                expectedConc2.loc[:,colname] = other.expectedConcentration[colname].ravel()
        expectedConcentration = pandas.concat([expectedConc1, expectedConc2], axis=0, ignore_index=True, sort=False)
        expectedConcentration.reset_index(drop=True, inplace=True)
        targetedData.expectedConcentration = copy.deepcopy(expectedConcentration)


        ## Masks
        targetedData.initialiseMasks()
        # sampleMask
        targetedData.sampleMask = numpy.concatenate([self.sampleMask, other.sampleMask], axis=0)
        # featureMask
        # if featureMask agree in both, keep that value. Otherwise let the default True value. If feature exist only in one, use that value.
        if (sum(~self.featureMask)!=0) | (sum(~other.featureMask)!=0):
            warnings.warn("Warning: featureMask are not empty, they will be merged. If both featureMasks do not agree, the default \'True\' value will be set. If the feature is only present in one dataset, the corresponding featureMask value will be kept.")
        for i in range(targetedData.featureMetadata.shape[0]):
            featureName = targetedData.featureMetadata.loc[i, 'Feature Name']
            featurePosition1 = self.featureMetadata['Feature Name'] == featureName
            featurePosition2 = other.featureMetadata['Feature Name'] == featureName
            # if both exist
            if (sum(featurePosition1)==1) & (sum(featurePosition2)==1):
                # only False if both are False (otherwise True, same as default)
                targetedData.featureMask[i] = self.featureMask[featurePosition1] | other.featureMask[featurePosition2]
            # if feature only exist in first input
            elif sum(featurePosition1==1):
                targetedData.featureMask[i] = self.featureMask[featurePosition1]
            # if feature only exist in second input
            elif sum(featurePosition2==1):
                targetedData.featureMask[i] = other.featureMask[featurePosition2]


        ## Excluded data with applyMask()
        # attribute doesn't exist the first time. From one round of __add__ onward the attribute is created and the length matches the number and order of 'Batch'
        if hasattr(self, 'sampleMetadataExcluded') & hasattr(other, 'sampleMetadataExcluded'):
            targetedData.sampleMetadataExcluded = concatenateList(self.sampleMetadataExcluded, other.sampleMetadataExcluded)
            targetedData.featureMetadataExcluded = concatenateList(self.featureMetadataExcluded, other.featureMetadataExcluded)
            targetedData.intensityDataExcluded = concatenateList(self.intensityDataExcluded, other.intensityDataExcluded)
            targetedData.expectedConcentrationExcluded = concatenateList(self.expectedConcentrationExcluded, other.expectedConcentrationExcluded)
            targetedData.excludedFlag = concatenateList(self.excludedFlag, other.excludedFlag)
            # add expectedConcentrationExcluded here too!
        elif hasattr(self, 'sampleMetadataExcluded'):
            targetedData.sampleMetadataExcluded = concatenateList(self.sampleMetadataExcluded, [])
            targetedData.featureMetadataExcluded = concatenateList(self.featureMetadataExcluded, [])
            targetedData.intensityDataExcluded = concatenateList(self.intensityDataExcluded, [])
            targetedData.expectedConcentrationExcluded = concatenateList(self.expectedConcentrationExcluded, [])
            targetedData.excludedFlag = concatenateList(self.excludedFlag, [])
        elif hasattr(other, 'sampleMetadataExcluded'):
            targetedData.sampleMetadataExcluded = concatenateList([], other.sampleMetadataExcluded)
            targetedData.featureMetadataExcluded = concatenateList([], other.featureMetadataExcluded)
            targetedData.intensityDataExcluded = concatenateList([], other.intensityDataExcluded)
            targetedData.expectedConcentrationExcluded = concatenateList([], other.expectedConcentrationExcluded)
            targetedData.excludedFlag = concatenateList([], other.excludedFlag)
        else:
            targetedData.sampleMetadataExcluded = concatenateList([], [])
            targetedData.featureMetadataExcluded = concatenateList([], [])
            targetedData.intensityDataExcluded = concatenateList([], [])
            targetedData.expectedConcentrationExcluded = concatenateList([], [])
            targetedData.excludedFlag = concatenateList([], [])


        ## calibration
        # change batch number inside each calibration['calibSampleMetadata']
        tmpCalibSelf  = copy.deepcopy(self.calibration)
        tmpCalibSelf  = updatecalibBatch(tmpCalibSelf, batchChangeSelf)
        tmpCalibOther = copy.deepcopy(other.calibration)
        tmpCalibOther = updatecalibBatch(tmpCalibOther, batchChangeOther)
        targetedData.calibration = flatten([tmpCalibSelf, tmpCalibOther])


        ## unexpected attributes
        expectedAttr = {'Attributes', 'VariableType', 'AnalyticalPlatform', '_Normalisation', '_name', 'fileName', 'filePath',
                        '_intensityData', 'sampleMetadata', 'featureMetadata', 'expectedConcentration','sampleMask',
                        'featureMask', 'calibration', 'sampleMetadataExcluded', 'intensityDataExcluded',
                        'featureMetadataExcluded', 'expectedConcentrationExcluded', 'excludedFlag'}
        selfAttr = set(self.__dict__.keys())
        selfAdditional = selfAttr - expectedAttr
        otherAttr = set(other.__dict__.keys())
        otherAdditional = otherAttr - expectedAttr
        # identify common and unique
        commonAttr = selfAdditional.intersection(otherAdditional)
        onlySelfAttr = selfAdditional - commonAttr
        onlyOtherAttr = otherAdditional - commonAttr
        # save a list [self, other] for each attribute
        if bool(commonAttr):
            print('The following additional attributes are present in both datasets and stored as lists:')
            print('\t' + str(commonAttr))
            for k in commonAttr:
                setattr(targetedData, k, [getattr(self,k), getattr(other,k)])
        if bool(onlySelfAttr):
            print('The following additional attributes are only present in the first dataset and stored as lists:')
            print('\t' + str(onlySelfAttr))
            for l in onlySelfAttr:
                setattr(targetedData, l, [getattr(self, l), None])
        if bool(onlyOtherAttr):
            print('The following additional attributes are only present in the second dataset and stored as lists:')
            print('\t' + str(onlyOtherAttr))
            for m in onlyOtherAttr:
                setattr(targetedData, m, [None, getattr(other, m)])

        ## run validation on the merged dataset
        validMergedDataset = targetedData.validateObject(verbose=False, raiseError=False, raiseWarning=False)
        if not validMergedDataset['BasicTargetedDataset']:
            raise ValueError('The merged dataset does not satisfy to the Basic TargetedDataset definition')

        ## Log
        targetedData.Attributes['Log'].append([datetime.now(), 'Concatenated datasets %s (%i samples and %i features) and %s (%i samples and %i features), to a dataset of %i samples and %i features.' % (self.name, self.noSamples, self.noFeatures, other.name, other.noSamples, other.noFeatures, targetedData.noSamples, targetedData.noFeatures)])
        print('Concatenated datasets %s (%i samples and %i features) and %s (%i samples and %i features), to a dataset of %i samples and %i features.' % (self.name, self.noSamples, self.noFeatures, other.name, other.noSamples, other.noFeatures, targetedData.noSamples, targetedData.noFeatures))

        ## Remind to mergeLimitsOfQuantification
        warnings.warn('Update the limits of quantification using `mergedDataset.mergeLimitsOfQuantification()` (keeps the lowest common denominator across all batch: highest LLOQ, lowest ULOQ)')

        return targetedData


    def __radd__(self, other):
        """
        Implements the summation of multiple :py:class:`TargetedDataset`
        
        `targetedDataset = sum([ targetedDatasetBatch1, targetedDatasetBatch2, targetedDatasetBatch3 ])`
        
        ..Note:: Sum always starts by the 0 integer and does `0.__add__(targetedDatasetBatch1)` which fails and then calls the reverse add method `targetedDatasetBatch1.__radd_(0)`
        """
        if other == 0:
            return self
        else:
            return self.__add__(other)


    def mergeLimitsOfQuantification(self, keepBatchLOQ=False, onlyLLOQ=False):
        """
        Update limits of quantification and apply LLOQ/ULOQ using the lowest common denominator across all batch (after a :py:meth:`~TargetedDataset.__add__`). Keep the highest LLOQ and lowest ULOQ.

        :param bool keepBatchLOQ: If ``True`` do not remove each batch LOQ (:py:attr:`featureMetadata['LLOQ_batchX']`, :py:attr:`featureMetadata['ULOQ_batchX']`)
        :param bool onlyLLOQ: if True only correct <LLOQ, if False correct <LLOQ and >ULOQ
        :raises ValueError: if targetedData does not satisfy to the BasicTargetedDataset definition on input
        :raises ValueError: if number of batch, LLOQ_batchX and ULOQ_batchX do not match
        :raises ValueError: if targetedData does not satisfy to the BasicTargetedDataset definition after LOQ merging
        :raises Warning: if :py:attr:`featureMetadata['LLOQ']` or :py:attr:`featureMetadata['ULOQ']` already exist and will be overwritten.
        """

        # Check dataset is fit for merging LOQ
        validateDataset = copy.deepcopy(self)
        validDataset = validateDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False)
        if not validDataset['BasicTargetedDataset']:
            raise ValueError('Import Error: targetedData does not satisfy to the BasicTargetedDataset definition')
        # find XLOQ_batchX, get batch ID, check agreement
        col_LLOQ = self.featureMetadata.columns[self.featureMetadata.columns.to_series().str.contains('LLOQ_batch')].tolist()
        col_LLOQ_batch = sorted([int(i.replace('LLOQ_batch', '')) for i in col_LLOQ])
        col_ULOQ = self.featureMetadata.columns[self.featureMetadata.columns.to_series().str.contains('ULOQ_batch')].tolist()
        col_ULOQ_batch = sorted([int(i.replace('ULOQ_batch', '')) for i in col_ULOQ])
        batches = sorted((numpy.unique(self.sampleMetadata.loc[:, 'Batch'].values[~numpy.isnan(self.sampleMetadata.loc[:, 'Batch'].values)])).astype(int))
        if (col_LLOQ_batch != batches) | (col_ULOQ_batch != batches):
            raise ValueError('Import Error: targetedData does not have the same number of batch, LLOQ_batchX and ULOQ_batchX: ' + str(batches) + ', ' + str(col_LLOQ) + ', ' + str(col_ULOQ) + '. LOQs must have already been merged!')

        # New LOQ
        common_LLOQ = self.featureMetadata[col_LLOQ].max(axis=1, skipna=False)
        common_ULOQ = self.featureMetadata[col_ULOQ].min(axis=1, skipna=False)
        if ('LLOQ' in self.featureMetadata.columns) | ('ULOQ' in self.featureMetadata.columns):
            warnings.warn('Previous featureMetadata[\'LLOQ\'] and [\'ULOQ\'] values will be overwritten.')
        self.featureMetadata['LLOQ'] = common_LLOQ
        self.featureMetadata['ULOQ'] = common_ULOQ

        # Remove old batch LOQ columns
        if not keepBatchLOQ:
            self.featureMetadata.drop(col_LLOQ, inplace=True, axis=1)
            self.featureMetadata.drop(col_ULOQ, inplace=True, axis=1)

        # _applyLimitsOfQuantification
        self._applyLimitsOfQuantification(onlyLLOQ=onlyLLOQ)

        # run validation on the merged LOQ
        validateMergeDataset = copy.deepcopy(self)
        validMergedDataset = validateMergeDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False)
        if not validMergedDataset['BasicTargetedDataset']:
            raise ValueError('The merged LOQ dataset does not satisfy to the Basic TargetedDataset definition')

        # Log
        self.Attributes['Log'].append([datetime.now(), 'LOQ merged (keepBatchLOQ =  %s, onlyLLOQ = %s).' % (keepBatchLOQ, onlyLLOQ)])
        if onlyLLOQ:
            print('Limits of quantification merged to the highest LLOQ across batch')
        else:
            print('Limits of quantification merged to the highest LLOQ and lowest ULOQ across batch')


    def exportDataset(self, destinationPath='.', saveFormat='CSV', withExclusions=True, escapeDelimiters=False, filterMetadata=True):
        """
        Calls :py:meth:`~Dataset.exportDataset` and raises a warning if normalisation is employed as :py:class:`TargetedDataset` :py:attr:`intensityData` can be left-censored.
        """
        # handle the dilution due to method... These lines are left here commented - as hopefully this will be handled more
        # elegantly through the intensityData getter
        # Export dataset...
        tmpData = copy.deepcopy(self)
        tmpData._intensityData = tmpData._intensityData * (100/tmpData.sampleMetadata['Dilution']).values[:, numpy.newaxis]
        super(TargetedDataset, tmpData).exportDataset(destinationPath=destinationPath, saveFormat=saveFormat, withExclusions=withExclusions, escapeDelimiters=escapeDelimiters, filterMetadata=filterMetadata)


    def _exportCSV(self, destinationPath, escapeDelimiters=False):
        """
        Replace `-numpy.inf` by `<LLOQ` and `numpy.inf` by `>ULOQ`

        Export the dataset to the directory *destinationPath* as a set of three CSV files:
            *destinationPath*_intensityData.csv
            *destinationPath*_sampleMetadata.csv
            *destinationPath*_featureMetadata.csv

        :param str destinationPath: Path to a directory in which the output will be saved
        :param bool escapeDelimiters: Remove characters commonly used as delimiters in csv files from metadata
        :raises IOError: If writing one of the files fails
        """

        sampleMetadata = self.sampleMetadata.copy(deep=True)
        featureMetadata = self.featureMetadata.copy(deep=True)

        intensityData = copy.deepcopy(self._intensityData)
        intensityData = pandas.DataFrame(intensityData)
        intensityData.replace(to_replace=-numpy.inf, value='<LLOQ', inplace=True)
        intensityData.replace(to_replace= numpy.inf, value='>ULOQ', inplace=True)

        if escapeDelimiters:
            # Remove any commas from metadata/feature tables - for subsequent import of resulting csv files to other software packages

            for column in sampleMetadata.columns:
                try:
                    if type(sampleMetadata[column][0]) is not datetime:
                        sampleMetadata[column] = sampleMetadata[column].str.replace(',', ';')
                except:
                    pass

            for column in featureMetadata.columns:
                try:
                    if type(featureMetadata[column][0]) is not datetime:
                        featureMetadata[column] = featureMetadata[column].str.replace(',', ';')
                except:
                    pass

        # Export sample metadata
        sampleMetadata.to_csv(destinationPath + '_sampleMetadata.csv', encoding='utf-8', date_format=self._timestampFormat)

        # Export feature metadata
        featureMetadata.to_csv(destinationPath + '_featureMetadata.csv', encoding='utf-8')

        # Export intensity data
        intensityData.to_csv(os.path.join(destinationPath + '_intensityData.csv'), encoding='utf-8', date_format=self._timestampFormat, header=False, index=False)


    def _exportUnifiedCSV(self, destinationPath, escapeDelimiters=False):
        """
        Replace `-numpy.inf` by `<LLOQ` and `numpy.inf` by `>ULOQ`

        Export the dataset to the directory *destinationPath* as a combined CSV file containing intensity data, and feature and sample metadata
            *destinationPath*_combinedData.csv

        :param str destinationPath: Path to a directory in which the output will be saved
        :param bool escapeDelimiters: Remove characters commonly used as delimiters in csv files from metadata
        :raises IOError: If writing one of the files fails
        """

        sampleMetadata = self.sampleMetadata.copy(deep=True)
        featureMetadata = self.featureMetadata.copy(deep=True)

        intensityData = copy.deepcopy(self._intensityData)
        intensityData = pandas.DataFrame(intensityData)
        intensityData.replace(to_replace=-numpy.inf, value='<LLOQ', inplace=True)
        intensityData.replace(to_replace=numpy.inf, value='>ULOQ', inplace=True)

        if escapeDelimiters:
            # Remove any commas from metadata/feature tables - for subsequent import of resulting csv files to other software packages

            for column in sampleMetadata.columns:
                try:
                    if type(sampleMetadata[column][0]) is not datetime:
                        sampleMetadata[column] = sampleMetadata[column].str.replace(',', ';')
                except:
                    pass

            for column in featureMetadata.columns:
                try:
                    if type(featureMetadata[column][0]) is not datetime:
                        featureMetadata[column] = featureMetadata[column].str.replace(',', ';')
                except:
                    pass

        # Export combined data in single file
        tmpXCombined = pandas.concat([featureMetadata.transpose(), intensityData], axis=0, sort=False)

        with warnings.catch_warnings():
            # Seems no way to avoid pandas complaining here (v0.18.1)
            warnings.simplefilter("ignore")
            tmpCombined = pandas.concat([sampleMetadata, tmpXCombined], axis=1, sort=False)

        # reorder rows to put metadata first
        tmpCombined = tmpCombined.reindex(tmpXCombined.index, axis=0)

        # Save
        tmpCombined.to_csv(os.path.join(destinationPath + '_combinedData.csv'), encoding='utf-8', date_format=self._timestampFormat)


    def validateObject(self, verbose=True, raiseError=False, raiseWarning=True):
        """
        Checks that all the attributes specified in the class definition are present and of the required class and/or values.

        Returns 4 boolean: is the object a *Dataset* < a *basic TargetedDataset* < *has the object parameters for QC* < *has the object sample metadata*.

        To employ all class methods, the most inclusive (*has the object sample metadata*) must be successful:

        * *'Basic TargetedDataset'* checks :py:class:`~TargetedDataset` types and uniqueness as well as additional attributes.
        * *'has parameters for QC'* is *'Basic TargetedDataset'* + sampleMetadata[['SampleType, AssayRole, Dilution, Run Order, Batch, Correction Batch, Sample Base Name]]
        * *'has sample metadata'* is *'has parameters for QC'* + sampleMetadata[['Sample ID', 'Subject ID', 'Matrix']]

        :py:attr:`~calibration['calibIntensityData']` must be initialised even if no samples are present
        :py:attr:`~calibration['calibSampleMetadata']` must be initialised even if no samples are present, use: ``pandas.DataFrame(None, columns=self.sampleMetadata.columns.values.tolist())``
        :py:attr:`~calibration['calibFeatureMetadata']` must be initialised even if no samples are present, use a copy of ``self.featureMetadata``
        :py:attr:`~calibration['calibExpectedConcentration']` must be initialised even if no samples are present, use: ``pandas.DataFrame(None, columns=self.expectedConcentration.columns.values.tolist())``
        Calibration features must be identical to the usual features. Number of calibration samples and features must match across the 4 calibration tables
        If *'sampleMetadataExcluded'*, *'intensityDataExcluded'*, *'featureMetadataExcluded'*, *'expectedConcentrationExcluded'* or *'excludedFlag'* exist, the existence and number of exclusions (based on *'sampleMetadataExcluded'*) is checked

        Column type() in pandas.DataFrame are established on the first sample (for non int/float)
        featureMetadata are search for column names containing *'LLOQ'* & *'ULOQ'* to allow for *'LLOQ_batch...'* after :py:meth:`~TargetedDataset.__add__`, the first column matching is then checked for dtype
        If datasets are merged, calibration is a list of dict, and number of features is only kept constant inside each dict
        Does not check for uniqueness in :py:attr:`~sampleMetadata['Sample File Name']`
        Does not check columns inside :py:attr:`~calibration['calibSampleMetadata']`
        Does not check columns inside :py:attr:`~calibration['calibFeatureMetadata']`
        Does not currently check for :py:attr:`~Attributes['Feature Name']`

        :param verbose: if True the result of each check is printed (default True)
        :type verbose: bool
        :param raiseError: if True an error is raised when a check fails and the validation is interrupted (default False)
        :type raiseError: bool
        :param raiseWarning: if True a warning is raised when a check fails
        :type raiseWarning: bool
        :return: A dictionary of 4 boolean with True if the Object conforms to the corresponding test. 'Dataset' conforms to :py:class:`Dataset`, 'BasicTargetedDataset' conforms to :py:class:`Dataset` + basic :py:class:`TargetedDataset`, 'QC' BasicTargetedDataset + object has QC parameters, 'sampleMetadata' QC + object has sample metadata information
        :rtype: dict

        :raises TypeError: if the Object class is wrong
        :raises AttributeError: if self.Attributes['methodName'] does not exist
        :raises TypeError: if self.Attributes['methodName'] is not a str
        :raises AttributeError: if self.Attributes['externalID'] does not exist
        :raises TypeError: if self.Attributes['externalID'] is not a list
        :raises TypeError: if self.VariableType is not an enum 'VariableType'
        :raises AttributeError: if self.fileName does not exist
        :raises TypeError: if self.fileName is not a str or list
        :raises AttributeError: if self.filePath does not exist
        :raises TypeError: if self.filePath is not a str or list
        :raises ValueError: if self.sampleMetadata does not have the same number of samples as self._intensityData
        :raises TypeError: if self.sampleMetadata['Sample File Name'] is not str
        :raises TypeError: if self.sampleMetadata['AssayRole'] is not an enum 'AssayRole'
        :raises TypeError: if self.sampleMetadata['SampleType'] is not an enum 'SampleType'
        :raises TypeError: if self.sampleMetadata['Dilution'] is not an int or float
        :raises TypeError: if self.sampleMetadata['Batch'] is not an int or float
        :raises TypeError: if self.sampleMetadata['Correction Batch'] is not an int or float
        :raises TypeError: if self.sampleMetadata['Run Order'] is not an int
        :raises TypeError: if self.sampleMetadata['Acquired Time'] is not a datetime
        :raises TypeError: if self.sampleMetadata['Sample Base Name'] is not str
        :raises LookupError: if self.sampleMetadata does not have a Subject ID column
        :raises TypeError: if self.sampleMetadata['Subject ID'] is not a str
        :raises TypeError: if self.sampleMetadata['Sample ID'] is not a str
        :raises ValueError: if self.featureMetadata does not have the same number of features as self._intensityData
        :raises TypeError: if self.featureMetadata['Feature Name'] is not a str
        :raises ValueError: if self.featureMetadata['Feature Name'] is not unique
        :raises LookupError: if self.featureMetadata does not have a calibrationMethod column
        :raises TypeError: if self.featureMetadata['calibrationMethod'] is not an enum 'CalibrationMethod'
        :raises LookupError: if self.featureMetadata does not have a quantificationType column
        :raises TypeError: if self.featureMetadata['quantificationType'] is not an enum 'QuantificationType'
        :raises LookupError: if self.featureMetadata does not have a Unit column
        :raises TypeError: if self.featureMetadata['Unit'] is not a str
        :raises LookupError: if self.featureMetadata does not have a LLOQ or similar column
        :raises TypeError: if self.featureMetadata['LLOQ'] or similar is not an int or float
        :raises LookupError: if self.featureMetadata does not have a ULOQ or similar column
        :raises TypeError: if self.featureMetadata['ULOQ'] or similar is not an int or float
        :raises LookupError: if self.featureMetadata does not have the 'externalID' as columns
        :raises AttributeError: if self.expectedConcentration does not exist
        :raises TypeError: if self.expectedConcentration is not a pandas.DataFrame
        :raises ValueError: if self.expectedConcentration does not have the same number of samples as self._intensityData
        :raises ValueError: if self.expectedConcentration does not have the same number of features as self._intensityData
        :raises ValueError: if self.expectedConcentration column name do not match self.featureMetadata['Feature Name']
        :raises ValueError: if self.sampleMask is not initialised
        :raises ValueError: if self.sampleMask does not have the same number of samples as self._intensityData
        :raises ValueError: if self.featureMask has not been initialised
        :raises ValueError: if self.featureMask does not have the same number of features as self._intensityData
        :raises AttributeError: if self.calibration does not exist
        :raises TypeError: if self.calibration is not a dict
        :raises AttributeError: if self.calibration['calibIntensityData'] does not exist
        :raises TypeError: if self.calibration['calibIntensityData'] is not a numpy.ndarray
        :raises ValueError: if self.calibration['calibIntensityData'] does not have the same number of features as self._intensityData
        :raises AttributeError: if self.calibration['calibSampleMetadata'] does not exist
        :raises TypeError: if self.calibration['calibSampleMetadata'] is not a pandas.DataFrame
        :raises ValueError: if self.calibration['calibSampleMetadata'] does not have the same number of samples as self.calibration['calibIntensityData']
        :raises AttributeError: if self.calibration['calibFeatureMetadata'] does not exist
        :raises TypeError: if self.calibration['calibFeatureMetadata'] is not a pandas.DataFrame
        :raises LookupError: if self.calibration['calibFeatureMetadata'] does not have a ['Feature Name'] column
        :raises ValueError: if self.calibration['calibFeatureMetadata'] does not have the same number of features as self._intensityData
        :raises AttributeError: if self.calibration['calibExpectedConcentration'] does not exist
        :raises TypeError: if self.calibration['calibExpectedConcentration'] is not a pandas.DataFrame
        :raises ValueError: if self.calibration['calibExpectedConcentration'] does not have the same number of samples as self.calibration['calibIntensityData']
        :raises ValueError: if self.calibration['calibExpectedConcentration'] does not have the same number of features as self.calibration['calibIntensityData']
        :raises ValueError: if self.calibration['calibExpectedConcentration'] column name do not match self.featureMetadata['Feature Name']
        """

        def conditionTest(successCond, successMsg, failureMsg, allFailures, verb, raiseErr, raiseWarn, exception):
            if not successCond:
                allFailures.append(failureMsg)
                msg = failureMsg
                if raiseWarn:
                    warnings.warn(msg)
                if raiseErr:
                    raise exception
            else:
                msg = successMsg
            if verb:
                print(msg)
            return (allFailures)

        ## init
        failureListBasic = []
        failureListQC    = []
        failureListMeta  = []
        # reference number of samples / features, from _intensityData
        refNumSamples = None
        refNumFeatures = None
        # reference ['Feature Name'], from featureMetadata
        refFeatureName = None
        # reference number of calibration samples, from calibration['calibIntensityData']
        refNumCalibSamples = None
        # reference number of exclusions in list, from sampleMetadataExcluded
        refNumExcluded = None

        # First check it conforms to Dataset
        if super().validateObject(verbose=verbose, raiseError=raiseError, raiseWarning=raiseWarning):
            ## Check object class
            condition = isinstance(self, TargetedDataset)
            success = 'Check Object class:\tOK'
            failure = 'Check Object class:\tFailure, not TargetedDataset, but ' + str(type(self))
            failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose, raiseError, raiseWarning, exception=TypeError(failure))

            ## Attributes
            ## methodName
            # exist
            condition = 'methodName' in self.Attributes
            success = 'Check self.Attributes[\'methodName\'] exists:\tOK'
            failure = 'Check self.Attributes[\'methodName\'] exists:\tFailure, no attribute \'self.Attributes[\'methodName\']\''
            failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose, raiseError, raiseWarning, exception=AttributeError(failure))
            if condition:
                # is a str
                condition = isinstance(self.Attributes['methodName'], str)
                success = 'Check self.Attributes[\'methodName\'] is a str:\tOK'
                failure = 'Check self.Attributes[\'methodName\'] is a str:\tFailure, \'self.Attributes[\'methodName\']\' is ' + str(type(self.Attributes['methodName']))
                failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose, raiseError, raiseWarning, exception=TypeError(failure))
            # end self.Attributes['methodName']
            ## externalID
            # exist
            condition = 'externalID' in self.Attributes
            success = 'Check self.Attributes[\'externalID\'] exists:\tOK'
            failure = 'Check self.Attributes[\'externalID\'] exists:\tFailure, no attribute \'self.Attributes[\'externalID\']\''
            failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose, raiseError, raiseWarning, exception=AttributeError(failure))
            if condition:
                # is a list
                condition = isinstance(self.Attributes['externalID'], list)
                success = 'Check self.Attributes[\'externalID\'] is a list:\tOK'
                failure = 'Check self.Attributes[\'externalID\'] is a list:\tFailure, \'self.Attributes[\'externalID\']\' is ' + str(type(self.Attributes['externalID']))
                failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose, raiseError, raiseWarning, exception=TypeError(failure))
            # end self.Attributes['externalID']

            ## self.VariableType
            # is a enum VariableType
            condition = isinstance(self.VariableType, VariableType)
            success = 'Check self.VariableType is an enum \'VariableType\':\tOK'
            failure = 'Check self.VariableType is an enum \'VariableType\':\tFailure, \'self.VariableType\' is' + str(type(self.VariableType))
            failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose, raiseError, raiseWarning, exception=TypeError(failure))
            # end Variabletype

            ## self.fileName
            # exist
            condition = hasattr(self, 'fileName')
            success = 'Check self.fileName exists:\tOK'
            failure = 'Check self.fileName exists:\tFailure, no attribute \'self.fileName\''
            failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose, raiseError, raiseWarning, exception=AttributeError(failure))
            if condition:
                # is a str
                condition = isinstance(self.fileName, (str, list))
                success = 'Check self.fileName is a str or list:\tOK'
                failure = 'Check self.fileName is a str or list:\tFailure, \'self.fileName\' is ' + str(type(self.fileName))
                failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose, raiseError, raiseWarning, exception=TypeError(failure))
                if isinstance(self.fileName, list):
                    for i in range(len(self.fileName)):
                        condition = isinstance(self.fileName[i], (str))
                        success = 'Check self.filename[' + str(i) + '] is str:\tOK'
                        failure = 'Check self.filename[' + str(i) + '] is str:\tFailure, \'self.fileName[' + str(i) + '] is' + str(type(self.fileName[i]))
                        failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose, raiseError, raiseWarning, exception=TypeError(failure))
                    # end self.fileName list
            # end self.fileName

            ## self.filePath
            # exist
            condition = hasattr(self, 'filePath')
            success = 'Check self.filePath exists:\tOK'
            failure = 'Check self.filePath exists:\tFailure, no attribute \'self.filePath\''
            failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose, raiseError, raiseWarning, exception=AttributeError(failure))
            if condition:
                # is a str
                condition = isinstance(self.filePath, (str, list))
                success = 'Check self.filePath is a str or list:\tOK'
                failure = 'Check self.filePath is a str or list:\tFailure, \'self.filePath\' is ' + str(type(self.filePath))
                failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose, raiseError, raiseWarning, exception=TypeError(failure))
                if isinstance(self.filePath, list):
                    for i in range(len(self.filePath)):
                        condition = isinstance(self.filePath[i], (str))
                        success = 'Check self.filePath[' + str(i) + '] is str:\tOK'
                        failure = 'Check self.filePath[' + str(i) + '] is str:\tFailure, \'self.filePath[' + str(i) + '] is' + str(type(self.filePath[i]))
                        failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose, raiseError, raiseWarning, exception=TypeError(failure))
                    # end self.filePath list
            # end self.filePath

            ## self._intensityData
            # Use _intensityData as size reference for all future tables
            if (self._intensityData.all() != numpy.array(None).all()):
                refNumSamples = self._intensityData.shape[0]
                refNumFeatures = self._intensityData.shape[1]
                if verbose:
                    print('---- self._intensityData used as size reference ----')
                    print('\t' + str(refNumSamples) + ' samples, ' + str(refNumFeatures) + ' features')
            # end self._intensityData

            ## self.sampleMetadata
            # number of samples
            condition = (self.sampleMetadata.shape[0] == refNumSamples)
            success = 'Check self.sampleMetadata number of samples (rows):\tOK'
            failure = 'Check self.sampleMetadata number of samples (rows):\tFailure, \'self.sampleMetadata\' has ' + str(self.sampleMetadata.shape[0]) + ' samples, ' + str(refNumSamples) + 'expected'
            failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose, raiseError, raiseWarning, exception=ValueError(failure))
            if condition:
                # sampleMetadata['Sample File Name'] is str
                condition = isinstance(self.sampleMetadata['Sample File Name'][0], str)
                success = 'Check self.sampleMetadata[\'Sample File Name\'] is str:\tOK'
                failure = 'Check self.sampleMetadata[\'Sample File Name\'] is str:\tFailure, \'self.sampleMetadata[\'Sample File Name\']\' is ' + str(type(self.sampleMetadata['Sample File Name'][0]))
                failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose, raiseError, raiseWarning, exception=TypeError(failure))

                ## Fields required for QC
                # sampleMetadata['AssayRole'] is enum AssayRole
                condition = isinstance(self.sampleMetadata['AssayRole'][0], AssayRole)
                success = 'Check self.sampleMetadata[\'AssayRole\'] is an enum \'AssayRole\':\tOK'
                failure = 'Check self.sampleMetadata[\'AssayRole\'] is an enum \'AssayRole\':\tFailure, \'self.sampleMetadata[\'AssayRole\']\' is ' + str(type(self.sampleMetadata['AssayRole'][0]))
                failureListQC = conditionTest(condition, success, failure, failureListQC, verbose, raiseError, raiseWarning, exception=TypeError(failure))
                # sampleMetadata['SampleType'] is enum SampleType
                condition = isinstance(self.sampleMetadata['SampleType'][0], SampleType)
                success = 'Check self.sampleMetadata[\'SampleType\'] is an enum \'SampleType\':\tOK'
                failure = 'Check self.sampleMetadata[\'SampleType\'] is an enum \'SampleType\':\tFailure, \'self.sampleMetadata[\'SampleType\']\' is ' + str(type(self.sampleMetadata['SampleType'][0]))
                failureListQC = conditionTest(condition, success, failure, failureListQC, verbose, raiseError, raiseWarning, exception=TypeError(failure))
                # sampleMetadata['Dilution'] is an int or float
                condition = isinstance(self.sampleMetadata['Dilution'][0], (int, float, numpy.integer, numpy.floating))
                success = 'Check self.sampleMetadata[\'Dilution\'] is int or float:\tOK'
                failure = 'Check self.sampleMetadata[\'Dilution\'] is int or float:\tFailure, \'self.sampleMetadata[\'Dilution\']\' is ' + str(type(self.sampleMetadata['Dilution'][0]))
                failureListQC = conditionTest(condition, success, failure, failureListQC, verbose, raiseError, raiseWarning, exception=TypeError(failure))
                # sampleMetadata['Batch'] is an int or float
                condition = isinstance(self.sampleMetadata['Batch'][0], (int, float, numpy.integer, numpy.floating))
                success = 'Check self.sampleMetadata[\'Batch\'] is int or float:\tOK'
                failure = 'Check self.sampleMetadata[\'Batch\'] is int or float:\tFailure, \'self.sampleMetadata[\'Batch\']\' is ' + str(type(self.sampleMetadata['Batch'][0]))
                failureListQC = conditionTest(condition, success, failure, failureListQC, verbose, raiseError, raiseWarning, exception=TypeError(failure))
                # sampleMetadata['Correction Batch'] is an int or float
                condition = isinstance(self.sampleMetadata['Correction Batch'][0], (int, float, numpy.integer, numpy.floating))
                success = 'Check self.sampleMetadata[\'Correction Batch\'] is int or float:\tOK'
                failure = 'Check self.sampleMetadata[\'Correction Batch\'] is int or float:\tFailure, \'self.sampleMetadata[\'Correction Batch\']\' is ' + str(type(self.sampleMetadata['Correction Batch'][0]))
                failureListQC = conditionTest(condition, success, failure, failureListQC, verbose, raiseError, raiseWarning, exception=TypeError(failure))
                # sampleMetadata['Run Order'] is an int
                condition = isinstance(self.sampleMetadata['Run Order'][0], (int, numpy.integer))
                success = 'Check self.sampleMetadata[\'Run Order\'] is int:\tOK'
                failure = 'Check self.sampleMetadata[\'Run Order\'] is int:\tFailure, \'self.sampleMetadata[\'Run Order\']\' is ' + str(type(self.sampleMetadata['Run Order'][0]))
                failureListQC = conditionTest(condition, success, failure, failureListQC, verbose, raiseError, raiseWarning, exception=TypeError(failure))
                # sampleMetadata['Acquired Time'] is datetime.datetime
                condition = isinstance(self.sampleMetadata['Acquired Time'][0], datetime)
                success = 'Check self.sampleMetadata[\'Acquired Time\'] is datetime:\tOK'
                failure = 'Check self.sampleMetadata[\'Acquired Time\'] is datetime:\tFailure, \'self.sampleMetadata[\'Acquired Time\']\' is ' + str(type(self.sampleMetadata['Acquired Time'][0]))
                failureListQC = conditionTest(condition, success, failure, failureListQC, verbose, raiseError, raiseWarning, exception=TypeError(failure))
                # sampleMetadata['Sample Base Name'] is str
                condition = isinstance(self.sampleMetadata['Sample Base Name'][0], str)
                success = 'Check self.sampleMetadata[\'Sample Base Name\'] is str:\tOK'
                failure = 'Check self.sampleMetadata[\'Sample Base Name\'] is str:\tFailure, \'self.sampleMetadata[\'Sample Base Name\']\' is ' + str(type(self.sampleMetadata['Sample Base Name'][0]))
                failureListQC = conditionTest(condition, success, failure, failureListQC, verbose, raiseError, raiseWarning, exception=TypeError(failure))

                ## Sample metadata fields
                # ['Subject ID']
                condition = ('Subject ID' in self.sampleMetadata.columns)
                success = 'Check self.sampleMetadata[\'Subject ID\'] exists:\tOK'
                failure = 'Check self.sampleMetadata[\'Subject ID\'] exists:\tFailure, \'self.sampleMetadata\' lacks a \'Subject ID\' column'
                failureListMeta = conditionTest(condition, success, failure, failureListMeta, verbose, raiseError, raiseWarning, exception=LookupError(failure))
                if condition:
                    # sampleMetadata['Subject ID'] is str
                    condition = (self.sampleMetadata['Subject ID'].dtype == numpy.dtype('O'))
                    success = 'Check self.sampleMetadata[\'Subject ID\'] is str:\tOK'
                    failure = 'Check self.sampleMetadata[\'Subject ID\'] is str:\tFailure, \'self.sampleMetadata[\'Subject ID\']\' is ' + str(type(self.sampleMetadata['Subject ID'][0]))
                    failureListMeta = conditionTest(condition, success, failure, failureListMeta, verbose, raiseError, raiseWarning, exception=TypeError(failure))
                # end self.sampleMetadata['Subject ID']
                # sampleMetadata['Sample ID'] is str
                condition = (self.sampleMetadata['Sample ID'].dtype == numpy.dtype('O'))
                success = 'Check self.sampleMetadata[\'Sample ID\'] is str:\tOK'
                failure = 'Check self.sampleMetadata[\'Sample ID\'] is str:\tFailure, \'self.sampleMetadata[\'Sample ID\']\' is ' + str(type(self.sampleMetadata['Sample ID'][0]))
                failureListMeta = conditionTest(condition, success, failure, failureListMeta, verbose, raiseError, raiseWarning, exception=TypeError(failure))
            # end self.sampleMetadata number of samples
            # end self.sampleMetadata

            ## self.featureMetadata
            # exist
            # number of features
            condition = (self.featureMetadata.shape[0] == refNumFeatures)
            success = 'Check self.featureMetadata number of features (rows):\tOK'
            failure = 'Check self.featureMetadata number of features (rows):\tFailure, \'self.featureMetadata\' has ' + str(self.featureMetadata.shape[0]) + ' features, ' + str(refNumFeatures) + ' expected'
            failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose, raiseError, raiseWarning, exception=ValueError(failure))
            if condition & (self.featureMetadata.shape[0] != 0):
                # No point checking columns if the number of columns is wrong or no features
                # featureMetadata['Feature Name'] is str
                condition = isinstance(self.featureMetadata['Feature Name'][0], str)
                success = 'Check self.featureMetadata[\'Feature Name\'] is str:\tOK'
                failure = 'Check self.featureMetadata[\'Feature Name\'] is str:\tFailure, \'self.featureMetadata[\'Feature Name\']\' is ' + str(type(self.featureMetadata['Feature Name'][0]))
                failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose, raiseError, raiseWarning, exception=TypeError(failure))
                if condition:
                    # featureMetadata['Feature Name'] are unique
                    u_ids, u_counts = numpy.unique(self.featureMetadata['Feature Name'], return_counts=True)
                    condition = all(u_counts == 1)
                    success = 'Check self.featureMetadata[\'Feature Name\'] are unique:\tOK'
                    failure = 'Check self.featureMetadata[\'Feature Name\'] are unique:\tFailure, the following \'self.featureMetadata[\'Feature Name\']\' are present more than once ' + str(u_ids[u_counts > 1].tolist())
                    failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose, raiseError, raiseWarning, exception=ValueError(failure))
                    # Use featureMetadata['Feature Name'] as reference for future tables
                    refFeatureName = self.featureMetadata['Feature Name'].values.tolist()
                    if verbose:
                        print('---- self.featureMetadata[\'Feature Name\'] used as Feature Name reference ----')
                # end self.featureMetadata['Feature Name']
                # ['calibrationMethod']
                condition = ('calibrationMethod' in self.featureMetadata.columns)
                success = 'Check self.featureMetadata[\'calibrationMethod\'] exists:\tOK'
                failure = 'Check self.featureMetadata[\'calibrationMethod\'] exists:\tFailure, \'self.featureMetadata\' lacks a \'calibrationMethod\' column'
                failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose, raiseError, raiseWarning, exception=LookupError(failure))
                if condition:
                    # featureMetadata['calibrationMethod'] is an enum 'CalibrationMethod'
                    condition = isinstance(self.featureMetadata['calibrationMethod'][0], CalibrationMethod)
                    success = 'Check self.featureMetadata[\'calibrationMethod\'] is an enum \'CalibrationMethod\':\tOK'
                    failure = 'Check self.featureMetadata[\'calibrationMethod\'] is an enum \'CalibrationMethod\':\tFailure, \'self.featureMetadata[\'calibrationMethod\']\' is ' + str(type(self.featureMetadata['calibrationMethod'][0]))
                    failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose, raiseError, raiseWarning, exception=TypeError(failure))
                # end self.featureMetadata['calibrationMethod']
                # ['quantificationType']
                condition = ('quantificationType' in self.featureMetadata.columns)
                success = 'Check self.featureMetadata[\'quantificationType\'] exists:\tOK'
                failure = 'Check self.featureMetadata[\'quantificationType\'] exists:\tFailure, \'self.featureMetadata\' lacks a \'quantificationType\' column'
                failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose, raiseError, raiseWarning, exception=LookupError(failure))
                if condition:
                    # featureMetadata['quantificationType'] is an enum 'QuantificationType'
                    condition = isinstance(self.featureMetadata['quantificationType'][0], QuantificationType)
                    success = 'Check self.featureMetadata[\'quantificationType\'] is an enum \'QuantificationType\':\tOK'
                    failure = 'Check self.featureMetadata[\'quantificationType\'] is an enum \'QuantificationType\':\tFailure, \'self.featureMetadata[\'quantificationType\']\' is ' + str(type(self.featureMetadata['quantificationType'][0]))
                    failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose, raiseError, raiseWarning, exception=TypeError(failure))
                # end self.featureMetadata['quantificationType']
                # ['Unit']
                condition = ('Unit' in self.featureMetadata.columns)
                success = 'Check self.featureMetadata[\'Unit\'] exists:\tOK'
                failure = 'Check self.featureMetadata[\'Unit\'] exists:\tFailure, \'self.featureMetadata\' lacks a \'Unit\' column'
                failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose, raiseError, raiseWarning, exception=LookupError(failure))
                if condition:
                    # featureMetadata['Unit'] is a str
                    condition = isinstance(self.featureMetadata['Unit'][0], str)
                    success = 'Check self.featureMetadata[\'Unit\'] is a str:\tOK'
                    failure = 'Check self.featureMetadata[\'Unit\'] is a str:\tFailure, \'self.featureMetadata[\'Unit\']\' is ' + str(type(self.featureMetadata['Unit'][0]))
                    failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose, raiseError, raiseWarning, exception=TypeError(failure))
                # end self.featureMetadata['Unit']
                # ['LLOQ']
                tmpLLOQMatch = self.featureMetadata.columns.to_series().str.contains('LLOQ')
                condition = (sum(tmpLLOQMatch) > 0)
                success = 'Check self.featureMetadata[\'LLOQ\'] or similar exists:\tOK'
                failure = 'Check self.featureMetadata[\'LLOQ\'] or similar exists:\tFailure, \'self.featureMetadata\' lacks a \'LLOQ\' or \'LLOQ_batch\' column'
                failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose, raiseError, raiseWarning, exception=LookupError(failure))
                if condition:
                    # featureMetadata['LLOQ'] is a float, try on first found
                    condition = ((self.featureMetadata.loc[:, tmpLLOQMatch].iloc[:, 0].dtype == numpy.dtype(numpy.float)) | (self.featureMetadata.loc[:, tmpLLOQMatch].iloc[:, 0].dtype == numpy.dtype(numpy.int32)) | (self.featureMetadata.loc[:, tmpLLOQMatch].iloc[:, 0].dtype == numpy.dtype(numpy.int64)))
                    success = 'Check self.featureMetadata[\'' + str(self.featureMetadata.columns[tmpLLOQMatch][0]) + '\'] is int or float:\tOK'
                    failure = 'Check self.featureMetadata[\'' + str(self.featureMetadata.columns[tmpLLOQMatch][0]) + '\'] is int or float:\tFailure, \'self.featureMetadata[\'' + str(self.featureMetadata.columns[tmpLLOQMatch][0]) + '\']\' is ' + str(self.featureMetadata.loc[:, tmpLLOQMatch].iloc[:, 0].dtype)
                    failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose, raiseError, raiseWarning, exception=TypeError(failure))
                # end self.featureMetadata['LLOQ']
                # ['ULOQ']
                tmpULOQMatch = self.featureMetadata.columns.to_series().str.contains('ULOQ')
                condition = (sum(tmpULOQMatch) > 0)
                success = 'Check self.featureMetadata[\'ULOQ\'] or similar exists:\tOK'
                failure = 'Check self.featureMetadata[\'ULOQ\'] or similar exists:\tFailure, \'self.featureMetadata\' lacks a \'ULOQ\' or \'ULOQ_batch\' column'
                failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose, raiseError, raiseWarning, exception=LookupError(failure))
                if condition:
                    # featureMetadata['ULOQ'] is a float, try on first found
                    condition = ((self.featureMetadata.loc[:, tmpULOQMatch].iloc[:, 0].dtype == numpy.dtype(numpy.float)) | (self.featureMetadata.loc[:, tmpULOQMatch].iloc[:, 0].dtype == numpy.dtype(numpy.int32)) | (self.featureMetadata.loc[:, tmpULOQMatch].iloc[:, 0].dtype == numpy.dtype(numpy.int64)))
                    success = 'Check self.featureMetadata[\'' + str(self.featureMetadata.columns[tmpULOQMatch][0]) + '\'] is int or float:\tOK'
                    failure = 'Check self.featureMetadata[\'' + str(self.featureMetadata.columns[tmpULOQMatch][0]) + '\'] is int or float:\tFailure, \'self.featureMetadata[\'' + str(self.featureMetadata.columns[tmpULOQMatch][0]) + '\']\' is ' + str(self.featureMetadata.loc[:, tmpULOQMatch].iloc[:, 0].dtype)
                    failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose, raiseError, raiseWarning, exception=TypeError(failure))
                # end self.featureMetadata['ULOQ']
                # 'externalID' in featureMetadata columns (need externalID to exist)
                if 'externalID' in self.Attributes:
                    if isinstance(self.Attributes['externalID'], list):
                        condition = set(self.Attributes['externalID']).issubset(self.featureMetadata.columns)
                        success = 'Check self.featureMetadata does have the \'externalID\' as columns:\tOK'
                        failure = 'Check self.featureMetadata does have the \'externalID\' as columns:\tFailure, \'self.featureMetadata\' lacks the \'externalID\' columns'
                        failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose, raiseError, raiseWarning, exception=LookupError(failure))
                # end 'externalID' columns
            # end self.featureMetadata number of features
            # end self.featureMetadata

            ## self.expectedConcentration
            # exist
            condition = hasattr(self, 'expectedConcentration')
            success = 'Check self.expectedConcentration exists:\tOK'
            failure = 'Check self.expectedConcentration exists:\tFailure, no attribute \'self.expectedConcentration\''
            failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose, raiseError, raiseWarning, exception=AttributeError(failure))
            if condition:
                # is a pandas.DataFrame
                condition = isinstance(self.expectedConcentration, pandas.DataFrame)
                success = 'Check self.expectedConcentration is a pandas.DataFrame:\tOK'
                failure = 'Check self.expectedConcentration is a pandas.DataFrame:\tFailure, \'self.expectedConcentration\' is ' + str(type(self.expectedConcentration))
                failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose, raiseError, raiseWarning, exception=TypeError(failure))
                if condition:
                    # number of samples
                    condition = (self.expectedConcentration.shape[0] == refNumSamples)
                    success = 'Check self.expectedConcentration number of samples (rows):\tOK'
                    failure = 'Check self.expectedConcentration number of samples (rows):\tFailure, \'self.expectedConcentration\' has ' + str(self.expectedConcentration.shape[0]) + ' features, ' + str(refNumSamples) + ' expected'
                    failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose, raiseError, raiseWarning, exception=ValueError(failure))
                    # number of features
                    condition = (self.expectedConcentration.shape[1] == refNumFeatures)
                    success = 'Check self.expectedConcentration number of features (columns):\tOK'
                    failure = 'Check self.expectedConcentration number of features (columns):\tFailure, \'self.expectedConcentration\' has ' + str(self.expectedConcentration.shape[1]) + ' features, ' + str(refNumFeatures) + ' expected'
                    failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose, raiseError, raiseWarning, exception=ValueError(failure))
                    if condition & (refNumFeatures != 0):
                        # expectedConcentration column names match ['Feature Name']
                        tmpDiff = pandas.DataFrame({'FeatName': refFeatureName, 'ColName': self.expectedConcentration.columns.values.tolist()})
                        condition = (self.expectedConcentration.columns.values.tolist() == refFeatureName)
                        success = 'Check self.expectedConcentration column name match self.featureMetadata[\'Feature Name\']:\tOK'
                        failure = 'Check self.expectedConcentration column name match self.featureMetadata[\'Feature Name\']:\tFailure, the following \'self.featureMetadata[\'Feature Name\']\' and \'self.expectedConcentration.columns\' differ ' + str(tmpDiff.loc[(tmpDiff['FeatName'] != tmpDiff['ColName']), ['FeatName', 'ColName']].values.tolist())
                        failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose, raiseError, raiseWarning, exception=ValueError(failure))
                    # end self.expectedConcentration number of features
                # end self.expectedConcentration is a pandas.DataFrame
            # end self.expectedConcentration

            ## self.sampleMask
            # is initialised
            condition = (self.sampleMask.shape != ())
            success = 'Check self.sampleMask is initialised:\tOK'
            failure = 'Check self.sampleMask is initialised:\tFailure, \'self.sampleMask\' is not initialised'
            failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose,raiseError, raiseWarning, exception=ValueError(failure))
            if condition:
                # number of samples
                condition = (self.sampleMask.shape == (refNumSamples,))
                success = 'Check self.sampleMask number of samples:\tOK'
                failure = 'Check self.sampleMask number of samples:\tFailure, \'self.sampleMask\' has ' + str(self.sampleMask.shape[0]) + ' samples, ' + str(refNumSamples) + ' expected'
                failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose, raiseError, raiseWarning, exception=ValueError(failure))
            ## end self.sampleMask

            ## self.featureMask
            # is initialised
            condition = (self.featureMask.shape != ())
            success = 'Check self.featureMask is initialised:\tOK'
            failure = 'Check self.featureMask is initialised:\tFailure, \'self.featureMask\' is not initialised'
            failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose, raiseError, raiseWarning, exception=ValueError(failure))
            if condition:
                # number of features
                condition = (self.featureMask.shape == (refNumFeatures,))
                success = 'Check self.featureMask number of features:\tOK'
                failure = 'Check self.featureMask number of features:\tFailure, \'self.featureMask\' has ' + str(self.featureMask.shape[0]) + ' features, ' + str(refNumFeatures) + ' expected'
                failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose, raiseError, raiseWarning, exception=ValueError(failure))
            ## end self.featureMask

            ## self.calibration
            # exist
            condition = hasattr(self, 'calibration')
            success = 'Check self.calibration exists:\tOK'
            failure = 'Check self.calibration exists:\tFailure, no attribute \'self.calibration\''
            failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose, raiseError, raiseWarning, exception=AttributeError(failure))
            if condition:
                # is a dict or a list
                condition = isinstance(self.calibration, (dict, list))
                success = 'Check self.calibration is a dict or list:\tOK'
                failure = 'Check self.calibration is a dict or list:\tFailure, \'self.calibration\' is ' + str(type(self.calibration))
                failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose, raiseError, raiseWarning, exception=TypeError(failure))
                if condition:
                    # self.calibration is a list of dict
                    if isinstance(self.calibration, list):
                        # use reference inside each calibration
                        refCalibNumSamples  = len(self.calibration) * [None]
                        refCalibNumFeatures = len(self.calibration) * [None]
                        refCalibFeatureName = len(self.calibration) * [None]
                        for i in range(len(self.calibration)):
                            # self.calibration[i] is a dict
                            condition = isinstance(self.calibration[i], dict)
                            success = 'Check self.calibration[' + str(i) + '] is a dict or list:\tOK'
                            failure = 'Check self.calibration[' + str(i) + '] is a dict or list:\tFailure, \'self.calibration\' is ' + str(type(self.calibration[i]))
                            failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose, raiseError, raiseWarning, exception=TypeError(failure))
                            if condition:
                                ## calibIntensityData
                                # exist
                                condition = 'calibIntensityData' in self.calibration[i]
                                success = 'Check self.calibration[' + str(i) + '][\'calibIntensityData\'] exists:\tOK'
                                failure = 'Check self.calibration[' + str(i) + '][\'calibIntensityData\'] exists:\tFailure, no attribute \'self.calibration[' + str(i) + '][\'calibIntensityData\']\''
                                failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose, raiseError, raiseWarning, exception=AttributeError(failure))
                                if condition:
                                    # is a numpy.ndarray
                                    condition = isinstance(self.calibration[i]['calibIntensityData'], numpy.ndarray)
                                    success = 'Check self.calibration[' + str(i) + '][\'calibIntensityData\'] is a numpy.ndarray:\tOK'
                                    failure = 'Check self.calibration[' + str(i) + '][\'calibIntensityData\'] is a numpy.ndarray:\tFailure, \'self.calibration[' + str(i) + '][\'calibIntensityData\']\' is ' + str(type(self.calibration[i]['calibIntensityData']))
                                    failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose, raiseError, raiseWarning, exception=TypeError(failure))
                                    if condition:
                                        if (self.calibration[i]['calibIntensityData'].all() != numpy.array(None).all()):
                                            # Use calibIntensityData as number of calib sample/feature reference
                                            refCalibNumSamples[i] = self.calibration[i]['calibIntensityData'].shape[0]
                                            refCalibNumFeatures[i] = self.calibration[i]['calibIntensityData'].shape[1]
                                            if verbose:
                                                print('---- self.calibration[' + str(i) + '][\'calibIntensityData\'] used as number of calibration samples/features reference ----')
                                                print('\t' + str(refCalibNumSamples[i]) + ' samples, ' + str(refCalibNumFeatures[i]) + ' features')
                                        # end calibIntensityData is a numpy.ndarray
                                # end calibIntensityData
                                ## calibSampleMetadata
                                # exist
                                condition = 'calibSampleMetadata' in self.calibration[i]
                                success = 'Check self.calibration[' + str(i) + '][\'calibSampleMetadata\'] exists:\tOK'
                                failure = 'Check self.calibration[' + str(i) + '][\'calibSampleMetadata\'] exists:\tFailure, no attribute \'self.calibration[' + str(i) + '][\'calibSampleMetadata\']\''
                                failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose, raiseError, raiseWarning, exception=AttributeError(failure))
                                if condition:
                                    # is a pandas.DataFrame
                                    condition = isinstance(self.calibration[i]['calibSampleMetadata'], pandas.DataFrame)
                                    success = 'Check self.calibration[' + str(i) + '][\'calibSampleMetadata\'] is a pandas.DataFrame:\tOK'
                                    failure = 'Check self.calibration[' + str(i) + '][\'calibSampleMetadata\'] is a pandas.DataFrame:\tFailure, \'self.calibration[' + str(i) + '][\'calibSampleMetadata\']\' is ' + str(type(self.calibration[i]['calibSampleMetadata']))
                                    failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose, raiseError, raiseWarning, exception=TypeError(failure))
                                    if condition:
                                        # number of samples
                                        condition = (self.calibration[i]['calibSampleMetadata'].shape[0] == refCalibNumSamples[i])
                                        success = 'Check self.calibration[' + str(i) + '][\'calibSampleMetadata\'] number of samples:\tOK'
                                        failure = 'Check self.calibration[' + str(i) + '][\'calibSampleMetadata\'] number of samples:\tFailure, \'self.calibration[' + str(i) + '][\'calibSampleMetadata\']\' has ' + str(self.calibration[i]['calibSampleMetadata'].shape[0]) + ' samples, ' + str(refCalibNumSamples[i]) + ' expected'
                                        failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose, raiseError, raiseWarning, exception=ValueError(failure))
                                    # end calibSampleMetadata is a pandas.DataFrame
                                # end calibSampleMetadata
                                ## calibFeatureMetadata
                                # exist
                                condition = 'calibFeatureMetadata' in self.calibration[i]
                                success = 'Check self.calibration[' + str(i) + '][\'calibFeatureMetadata\'] exists:\tOK'
                                failure = 'Check self.calibration[' + str(i) + '][\'calibFeatureMetadata\'] exists:\tFailure, no attribute \'self.calibration[' + str(i) + '][\'calibFeatureMetadata\']\''
                                failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose, raiseError, raiseWarning, exception=AttributeError(failure))
                                if condition:
                                    # is a pandas.DataFrame
                                    condition = isinstance(self.calibration[i]['calibFeatureMetadata'], pandas.DataFrame)
                                    success = 'Check self.calibration[' + str(i) + '][\'calibFeatureMetadata\'] is a pandas.DataFrame:\tOK'
                                    failure = 'Check self.calibration[' + str(i) + '][\'calibFeatureMetadata\'] is a pandas.DataFrame:\tFailure, \'self.calibration[' + str(i) + '][\'calibFeatureMetadata\']\' is ' + str(type(self.calibration[i]['calibFeatureMetadata']))
                                    failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose, raiseError, raiseWarning, exception=TypeError(failure))
                                    if condition:
                                        # number of features
                                        condition = (self.calibration[i]['calibFeatureMetadata'].shape[0] == refCalibNumFeatures[i])
                                        success = 'Check self.calibration[' + str(i) + '][\'calibFeatureMetadata\'] number of features:\tOK'
                                        failure = 'Check self.calibration[' + str(i) + '][\'calibFeatureMetadata\'] number of features:\tFailure, \'self.calibration[' + str(i) + '][\'calibFeatureMetadata\']\' has ' + str(self.calibration[i]['calibFeatureMetadata'].shape[0]) + ' features, ' + str(refCalibNumFeatures[i]) + ' expected'
                                        failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose, raiseError, raiseWarning, exception=ValueError(failure))
                                        if condition & (refCalibNumFeatures[i] != 0):
                                            # Feature Name exist
                                            condition = ('Feature Name' in self.calibration[i]['calibFeatureMetadata'].columns.tolist())
                                            success = 'Check self.calibration[' + str(i) + '][\'calibFeatureMetadata\'][\'Feature Name\'] exist:\tOK'
                                            failure = 'Check self.calibration[' + str(i) + '][\'calibFeatureMetadata\'][\'Feature Name\'] exist:\tFailure, no column \'self.calibration[' + str(i) + '][\'calibFeatureMetadata\'][\'Feature Name\']'
                                            failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose,raiseError, raiseWarning, exception=LookupError(failure))
                                            if condition:
                                                # store the featureMetadata columns as reference
                                                refCalibFeatureName[i] = self.calibration[i]['calibFeatureMetadata']['Feature Name'].values.tolist()
                                    # end calibFeatureMetadata is a pandas.DataFrame
                                # end calibFeatureMetadata
                                ## calibExpectedConcentration
                                # exist
                                condition = 'calibExpectedConcentration' in self.calibration[i]
                                success = 'Check self.calibration[' + str(i) + '][\'calibExpectedConcentration\'] exists:\tOK'
                                failure = 'Check self.calibration[' + str(i) + '][\'calibExpectedConcentration\'] exists:\tFailure, no attribute \'self.calibration[' + str(i) + '][\'calibExpectedConcentration\']\''
                                failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose, raiseError, raiseWarning, exception=AttributeError(failure))
                                if condition:
                                    # is a pandas.DataFrame
                                    condition = isinstance(self.calibration[i]['calibExpectedConcentration'], pandas.DataFrame)
                                    success = 'Check self.calibration[' + str(i) + '][\'calibExpectedConcentration\'] is a pandas.DataFrame:\tOK'
                                    failure = 'Check self.calibration[' + str(i) + '][\'calibExpectedConcentration\'] is a pandas.DataFrame:\tFailure, \'self.calibration[' + str(i) + '][\'calibExpectedConcentration\']\' is ' + str(type(self.calibration[i]['calibExpectedConcentration']))
                                    failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose, raiseError, raiseWarning, exception=TypeError(failure))
                                    if condition:
                                        # number of samples
                                        condition = (self.calibration[i]['calibExpectedConcentration'].shape[0] == refCalibNumSamples[i])
                                        success = 'Check self.calibration[' + str(i) + '][\'calibExpectedConcentration\'] number of samples:\tOK'
                                        failure = 'Check self.calibration[' + str(i) + '][\'calibExpectedConcentration\'] number of samples:\tFailure, \'self.calibration[' + str(i) + '][\'calibExpectedConcentration\']\' has ' + str(self.calibration[i]['calibExpectedConcentration'].shape[0]) + ' samples, ' + str(refCalibNumSamples[i]) + ' expected'
                                        failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose, raiseError, raiseWarning, exception=ValueError(failure))
                                        # number of features
                                        condition = (self.calibration[i]['calibExpectedConcentration'].shape[1] == refCalibNumFeatures[i])
                                        success = 'Check self.calibration[' + str(i) + '][\'calibExpectedConcentration\'] number of features:\tOK'
                                        failure = 'Check self.calibration[' + str(i) + '][\'calibExpectedConcentration\'] number of features:\tFailure, \'self.calibration[' + str(i) + '][\'calibExpectedConcentration\']\' has ' + str(self.calibration[i]['calibExpectedConcentration'].shape[1]) + ' features, ' + str(refCalibNumFeatures[i]) + ' expected'
                                        failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose, raiseError, raiseWarning, exception=ValueError(failure))
                                        if condition & (refCalibNumFeatures[i] != 0):
                                            # calibExpectedConcentration column names match ['Feature Name']
                                            tmpDiff = pandas.DataFrame({'FeatName': refCalibFeatureName[i],'ColName': self.calibration[i]['calibExpectedConcentration'].columns.values.tolist()})
                                            condition = (self.calibration[i]['calibExpectedConcentration'].columns.values.tolist() == refCalibFeatureName[i])
                                            success = 'Check self.calibration[' + str(i) + '][\'calibExpectedConcentration\'] column name match self.calibration[' + str(i) + '][\'calibFeatureMetadata\'][\'Feature Name\']:\tOK'
                                            failure = 'Check self.calibration[' + str(i) + '][\'calibExpectedConcentration\'] column name match self.calibration[' + str(i) + '][\'calibFeatureMetadata\'][\'Feature Name\']:\tFailure, the following \'self.calibration[' + str(i) + '][\'calibFeatureMetadata\'][\'Feature Name\']\' and \'self.calibration[' + str(i) + '][\'calibExpectedConcentration\'].columns\' differ ' + str(tmpDiff.loc[(tmpDiff['FeatName'] != tmpDiff['ColName']), ['FeatName','ColName']].values.tolist())
                                            failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose, raiseError, raiseWarning, exception=ValueError(failure))
                                        # end calibExpectedConcentration number of features
                                    # end calibExpectedConcentration is a pandas.DataFrame
                                # end calibExpectedConcentration
                            # end self.calibration[i] is a dict
                    # end self.calibration list
                    else:
                    ## self.calibration is a dict
                        ## calibIntensityData
                        # exist
                        condition = 'calibIntensityData' in self.calibration
                        success = 'Check self.calibration[\'calibIntensityData\'] exists:\tOK'
                        failure = 'Check self.calibration[\'calibIntensityData\'] exists:\tFailure, no attribute \'self.calibration[\'calibIntensityData\']\''
                        failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose, raiseError, raiseWarning, exception=AttributeError(failure))
                        if condition:
                            # is a numpy.ndarray
                            condition = isinstance(self.calibration['calibIntensityData'], numpy.ndarray)
                            success = 'Check self.calibration[\'calibIntensityData\'] is a numpy.ndarray:\tOK'
                            failure = 'Check self.calibration[\'calibIntensityData\'] is a numpy.ndarray:\tFailure, \'self.calibration[\'calibIntensityData\']\' is ' + str(type(self.calibration['calibIntensityData']))
                            failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose, raiseError, raiseWarning, exception=TypeError(failure))
                            if condition:
                                if (self.calibration['calibIntensityData'].all() != numpy.array(None).all()):
                                    # number of features
                                    condition = (self.calibration['calibIntensityData'].shape[1] == refNumFeatures)
                                    success = 'Check self.calibration[\'calibIntensityData\'] number of features:\tOK'
                                    failure = 'Check self.calibration[\'calibIntensityData\'] number of features:\tFailure, \'self.calibration[\'calibIntensityData\']\' has ' + str(self.calibration['calibIntensityData'].shape[1]) + ' features, ' + str(refNumFeatures) + ' expected'
                                    failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose, raiseError, raiseWarning, exception=ValueError(failure))
                                    # Use calibIntensityData as number of calib sample reference
                                    refNumCalibSamples = self.calibration['calibIntensityData'].shape[0]
                                    if verbose:
                                        print('---- self.calibration[\'calibIntensityData\'] used as number of calibration samples reference ----')
                                        print('\t' + str(refNumCalibSamples) + ' samples')
                            # end calibIntensityData is a numpy.ndarray
                        # end calibIntensityData
                        ## calibSampleMetadata
                        # exist
                        condition = 'calibSampleMetadata' in self.calibration
                        success = 'Check self.calibration[\'calibSampleMetadata\'] exists:\tOK'
                        failure = 'Check self.calibration[\'calibSampleMetadata\'] exists:\tFailure, no attribute \'self.calibration[\'calibSampleMetadata\']\''
                        failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose, raiseError, raiseWarning, exception=AttributeError(failure))
                        if condition:
                            # is a pandas.DataFrame
                            condition = isinstance(self.calibration['calibSampleMetadata'], pandas.DataFrame)
                            success = 'Check self.calibration[\'calibSampleMetadata\'] is a pandas.DataFrame:\tOK'
                            failure = 'Check self.calibration[\'calibSampleMetadata\'] is a pandas.DataFrame:\tFailure, \'self.calibration[\'calibSampleMetadata\']\' is ' + str(type(self.calibration['calibSampleMetadata']))
                            failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose, raiseError, raiseWarning, exception=TypeError(failure))
                            if condition:
                                # number of samples
                                condition = (self.calibration['calibSampleMetadata'].shape[0] == refNumCalibSamples)
                                success = 'Check self.calibration[\'calibSampleMetadata\'] number of samples:\tOK'
                                failure = 'Check self.calibration[\'calibSampleMetadata\'] number of samples:\tFailure, \'self.calibration[\'calibSampleMetadata\']\' has ' + str(self.calibration['calibSampleMetadata'].shape[0]) + ' samples, ' + str(refNumCalibSamples) + ' expected'
                                failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose, raiseError, raiseWarning, exception=ValueError(failure))
                            # end calibSampleMetadata is a pandas.DataFrame
                        # end calibSampleMetadata
                        ## calibFeatureMetadata
                        # exist
                        condition = 'calibFeatureMetadata' in self.calibration
                        success = 'Check self.calibration[\'calibFeatureMetadata\'] exists:\tOK'
                        failure = 'Check self.calibration[\'calibFeatureMetadata\'] exists:\tFailure, no attribute \'self.calibration[\'calibFeatureMetadata\']\''
                        failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose, raiseError, raiseWarning, exception=AttributeError(failure))
                        if condition:
                            # is a pandas.DataFrame
                            condition = isinstance(self.calibration['calibFeatureMetadata'], pandas.DataFrame)
                            success = 'Check self.calibration[\'calibFeatureMetadata\'] is a pandas.DataFrame:\tOK'
                            failure = 'Check self.calibration[\'calibFeatureMetadata\'] is a pandas.DataFrame:\tFailure, \'self.calibration[\'calibFeatureMetadata\']\' is ' + str(type(self.calibration['calibFeatureMetadata']))
                            failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose, raiseError, raiseWarning, exception=TypeError(failure))
                            if condition:
                                # number of features
                                condition = (self.calibration['calibFeatureMetadata'].shape[0] == refNumFeatures)
                                success = 'Check self.calibration[\'calibFeatureMetadata\'] number of features:\tOK'
                                failure = 'Check self.calibration[\'calibFeatureMetadata\'] number of features:\tFailure, \'self.calibration[\'calibFeatureMetadata\']\' has ' + str(self.calibration['calibFeatureMetadata'].shape[0]) + ' features, ' + str(refNumFeatures) + ' expected'
                                failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose, raiseError, raiseWarning, exception=ValueError(failure))
                                if condition & (refNumFeatures != 0):
                                    # Feature Name exist
                                    condition = ('Feature Name' in self.calibration['calibFeatureMetadata'].columns.tolist())
                                    success = 'Check self.calibration[\'calibFeatureMetadata\'][\'Feature Name\'] exist:\tOK'
                                    failure = 'Check self.calibration[\'calibFeatureMetadata\'][\'Feature Name\'] exist:\tFailure, no column \'self.calibration[\'calibFeatureMetadata\'][\'Feature Name\']'
                                    failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose, raiseError, raiseWarning, exception=LookupError(failure))
                            # end calibFeatureMetadata is a pandas.DataFrame
                        # end calibFeatureMetadata
                        ## calibExpectedConcentration
                        # exist
                        condition = 'calibExpectedConcentration' in self.calibration
                        success = 'Check self.calibration[\'calibExpectedConcentration\'] exists:\tOK'
                        failure = 'Check self.calibration[\'calibExpectedConcentration\'] exists:\tFailure, no attribute \'self.calibration[\'calibExpectedConcentration\']\''
                        failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose, raiseError, raiseWarning, exception=AttributeError(failure))
                        if condition:
                            # is a pandas.DataFrame
                            condition = isinstance(self.calibration['calibExpectedConcentration'], pandas.DataFrame)
                            success = 'Check self.calibration[\'calibExpectedConcentration\'] is a pandas.DataFrame:\tOK'
                            failure = 'Check self.calibration[\'calibExpectedConcentration\'] is a pandas.DataFrame:\tFailure, \'self.calibration[\'calibExpectedConcentration\']\' is ' + str(type(self.calibration['calibExpectedConcentration']))
                            failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose, raiseError, raiseWarning, exception=TypeError(failure))
                            if condition:
                                # number of samples
                                condition = (self.calibration['calibExpectedConcentration'].shape[0] == refNumCalibSamples)
                                success = 'Check self.calibration[\'calibExpectedConcentration\'] number of samples:\tOK'
                                failure = 'Check self.calibration[\'calibExpectedConcentration\'] number of samples:\tFailure, \'self.calibration[\'calibExpectedConcentration\']\' has ' + str(self.calibration['calibExpectedConcentration'].shape[0]) + ' samples, ' + str(refNumCalibSamples) + ' expected'
                                failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose, raiseError, raiseWarning, exception=ValueError(failure))
                                # number of features
                                condition = (self.calibration['calibExpectedConcentration'].shape[1] == refNumFeatures)
                                success = 'Check self.calibration[\'calibExpectedConcentration\'] number of features:\tOK'
                                failure = 'Check self.calibration[\'calibExpectedConcentration\'] number of features:\tFailure, \'self.calibration[\'calibExpectedConcentration\']\' has ' + str(self.calibration['calibExpectedConcentration'].shape[1]) + ' features, ' + str(refNumFeatures) + ' expected'
                                failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose, raiseError, raiseWarning, exception=ValueError(failure))
                                if condition & (refNumFeatures != 0):
                                    # calibExpectedConcentration column names match ['Feature Name']
                                    tmpDiff = pandas.DataFrame({'FeatName': refFeatureName, 'ColName': self.calibration['calibExpectedConcentration'].columns.values.tolist()})
                                    condition = (self.calibration['calibExpectedConcentration'].columns.values.tolist() == refFeatureName)
                                    success = 'Check self.calibration[\'calibExpectedConcentration\'] column name match self.featureMetadata[\'Feature Name\']:\tOK'
                                    failure = 'Check self.calibration[\'calibExpectedConcentration\'] column name match self.featureMetadata[\'Feature Name\']:\tFailure, the following \'self.featureMetadata[\'Feature Name\']\' and \'self.calibration[\'calibExpectedConcentration\'].columns\' differ ' + str(tmpDiff.loc[(tmpDiff['FeatName'] != tmpDiff['ColName']), ['FeatName','ColName']].values.tolist())
                                    failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose, raiseError, raiseWarning, exception=ValueError(failure))
                                # end calibExpectedConcentration number of features
                            # end calibExpectedConcentration is a pandas.DataFrame
                        # end calibExpectedConcentration
                    # end self.calib is a dict
                # self.calibration is a dict or a list
            # end self.calibration


            ## List additional attributes (print + log)
            expectedSet = set({'Attributes', 'VariableType', '_Normalisation', '_name', 'fileName', 'filePath',
                               '_intensityData', 'sampleMetadata', 'featureMetadata', 'expectedConcentration', 'sampleMask',
                               'featureMask', 'calibration', 'sampleMetadataExcluded', 'intensityDataExcluded',
                               'featureMetadataExcluded', 'expectedConcentrationExcluded', 'excludedFlag'})
            objectSet = set(self.__dict__.keys())
            additionalAttributes = objectSet - expectedSet
            if len(additionalAttributes) > 0:
                if verbose:
                    print('--------')
                    print(str(len(additionalAttributes)) + ' additional attributes in the object:')
                    print('\t' + str(list(additionalAttributes)))
            else:
                if verbose:
                    print('--------')
                    print('No additional attributes in the object')

            ## Log and final Output
            # Basic failure might compromise logging, failure of QC compromises sample meta
            if len(failureListBasic) == 0:
                # Prepare log text and bool
                if len(failureListQC) != 0:
                    QCText = 'lacks parameters for QC'
                    QCBool = False
                    MetaText = 'lacks sample metadata'
                    MetaBool = False
                else:
                    QCText = 'has parameters for QC'
                    QCBool = True
                    if len(failureListMeta) != 0:
                        MetaText = 'lacks sample metadata'
                        MetaBool = False
                    else:
                        MetaText = 'has sample metadata'
                        MetaBool = True
                # Log
                self.Attributes['Log'].append([datetime.now(), 'Dataset conforms to basic TargetedDataset (0 errors), %s (%d errors), %s (%d errors), (%i samples and %i features), with %d additional attributes in the object: %s. QC errors: %s, Meta errors: %s' % (QCText, len(failureListQC), MetaText, len(failureListMeta), self.noSamples, self.noFeatures, len(additionalAttributes), list(additionalAttributes), list(failureListQC), list(failureListMeta))])
                # print results
                if verbose:
                    print('--------')
                    print('Conforms to Dataset:\t 0 errors found')
                    print('Conforms to basic TargetedDataset:\t 0 errors found')
                    if QCBool:
                        print('Has required parameters for QC:\t %d errors found' % ((len(failureListQC))))
                    else:
                        print('Does not have QC parameters:\t %d errors found' % ((len(failureListQC))))
                    if MetaBool:
                        print('Has sample metadata information:\t %d errors found' % ((len(failureListMeta))))
                    else:
                        print('Does not have sample metadata information:\t %d errors found' % ((len(failureListMeta))))
                # output
                if (not QCBool) & raiseWarning:
                    warnings.warn('Does not have QC parameters:\t %d errors found' % ((len(failureListQC))))
                if (not MetaBool) & raiseWarning:
                    warnings.warn('Does not have sample metadata information:\t %d errors found' % ((len(failureListMeta))))
                return ({'Dataset': True, 'BasicTargetedDataset': True, 'QC': QCBool, 'sampleMetadata': MetaBool})

            # Try logging to something that might not have a log
            else:
                # try logging
                try:
                    self.Attributes['Log'].append([datetime.now(), 'Failed basic TargetedDataset validation, with the following %d issues: %s' % (len(failureListBasic), failureListBasic)])
                except (AttributeError, KeyError, TypeError):
                    if verbose:
                        print('--------')
                        print('Logging failed')
                # print results
                if verbose:
                    print('--------')
                    print('Conforms to Dataset:\t 0 errors found')
                    print('Does not conform to basic TargetedDataset:\t %i errors found' % (len(failureListBasic)))
                    print('Does not have QC parameters')
                    print('Does not have sample metadata information')
                # output
                if raiseWarning:
                    warnings.warn('Does not conform to basic TargetedDataset:\t %i errors found' % (len(failureListBasic)))
                    warnings.warn('Does not have QC parameters')
                    warnings.warn('Does not have sample metadata information')
                return ({'Dataset': True, 'BasicTargetedDataset': False, 'QC': False, 'sampleMetadata': False})

        # If it's not a Dataset, no point checking anything more
        else:
            # try logging
            try:
                self.Attributes['Log'].append([datetime.now(), 'Failed basic TargetedDataset validation, Failed Dataset validation'])
            except (AttributeError, KeyError, TypeError):
                if verbose:
                    print('--------')
                    print('Logging failed')
            # print results
            if verbose:
                print('--------')
                print('Does not conform to Dataset')
                print('Does not conform to basic TargetedDataset')
                print('Does not have QC parameters')
                print('Does not have sample metadata information')
            # output
            if raiseWarning:
                warnings.warn('Does not conform to basic TargetedDataset')
                warnings.warn('Does not have QC parameters')
                warnings.warn('Does not have sample metadata information')
            return ({'Dataset': False, 'BasicTargetedDataset': False, 'QC': False, 'sampleMetadata': False})


    def applyMasks(self):
        """
        Permanently delete elements masked (those set to ``False``) in :py:attr:`~Dataset.sampleMask` and :py:attr:`~Dataset.featureMask`, from :py:attr:`~Dataset.featureMetadata`, :py:attr:`~Dataset.sampleMetadata`, :py:attr:`~Dataset.intensityData` and py:attr:`TargetedDataset.expectedConcentration`.

        Features are excluded in each :py:attr:`~TargetedDataset.calibration` based on the internal :py:attr:`~TargetedDataset.calibration['calibFeatureMetadata']` (iterate through the list of calibration if 2+ datasets have been joined with :py:meth:`~TargetedDataset.__add__`).
        """

        def findAndRemoveFeatures(calibDict, featureNameList):
            """
            Finds and remove all features with Feature Name in featureNameList, from the numpy.ndarray and pandas.Dataframe in calibDict.
            Do not expect features in calibration ordered the same as in featureMetadata (but it should), therefore work on feature names.

            :param calibDict: self.calibration dictionary
            :param featureNameList: list of Feature Name to remove
            :return: newCalibDict with feature removed
            """

            # init new mask
            toRemoveFeatMask = calibDict['calibFeatureMetadata']['Feature Name'].isin(featureNameList).values  # True for feature to remove

            newCalibDict = dict()
            newCalibDict['calibSampleMetadata'] = calibDict['calibSampleMetadata']

            # resize all frames
            dictKeys = set(calibDict.keys()) - set(['calibFeatureMetadata', 'calibSampleMetadata'])
            for i in dictKeys:
                # numpy.ndarray
                if isinstance(calibDict[i], numpy.ndarray):
                    newCalibDict[i] = calibDict[i][:, ~toRemoveFeatMask]
                # pandas.DataFrame
                elif isinstance(calibDict[i], pandas.DataFrame):
                    newCalibDict[i] = calibDict[i].loc[:, ~toRemoveFeatMask]
                else:
                    newCalibDict[i] = calibDict[i]

            # calibFeatureMetadata
            newCalibDict['calibFeatureMetadata'] = calibDict['calibFeatureMetadata'].loc[~toRemoveFeatMask, :]
            newCalibDict['calibFeatureMetadata'].reset_index(drop=True, inplace=True)

            return newCalibDict

        # Only filter TargetedDataset.expectedConcentration as it is not present in Dataset, others are done in Dataset.applyMasks
        if (sum(self.sampleMask == False) > 0) | (sum(self.featureMask == False) > 0):

            # Instantiate lists if first application
            if not hasattr(self, 'sampleMetadataExcluded'):
                self.expectedConcentrationExcluded = []

            # Samples
            if sum(self.sampleMask) != len(self.sampleMask):
                # Account for if self.sampleMask is a pandas.series
                try:
                    self.sampleMask = self.sampleMask.values
                except:
                    pass

                # Save excluded samples
                self.expectedConcentrationExcluded.append(self.expectedConcentration.loc[~self.sampleMask, :])
                # Delete excluded samples
                self.expectedConcentration = self.expectedConcentration.loc[self.sampleMask]
                self.expectedConcentration.reset_index(drop=True, inplace=True)

            # Features
            if sum(self.featureMask) != len(self.featureMask):
                # Account for if self.featureMask is a pandas.series
                try:
                    self.featureMask = self.featureMask.values
                except:
                    pass
                # Start by removing features from self.calibration
                featureNameList = self.featureMetadata['Feature Name'].values[~self.featureMask].tolist()
                # list of dict if 2+ joined targetedDatasets
                if isinstance(self.calibration, list):
                    # remove in each calibration
                    for j in range(len(self.calibration)):
                        self.calibration[j] = findAndRemoveFeatures(self.calibration[j], featureNameList)
                # dict 1 targetedDataset
                elif isinstance(self.calibration, dict):
                    self.calibration = findAndRemoveFeatures(self.calibration, featureNameList)

                # Save excluded features
                self.expectedConcentrationExcluded.append(self.expectedConcentration.loc[:, ~self.featureMask])
                # Delete excluded features
                self.expectedConcentration = self.expectedConcentration.loc[:, self.featureMask]
                self.expectedConcentration.reset_index(drop=True, inplace=True)

        # applyMasks to the rest of TargetedDataset
        super().applyMasks()


    def updateMasks(self, filterSamples=True, filterFeatures=True, sampleTypes=[SampleType.StudySample, SampleType.StudyPool],
                    assayRoles=[AssayRole.Assay, AssayRole.PrecisionReference],
                    quantificationTypes=[QuantificationType.IS, QuantificationType.QuantOwnLabeledAnalogue, QuantificationType.QuantAltLabeledAnalogue, QuantificationType.QuantOther, QuantificationType.Monitored],
                    calibrationMethods=[CalibrationMethod.backcalculatedIS, CalibrationMethod.noIS, CalibrationMethod.noCalibration, CalibrationMethod.otherCalibration],
                    rsdThreshold=None, **kwargs):
        """
        Update :py:attr:`~Dataset.sampleMask` and :py:attr:`~Dataset.featureMask` according to QC parameters.

        :py:meth:`updateMasks` sets :py:attr:`~Dataset.sampleMask` or :py:attr:`~Dataset.featureMask` to ``False`` for those items failing analytical criteria.

        Similar to :py:meth:`~MSDataset.updateMasks`, without `blankThreshold` or `artifactual` filtering

        .. note:: To avoid reintroducing items manually excluded, this method only ever sets items to ``False``, therefore if you wish to move from more stringent criteria to a less stringent set, you will need to reset the mask to all ``True`` using :py:meth:`~Dataset.initialiseMasks`.

        :param bool filterSamples: If ``False`` don't modify sampleMask
        :param bool filterFeatures: If ``False`` don't modify featureMask
        :param sampleTypes: List of types of samples to retain
        :type sampleTypes: SampleType
        :param assayRoles: List of assays roles to retain
        :type assayRoles: AssayRole
        :param quantificationTypes: List of quantification types to retain
        :type quantificationTypes: QuantificationType
        :param calibrationMethods: List of calibratio methods to retain
        :type calibrationMethods: CalibrationMethod
        :raise TypeError: if sampleTypes is not a list
        :raise TypeError: if sampleTypes are not a SampleType enum
        :raise TypeError: if assayRoles is not a list
        :raise TypeError: if assayRoles are not an AssayRole enum
        :raise TypeError: if quantificationTypes is not a list
        :raise TypeError: if quantificationTypes are not a QuantificationType enum
        :raise TypeError: if calibrationMethods is not a list
        :raise TypeError: if calibrationMethods are not a CalibrationMethod enum
        """
        # Check sampleTypes, assayRoles, quantificationTypes and calibrationMethods are lists
        if not isinstance(sampleTypes, list):
            raise TypeError('sampleTypes must be a list of SampleType enums')
        if not isinstance(assayRoles, list):
            raise TypeError('assayRoles must be a list of AssayRole enums')
        if not isinstance(quantificationTypes, list):
            raise TypeError('quantificationTypes must be a list of QuantificationType enums')
        if not isinstance(assayRoles, list):
            raise TypeError('calibrationMethods must be a list of CalibrationMethod enums')
        # Check sampleTypes, assayRoles, quantificationTypes and calibrationMethods are enums
        if not all(isinstance(item, SampleType) for item in sampleTypes):
            raise TypeError('sampleTypes must be SampleType enums.')
        if not all(isinstance(item, AssayRole) for item in assayRoles):
            raise TypeError('assayRoles must be AssayRole enums.')
        if not all(isinstance(item, QuantificationType) for item in quantificationTypes):
            raise TypeError('quantificationTypes must be QuantificationType enums.')
        if not all(isinstance(item, CalibrationMethod) for item in calibrationMethods):
            raise TypeError('calibrationMethods must be CalibrationMethod enums.')
        if rsdThreshold is None:
            if 'rsdThreshold' in self.Attributes:
                rsdThreshold = self.Attributes['rsdThreshold']
            else:
                rsdThreshold = None
        if rsdThreshold is not None and not isinstance(rsdThreshold, (float, int)):
            raise TypeError('rsdThreshold should either be a float or None')

        # Feature Exclusions
        if filterFeatures:
            quantTypeMask = self.featureMetadata['quantificationType'].isin(quantificationTypes)
            calibMethodMask = self.featureMetadata['calibrationMethod'].isin(calibrationMethods)

            featureMask = numpy.logical_and(quantTypeMask, calibMethodMask).values

            self.featureMask = numpy.logical_and(featureMask, self.featureMask)
            if rsdThreshold is not None:
                self.featureMask &= self.rsdSP <= rsdThreshold

            self.featureMetadata['Passing Selection'] = self.featureMask

        # Sample Exclusions
        if filterSamples:
            sampleMask = self.sampleMetadata['SampleType'].isin(sampleTypes)
            assayMask = self.sampleMetadata['AssayRole'].isin(assayRoles)

            sampleMask = numpy.logical_and(sampleMask, assayMask).values

            self.sampleMask = numpy.logical_and(sampleMask, self.sampleMask)

        self.Attributes['Log'].append([datetime.now(), 'Dataset filtered with: filterSamples=%s, filterFeatures=%s, sampleTypes=%s, assayRoles=%s, quantificationTypes=%s, calibrationMethods=%s' % (filterSamples, filterFeatures, sampleTypes, assayRoles, quantificationTypes, calibrationMethods)])


    def addSampleInfo(self, descriptionFormat=None, filePath=None, **kwargs):
        """
        Load additional metadata and map it in to the :py:attr:`~Dataset.sampleMetadata` table.

        Possible options:

        * **'NPC Subject Info'** Map subject metadata from a NPC sample manifest file (format defined in 'PCSOP.082')
        * **'Raw Data'** Extract analytical parameters from raw data files
        * **'ISATAB'** ISATAB study designs
        * **'Filenames'** Parses sample information out of the filenames, based on the named capture groups in the regex passed in *filenamespec*
        * **'Basic CSV'** Joins the :py:attr:`sampleMetadata` table with the data in the ``csv`` file at *filePath=*, matching on the 'Sample File Name' column in both.
        * **'Batches'** Interpolate batch numbers for samples between those with defined batch numbers based on sample acquisitions times

        :param str descriptionFormat: Format of metadata to be added
        :param str filePath: Path to the additional data to be added
        :param filenameSpec: Only used if *descriptionFormat* is 'Filenames'. A regular expression that extracts sample-type information into the following named capture groups: 'fileName', 'baseName', 'study', 'chromatography' 'ionisation', 'instrument', 'groupingKind' 'groupingNo', 'injectionKind', 'injectionNo', 'reference', 'exclusion' 'reruns', 'extraInjections', 'exclusion2'. if ``None`` is passed, use the *filenameSpec* key in *Attributes*, loaded from the SOP json
        :type filenameSpec: None or str
        :raises NotImplementedError: if the descriptionFormat is not understood
        """

        if descriptionFormat == 'Filenames':
            filenameSpec = kwargs.get('filenameSpec', None) # default to None if not provided
            if filenameSpec is None:
                raise AttributeError('A \'filenameSpec\' must be provided with \'descriptionFormat==\'Filenames\'\'')
            self._getSampleMetadataFromFilename(filenameSpec)
        elif descriptionFormat == 'Batches':
            self._fillBatches()
        else:
            super().addSampleInfo(descriptionFormat=descriptionFormat, filePath=filePath, **kwargs)


    def _matchDatasetToLIMS(self, pathToLIMSfile):
        """
        Establish the `Sampling ID` by matching the `Sample Base Name` with the LIMS file information.

        :param str pathToLIMSfile: Path to LIMS file for map Sampling ID
        """

        # Detect if requires NMR specific alterations
        if 'expno' in self.sampleMetadata.columns:
            from . import NMRDataset
            NMRDataset._matchDatasetToLIMS(self,pathToLIMSfile)
        else:
            super()._matchDatasetToLIMS(pathToLIMSfile)


    def _getSampleMetadataFromFilename(self, filenameSpec):
        """
        Infer sample acquisition metadata from standardised filename template.
        Similar to :py:meth:`~MSDataset._getSampleMetadataFromFilename`
        """

        # If the dilution series design is not defined in the SOP, load the default.
        if not 'dilutionMap' in self.Attributes.keys():
            dilutionMap = pandas.read_csv(os.path.join(toolboxPath(), 'StudyDesigns', 'DilutionSeries.csv'), index_col='Sample Name')
            self.Attributes['dilutionMap'] = dilutionMap['Dilution Factor (%)'].to_dict()

        # Strip any whitespace from 'Sample File Name'
        self.sampleMetadata['Sample File Name'] = self.sampleMetadata['Sample File Name'].str.strip()

        # Break filename down into constituent parts.
        baseNameParser = re.compile(filenameSpec, re.VERBOSE)
        fileNameParts = self.sampleMetadata['Sample File Name'].str.extract(baseNameParser, expand=False)

        # Deal with badly ordered exclusions
        fileNameParts['exclusion'].loc[fileNameParts['exclusion2'].isnull() == False] = fileNameParts['exclusion2'].loc[fileNameParts['exclusion2'].isnull() == False]
        fileNameParts.drop('exclusion2', axis=1, inplace=True)

        # Pass masks into enum fields
        fileNameParts.loc[:, 'AssayRole'] = AssayRole.Assay
        fileNameParts.loc[fileNameParts['reference'] == 'SR', 'AssayRole'] = AssayRole.PrecisionReference
        fileNameParts.loc[fileNameParts['baseName'].str.match('.+[B]\d+?[SE]\d+?', na=False).astype(bool), 'AssayRole'] = AssayRole.PrecisionReference
        fileNameParts.loc[fileNameParts['reference'] == 'LTR', 'AssayRole'] = AssayRole.PrecisionReference
        fileNameParts.loc[fileNameParts['reference'] == 'MR', 'AssayRole'] = AssayRole.PrecisionReference
        fileNameParts.loc[fileNameParts['injectionKind'] == 'SRD', 'AssayRole'] = AssayRole.LinearityReference
        fileNameParts.loc[fileNameParts['groupingKind'].str.match('Blank', na=False).astype(bool), 'AssayRole'] = AssayRole.LinearityReference
        fileNameParts.loc[fileNameParts['groupingKind'].str.match('E?IC', na=False).astype(bool), 'AssayRole'] = AssayRole.Assay

        fileNameParts.loc[:, 'SampleType'] = SampleType.StudySample
        fileNameParts.loc[fileNameParts['reference'] == 'SR', 'SampleType'] = SampleType.StudyPool
        fileNameParts.loc[fileNameParts['baseName'].str.match('.+[B]\d+?[SE]\d+?', na=False).astype(bool), 'SampleType'] = SampleType.StudyPool
        fileNameParts.loc[fileNameParts['reference'] == 'LTR', 'SampleType'] = SampleType.ExternalReference
        fileNameParts.loc[fileNameParts['reference'] == 'MR', 'SampleType'] = SampleType.MethodReference
        fileNameParts.loc[fileNameParts['injectionKind'] == 'SRD', 'SampleType'] = SampleType.StudyPool
        fileNameParts.loc[fileNameParts['groupingKind'].str.match('Blank', na=False).astype(bool), 'SampleType'] = SampleType.ProceduralBlank
        fileNameParts.loc[fileNameParts['groupingKind'].str.match('E?IC', na=False).astype(bool), 'SampleType'] = SampleType.StudyPool

        # Skipped runs
        fileNameParts['Skipped'] = fileNameParts['exclusion'].str.match('[Xx]', na=False)

        # Get matrix
        fileNameParts['Matrix'] = fileNameParts['groupingKind'].str.extract('^([AC-Z]{1,2})(?<!IC)$', expand=False)
        fileNameParts['Matrix'].fillna('', inplace=True)

        # Get well numbers
        fileNameParts.loc[
            fileNameParts['groupingKind'].str.match('Blank|E?IC', na=False).astype(bool), 'injectionNo'] = -1
        fileNameParts['Well'] = pandas.to_numeric(fileNameParts['injectionNo'])

        # Plate / grouping no
        fileNameParts['Plate'] = pandas.to_numeric(fileNameParts['groupingNo'])

        # Get batch where it is explicit in file name
        fileNameParts['Batch'] = pandas.to_numeric(fileNameParts['baseName'].str.extract('B(\d+?)[SE]', expand=False))
        fileNameParts['Correction Batch'] = numpy.nan

        # Map dilution series names to dilution level
        fileNameParts['Dilution'] = fileNameParts['baseName'].str.extract('(?:.+_?)(SRD\d\d)(?:_?.*)', expand=False).replace(self.Attributes['dilutionMap'])
        fileNameParts['Dilution'] = fileNameParts['Dilution'].astype(float)
        # Blank out NAs for neatness
        fileNameParts['reruns'].fillna('', inplace=True)
        fileNameParts['extraInjections'].fillna('', inplace=True)

        # Drop unwanted columns
        fileNameParts.drop(['exclusion', 'reference', 'groupingKind', 'injectionNo', 'injectionKind', 'groupingNo'], axis=1, inplace=True)

        # Swap in user freindly file names
        fileNameParts.rename(columns={'chromatography': 'Chromatography'}, inplace=True)
        fileNameParts.rename(columns={'instrument': 'Instrument'}, inplace=True)
        fileNameParts.rename(columns={'study': 'Study'}, inplace=True)
        fileNameParts.rename(columns={'baseName': 'Sample Base Name'}, inplace=True)
        fileNameParts.rename(columns={'fileName': 'Sample File Name'}, inplace=True)
        fileNameParts.rename(columns={'suplementalInfo': 'Suplemental Info'}, inplace=True)
        fileNameParts.rename(columns={'ionisation': 'Ionisation'}, inplace=True)
        fileNameParts.rename(columns={'extraInjections': 'Suplemental Injections'}, inplace=True)
        fileNameParts.rename(columns={'reruns': 'Re-Run'}, inplace=True)

        # Merge metadata back into the sampleInfo table.
        # first remove duplicate columns (from _dataset _init_)
        if 'AssayRole' in self.sampleMetadata.columns: self.sampleMetadata.drop(['AssayRole'], axis=1, inplace=True)
        if 'SampleType' in self.sampleMetadata.columns: self.sampleMetadata.drop(['SampleType'], axis=1, inplace=True)
        if 'Sample Base Name' in self.sampleMetadata.columns: self.sampleMetadata.drop(['Sample Base Name'], axis=1, inplace=True)
        if 'Dilution' in self.sampleMetadata.columns: self.sampleMetadata.drop(['Dilution'], axis=1, inplace=True)
        if 'Batch' in self.sampleMetadata.columns: self.sampleMetadata.drop(['Batch'], axis=1, inplace=True)
        if 'Correction Batch' in self.sampleMetadata.columns: self.sampleMetadata.drop(['Correction Batch'], axis=1, inplace=True)
        # merge
        self.sampleMetadata = pandas.merge(self.sampleMetadata, fileNameParts, left_on='Sample File Name', right_on='Sample File Name', how='left', sort=False)

        # Add 'Exclusion Details' column
        self.sampleMetadata['Exclusion Details'] = ''

        self.Attributes['Log'].append([datetime.now(), 'Sample metadata parsed from filenames.'])


    def _fillBatches(self):
        """
        Use sample names and acquisition times to infer batch info
        Similar to :py:meth:`~MSDataset._fillBatches`
        """

        batchRE = r"""
            B
            (?P<observebatch>\d+?)
            (?P<startend>[SE])
            (?P<sequence>\d+?)
            _SR
            (?:_(?P<extraInjections>\d+?|\w+?))?
            $
            """
        batchRE = re.compile(batchRE, re.VERBOSE)
        # We canot infer batches unless we have runorder
        if 'Run Order' in self.sampleMetadata.keys():
            currentBatch = 0
            # Loop over samples in run order
            for index, row in self.sampleMetadata.sort_values(by='Run Order').iterrows():
                nameComponents = batchRE.search(row['Sample File Name'])
                if nameComponents:
                    # Batch start
                    if nameComponents.group('startend') == 'S':
                        # New batch - increment batch no
                        if nameComponents.group('sequence') == '1':
                            currentBatch = currentBatch + 1

                # Don't include the dilution series or blanks
                if not ((row['AssayRole'] == AssayRole.LinearityReference) or (row['SampleType'] == SampleType.ProceduralBlank)):
                    self.sampleMetadata.loc[index, 'Batch'] = currentBatch
                    self.sampleMetadata.loc[index, 'Correction Batch'] = currentBatch

        else:
            warnings.warn('Unable to infer batches without run order, skipping.')
            return


    def accuracyPrecision(self, onlyPrecisionReferences=False):
        """
        Return Precision (percent RSDs) and Accuracy for each SampleType and each unique concentration.
        Statistic grouped by SampleType, Feature and unique concentration.

        :param TargetedDataset dataset: TargetedDataset object to generate the accuracy and precision for.
        :param bool onlyPrecisionReference: If ``True`` only use samples with the `AssayRole` PrecisionReference.
        :returns: Dict of Accuracy and Precision dict for each group.
        :rtype: dict(str:dict(str:pandas.DataFrame))
        :raises TypeError: if dataset is not an instance of TargetedDataset
        """
        #from ..enumerations import AssayRole
        #from ..objects import TargetedDataset

        def calcAccuracy(measuredConc, expectedConc):
            """
            Calculate the accuracy of measurement for a column of data.
            accuracy = (mean(measuredConcentration)/expectedConcentration)*100

            :param numpy.ndarray measuredConc: *n* by 1 numpy array of data, with a single feature in column, and samples in rows
            :param float expectedConc: expected concentration
            :return: accuracy value
            :rtype: float
            """
            accuracy = (numpy.mean(measuredConc) / expectedConc) * 100
            return accuracy

        def calcPrecision(measuredConc):
            """
            Calculate the precision of measurement (percent RSD) for a column of data.
            Allow for -inf, inf values in input.

            :param numpy.ndarray measuredConc: *n* by 1 numpy array of data, with a single feature in column, and samples in rows
            :return: precisin value
            :rtype: float
            """
            std = numpy.std(measuredConc)
            rsd = (std / numpy.mean(measuredConc)) * 100
            if numpy.isnan(rsd):
                rsd = numpy.inf
            return rsd

        #if not isinstance(dataset, TargetedDataset):
        #    raise TypeError('dataset must be an instance of TargetedDataset.')

        # Init
        accuracy = dict()
        precision = dict()

        # Restrict to PrecisionReference if necessary
        if onlyPrecisionReferences:
            startMask = (self.sampleMetadata['AssayRole'].values == AssayRole.PrecisionReference)
        else:
            startMask = numpy.squeeze(numpy.ones([self.sampleMetadata.shape[0], 1], dtype=bool), axis=1)

        # Unique concentrations
        uniqueConc = pandas.unique(self.expectedConcentration.loc[startMask, :].values.ravel()).tolist()
        uniqueConc = sorted([x for x in uniqueConc if str(x) != 'nan'])

        # Each SampleType
        sampleTypes = self.sampleMetadata['SampleType'].unique()
        for sampleType in sampleTypes:
            # init
            acc =  pandas.DataFrame(numpy.full([len(uniqueConc), self.featureMetadata.shape[0]], numpy.nan), index=uniqueConc, columns=self.featureMetadata['Feature Name'].values)
            prec = pandas.DataFrame(numpy.full([len(uniqueConc), self.featureMetadata.shape[0]], numpy.nan), index=uniqueConc, columns=self.featureMetadata['Feature Name'].values)
            # Restrict to sampleType
            # Allow for the case where sampleType is not defined
            if pandas.isnull(sampleType):
                sampleTypeMask = numpy.logical_and(startMask, self.sampleMetadata['SampleType'].isnull())
            else:
                sampleTypeMask = numpy.logical_and(startMask, self.sampleMetadata['SampleType'].values == sampleType)
            # Each feature
            for feat in self.featureMetadata['Feature Name'].tolist():
                # Each unique concentrations
                for conc in uniqueConc:
                    # Restrict to concentration
                    mask = numpy.logical_and(sampleTypeMask, self.expectedConcentration[feat].values == conc)
                    # minimum of samples
                    if sum(mask) < 2:
                        continue
                    # fill accuracy/precision df
                    featID = (self.featureMetadata['Feature Name'] == feat).values
                    acc.loc[conc, feat] = calcAccuracy(self.intensityData[mask, featID], conc)
                    prec.loc[conc, feat] = calcPrecision(self.intensityData[mask, featID])
            # Store accuracy/precision + clean empty rows
            accuracy[sampleType] = acc.dropna(axis=0, how='all')
            precision[sampleType] = prec.dropna(axis=0, how='all')

        # All samples
        acc  = pandas.DataFrame(numpy.full([len(uniqueConc), self.featureMetadata.shape[0]], numpy.nan), index=uniqueConc, columns=self.featureMetadata['Feature Name'].values)
        prec = pandas.DataFrame(numpy.full([len(uniqueConc), self.featureMetadata.shape[0]], numpy.nan), index=uniqueConc, columns=self.featureMetadata['Feature Name'].values)
        # Each feature
        for feat in self.featureMetadata['Feature Name'].tolist():
            # Each unique concentrations
            for conc in uniqueConc:
                # Restrict to concentration
                mask = numpy.logical_and(startMask, self.expectedConcentration[feat].values == conc)
                # minimum of samples
                if sum(mask) < 2:
                    continue
                # fill accuracy/precision df
                featID = (self.featureMetadata['Feature Name'] == feat).values
                acc.loc[conc, feat] = calcAccuracy(self.intensityData[mask, featID], conc)
                prec.loc[conc, feat] = calcPrecision(self.intensityData[mask, featID])
        # Store accuracy/precision
        accuracy['All Samples'] = acc.dropna(axis=0, how='all')
        precision['All Samples'] = prec.dropna(axis=0, how='all')

        # Output
        return {'Accuracy': accuracy, 'Precision': precision}


def main():
    pass

if __name__=='__main__':
    main()
