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
from ..utilities.iotools._importBrukerIvDr import importBrukerXML
from ..utilities.extractParams import buildFileList
from ..enumerations import VariableType, AssayRole, SampleType, QuantificationType, CalibrationMethod, \
    AnalyticalPlatform


class NMRTargetedDataset(Dataset):
    """
    NMRTargetedDataset(dataPath, fileType='Bruker', sop='Generic', \*\*kwargs)

    :py:class:`~NMRTargetedDataset` extends :py:class:`Dataset` to represent quantitative NMR datasets, where compounds were extracted from the standard NMR experiments, through means of targeted curve-fitting or other targeted approaches. By definition, these are identified and the concentration units known.
    The :py:class:`~NMRTargetedDataset` class include methods to apply limits of detection/quantification (LOD/LOQ), merge multiple batches precision for each measurement

    In addition to the structure of :py:class:`~Dataset`, :py:class:`~NMRTargetedDataset` requires the following attributes:

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


    Currently **Bruker quantification results** obtained from Bruker Biospin ivDr methods can be imported.
    To create an import for any other form of semi-quantitative or quantitative results, the procedure is as follow:

        * Create a new ``fileType == 'myMethod'`` entry in :py:meth:`~NMRTargetedDataset.__init__`
        * Define functions to populate all expected dataframes (using file readers, JSON,...)
        * Separate calibration samples from study samples (store in :py:attr:`~NMRTargetedDataset.calibration`). *If none exist, intialise empty dataframes with the correct number of columns and column names.*
        * Execute pre-processing steps if required (note: all feature values should be expressed in the unit listed in :py:attr:`~NMRTargetedDataset.featureMetadata['Unit']`)
        * Apply limits of quantification using :py:meth:`~NMRTargetedDataset._applyLimitsOfQuantification`.

    The resulting :py:class:`~NMRTargetedDatset` created must satisfy to the criteria for *TargetedDataset*, which can be checked with :py:meth:`~TargetedDataset.validatedObject` (list the minimum requirements for all class methods).

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

    def __init__(self, datapath, fileType='BrukerQuant-UR', sop='Generic', **kwargs):
        """
        Initialisation and pre-processing of input data (load files and match data and calibration and SOP, apply limits of quantification if needed).
        """

        super().__init__(sop=sop, **kwargs)
        self.filePath, fileName = os.path.split(datapath)
        self.fileName, fileExtension = os.path.splitext(fileName)

        self.name = self.fileName

        # Load files and match data, calibration report and SOP, then Apply the limits of quantification
        if fileType == 'Bruker Quantification':
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
                raise ValueError(
                    'Import Error: The imported dataset does not satisfy to the Basic TargetedDataset definition')
            elif not validDataset['NMRTargetedDataset']:
                raise ValueError(
                    'Import Error: The imported dataset does not satisfy to the NMRTargetedDataset definition')
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
                raise ValueError(
                    'The unit \'' + str(unit) + '\' is not present in the input data, available units: ' + str(avUnit))
            keepMask = (self.featureMetadata['Unit'] == unit).values
            self.featureMetadata = self.featureMetadata.loc[keepMask, :]
            self.featureMetadata.reset_index(drop=True, inplace=True)
            self.intensityData = self.intensityData[:, keepMask]

        ## Check all features are unique, and
        u_ids, u_counts = numpy.unique(self.featureMetadata['Feature Name'], return_counts=True)
        if not all(u_counts == 1):
            dupFeat = u_ids[u_counts != 1].tolist()
            warnings.warn(
                'The following features are present more than once, only the first occurence will be kept: ' + str(
                    dupFeat) + '. For further filtering, available units are: ' + str(avUnit))
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
        self.featureMetadata.loc[
            self.featureMetadata['type'] == 'quantification', 'quantificationType'] = QuantificationType.QuantOther
        self.featureMetadata.loc[
            self.featureMetadata['type'] != 'quantification', 'quantificationType'] = QuantificationType.Monitored
        self.featureMetadata.drop('type', inplace=True, axis=1)
        # calibrationMethod
        self.featureMetadata['calibrationMethod'] = numpy.nan
        self.featureMetadata.loc[self.featureMetadata[
                                     'quantificationType'] == QuantificationType.QuantOther, 'calibrationMethod'] = CalibrationMethod.otherCalibration
        self.featureMetadata.loc[self.featureMetadata[
                                     'quantificationType'] == QuantificationType.Monitored, 'calibrationMethod'] = CalibrationMethod.noCalibration
        # rename columns
        self.featureMetadata.rename(
            columns={'loq': 'LLOQ', 'lod': 'LOD', 'Lower Reference Bound': 'Lower Reference Percentile',
                     'Upper Reference Bound': 'Upper Reference Percentile'}, inplace=True)
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
        self.expectedConcentration = pandas.DataFrame(None, index=list(self.sampleMetadata.index),
                                                      columns=self.featureMetadata['Feature Name'].tolist())

        ## Initialise empty Calibration info
        self.calibration = dict()
        self.calibration['calibIntensityData'] = numpy.ndarray((0, self.featureMetadata.shape[0]))
        self.calibration['calibSampleMetadata'] = pandas.DataFrame(None, columns=self.sampleMetadata.columns)
        self.calibration['calibSampleMetadata']['Metadata Available'] = False
        self.calibration['calibFeatureMetadata'] = pandas.DataFrame(
            {'Feature Name': self.featureMetadata['Feature Name'].tolist()})
        self.calibration['calibExpectedConcentration'] = pandas.DataFrame(None, columns=self.featureMetadata[
            'Feature Name'].tolist())

        ## Summary
        print('Targeted Method: ' + self.Attributes['methodName'])
        print(str(self.sampleMetadata.shape[0]) + ' study samples')
        print(str(self.featureMetadata.shape[0]) + ' features (' + str(
            sum(self.featureMetadata['quantificationType'] == QuantificationType.IS)) + ' IS, ' + str(sum(
            self.featureMetadata[
                'quantificationType'] == QuantificationType.QuantOwnLabeledAnalogue)) + ' quantified and validated with own labeled analogue, ' + str(
            sum(self.featureMetadata[
                    'quantificationType'] == QuantificationType.QuantAltLabeledAnalogue)) + ' quantified and validated with alternative labeled analogue, ' + str(
            sum(self.featureMetadata[
                    'quantificationType'] == QuantificationType.QuantOther)) + ' other quantification, ' + str(sum(
            self.featureMetadata[
                'quantificationType'] == QuantificationType.Monitored)) + ' monitored for relative information)')
        print('-----')

        ## Apply limit of quantification by default??
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

    def _applyLimitsOfQuantification(self, **kwargs):
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

        sampleMetadata = copy.deepcopy(self.sampleMetadata)
        featureMetadata = copy.deepcopy(self.featureMetadata)
        intensityData = copy.deepcopy(self._intensityData)

        if ((not hasattr(self, 'sampleMetadataExcluded')) | (not hasattr(self, 'featureMetadataExcluded')) | (
        not hasattr(self, 'intensityDataExcluded')) | (not hasattr(self, 'expectedConcentrationExcluded')) | (
        not hasattr(self, 'excludedFlag'))):
            sampleMetadataExcluded = []
            featureMetadataExcluded = []
            intensityDataExcluded = []
            excludedFlag = []
        else:
            sampleMetadataExcluded = copy.deepcopy(self.sampleMetadataExcluded)
            featureMetadataExcluded = copy.deepcopy(self.featureMetadataExcluded)
            intensityDataExcluded = copy.deepcopy(self.intensityDataExcluded)
            excludedFlag = copy.deepcopy(self.excludedFlag)

        ## Check input columns
        if 'LOD' not in featureMetadata.columns:
            raise AttributeError('the featureMetadata[\'LLOQ\'] column is absent')

        ## Features only Monitored are not processed and passed untouched (concatenated back at the end)
        untouched = (featureMetadata['quantificationType'] == QuantificationType.Monitored).values
        if sum(untouched) != 0:
            print('The following features are only monitored and therefore not processed for LOQs: ' + str(
                featureMetadata.loc[untouched, 'Feature Name'].values.tolist()))
            untouchedFeatureMetadata = featureMetadata.loc[untouched, :]
            featureMetadata = featureMetadata.loc[~untouched, :]
            untouchedIntensityData = intensityData[:, untouched]
            intensityData = intensityData[:, ~untouched]

        ## Values replacement (-inf / +inf)
        # iterate over the features
        for i in range(0, featureMetadata.shape[0]):
            # LOD/LOQ
            if not numpy.isnan(featureMetadata['LLOQ'].values[i]):
                toReplaceLLOQ = intensityData[:, i] < featureMetadata['LLOQ'].values[i]
                intensityData[toReplaceLLOQ, i] = -numpy.inf

        # Remove excess info
        featureMetadata.reset_index(drop=True, inplace=True)

        ## return dataset with limits of quantification applied
        self.featureMetadata = featureMetadata
        self._intensityData = intensityData
        self.sampleMetadataExcluded = sampleMetadataExcluded
        self.featureMetadataExcluded = featureMetadataExcluded
        self.intensityDataExcluded = intensityDataExcluded
        self.excludedFlag = excludedFlag

        ## Output and Log
        print('Values <LLOQ replaced by -inf')

        # log the modifications
        logLimits = 'Limits of quantification applied to LLOQ'

        self.Attributes['Log'].append([datetime.now(), '%s (%i samples, %i features). LLOQ are replaced by -inf.%s' % (
        logLimits, self.noSamples, self.noFeatures)])

    def exportDataset(self, destinationPath='.', saveFormat='CSV', withExclusions=True, escapeDelimiters=False,
                      filterMetadata=True):
        """
        Calls :py:meth:`~Dataset.exportDataset` and raises a warning if normalisation is employed as :py:class:`TargetedDataset` :py:attr:`intensityData` can be left-censored.
        """
        # handle the dilution due to method... hopefully this will be handled more elegantly through a getter
        # elegantly through the intensityData getter
        # Export dataset...
        tmpData = copy.deepcopy(self)
        tmpData._intensityData = tmpData._intensityData * (100 / tmpData.sampleMetadata['Dilution']).values[:,
                                                          numpy.newaxis]

        super(NMRTargetedDataset, tmpData).exportDataset(destinationPath=destinationPath, saveFormat=saveFormat,
                                                      withExclusions=withExclusions, escapeDelimiters=escapeDelimiters,
                                                      filterMetadata=filterMetadata)

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

        # Export sample metadata
        sampleMetadata.to_csv(destinationPath + '_sampleMetadata.csv', encoding='utf-8',
                              date_format=self._timestampFormat)

        # Export feature metadata
        featureMetadata.to_csv(destinationPath + '_featureMetadata.csv', encoding='utf-8')

        # Export intensity data
        intensityData.to_csv(os.path.join(destinationPath + '_intensityData.csv'), encoding='utf-8',
                             date_format=self._timestampFormat, header=False, index=False)

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
        tmpCombined.to_csv(os.path.join(destinationPath + '_combinedData.csv'), encoding='utf-8',
                           date_format=self._timestampFormat)

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
            toRemoveFeatMask = calibDict['calibFeatureMetadata']['Feature Name'].isin(
                featureNameList).values  # True for feature to remove

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

    def updateMasks(self, filterSamples=True, filterFeatures=True,
                    sampleTypes=[SampleType.StudySample, SampleType.StudyPool],
                    assayRoles=[AssayRole.Assay, AssayRole.PrecisionReference],
                    quantificationTypes=[QuantificationType.QuantOther,
                                         QuantificationType.Monitored],
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
        if rsdThreshold is None:
            if 'rsdThreshold' in self.Attributes:
                rsdThreshold = self.Attributes['rsdThreshold']
            else:
                rsdThreshold = None
        if rsdThreshold is not None and not isinstance(rsdThreshold, (float, int)):
            raise TypeError('rsdThreshold should either be a float or None')

        # Feature Exclusions
        if filterFeatures:
            featureMask = self.featureMetadata['quantificationType'].isin(quantificationTypes)

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

        self.Attributes['Log'].append([datetime.now(),
                                       'Dataset filtered with: filterSamples=%s, filterFeatures=%s, sampleTypes=%s, assayRoles=%s, quantificationTypes=%s' % (
                                       filterSamples, filterFeatures, sampleTypes, assayRoles, quantificationTypes)])

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
            filenameSpec = kwargs.get('filenameSpec', None)  # default to None if not provided
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
            NMRDataset._matchDatasetToLIMS(self, pathToLIMSfile)
        else:
            super()._matchDatasetToLIMS(pathToLIMSfile)

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

        # if not isinstance(dataset, TargetedDataset):
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
            acc = pandas.DataFrame(numpy.full([len(uniqueConc), self.featureMetadata.shape[0]], numpy.nan),
                                   index=uniqueConc, columns=self.featureMetadata['Feature Name'].values)
            prec = pandas.DataFrame(numpy.full([len(uniqueConc), self.featureMetadata.shape[0]], numpy.nan),
                                    index=uniqueConc, columns=self.featureMetadata['Feature Name'].values)
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
        acc = pandas.DataFrame(numpy.full([len(uniqueConc), self.featureMetadata.shape[0]], numpy.nan),
                               index=uniqueConc, columns=self.featureMetadata['Feature Name'].values)
        prec = pandas.DataFrame(numpy.full([len(uniqueConc), self.featureMetadata.shape[0]], numpy.nan),
                                index=uniqueConc, columns=self.featureMetadata['Feature Name'].values)
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
        failureListQC = []
        failureListMeta = []
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
            condition = isinstance(self, NMRTargetedDataset)
            success = 'Check Object class:\tOK'
            failure = 'Check Object class:\tFailure, not TargetedDataset, but ' + str(type(self))
            failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose, raiseError,
                                             raiseWarning, exception=TypeError(failure))

            ## Attributes
            ## methodName
            # exist
            condition = 'methodName' in self.Attributes
            success = 'Check self.Attributes[\'methodName\'] exists:\tOK'
            failure = 'Check self.Attributes[\'methodName\'] exists:\tFailure, no attribute \'self.Attributes[\'methodName\']\''
            failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose, raiseError,
                                             raiseWarning, exception=AttributeError(failure))
            if condition:
                # is a str
                condition = isinstance(self.Attributes['methodName'], str)
                success = 'Check self.Attributes[\'methodName\'] is a str:\tOK'
                failure = 'Check self.Attributes[\'methodName\'] is a str:\tFailure, \'self.Attributes[\'methodName\']\' is ' + str(
                    type(self.Attributes['methodName']))
                failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose, raiseError,
                                                 raiseWarning, exception=TypeError(failure))
            # end self.Attributes['methodName']
            ## externalID
            # exist
            condition = 'externalID' in self.Attributes
            success = 'Check self.Attributes[\'externalID\'] exists:\tOK'
            failure = 'Check self.Attributes[\'externalID\'] exists:\tFailure, no attribute \'self.Attributes[\'externalID\']\''
            failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose, raiseError,
                                             raiseWarning, exception=AttributeError(failure))
            if condition:
                # is a list
                condition = isinstance(self.Attributes['externalID'], list)
                success = 'Check self.Attributes[\'externalID\'] is a list:\tOK'
                failure = 'Check self.Attributes[\'externalID\'] is a list:\tFailure, \'self.Attributes[\'externalID\']\' is ' + str(
                    type(self.Attributes['externalID']))
                failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose, raiseError,
                                                 raiseWarning, exception=TypeError(failure))
            # end self.Attributes['externalID']

            ## self.VariableType
            # is a enum VariableType
            condition = isinstance(self.VariableType, VariableType)
            success = 'Check self.VariableType is an enum \'VariableType\':\tOK'
            failure = 'Check self.VariableType is an enum \'VariableType\':\tFailure, \'self.VariableType\' is' + str(
                type(self.VariableType))
            failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose, raiseError,
                                             raiseWarning, exception=TypeError(failure))
            # end Variabletype

            ## self.fileName
            # exist
            condition = hasattr(self, 'fileName')
            success = 'Check self.fileName exists:\tOK'
            failure = 'Check self.fileName exists:\tFailure, no attribute \'self.fileName\''
            failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose, raiseError,
                                             raiseWarning, exception=AttributeError(failure))
            if condition:
                # is a str
                condition = isinstance(self.fileName, (str, list))
                success = 'Check self.fileName is a str or list:\tOK'
                failure = 'Check self.fileName is a str or list:\tFailure, \'self.fileName\' is ' + str(
                    type(self.fileName))
                failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose, raiseError,
                                                 raiseWarning, exception=TypeError(failure))
                if isinstance(self.fileName, list):
                    for i in range(len(self.fileName)):
                        condition = isinstance(self.fileName[i], (str))
                        success = 'Check self.filename[' + str(i) + '] is str:\tOK'
                        failure = 'Check self.filename[' + str(i) + '] is str:\tFailure, \'self.fileName[' + str(
                            i) + '] is' + str(type(self.fileName[i]))
                        failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose,
                                                         raiseError, raiseWarning, exception=TypeError(failure))
                    # end self.fileName list
            # end self.fileName

            ## self.filePath
            # exist
            condition = hasattr(self, 'filePath')
            success = 'Check self.filePath exists:\tOK'
            failure = 'Check self.filePath exists:\tFailure, no attribute \'self.filePath\''
            failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose, raiseError,
                                             raiseWarning, exception=AttributeError(failure))
            if condition:
                # is a str
                condition = isinstance(self.filePath, (str, list))
                success = 'Check self.filePath is a str or list:\tOK'
                failure = 'Check self.filePath is a str or list:\tFailure, \'self.filePath\' is ' + str(
                    type(self.filePath))
                failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose, raiseError,
                                                 raiseWarning, exception=TypeError(failure))
                if isinstance(self.filePath, list):
                    for i in range(len(self.filePath)):
                        condition = isinstance(self.filePath[i], (str))
                        success = 'Check self.filePath[' + str(i) + '] is str:\tOK'
                        failure = 'Check self.filePath[' + str(i) + '] is str:\tFailure, \'self.filePath[' + str(
                            i) + '] is' + str(type(self.filePath[i]))
                        failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose,
                                                         raiseError, raiseWarning, exception=TypeError(failure))
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
            failure = 'Check self.sampleMetadata number of samples (rows):\tFailure, \'self.sampleMetadata\' has ' + str(
                self.sampleMetadata.shape[0]) + ' samples, ' + str(refNumSamples) + 'expected'
            failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose, raiseError,
                                             raiseWarning, exception=ValueError(failure))
            if condition:
                # sampleMetadata['Sample File Name'] is str
                condition = isinstance(self.sampleMetadata['Sample File Name'][0], str)
                success = 'Check self.sampleMetadata[\'Sample File Name\'] is str:\tOK'
                failure = 'Check self.sampleMetadata[\'Sample File Name\'] is str:\tFailure, \'self.sampleMetadata[\'Sample File Name\']\' is ' + str(
                    type(self.sampleMetadata['Sample File Name'][0]))
                failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose, raiseError,
                                                 raiseWarning, exception=TypeError(failure))

                ## Fields required for QC
                # sampleMetadata['AssayRole'] is enum AssayRole
                condition = isinstance(self.sampleMetadata['AssayRole'][0], AssayRole)
                success = 'Check self.sampleMetadata[\'AssayRole\'] is an enum \'AssayRole\':\tOK'
                failure = 'Check self.sampleMetadata[\'AssayRole\'] is an enum \'AssayRole\':\tFailure, \'self.sampleMetadata[\'AssayRole\']\' is ' + str(
                    type(self.sampleMetadata['AssayRole'][0]))
                failureListQC = conditionTest(condition, success, failure, failureListQC, verbose, raiseError,
                                              raiseWarning, exception=TypeError(failure))
                # sampleMetadata['SampleType'] is enum SampleType
                condition = isinstance(self.sampleMetadata['SampleType'][0], SampleType)
                success = 'Check self.sampleMetadata[\'SampleType\'] is an enum \'SampleType\':\tOK'
                failure = 'Check self.sampleMetadata[\'SampleType\'] is an enum \'SampleType\':\tFailure, \'self.sampleMetadata[\'SampleType\']\' is ' + str(
                    type(self.sampleMetadata['SampleType'][0]))
                failureListQC = conditionTest(condition, success, failure, failureListQC, verbose, raiseError,
                                              raiseWarning, exception=TypeError(failure))
                # sampleMetadata['Dilution'] is an int or float
                condition = isinstance(self.sampleMetadata['Dilution'][0], (int, float, numpy.integer, numpy.floating))
                success = 'Check self.sampleMetadata[\'Dilution\'] is int or float:\tOK'
                failure = 'Check self.sampleMetadata[\'Dilution\'] is int or float:\tFailure, \'self.sampleMetadata[\'Dilution\']\' is ' + str(
                    type(self.sampleMetadata['Dilution'][0]))
                failureListQC = conditionTest(condition, success, failure, failureListQC, verbose, raiseError,
                                              raiseWarning, exception=TypeError(failure))
                # sampleMetadata['Batch'] is an int or float
                condition = isinstance(self.sampleMetadata['Batch'][0], (int, float, numpy.integer, numpy.floating))
                success = 'Check self.sampleMetadata[\'Batch\'] is int or float:\tOK'
                failure = 'Check self.sampleMetadata[\'Batch\'] is int or float:\tFailure, \'self.sampleMetadata[\'Batch\']\' is ' + str(
                    type(self.sampleMetadata['Batch'][0]))
                failureListQC = conditionTest(condition, success, failure, failureListQC, verbose, raiseError,
                                              raiseWarning, exception=TypeError(failure))
                # sampleMetadata['Correction Batch'] is an int or float
                condition = isinstance(self.sampleMetadata['Correction Batch'][0],
                                       (int, float, numpy.integer, numpy.floating))
                success = 'Check self.sampleMetadata[\'Correction Batch\'] is int or float:\tOK'
                failure = 'Check self.sampleMetadata[\'Correction Batch\'] is int or float:\tFailure, \'self.sampleMetadata[\'Correction Batch\']\' is ' + str(
                    type(self.sampleMetadata['Correction Batch'][0]))
                failureListQC = conditionTest(condition, success, failure, failureListQC, verbose, raiseError,
                                              raiseWarning, exception=TypeError(failure))
                # sampleMetadata['Run Order'] is an int
                condition = isinstance(self.sampleMetadata['Run Order'][0], (int, numpy.integer))
                success = 'Check self.sampleMetadata[\'Run Order\'] is int:\tOK'
                failure = 'Check self.sampleMetadata[\'Run Order\'] is int:\tFailure, \'self.sampleMetadata[\'Run Order\']\' is ' + str(
                    type(self.sampleMetadata['Run Order'][0]))
                failureListQC = conditionTest(condition, success, failure, failureListQC, verbose, raiseError,
                                              raiseWarning, exception=TypeError(failure))
                # sampleMetadata['Acquired Time'] is datetime.datetime
                condition = isinstance(self.sampleMetadata['Acquired Time'][0], datetime)
                success = 'Check self.sampleMetadata[\'Acquired Time\'] is datetime:\tOK'
                failure = 'Check self.sampleMetadata[\'Acquired Time\'] is datetime:\tFailure, \'self.sampleMetadata[\'Acquired Time\']\' is ' + str(
                    type(self.sampleMetadata['Acquired Time'][0]))
                failureListQC = conditionTest(condition, success, failure, failureListQC, verbose, raiseError,
                                              raiseWarning, exception=TypeError(failure))
                # sampleMetadata['Sample Base Name'] is str
                condition = isinstance(self.sampleMetadata['Sample Base Name'][0], str)
                success = 'Check self.sampleMetadata[\'Sample Base Name\'] is str:\tOK'
                failure = 'Check self.sampleMetadata[\'Sample Base Name\'] is str:\tFailure, \'self.sampleMetadata[\'Sample Base Name\']\' is ' + str(
                    type(self.sampleMetadata['Sample Base Name'][0]))
                failureListQC = conditionTest(condition, success, failure, failureListQC, verbose, raiseError,
                                              raiseWarning, exception=TypeError(failure))

                ## Sample metadata fields
                # ['Subject ID']
                condition = ('Subject ID' in self.sampleMetadata.columns)
                success = 'Check self.sampleMetadata[\'Subject ID\'] exists:\tOK'
                failure = 'Check self.sampleMetadata[\'Subject ID\'] exists:\tFailure, \'self.sampleMetadata\' lacks a \'Subject ID\' column'
                failureListMeta = conditionTest(condition, success, failure, failureListMeta, verbose, raiseError,
                                                raiseWarning, exception=LookupError(failure))
                if condition:
                    # sampleMetadata['Subject ID'] is str
                    condition = (self.sampleMetadata['Subject ID'].dtype == numpy.dtype('O'))
                    success = 'Check self.sampleMetadata[\'Subject ID\'] is str:\tOK'
                    failure = 'Check self.sampleMetadata[\'Subject ID\'] is str:\tFailure, \'self.sampleMetadata[\'Subject ID\']\' is ' + str(
                        type(self.sampleMetadata['Subject ID'][0]))
                    failureListMeta = conditionTest(condition, success, failure, failureListMeta, verbose, raiseError,
                                                    raiseWarning, exception=TypeError(failure))
                # end self.sampleMetadata['Subject ID']
                # sampleMetadata['Sample ID'] is str
                condition = (self.sampleMetadata['Sample ID'].dtype == numpy.dtype('O'))
                success = 'Check self.sampleMetadata[\'Sample ID\'] is str:\tOK'
                failure = 'Check self.sampleMetadata[\'Sample ID\'] is str:\tFailure, \'self.sampleMetadata[\'Sample ID\']\' is ' + str(
                    type(self.sampleMetadata['Sample ID'][0]))
                failureListMeta = conditionTest(condition, success, failure, failureListMeta, verbose, raiseError,
                                                raiseWarning, exception=TypeError(failure))
            # end self.sampleMetadata number of samples
            # end self.sampleMetadata

            ## self.featureMetadata
            # exist
            # number of features
            condition = (self.featureMetadata.shape[0] == refNumFeatures)
            success = 'Check self.featureMetadata number of features (rows):\tOK'
            failure = 'Check self.featureMetadata number of features (rows):\tFailure, \'self.featureMetadata\' has ' + str(
                self.featureMetadata.shape[0]) + ' features, ' + str(refNumFeatures) + ' expected'
            failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose, raiseError,
                                             raiseWarning, exception=ValueError(failure))
            if condition & (self.featureMetadata.shape[0] != 0):
                # No point checking columns if the number of columns is wrong or no features
                # featureMetadata['Feature Name'] is str
                condition = isinstance(self.featureMetadata['Feature Name'][0], str)
                success = 'Check self.featureMetadata[\'Feature Name\'] is str:\tOK'
                failure = 'Check self.featureMetadata[\'Feature Name\'] is str:\tFailure, \'self.featureMetadata[\'Feature Name\']\' is ' + str(
                    type(self.featureMetadata['Feature Name'][0]))
                failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose, raiseError,
                                                 raiseWarning, exception=TypeError(failure))
                if condition:
                    # featureMetadata['Feature Name'] are unique
                    u_ids, u_counts = numpy.unique(self.featureMetadata['Feature Name'], return_counts=True)
                    condition = all(u_counts == 1)
                    success = 'Check self.featureMetadata[\'Feature Name\'] are unique:\tOK'
                    failure = 'Check self.featureMetadata[\'Feature Name\'] are unique:\tFailure, the following \'self.featureMetadata[\'Feature Name\']\' are present more than once ' + str(
                        u_ids[u_counts > 1].tolist())
                    failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose, raiseError,
                                                     raiseWarning, exception=ValueError(failure))
                    # Use featureMetadata['Feature Name'] as reference for future tables
                    refFeatureName = self.featureMetadata['Feature Name'].values.tolist()
                    if verbose:
                        print('---- self.featureMetadata[\'Feature Name\'] used as Feature Name reference ----')
                # end self.featureMetadata['Feature Name']
                # ['calibrationMethod']
                condition = ('calibrationMethod' in self.featureMetadata.columns)
                success = 'Check self.featureMetadata[\'calibrationMethod\'] exists:\tOK'
                failure = 'Check self.featureMetadata[\'calibrationMethod\'] exists:\tFailure, \'self.featureMetadata\' lacks a \'calibrationMethod\' column'
                failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose, raiseError,
                                                 raiseWarning, exception=LookupError(failure))
                if condition:
                    # featureMetadata['calibrationMethod'] is an enum 'CalibrationMethod'
                    condition = isinstance(self.featureMetadata['calibrationMethod'][0], CalibrationMethod)
                    success = 'Check self.featureMetadata[\'calibrationMethod\'] is an enum \'CalibrationMethod\':\tOK'
                    failure = 'Check self.featureMetadata[\'calibrationMethod\'] is an enum \'CalibrationMethod\':\tFailure, \'self.featureMetadata[\'calibrationMethod\']\' is ' + str(
                        type(self.featureMetadata['calibrationMethod'][0]))
                    failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose, raiseError,
                                                     raiseWarning, exception=TypeError(failure))
                # end self.featureMetadata['calibrationMethod']
                # ['quantificationType']
                condition = ('quantificationType' in self.featureMetadata.columns)
                success = 'Check self.featureMetadata[\'quantificationType\'] exists:\tOK'
                failure = 'Check self.featureMetadata[\'quantificationType\'] exists:\tFailure, \'self.featureMetadata\' lacks a \'quantificationType\' column'
                failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose, raiseError,
                                                 raiseWarning, exception=LookupError(failure))
                if condition:
                    # featureMetadata['quantificationType'] is an enum 'QuantificationType'
                    condition = isinstance(self.featureMetadata['quantificationType'][0], QuantificationType)
                    success = 'Check self.featureMetadata[\'quantificationType\'] is an enum \'QuantificationType\':\tOK'
                    failure = 'Check self.featureMetadata[\'quantificationType\'] is an enum \'QuantificationType\':\tFailure, \'self.featureMetadata[\'quantificationType\']\' is ' + str(
                        type(self.featureMetadata['quantificationType'][0]))
                    failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose, raiseError,
                                                     raiseWarning, exception=TypeError(failure))
                # end self.featureMetadata['quantificationType']
                # ['Unit']
                condition = ('Unit' in self.featureMetadata.columns)
                success = 'Check self.featureMetadata[\'Unit\'] exists:\tOK'
                failure = 'Check self.featureMetadata[\'Unit\'] exists:\tFailure, \'self.featureMetadata\' lacks a \'Unit\' column'
                failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose, raiseError,
                                                 raiseWarning, exception=LookupError(failure))
                if condition:
                    # featureMetadata['Unit'] is a str
                    condition = isinstance(self.featureMetadata['Unit'][0], str)
                    success = 'Check self.featureMetadata[\'Unit\'] is a str:\tOK'
                    failure = 'Check self.featureMetadata[\'Unit\'] is a str:\tFailure, \'self.featureMetadata[\'Unit\']\' is ' + str(
                        type(self.featureMetadata['Unit'][0]))
                    failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose, raiseError,
                                                     raiseWarning, exception=TypeError(failure))
                # end self.featureMetadata['Unit']
                # ['LLOQ']
                tmpLLOQMatch = self.featureMetadata.columns.to_series().str.contains('LLOQ')
                condition = (sum(tmpLLOQMatch) > 0)
                success = 'Check self.featureMetadata[\'LLOQ\'] or similar exists:\tOK'
                failure = 'Check self.featureMetadata[\'LLOQ\'] or similar exists:\tFailure, \'self.featureMetadata\' lacks a \'LLOQ\' or \'LLOQ_batch\' column'
                failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose, raiseError,
                                                 raiseWarning, exception=LookupError(failure))
                if condition:
                    # featureMetadata['LLOQ'] is a float, try on first found
                    condition = ((self.featureMetadata.loc[:, tmpLLOQMatch].iloc[:, 0].dtype == numpy.dtype(
                        numpy.floating)) | (self.featureMetadata.loc[:, tmpLLOQMatch].iloc[:, 0].dtype == numpy.dtype(
                        numpy.int32)) | (self.featureMetadata.loc[:, tmpLLOQMatch].iloc[:, 0].dtype == numpy.dtype(
                        numpy.int64)))
                    success = 'Check self.featureMetadata[\'' + str(
                        self.featureMetadata.columns[tmpLLOQMatch][0]) + '\'] is int or float:\tOK'
                    failure = 'Check self.featureMetadata[\'' + str(self.featureMetadata.columns[tmpLLOQMatch][
                                                                        0]) + '\'] is int or float:\tFailure, \'self.featureMetadata[\'' + str(
                        self.featureMetadata.columns[tmpLLOQMatch][0]) + '\']\' is ' + str(
                        self.featureMetadata.loc[:, tmpLLOQMatch].iloc[:, 0].dtype)
                    failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose, raiseError,
                                                     raiseWarning, exception=TypeError(failure))
                # end self.featureMetadata['LLOQ']
                # ['ULOQ']
                tmpULOQMatch = self.featureMetadata.columns.to_series().str.contains('ULOQ')
                condition = (sum(tmpULOQMatch) > 0)
                success = 'Check self.featureMetadata[\'ULOQ\'] or similar exists:\tOK'
                failure = 'Check self.featureMetadata[\'ULOQ\'] or similar exists:\tFailure, \'self.featureMetadata\' lacks a \'ULOQ\' or \'ULOQ_batch\' column'
                failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose, raiseError,
                                                 raiseWarning, exception=LookupError(failure))
                if condition:
                    # featureMetadata['ULOQ'] is a float, try on first found
                    condition = ((self.featureMetadata.loc[:, tmpULOQMatch].iloc[:, 0].dtype == numpy.dtype(
                        numpy.floating)) | (self.featureMetadata.loc[:, tmpULOQMatch].iloc[:, 0].dtype == numpy.dtype(
                        numpy.int32)) | (self.featureMetadata.loc[:, tmpULOQMatch].iloc[:, 0].dtype == numpy.dtype(
                        numpy.int64)))
                    success = 'Check self.featureMetadata[\'' + str(
                        self.featureMetadata.columns[tmpULOQMatch][0]) + '\'] is int or float:\tOK'
                    failure = 'Check self.featureMetadata[\'' + str(self.featureMetadata.columns[tmpULOQMatch][
                                                                        0]) + '\'] is int or float:\tFailure, \'self.featureMetadata[\'' + str(
                        self.featureMetadata.columns[tmpULOQMatch][0]) + '\']\' is ' + str(
                        self.featureMetadata.loc[:, tmpULOQMatch].iloc[:, 0].dtype)
                    failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose, raiseError,
                                                     raiseWarning, exception=TypeError(failure))
                # end self.featureMetadata['ULOQ']
                # 'externalID' in featureMetadata columns (need externalID to exist)
                if 'externalID' in self.Attributes:
                    if isinstance(self.Attributes['externalID'], list):
                        condition = set(self.Attributes['externalID']).issubset(self.featureMetadata.columns)
                        success = 'Check self.featureMetadata does have the \'externalID\' as columns:\tOK'
                        failure = 'Check self.featureMetadata does have the \'externalID\' as columns:\tFailure, \'self.featureMetadata\' lacks the \'externalID\' columns'
                        failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose,
                                                         raiseError, raiseWarning, exception=LookupError(failure))
                # end 'externalID' columns
            # end self.featureMetadata number of features
            # end self.featureMetadata

            ## self.expectedConcentration
            # exist
            condition = hasattr(self, 'expectedConcentration')
            success = 'Check self.expectedConcentration exists:\tOK'
            failure = 'Check self.expectedConcentration exists:\tFailure, no attribute \'self.expectedConcentration\''
            failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose, raiseError,
                                             raiseWarning, exception=AttributeError(failure))
            if condition:
                # is a pandas.DataFrame
                condition = isinstance(self.expectedConcentration, pandas.DataFrame)
                success = 'Check self.expectedConcentration is a pandas.DataFrame:\tOK'
                failure = 'Check self.expectedConcentration is a pandas.DataFrame:\tFailure, \'self.expectedConcentration\' is ' + str(
                    type(self.expectedConcentration))
                failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose, raiseError,
                                                 raiseWarning, exception=TypeError(failure))
                if condition:
                    # number of samples
                    condition = (self.expectedConcentration.shape[0] == refNumSamples)
                    success = 'Check self.expectedConcentration number of samples (rows):\tOK'
                    failure = 'Check self.expectedConcentration number of samples (rows):\tFailure, \'self.expectedConcentration\' has ' + str(
                        self.expectedConcentration.shape[0]) + ' features, ' + str(refNumSamples) + ' expected'
                    failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose, raiseError,
                                                     raiseWarning, exception=ValueError(failure))
                    # number of features
                    condition = (self.expectedConcentration.shape[1] == refNumFeatures)
                    success = 'Check self.expectedConcentration number of features (columns):\tOK'
                    failure = 'Check self.expectedConcentration number of features (columns):\tFailure, \'self.expectedConcentration\' has ' + str(
                        self.expectedConcentration.shape[1]) + ' features, ' + str(refNumFeatures) + ' expected'
                    failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose, raiseError,
                                                     raiseWarning, exception=ValueError(failure))
                    if condition & (refNumFeatures != 0):
                        # expectedConcentration column names match ['Feature Name']
                        tmpDiff = pandas.DataFrame(
                            {'FeatName': refFeatureName, 'ColName': self.expectedConcentration.columns.values.tolist()})
                        condition = (self.expectedConcentration.columns.values.tolist() == refFeatureName)
                        success = 'Check self.expectedConcentration column name match self.featureMetadata[\'Feature Name\']:\tOK'
                        failure = 'Check self.expectedConcentration column name match self.featureMetadata[\'Feature Name\']:\tFailure, the following \'self.featureMetadata[\'Feature Name\']\' and \'self.expectedConcentration.columns\' differ ' + str(
                            tmpDiff.loc[
                                (tmpDiff['FeatName'] != tmpDiff['ColName']), ['FeatName', 'ColName']].values.tolist())
                        failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose,
                                                         raiseError, raiseWarning, exception=ValueError(failure))
                    # end self.expectedConcentration number of features
                # end self.expectedConcentration is a pandas.DataFrame
            # end self.expectedConcentration

            ## self.sampleMask
            # is initialised
            condition = (self.sampleMask.shape != ())
            success = 'Check self.sampleMask is initialised:\tOK'
            failure = 'Check self.sampleMask is initialised:\tFailure, \'self.sampleMask\' is not initialised'
            failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose, raiseError,
                                             raiseWarning, exception=ValueError(failure))
            if condition:
                # number of samples
                condition = (self.sampleMask.shape == (refNumSamples,))
                success = 'Check self.sampleMask number of samples:\tOK'
                failure = 'Check self.sampleMask number of samples:\tFailure, \'self.sampleMask\' has ' + str(
                    self.sampleMask.shape[0]) + ' samples, ' + str(refNumSamples) + ' expected'
                failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose, raiseError,
                                                 raiseWarning, exception=ValueError(failure))
            ## end self.sampleMask

            ## self.featureMask
            # is initialised
            condition = (self.featureMask.shape != ())
            success = 'Check self.featureMask is initialised:\tOK'
            failure = 'Check self.featureMask is initialised:\tFailure, \'self.featureMask\' is not initialised'
            failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose, raiseError,
                                             raiseWarning, exception=ValueError(failure))
            if condition:
                # number of features
                condition = (self.featureMask.shape == (refNumFeatures,))
                success = 'Check self.featureMask number of features:\tOK'
                failure = 'Check self.featureMask number of features:\tFailure, \'self.featureMask\' has ' + str(
                    self.featureMask.shape[0]) + ' features, ' + str(refNumFeatures) + ' expected'
                failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose, raiseError,
                                                 raiseWarning, exception=ValueError(failure))
            ## end self.featureMask

            ## self.calibration
            # exist
            condition = hasattr(self, 'calibration')
            success = 'Check self.calibration exists:\tOK'
            failure = 'Check self.calibration exists:\tFailure, no attribute \'self.calibration\''
            failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose, raiseError,
                                             raiseWarning, exception=AttributeError(failure))
            if condition:
                # is a dict or a list
                condition = isinstance(self.calibration, (dict, list))
                success = 'Check self.calibration is a dict or list:\tOK'
                failure = 'Check self.calibration is a dict or list:\tFailure, \'self.calibration\' is ' + str(
                    type(self.calibration))
                failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose, raiseError,
                                                 raiseWarning, exception=TypeError(failure))
                if condition:
                    # self.calibration is a list of dict
                    if isinstance(self.calibration, list):
                        # use reference inside each calibration
                        refCalibNumSamples = len(self.calibration) * [None]
                        refCalibNumFeatures = len(self.calibration) * [None]
                        refCalibFeatureName = len(self.calibration) * [None]
                        for i in range(len(self.calibration)):
                            # self.calibration[i] is a dict
                            condition = isinstance(self.calibration[i], dict)
                            success = 'Check self.calibration[' + str(i) + '] is a dict or list:\tOK'
                            failure = 'Check self.calibration[' + str(
                                i) + '] is a dict or list:\tFailure, \'self.calibration\' is ' + str(
                                type(self.calibration[i]))
                            failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose,
                                                             raiseError, raiseWarning, exception=TypeError(failure))
                            if condition:
                                ## calibIntensityData
                                # exist
                                condition = 'calibIntensityData' in self.calibration[i]
                                success = 'Check self.calibration[' + str(i) + '][\'calibIntensityData\'] exists:\tOK'
                                failure = 'Check self.calibration[' + str(
                                    i) + '][\'calibIntensityData\'] exists:\tFailure, no attribute \'self.calibration[' + str(
                                    i) + '][\'calibIntensityData\']\''
                                failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose,
                                                                 raiseError, raiseWarning,
                                                                 exception=AttributeError(failure))
                                if condition:
                                    # is a numpy.ndarray
                                    condition = isinstance(self.calibration[i]['calibIntensityData'], numpy.ndarray)
                                    success = 'Check self.calibration[' + str(
                                        i) + '][\'calibIntensityData\'] is a numpy.ndarray:\tOK'
                                    failure = 'Check self.calibration[' + str(
                                        i) + '][\'calibIntensityData\'] is a numpy.ndarray:\tFailure, \'self.calibration[' + str(
                                        i) + '][\'calibIntensityData\']\' is ' + str(
                                        type(self.calibration[i]['calibIntensityData']))
                                    failureListBasic = conditionTest(condition, success, failure, failureListBasic,
                                                                     verbose, raiseError, raiseWarning,
                                                                     exception=TypeError(failure))
                                    if condition:
                                        if (self.calibration[i]['calibIntensityData'].all() != numpy.array(None).all()):
                                            # Use calibIntensityData as number of calib sample/feature reference
                                            refCalibNumSamples[i] = self.calibration[i]['calibIntensityData'].shape[0]
                                            refCalibNumFeatures[i] = self.calibration[i]['calibIntensityData'].shape[1]
                                            if verbose:
                                                print('---- self.calibration[' + str(
                                                    i) + '][\'calibIntensityData\'] used as number of calibration samples/features reference ----')
                                                print('\t' + str(refCalibNumSamples[i]) + ' samples, ' + str(
                                                    refCalibNumFeatures[i]) + ' features')
                                        # end calibIntensityData is a numpy.ndarray
                                # end calibIntensityData
                                ## calibSampleMetadata
                                # exist
                                condition = 'calibSampleMetadata' in self.calibration[i]
                                success = 'Check self.calibration[' + str(i) + '][\'calibSampleMetadata\'] exists:\tOK'
                                failure = 'Check self.calibration[' + str(
                                    i) + '][\'calibSampleMetadata\'] exists:\tFailure, no attribute \'self.calibration[' + str(
                                    i) + '][\'calibSampleMetadata\']\''
                                failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose,
                                                                 raiseError, raiseWarning,
                                                                 exception=AttributeError(failure))
                                if condition:
                                    # is a pandas.DataFrame
                                    condition = isinstance(self.calibration[i]['calibSampleMetadata'], pandas.DataFrame)
                                    success = 'Check self.calibration[' + str(
                                        i) + '][\'calibSampleMetadata\'] is a pandas.DataFrame:\tOK'
                                    failure = 'Check self.calibration[' + str(
                                        i) + '][\'calibSampleMetadata\'] is a pandas.DataFrame:\tFailure, \'self.calibration[' + str(
                                        i) + '][\'calibSampleMetadata\']\' is ' + str(
                                        type(self.calibration[i]['calibSampleMetadata']))
                                    failureListBasic = conditionTest(condition, success, failure, failureListBasic,
                                                                     verbose, raiseError, raiseWarning,
                                                                     exception=TypeError(failure))
                                    if condition:
                                        # number of samples
                                        condition = (self.calibration[i]['calibSampleMetadata'].shape[0] ==
                                                     refCalibNumSamples[i])
                                        success = 'Check self.calibration[' + str(
                                            i) + '][\'calibSampleMetadata\'] number of samples:\tOK'
                                        failure = 'Check self.calibration[' + str(
                                            i) + '][\'calibSampleMetadata\'] number of samples:\tFailure, \'self.calibration[' + str(
                                            i) + '][\'calibSampleMetadata\']\' has ' + str(
                                            self.calibration[i]['calibSampleMetadata'].shape[0]) + ' samples, ' + str(
                                            refCalibNumSamples[i]) + ' expected'
                                        failureListBasic = conditionTest(condition, success, failure, failureListBasic,
                                                                         verbose, raiseError, raiseWarning,
                                                                         exception=ValueError(failure))
                                    # end calibSampleMetadata is a pandas.DataFrame
                                # end calibSampleMetadata
                                ## calibFeatureMetadata
                                # exist
                                condition = 'calibFeatureMetadata' in self.calibration[i]
                                success = 'Check self.calibration[' + str(i) + '][\'calibFeatureMetadata\'] exists:\tOK'
                                failure = 'Check self.calibration[' + str(
                                    i) + '][\'calibFeatureMetadata\'] exists:\tFailure, no attribute \'self.calibration[' + str(
                                    i) + '][\'calibFeatureMetadata\']\''
                                failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose,
                                                                 raiseError, raiseWarning,
                                                                 exception=AttributeError(failure))
                                if condition:
                                    # is a pandas.DataFrame
                                    condition = isinstance(self.calibration[i]['calibFeatureMetadata'],
                                                           pandas.DataFrame)
                                    success = 'Check self.calibration[' + str(
                                        i) + '][\'calibFeatureMetadata\'] is a pandas.DataFrame:\tOK'
                                    failure = 'Check self.calibration[' + str(
                                        i) + '][\'calibFeatureMetadata\'] is a pandas.DataFrame:\tFailure, \'self.calibration[' + str(
                                        i) + '][\'calibFeatureMetadata\']\' is ' + str(
                                        type(self.calibration[i]['calibFeatureMetadata']))
                                    failureListBasic = conditionTest(condition, success, failure, failureListBasic,
                                                                     verbose, raiseError, raiseWarning,
                                                                     exception=TypeError(failure))
                                    if condition:
                                        # number of features
                                        condition = (self.calibration[i]['calibFeatureMetadata'].shape[0] ==
                                                     refCalibNumFeatures[i])
                                        success = 'Check self.calibration[' + str(
                                            i) + '][\'calibFeatureMetadata\'] number of features:\tOK'
                                        failure = 'Check self.calibration[' + str(
                                            i) + '][\'calibFeatureMetadata\'] number of features:\tFailure, \'self.calibration[' + str(
                                            i) + '][\'calibFeatureMetadata\']\' has ' + str(
                                            self.calibration[i]['calibFeatureMetadata'].shape[0]) + ' features, ' + str(
                                            refCalibNumFeatures[i]) + ' expected'
                                        failureListBasic = conditionTest(condition, success, failure, failureListBasic,
                                                                         verbose, raiseError, raiseWarning,
                                                                         exception=ValueError(failure))
                                        if condition & (refCalibNumFeatures[i] != 0):
                                            # Feature Name exist
                                            condition = ('Feature Name' in self.calibration[i][
                                                'calibFeatureMetadata'].columns.tolist())
                                            success = 'Check self.calibration[' + str(
                                                i) + '][\'calibFeatureMetadata\'][\'Feature Name\'] exist:\tOK'
                                            failure = 'Check self.calibration[' + str(
                                                i) + '][\'calibFeatureMetadata\'][\'Feature Name\'] exist:\tFailure, no column \'self.calibration[' + str(
                                                i) + '][\'calibFeatureMetadata\'][\'Feature Name\']'
                                            failureListBasic = conditionTest(condition, success, failure,
                                                                             failureListBasic, verbose, raiseError,
                                                                             raiseWarning,
                                                                             exception=LookupError(failure))
                                            if condition:
                                                # store the featureMetadata columns as reference
                                                refCalibFeatureName[i] = self.calibration[i]['calibFeatureMetadata'][
                                                    'Feature Name'].values.tolist()
                                    # end calibFeatureMetadata is a pandas.DataFrame
                                # end calibFeatureMetadata
                                ## calibExpectedConcentration
                                # exist
                                condition = 'calibExpectedConcentration' in self.calibration[i]
                                success = 'Check self.calibration[' + str(
                                    i) + '][\'calibExpectedConcentration\'] exists:\tOK'
                                failure = 'Check self.calibration[' + str(
                                    i) + '][\'calibExpectedConcentration\'] exists:\tFailure, no attribute \'self.calibration[' + str(
                                    i) + '][\'calibExpectedConcentration\']\''
                                failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose,
                                                                 raiseError, raiseWarning,
                                                                 exception=AttributeError(failure))
                                if condition:
                                    # is a pandas.DataFrame
                                    condition = isinstance(self.calibration[i]['calibExpectedConcentration'],
                                                           pandas.DataFrame)
                                    success = 'Check self.calibration[' + str(
                                        i) + '][\'calibExpectedConcentration\'] is a pandas.DataFrame:\tOK'
                                    failure = 'Check self.calibration[' + str(
                                        i) + '][\'calibExpectedConcentration\'] is a pandas.DataFrame:\tFailure, \'self.calibration[' + str(
                                        i) + '][\'calibExpectedConcentration\']\' is ' + str(
                                        type(self.calibration[i]['calibExpectedConcentration']))
                                    failureListBasic = conditionTest(condition, success, failure, failureListBasic,
                                                                     verbose, raiseError, raiseWarning,
                                                                     exception=TypeError(failure))
                                    if condition:
                                        # number of samples
                                        condition = (self.calibration[i]['calibExpectedConcentration'].shape[0] ==
                                                     refCalibNumSamples[i])
                                        success = 'Check self.calibration[' + str(
                                            i) + '][\'calibExpectedConcentration\'] number of samples:\tOK'
                                        failure = 'Check self.calibration[' + str(
                                            i) + '][\'calibExpectedConcentration\'] number of samples:\tFailure, \'self.calibration[' + str(
                                            i) + '][\'calibExpectedConcentration\']\' has ' + str(
                                            self.calibration[i]['calibExpectedConcentration'].shape[
                                                0]) + ' samples, ' + str(refCalibNumSamples[i]) + ' expected'
                                        failureListBasic = conditionTest(condition, success, failure, failureListBasic,
                                                                         verbose, raiseError, raiseWarning,
                                                                         exception=ValueError(failure))
                                        # number of features
                                        condition = (self.calibration[i]['calibExpectedConcentration'].shape[1] ==
                                                     refCalibNumFeatures[i])
                                        success = 'Check self.calibration[' + str(
                                            i) + '][\'calibExpectedConcentration\'] number of features:\tOK'
                                        failure = 'Check self.calibration[' + str(
                                            i) + '][\'calibExpectedConcentration\'] number of features:\tFailure, \'self.calibration[' + str(
                                            i) + '][\'calibExpectedConcentration\']\' has ' + str(
                                            self.calibration[i]['calibExpectedConcentration'].shape[
                                                1]) + ' features, ' + str(refCalibNumFeatures[i]) + ' expected'
                                        failureListBasic = conditionTest(condition, success, failure, failureListBasic,
                                                                         verbose, raiseError, raiseWarning,
                                                                         exception=ValueError(failure))
                                        if condition & (refCalibNumFeatures[i] != 0):
                                            # calibExpectedConcentration column names match ['Feature Name']
                                            tmpDiff = pandas.DataFrame({'FeatName': refCalibFeatureName[i],
                                                                        'ColName': self.calibration[i][
                                                                            'calibExpectedConcentration'].columns.values.tolist()})
                                            condition = (self.calibration[i][
                                                             'calibExpectedConcentration'].columns.values.tolist() ==
                                                         refCalibFeatureName[i])
                                            success = 'Check self.calibration[' + str(
                                                i) + '][\'calibExpectedConcentration\'] column name match self.calibration[' + str(
                                                i) + '][\'calibFeatureMetadata\'][\'Feature Name\']:\tOK'
                                            failure = 'Check self.calibration[' + str(
                                                i) + '][\'calibExpectedConcentration\'] column name match self.calibration[' + str(
                                                i) + '][\'calibFeatureMetadata\'][\'Feature Name\']:\tFailure, the following \'self.calibration[' + str(
                                                i) + '][\'calibFeatureMetadata\'][\'Feature Name\']\' and \'self.calibration[' + str(
                                                i) + '][\'calibExpectedConcentration\'].columns\' differ ' + str(
                                                tmpDiff.loc[(tmpDiff['FeatName'] != tmpDiff['ColName']), ['FeatName',
                                                                                                          'ColName']].values.tolist())
                                            failureListBasic = conditionTest(condition, success, failure,
                                                                             failureListBasic, verbose, raiseError,
                                                                             raiseWarning,
                                                                             exception=ValueError(failure))
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
                        failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose,
                                                         raiseError, raiseWarning, exception=AttributeError(failure))
                        if condition:
                            # is a numpy.ndarray
                            condition = isinstance(self.calibration['calibIntensityData'], numpy.ndarray)
                            success = 'Check self.calibration[\'calibIntensityData\'] is a numpy.ndarray:\tOK'
                            failure = 'Check self.calibration[\'calibIntensityData\'] is a numpy.ndarray:\tFailure, \'self.calibration[\'calibIntensityData\']\' is ' + str(
                                type(self.calibration['calibIntensityData']))
                            failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose,
                                                             raiseError, raiseWarning, exception=TypeError(failure))
                            if condition:
                                if (self.calibration['calibIntensityData'].all() != numpy.array(None).all()):
                                    # number of features
                                    condition = (self.calibration['calibIntensityData'].shape[1] == refNumFeatures)
                                    success = 'Check self.calibration[\'calibIntensityData\'] number of features:\tOK'
                                    failure = 'Check self.calibration[\'calibIntensityData\'] number of features:\tFailure, \'self.calibration[\'calibIntensityData\']\' has ' + str(
                                        self.calibration['calibIntensityData'].shape[1]) + ' features, ' + str(
                                        refNumFeatures) + ' expected'
                                    failureListBasic = conditionTest(condition, success, failure, failureListBasic,
                                                                     verbose, raiseError, raiseWarning,
                                                                     exception=ValueError(failure))
                                    # Use calibIntensityData as number of calib sample reference
                                    refNumCalibSamples = self.calibration['calibIntensityData'].shape[0]
                                    if verbose:
                                        print(
                                            '---- self.calibration[\'calibIntensityData\'] used as number of calibration samples reference ----')
                                        print('\t' + str(refNumCalibSamples) + ' samples')
                            # end calibIntensityData is a numpy.ndarray
                        # end calibIntensityData
                        ## calibSampleMetadata
                        # exist
                        condition = 'calibSampleMetadata' in self.calibration
                        success = 'Check self.calibration[\'calibSampleMetadata\'] exists:\tOK'
                        failure = 'Check self.calibration[\'calibSampleMetadata\'] exists:\tFailure, no attribute \'self.calibration[\'calibSampleMetadata\']\''
                        failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose,
                                                         raiseError, raiseWarning, exception=AttributeError(failure))
                        if condition:
                            # is a pandas.DataFrame
                            condition = isinstance(self.calibration['calibSampleMetadata'], pandas.DataFrame)
                            success = 'Check self.calibration[\'calibSampleMetadata\'] is a pandas.DataFrame:\tOK'
                            failure = 'Check self.calibration[\'calibSampleMetadata\'] is a pandas.DataFrame:\tFailure, \'self.calibration[\'calibSampleMetadata\']\' is ' + str(
                                type(self.calibration['calibSampleMetadata']))
                            failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose,
                                                             raiseError, raiseWarning, exception=TypeError(failure))
                            if condition:
                                # number of samples
                                condition = (self.calibration['calibSampleMetadata'].shape[0] == refNumCalibSamples)
                                success = 'Check self.calibration[\'calibSampleMetadata\'] number of samples:\tOK'
                                failure = 'Check self.calibration[\'calibSampleMetadata\'] number of samples:\tFailure, \'self.calibration[\'calibSampleMetadata\']\' has ' + str(
                                    self.calibration['calibSampleMetadata'].shape[0]) + ' samples, ' + str(
                                    refNumCalibSamples) + ' expected'
                                failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose,
                                                                 raiseError, raiseWarning,
                                                                 exception=ValueError(failure))
                            # end calibSampleMetadata is a pandas.DataFrame
                        # end calibSampleMetadata
                        ## calibFeatureMetadata
                        # exist
                        condition = 'calibFeatureMetadata' in self.calibration
                        success = 'Check self.calibration[\'calibFeatureMetadata\'] exists:\tOK'
                        failure = 'Check self.calibration[\'calibFeatureMetadata\'] exists:\tFailure, no attribute \'self.calibration[\'calibFeatureMetadata\']\''
                        failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose,
                                                         raiseError, raiseWarning, exception=AttributeError(failure))
                        if condition:
                            # is a pandas.DataFrame
                            condition = isinstance(self.calibration['calibFeatureMetadata'], pandas.DataFrame)
                            success = 'Check self.calibration[\'calibFeatureMetadata\'] is a pandas.DataFrame:\tOK'
                            failure = 'Check self.calibration[\'calibFeatureMetadata\'] is a pandas.DataFrame:\tFailure, \'self.calibration[\'calibFeatureMetadata\']\' is ' + str(
                                type(self.calibration['calibFeatureMetadata']))
                            failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose,
                                                             raiseError, raiseWarning, exception=TypeError(failure))
                            if condition:
                                # number of features
                                condition = (self.calibration['calibFeatureMetadata'].shape[0] == refNumFeatures)
                                success = 'Check self.calibration[\'calibFeatureMetadata\'] number of features:\tOK'
                                failure = 'Check self.calibration[\'calibFeatureMetadata\'] number of features:\tFailure, \'self.calibration[\'calibFeatureMetadata\']\' has ' + str(
                                    self.calibration['calibFeatureMetadata'].shape[0]) + ' features, ' + str(
                                    refNumFeatures) + ' expected'
                                failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose,
                                                                 raiseError, raiseWarning,
                                                                 exception=ValueError(failure))
                                if condition & (refNumFeatures != 0):
                                    # Feature Name exist
                                    condition = ('Feature Name' in self.calibration[
                                        'calibFeatureMetadata'].columns.tolist())
                                    success = 'Check self.calibration[\'calibFeatureMetadata\'][\'Feature Name\'] exist:\tOK'
                                    failure = 'Check self.calibration[\'calibFeatureMetadata\'][\'Feature Name\'] exist:\tFailure, no column \'self.calibration[\'calibFeatureMetadata\'][\'Feature Name\']'
                                    failureListBasic = conditionTest(condition, success, failure, failureListBasic,
                                                                     verbose, raiseError, raiseWarning,
                                                                     exception=LookupError(failure))
                            # end calibFeatureMetadata is a pandas.DataFrame
                        # end calibFeatureMetadata
                        ## calibExpectedConcentration
                        # exist
                        condition = 'calibExpectedConcentration' in self.calibration
                        success = 'Check self.calibration[\'calibExpectedConcentration\'] exists:\tOK'
                        failure = 'Check self.calibration[\'calibExpectedConcentration\'] exists:\tFailure, no attribute \'self.calibration[\'calibExpectedConcentration\']\''
                        failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose,
                                                         raiseError, raiseWarning, exception=AttributeError(failure))
                        if condition:
                            # is a pandas.DataFrame
                            condition = isinstance(self.calibration['calibExpectedConcentration'], pandas.DataFrame)
                            success = 'Check self.calibration[\'calibExpectedConcentration\'] is a pandas.DataFrame:\tOK'
                            failure = 'Check self.calibration[\'calibExpectedConcentration\'] is a pandas.DataFrame:\tFailure, \'self.calibration[\'calibExpectedConcentration\']\' is ' + str(
                                type(self.calibration['calibExpectedConcentration']))
                            failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose,
                                                             raiseError, raiseWarning, exception=TypeError(failure))
                            if condition:
                                # number of samples
                                condition = (self.calibration['calibExpectedConcentration'].shape[
                                                 0] == refNumCalibSamples)
                                success = 'Check self.calibration[\'calibExpectedConcentration\'] number of samples:\tOK'
                                failure = 'Check self.calibration[\'calibExpectedConcentration\'] number of samples:\tFailure, \'self.calibration[\'calibExpectedConcentration\']\' has ' + str(
                                    self.calibration['calibExpectedConcentration'].shape[0]) + ' samples, ' + str(
                                    refNumCalibSamples) + ' expected'
                                failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose,
                                                                 raiseError, raiseWarning,
                                                                 exception=ValueError(failure))
                                # number of features
                                condition = (self.calibration['calibExpectedConcentration'].shape[1] == refNumFeatures)
                                success = 'Check self.calibration[\'calibExpectedConcentration\'] number of features:\tOK'
                                failure = 'Check self.calibration[\'calibExpectedConcentration\'] number of features:\tFailure, \'self.calibration[\'calibExpectedConcentration\']\' has ' + str(
                                    self.calibration['calibExpectedConcentration'].shape[1]) + ' features, ' + str(
                                    refNumFeatures) + ' expected'
                                failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose,
                                                                 raiseError, raiseWarning,
                                                                 exception=ValueError(failure))
                                if condition & (refNumFeatures != 0):
                                    # calibExpectedConcentration column names match ['Feature Name']
                                    tmpDiff = pandas.DataFrame({'FeatName': refFeatureName, 'ColName': self.calibration[
                                        'calibExpectedConcentration'].columns.values.tolist()})
                                    condition = (self.calibration[
                                                     'calibExpectedConcentration'].columns.values.tolist() == refFeatureName)
                                    success = 'Check self.calibration[\'calibExpectedConcentration\'] column name match self.featureMetadata[\'Feature Name\']:\tOK'
                                    failure = 'Check self.calibration[\'calibExpectedConcentration\'] column name match self.featureMetadata[\'Feature Name\']:\tFailure, the following \'self.featureMetadata[\'Feature Name\']\' and \'self.calibration[\'calibExpectedConcentration\'].columns\' differ ' + str(
                                        tmpDiff.loc[(tmpDiff['FeatName'] != tmpDiff['ColName']), ['FeatName',
                                                                                                  'ColName']].values.tolist())
                                    failureListBasic = conditionTest(condition, success, failure, failureListBasic,
                                                                     verbose, raiseError, raiseWarning,
                                                                     exception=ValueError(failure))
                                # end calibExpectedConcentration number of features
                            # end calibExpectedConcentration is a pandas.DataFrame
                        # end calibExpectedConcentration
                    # end self.calib is a dict
                # self.calibration is a dict or a list
            # end self.calibration

            ## List additional attributes (print + log)
            expectedSet = set({'Attributes', 'VariableType', '_Normalisation', '_name', 'fileName', 'filePath',
                               '_intensityData', 'sampleMetadata', 'featureMetadata', 'expectedConcentration',
                               'sampleMask',
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
                self.Attributes['Log'].append([datetime.now(),
                                               'Dataset conforms to basic TargetedDataset (0 errors), %s (%d errors), %s (%d errors), (%i samples and %i features), with %d additional attributes in the object: %s. QC errors: %s, Meta errors: %s' % (
                                               QCText, len(failureListQC), MetaText, len(failureListMeta),
                                               self.noSamples, self.noFeatures, len(additionalAttributes),
                                               list(additionalAttributes), list(failureListQC), list(failureListMeta))])
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
                    warnings.warn(
                        'Does not have sample metadata information:\t %d errors found' % ((len(failureListMeta))))
                return ({'Dataset': True, 'BasicTargetedDataset': True, 'QC': QCBool, 'sampleMetadata': MetaBool})

            # Try logging to something that might not have a log
            else:
                # try logging
                try:
                    self.Attributes['Log'].append([datetime.now(),
                                                   'Failed basic TargetedDataset validation, with the following %d issues: %s' % (
                                                   len(failureListBasic), failureListBasic)])
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
                    warnings.warn(
                        'Does not conform to basic TargetedDataset:\t %i errors found' % (len(failureListBasic)))
                    warnings.warn('Does not have QC parameters')
                    warnings.warn('Does not have sample metadata information')
                return ({'Dataset': True, 'BasicTargetedDataset': False, 'QC': False, 'sampleMetadata': False})

        # If it's not a Dataset, no point checking anything more
        else:
            # try logging
            try:
                self.Attributes['Log'].append(
                    [datetime.now(), 'Failed basic TargetedDataset validation, Failed Dataset validation'])
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

    def __add__(self, other):
        """
        Implements the concatenation of 2 :py:class:`NMRTargetedDataset`

        `targetedNMRDataset = targetedNMRDatasetDatasetBatch1 + targetedNMRDatasetBatch2`

        `targetedNMRDataset = sum([targetedDatasetBatch1, targetedNMRDatasetBatch2`, targetedNMRDatasetBatch3])'

        This __add__ method provides two behaviours:
        1) If the features are exactly the same in both NMRTargetedDataset
        objects and there are complementary samples, the new object will be a sample-wise concatenation of
        the two datasets. In this scenario, if there are duplicated Sample File Names
        an exception will be thrown listing those.
        2) If the features are different (number and name), it is assumed that both datasets contain the same
        samples and sum of the two objects is meant to add new measurements.
        For this case, all the Sample File Names must be present in all merged batches.
        The __add__ method will automatically detect which behaviour is applicable and perform the appropriate checks.
        If new samples are added, the intensityData matrix is expected to increase in rows, and the sampleMetadata
        dataframe concatenated,
        while featureMetadata information remains the same. If
        :raises ValueError: if the targeted methods employed differ
        :raises ValueError: if an object doesn't pass validation before merge
        :raises ValueError: if the merge object doesn't pass validation
        :raises Warning: to update LOQ using :py:meth:`~NMRTargetedDataset.mergeLimitsOfQuantification`
        """

        # Refactor the logic here - there are 2 possible types of merge - either a new feature is added
        if self.noFeatures == other.noFeatures:
            stackSamplesMerge = True

        if self.noSamples != other.noSamples:
            stackFeaturesMerge = True

        ## Input checks
        # Validate both objects.
        validSelfDataset = self.validateObject(verbose=False, raiseError=False, raiseWarning=False)
        validOtherDataset = other.validateObject(verbose=False, raiseError=False, raiseWarning=False)
        if not validSelfDataset['NMRTargetedDataset']:
            raise ValueError('self does not satisfy to the Basic TargetedDataset definition, '
                             'check with self.validateObject(verbose=True, raiseError=False)')
        if not validOtherDataset['NMRTargetedDataset']:
            raise ValueError('other does not satisfy to the Basic TargetedDataset definition, '
                             'check with other.validateObject(verbose=True, raiseError=False)')
        ## Attributes
        if self.Attributes['methodName'] != other.Attributes['methodName']:
            raise ValueError(
                'Cannot concatenate different targeted methods: \'' + self.Attributes['methodName'] + '\' and \'' +
                other.Attributes['methodName'] + '\'')

        if stackSamplesMerge:
            pass

        elif stackFeaturesMerge:
            #  Warning if duplicate 'Sample File Name' in sampleMetadata
            u_ids, u_counts = numpy.unique(
                pandas.concat([self.sampleMetadata['Sample File Name'], other.sampleMetadata['Sample File Name']],
                              ignore_index=True, sort=False), return_counts=True)
        if any(u_counts > 1):
            warnings.warn('Warning: The following \'Sample File Name\' are present more than once: ' + str(
                u_ids[u_counts > 1].tolist()))


        ## Initialise an empty TargetedDataset to overwrite
        targetedData = NMRTargetedDataset(datapath='', fileType='empty')

        # copy from the first (mainly dataset parameters, methodName, chromatography and ionisation)
        targetedData.Attributes = copy.deepcopy(self.Attributes)

        # append both logs
        targetedData.Attributes['Log'] = self.Attributes['Log'] + other.Attributes['Log']

        ## _Normalisation
        targetedData._Normalisation = normalisation.NullNormaliser()

        ## VariableType
        targetedData.VariableType = VariableType.Discrete

        targetedData.AnalyticalPlatform = AnalyticalPlatform.NMR

        ## _name
        targetedData.name = self.name + ' - ' + other.name

        ## fileName
        targetedData.fileName = flatten([self.fileName, other.fileName]) # what to do with this for NMR??

        ## filePath
        targetedData.filePath = flatten([self.filePath, other.filePath]) # what to do with this for NMR ??

        if stackSamplesMerge:
            ## sampleMetadata
            tmpSampleMetadata1 = copy.deepcopy(self.sampleMetadata)
            tmpSampleMetadata2 = copy.deepcopy(other.sampleMetadata)
            # reindex the 'Batch' value across both targetedDataset (self starts at 1, other at max(self)+1)
            tmpSampleMetadata1['Batch'], batchChangeSelf = reNumber(tmpSampleMetadata1['Batch'], 1)
            tmpSampleMetadata2['Batch'], batchChangeOther = reNumber(tmpSampleMetadata2['Batch'],
                                                                     tmpSampleMetadata1['Batch'].values.max() + 1)
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
                    if (col in self.featureMetadata.columns) and (col in other.featureMetadata.columns) and (
                            col not in mergeCol):
                        mergeCol.append(col)
            # take each dataset featureMetadata column names, modify them and rename columns
            tmpFeatureMetadata1 = copy.deepcopy(self.featureMetadata)
            updatedCol1 = batchListReNumber(tmpFeatureMetadata1.columns.tolist(), batchChangeSelf, mergeCol)
            tmpFeatureMetadata1.columns = updatedCol1
            tmpFeatureMetadata2 = copy.deepcopy(other.featureMetadata)
            updatedCol2 = batchListReNumber(tmpFeatureMetadata2.columns.tolist(), batchChangeOther, mergeCol)
            tmpFeatureMetadata2.columns = updatedCol2
            # Merge featureMetadata on the mergeCol, no columns with identical name exist
            tmpFeatureMetadata = tmpFeatureMetadata1.merge(tmpFeatureMetadata2, how='outer', on=mergeCol, left_on=None,
                                                       right_on=None, left_index=False, right_index=False, sort=False,
                                                       copy=True, indicator=False)
            targetedData.featureMetadata = copy.deepcopy(tmpFeatureMetadata)

            ## featureMetadataNotExported
            # add _batchX to the column names to exclude. The expected columns are 'mergeCol' from featureMetadata. No modification for sampleMetadataNotExported which has been copied with the other Attributes (and is an SOP parameter)
            notExportedSelf = batchListReNumber(self.Attributes['featureMetadataNotExported'], batchChangeSelf, mergeCol)
            notExportedOther = batchListReNumber(other.Attributes['featureMetadataNotExported'], batchChangeOther, mergeCol)
            targetedData.Attributes['featureMetadataNotExported'] = list(set().union(notExportedSelf, notExportedOther))

            ## _intensityData
            # samples are simply concatenated, but features are merged. Reproject each dataset on the merge feature list before concatenation.
            # init with nan
            intensityData1 = numpy.full([self._intensityData.shape[0], targetedData.featureMetadata.shape[0]], numpy.nan)
            intensityData2 = numpy.full([other._intensityData.shape[0], targetedData.featureMetadata.shape[0]], numpy.nan)
            # iterate over the merged features
            for i in range(targetedData.featureMetadata.shape[0]):
                featureName = targetedData.featureMetadata.loc[i, 'Feature Name']
                featurePosition1 = self.featureMetadata['Feature Name'] == featureName
                featurePosition2 = other.featureMetadata['Feature Name'] == featureName
                if sum(featurePosition1) == 1:
                    intensityData1[:, i] = self._intensityData[:, featurePosition1].ravel()
                elif sum(featurePosition1) > 1:
                    raise ValueError('Duplicate feature name in first input: ' + featureName)
                if sum(featurePosition2) == 1:
                    intensityData2[:, i] = other._intensityData[:, featurePosition2].ravel()
                elif sum(featurePosition2) > 1:
                    raise ValueError('Duplicate feature name in second input: ' + featureName)
            intensityData = numpy.concatenate([intensityData1, intensityData2], axis=0)
            targetedData._intensityData = copy.deepcopy(intensityData)

            ## Masks
            targetedData.initialiseMasks()

            # sampleMask
            targetedData.sampleMask = numpy.concatenate([self.sampleMask, other.sampleMask], axis=0)

            # featureMask
            # if featureMask agree in both, keep that value. Otherwise let the default True value. If feature exist only in one, use that value.
            if (sum(~self.featureMask) != 0) | (sum(~other.featureMask) != 0):
                warnings.warn(
                    "Warning: featureMask are not empty, they will be merged. If both featureMasks do not agree, the default \'True\' value will be set. If the feature is only present in one dataset, the corresponding featureMask value will be kept.")
            for i in range(targetedData.featureMetadata.shape[0]):
                featureName = targetedData.featureMetadata.loc[i, 'Feature Name']
                featurePosition1 = self.featureMetadata['Feature Name'] == featureName
                featurePosition2 = other.featureMetadata['Feature Name'] == featureName
                # if both exist
                if (sum(featurePosition1) == 1) & (sum(featurePosition2) == 1):
                    # only False if both are False (otherwise True, same as default)
                    targetedData.featureMask[i] = self.featureMask[featurePosition1] | other.featureMask[featurePosition2]
                # if feature only exist in first input
                elif sum(featurePosition1 == 1):
                    targetedData.featureMask[i] = self.featureMask[featurePosition1]
                # if feature only exist in second input
                elif sum(featurePosition2 == 1):
                    targetedData.featureMask[i] = other.featureMask[featurePosition2]

            ## Excluded data with applyMask()
            # attribute doesn't exist the first time. From one round of __add__ onward the attribute is created and the length matches the number and order of 'Batch'
            if hasattr(self, 'sampleMetadataExcluded') & hasattr(other, 'sampleMetadataExcluded'):
                targetedData.sampleMetadataExcluded = concatenateList(self.sampleMetadataExcluded,
                                                                      other.sampleMetadataExcluded)
                targetedData.featureMetadataExcluded = concatenateList(self.featureMetadataExcluded,
                                                                       other.featureMetadataExcluded)
                targetedData.intensityDataExcluded = concatenateList(self.intensityDataExcluded,
                                                                     other.intensityDataExcluded)
                targetedData.excludedFlag = concatenateList(self.excludedFlag, other.excludedFlag)
                # add expectedConcentrationExcluded here too!
            elif hasattr(self, 'sampleMetadataExcluded'):
                targetedData.sampleMetadataExcluded = concatenateList(self.sampleMetadataExcluded, [])
                targetedData.featureMetadataExcluded = concatenateList(self.featureMetadataExcluded, [])
                targetedData.intensityDataExcluded = concatenateList(self.intensityDataExcluded, [])
                targetedData.excludedFlag = concatenateList(self.excludedFlag, [])
        elif hasattr(other, 'sampleMetadataExcluded'):
            targetedData.sampleMetadataExcluded = concatenateList([], other.sampleMetadataExcluded)
            targetedData.featureMetadataExcluded = concatenateList([], other.featureMetadataExcluded)
            targetedData.intensityDataExcluded = concatenateList([], other.intensityDataExcluded)
            targetedData.excludedFlag = concatenateList([], other.excludedFlag)
        else:
            targetedData.sampleMetadataExcluded = concatenateList([], [])
            targetedData.featureMetadataExcluded = concatenateList([], [])
            targetedData.intensityDataExcluded = concatenateList([], [])
            targetedData.excludedFlag = concatenateList([], [])

        # Deal with unexpected attributes
        expectedAttr = {'Attributes', 'VariableType', 'AnalyticalPlatform', '_Normalisation', '_name', 'fileName',
                        'filePath',
                        '_intensityData', 'sampleMetadata', 'featureMetadata', 'sampleMask',
                        'featureMask', 'calibration', 'sampleMetadataExcluded', 'intensityDataExcluded',
                        'featureMetadataExcluded', 'excludedFlag'}
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
                setattr(targetedData, k, [getattr(self, k), getattr(other, k)])
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
        if not validMergedDataset['NMRTargetedDataset']:
            raise ValueError('The merged dataset does not satisfy to the NMRTargetedDataset definition')

        ## Log
        targetedData.Attributes['Log'].append([datetime.now(),
                                               'Concatenated datasets %s (%i samples and %i features) and %s (%i samples and %i features), to a dataset of %i samples and %i features.' % (
                                               self.name, self.noSamples, self.noFeatures, other.name, other.noSamples,
                                               other.noFeatures, targetedData.noSamples, targetedData.noFeatures)])
        print(
            'Concatenated datasets %s (%i samples and %i features) and %s (%i samples and %i features), to a dataset of %i samples and %i features.' % (
            self.name, self.noSamples, self.noFeatures, other.name, other.noSamples, other.noFeatures,
            targetedData.noSamples, targetedData.noFeatures))

        return targetedData

    def __radd__(self, other):
        """
        Implements the summation of multiple :py:class:`TargetedNMRDataset`

        `targetedDataset = sum([ targetedNMRDatasetBatch1, targetedNMRDatasetBatch2, targetedNMRDatasetBatch3 ])`

        ..Note:: Sum always starts by the 0 integer and does `0.__add__(targetedNMRDatasetBatch1)` which fails and then calls the reverse add method `targetedDatasetBatch1.__radd_(0)`
        """
        if other == 0:
            return self
        else:
            return self.__add__(other)


def main():
    pass


if __name__ == '__main__':
    main()
