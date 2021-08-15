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
from ..utilities import normalisation, rsd, calcAccuracy, calcPrecision, importBrukerXML, \
    readSkylineData, buildFileList
from ..enumerations import VariableType, AssayRole, SampleType, QuantificationType, CalibrationMethod, \
    AnalyticalPlatform


class AbstractTargetedDataset(Dataset):
    """
    Base class for nPYc TargetedDataset objects.

    :param str sop: Load configuration parameters from specified SOP JSON file
    :param datapath:
    """
    def __init__(self, fileType='empty', sop='Generic', **kwargs):
        """
        Initialize
        """

        super().__init__(sop=sop, **kwargs)

        self.expectedConcentration = None
        self.expectedConcentrationExcluded = None
        self._lodMatrix = numpy.array(None)

        # Check the final object is valid and log - in theory if all code works well this is unecessary
        # TODO: make validate object for this class lightweight and then remove from __init__
        if fileType != 'empty':
            validDataset = self.validateObject(verbose=False, raiseError=False, raiseWarning=False)
            if not validDataset['BasicTargetedDataset']:
                raise ValueError(
                    'Import Error: The imported dataset does not satisfy to the Basic TargetedDataset definition')

    # TODO: rsd methods could be pushed to top level dataset object
    @property
    def rsdSP(self):
        """
        Returns percentage :term:`relative standard deviations<RSD>` for each feature in the dataset,
        calculated on samples with the Assay Role :py:attr:`~nPYc.enumerations.AssayRole.PrecisionReference`
        and Sample Type :py:attr:`~nPYc.enumerations.SampleType.StudyPool` in :py:attr:`~Dataset.sampleMetadata`.
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

    # TODO: rsd methods could be pushed to top level dataset object
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

    @property
    def intensityData(self):
        """
        Return Intensity data matrix filtered by LODs
        """
        intensityData = self._intensityData * (100 / self.sampleMetadata['Dilution']).values[:,
                                                          numpy.newaxis]
        return intensityData

    @property
    def rawIntensityData(self):
        return self._intensityData

    @property
    def lodMatrix(self):
        return self._lodMatrix

    # TODO: Generalize to allow calculation per arbitrary sample type, or list of Sample IDs
    def calculateAccuracyPrecision(self, metric='Accuracy', onlyPrecisionReferences=False):
        """
        Return Precision (percent RSDs) and Accuracy for each SampleType and each unique concentration.
        Statistic grouped by SampleType, Feature and unique concentration.

        :param TargetedDataset dataset: TargetedDataset object to generate the accuracy and precision for.
        :param bool onlyPrecisionReference: If ``True`` only use samples with the `AssayRole` PrecisionReference.
        :returns: Dict of Accuracy and Precision dict for each group.
        :rtype: dict(str:dict(str:pandas.DataFrame))
        :raises TypeError: if dataset is not an instance of TargetedDataset
        """

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

    def addSampleInfo(self, descriptionFormat=None, filePath=None, **kwargs):
        """
        Load additional metadata and map it in to the :py:attr:`~Dataset.sampleMetadata` table.

        Possible options:

        * **'NPC Subject Info'** Map subject metadata from a NPC sample manifest file (format defined in 'PCSOP.082')
        * **'Raw Data'** Extract analytical parameters from raw data files
        * **'ISATAB'** ISATAB study designs
        * **'Filenames'** Parses sample information out of the filenames, based on the named capture groups in the regex passed in *filenamespec*
        * **'Basic CSV'** Joins the :py:attr:`sampleMetadata` table with the data in the ``csv`` file at *filePath=*, matching on the 'Sample File Name' column in both.

        :param str descriptionFormat: Format of metadata to be added
        :param str filePath: Path to the additional data to be added
        :param filenameSpec: Only used if *descriptionFormat* is 'Filenames'. A regular expression that extracts sample-type information into the following named capture groups: 'fileName', 'baseName', 'study', 'chromatography' 'ionisation', 'instrument', 'groupingKind' 'groupingNo', 'injectionKind', 'injectionNo', 'reference', 'exclusion' 'reruns', 'extraInjections', 'exclusion2'. if ``None`` is passed, use the *filenameSpec* key in *Attributes*, loaded from the SOP json
        :type filenameSpec: None or str
        :raises NotImplementedError: if the descriptionFormat is not understood
        """
        super().addSampleInfo(descriptionFormat=descriptionFormat, filePath=filePath, **kwargs)

    def addQuantificationInfo(self, filePath=None, descriptionFormat='CSV', **kwargs):
        """
        Load the expected quantification values for specific features in a set of samples and write to the :py:attr:`~TargetedDataset.expectedConcentration` table.
        Possible descriptionFormat:
        * **'CSV'** CSV file containing a series of features in columns, and Sample File Names as index
        :param str filePath: Path to the quantification info file
        :param str descriptionFormat: Format of quantification info file to be added
        :raises NotImplementedError: if the descriptionFormat is not understood
        :raises ValueError: If the input file contains format errors, duplicated columns, or mismatched samples/features.
        """

        try:
            if descriptionFormat == 'CSV':
                concentrationFile = pandas.read_csv(filePath)

                # File checks
                if any(~concentrationFile['Sample File Name'].isin(self.sampleMetadata['Sample File Name'])):
                    raise ValueError('CSV file contains Sample File Name not found in dataset')
                if any(~concentrationFile.columns.isin(self.featureMetadata['Feature Name'])):
                    raise ValueError('CSV file contains Feature Name not found in dataset')
                if any(~concentrationFile.apply(lambda x: pandas.to_numeric(x, errors='coerce').notnull().all())):
                    raise ValueError('CSV file contains non-numeric values')
                if any(concentrationFile.columns.duplicated()):
                    raise ValueError('Duplicated features found in CSV file provided')
                if any(concentrationFile['Sample File Name'].duplicated()):
                    raise ValueError('Duplicated Sample File Name found in CSV file provided')

                self.expectedConcentration = concentrationFile
            else:
                return NotImplementedError
        except IOError:
            print("{0} is not a valid CSV file".format(filePath))
        except LookupError as lokerr:
            raise lokerr

    def applyMasks(self):
        """
        Permanently delete elements masked (those set to ``False``) in :py:attr:`~Dataset.sampleMask` and :py:attr:`~Dataset.featureMask`, from :py:attr:`~Dataset.featureMetadata`, :py:attr:`~Dataset.sampleMetadata`, :py:attr:`~Dataset.intensityData` and py:attr:`TargetedDataset.expectedConcentration`.
        """
        # Filter TargetedDataset.expectedConcentration as it is not present in Dataset
        if (sum(self.sampleMask is False) > 0) | (sum(self.featureMask is False) > 0):

            # Instantiate lists if first application
            if not hasattr(self, 'sampleMetadataExcluded'):
                self.expectedConcentrationExcluded = []

            # Samples
            if sum(self.sampleMask) != len(self.sampleMask):
                # Save excluded samples
                self.expectedConcentrationExcluded.append(self.expectedConcentration.loc[~self.sampleMask, :])
                # Delete excluded samples
                self.expectedConcentration = self.expectedConcentration.loc[self.sampleMask]
                self.expectedConcentration.reset_index(drop=True, inplace=True)

            # Features
            if sum(self.featureMask) != len(self.featureMask):
                # Save excluded features
                self.expectedConcentrationExcluded.append(self.expectedConcentration.loc[:, ~self.featureMask])
                # Delete excluded features
                self.expectedConcentration = self.expectedConcentration.loc[:, self.featureMask]
                self.expectedConcentration.reset_index(drop=True, inplace=True)

        # applyMasks to the rest of TargetedDataset
        super().applyMasks()

    # TODO: What filters are general? And are they suitable to be automated in this manner?
    def updateMasks(self, filterSamples=True, filterFeatures=True,
                    sampleTypes=[SampleType.StudySample, SampleType.StudyPool],
                    assayRoles=[AssayRole.Assay, AssayRole.PrecisionReference],
                    quantificationTypes=[QuantificationType.IS, QuantificationType.QuantOwnLabeledAnalogue,
                                         QuantificationType.QuantAltLabeledAnalogue, QuantificationType.QuantOther,
                                         QuantificationType.Monitored],
                    calibrationMethods=[CalibrationMethod.backcalculatedIS, CalibrationMethod.noIS,
                                        CalibrationMethod.noCalibration, CalibrationMethod.otherCalibration],
                    featureFilters={'accuracyFilter': False, 'precisionFilter': False}, **kwargs):
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

        if any([type(x) is not bool for x in featureFilters.values()]):
            raise TypeError('Only bool values should be passed in featureFilters')
        # Fill in dictionary provided with default value if no argument is passed
        default_args = {'rsdFilter': False, 'accuracyFilter': False, 'precisionFilter': False}

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

        # Feature Exclusions
        if filterFeatures:
            quantTypeMask = self.featureMetadata['quantificationType'].isin(quantificationTypes)
            calibMethodMask = self.featureMetadata['calibrationMethod'].isin(calibrationMethods)

            featureMask = numpy.logical_and(quantTypeMask, calibMethodMask).values

            self.featureMask = numpy.logical_and(featureMask, self.featureMask)
            self.featureMetadata['Passing Selection'] = self.featureMask

        # Sample Exclusions
        if filterSamples:
            sampleMask = self.sampleMetadata['SampleType'].isin(sampleTypes)
            assayMask = self.sampleMetadata['AssayRole'].isin(assayRoles)

            sampleMask = numpy.logical_and(sampleMask, assayMask).values

            self.sampleMask = numpy.logical_and(sampleMask, self.sampleMask)

        self.Attributes['Log'].append([datetime.now(),
                                       'Dataset filtered with: filterSamples=%s, filterFeatures=%s, sampleTypes=%s, assayRoles=%s, quantificationTypes=%s, calibrationMethods=%s' % (
                                           filterSamples, filterFeatures, sampleTypes, assayRoles, quantificationTypes,
                                           calibrationMethods)])

    def exportDataset(self, destinationPath='.', saveFormat='CSV', lodFilteredIntensities=True, withExclusions=True, escapeDelimiters=False,
                      filterMetadata=True):
        """
        Calls :py:meth:`~Dataset.exportDataset` and raises a warning if normalisation is employed as :py:class:`TargetedDataset` :py:attr:`intensityData` can be left-censored.
        """
        # handle the dilution due to method... These lines are left here commented - as hopefully this will be handled more
        # elegantly through the intensityData getter
        # Export dataset...
        tmpData = copy.deepcopy(self)
        tmpData._intensityData = tmpData._intensityData * (100 / tmpData.sampleMetadata['Dilution']).values[:,
                                                          numpy.newaxis]
        super(AbstractTargetedDataset, tmpData).exportDataset(destinationPath=destinationPath, saveFormat=saveFormat,
                                                      withExclusions=withExclusions, escapeDelimiters=escapeDelimiters,
                                                      filterMetadata=filterMetadata)

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
        :raises TypeError: if self.VariableType is not an enum 'VariableType'
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
        :raises AttributeError: if self.expectedConcentration does not exist
        :raises TypeError: if self.expectedConcentration is not a pandas.DataFrame
        :raises ValueError: if self.expectedConcentration does not have the same number of samples as self._intensityData
        :raises ValueError: if self.expectedConcentration does not have the same number of features as self._intensityData
        :raises ValueError: if self.expectedConcentration column name do not match self.featureMetadata['Feature Name']
        """

        def conditionTest(successCond, successMsg, failureMsg, allFailures,
                          verb, raiseErr, raiseWarn, exception):
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

            return allFailures

        # init
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
            condition = isinstance(self, AbstractTargetedDataset)
            success = 'Check Object class:\tOK'
            failure = 'Check Object class:\tFailure, not TargetedDataset, but ' + str(type(self))
            failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose, raiseError,
                                             raiseWarning, exception=TypeError(failure))

            ## Attributes
            # methodName exist
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

            ## self.featureMetadata - specific TargetedDataset Fields
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

            # ['LLOQ']
            tmpLLOQMatch = self.featureMetadata.columns.to_series().str.contains('LLOQ')
            condition = (sum(tmpLLOQMatch) > 0)
            success = 'Check self.featureMetadata[\'LLOQ\'] or similar exists:\tOK'
            failure = 'Check self.featureMetadata[\'LLOQ\'] or similar exists:\tFailure, \'self.featureMetadata\' lacks a \'LLOQ\' or \'LLOQ_batch\' column'
            failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose, raiseError,
                                             raiseWarning, exception=LookupError(failure))
            if condition:
                # featureMetadata['LLOQ'] is a float, try on first found
                condition = ((self.featureMetadata.loc[:, tmpLLOQMatch].iloc[:, 0].dtype == numpy.dtype(float)) | (
                            self.featureMetadata.loc[:, tmpLLOQMatch].iloc[:, 0].dtype == numpy.dtype(
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

            # ['ULOQ']
            tmpULOQMatch = self.featureMetadata.columns.to_series().str.contains('ULOQ')
            condition = (sum(tmpULOQMatch) > 0)
            success = 'Check self.featureMetadata[\'ULOQ\'] or similar exists:\tOK'
            failure = 'Check self.featureMetadata[\'ULOQ\'] or similar exists:\tFailure, \'self.featureMetadata\' lacks a \'ULOQ\' or \'ULOQ_batch\' column'
            failureListBasic = conditionTest(condition, success, failure, failureListBasic, verbose, raiseError,
                                             raiseWarning, exception=LookupError(failure))
            if condition:
                # featureMetadata['ULOQ'] is a float, try on first found
                condition = ((self.featureMetadata.loc[:, tmpULOQMatch].iloc[:, 0].dtype == numpy.dtype(float)) | (
                            self.featureMetadata.loc[:, tmpULOQMatch].iloc[:, 0].dtype == numpy.dtype(
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

            # self.expectedConcentration exist
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

            ## List additional attributes (print + log)
            expectedSet = set({'Attributes', 'VariableType', '_Normalisation', '_name',
                               '_intensityData', 'featureMetadata', 'expectedConcentration',
                               'sampleMask',
                               'featureMask', 'sampleMetadataExcluded', 'intensityDataExcluded',
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
            return {'Dataset': False, 'BasicTargetedDataset': False, 'QC': False, 'sampleMetadata': False}

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

        intensityData = self.intensityData
        intensityData = pandas.DataFrame(intensityData)
        intensityData.replace(to_replace=-numpy.inf, value='<LLOQ', inplace=True)
        intensityData.replace(to_replace=numpy.inf, value='>ULOQ', inplace=True)
        intensityData.replace(to_replace=numpy.nan, value='NA', inplace=True)

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

        intensityData = self.intensityData
        intensityData = pandas.DataFrame(intensityData)
        intensityData.replace(to_replace=-numpy.inf, value='<LLOQ', inplace=True)
        intensityData.replace(to_replace=numpy.inf, value='>ULOQ', inplace=True)
        intensityData.replace(to_replace=numpy.nan, value='NA', inplace=True)

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

