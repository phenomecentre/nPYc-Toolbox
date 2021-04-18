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
from ._abstractTargetedDataset import AbstractTargetedDataset
from ..utilities import normalisation, rsd, calcAccuracy, calcPrecision, importBrukerXML, \
    readSkylineData, buildFileList
from ..enumerations import VariableType, AssayRole, SampleType, QuantificationType, CalibrationMethod, \
    AnalyticalPlatform


class MSTargetedDataset(AbstractTargetedDataset):

    def __init__(self, datapath, fileType='Skyline', sop='Generic', **kwargs):
        """
        Initialisation and pre-processing of input data (load files and match data and calibration and SOP,
        apply limits of quantification).
        """

        super().__init__(sop=sop, **kwargs)
        self.filePath, fileName = os.path.split(datapath)
        self.fileName, fileExtension = os.path.splitext(fileName)
        self.name = self.fileName

        # Load files and match data, calibration report and SOP, then Apply the limits of quantification
        if fileType == 'Skyline':
            # Read files, filter calibration samples, filter IS, applyLLOQ, clean object
            self._loadSkylineDataset(datapath, **kwargs)
            # Finalise object
            self.VariableType = VariableType.Discrete
            self.AnalyticalPlatform = AnalyticalPlatform.MS
            self.initialiseMasks()
            self.lodData = None
            self.ISIntensityData = None
            self.expectedConcentration = None
        elif fileType == 'TargetLynx':
            return NotImplementedError
        elif fileType == 'Biocrates':
            self._loadBiocratesTargeted(datapath, sop=sop)
            self.VariableType = VariableType.Discrete
            self.AnalyticalPlatform = AnalyticalPlatform.MS
            self.initialiseMasks()
            self.lodData = None
            self.ISIntensityData = None
            self.expectedConcentration = None
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

        self.Attributes['Log'].append([datetime.now(),
                                       '%s instance initiated, with %d samples, %d features, from %s'
                                       % (self.__class__.__name__, self.noSamples, self.noFeatures, datapath)])

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
        col_LLOQ = self.featureMetadata.columns[
            self.featureMetadata.columns.to_series().str.contains('LLOQ_batch')].tolist()
        col_LLOQ_batch = sorted([int(i.replace('LLOQ_batch', '')) for i in col_LLOQ])
        col_ULOQ = self.featureMetadata.columns[
            self.featureMetadata.columns.to_series().str.contains('ULOQ_batch')].tolist()
        col_ULOQ_batch = sorted([int(i.replace('ULOQ_batch', '')) for i in col_ULOQ])
        batches = sorted((numpy.unique(self.sampleMetadata.loc[:, 'Batch'].values[
                                           ~numpy.isnan(self.sampleMetadata.loc[:, 'Batch'].values)])).astype(int))
        if (col_LLOQ_batch != batches) | (col_ULOQ_batch != batches):
            raise ValueError(
                'Import Error: targetedData does not have the same number of batch, LLOQ_batchX and ULOQ_batchX: ' + str(
                    batches) + ', ' + str(col_LLOQ) + ', ' + str(col_ULOQ) + '. LOQs must have already been merged!')

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
        self.Attributes['Log'].append(
            [datetime.now(), 'LOQ merged (keepBatchLOQ =  %s, onlyLLOQ = %s).' % (keepBatchLOQ, onlyLLOQ)])
        if onlyLLOQ:
            print('Limits of quantification merged to the highest LLOQ across batch')
        else:
            print('Limits of quantification merged to the highest LLOQ and lowest ULOQ across batch')

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

        sampleMetadata = copy.deepcopy(self.sampleMetadata)
        featureMetadata = copy.deepcopy(self.featureMetadata)
        intensityData = copy.deepcopy(self._intensityData)
        expectedConcentration = copy.deepcopy(self.expectedConcentration)
        calibration = copy.deepcopy(self.calibration)
        if ((not hasattr(self, 'sampleMetadataExcluded')) | (not hasattr(self, 'featureMetadataExcluded')) | (
        not hasattr(self, 'intensityDataExcluded')) | (not hasattr(self, 'expectedConcentrationExcluded')) | (
        not hasattr(self, 'excludedFlag'))):
            sampleMetadataExcluded = []
            featureMetadataExcluded = []
            intensityDataExcluded = []
            expectedConcentrationExcluded = []
            excludedFlag = []
        else:
            sampleMetadataExcluded = copy.deepcopy(self.sampleMetadataExcluded)
            featureMetadataExcluded = copy.deepcopy(self.featureMetadataExcluded)
            intensityDataExcluded = copy.deepcopy(self.intensityDataExcluded)
            expectedConcentrationExcluded = copy.deepcopy(self.expectedConcentrationExcluded)
            excludedFlag = copy.deepcopy(self.excludedFlag)

        ## Check input columns
        if 'LLOQ' not in featureMetadata.columns:
            raise AttributeError('the featureMetadata[\'LLOQ\'] column is absent')
        if onlyLLOQ == False:
            if 'ULOQ' not in featureMetadata.columns:
                raise AttributeError('featureMetadata[\'ULOQ\'] column is absent')

        ## Features only Monitored are not processed and passed untouched (concatenated back at the end)
        untouched = (featureMetadata['quantificationType'] == QuantificationType.Monitored).values
        if sum(untouched) != 0:
            print('The following features are only monitored and therefore not processed for LOQs: ' + str(
                featureMetadata.loc[untouched, 'Feature Name'].values.tolist()))
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
        unusableFeat = featureMetadata['LLOQ'].isnull().values & (
                    featureMetadata['quantificationType'] != QuantificationType.QuantOther).values
        if not onlyLLOQ:
            unusableFeat = unusableFeat | (featureMetadata['ULOQ'].isnull().values & (
                        featureMetadata['quantificationType'] != QuantificationType.QuantOther).values)
        if sum(unusableFeat) != 0:
            print(str(sum(unusableFeat)) + ' features cannot be pre-processed:')
            print('\t' + str(
                sum(unusableFeat)) + ' features lack the required information to apply limits of quantification')
            # store
            sampleMetadataExcluded.append(sampleMetadata)
            featureMetadataExcluded.append(featureMetadata.loc[unusableFeat, :])
            intensityDataExcluded.append(intensityData[:, unusableFeat])
            expectedConcentrationExcluded.append(expectedConcentration.loc[:, unusableFeat])
            excludedFlag.append('Features')
            # remove
            featureMetadata = featureMetadata.loc[~unusableFeat, :]
            intensityData = intensityData[:, ~unusableFeat]
            expectedConcentration = expectedConcentration.loc[:, ~unusableFeat]
            if isinstance(calibration, dict):
                calibration['calibFeatureMetadata'] = calibration['calibFeatureMetadata'].loc[~unusableFeat, :]
                calibration['calibIntensityData'] = calibration['calibIntensityData'][:, ~unusableFeat]
                calibration['calibExpectedConcentration'] = calibration['calibExpectedConcentration'].loc[:,
                                                            ~unusableFeat]

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
            expectedConcentration = pandas.concat([expectedConcentration, untouchedExpectedConcentration], axis=1,
                                                  sort=False)
            # reorder the calib
            if isinstance(calibration, dict):
                calibration['calibFeatureMetadata'] = pandas.concat(
                    [calibration['calibFeatureMetadata'], untouchedCalibFeatureMetadata], axis=0, sort=False)
                calibration['calibIntensityData'] = numpy.concatenate(
                    (calibration['calibIntensityData'], untouchedCalibIntensityData), axis=1)
                calibration['calibExpectedConcentration'] = pandas.concat(
                    [calibration['calibExpectedConcentration'], untouchedCalibExpectedConcentration], axis=1,
                    sort=False)

        # Remove excess info
        featureMetadata.reset_index(drop=True, inplace=True)
        expectedConcentration.reset_index(drop=True, inplace=True)
        if isinstance(calibration, dict):
            calibration['calibFeatureMetadata'].reset_index(drop=True, inplace=True)
            calibration['calibExpectedConcentration'].reset_index(drop=True, inplace=True)

        ## return dataset with limits of quantification applied
        self.featureMetadata = featureMetadata
        self._intensityData = intensityData
        self.expectedConcentration = expectedConcentration
        self.calibration = calibration
        self.sampleMetadataExcluded = sampleMetadataExcluded
        self.featureMetadataExcluded = featureMetadataExcluded
        self.intensityDataExcluded = intensityDataExcluded
        self.expectedConcentrationExcluded = expectedConcentrationExcluded
        self.excludedFlag = excludedFlag
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
            logUntouchedFeatures = ' ' + str(sum(untouched)) + ' features only monitored and not processed: ' + str(
                untouchedFeatureMetadata.loc[:, 'Feature Name'].values.tolist()) + '.'
        else:
            logUntouchedFeatures = ''
        self.Attributes['Log'].append([datetime.now(), '%s (%i samples, %i features). LLOQ are replaced by -inf.%s' % (
        logLimits, self.noSamples, self.noFeatures, logUntouchedFeatures)])

    def __add__(self, other):
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
                if isinstance(x, collections.Iterable) and not (isinstance(el, str) | isinstance(el, dict)):
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
                        updatedcalib[j]['calibSampleMetadata'].loc[
                            calib[j]['calibSampleMetadata']['Batch'] == batchNum, 'Batch'] = batchChange[batchNum]

            elif isinstance(calib, dict):
                updatedcalib = copy.deepcopy(calib)
                # modify batch number
                for batchNum in batchChange.keys():
                    updatedcalib['calibSampleMetadata'].loc[
                        calib['calibSampleMetadata']['Batch'] == batchNum, 'Batch'] = batchChange[batchNum]

            return updatedcalib

        ## Input checks
        # Run validator (checks for duplicates in featureMetadata['Feature Name']). No check for AssayRole and SampleType as sample info data might not have been imported yet
        validSelfDataset = self.validateObject(verbose=False, raiseError=False, raiseWarning=False)
        validOtherDataset = other.validateObject(verbose=False, raiseError=False, raiseWarning=False)
        if not validSelfDataset['BasicTargetedDataset']:
            raise ValueError(
                'self does not satisfy to the Basic TargetedDataset definition, check with self.validateObject(verbose=True, raiseError=False)')
        if not validOtherDataset['BasicTargetedDataset']:
            raise ValueError(
                'other does not satisfy to the Basic TargetedDataset definition, check with other.validateObject(verbose=True, raiseError=False)')
        # Warning if duplicate 'Sample File Name' in sampleMetadata
        u_ids, u_counts = numpy.unique(
            pandas.concat([self.sampleMetadata['Sample File Name'], other.sampleMetadata['Sample File Name']],
                          ignore_index=True, sort=False), return_counts=True)
        if any(u_counts > 1):
            warnings.warn('Warning: The following \'Sample File Name\' are present more than once: ' + str(
                u_ids[u_counts > 1].tolist()))

        if self.AnalyticalPlatform != other.AnalyticalPlatform:
            raise ValueError('Can only add Targeted datasets with the same AnalyticalPlatform Attribute')

        ## Initialise an empty TargetedDataset to overwrite
        targetedData = AbstractTargetedDataset(datapath='', fileType='empty')

        ## Attributes
        if self.Attributes['methodName'] != other.Attributes['methodName']:
            raise ValueError(
                'Cannot concatenate different targeted methods: \'' + self.Attributes['methodName'] + '\' and \'' +
                other.Attributes['methodName'] + '\'')
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
        targetedData.name = self.name + '-' + other.name

        ## fileName
        targetedData.fileName = flatten([self.fileName, other.fileName])

        ## filePath
        targetedData.filePath = flatten([self.filePath, other.filePath])

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

        ## expectedConcentration
        # same approach as _intensityData, samples are concatenated but features are merged. validObject() on input ensures expectedConcentration.columns match featureMetadata['Feature Name']
        expectedConc1 = pandas.DataFrame(
            numpy.full([self.expectedConcentration.shape[0], targetedData.featureMetadata.shape[0]], numpy.nan),
            columns=targetedData.featureMetadata['Feature Name'].tolist())
        expectedConc2 = pandas.DataFrame(
            numpy.full([other.expectedConcentration.shape[0], targetedData.featureMetadata.shape[0]], numpy.nan),
            columns=targetedData.featureMetadata['Feature Name'].tolist())
        # iterate over the merged features
        for colname in targetedData.featureMetadata['Feature Name'].tolist():
            if colname in self.expectedConcentration.columns:
                expectedConc1.loc[:, colname] = self.expectedConcentration[colname].ravel()
            if colname in other.expectedConcentration.columns:
                expectedConc2.loc[:, colname] = other.expectedConcentration[colname].ravel()
        expectedConcentration = pandas.concat([expectedConc1, expectedConc2], axis=0, ignore_index=True, sort=False)
        expectedConcentration.reset_index(drop=True, inplace=True)
        targetedData.expectedConcentration = copy.deepcopy(expectedConcentration)

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
            targetedData.expectedConcentrationExcluded = concatenateList(self.expectedConcentrationExcluded,
                                                                         other.expectedConcentrationExcluded)
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
        tmpCalibSelf = copy.deepcopy(self.calibration)
        tmpCalibSelf = updatecalibBatch(tmpCalibSelf, batchChangeSelf)
        tmpCalibOther = copy.deepcopy(other.calibration)
        tmpCalibOther = updatecalibBatch(tmpCalibOther, batchChangeOther)
        targetedData.calibration = flatten([tmpCalibSelf, tmpCalibOther])

        ## unexpected attributes
        expectedAttr = {'Attributes', 'VariableType', 'AnalyticalPlatform', '_Normalisation', '_name', 'fileName',
                        'filePath',
                        '_intensityData', 'sampleMetadata', 'featureMetadata', 'expectedConcentration', 'sampleMask',
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
        if not validMergedDataset['BasicTargetedDataset']:
            raise ValueError('The merged dataset does not satisfy to the Basic TargetedDataset definition')

        ## Log
        targetedData.Attributes['Log'].append([datetime.now(),
                                               'Concatenated datasets %s (%i samples and %i features) and %s (%i samples and %i features), to a dataset of %i samples and %i features.' % (
                                               self.name, self.noSamples, self.noFeatures, other.name, other.noSamples,
                                               other.noFeatures, targetedData.noSamples, targetedData.noFeatures)])
        print(
            'Concatenated datasets %s (%i samples and %i features) and %s (%i samples and %i features), to a dataset of %i samples and %i features.' % (
            self.name, self.noSamples, self.noFeatures, other.name, other.noSamples, other.noFeatures,
            targetedData.noSamples, targetedData.noFeatures))

        ## Remind to mergeLimitsOfQuantification
        warnings.warn(
            'Update the limits of quantification using `mergedDataset.mergeLimitsOfQuantification()` (keeps the lowest common denominator across all batch: highest LLOQ, lowest ULOQ)')

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
