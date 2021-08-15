import copy
import os
import re
from datetime import datetime
import numpy
import pandas
import warnings
from .._toolboxPath import toolboxPath
from ._abstractTargetedDataset import AbstractTargetedDataset
from ..utilities import normalisation, rsd, calcAccuracy, calcPrecision, importBrukerXML, \
    readSkylineData, buildFileList
from ..enumerations import VariableType, AssayRole, SampleType, QuantificationType, CalibrationMethod, \
    AnalyticalPlatform


class NMRTargetedDataset(AbstractTargetedDataset):

    def __init__(self, datapath, fileType='Bruker IvDR', sop='Generic', **kwargs):
        """
        Initialisation and pre-processing of input data (load files and match data and calibration and SOP,
        apply limits of quantification).
        """

        super().__init__(sop=sop, **kwargs)

        # Load files and match data, calibration report and SOP, then Apply the limits of quantification
        if fileType == 'Bruker IvDR':
            self._loadBrukerNMRTargeted(datapath, sop=sop, **kwargs)
            self.VariableType = VariableType.Discrete
            self.AnalyticalPlatform = AnalyticalPlatform.NMR
            self.initialiseMasks()
        elif fileType == 'empty':
            # Build empty object for testing
            pass
        else:
            raise NotImplementedError

        # Check the final object is valid and log
        #if fileType != 'empty':
            #validDataset = self.validateObject(verbose=False, raiseError=False, raiseWarning=False)
            #if not validDataset['BasicTargetedDataset']:
            #    raise ValueError(
            #        'Import Error: The imported dataset does not satisfy to the Basic TargetedDataset definition')

        self.Attributes['Log'].append([datetime.now(),
                                       '%s instance initiated, with %d samples, %d features, from %s'
                                       % (self.__class__.__name__, self.noSamples, self.noFeatures, datapath)])

    def validateObject(self, verbose=True, raiseError=False, raiseWarning=True):
        """
        Checks that all the attributes specified in the class definition are present and of the required class and/or values.

        Returns 4 boolean: is the object a *Dataset* < a *basic TargetedDataset*
            < *has the object parameters for QC*
            < *has the object sample metadata*.

        To employ all class methods, the most inclusive (*has the object sample metadata*) must be successful:

        * *'Basic TargetedDataset'* checks :py:class:`~TargetedDataset` types and uniqueness as well as additional attributes.
        * *'has parameters for QC'* is *'Basic TargetedDataset'* + sampleMetadata[['SampleType, AssayRole, Dilution, Run Order, Batch, Correction Batch, Sample Base Name]]
        * *'has sample metadata'* is *'has parameters for QC'* + sampleMetadata[['Sample ID', 'Subject ID', 'Matrix']]

        If *'sampleMetadataExcluded'*, *'intensityDataExcluded'*, *'featureMetadataExcluded'*, *'expectedConcentrationExcluded'* or *'excludedFlag'* exist, the existence and number of exclusions (based on *'sampleMetadataExcluded'*) is checked

        Column type() in pandas.DataFrame are established on the first sample (for non int/float)
        featureMetadata are search for column names containing *'LLOQ'* & *'ULOQ'* to allow for *'LLOQ_batch...'* after :py:meth:`~TargetedDataset.__add__`, the first column matching is then checked for dtype
        If datasets are merged, calibration is a list of dict, and number of features is only kept constant inside each dict
        Does not check for uniqueness in :py:attr:`~sampleMetadata['Sample File Name']`
        Does not currently check for :py:attr:`~Attributes['Feature Name']`

        :param verbose: if True the result of each check is printed (default True)
        :type verbose: bool
        :param raiseError: if True an error is raised when a check fails and the validation is interrupted (default False)
        :type raiseError: bool
        :param raiseWarning: if True a warning is raised when a check fails
        :type raiseWarning: bool
        :return: A dictionary of 4 boolean with True if the Object conforms to the corresponding test. 'Dataset' conforms to :py:class:`Dataset`, 'BasicTargetedDataset' conforms to :py:class:`Dataset` + basic :py:class:`TargetedDataset`, 'QC' BasicTargetedDataset + object has QC parameters, 'sampleMetadata' QC + object has sample metadata information
        :rtype: dict
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

        # TODO: review at the end what checks are not covered in abstractTargetedDataset

        super().validateObject(verbose, raiseError, raiseWarning)

    def __add__(self, other):
        return NotImplementedError

    def __radd__(self, other):
        """
        :param other: Another NMRTargetedDataset
        :return: NMRTargetedDataset containing all samples in the individual dataset objects
        """
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def _loadBrukerNMRTargeted(self, datapath, unit=None, pdata=1, fileNamePattern=None, **kwargs):
        """
        Import a dataset from Bruker IvDr .xml files.
        :param datapath:
        :param unit:
        :param pdata:
        :param fileNamePattern:
        :param kwargs:
        :return:
        """

        if not isinstance(fileNamePattern, str):
            raise TypeError('\'fileNamePattern\' must be a string')
        if not isinstance(pdata, int):
            raise TypeError('\'pdata\' must be an integer')
        if unit is not None:
            if not isinstance(unit, str):
                raise TypeError('\'unit\' must be a string')

        # Build a list of xml files matching the pdata in the right folder
        pattern = re.compile(fileNamePattern)
        filelist = buildFileList(datapath, pattern)
        pdataPattern = re.compile('.*?pdata.*?%i' % (pdata))
        filelist = [x for x in filelist if pdataPattern.match(x)]
        # Import the data
        try:
            intensityData, sampleMetadata, featureMetadata, lodData = importBrukerXML(filelist)
        except IOError as ioerr:
            print('I')
            raise ioerr

        # Filter based on units if unit is provided
        avUnit = featureMetadata['Unit'].unique().tolist()
        if unit is not None:
            if unit not in featureMetadata['Unit'].unique().tolist():
                raise ValueError(
                    'The unit \'' + str(unit) + '\' is not present in the input data, available units: ' + str(avUnit))
            keepMask = (featureMetadata['Unit'] == unit).values
            featureMetadata = featureMetadata.loc[keepMask, :]
            featureMetadata.reset_index(drop=True, inplace=True)
            intensityData = intensityData[:, keepMask]

        # Check all features are unique - TODO: is this necessary??
        u_ids, u_counts = numpy.unique(featureMetadata['Feature Name'], return_counts=True)
        if not all(u_counts == 1):
            dupFeat = u_ids[u_counts != 1].tolist()
            warnings.warn(
                'The following features are present more than once, only the first occurence will be kept: ' + str(
                    dupFeat) + '. For further filtering, available units are: ' + str(avUnit))
            # only keep the first of duplicated features
            keepMask = ~featureMetadata['Feature Name'].isin(dupFeat).values
            keepFirstVal = [(featureMetadata['Feature Name'] == Feat).idxmax() for Feat in dupFeat]
            keepMask[keepFirstVal] = True
            featureMetadata = featureMetadata.loc[keepMask, :]
            featureMetadata.reset_index(drop=True, inplace=True)
            intensityData = intensityData[:, keepMask]

        # quantificationType
        featureMetadata['quantificationType'] = QuantificationType.BrukerivDrQuant
        featureMetadata.drop('type', inplace=True, axis=1)
        # calibrationMethod
        featureMetadata['calibrationMethod'] = CalibrationMethod.nmrCalibration

        # rename columns
        featureMetadata.rename(columns={'loq': 'LLOQ', 'lod': 'LOD', 'Lower Reference Bound': 'Lower Reference Percentile',
                     'Upper Reference Bound': 'Upper Reference Percentile'}, inplace=True)

        # replace '-' with nan
        #featureMetadata['LOD'].replace('-', numpy.nan, inplace=True)
        #featureMetadata['LOD'] = [float(x) for x in self.featureMetadata['LOD'].tolist()]

        # Acquired date??
        sampleMetadata['Order'] = sampleMetadata.sort_values(by='Acquired Time').index
        sampleMetadata['Run Order'] = sampleMetadata.sort_values(by='Order').index
        sampleMetadata.drop('Order', axis=1, inplace=True)
        # initialise the Batch to 1
        sampleMetadata['Batch'] = [1] * sampleMetadata.shape[0]
        sampleMetadata['Metadata Available'] = False

        ## Initialise sampleMetadata
        sampleMetadata['AssayRole'] = numpy.nan
        sampleMetadata['SampleType'] = numpy.nan
        sampleMetadata['Dilution'] = 100
        sampleMetadata['Correction Batch'] = numpy.nan
        sampleMetadata['Sample ID'] = numpy.nan
        sampleMetadata['Exclusion Details'] = None

        self._intensityData = intensityData
        self.sampleMetadata = sampleMetadata
        self.featureMetadata = featureMetadata
        self._lodMatrix = lodData

        self.expectedConcentration = pandas.DataFrame(None, index=list(self.sampleMetadata.index),
                                                      columns=self.featureMetadata['Feature Name'].tolist())

        # Summary
        print('NMR Targeted Method: ' + self.Attributes['methodName'])
        print(str(self.sampleMetadata.shape[0]) + ' study samples')
        print(str(self.featureMetadata.shape[0]) + ' features')

        return None
