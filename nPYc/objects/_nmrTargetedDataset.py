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
        self.filePath, fileName = os.path.split(datapath)
        self.fileName, fileExtension = os.path.splitext(fileName)
        self.name = self.fileName

        # Load files and match data, calibration report and SOP, then Apply the limits of quantification
        if fileType == 'Bruker IvDR':
            self._loadBrukerNMRTargeted(datapath, sop=sop)
            self.VariableType = VariableType.Discrete
            self.AnalyticalPlatform = AnalyticalPlatform.MS
            self.initialiseMasks()
            self._lodData = None
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

    @property
    def intensityData(self):
        """
        Return intensity data matrix filtered by LODs
        """
        intensityData = self._intensityData * (100 / self.sampleMetadata['Dilution']).values[:,
                                                          numpy.newaxis]

        # Filter data as < limit of quantification on a sample per sample basis.
        if self._lodData is not None:
            intensityData[self._lodData == True] = -numpy.inf

        return intensityData

    @property
    def rawIntensityData(self):
        """
        Return the raw
        :return: raw (non-LOD masked intensityData)
        """
        intensityData = self._intensityData * (100 / self.sampleMetadata['Dilution']).values[:,
                                                          numpy.newaxis]
        return intensityData

    def _loadBrukerNMRTargeted(self, filepath, unit=None, pdata=1, fileNamePattern=None, **kwargs):
        """

        :param filepath:
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

        ## Build a list of xml files matching the pdata in the right folder
        pattern = re.compile(fileNamePattern)
        filelist = buildFileList(filepath, pattern)
        pdataPattern = re.compile('.*?pdata.*?%i' % (pdata))
        filelist = [x for x in filelist if pdataPattern.match(x)]

        try:
            intensityData, sampleMetadata, featureMetadata, lodData = importBrukerXML(filelist)
        except IOError as ioerr:
            print('I')
            raise ioerr

        ## Filter unit if required
        avUnit = featureMetadata['Unit'].unique().tolist()
        if unit is not None:
            if unit not in featureMetadata['Unit'].unique().tolist():
                raise ValueError(
                    'The unit \'' + str(unit) + '\' is not present in the input data, available units: ' + str(avUnit))
            keepMask = (featureMetadata['Unit'] == unit).values
            featureMetadata = featureMetadata.loc[keepMask, :]
            featureMetadata.reset_index(drop=True, inplace=True)
            intensityData = intensityData[:, keepMask]

        ## Check all features are unique
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
        featureMetadata['LLOQ'].replace('-', numpy.nan, inplace=True)
        featureMetadata['LLOQ'] = [float(x) for x in self.featureMetadata['LLOQ'].tolist()]
        featureMetadata['LOD'].replace('-', numpy.nan, inplace=True)
        featureMetadata['LOD'] = [float(x) for x in self.featureMetadata['LOD'].tolist()]
        # ULOQ
        featureMetadata['ULOQ'] = numpy.nan

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
        self._lodData = lodData

        self.expectedConcentration = pandas.DataFrame(None, index=list(self.sampleMetadata.index),
                                                      columns=self.featureMetadata['Feature Name'].tolist())

        ## Summary
        print('NMR Targeted Method: ' + self.Attributes['methodName'])
        print(str(self.sampleMetadata.shape[0]) + ' study samples')
        print(str(self.featureMetadata.shape[0]) + ' features')

        return None

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