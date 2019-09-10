import numpy
import datetime
import pandas
from xml.etree.ElementTree import ElementTree
from copy import deepcopy
from ...enumerations import QuantificationType, CalibrationMethod


def _loadTargetLynxDataset(self, datapath, calibrationReportPath, keepIS=False, noiseFilled=False, keepPeakInfo=False,
                           keepExcluded=False, **kwargs):
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
        print('IS features are kept for processing:', sum(self.featureMetadata['IS'].values), 'IS features,',
              sum(~self.featureMetadata['IS'].values), 'other features.')
        print('-----')
        self.Attributes['Log'].append([datetime.now(),
                                       'IS features kept for processing (%d samples). %d IS, %d other features.' % (
                                       self.noSamples, sum(self.featureMetadata['IS'].values),
                                       sum(~self.featureMetadata['IS'].values))])
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
    for j in ['keepIS', 'noiseFilled', 'keepPeakInfo', 'keepExcluded']:
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
    sampleMetadata, featureMetadata, intensityData, expectedConcentration, peakResponse, peakArea, peakConcentrationDeviation, peakIntegrationFlag, peakRT = self.__getDatasetFromXML(
        datapath)
    # Read calibration information from .csv (dumb, no metadata alteration, only checks for required columns)
    calibReport = self.__getCalibrationFromReport(calibrationReportPath)
    # Match XML, Calibration Report & SOP
    sampleMetadata, featureMetadata, intensityData, expectedConcentration, excludedImportSampleMetadata, excludedImportFeatureMetadata, excludedImportIntensityData, excludedImportExpectedConcentration, excludedImportFlag, peakResponse, peakArea, peakConcentrationDeviation, peakIntegrationFlag, peakRT = self.__matchDatasetToCalibrationReport(
        sampleMetadata, featureMetadata, intensityData, expectedConcentration, peakResponse, peakArea,
        peakConcentrationDeviation, peakIntegrationFlag, peakRT, calibReport)

    self.sampleMetadata = sampleMetadata
    self.featureMetadata = featureMetadata
    self._intensityData = intensityData
    self.expectedConcentration = expectedConcentration
    self.sampleMetadataExcluded = excludedImportSampleMetadata
    self.featureMetadataExcluded = excludedImportFeatureMetadata
    self.intensityDataExcluded = excludedImportIntensityData
    self.expectedConcentrationExcluded = excludedImportExpectedConcentration
    self.excludedFlag = excludedImportFlag
    self.peakInfo = {'peakResponse': peakResponse, 'peakArea': peakArea,
                     'peakConcentrationDeviation': peakConcentrationDeviation,
                     'peakIntegrationFlag': peakIntegrationFlag, 'peakRT': peakRT}

    # add Dataset mandatory columns
    self.sampleMetadata['AssayRole'] = numpy.nan
    self.sampleMetadata['SampleType'] = numpy.nan
    self.sampleMetadata['Dilution'] = numpy.nan
    self.sampleMetadata['Correction Batch'] = numpy.nan
    self.sampleMetadata['Sample ID'] = numpy.nan
    self.sampleMetadata['Exclusion Details'] = numpy.nan
    # self.sampleMetadata['Batch']             = numpy.nan #already created

    # clear SOP parameters not needed after __matchDatasetToCalibrationReport
    AttributesToRemove = ['compoundID', 'compoundName', 'IS', 'unitFinal', 'unitCorrectionFactor', 'calibrationMethod',
                          'calibrationEquation', 'quantificationType']
    AttributesToRemove.extend(self.Attributes['externalID'])
    for k in AttributesToRemove:
        del self.Attributes[k]

    self.Attributes['Log'].append([datetime.now(),
                                   'TargetLynx data file with %d samples, %d features, loaded from \%s, calibration report read from \%s\'' % (
                                   self.noSamples, self.noFeatures, datapath, calibrationReportPath)])


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

    inputData = ElementTree(file=path).getroot()[2][0]
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
    peakIntegrationFlag = peak_integrationFlag  # already dataframe
    peakIntegrationFlag.reset_index(drop=True, inplace=True)
    peakRT = pandas.DataFrame(peak_RT)

    # Convert to DataFrames
    featureMetadata = pandas.concat([pandas.DataFrame(featureMetadata[c], columns=[c]) for c in featureMetadata.keys()],
                                    axis=1, sort=False)
    sampleMetadata = pandas.concat([pandas.DataFrame(sampleMetadata[c], columns=[c]) for c in sampleMetadata.keys()],
                                   axis=1, sort=False)
    expectedConcentration.columns = featureMetadata['Feature Name'].values.tolist()
    peakIntegrationFlag.columns = featureMetadata['Feature Name'].values.tolist()
    peakResponse.columns = featureMetadata['Feature Name'].values.tolist()
    peakArea.columns = featureMetadata['Feature Name'].values.tolist()
    peakConcentrationDeviation.columns = featureMetadata['Feature Name'].values.tolist()
    peakRT.columns = featureMetadata['Feature Name'].values.tolist()
    sampleMetadata['Metadata Available'] = False

    return sampleMetadata, featureMetadata, intensityData, expectedConcentration, peakResponse, peakArea, peakConcentrationDeviation, peakIntegrationFlag, peakRT


def __matchDatasetToCalibrationReport(self, sampleMetadata, featureMetadata, intensityData, expectedConcentration,
                                      peakResponse, peakArea, peakConcentrationDeviation, peakIntegrationFlag,
                                      peakRT, calibReport):
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
    excludedSampleMetadata = []
    excludedFeatureMetadata = []
    excludedIntensityData = []
    excludedExpectedConcentration = []
    excludedFlag = []

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
    SOPColumnsToLoad = ['compoundID', 'compoundName', 'IS', 'unitFinal', 'unitCorrectionFactor',
                        'calibrationMethod', 'calibrationEquation', 'quantificationType']
    SOPColumnsToLoad.extend(self.Attributes['externalID'])
    SOPFeatureMetadata = pandas.DataFrame.from_dict(dict((k, self.Attributes[k]) for k in SOPColumnsToLoad),
                                                    orient='columns')
    SOPFeatureMetadata['compoundID'] = pandas.to_numeric(SOPFeatureMetadata['compoundID'])
    SOPFeatureMetadata['unitCorrectionFactor'] = pandas.to_numeric(SOPFeatureMetadata['unitCorrectionFactor'])
    SOPFeatureMetadata['IS'] = SOPFeatureMetadata['IS'].map({'True': True, 'False': False})
    SOPFeatureMetadata['Unit'] = SOPFeatureMetadata['unitFinal']
    SOPFeatureMetadata.drop('unitFinal', inplace=True, axis=1)

    # convert quantificationType from str to enum
    if 'quantificationType' in SOPFeatureMetadata.columns:
        for qType in QuantificationType:
            SOPFeatureMetadata.loc[
                SOPFeatureMetadata['quantificationType'].values == qType.name, 'quantificationType'] = qType
    # convert calibrationMethod from str to enum
    if 'calibrationMethod' in SOPFeatureMetadata.columns:
        for cMethod in CalibrationMethod:
            SOPFeatureMetadata.loc[
                SOPFeatureMetadata['calibrationMethod'].values == cMethod.name, 'calibrationMethod'] = cMethod

    # check that all quantificationType='IS' are also flagged as IS
    # (both have same number of feature + intersection has same number of feature as one of them)
    if (sum((SOPFeatureMetadata['quantificationType'] == QuantificationType.IS)) != sum(
            SOPFeatureMetadata['IS'])) | (sum(
            (SOPFeatureMetadata['quantificationType'] == QuantificationType.IS) & SOPFeatureMetadata['IS']) != sum(
            SOPFeatureMetadata['IS'])):
        raise ValueError(
            'Check SOP file, features with quantificationType=\'IS\' must have been flagged as IS=\'True\'')

    # check that all quantificationType='Monitored' have a calibrationMethod='noCalibration'
    # (both have same number of feature + intersection has same number of feature as one of them)
    if (sum((SOPFeatureMetadata['quantificationType'] == QuantificationType.Monitored)) != (
    sum(SOPFeatureMetadata['calibrationMethod'] == CalibrationMethod.noCalibration))) | (sum(
            (SOPFeatureMetadata['quantificationType'] == QuantificationType.Monitored) & (
                    SOPFeatureMetadata['calibrationMethod'] == CalibrationMethod.noCalibration)) != sum(
            SOPFeatureMetadata['quantificationType'] == QuantificationType.Monitored)):
        raise ValueError(
            'Check SOP file, features with quantificationType=\'Monitored\' must have a calibrationMethod=\'noCalibration\'\n quantificationType are:\n\'IS\' (expects calibrationMethod=noIS)\n\'QuantOwnLabeledAnalogue\' (would expect \'backcalculatedIS\' but could use \'noIS\' or \'otherCalibration\')\n\'QuantAltLabeledAnalogue\' (would expect \'backcalculatedIS\' but could use \'noIS\' or \'otherCalibration\')\n\'QuantOther\' (can take any CalibrationMethod)\n\'Monitored\' (which expects \'noCalibration\')')

    # check number of compounds in SOP & calibReport
    if SOPFeatureMetadata.shape[0] != calibReport.shape[0]:
        raise ValueError('SOP and Calibration Report number of compounds differ')
    featureCalibSOP = pandas.merge(left=SOPFeatureMetadata, right=calibReport, how='inner', left_on='compoundName',
                                   right_on='Compound', sort=False)
    featureCalibSOP.drop('TargetLynx ID', inplace=True, axis=1)

    # check we still have the same number of features (inner join)
    if featureCalibSOP.shape[0] != SOPFeatureMetadata.shape[0]:
        raise ValueError('SOP and Calibration Report compounds differ')

    # check compound names match in SOP and calibReport after join
    if sum(featureCalibSOP['compoundName'] != featureCalibSOP['Compound']) != 0:
        raise ValueError('SOP and Calibration Report compounds names differ: ' + str(featureCalibSOP.loc[(
                                                                                                                     featureCalibSOP[
                                                                                                                         'compoundName'] !=
                                                                                                                     featureCalibSOP[
                                                                                                                         'Compound']), [
                                                                                                             'compoundName',
                                                                                                             'Compound']].values.tolist()))
    featureCalibSOP.drop('Compound', inplace=True, axis=1)

    ## Match calibSOP & featureMetadata
    # left join to keep feature order and limit to features in XML
    finalFeatureMetadata = pandas.merge(left=featureMetadata, right=featureCalibSOP, how='left',
                                        left_on='TargetLynx Feature ID', right_on='compoundID', sort=False)

    # limit to compounds present in the SOP (no report of SOP compounds not in XML)
    if finalFeatureMetadata['compoundID'].isnull().sum() != 0:
        warnings.warn("Warning: Only " + str(finalFeatureMetadata[
                                                 'compoundID'].notnull().sum()) + " features shared across the SOP/Calibration report (" + str(
            featureCalibSOP.shape[0]) + " total) and the TargetLynx output file (" + str(
            featureMetadata.shape[0]) + " total). " + str(finalFeatureMetadata[
                                                              'compoundID'].isnull().sum()) + " features discarded from the TargetLynx output file.")
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
        warnings.warn(
            'TargetLynx feature names & SOP/Calibration Report compounds names differ; SOP names will be used: ' + str(
                finalFeatureMetadata.loc[
                    (finalFeatureMetadata['Feature Name'] != finalFeatureMetadata['compoundName']), ['Feature Name',
                                                                                                     'compoundName']].values.tolist()))
        finalFeatureMetadata['Feature Name'] = finalFeatureMetadata['compoundName']
        finalExpectedConcentration.columns = finalFeatureMetadata['Feature Name'].values.tolist()
        finalPeakResponse.columns = finalFeatureMetadata['Feature Name'].values.tolist()
        finalPeakArea.columns = finalFeatureMetadata['Feature Name'].values.tolist()
        finalPeakConcentrationDeviation.columns = finalFeatureMetadata['Feature Name'].values.tolist()
        finalPeakIntegrationFlag.columns = finalFeatureMetadata['Feature Name'].values.tolist()
        finalPeakRT.columns = finalFeatureMetadata['Feature Name'].values.tolist()
    finalFeatureMetadata.drop('compoundName', inplace=True, axis=1)

    ## Add information to the sampleMetada
    finalSampleMetadata = deepcopy(sampleMetadata)
    # Add chromatography
    finalSampleMetadata.join(pandas.DataFrame([self.Attributes['chromatography']] * finalSampleMetadata.shape[0],
                                              columns=['Chromatograpy']))
    # Add ionisation
    finalSampleMetadata.join(
        pandas.DataFrame([self.Attributes['ionisation']] * finalSampleMetadata.shape[0], columns=['Ionisation']))
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
    finalSampleMetadata['Other'] = (
                ~finalSampleMetadata['Calibrant'] & ~finalSampleMetadata['Study Sample'] & ~finalSampleMetadata[
            'Blank'] & ~finalSampleMetadata[
            'QC'])  # & ~sampleMetadata['Solvent'] & ~sampleMetadata['Recovery'] & ~sampleMetadata['Donor'] & ~sampleMetadata['Receptor']
    # Add Acquired Time
    finalSampleMetadata['Acquired Time'] = numpy.nan
    for i in range(finalSampleMetadata.shape[0]):
        try:
            finalSampleMetadata.loc[i, 'Acquired Time'] = datetime.strptime(
                str(finalSampleMetadata.loc[i, 'Acqu Date']) + " " + str(finalSampleMetadata.loc[i, 'Acqu Time']),
                '%d-%b-%y %H:%M:%S')
        except ValueError:
            pass
    finalSampleMetadata['Acquired Time'] = finalSampleMetadata['Acquired Time'].dt.to_pydatetime()
    # Add Run Order
    finalSampleMetadata['Order'] = finalSampleMetadata.sort_values(by='Acquired Time').index
    finalSampleMetadata['Run Order'] = finalSampleMetadata.sort_values(by='Order').index
    finalSampleMetadata.drop('Order', axis=1, inplace=True)
    # Initialise the Batch to 1
    finalSampleMetadata['Batch'] = [1] * finalSampleMetadata.shape[0]

    ## Apply unitCorrectionFactor
    finalFeatureMetadata['LLOQ'] = finalFeatureMetadata['LLOQ'] * finalFeatureMetadata[
        'unitCorrectionFactor']  # NaN will be kept
    finalFeatureMetadata['ULOQ'] = finalFeatureMetadata['ULOQ'] * finalFeatureMetadata['unitCorrectionFactor']
    finalIntensityData = finalIntensityData * finalFeatureMetadata['unitCorrectionFactor'].values
    finalExpectedConcentration = finalExpectedConcentration * finalFeatureMetadata['unitCorrectionFactor'].values

    ## Summary
    print('TagetLynx output, Calibration report and SOP information matched:')
    print('Targeted Method: ' + self.Attributes['methodName'])
    print(str(finalSampleMetadata.shape[0]) + ' samples (' + str(
        sum(finalSampleMetadata['Calibrant'])) + ' calibration points, ' + str(
        sum(finalSampleMetadata['Study Sample'])) + ' study samples)')
    print(str(finalFeatureMetadata.shape[0]) + ' features (' + str(sum(finalFeatureMetadata['IS'])) + ' IS, ' + str(
        sum(finalFeatureMetadata[
                'quantificationType'] == QuantificationType.QuantOwnLabeledAnalogue)) + ' quantified and validated with own labeled analogue, ' + str(
        sum(finalFeatureMetadata[
                'quantificationType'] == QuantificationType.QuantAltLabeledAnalogue)) + ' quantified and validated with alternative labeled analogue, ' + str(
        sum(finalFeatureMetadata[
                'quantificationType'] == QuantificationType.QuantOther)) + ' other quantification, ' + str(sum(
        finalFeatureMetadata[
            'quantificationType'] == QuantificationType.Monitored)) + ' monitored for relative information)')
    if len(excludedFeatureMetadata) != 0:
        print(str(excludedFeatureMetadata[0].shape[0]) + ' features excluded as missing from the SOP')
    print('All concentrations converted to final units')
    print('-----')

    return finalSampleMetadata, finalFeatureMetadata, finalIntensityData, finalExpectedConcentration, excludedSampleMetadata, excludedFeatureMetadata, excludedIntensityData, excludedExpectedConcentration, excludedFlag, finalPeakResponse, finalPeakArea, finalPeakConcentrationDeviation, finalPeakIntegrationFlag, finalPeakRT
