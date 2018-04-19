import sys
import os
import numpy
import pandas
from collections import OrderedDict
import matplotlib.pyplot as plt
import seaborn as sns
import copy
from matplotlib import gridspec
from .._toolboxPath import toolboxPath
from ..objects import TargetedDataset
from ..plotting import plotFeatureLOQ, plotLOQRunOrder, plotAccuracyPrecision, plotTIC, histogram, plotLRTIC, jointplotRSDvCorrelation, plotRSDs, plotIonMap, plotBatchAndROCorrection, plotScores, plotLoadings
from ._generateSampleReport import _generateSampleReport
from ..utilities import generateLRmask, rsd
from ..utilities._internal import _copyBackingFiles as copyBackingFiles
from ..enumerations import AssayRole, SampleType, CalibrationMethod, QuantificationType
from pyChemometrics.ChemometricsPCA import ChemometricsPCA
from ._generateBasicPCAReport import generateBasicPCAReport
from IPython.display import display
from io import StringIO
import warnings
import re
import shutil
from ..__init__ import __version__ as version


def _generateReportTargeted(tDataIn, reportType, withExclusions=False, output=None, numberPlotPerRowLOQ=3, numberPlotPerRowFeature=2, percentRange=20, pcaModel=None):
    """
    Summarise different aspects of a Targeted Dataset

    Generate reports for ``feature summary``, ``merge LOQ assessment`` or ``final report``

    * **'feature summary'** Generates feature summary report, ...
    * **'merge loq assessment'** Generates a report before :py:meth:`~TargetedData.mergeLimitsOfQuantification`, highlighting the impact of updating limits of quantification across batch. List and plot limits of quantification that are altered, number of samples impacted.
    * **'final report'** Generates a summary of the final dataset, lists sample numbers present, a selection of figures summarising dataset quality, and a final list of samples missing from acquisition.

    :param TargetedDataset tDataIn: TargetedDataset to report on
    :param str reportType: Type or report to generate, one of ``feature summary``, ``merge loq assessment`` or ``final report``
    :param bool withExclusions: If ``True``, only report on features and samples not masked by sample and feature masks
    :param output: If ``None`` plot interactively, otherwise save report to the path specified
    :type output: None or str
    :param int numberPlotPerRowLOQ: Only if ``merge loq assessment``, the number of subplots to place on each row
    :param int numberPlotPerRowFeature: Only if ``feature summary`` or ``final report``, the number of subplots to place on each row
    :param percentRange: ``None`` or Float, percentage range for acceptable accuracy [100 - percentRange, 100 + percentRange] and precision [0, percentRange]
    :type percentRange: None or float
    :param PCAmodel pcaModel: Only if ``final report``, if PCAmodel object is available PCA scores plots coloured by sample type will be added to report
    :raises ValueError: If 'tData' does not satisfy to BasicTargetedDataset definition
    :raises ValueError: If 'reportType' is not ``feature summary``, ``merge LOQ assessment`` or ``final report``
    :raises TypeError: If 'withExclusion' is not a bool
    :raises TypeError: If 'output' is not None or str
    :raises TypeError: If 'numberPlotPerRowLOQ' is not int
    :raises TypeError: If 'numberPlotPerRowFeature' is not int
    :raises TypeError: If 'percentRange' is not None or float
    """

    # Check inputs
    # Dataset minimum requirement
    tmpTData = copy.deepcopy(tDataIn)  # to not log validateObject
    validDataset = tmpTData.validateObject(verbose=False, raiseError=False, raiseWarning=False)
    if not validDataset['BasicTargetedDataset']:
        raise ValueError('Import Error: tData does not satisfy to the BasicTargetedDataset definition')

    acceptAllOptions = {'sample summary', 'feature summary', 'merge loq assessment', 'final report'}
    if not isinstance(reportType, str) & (reportType in acceptAllOptions):
        raise ValueError('reportType must be == ' + str(acceptAllOptions))

    if not isinstance(withExclusions, bool):
        raise TypeError('withExclusions must be a bool')

    if output is not None:
        if not isinstance(output, str):
            raise TypeError('output must be a string')

    if not isinstance(numberPlotPerRowLOQ, int):
        raise TypeError('numberPlotPerRowLOQ must be an int')

    if not isinstance(numberPlotPerRowFeature, int):
        raise TypeError('numberPlotPerRowFeature must be an int')

    if percentRange is not None:
        if not isinstance(percentRange, (int, float)):
            raise TypeError('percentRange must be an \'None\' or float')


    sns.set_style("whitegrid")

    # Create directory to save output
    if output:
        reportTypeCases = {'feature summary':'featureSummary', 'merge loq assessment': 'mergeLoqAssessment', 'final report': 'finalSummary'}
        #reportTypeCase = reportType.title().replace(" ", "")
        #reportTypeCase = reportTypeCase[0].lower() + reportTypeCase[1:]
        saveDir = os.path.join(output, 'graphics', 'report_' + reportTypeCases[reportType])

        # If directory exists delete directory and contents
        if os.path.exists(saveDir):
            shutil.rmtree(saveDir)

        # Create directory to save output
        os.makedirs(saveDir)

    else:
        saveAs = None

    # Apply sample/feature masks if exclusions to be applied
    tData = copy.deepcopy(tDataIn)
    if withExclusions:
        tData.applyMasks()

    # Define sample masks
    SSmask = (tData.sampleMetadata['SampleType'].values == SampleType.StudySample) & (tData.sampleMetadata['AssayRole'].values == AssayRole.Assay)
    SPmask = (tData.sampleMetadata['SampleType'].values == SampleType.StudyPool) & (tData.sampleMetadata['AssayRole'].values == AssayRole.PrecisionReference)
    ERmask = (tData.sampleMetadata['SampleType'].values == SampleType.ExternalReference) & (tData.sampleMetadata['AssayRole'].values == AssayRole.PrecisionReference)
    LRmask = (tData.sampleMetadata['SampleType'].values == SampleType.StudyPool) & (tData.sampleMetadata['AssayRole'].values == AssayRole.LinearityReference)
    [ns, nv] = tData.intensityData.shape

    # The quantificationTypes present (force the ordering)
    allQType = pandas.DataFrame({'qType': [QuantificationType.QuantOwnLabeledAnalogue,
                                           QuantificationType.QuantAltLabeledAnalogue,
                                           QuantificationType.QuantOther,
                                           QuantificationType.Monitored,
                                           QuantificationType.IS],
                                 'text': ['quantified and validated with own labeled analogue',
                                          'quantified and validated with alternative labeled analogue',
                                          'quantified using an external method',
                                          'monitored for relative information',
                                          'are internal standards'],
                                 'count': [str(sum(tData.featureMetadata['quantificationType'].values == QuantificationType.QuantOwnLabeledAnalogue)),
                                           str(sum(tData.featureMetadata['quantificationType'].values == QuantificationType.QuantAltLabeledAnalogue)),
                                           str(sum(tData.featureMetadata['quantificationType'].values == QuantificationType.QuantOther)),
                                           str(sum(tData.featureMetadata['quantificationType'].values == QuantificationType.Monitored)),
                                           str(sum(tData.featureMetadata['quantificationType'].values == QuantificationType.IS))]})
    presentQType = allQType['qType'][allQType.qType.isin(tData.featureMetadata['quantificationType'].unique().tolist())].tolist()
    textQType = allQType['text'][allQType.qType.isin(tData.featureMetadata['quantificationType'].unique().tolist())].tolist()
    countQType = allQType['count'][allQType.qType.isin(tData.featureMetadata['quantificationType'].unique().tolist())].tolist()
    nQType = len(presentQType)


    # Set up template item and save required info
    item = dict()
    item['Name'] = tData.name
    item['TargMethod'] = tData.Attributes['methodName']
    item['ReportType'] = reportType
    item['Nfeatures'] = str(nv)
    item['Nsamples'] = str(ns)
    item['SScount'] = str(sum(SSmask))
    item['SPcount'] = str(sum(SPmask))
    item['ERcount'] = str(sum(ERmask))
    item['QType'] = presentQType
    item['NQType'] = nQType
    item['QTypeIter'] = list(range(0, item['NQType']))
    item['TextQType'] = textQType
    item['CountQType'] = countQType


    # Feature summary report
    if reportType == 'feature summary':
        """
        Generates feature summary report, present the acquisition structure.
        For each QuantificationType show a table summary the compound information, plot and list accuracy and precision, violin plot of sample concentrations
        """

        # Prepare numbering for iteration over quantificationTypes
        figTabNumber = dict({'1': ['1'], '2': ['2'], '3': ['3'], '4': ['4']})
        if nQType > 1:
            allSuffix = ['-A', '-B', '-C', '-D', '-E']
            suffix = allSuffix[0:nQType]
            for i in figTabNumber.keys():
                figTabNumber[i] = [i+x for x in suffix]
        else:
            suffix = ['']
        item['figTabNumber'] = figTabNumber

        item['FeatureQuantParamTable'] = []
        item['FeatureConcentrationDistribution'] = []
        # Accuracy and precision data is present
        if _getAccuracyPrecisionTable(tData, table='accuracy').shape[0] != 0:
            withAccPrec = True
            withRSD = False
            item['FeatureAccuracyPlot'] = []
            item['FeaturePrecisionPlot'] = []
            item['FeatureAccPreTable'] = []
        else:
            # check safely RSD can be calculated
            try:
                tmpRSD = _getRSDTable(tData)
            except ValueError:
                tmpRSD = pandas.DataFrame(None)
            if tmpRSD.shape[0] != 0:
                withAccPrec = False
                withRSD = True
                item['FeatureRSDPlot'] = []
                item['FeatureRSDTable'] = []
            else:
                withAccPrec = False
                withRSD = False

        # Overall feature quantification parameters
        # add number of <LLOQ >ULOQ and percentage
        LLOQTotal = (pandas.DataFrame(tData._intensityData) < tData.featureMetadata.loc[:, 'LLOQ'].values).sum(axis=0).tolist()
        LLOQTotalPercent = ['(' + str(round(LLOQTotal[i] / tData.sampleMetadata.shape[0] * 100, 1)) + ' %)' for i in range(0, len(LLOQTotal))]
        LLOQSummary = [str(LLOQTotal[i]) + ' ' + LLOQTotalPercent[i] for i in range(0, len(LLOQTotal))]

        ULOQTotal = (pandas.DataFrame(tData._intensityData) > tData.featureMetadata.loc[:, 'ULOQ'].values).sum(axis=0).tolist()
        ULOQTotalPercent = ['(' + str(round(ULOQTotal[i] / tData.sampleMetadata.shape[0] * 100, 1)) + ' %)' for i in range(0, len(ULOQTotal))]
        ULOQSummary = [str(ULOQTotal[i]) + ' ' + ULOQTotalPercent[i] for i in range(0, len(ULOQTotal))]

        tData.featureMetadata['<LLOQ'] = LLOQSummary
        tData.featureMetadata['>ULOQ'] = ULOQSummary

        # reporting columns
        quantParamColumns = ['Feature Name', 'Unit', 'LLOQ', '<LLOQ', 'ULOQ', '>ULOQ', 'quantificationType', 'calibrationMethod']
        quantParamColumns.extend(tData.Attributes['externalID'])
        # add method specific quantification parameter columns
        if 'additionalQuantParamColumns' in tData.Attributes.keys():
            for col in tData.Attributes['additionalQuantParamColumns']:
                if (col in tData.featureMetadata.columns) and (col not in quantParamColumns):
                    quantParamColumns.append(col)

        # Feature Summary
        if output is None:
            print('\nData consists of ' + item['Nfeatures'] + ' features:')
            for i in range(0, item['NQType']):
                print('\t' + item['CountQType'][i] + ' features ' + item['TextQType'][i] + '.')
        # Summary table
        item['FeatureQuantParamTableOverall'] = tData.featureMetadata.loc[:, quantParamColumns]
        if not output:
            display(item['FeatureQuantParamTableOverall'])
            print('\n')


        # Figure 1: Acquisition structure colored by limits of quantification
        if output:
            item['AcquisitionStructure'] = os.path.join(saveDir, item['Name'] + '_AcquisitionStructure.' + tData.Attributes['figureFormat'])
            saveAs = item['AcquisitionStructure']
        else:
            print('Figure 1: Acquisition structure colored by Limits Of Quantification.')

        plotLOQRunOrder(tData,
                        addCalibration=True,
                        compareBatch=True,
                        title='',
                        savePath=saveAs,
                        figureFormat=tData.Attributes['figureFormat'],
                        dpi=tData.Attributes['dpi'],
                        figureSize=tData.Attributes['figureSize'])


        ## Iterate over Quantification Types
        for i in range(0, item['NQType']):

            # Subset only the features of interest
            tmpData = copy.deepcopy(tData)
            tmpData.updateMasks(filterSamples=False, filterFeatures=True, quantificationTypes=[item['QType'][i]])
            tmpData.applyMasks()


            # Title
            if output is None:
                print('\033[1m' + 'Features ' + item['TextQType'][i] + ' (' + item['CountQType'][i] + ')' + '\033[0m')


            # Table 1: Feature quantification parameters
            item['FeatureQuantParamTable'].append(tmpData.featureMetadata.loc[:, quantParamColumns])
            if not output:
                print('\nTable ' + item['figTabNumber']['1'][i] + ': Quantification parameters for features ' + item['TextQType'][i] + '.')
                display(item['FeatureQuantParamTable'][i])
                print('\n')


            # Figure 2: Feature Accuracy Plot
            if withAccPrec:
                if output:
                    item['FeatureAccuracyPlot'].append(os.path.join(saveDir, item['Name'] + '_FeatureAccuracy' + suffix[i] + '.' + tmpData.Attributes['figureFormat']))
                    saveAs = item['FeatureAccuracyPlot'][i]
                else:
                    print('\nFigure ' + item['figTabNumber']['2'][i] + ': Measurements accuracy for features ' + item['TextQType'][i] + '.')
                plotAccuracyPrecision(tmpData,
                                      accuracy=True,
                                      percentRange=percentRange,
                                      savePath=saveAs,
                                      figureFormat=tmpData.Attributes['figureFormat'],
                                      dpi=tmpData.Attributes['dpi'],
                                      figureSize=tmpData.Attributes['figureSize'])
            else:
                if not output:
                    print('Figure ' + item['figTabNumber']['2'][i] + ': Measurements accuracy for features ' + item['TextQType'][i] + '.')
                    print('Unable to calculate (not enough samples with expected concentrations present in dataset).\n')


            # Figure 3: Feature Precision Plot, or RSD Plot
            if withAccPrec:
                if output:
                    item['FeaturePrecisionPlot'].append(os.path.join(saveDir, item['Name'] + '_FeaturePrecision' + suffix[i] + '.' + tmpData.Attributes['figureFormat']))
                    saveAs = item['FeaturePrecisionPlot'][i]
                else:
                    print('\nFigure ' + item['figTabNumber']['3'][i] + ': Measurements precision for features ' + item['TextQType'][i] + '.')
                plotAccuracyPrecision(tmpData,
                                      accuracy=False,
                                      percentRange=percentRange,
                                      savePath=saveAs,
                                      figureFormat=tmpData.Attributes['figureFormat'],
                                      dpi=tmpData.Attributes['dpi'],
                                      figureSize=tmpData.Attributes['figureSize'])
            elif withRSD:
                if output:
                    item['FeatureRSDPlot'].append(os.path.join(saveDir, item['Name'] + '_FeatureRSD' + suffix[i] + '.' + tmpData.Attributes['figureFormat']))
                    saveAs = item['FeatureRSDPlot'][i]
                else:
                    print('\nFigure ' + item['figTabNumber']['3'][i] + ': Measurements RSD for features ' + item['TextQType'][i] + ' in all samples (by sample type).')
                plotRSDs(tmpData,
                         ratio=False,
                         logx=True,
                         color='matchReport',
                         featName=True,
                         savePath=saveAs,
                         figureFormat=tmpData.Attributes['figureFormat'],
                         dpi=tmpData.Attributes['dpi'],
                         figureSize=(tmpData.Attributes['figureSize'][0], tmpData.Attributes['figureSize'][1] * (tmpData.noFeatures / 35)))
            else:
                if not output:
                    print('Figure ' + item['figTabNumber']['3'][i] + ': Measurements precision for features ' + item['TextQType'][i] + '.')
                    print('Unable to calculate (not enough samples with expected concentrations present in dataset).\n')


            # Figure 4: Measured concentrations distribution, split by sample types.
            if output:
                item['FeatureConcentrationDistribution'].append(os.path.join(saveDir, item['Name'] + '_FeatureConcentrationDistribution' + suffix[i] + '.' + tmpData.Attributes['figureFormat']))
                saveAs = item['FeatureConcentrationDistribution'][i]
            else:
                print('\nFigure ' + item['figTabNumber']['4'][i] + ': Measured concentrations distribution, split by sample types, for features ' + item['TextQType'][i] + '.')
            plotFeatureLOQ(tmpData,
                           splitByBatch=True,
                           plotBatchLOQ=False,
                           zoomLOQ=False,
                           logY=False,
                           tightYLim=True,
                           nbPlotPerRow=numberPlotPerRowFeature,
                           savePath=saveAs,
                           figureFormat=tmpData.Attributes['figureFormat'],
                           dpi=tmpData.Attributes['dpi'],
                           figureSize=tmpData.Attributes['figureSize'])


            # Table 2: Feature Accuracy Precision Table, or RSD Table
            if withAccPrec:
                item['FeatureAccPreTable'].append(_getAccuracyPrecisionTable(tmpData, table='both'))
                if not output:
                    print('\nTable ' + item['figTabNumber']['2'][i] + ': Measurement accuracy (%) and precision (% RSD) for features ' + item['TextQType'][i] + '.')
                    display(item['FeatureAccPreTable'][i])
                    print('\n')
            elif withRSD:
                item['FeatureRSDTable'].append(_getRSDTable(tmpData))
                if not output:
                    print('\nTable ' + item['figTabNumber']['2'][i] + ': RSD for features ' + item['TextQType'][i] + '.')
                    display(item['FeatureRSDTable'][i])
                    print('\n')
            else:
                if not output:
                    print('Table ' + item['figTabNumber']['2'][i] + ': Measurement accuracy (%) and precision (% RSD) for features ' + item['TextQType'][i] + '.')
                    print('Unable to calculate (not enough samples with expected concentrations present in dataset).\n')


    # Merge LOQ assessment report
    if reportType == 'merge loq assessment':
        """
        Generates a report before :py:meth:`~TargetedData.mergeLimitsOfQuantification`, highlighting the impact of updating limits of quantification across batch. List and plot limits of quantification that are altered, number of samples impacted.
        """

        # pre-mergeLOQ dataset with post-mergeLOQ limits
        mergeLOQData = _postMergeLOQDataset(tData)
        LOQSummaryTable = _prePostMergeLOQSummaryTable(mergeLOQData)
        # Save to item
        item['MonitoredFeaturesRatio'] = str(mergeLOQData.noFeatures) + ' out of ' + str(tData.noFeatures)

        if not output:
            print('Only quantified features are assessed for the merge of limits of quantification (' + item['MonitoredFeaturesRatio'] + ').\n')

        # Table 1: Limits of quantification
        # Save to item
        item['LOQSummaryTable'] = LOQSummaryTable['LOQTable']

        if not output:
            print('Table 1: Limits of Quantification pre and post merging to the lowest common denominator.')
            display(item['LOQSummaryTable'])
            print('\n')

        # Table 2: Number of samples <LLOQ
        # Save to item
        item['LLOQSummaryTable'] = LOQSummaryTable['LLOQTable']

        if not output:
            print('Table 2: Number of sample measurements lower than the Lowest Limit of Quantification, pre and post merging to the lowest common denominator.')
            display(item['LLOQSummaryTable'])
            print('\n')

        # Table 3: Number of samples >ULOQ
        # Save to item
        item['ULOQSummaryTable'] = LOQSummaryTable['ULOQTable']

        if not output:
            print('Table 3: Number of sample measurements greater than the Upper Limit of Quantification, pre and post merging to the lowest common denominator.')
            display(item['ULOQSummaryTable'])
            print('\n')

        # Figure 1: Measured concentrations pre and post LOQ
        if output:
            item['ConcentrationPrePostMergeLOQ'] = os.path.join(saveDir, item['Name'] + '_ConcentrationPrePostMergeLOQ.' + tData.Attributes['figureFormat'])
            saveAs = item['ConcentrationPrePostMergeLOQ']
        else:
            print('Figure 1: Measured concentrations pre and post LOQ merge, split by batch and sample types.')

        plotFeatureLOQ(mergeLOQData,
                       splitByBatch=True,
                       plotBatchLOQ=True,
                       zoomLOQ=True,
                       logY=False,
                       tightYLim=False,
                       nbPlotPerRow=numberPlotPerRowLOQ,
                       savePath=saveAs,
                       figureFormat=tData.Attributes['figureFormat'],
                       dpi=tData.Attributes['dpi'],
                       figureSize=tData.Attributes['figureSize'])


    # Final summary report
    if reportType == 'final report':
        """
        Generates a summary of the final dataset, lists sample numbers present, a selection of figures summarising dataset quality, and a final list of samples missing from acquisition.
        """

        # Prepare numbering for iteration over quantificationTypes
        figTabNumber = dict({'1': ['1'], '2': ['2'], '3': ['3'], '4': ['4']})
        if nQType > 1:
            allSuffix = ['-A', '-B', '-C', '-D', '-E']
            suffix = allSuffix[0:nQType]
            for i in figTabNumber.keys():
                figTabNumber[i] = [i + x for x in suffix]
        else:
            suffix = ['']
        item['figTabNumber'] = figTabNumber

        item['FeatureQuantParamTable'] = []
        item['FeatureConcentrationDistribution'] = []
        # Accuracy and precision data is present
        if _getAccuracyPrecisionTable(tData, table='accuracy').shape[0] != 0:
            withAccPrec = True
            item['FeatureAccuracyPlot'] = []
            item['FeaturePrecisionPlot'] = []
            item['FeatureAccPreTable'] = []
        else:
            # check safely RSD can be calculated
            try:
                tmpRSD = _getRSDTable(tData)
            except ValueError:
                tmpRSD = pandas.DataFrame(None)
            if tmpRSD.shape[0] != 0:
                withAccPrec = False
                withRSD = True
                item['FeatureRSDPlot'] = []
                item['FeatureRSDTable'] = []
            else:
                withAccPrec = False
                withRSD = False


        # Feature quantification parameters
        # add number of <LLOQ >ULOQ and percentage
        LLOQTotal = (pandas.DataFrame(tData._intensityData) < tData.featureMetadata.loc[:, 'LLOQ'].values).sum(axis=0).tolist()
        LLOQTotalPercent = ['(' + str(round(LLOQTotal[i] / tData.sampleMetadata.shape[0] * 100, 1)) + ' %)' for i in range(0, len(LLOQTotal))]
        LLOQSummary = [str(LLOQTotal[i]) + ' ' + LLOQTotalPercent[i] for i in range(0,len(LLOQTotal))]

        ULOQTotal = (pandas.DataFrame(tData._intensityData) > tData.featureMetadata.loc[:, 'ULOQ'].values).sum( axis=0).tolist()
        ULOQTotalPercent = ['(' + str(round(ULOQTotal[i] / tData.sampleMetadata.shape[0] * 100, 1)) + ' %)' for i in range(0, len(ULOQTotal))]
        ULOQSummary = [str(ULOQTotal[i]) + ' ' + ULOQTotalPercent[i] for i in range(0,len(ULOQTotal))]

        tData.featureMetadata['<LLOQ'] = LLOQSummary
        tData.featureMetadata['>ULOQ'] = ULOQSummary

        # reporting columns
        quantParamColumns = ['Feature Name', 'Unit', 'LLOQ','<LLOQ', 'ULOQ', '>ULOQ', 'quantificationType', 'calibrationMethod']
        quantParamColumns.extend(tData.Attributes['externalID'])
        # add method specific quantification parameter columns
        if 'additionalQuantParamColumns' in tData.Attributes.keys():
            for col in tData.Attributes['additionalQuantParamColumns']:
                if (col in tData.featureMetadata.columns) and (col not in quantParamColumns):
                    quantParamColumns.append(col)

        ## Table 1: Sample summary

        # Generate sample summary
        sampleSummary = _generateSampleReport(tData, withExclusions=True, output=None, returnOutput=True)

        # Extract summary table for samples acquired
        sampleSummaryTable = copy.deepcopy(sampleSummary['Acquired'])

        # Drop unwanted columns
        sampleSummaryTable.drop(['Marked for Exclusion'], axis=1, inplace=True)
        if 'LIMS marked as missing' in sampleSummaryTable.columns:
            sampleSummaryTable.drop(['LIMS marked as missing', 'Missing from LIMS'], axis=1, inplace=True)
        if 'Missing Subject Information' in sampleSummaryTable.columns:
            sampleSummaryTable.drop(['Missing Subject Information'], axis=1, inplace=True)

        # Rename 'already excluded'
        sampleSummaryTable.rename(columns={'Already Excluded': 'Excluded'}, inplace=True)

        # Add 'unavailable' column
        if 'NotAcquired' in sampleSummary:
            sampleSummaryTable = sampleSummaryTable.join(pandas.DataFrame(data=sampleSummary['NotAcquired']['Marked as Sample'] - sampleSummary['NotAcquired']['Already Excluded'], columns=['Unavailable']), how='left', sort=False)
        else:
            sampleSummaryTable['Unavailable'] = 0

        # Update 'All', 'Unavailable' to only reflect sample types present in data
        sampleSummaryTable.loc['All', 'Unavailable'] = sum(sampleSummaryTable['Unavailable'][1:])

        # Save to item
        item['SampleSummaryTable'] = sampleSummaryTable

        # Save details of study samples missing from dataset
        if sampleSummaryTable['Unavailable']['Study Sample'] != 0:
            item['SamplesMissingInfo'] = sampleSummary['NotAcquired Details'].loc[sampleSummary['NotAcquired Details']['Sampling ID'].isnull() == False, :]
            item['SamplesMissingInfo'] = item['SamplesMissingInfo'].drop(['LIMS Marked Missing'], axis=1)
            item['SamplesMissingNo'] = str(sampleSummaryTable['Unavailable']['Study Sample'])

        # Save details of study samples excluded from dataset
        if hasattr(sampleSummaryTable, 'Excluded'):
            if sampleSummaryTable['Excluded']['Study Sample'] != 0:
                item['SamplesExcludedInfo'] = sampleSummary['Excluded Details'].loc[(sampleSummary['Excluded Details']['SampleType'] == SampleType.StudySample) & (sampleSummary['Excluded Details']['AssayRole'] == AssayRole.Assay),:]
                item['SamplesExcludedInfo'] = item['SamplesExcludedInfo'].drop(['Sample Base Name', 'SampleType', 'AssayRole'], axis=1)
                item['SamplesExcludedNo'] = str(sampleSummaryTable['Excluded']['Study Sample'])

        if not output:
            print('Final Dataset for: ' + item['Name'])
            print('\n\t' + 'Method: ' + item['TargMethod'] + '\n\t' + item['Nsamples'] + ' samples\n\t' + item['Nfeatures'] + ' features')
            ## Feature Summary
            for i in range(0, item['NQType']):
                print('\t\t' + item['CountQType'][i] + ' features ' + item['TextQType'][i] + '.')

            print('\nSample Summary')
            display(item['SampleSummaryTable'])
            print('\n')


        ## Overall feature quantification parameters
        item['FeatureQuantParamTableOverall'] = tData.featureMetadata.loc[:, quantParamColumns]
        if not output:
            print('Feature Summary')
            display(item['FeatureQuantParamTableOverall'])
            print('\n')


        ## Figure 1: Acquisition structure colored by limits of quantification
        nBatchCollect = len((numpy.unique(
            tData.sampleMetadata['Batch'].values[~numpy.isnan(tData.sampleMetadata['Batch'].values)])).astype(int))
        if nBatchCollect == 1:
            item['nBatchesCollect'] = '1 batch'
        else:
            item['nBatchesCollect'] = str(nBatchCollect) + ' batches'
        start = pandas.to_datetime(str(tData.sampleMetadata['Acquired Time'].loc[tData.sampleMetadata['Run Order'] == min(tData.sampleMetadata['Run Order'][tData.sampleMask])].values[0]))
        end = pandas.to_datetime(str(tData.sampleMetadata['Acquired Time'].loc[tData.sampleMetadata['Run Order'] == max(tData.sampleMetadata['Run Order'][tData.sampleMask])].values[0]))
        item['start'] = start.strftime('%d/%m/%y')
        item['end'] = end.strftime('%d/%m/%y')


        if output:
            item['AcquisitionStructure'] = os.path.join(saveDir, item['Name'] + '_AcquisitionStructure.' + tData.Attributes['figureFormat'])
            saveAs = item['AcquisitionStructure']
        else:
            print('Acquisition Structure')
            print('\n\tSamples acquired in ' + item['nBatchesCollect'] + ' between ' + item['start'] + ' and ' + item['end'])
            print('\nFigure 1: Acquisition structure colored by Limits Of Quantification.')

        plotLOQRunOrder(tData,
                        addCalibration=True,
                        compareBatch=True,
                        title='',
                        savePath=saveAs,
                        figureFormat=tData.Attributes['figureFormat'],
                        dpi=tData.Attributes['dpi'],
                        figureSize=tData.Attributes['figureSize'])

        ## Iterate over Quantification Types (Quant parameters table, Acc Prec plots, Acc Prec tables)
        for i in range(0, item['NQType']):

            # Subset only the features of interest
            tmpData = copy.deepcopy(tData)
            tmpData.updateMasks(filterSamples=False, filterFeatures=True, quantificationTypes=[item['QType'][i]])
            tmpData.applyMasks()

            # Title
            if output is None:
                print('\033[1m' + 'Features ' + item['TextQType'][i] + ' (' + item['CountQType'][i] + ')' + '\033[0m')

            ## Table 2: Feature quantification parameters
            item['FeatureQuantParamTable'].append(tmpData.featureMetadata.loc[:, quantParamColumns])
            if not output:
                print('\nTable ' + item['figTabNumber']['2'][i] + ': Quantification parameters for features ' + item['TextQType'][i] + '.')
                display(item['FeatureQuantParamTable'][i])
                print('\n')

            ## Figure 2: Feature Accuracy Plot
            if withAccPrec:
                if output:
                    item['FeatureAccuracyPlot'].append(os.path.join(saveDir, item['Name'] + '_FeatureAccuracy' + suffix[i] + '.' + tmpData.Attributes['figureFormat']))
                    saveAs = item['FeatureAccuracyPlot'][i]
                else:
                    print('\nFigure ' + item['figTabNumber']['2'][i] + ': Measurements accuracy for features ' + item['TextQType'][i] + '.')
                plotAccuracyPrecision(tmpData,
                                      accuracy=True,
                                      percentRange=percentRange,
                                      savePath=saveAs,
                                      figureFormat=tmpData.Attributes['figureFormat'],
                                      dpi=tmpData.Attributes['dpi'],
                                      figureSize=tmpData.Attributes['figureSize'])
            else:
                if not output:
                    print('Figure ' + item['figTabNumber']['2'][i] + ': Measurements accuracy for features ' + item['TextQType'][i] + '.')
                    print('Unable to calculate (not enough samples with expected concentrations present in dataset).\n')

            ## Figure 3: Feature Precision Plot, or RSD Plot
            if withAccPrec:
                if output:
                    item['FeaturePrecisionPlot'].append(os.path.join(saveDir, item['Name'] + '_FeaturePrecision' + suffix[i] + '.' + tmpData.Attributes['figureFormat']))
                    saveAs = item['FeaturePrecisionPlot'][i]
                else:
                    print('\nFigure ' + item['figTabNumber']['3'][i] + ': Measurements precision for features ' + item['TextQType'][i] + '.')
                plotAccuracyPrecision(tmpData,
                                      accuracy=False,
                                      percentRange=percentRange,
                                      savePath=saveAs,
                                      figureFormat=tmpData.Attributes['figureFormat'],
                                      dpi=tmpData.Attributes['dpi'],
                                      figureSize=tmpData.Attributes['figureSize'])
            elif withRSD:
                if output:
                    item['FeatureRSDPlot'].append(os.path.join(saveDir, item['Name'] + '_FeatureRSD' + suffix[i] + '.' + tmpData.Attributes['figureFormat']))
                    saveAs = item['FeatureRSDPlot'][i]
                else:
                    print('\nFigure ' + item['figTabNumber']['3'][i] + ': Measurements RSD for features ' + item['TextQType'][i] + ' in all samples (by sample type).')
                plotRSDs(tmpData,
                         ratio=False,
                         logx=True,
                         color='matchReport',
                         featName=True,
                         savePath=saveAs,
                         figureFormat=tmpData.Attributes['figureFormat'],
                         dpi=tmpData.Attributes['dpi'],
                         figureSize=(tmpData.Attributes['figureSize'][0], tmpData.Attributes['figureSize'][1] * (tmpData.noFeatures / 35)))
            else:
                if not output:
                    print('Figure ' + item['figTabNumber']['3'][i] + ': Measurements precision for features ' + item['TextQType'][i] + '.')
                    print('Unable to calculate (not enough samples with expected concentrations present in dataset).\n')

            ## Figure 4: Measured concentrations distribution, split by sample types.
            if output:
                item['FeatureConcentrationDistribution'].append(os.path.join(saveDir, item['Name'] + '_FeatureConcentrationDistribution' + suffix[i] + '.' + tmpData.Attributes['figureFormat']))
                saveAs = item['FeatureConcentrationDistribution'][i]
            else:
                print('\nFigure ' + item['figTabNumber']['4'][i] + ': Measured concentrations distribution, split by sample types, for features ' + item['TextQType'][i] + '.')
            plotFeatureLOQ(tmpData,
                           splitByBatch=True,
                           plotBatchLOQ=False,
                           zoomLOQ=False,
                           logY=False,
                           tightYLim=True,
                           nbPlotPerRow=numberPlotPerRowFeature,
                           savePath=saveAs,
                           figureFormat=tmpData.Attributes['figureFormat'],
                           dpi=tmpData.Attributes['dpi'],
                           figureSize=tmpData.Attributes['figureSize'])

            ## Table 3: Feature Accuracy Precision Table, or RSD Table
            if withAccPrec:
                item['FeatureAccPreTable'].append(_getAccuracyPrecisionTable(tmpData, table='both'))
                if not output:
                    print('\nTable ' + item['figTabNumber']['3'][i] + ': Measurement accuracy (%) and precision (% RSD) for features ' + item['TextQType'][i] + '.')
                    display(item['FeatureAccPreTable'][i])
                    print('\n')
            elif withRSD:
                item['FeatureRSDTable'].append(_getRSDTable(tmpData))
                if not output:
                    print('\nTable ' + item['figTabNumber']['2'][i] + ': RSD for features ' + item['TextQType'][i] + '.')
                    display(item['FeatureRSDTable'][i])
                    print('\n')
            else:
                if not output:
                    print('Table ' + item['figTabNumber']['3'][i] + ': Measurement accuracy (%) and precision (% RSD) for features ' + item['TextQType'][i] + '.')
                    print('Unable to calculate (not enough samples with expected concentrations present in dataset).\n')
        # End iterate over Quantification Types


        ## Figure 5 and 6: (if available) PCA scores and loadings plots by sample type
        if pcaModel is not None:

            tData.sampleMetadata.loc[~SSmask & ~SPmask & ~ERmask, 'Plot Sample Type'] = 'Sample'
            tData.sampleMetadata.loc[SSmask, 'Plot Sample Type'] = 'Study Sample'
            tData.sampleMetadata.loc[SPmask, 'Plot Sample Type'] = 'Study Pool'
            tData.sampleMetadata.loc[ERmask, 'Plot Sample Type'] = 'External Reference'

            if pcaModel:
                if output:
                    pcaPath = output
                else:
                    pcaPath = None
                pcaModel = generateBasicPCAReport(pcaModel, tData, figureCounter=6, output=pcaPath, fileNamePrefix='')

            #if output:
            #    item['pcaPlots'] = generateBasicPCAReport(pcaModel, tData, figureCounter=5, output=saveAs)
            #else:
            #    item['pcaPlots'] = generateBasicPCAReport(pcaModel, tData, figureCounter=5, output=output)


        ## Add final tables of excluded/missing study samples
        if not output:

            if (('SamplesMissingInfo' in item) | ('SamplesExcludedInfo' in item)):

                print('Samples Missing from Acquisition\n')

                if 'SamplesMissingInfo' in item:
                    print('Samples unavailable for acquisition (' + item['SamplesMissingNo'] + ')')
                    display(item['SamplesMissingInfo'])
                    print('\n')

                if 'SamplesExcludedInfo' in item:
                    print('Samples excluded on analytical criteria (' + item['SamplesExcludedNo'] + ')')
                    display(item['SamplesExcludedInfo'])
                    print('\n')


    # Generate HTML report
    if output:

        # Make paths for graphics local not absolute for use in the HTML.
        for key in item:
            if isinstance(item[key], list):
                for i in range(0, len(item[key])):
                    if os.path.join(output, 'graphics') in str(item[key][i]):
                        item[key][i] = re.sub('.*graphics', 'graphics', item[key][i])
            elif os.path.join(output, 'graphics') in str(item[key]):
                item[key] = re.sub('.*graphics', 'graphics', item[key])

        # Generate report
        from jinja2 import Environment, FileSystemLoader

        env = Environment(loader=FileSystemLoader(os.path.join(toolboxPath(), 'Templates')))
        template = env.get_template('generateReportTargeted.html')
        filename = os.path.join(output, tData.name + '_report_' + reportTypeCases[reportType] + '.html')

        f = open(filename,'w')
        f.write(template.render(item=item, version=version, graphicsPath='/report_' + reportTypeCases[reportType], pcaPlots=pcaModel))
        f.close()

        copyBackingFiles(toolboxPath(), saveDir)


def _postMergeLOQDataset(tData):
    """
    Return a dataset with merged LOQs, but without updated :py:attr:`_intensityData` values (post-merge LOQ with pre-merge intensity)

    :param TargetedDataset tData: :py:class:`TargetedDataset` concatenated using :py:meth:`__add__` but without merged LOQ.
    """
    # Prepare a targetedDataset with mergeLOQ
    mergedLOQData = copy.deepcopy(tData)
    # silence text output
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    mergedLOQData.mergeLimitsOfQuantification(onlyLLOQ=False, keepBatchLOQ=True)
    sys.stdout = old_stdout

    # copy merged LOQ in pre-merge dataset
    outDataset = copy.deepcopy(tData)
    outDataset.featureMetadata['LLOQ'] = mergedLOQData.featureMetadata['LLOQ']
    outDataset.featureMetadata['ULOQ'] = mergedLOQData.featureMetadata['ULOQ']

    return outDataset


def _prePostMergeLOQSummaryTable(tData):
    """
    Summarise changes in LOQ, number of samples <LLOQ and >ULOQ, before and after :py:meth:`mergeLimitsOfQuantification`

    :param TargetedDataset tData: :py:class:`TargetedDataset` with pre-merge intensityData and post-merge LOQ as generated by py:meth:`_postMergeLOQDataset`
    :return: A dictionary of 'LOQTable', 'LLOQTable' and 'ULOQTable' summarising changes in LOQ and number of samples <LLOQ and >ULOQ.
    """

    # Identify batch and batch LOQ columns
    col_LLOQ = sorted(tData.featureMetadata.columns[tData.featureMetadata.columns.to_series().str.contains('LLOQ_batch')].tolist())
    col_ULOQ = sorted(tData.featureMetadata.columns[tData.featureMetadata.columns.to_series().str.contains('ULOQ_batch')].tolist())
    batches = sorted((numpy.unique(tData.sampleMetadata.loc[:, 'Batch'].values[~numpy.isnan(tData.sampleMetadata.loc[:, 'Batch'].values)])).astype(int))
    number_of_batch = len(batches)

    ## LOQTable, summarisethe change in each LOQ
    LOQTable = pandas.DataFrame(None, index=tData.featureMetadata['Feature Name'].values)

    # All LLOQ columns
    LOQTable['LLOQ'] = tData.featureMetadata['LLOQ'].values
    for col in col_LLOQ:
        LOQTable[col] = tData.featureMetadata[col].values
    LOQTable['LLOQ Diff.'] = ''
    LOQTable.loc[LOQTable[col_LLOQ].apply(lambda x: min(x) != max(x), 1), 'LLOQ Diff.'] = 'X'

    # All ULOQ columns
    LOQTable['ULOQ'] = tData.featureMetadata['ULOQ'].values
    for col in col_ULOQ:
        LOQTable[col] = tData.featureMetadata[col].values
    LOQTable['ULOQ Diff.'] = ''
    LOQTable.loc[LOQTable[col_ULOQ].apply(lambda x: min(x) != max(x), 1), 'ULOQ Diff.'] = 'X'


    ## LLOQTable, summarise the number of <LLOQ samples
    LLOQTable = pandas.DataFrame(None, index=tData.featureMetadata['Feature Name'].values)

    # Batch LLOQ
    for i in range(number_of_batch):
        # Limit to the proper batch and quantified features
        batchMask = (tData.sampleMetadata['Batch'].values == batches[i])
        old_LLOQ_values = tData.featureMetadata.loc[:, col_LLOQ[i]].values

        # Count <LLOQ and append
        nbLLOQ = (pandas.DataFrame(tData._intensityData).loc[batchMask, :] < old_LLOQ_values).sum(axis=0).values
        LLOQTable['Batch ' + str(batches[i])] = nbLLOQ

    # Sum previous
    LLOQTable['Prev. Total'] = LLOQTable.sum(axis=1)
    LLOQTable['Prev. Total (%)'] = round(LLOQTable['Prev. Total'] / tData.sampleMetadata.shape[0] * 100, 1)

    # Merged LLOQ
    new_LLOQ_values = tData.featureMetadata.loc[:, 'LLOQ'].values
    LLOQTable['New Total'] = (pandas.DataFrame(tData._intensityData) < new_LLOQ_values).sum(axis=0).values
    LLOQTable['New Total (%)'] = round(LLOQTable['New Total'] / tData.sampleMetadata.shape[0] * 100, 1)

    # Highlight difference
    LLOQTable['Diff.'] = ''
    LLOQTable.loc[(LLOQTable['Prev. Total'] != LLOQTable['New Total']), 'Diff.'] = 'X'

    # Finish LLOQ table
    LLOQTable.columns.name = '<LLOQ'


    ## ULOQTable, summarise the number of >ULOQ samples
    ULOQTable = pandas.DataFrame(None, index=tData.featureMetadata['Feature Name'].values)

    # Batch ULOQ
    for i in range(number_of_batch):
        # Limit to the proper batch and quantified features
        batchMask = (tData.sampleMetadata['Batch'].values == batches[i])
        old_ULOQ_values = tData.featureMetadata.loc[:, col_ULOQ[i]].values

        # Count <ULOQ and append
        nbULOQ = (pandas.DataFrame(tData._intensityData).loc[batchMask, :] > old_ULOQ_values).sum(axis=0).values
        ULOQTable['Batch ' + str(batches[i])] = nbULOQ

    # Sum previous
    ULOQTable['Prev. Total'] = ULOQTable.sum(axis=1)
    ULOQTable['Prev. Total (%)'] = round(ULOQTable['Prev. Total'] / tData.sampleMetadata.shape[0] * 100, 1)

    # Merged ULOQ
    new_ULOQ_values = tData.featureMetadata.loc[:, 'ULOQ'].values
    ULOQTable['New Total'] = (pandas.DataFrame(tData._intensityData) > new_ULOQ_values).sum(axis=0).values
    ULOQTable['New Total (%)'] = round(ULOQTable['New Total'] / tData.sampleMetadata.shape[0] * 100, 1)

    # Highlight difference
    ULOQTable['Diff.'] = ''
    ULOQTable.loc[(ULOQTable['Prev. Total'] != ULOQTable['New Total']), 'Diff.'] = 'X'

    # Finish ULOQ table
    ULOQTable.columns.name = '>ULOQ'

    return {'LOQTable': LOQTable, 'LLOQTable': LLOQTable, 'ULOQTable': ULOQTable}


def _getAccuracyPrecisionTable(tData, table='both'):
    """
    Return a table of Accuracy or Precision.

    :param TargetedDataset tData: TargetedDataset object to plot
    :param str table: If ``accuracy` returns the Accuracy of each measurements, if ``precision`` returns the Precision of measurements, if ``both`` returns the combined table.
    """

    def generateTable(statistic):

        ## Prepare data (loop over features, append to an output df, remove all rows with NA, define a y-axis)
        # limit to sample types with existing data
        sType = []
        for skey in statistic.keys():
            if statistic[skey].shape[0] != 0:
                sType.append(skey)

        # define output df
        Conc = statistic['All Samples'].index.tolist()  # [str(i) for i in statistic['All Samples'].index]
        nConc = len(Conc)
        nFeat = tData.noFeatures
        featList = tData.featureMetadata['Feature Name'].tolist()
        cols = ['Feature', 'Sample Type']
        cols.extend(Conc)
        statTable = pandas.DataFrame(columns=cols)

        # iterate over features
        for i in range(0, nFeat):
            featID = featList[i]
            tmpStatTable = pandas.DataFrame(columns=cols, index=sType)
            # iterate over each possible concentration
            for currConc in Conc:
                # iterate over each SampleType
                for skey in sType:
                    if currConc in statistic[skey].index.tolist():
                        tmpStatTable.loc[skey, currConc] = statistic[skey].loc[currConc, featID]
            # remove empty rows and finish off table
            tmpStatTable = tmpStatTable.dropna(how='all')
            tmpStatTable['Feature'] = featID
            tmpStatTable['Sample Type'] = sType
            statTable = statTable.append(tmpStatTable, ignore_index=True)

        statTable['Sample Type'] = [str(x) for x in statTable['Sample Type'].tolist()]
        statTable.fillna(value='', inplace=True)

        return statTable


    if table not in ['both', 'accuracy', 'precision']:
        raise ValueError('Parameter table must be \'both\', \'accuracy\' or \'precision\'')

    # Init
    accPrec = tData.accuracyPrecision()

    # Restructure table
    if table == 'accuracy':
        outTable = generateTable(accPrec['Accuracy'])
        outTable = outTable.set_index(['Feature', 'Sample Type'])
        outTable.columns.name = 'Accuracy'

    elif table == 'precision':
        outTable = generateTable(accPrec['Precision'])
        outTable = outTable.set_index(['Feature', 'Sample Type'])
        outTable.columns.name = 'Precision'

    elif table == 'both':
        # Load both table
        acc = generateTable(accPrec['Accuracy'])
        pre = generateTable(accPrec['Precision'])

        ## put both tables together, generate the multiIndex columns first, then multiIndex rows
        # Store row info (feature, sample type) and remove
        refRows = acc.loc[:, ['Feature', 'Sample Type']]
        tmpAcc  = acc.drop(['Feature', 'Sample Type'], axis=1)

        # Store column info, reinitialise column names
        refCols = tmpAcc.columns.tolist()
        tmpAcc.columns = [i for i in range(tmpAcc.shape[1])]

        # Filter precision too
        tmpPre = pre.drop(['Feature', 'Sample Type'], axis=1)
        tmpPre.columns = [i for i in range(tmpPre.shape[1])]

        # Join both, Accuracy on top, Precision at the bottom
        tmpOut = pandas.concat([tmpAcc.transpose(), tmpPre.transpose()])
        tmpOut.reset_index(drop=True, inplace=True)

        # Reorder the rows by concentration
        newRefCols    = pandas.DataFrame({'Concentration': refCols * 2, 'Measure': ['Acc.'] * len(refCols) + ['Prec.'] * len(refCols)})
        rowReordering = newRefCols.sort_values(by='Concentration').index.tolist()
        tmpOut     = tmpOut.loc[rowReordering, :]
        newRefCols = newRefCols.loc[rowReordering, :]

        # Add the future column multiIndex
        tmpOut   = newRefCols.join(tmpOut)
        tmpOut   = tmpOut.set_index(['Concentration', 'Measure'])
        outTable = tmpOut.transpose()

        # Add the row multiIndex
        tuplesRow = list(map(tuple, refRows.values))
        rowIdx    = pandas.MultiIndex.from_tuples(tuplesRow, names=['Feature', 'Sample Type'])
        outTable.index = rowIdx

    return outTable


def _getRSDTable(tData):
    """
    Return a table of RSD

    :param TargetedDataset tData: TargetedDataset object to plot
    """

    ## Calculate RSD for every SampleType with enough PrecisionReference samples.
    rsdVal = dict()

    precRefMask = tData.sampleMetadata.loc[:, 'AssayRole'].values == AssayRole.PrecisionReference
    precRefMask = numpy.logical_and(precRefMask, tData.sampleMask)
    sTypes = list(set(tData.sampleMetadata.loc[precRefMask, 'SampleType'].values))

    rsdVal[SampleType.StudyPool] = tData.rsdSP
    ssMask = (tData.sampleMetadata['SampleType'].values == SampleType.StudySample) & tData.sampleMask
    rsdVal[SampleType.StudySample] = rsd(tData.intensityData[ssMask, :])

    # Only keep features with finite values for SP and SS
    finiteMask = (rsdVal[SampleType.StudyPool] < numpy.finfo(numpy.float64).max) & tData.featureMask
    finiteMask = finiteMask & (rsdVal[SampleType.StudySample] < numpy.finfo(numpy.float64).max)

    for sType in sTypes:
        if not sTypes == SampleType.StudyPool:
            sTypeMask = tData.sampleMetadata.loc[:, 'SampleType'].values == sType
            # precRefMask limits to Precision Reference and tData.sampleMask
            sTypeMask = numpy.logical_and(sTypeMask, precRefMask)

            # minimum 3 points needed
            if sum(sTypeMask) >= 3:
                rsdVal[sType] = rsd(tData.intensityData[sTypeMask, :])
                finiteMask = finiteMask & (rsdVal[sType] < numpy.finfo(numpy.float64).max)

    # apply ginite mask
    for sType in rsdVal.keys():
        rsdVal[sType] = rsdVal[sType][finiteMask]

    rsdTable = pandas.DataFrame(rsdVal, index=tData.featureMetadata['Feature Name'].tolist())
    rsdTable.columns.name = 'RSD'

    return(rsdTable)
