import sys
import os
import numpy
import pandas
from collections import OrderedDict
import matplotlib.pyplot as plt
import seaborn as sns
import copy
from IPython.display import display
import warnings
import re
import shutil
from matplotlib import gridspec
from .._toolboxPath import toolboxPath
from ..objects import MSDataset
from pyChemometrics.ChemometricsPCA import ChemometricsPCA
from ..plotting import plotTIC, histogram, plotLRTIC, jointplotRSDvCorrelation, plotRSDs, plotIonMap, plotBatchAndROCorrection, plotScores, plotLoadings, plotTargetedFeatureDistribution
from ._generateSampleReport import _generateSampleReport
from ..utilities import generateLRmask, rsd
from ..utilities._internal import _vcorrcoef
from ..utilities._internal import _copyBackingFiles as copyBackingFiles
from ..enumerations import AssayRole, SampleType
import operator


from ..__init__ import __version__ as version


def _finalReportPeakPantheR(dataset, destinationPath=None):
    """
    Summarise different aspects of an MS dataset

    Generate reports for ``feature summary``, ``correlation to dilution``, ``batch correction assessment``, ``batch correction summary``, ``feature selection``, ``final report``, ``final report abridged``, or ``final report targeted abridged``

    * **'feature summary'** Generates feature summary report, plots figures including those for feature abundance, sample TIC and acquisition structure, correlation to dilution, RSD and an ion map.
    * **'correlation to dilution'** Generates a more detailed report on correlation to dilution, broken down by batch subset with TIC, detector voltage, a summary, and heatmap indicating potential saturation or other issues.
    * **'batch correction assessment'** Generates a report before batch correction showing TIC overall and intensity and batch correction fit for a subset of features, to aid specification of batch start and end points.
    * **'batch correction summary'** Generates a report post batch correction with pertinant figures (TIC, RSD etc.) before and after.
    * **'feature selection'** Generates a summary of the number of features passing feature selection (with current settings as definite in the SOP), and a heatmap showing how this number would be affected by changes to RSD and correlation to dilution thresholds.
    * **'final report'** Generates a summary of the final dataset, lists sample numbers present, a selection of figures summarising dataset quality, and a final list of samples missing from acquisition.
    * **'final report abridged'** Generates an abridged summary of the final dataset, lists sample numbers present, a selection of figures summarising dataset quality, and a final list of samples missing from acquisition.
    * **'final report targeted abridged'** Generates an abridged summary of the final targeted (peakPantheR) dataset, lists sample numbers present, a selection of figures summarising dataset quality, feature distributions, and a final list of samples missing from acquisition.

    :param MSDataset msDataTrue: MSDataset to report on
    :param str reportType: Type of report to generate, one of ``feature summary``, ``correlation to dilution``, ``batch correction``, ``feature selection``, ``final report``, ``final report abridged``, or ``final report targeted abridged``
    :param bool withExclusions: If ``True``, only report on features and samples not masked by the sample and feature masks
    :param None or bool withArtifactualFiltering: If ``None`` use the value from ``Attributes['artifactualFilter']``. If ``True`` apply artifactual filtering to the ``feature selection`` report and ``final report``
    :param destinationPath: If ``None`` plot interactively, otherwise save report to the path specified
    :type destinationPath: None or str
    :param MSDataset msDataCorrected: Only if ``batch correction``, if msDataCorrected included will generate report post correction
    :param PCAmodel pcaModel: Only if ``final report``, if PCAmodel object is available PCA scores plots coloured by sample type will be added to report
    """

    """
    Generates a summary of the final dataset, lists sample numbers present, a selection of figures summarising dataset quality, and a final list of samples missing from acquisition.
    """

	# Create save directory if required
    if destinationPath is not None:
        if not os.path.exists(destinationPath):
            os.makedirs(destinationPath)
        if not os.path.exists(os.path.join(destinationPath, 'graphics')):
            os.makedirs(os.path.join(destinationPath, 'graphics'))
        graphicsPath = os.path.join(destinationPath, 'graphics', 'report_finalSummary')
        if not os.path.exists(graphicsPath):
            os.makedirs(graphicsPath)
    else:
        graphicsPath = None
        saveAs = None

    # Ensure we have 'Passing Selection' column in dataset object
    if not hasattr(dataset.featureMetadata, 'Passing Selection'):
        dataset.saveFeatureMask()

	# Use cpdName (targeted) to label RSD plot if available
    if (hasattr(dataset.featureMetadata, 'cpdName')):
        featureName = 'cpdName'
        featName=True
        figureSize=(dataset.Attributes['figureSize'][0], dataset.Attributes['figureSize'][1] * (dataset.noFeatures / 35))
    else:
        featureName = 'Feature Name'
        featName=False
        figureSize=dataset.Attributes['figureSize']

	# Define sample masks
    SSmask = (dataset.sampleMetadata['SampleType'].values == SampleType.StudySample) & \
             (dataset.sampleMetadata['AssayRole'].values == AssayRole.Assay)
    SPmask = (dataset.sampleMetadata['SampleType'].values == SampleType.StudyPool) & \
             (dataset.sampleMetadata['AssayRole'].values == AssayRole.PrecisionReference)
    ERmask = (dataset.sampleMetadata['SampleType'].values == SampleType.ExternalReference) & \
             (dataset.sampleMetadata['AssayRole'].values == AssayRole.PrecisionReference)
    LRmask = (dataset.sampleMetadata['SampleType'].values == SampleType.StudyPool) & \
             (dataset.sampleMetadata['AssayRole'].values == AssayRole.LinearityReference)

    # Set up template item and save required info
    item = dict()
    item['Name'] = dataset.name
    item['ReportType'] = 'feature summary' # TODO check what this means!
    item['Nsamples'] = dataset.intensityData.shape[0]
    item['Nfeatures'] = dataset.intensityData.shape[1]
    item['NfeaturesPassing'] = sum(dataset.featureMetadata['Passing Selection'])
    item['NfeaturesFailing'] = item['Nfeatures'] - item['NfeaturesPassing']
    if item['NfeaturesFailing'] != 0:
        hLine = [item['NfeaturesFailing']]
    else:
        hLine = None
    item['SScount'] = str(sum(SSmask))
    item['SPcount'] = str(sum(SPmask))
    item['ERcount'] = str(sum(ERmask))
    item['LRcount'] = str(sum(LRmask))
    item['corrMethod'] = dataset.Attributes['corrMethod']
    figNo = 1


    # Final dataset summary
    if not destinationPath:
        print('Final Dataset\n')
        print(str(item['Nsamples']) + ' samples')
        print(str(item['Nfeatures']) + ' features')
        if item['NfeaturesFailing'] != 0:
            print('\t' + str(item['NfeaturesPassing']) + ' detected and passing feature selection')
            print('\t' + str(item['NfeaturesFailing']) + ' not detected or not present in sufficient concentration to be measured accurately')

    # Table 1: Sample summary

    # Generate sample summary

    sampleSummary = _generateSampleReport(dataset, destinationPath=None, returnOutput=True)

    # Tidy table for final report format
    sampleSummary['Acquired'].drop('Marked for Exclusion', inplace=True, axis=1)

    if hasattr(sampleSummary['Acquired'], 'Already Excluded'):
        sampleSummary['Acquired'].rename(columns={'Already Excluded': 'Excluded'}, inplace=True)

    sampleSummary['isFinalReport'] = True
    if 'StudySamples Exclusion Details' in sampleSummary:
        sampleSummary['studySamplesExcluded'] = True
    else:
        sampleSummary['studySamplesExcluded'] = False
    item['sampleSummary'] = sampleSummary

    if not destinationPath:
        print('\n\nSample Summary')
        print('\nTable 1: Sample summary table.')
        display(sampleSummary['Acquired'])
        print('\n*Details of any missing/excluded study samples given at the end of the report\n')


    # Table 2: Feature Selection parameters
    FeatureSelectionTable = pandas.DataFrame(
        data=['yes', dataset.Attributes['corrMethod'], dataset.Attributes['corrThreshold']],
        index=['Correlation to Dilution', 'Correlation to Dilution: Method', 'Correlation to Dilution: Threshold'],
        columns=['Value Applied'])

    if sum(dataset.corrExclusions) != dataset.noSamples:
        temp = ', '.join(dataset.sampleMetadata.loc[dataset.corrExclusions == False, 'Sample File Name'].values)
        FeatureSelectionTable = FeatureSelectionTable.append(
            pandas.DataFrame(data=temp, index=['Correlation to Dilution: Sample Exclusions'], columns=['Value Applied']))
    else:
        FeatureSelectionTable = FeatureSelectionTable.append(
            pandas.DataFrame(data=['none'], index=['Correlation To Dilution: Sample Exclusions'], columns=['Value Applied']))
    FeatureSelectionTable = FeatureSelectionTable.append(
        pandas.DataFrame(data=['yes', dataset.Attributes['rsdThreshold'], 'yes'],
                         index=['Relative Standard Devation (RSD)', 'RSD of SP Samples: Threshold',
                                'RSD of SS Samples > RSD of SP Samples'], columns=['Value Applied']))

    item['FeatureSelectionTable'] = FeatureSelectionTable
    
    
    nBatchCollect = len((numpy.unique(dataset.sampleMetadata['Batch'].values[~numpy.isnan(dataset.sampleMetadata['Batch'].values)])).astype(int))
    if nBatchCollect == 1:
        item['batchesCollect'] = '1 batch'
    else:
        item['batchesCollect'] = str(nBatchCollect) + ' batches'
    
    if hasattr(dataset, 'fit'):
        nBatchCorrect = len((numpy.unique(dataset.sampleMetadata['Correction Batch'].values[~numpy.isnan(dataset.sampleMetadata['Correction Batch'].values)])).astype(int))
        if nBatchCorrect == 1:
            item['batchesCorrect'] = 'Run-order and batch correction applied (LOESS regression fitted to SP samples in 1 batch)'
        else:
            item['batchesCorrect'] = 'Run-order and batch correction applied (LOESS regression fitted to SP samples in ' + str(nBatchCorrect) + ' batches)'
    else:
        item['batchesCorrect'] =  'Run-order and batch correction not required' 
 
    start = pandas.to_datetime(str(dataset.sampleMetadata['Acquired Time'].loc[dataset.sampleMetadata['Run Order'] == min(dataset.sampleMetadata['Run Order'][dataset.sampleMask])].values[0]))
    end = pandas.to_datetime(str(dataset.sampleMetadata['Acquired Time'].loc[dataset.sampleMetadata['Run Order'] == max(dataset.sampleMetadata['Run Order'][dataset.sampleMask])].values[0]))
    item['start'] = start.strftime('%d/%m/%y')
    item['end'] = end.strftime('%d/%m/%y')
    
    if not destinationPath:
        print('\nFeature Summary')

        print('\nSamples acquired in ' + item['batchesCollect'] + ' between ' + item['start'] + ' and ' + item['end'])
        print(item['batchesCorrect']) 
        
        print('\nTable 2: Features selected based on the following criteria:')
        display(item['FeatureSelectionTable'])
        if item['NfeaturesFailing'] != 0:
            print('\n*Features not passing these criteria are reported and exported as part of the final dataset, however it should be noted that these are not detected or not present in sufficient concentration to be measured accurately, thus results should be interpreted accordingly')
         
    
    # Separate into features passing and failing feature selection for rest of report

    # Sort features by featureMask then by rsdSR
    dataset.featureMetadata['rsdSP'] = dataset.rsdSP
    dataset.featureMetadata.sort_values(by=['Passing Selection', 'rsdSP'], ascending=[False, True], inplace=True)
    orderNew = dataset.featureMetadata.index
    dataset._intensityData = dataset._intensityData[:,orderNew]
    dataset.featureMetadata.drop('rsdSP', axis=1, inplace=True)
    dataset.featureMetadata.reset_index(drop=True, inplace=True)
    
    # Figure: Distribution of RSDs in SP and SS
    if destinationPath:
        item['finalRSDdistributionFigure'] = os.path.join(graphicsPath, item['Name'] + '_finalRSDdistributionFigure.' +
                                                          dataset.Attributes['figureFormat'])
        saveAs = item['finalRSDdistributionFigure']
    else:
        print('\n\nFigure ' + str(figNo) + ': Residual Standard Deviation (RSD) distribution for all samples and all features in final dataset (by sample type).')
        figNo = figNo+1

    plotRSDs(dataset,
            featureName=featureName,
            ratio=False,
            logx=True,
            sortOrder=False,
            withExclusions=False,
            color='matchReport',
            featName=featName,
            hLines=hLine,
            savePath=saveAs,
            figureFormat=dataset.Attributes['figureFormat'],
            dpi=dataset.Attributes['dpi'],
            figureSize=figureSize)
    
    if not destinationPath:
          if item['NfeaturesFailing'] != 0:
            print('\n*Features sorted by RSD in SP samples; with features passing selection (i.e., able to be accurately measured) above the line and those failing (i.e., not able to be accurately measured) below the line')
      

    # Figure: Histogram of log mean abundance by sample type
    if destinationPath:
        item['finalFeatureIntensityHist'] = os.path.join(graphicsPath, item['Name'] + '_finalFeatureIntensityHist.' +
                                                         dataset.Attributes['figureFormat'])
        saveAs = item['finalFeatureIntensityHist']
    else:
        print('\n\nFigure ' + str(figNo) + ': Feature intensity histogram for all samples and all features passing selection (i.e., able to be accurately measured) in final dataset (by sample type).')
        figNo = figNo+1

    _plotAbundanceBySampleType(dataset, SSmask, SPmask, ERmask, saveAs)

    
    # Figures: Distributions for each feature PASSING SELECTION
    figuresFeatureDistributionPassing = OrderedDict()
    temp = dict()
    if destinationPath:
        temp['FeatureConcentrationDistributionPassing'] = os.path.join(graphicsPath, item['Name'] + '_FeatureConcentrationDistributionPassing_')
        saveAs = temp['FeatureConcentrationDistributionPassing']
    else:
        print('Figure ' + str(figNo) + ': Relative concentration distributions, for features passing selection (i.e., able to be accurately measured) in final dataset (by sample type).')
        figNo = figNo+1

    figuresFeatureDistributionPassing = plotTargetedFeatureDistribution(
               dataset,
               featureMask=dataset.featureMetadata['Passing Selection'],
               featureName=featureName,
               logx=False,
               figures=figuresFeatureDistributionPassing,
               savePath=saveAs,
               figureFormat=dataset.Attributes['figureFormat'],
               dpi=dataset.Attributes['dpi'],
               figureSize=dataset.Attributes['figureSize'])

    for key in figuresFeatureDistributionPassing:
        if os.path.join(destinationPath, 'graphics') in str(figuresFeatureDistributionPassing[key]):
            figuresFeatureDistributionPassing[key] = re.sub('.*graphics', 'graphics', figuresFeatureDistributionPassing[key])

    item['FeatureConcentrationDistributionPassing'] = figuresFeatureDistributionPassing
    
    
    # Figures: Distributions for each feature FAILING SELECTION 
    if item['NfeaturesFailing'] != 0:
        figuresFeatureDistributionFailing = OrderedDict()
        temp = dict()
        if destinationPath:
            temp['FeatureConcentrationDistributionFailing'] = os.path.join(graphicsPath, item['Name'] + '_FeatureConcentrationDistributionFailing_')
            saveAs = temp['FeatureConcentrationDistributionFailing']
        else:
            print('Figure ' + str(figNo) + ': Relative concentration distributions, for features failing selection (i.e., not detected, or not able to be accurately measured) in final dataset (by sample type).')
            figNo = figNo+1
    
        figuresFeatureDistributionFailing = plotTargetedFeatureDistribution(
                   dataset,
                   featureMask=dataset.featureMetadata['Passing Selection']==False,
                   featureName=featureName,
                   logx=False,
                   figures=figuresFeatureDistributionFailing,
                   savePath=saveAs,
                   figureFormat=dataset.Attributes['figureFormat'],
                   dpi=dataset.Attributes['dpi'],
                   figureSize=dataset.Attributes['figureSize'])
    
        for key in figuresFeatureDistributionFailing:
            if os.path.join(destinationPath, 'graphics') in str(figuresFeatureDistributionFailing[key]):
                figuresFeatureDistributionFailing[key] = re.sub('.*graphics', 'graphics', figuresFeatureDistributionFailing[key])
    
        item['FeatureConcentrationDistributionFailing'] = figuresFeatureDistributionFailing


    # Table 3: Summary of samples excluded
    if not destinationPath:
        if 'StudySamples Exclusion Details' in sampleSummary:
            print('Missing/Excluded Study Samples')
            print('\nTable 3: Details of missing/excluded study samples')
            display(sampleSummary['StudySamples Exclusion Details'])


    # Write HTML if saving
    if destinationPath:

        # Make paths for graphics local not absolute for use in the HTML.
        for key in item:
            if os.path.join(destinationPath, 'graphics') in str(item[key]):
                item[key] = re.sub('.*graphics', 'graphics', item[key])

        # Generate report
        from jinja2 import Environment, FileSystemLoader

        env = Environment(loader=FileSystemLoader(os.path.join(toolboxPath(), 'Templates')))
 
        template = env.get_template('MS_peakPantheR_FinalSummaryReport.html')
        filename = os.path.join(destinationPath, dataset.name + '_report_finalSummary.html')

        f = open(filename,'w')
        f.write(template.render(item=item,
                                attributes=dataset.Attributes,
                                version=version,
                                graphicsPath=graphicsPath))
        f.close()
        copyBackingFiles(toolboxPath(), os.path.join(destinationPath, 'graphics'))

    return None


def _plotAbundanceBySampleType(dataset, SSmask, SPmask, ERmask, saveAs):

    # Load toolbox wide color scheme
    if 'sampleTypeColours' in dataset.Attributes.keys():
        sTypeColourDict = copy.deepcopy(dataset.Attributes['sampleTypeColours'])
        for stype in SampleType:
            if stype.name in sTypeColourDict.keys():
                sTypeColourDict[stype] = sTypeColourDict.pop(stype.name)
    else:
        sTypeColourDict = {SampleType.StudySample: 'b', SampleType.StudyPool: 'g', SampleType.ExternalReference: 'r',
                            SampleType.MethodReference: 'm', SampleType.ProceduralBlank: 'c', 'Other': 'grey'}

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        # Just for features which are passing selection
        intensityData = dataset.intensityData[:, dataset.featureMask]
        meanIntensities = OrderedDict()
        temp = numpy.nanmean(intensityData[SSmask, :], axis=0)
        temp[numpy.isinf(temp)] = numpy.nan
        meanIntensities['Study Sample'] = temp
        colour = [sTypeColourDict[SampleType.StudySample]]

    if sum(SPmask) != 0:
        temp = numpy.nanmean(intensityData[SPmask, :], axis=0)
        temp[numpy.isinf(temp)] = numpy.nan
        meanIntensities['Study Reference'] = temp
        colour.append(sTypeColourDict[SampleType.StudyPool])
    if sum(ERmask) != 0:
        temp = numpy.nanmean(intensityData[ERmask, :], axis=0)
        temp[numpy.isinf(temp)] = numpy.nan
        meanIntensities['Long-Term Reference'] = temp
        colour.append(sTypeColourDict[SampleType.ExternalReference])

    histogram(meanIntensities,
        xlabel='Mean Feature Intensity',
        color=colour,
        title='',
        histBins=dataset.Attributes['histBins'],
        logx=True,
        savePath=saveAs,
        figureFormat=dataset.Attributes['figureFormat'],
        dpi=dataset.Attributes['dpi'],
        figureSize=dataset.Attributes['figureSize'])