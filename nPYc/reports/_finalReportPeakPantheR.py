import os
import numpy
import pandas
from collections import OrderedDict
import copy
from IPython.display import display
import re
import shutil
from .._toolboxPath import toolboxPath
from ..objects import MSDataset
from ..plotting import plotRSDs, plotIonMap, plotTargetedFeatureDistribution, plotAbundanceBySampleType
from ._generateSampleReport import _generateSampleReport
from ..utilities.ms import generateTypeRoleMasks
from ..utilities._internal import _copyBackingFiles as copyBackingFiles
from ..utilities._errorHandling import npycToolboxError


from ..__init__ import __version__ as version


def _finalReportPeakPantheR(datasetOriginal, destinationPath=None, labelFeaturesBy='Feature Name', orderFeaturesBy='rsdSP', withExclusions=False):
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
        graphicsPath = os.path.join(destinationPath, 'graphics', 'finalSummary')
        if not os.path.exists(graphicsPath):
            os.makedirs(graphicsPath)

        # Copy required file for final report
        shutil.copy2(os.path.join(toolboxPath(), 'Templates', 'NPC_assay_coverage.pdf'),
                     os.path.join(destinationPath, 'graphics', 'NPC_assay_coverage.pdf'))
    else:
        graphicsPath = None
        saveAs = None


    # Do some checks
    if (labelFeaturesBy is not None) and (not hasattr(datasetOriginal.featureMetadata, labelFeaturesBy)):
        raise npycToolboxError('Unable to label features by: ' + labelFeaturesBy + ' as column not present in `dataset.featureMetadata`')

    if (orderFeaturesBy is not None) and (not hasattr(datasetOriginal.featureMetadata, orderFeaturesBy)):
        raise npycToolboxError('Unable to label features by: ' + orderFeaturesBy + ' as column not present in `dataset.featureMetadata`')

    # Apply sample/feature masks if exclusions to be applied
    dataset = copy.deepcopy(datasetOriginal)
    if withExclusions:
        dataset.applyMasks()

    # Set up template item and save required info
    item = dict()
    item['Name'] = dataset.name
    item['Nsamples'] = dataset.intensityData.shape[0]
    item['Nfeatures'] = dataset.intensityData.shape[1]
    item['NfeaturesPassing'] = sum(dataset.featureMask)
    nfeaturesFailing = item['Nfeatures'] - item['NfeaturesPassing']
    if nfeaturesFailing != 0:
        item['NfeaturesFailing'] = nfeaturesFailing
        hLine = [item['NfeaturesFailing']]
    else:
        hLine = None
    figNo = 1

    # Final dataset summary
    if not destinationPath:
        print('Final Dataset\n')
        print(str(item['Nsamples']) + ' samples')
        print(str(item['Nfeatures']) + ' features')
        if nfeaturesFailing != 0:
            print('\t' + str(item['NfeaturesPassing']) + ' detected and passing feature selection')
            print('\t' + str(item['NfeaturesFailing']) + ' not detected or not present in sufficient concentration to be measured precisely')

    # Table 1: Sample summary

    # Generate sample summary

    sampleSummary = _generateSampleReport(dataset, destinationPath=None, returnOutput=True)

    sampleSummary['isFinalReport'] = True
    #if 'StudySamples Exclusion Details' in sampleSummary:
    #    sampleSummary['studySamplesExcluded'] = True
    #else:
    #    sampleSummary['studySamplesExcluded'] = False
    item['sampleSummary'] = sampleSummary

    if not destinationPath:
        print('\n\nSample Summary')
        print('\nTable 1: Sample summary table.')
        display(sampleSummary['Dataset'])
        print('\n*Details of any missing/excluded study samples given at the end of the report\n')


    # Table 2: Feature Selection parameters
    FeatureSelectionTable = pandas.DataFrame(
        data=['yes', dataset.Attributes['corrMethod'], dataset.Attributes['corrThreshold']],
        index=['Correlation to Dilution', 'Correlation to Dilution: Method', 'Correlation to Dilution: Threshold'],
        columns=['Value Applied'])

    if sum(dataset.corrExclusions) != dataset.noSamples:
        temp = ', '.join(dataset.sampleMetadata.loc[dataset.corrExclusions == False, 'Sample File Name'].values)
        FeatureSelectionTable = pandas.concat([FeatureSelectionTable,
            pandas.DataFrame(data=temp, index=['Correlation to Dilution: Sample Exclusions'], columns=['Value Applied'])])
    else:
        FeatureSelectionTable = pandas.concat([FeatureSelectionTable,
            pandas.DataFrame(data=['none'], index=['Correlation To Dilution: Sample Exclusions'], columns=['Value Applied'])])
    FeatureSelectionTable = pandas.concat([FeatureSelectionTable,
        pandas.DataFrame(data=['yes', dataset.Attributes['rsdThreshold'], 'yes'],
                         index=['Relative Standard Devation (RSD)', 'RSD of SR Samples: Threshold',
                                'RSD of SS Samples > RSD of SR Samples'], columns=['Value Applied'])])

    item['FeatureSelectionTable'] = FeatureSelectionTable
    
    
    nBatchCollect = len((numpy.unique(dataset.sampleMetadata['Batch'].values[~numpy.isnan(dataset.sampleMetadata['Batch'].values)])).astype(int))
    if nBatchCollect == 1:
        item['batchesCollect'] = '1 batch'
    else:
        item['batchesCollect'] = str(nBatchCollect) + ' batches'
    
    if hasattr(dataset, 'fit'):
        nBatchCorrect = len((numpy.unique(dataset.sampleMetadata['Correction Batch'].values[~numpy.isnan(dataset.sampleMetadata['Correction Batch'].values)])).astype(int))
        if nBatchCorrect == 1:
            item['batchesCorrect'] = 'Run-order and batch correction applied (LOESS regression fitted to SR samples in 1 batch)'
        else:
            item['batchesCorrect'] = 'Run-order and batch correction applied (LOESS regression fitted to SR samples in ' + str(nBatchCorrect) + ' batches)'
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
        if nfeaturesFailing != 0:
            print('\n*Features not passing these criteria are reported and exported as part of the final dataset, however it should be noted that these are not detected or not present in sufficient concentration to be measured precisely, thus results should be interpreted accordingly')
         
    
    # Separate into features passing and failing feature selection for rest of report
    
    # Figure: Distribution of RSDs in SP and SS
    if destinationPath:
        item['finalRSDdistributionFigure'] = os.path.join(graphicsPath, item['Name'] + '_rsdHist.' +
                                                          dataset.Attributes['figureFormat'])
        saveAs = item['finalRSDdistributionFigure']
        item['orderFeaturesBy'] = orderFeaturesBy
    else:
        print('\n\nFigure ' + str(figNo) + ': Residual Standard Deviation (RSD) distribution for all samples and all features in final dataset (by sample type), ordered by ' + orderFeaturesBy)
        figNo = figNo+1

    plotRSDs(dataset,
            featureName=labelFeaturesBy,
            ratio=False,
            logx=True,
            sortOrder=orderFeaturesBy,
            withExclusions=False,
            featName=True,
            hLines=hLine,
            savePath=saveAs,
            figureFormat=dataset.Attributes['figureFormat'],
            dpi=dataset.Attributes['dpi'],
            figureSize=(dataset.Attributes['figureSize'][0], dataset.Attributes['figureSize'][1] * (dataset.noFeatures / 35)))
    
    if not destinationPath:
          if nfeaturesFailing != 0:
            print('\n*Features passing selection (i.e., able to be precisely measured) plotted above the line and those failing (i.e., not able to be precisely measured) below the line')
      

    # Figure: Histogram of log mean abundance by sample type
    if destinationPath:
        item['finalFeatureIntensityHist'] = os.path.join(graphicsPath, item['Name'] + '_intensityHist.' +
                                                         dataset.Attributes['figureFormat'])
        saveAs = item['finalFeatureIntensityHist']
    else:
        print('\n\nFigure ' + str(figNo) + ': Feature intensity histogram for all samples and all features passing selection (i.e., able to be precisely measured) in final dataset (by sample type).')
        figNo = figNo+1

    plotAbundanceBySampleType(dataset,
                              saveAs)

    # Figure: Ion map
    if 'm/z' in dataset.featureMetadata.columns and 'Retention Time' in dataset.featureMetadata.columns:
        if destinationPath:
            item['finalIonMap'] = os.path.join(graphicsPath, item['Name'] + '_ionMap.' + dataset.Attributes['figureFormat'])
            saveAs = item['finalIonMap']
        else:
            print('Figure ' + str(figNo) + ': Ion map of all features (coloured by log median intensity).')
            figNo = figNo+1

        plotIonMap(dataset,
                   savePath=saveAs,
                   figureFormat=dataset.Attributes['figureFormat'],
                   dpi=dataset.Attributes['dpi'],
                   figureSize=dataset.Attributes['figureSize'])

    else:
        if not destinationPath:
            print('No Retention Time and m/z information, unable to plot the ion map.\n')

    
    # Figures: Distributions for each feature PASSING SELECTION
    figuresFeatureDistributionPassing = OrderedDict()
    temp = dict()
    if destinationPath:
        temp['FeatureConcentrationDistributionPassing'] = os.path.join(graphicsPath, item['Name'] + '_featurePassViolin')
        saveAs = temp['FeatureConcentrationDistributionPassing']
    else:
        print('Figure ' + str(figNo) + ': Relative concentration distributions, for features passing selection (i.e., able to be precisely measured) in final dataset (by sample type).')
        figNo = figNo+1


    figuresFeatureDistributionPassing = plotTargetedFeatureDistribution(
               dataset,
               featureMask=dataset.featureMask,
               labelFeaturesBy=labelFeaturesBy,
               orderFeaturesBy=orderFeaturesBy,
               logx=False,
               figures=figuresFeatureDistributionPassing,
               savePath=saveAs)

    for key in figuresFeatureDistributionPassing:
        if os.path.join(destinationPath, 'graphics') in str(figuresFeatureDistributionPassing[key]):
            figuresFeatureDistributionPassing[key] = re.sub('.*graphics', 'graphics', figuresFeatureDistributionPassing[key])

    item['FeatureConcentrationDistributionPassing'] = figuresFeatureDistributionPassing
    
    
    # Figures: Distributions for each feature FAILING SELECTION 
    if nfeaturesFailing != 0:
        figuresFeatureDistributionFailing = OrderedDict()
        temp = dict()
        if destinationPath:
            temp['FeatureConcentrationDistributionFailing'] = os.path.join(graphicsPath, item['Name'] + '_featureFailViolin')
            saveAs = temp['FeatureConcentrationDistributionFailing']
        else:
            print('Figure ' + str(figNo) + ': Relative concentration distributions, for features failing selection (i.e., not detected, or not able to be precisely measured) in final dataset (by sample type).')
            figNo = figNo+1
    
        figuresFeatureDistributionFailing = plotTargetedFeatureDistribution(
                   dataset,
                   featureMask=dataset.featureMask == False,
                   labelFeaturesBy=labelFeaturesBy,
                   orderFeaturesBy=orderFeaturesBy,
                   logx=False,
                   figures=figuresFeatureDistributionFailing,
                   savePath=saveAs)
    
        for key in figuresFeatureDistributionFailing:
            if os.path.join(destinationPath, 'graphics') in str(figuresFeatureDistributionFailing[key]):
                figuresFeatureDistributionFailing[key] = re.sub('.*graphics', 'graphics', figuresFeatureDistributionFailing[key])
    
        item['FeatureConcentrationDistributionFailing'] = figuresFeatureDistributionFailing


    # Table 3: Summary of samples excluded
    if not destinationPath:
        if hasattr(sampleSummary, 'Missing/excluded SS Details'):
            print('Missing/Excluded Study Samples')
            print('\nTable 3: Details of missing/excluded study samples')
            display(sampleSummary['Missing/excluded SS Details'])


    # Write HTML if saving
    if destinationPath:

        # Make paths for graphics local not absolute for use in the HTML.
        for key in item:
            if os.path.join(destinationPath, 'graphics') in str(item[key]):
                #print(item[key])
                item[key] = re.sub('.*graphics', 'graphics', item[key])

        # Generate report
        from jinja2 import Environment, FileSystemLoader

        env = Environment(loader=FileSystemLoader(os.path.join(toolboxPath(), 'Templates')))
 
        template = env.get_template('MS_peakPantheR_FinalSummaryReport.html')
        filename = os.path.join(destinationPath, dataset.name + '_finalSummary.html')

        f = open(filename,'w')
        f.write(template.render(item=item,
                                attributes=dataset.Attributes,
                                version=version,
                                graphicsPath=graphicsPath))
        f.close()
        copyBackingFiles(toolboxPath(), os.path.join(destinationPath, 'graphics'))

    return None