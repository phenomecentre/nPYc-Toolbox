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
from ..plotting import plotTIC, histogram, plotLRTIC, jointplotRSDvCorrelation, plotRSDs, plotIonMap, plotBatchAndROCorrection, plotScores, plotLoadings
from ._generateSampleReport import _generateSampleReport
from ..utilities import generateLRmask, rsd
from ..utilities._internal import _vcorrcoef
from ..utilities._internal import _copyBackingFiles as copyBackingFiles
from ..enumerations import AssayRole, SampleType
from ._generateBasicPCAReport import generateBasicPCAReport

from ..__init__ import __version__ as version


def _generateReportMS(dataset, reportType, withExclusions=False, withArtifactualFiltering=None, output=None,
                          msDataCorrected=None, pcaModel=None, batch_correction_window=11):
    """
    Summarise different aspects of an MS dataset

    Generate reports for ``feature summary``, ``correlation to dilution``, ``batch correction assessment``, ``batch correction summary``, ``feature selection``, or ``final report``

    * **'feature summary'** Generates feature summary report, plots figures including those for feature abundance, sample TIC and acquisition structure, correlation to dilution, RSD and an ion map.
    * **'correlation to dilution'** Generates a more detailed report on correlation to dilution, broken down by batch subset with TIC, detector voltage, a summary, and heatmap indicating potential saturation or other issues.
    * **'batch correction assessment'** Generates a report before batch correction showing TIC overall and intensity and batch correction fit for a subset of features, to aid specification of batch start and end points.
    * **'batch correction summary'** Generates a report post batch correction with pertinant figures (TIC, RSD etc.) before and after.
    * **'feature selection'** Generates a summary of the number of features passing feature selection (with current settings as definite in the SOP), and a heatmap showing how this number would be affected by changes to RSD and correlation to dilution thresholds.
    * **'final report'** Generates a summary of the final dataset, lists sample numbers present, a selection of figures summarising dataset quality, and a final list of samples missing from acquisition.

    :param MSDataset msDataTrue: MSDataset to report on
    :param str reportType: Type of report to generate, one of ``feature summary``, ``correlation to dilution``, ``batch correction``, ``feature selection``, or ``final report``
    :param bool withExclusions: If ``True``, only report on features and samples not masked by the sample and feature masks
    :param None or bool withArtifactualFiltering: If ``None`` use the value from ``Attributes['artifactualFilter']``. If ``True`` apply artifactual filtering to the ``feature selection`` report and ``final report``
    :param output: If ``None`` plot interactively, otherwise save report to the path specified
    :type output: None or str
    :param MSDataset msDataCorrected: Only if ``batch correction``, if msDataCorrected included will generate report post correction
    :param PCAmodel pcaModel: Only if ``final report``, if PCAmodel object is available PCA scores plots coloured by sample type will be added to report
    """

    acceptableOptions = {'feature summary', 'correlation to dilution',
                         'batch correction assessment',
                         'batch correction summary', 'feature selection', 'final report'}

    # Check inputs
    if not isinstance(dataset, MSDataset):
        raise TypeError('msData must be an instance of MSDataset')

    if not isinstance(reportType, str) & (reportType.lower() in acceptableOptions):
        raise ValueError('reportType must be one of: ' + str(acceptableOptions))

    if not isinstance(withExclusions, bool):
        raise TypeError('withExclusions must be a bool')

    if withArtifactualFiltering is not None:
        if not isinstance(withArtifactualFiltering, bool):
            raise TypeError('withArtifactualFiltering must be a bool')
    if withArtifactualFiltering is None:
        withArtifactualFiltering = dataset.Attributes['artifactualFilter']
    # if self.Attributes['artifactualFilter'] is False, can't/shouldn't apply it.
    # However if self.Attributes['artifactualFilter'] is True, the user can have the choice to not apply it (withArtifactualFilering=False).
    if (withArtifactualFiltering is True) & (dataset.Attributes['artifactualFilter'] is False):
        warnings.warn("Warning: Attributes['artifactualFilter'] set to \'False\', artifactual filtering cannot be applied.")
        withArtifactualFiltering = False

    if output is not None:
        if not isinstance(output, str):
            raise TypeError('output must be a string')

    if msDataCorrected is not None:
        if not isinstance(msDataCorrected, MSDataset):
            raise TypeError('msDataCorrected must be an instance of nPYc.MSDataset')

    if pcaModel is not None:
        if not isinstance(pcaModel, ChemometricsPCA):
            raise TypeError('pcaModel must be a ChemometricsPCA object')

    sns.set_style("whitegrid")

    # Create directory to save output
    if output:
        if not os.path.exists(output):
            os.makedirs(output)
        if not os.path.exists(os.path.join(output, 'graphics')):
            os.makedirs(os.path.join(output, 'graphics'))

    # Apply sample/feature masks if exclusions to be applied
    msData = copy.deepcopy(dataset)
    if withExclusions:
        msData.applyMasks()

    if reportType.lower() == 'feature summary':
        _featureReport(msData, output)
    elif reportType.lower() == 'correlation to dilution':
        _featureCorrelationToDilutionReport(msData, output)
    elif reportType.lower() == 'feature selection':
        _featureSelectionReport(msData, output)
    elif reportType.lower() == 'batch correction assessment':
        _batchCorrectionAssessmentReport(msData, output)
    elif reportType.lower() == 'batch correction summary':
        _batchCorrectionSummaryReport(msData, msDataCorrected, output)
    elif reportType.lower() == 'final report':
        _finalReport(msData, output, pcaModel)


def _finalReport(dataset, output=None, pcaModel=None, withArtifactualFiltering=True):
    """
    Generates a summary of the final dataset, lists sample numbers present, a selection of figures summarising dataset quality, and a final list of samples missing from acquisition.
    """

    # Table 1: Sample summary
    # Generate sample summary
    sampleSummary = _generateSampleReport(dataset, withExclusions=True, output=None, returnOutput=True)
    
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
    item['ReportType'] = 'feature summary'
    item['Nfeatures'] = dataset.intensityData.shape[1]
    item['Nsamples'] = dataset.intensityData.shape[0]
    item['SScount'] = str(sum(SSmask))
    item['SPcount'] = str(sum(SPmask))
    item['ERcount'] = str(sum(ERmask))
    item['LRcount'] = str(sum(LRmask))
    item['corrMethod'] = dataset.Attributes['corrMethod']
    sampleSummary['isFinalReport'] = True
    item['sampleSummary'] = sampleSummary
    ##
    # Report stats
    ##
    if output is not None:
        if not os.path.exists(output):
            os.makedirs(output)
        if not os.path.exists(os.path.join(output, 'graphics')):
            os.makedirs(os.path.join(output, 'graphics'))
        graphicsPath = os.path.join(output, 'graphics', 'report_FinalSummary')
        if not os.path.exists(graphicsPath):
            os.makedirs(graphicsPath)
    else:
        graphicsPath = None
        saveAs = None

    if not output:
        print('Table 1: Summary of samples present')
        display(sampleSummary['Acquired'])
        #if 'Excluded Details' in sampleSummary:
        #    print('Table 2: Summary of samples excuded')
        #    display(sampleSummary['Excluded Details'])

    # Figure 1: Acquisition Structure, TIC by sample and batch
    nBatchCollect = len((numpy.unique(
        dataset.sampleMetadata['Batch'].values[~numpy.isnan(dataset.sampleMetadata['Batch'].values)])).astype(int))
    if nBatchCollect == 1:
        item['nBatchesCollect'] = '1 batch'
    else:
        item['nBatchesCollect'] = str(nBatchCollect) + ' batches'

    nBatchCorrect = len((numpy.unique(dataset.sampleMetadata['Correction Batch'].values[
                                          ~numpy.isnan(dataset.sampleMetadata['Correction Batch'].values)])).astype(int))
    if nBatchCorrect == 1:
        item['nBatchesCorrect'] = '1 batch'
    else:
        item['nBatchesCorrect'] = str(nBatchCorrect) + ' batches'

    start = pandas.to_datetime(str(dataset.sampleMetadata['Acquired Time'].loc[dataset.sampleMetadata['Run Order'] == min(
            dataset.sampleMetadata['Run Order'][dataset.sampleMask])].values[0]))
    end = pandas.to_datetime(str(dataset.sampleMetadata['Acquired Time'].loc[dataset.sampleMetadata['Run Order'] == max(
            dataset.sampleMetadata['Run Order'][dataset.sampleMask])].values[0]))
    item['start'] = start.strftime('%d/%m/%y')
    item['end'] = end.strftime('%d/%m/%y')

    if output:
        item['finalTICbatches'] = os.path.join(graphicsPath,
                                               item['Name'] + '_finalTICbatches.' + dataset.Attributes['figureFormat'])
        saveAs = item['finalTICbatches']
    else:
        print('Acquisition Structure')
        print(
            '\n\tSamples acquired in ' + item['nBatchesCollect'] + ' between ' + item['start'] + ' and ' + item['end'])
        print('\n\tBatch correction applied (LOESS regression fitted to SP samples in ' + item[
            'nBatchesCorrect'] + ') for run-order correction and batch alignment\n')
        print('Figure 1: Acquisition Structure')

    plotTIC(dataset,
            savePath=saveAs,
            addBatchShading=True,
            figureFormat=dataset.Attributes['figureFormat'],
            dpi=dataset.Attributes['dpi'],
            figureSize=dataset.Attributes['figureSize'])

    # Table 2: Feature Selection parameters
    FeatureSelectionTable = pandas.DataFrame(
        data=['yes', dataset.Attributes['corrMethod'], dataset.Attributes['corrThreshold']],
        index=['Correlation to Dilution', 'Correlation to Dilution: Method', 'Correlation to Dilution: Threshold'],
        columns=['Applied'])

    if sum(dataset.corrExclusions) != dataset.noSamples:
        temp = ', '.join(dataset.sampleMetadata.loc[dataset.corrExclusions == False, 'Sample File Name'].values)
        FeatureSelectionTable = FeatureSelectionTable.append(
            pandas.DataFrame(data=temp, index=['Correlation to Dilution: Sample Exclusions'], columns=['Applied']))
    else:
        FeatureSelectionTable = FeatureSelectionTable.append(
            pandas.DataFrame(data=['none'], index=['Correlation To Dilution: Sample Exclusions'], columns=['Applied']))
    FeatureSelectionTable = FeatureSelectionTable.append(
        pandas.DataFrame(data=['yes', dataset.Attributes['rsdThreshold'], 'yes'],
                         index=['Relative Standard Devation (RSD)', 'RSD of SP Samples: Threshold',
                                'RSD of SS Samples > RSD of SP Samples'], columns=['Applied']))
    if withArtifactualFiltering:
        FeatureSelectionTable = FeatureSelectionTable.append(pandas.DataFrame(
            data=['yes', dataset.Attributes['deltaMzArtifactual'], dataset.Attributes['overlapThresholdArtifactual'],
                  dataset.Attributes['corrThresholdArtifactual']],
            index=['Artifactual Filtering', 'Artifactual Filtering: Delta m/z',
                   'Artifactual Filtering: Overlap Threshold', 'Artifactual Filtering: Correlation Threshold'],
            columns=['Applied']))

    item['FeatureSelectionTable'] = FeatureSelectionTable

    if not output:
        print('Feature Selection Summary')
        print('Features selected based on:')
        display(item['FeatureSelectionTable'])
        print('\n')

    # Figure 2: Final TIC
    if output:
        item['finalTIC'] = os.path.join(graphicsPath, item['Name'] + '_finalTIC.' + dataset.Attributes['figureFormat'])
        saveAs = item['finalTIC']
    else:
        print('Figure 2: Total Ion Count (TIC) for all samples and all features in final dataset.')

    plotTIC(dataset,
            addViolin=True,
            title='',
            savePath=saveAs,
            figureFormat=dataset.Attributes['figureFormat'],
            dpi=dataset.Attributes['dpi'],
            figureSize=dataset.Attributes['figureSize'])

    # Figure 3: Histogram of log mean abundance by sample type
    if output:
        item['finalFeatureIntensityHist'] = os.path.join(graphicsPath, item['Name'] + '_finalFeatureIntensityHist.' +
                                                         dataset.Attributes['figureFormat'])
        saveAs = item['finalFeatureIntensityHist']
    else:
        print(
            'Figure 3: Feature intensity histogram for all samples and all features in final dataset (by sample type)')

    _plotAbundanceBySampleType(dataset.intensityData, SSmask, SPmask, ERmask, saveAs, dataset)

    # Figure 4: Histogram of RSDs in SP and SS
    if output:
        item['finalRSDdistributionFigure'] = os.path.join(graphicsPath, item['Name'] + '_finalRSDdistributionFigure.' +
                                                          dataset.Attributes['figureFormat'])
        saveAs = item['finalRSDdistributionFigure']
    else:
        print(
            'Figure 4: Residual Standard Deviation (RSD) distribution for all samples and all features in final dataset (by sample type)')

    plotRSDs(dataset,
             ratio=False,
             logx=True,
             color='matchReport',
             savePath=saveAs,
             figureFormat=dataset.Attributes['figureFormat'],
             dpi=dataset.Attributes['dpi'],
             figureSize=dataset.Attributes['figureSize'])

    # Figure 5: Ion map
    if output:
        item['finalIonMap'] = os.path.join(graphicsPath, item['Name'] + '_finalIonMap.' + dataset.Attributes['figureFormat'])
        saveAs = item['finalIonMap']
    else:
        print('Figure 5: Ion map of all features (coloured by log median intensity).')

    plotIonMap(dataset,
               savePath=saveAs,
               figureFormat=dataset.Attributes['figureFormat'],
               dpi=dataset.Attributes['dpi'],
               figureSize=dataset.Attributes['figureSize'])

    # Figures 6 and 7: (if available) PCA scores and loadings plots by sample type
    ##
    # PCA plots
    ##

    if not 'Plot Sample Type' in dataset.sampleMetadata.columns:
        dataset.sampleMetadata.loc[~SSmask & ~SPmask & ~ERmask, 'Plot Sample Type'] = 'Sample'
        dataset.sampleMetadata.loc[SSmask, 'Plot Sample Type'] = 'Study Sample'
        dataset.sampleMetadata.loc[SPmask, 'Plot Sample Type'] = 'Study Pool'
        dataset.sampleMetadata.loc[ERmask, 'Plot Sample Type'] = 'External Reference'

    if pcaModel:
        if output:
            pcaPath = output

        else:
            pcaPath = None
        pcaModel = generateBasicPCAReport(pcaModel, dataset, figureCounter=6, output=pcaPath, fileNamePrefix='')


    ##
    # Sample summary
    ##
    if not output:
        print('Table 1: Summary of samples present')
        display(sampleSummary['Acquired'])
        if 'StudySamples Exclusion Details' in sampleSummary:
            print('Table 2: Summary of samples excluded')
            display(sampleSummary['StudySamples Exclusion Details'])

    ##
    # Write HTML if saving
    ##
    if output:
        # Make paths for graphics local not absolute for use in the HTML.
        for key in item:
            if os.path.join(output, 'graphics') in str(item[key]):
                item[key] = re.sub('.*graphics', 'graphics', item[key])

        # Generate report
        from jinja2 import Environment, FileSystemLoader

        env = Environment(loader=FileSystemLoader(os.path.join(toolboxPath(), 'Templates')))
        template = env.get_template('MS_FinalSummaryReport.html')
        filename = os.path.join(output, dataset.name + '_report_FinalSummary.html')

        f = open(filename,'w')
        f.write(template.render(item=item,
                                attributes=dataset.Attributes,
                                version=version,
                                graphicsPath=graphicsPath))
        f.close()
        copyBackingFiles(toolboxPath(), output)
    return None


def _featureReport(dataset, output=None):
    """
    Generates feature summary report, plots figures including those for feature abundance, sample TIC and acquisition structure, correlation to dilution, RSD and an ion map.
    """

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
    item['ReportType'] = 'feature summary'
    item['Nfeatures'] = dataset.intensityData.shape[1]
    item['Nsamples'] = dataset.intensityData.shape[0]
    item['SScount'] = str(sum(SSmask))
    item['SPcount'] = str(sum(SPmask))
    item['ERcount'] = str(sum(ERmask))
    item['LRcount'] = str(sum(LRmask))
    item['corrMethod'] = dataset.Attributes['corrMethod']

    ##
    # Report stats
    ##
    if output is not None:
        if not os.path.exists(output):
            os.makedirs(output)
        if not os.path.exists(os.path.join(output, 'graphics')):
            os.makedirs(os.path.join(output, 'graphics'))
        graphicsPath = os.path.join(output, 'graphics', 'report_featureSummary')
        if not os.path.exists(graphicsPath):
            os.makedirs(graphicsPath)
    else:
        graphicsPath = None
        saveAs = None


    # Generate correlation to dilution for each batch subset - plot TIC and histogram of correlation to dilution

    # Mean intensities of Study Pool samples (for future plotting segmented by intensity)
    meanIntensitiesSP = numpy.log(numpy.nanmean(dataset.intensityData[SPmask, :], axis=0))
    meanIntensitiesSP[numpy.mean(dataset.intensityData[SPmask, :], axis=0) == 0] = numpy.nan
    meanIntensitiesSP[numpy.isinf(meanIntensitiesSP)] = numpy.nan

    # Figure 1: Histogram of log mean abundance by sample type
    if output:
        item['FeatureIntensityFigure'] = os.path.join(graphicsPath,
                                                      item['Name'] + '_meanIntensityFeature.' + dataset.Attributes[
                                                          'figureFormat'])
        saveAs = item['FeatureIntensityFigure']
    else:
        print('Figure 1: Feature intensity histogram for all samples and all features in dataset (by sample type).')

    _plotAbundanceBySampleType(dataset.intensityData, SSmask, SPmask, ERmask, saveAs, dataset)

    # Figure 2: Sample intensity TIC and distribution by sample type
    if output:
        item['SampleIntensityFigure'] = os.path.join(graphicsPath, item['Name'] + '_meanIntensitySample.' + dataset.Attributes[
            'figureFormat'])
        saveAs = item['SampleIntensityFigure']
    else:
        print('Figure 2: Sample Total Ion Count (TIC) and distribtion (coloured by sample type).')

    # TIC all samples
    plotTIC(dataset,
            addViolin=True,
            savePath=saveAs,
            title='',
            figureFormat=dataset.Attributes['figureFormat'],
            dpi=dataset.Attributes['dpi'],
            figureSize=dataset.Attributes['figureSize'])

    # Figure 3: Acquisition structure and detector voltage
    if output:
        item['AcquisitionStructureFigure'] = os.path.join(graphicsPath,
                                                          item['Name'] + '_acquisitionStructure.' + dataset.Attributes[
                                                              'figureFormat'])
        saveAs = item['AcquisitionStructureFigure']
    else:
        print('Figure 3: Acquisition structure (coloured by detector voltage).')

    # TIC all samples
    plotTIC(dataset,
            addViolin=False,
            addBatchShading=True,
            addLineAtGaps=True,
            colourByDetectorVoltage=True,
            savePath=saveAs,
            title='',
            figureFormat=dataset.Attributes['figureFormat'],
            dpi=dataset.Attributes['dpi'],
            figureSize=dataset.Attributes['figureSize'])

    # Correlation to dilution figures:
    if sum(LRmask) != 0:

        # Figure 4: Histogram of correlation to dilution by abundance percentiles
        if output:
            item['CorrelationByPercFigure'] = os.path.join(graphicsPath,
                                                           item['Name'] + '_correlationByPerc.' + dataset.Attributes[
                                                               'figureFormat'])
            saveAs = item['CorrelationByPercFigure']
        else:
            print('Figure 4: Histogram of ' + item[
                'corrMethod'] + ' correlation of features to serial dilution, segmented by percentile.')

        histogram(dataset.correlationToDilution,
                  xlabel='Correlation to Dilution',
                  histBins=dataset.Attributes['histBins'],
                  quantiles=dataset.Attributes['quantiles'],
                  inclusionVector=numpy.exp(meanIntensitiesSP),
                  savePath=saveAs,
                  figureFormat=dataset.Attributes['figureFormat'],
                  dpi=dataset.Attributes['dpi'],
                  figureSize=dataset.Attributes['figureSize'])

        # Figure 5: TIC of linearity reference samples
        if output:
            item['TICinLRfigure'] = os.path.join(graphicsPath,
                                                 item['Name'] + '_TICinLR.' + dataset.Attributes['figureFormat'])
            saveAs = item['TICinLRfigure']
        else:
            print('Figure 5: TIC of linearity reference (LR) samples coloured by sample dilution.')

        plotLRTIC(dataset,
                  sampleMask=LRmask,
                  savePath=saveAs,
                  figureFormat=dataset.Attributes['figureFormat'],
                  dpi=dataset.Attributes['dpi'],
                  figureSize=dataset.Attributes['figureSize'])

    else:
        if not output:
            print('Figure 4: Histogram of ' + item[
                'corrMethod'] + ' correlation of features to serial dilution, segmented by percentile.')
            print('Unable to calculate (no linearity reference samples present in dataset).\n')

            print('Figure 5: TIC of linearity reference (LR) samples coloured by sample dilution')
            print('Unable to calculate (no linearity reference samples present in dataset).\n')

    # Figure 6: Histogram of RSD in SP samples by abundance percentiles
    if output:
        item['RsdByPercFigure'] = os.path.join(graphicsPath,
                                               item['Name'] + '_rsdByPerc.' + dataset.Attributes['figureFormat'])
        saveAs = item['RsdByPercFigure']
    else:
        print(
            'Figure 6: Histogram of Residual Standard Deviation (RSD) in study pool (SP) samples, segmented by abundance percentiles.')

    histogram(dataset.rsdSP,
              xlabel='RSD',
              histBins=dataset.Attributes['histBins'],
              quantiles=dataset.Attributes['quantiles'],
              inclusionVector=numpy.exp(meanIntensitiesSP),
              logx=False,
              xlim=(0, 100),
              savePath=saveAs,
              figureFormat=dataset.Attributes['figureFormat'],
              dpi=dataset.Attributes['dpi'],
              figureSize=dataset.Attributes['figureSize'])

    # Figure 7: Scatterplot of RSD vs correlation to dilution
    if sum(LRmask) != 0:
        if output:
            item['RsdVsCorrelationFigure'] = os.path.join(graphicsPath,
                                                          item['Name'] + '_rsdVsCorrelation.' + dataset.Attributes[
                                                              'figureFormat'])
            saveAs = item['RsdVsCorrelationFigure']
        else:
            print('Figure 7: Scatterplot of RSD vs correlation to dilution.')

        jointplotRSDvCorrelation(dataset.rsdSP,
                                 dataset.correlationToDilution,
                                 savePath=saveAs,
                                 figureFormat=dataset.Attributes['figureFormat'],
                                 dpi=dataset.Attributes['dpi'],
                                 figureSize=dataset.Attributes['figureSize'])

    else:
        if not output:
            print('Figure 7: Scatterplot of RSD vs correlation to dilution.')
            print('Unable to calculate (no serial dilution samples present in dataset).\n')

    if 'Peak Width' in dataset.featureMetadata.columns:
        # Figure 8: Histogram of chromatographic peak width
        if output:
            item['PeakWidthFigure'] = os.path.join(graphicsPath,
                                                   item['Name'] + '_peakWidth.' + dataset.Attributes['figureFormat'])
            saveAs = item['PeakWidthFigure']
        else:
            print('Figure 8: Histogram of chromatographic peak width.')

        histogram(dataset.featureMetadata['Peak Width'],
                  xlabel='Peak Width (minutes)',
                  histBins=dataset.Attributes['histBins'],
                  savePath=saveAs,
                  figureFormat=dataset.Attributes['figureFormat'],
                  dpi=dataset.Attributes['dpi'],
                  figureSize=dataset.Attributes['figureSize'])
    else:
        if not output:
            print('\x1b[31;1m No peak width data to plot')
            print('Figure 8: Histogram of chromatographic peak width.')

    # Figure 9: Residual Standard Deviation (RSD) distribution for all samples and all features in dataset (by sample type)
    if output:
        item['RSDdistributionFigure'] = os.path.join(graphicsPath,
                                                     item['Name'] + '_RSDdistributionFigure.' + dataset.Attributes[
                                                         'figureFormat'])
        saveAs = item['RSDdistributionFigure']
    else:
        print('Figure 9: RSD distribution for all samples and all features in dataset (by sample type).')

    plotRSDs(dataset,
             ratio=False,
             logx=True,
             color='matchReport',
             savePath=saveAs,
             figureFormat=dataset.Attributes['figureFormat'],
             dpi=dataset.Attributes['dpi'],
             figureSize=dataset.Attributes['figureSize'])

    # Figure 10: Ion map
    if output:
        item['IonMap'] = os.path.join(graphicsPath, item['Name'] + '_ionMap.' + dataset.Attributes['figureFormat'])
        saveAs = item['IonMap']
    else:
        print('Figure 10: Ion map of all features (coloured by log median intensity).')

    plotIonMap(dataset,
               savePath=saveAs,
               figureFormat=dataset.Attributes['figureFormat'],
               dpi=dataset.Attributes['dpi'],
               figureSize=dataset.Attributes['figureSize'])

    # Write HTML if saving
    ##
    if output:
        # Make paths for graphics local not absolute for use in the HTML.
        for key in item:
            if os.path.join(output, 'graphics') in str(item[key]):
                item[key] = re.sub('.*graphics', 'graphics', item[key])

        # Generate report
        from jinja2 import Environment, FileSystemLoader

        env = Environment(loader=FileSystemLoader(os.path.join(toolboxPath(), 'Templates')))
        template = env.get_template('MS_FeatureSummaryReport.html')
        filename = os.path.join(output, dataset.name + '_report_featureSummary.html')

        f = open(filename, 'w')
        f.write(template.render(item=item,
                                attributes=dataset.Attributes,
                                version=version,
                                graphicsPath=graphicsPath))
        f.close()

        copyBackingFiles(toolboxPath(), output)

    return None


def _featureSelectionReport(dataset, output=None, withArtifactualFiltering=False):
    """
    Report on feature quality
    Generates a summary of the number of features passing feature selection (with current settings as definite in the SOP), and a heatmap showing how this number would be affected by changes to RSD and correlation to dilution thresholds.
    """
    
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
    item['ReportType'] = 'feature summary'
    item['Nfeatures'] = dataset.intensityData.shape[1]
    item['Nsamples'] = dataset.intensityData.shape[0]
    item['SScount'] = str(sum(SSmask))
    item['SPcount'] = str(sum(SPmask))
    item['ERcount'] = str(sum(ERmask))
    item['LRcount'] = str(sum(LRmask))
    item['corrMethod'] = dataset.Attributes['corrMethod']
    
    
    ##
    # Report stats
    ##
    if output is not None:
        if not os.path.exists(output):
            os.makedirs(output)
        if not os.path.exists(os.path.join(output, 'graphics')):
            os.makedirs(os.path.join(output, 'graphics'))
        graphicsPath = os.path.join(output, 'graphics', 'report_featureSelectionSummary')
        if not os.path.exists(graphicsPath):
            os.makedirs(graphicsPath)
    else:
        graphicsPath = None

    # Feature selection parameters and numbers passing
    
    # rsdSP <= rsdSS
    rsdSS = rsd(dataset.intensityData[SSmask, :])
    item['rsdSPvsSSvarianceRatio'] = str(dataset.Attributes['varianceRatio'])
    item['rsdSPvsSSPassed'] = sum((dataset.rsdSP * dataset.Attributes['varianceRatio']) <= rsdSS)

    # Correlation to dilution
    item['corrMethod'] = dataset.Attributes['corrMethod']
    item['corrThreshold'] = dataset.Attributes['corrThreshold']
    if sum(dataset.corrExclusions) != dataset.noSamples:
        item['corrExclusions'] = str(
            dataset.sampleMetadata.loc[dataset.corrExclusions == False, 'Sample File Name'].values)
    else:
        item['corrExclusions'] = 'none'
    item['corrPassed'] = sum(dataset.correlationToDilution >= dataset.Attributes['corrThreshold'])

    # rsdSP
    item['rsdThreshold'] = dataset.Attributes['rsdThreshold']
    item['rsdPassed'] = sum(dataset.rsdSP <= dataset.Attributes['rsdThreshold'])

    # Artifactual filtering
    passMask = (dataset.correlationToDilution >= dataset.Attributes['corrThreshold']) & (
                dataset.rsdSP <= dataset.Attributes['rsdThreshold']) & (
                           (dataset.rsdSP * dataset.Attributes['varianceRatio']) <= rsdSS) & (dataset.featureMask == True)
    if withArtifactualFiltering:
        passMask = dataset.artifactualFilter(featMask=passMask)

    if 'blankThreshold' in dataset.Attributes.keys():
        from ..utilities._filters import blankFilter

        blankThreshold = dataset.Attributes['blankThreshold']

        blankMask = blankFilter(dataset)

        passMask = numpy.logical_and(passMask, blankMask)

        item['BlankPassed'] = sum(blankMask)

    if withArtifactualFiltering:
        item['artifactualPassed'] = sum(passMask)
    item['featuresPassed'] = sum(passMask)

    # Heatmap of the number of features passing selection with different RSD and correlation to dilution thresholds
    rsdVals = numpy.arange(5, 55, 5)
    rVals = numpy.arange(0.5, 1.01, 0.05)
    rValsRep = numpy.tile(numpy.arange(0.5, 1.01, 0.05), [1, len(rsdVals)])
    rsdValsRep = numpy.reshape(numpy.tile(numpy.arange(5, 55, 5), [len(rVals), 1]), rValsRep.shape, order='F')
    featureNos = numpy.zeros(rValsRep.shape, dtype=numpy.int)
    if withArtifactualFiltering:
        # with blankThreshold in heatmap
        if 'blankThreshold' in dataset.Attributes.keys():
            for rsdNo in range(rValsRep.shape[1]):
                featureNos[0, rsdNo] = sum(dataset.artifactualFilter(featMask=(
                            (dataset.correlationToDilution >= rValsRep[0, rsdNo]) & (
                                dataset.rsdSP <= rsdValsRep[0, rsdNo]) & (
                                        (dataset.rsdSP * dataset.Attributes['varianceRatio']) <= rsdSS) & (
                                        dataset.featureMask == True) & (blankMask == True))))
        # without blankThreshold
        else:
            for rsdNo in range(rValsRep.shape[1]):
                featureNos[0, rsdNo] = sum(dataset.artifactualFilter(featMask=(
                            (dataset.correlationToDilution >= rValsRep[0, rsdNo]) & (
                                dataset.rsdSP <= rsdValsRep[0, rsdNo]) & (
                                        (dataset.rsdSP * dataset.Attributes['varianceRatio']) <= rsdSS) & (
                                        dataset.featureMask == True))))
    else:
        # with blankThreshold in heatmap
        if 'blankThreshold' in dataset.Attributes.keys():
            for rsdNo in range(rValsRep.shape[1]):
                featureNos[0, rsdNo] = sum(
                    (dataset.correlationToDilution >= rValsRep[0, rsdNo]) & (dataset.rsdSP <= rsdValsRep[0, rsdNo]) & (
                                (dataset.rsdSP * dataset.Attributes['varianceRatio']) <= rsdSS) & (
                                dataset.featureMask == True) & (blankMask == True))
        # without blankThreshold
        else:
            for rsdNo in range(rValsRep.shape[1]):
                featureNos[0, rsdNo] = sum(
                    (dataset.correlationToDilution >= rValsRep[0, rsdNo]) & (dataset.rsdSP <= rsdValsRep[0, rsdNo]) & (
                                (dataset.rsdSP * dataset.Attributes['varianceRatio']) <= rsdSS) & (
                                dataset.featureMask == True))
    test = pandas.DataFrame(data=numpy.transpose(numpy.concatenate([rValsRep, rsdValsRep, featureNos])),
                            columns=['Correlation to dilution', 'RSD', 'nFeatures'])
    test = test.pivot('Correlation to dilution', 'RSD', 'nFeatures')

    fig, ax = plt.subplots(1, figsize=dataset.Attributes['figureSize'], dpi=dataset.Attributes['dpi'])
    sns.heatmap(test, annot=True, fmt='g', cbar=False)
    plt.tight_layout()

    if output:
        item['NoFeaturesHeatmap'] = os.path.join(graphicsPath,
                                                 item['Name'] + '_noFeatures.' + dataset.Attributes['figureFormat'])
        plt.savefig(item['NoFeaturesHeatmap'], format=dataset.Attributes['figureFormat'], dpi=dataset.Attributes['dpi'])
        plt.close()

    else:
        print(
            'Heatmap of the number of features passing selection with different Residual Standard Deviation (RSD) and correlation to dilution thresholds')
        plt.show()

        print('Summary of current feature filtering parameters and number of features passing at each stage\n')
        print('Number of features in original dataset: ' + str(item['Nfeatures']) + '\n\n' +
              'Features filtered on:\n' +
              'Correlation (' + item['corrMethod'] + ', exclusions: ' + item[
                  'corrExclusions'] + ') to dilution greater than ' + str(item['corrThreshold']) + ': ' + str(
            item['corrPassed']) + ' passed selection\n' +
              'Relative Standard Deviation (RSD) in study pool (SP) samples below ' + str(
            item['rsdThreshold']) + ': ' + str(item['rsdPassed']) + ' passed selection\n' +
              'RSD in study samples (SS) * ' + item['rsdSPvsSSvarianceRatio'] + ' >= RSD in SP samples: ' + str(
            item['rsdSPvsSSPassed']) + ' passed selection')
        if blankThreshold:
            print('%i features above blank threshold.' % (item['BlankPassed']))
        if withArtifactualFiltering:
            print('Artifactual features filtering: ' + str(item['artifactualPassed']) + ' passed selection')
        print('\nTotal number of features after filtering: ' + str(item['featuresPassed']))

    # Write HTML if saving
    ##
    if output:
        # Make paths for graphics local not absolute for use in the HTML.
        for key in item:
            if os.path.join(output, 'graphics') in str(item[key]):
                item[key] = re.sub('.*graphics', 'graphics', item[key])

        # Generate report
        from jinja2 import Environment, FileSystemLoader

        env = Environment(loader=FileSystemLoader(os.path.join(toolboxPath(), 'Templates')))
        template = env.get_template('MS_FeatureSelectionReport.html')
        filename = os.path.join(output, dataset.name + '_report_featureSelectionSummary.html')

        f = open(filename, 'w')
        f.write(template.render(item=item,
                                attributes=dataset.Attributes,
                                version=version,
                                graphicsPath=graphicsPath))
        f.close()

        copyBackingFiles(toolboxPath(), output)

    return None


def _batchCorrectionAssessmentReport(dataset, output=None, batch_correction_window=11):
    """
    Generates a report before batch correction showing TIC overall and intensity and batch correction fit for a subset of features, to aid specification of batch start and end points.
    """

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
    item['ReportType'] = 'feature summary'
    item['Nfeatures'] = dataset.intensityData.shape[1]
    item['Nsamples'] = dataset.intensityData.shape[0]
    item['SScount'] = str(sum(SSmask))
    item['SPcount'] = str(sum(SPmask))
    item['ERcount'] = str(sum(ERmask))
    item['LRcount'] = str(sum(LRmask))
    item['corrMethod'] = dataset.Attributes['corrMethod']

    ##
    # Report stats
    ##
    if output is not None:
        if not os.path.exists(output):
            os.makedirs(output)
        if not os.path.exists(os.path.join(output, 'graphics')):
            os.makedirs(os.path.join(output, 'graphics'))
        graphicsPath = os.path.join(output, 'graphics', 'report_batchCorrectionAssessment')
        if not os.path.exists(graphicsPath):
            os.makedirs(graphicsPath)
    else:
        graphicsPath = None
        saveAs = None


    # Pre-correction report (report is example of results when batch correction applied)

    # Check inputs
    if not hasattr(dataset.sampleMetadata, 'Correction Batch'):
        raise ValueError("Correction Batch information missing, run addSampleInfo(descriptionFormat=\'Batches\')")

    # Figure 1: TIC for all samples by sample type and detector voltage change
    if output:
        item['TICdetectorBatches'] = os.path.join(graphicsPath, item['Name'] + '_TICdetectorBatches.' + dataset.Attributes[
            'figureFormat'])
        saveAs = item['TICdetectorBatches']
    else:
        print('Overall Total Ion Count (TIC) for all samples and features, coloured by batch.')

    plotTIC(dataset,
            addViolin=True,
            addBatchShading=True,
            savePath=saveAs,
            figureFormat=dataset.Attributes['figureFormat'],
            dpi=dataset.Attributes['dpi'],
            figureSize=dataset.Attributes['figureSize'])

    # Remaining figures: Sample of fits for selection of features
    (preData, postData, maskNum) = batchCorrectionTest(dataset, nFeatures=10, window=batch_correction_window)
    item['NoBatchPlotFeatures'] = len(maskNum)

    if output:
        figuresCorrectionExamples = OrderedDict()  # To save figures
    else:
        print(
            'Example batch correction plots for a subset of features, results of batch correction with specified batches.')
        figuresCorrectionExamples = None

    for feature in range(len(maskNum)):

        featureName = str(numpy.squeeze(preData.featureMetadata.loc[feature, 'Feature Name'])).replace('/', '-')
        if output:
            figuresCorrectionExamples['Feature ' + featureName] = os.path.join(graphicsPath, item[
                'Name'] + '_batchPlotFeature_' + featureName + '.' + dataset.Attributes['figureFormat'])
            saveAs = figuresCorrectionExamples['Feature ' + featureName]
        else:
            print('Feature ' + featureName)

        plotBatchAndROCorrection(preData,
                                 postData,
                                 feature,
                                 logy=True,
                                 savePath=saveAs,
                                 figureFormat=dataset.Attributes['figureFormat'],
                                 dpi=dataset.Attributes['dpi'],
                                 figureSize=dataset.Attributes['figureSize'])

    if figuresCorrectionExamples is not None:
        # Make paths for graphics local not absolute for use in the HTML.
        for key in figuresCorrectionExamples:
            if os.path.join(output, 'graphics') in str(figuresCorrectionExamples[key]):
                figuresCorrectionExamples[key] = re.sub('.*graphics', 'graphics', figuresCorrectionExamples[key])
        # Save to item
        item['figuresCorrectionExamples'] = figuresCorrectionExamples

    # Write HTML if saving
    ##
    if output:
        # Make paths for graphics local not absolute for use in the HTML.
        for key in item:
            if os.path.join(output, 'graphics') in str(item[key]):
                item[key] = re.sub('.*graphics', 'graphics', item[key])

        # Generate report
        from jinja2 import Environment, FileSystemLoader

        env = Environment(loader=FileSystemLoader(os.path.join(toolboxPath(), 'Templates')))
        template = env.get_template('MS_BatchCorrectionAssessmentReport.html')
        filename = os.path.join(output, dataset.name + '_report_batchCorrectionAssessment.html')

        f = open(filename, 'w')
        f.write(template.render(item=item,
                                attributes=dataset.Attributes,
                                version=version,
                                graphicsPath=graphicsPath))
        f.close()

        copyBackingFiles(toolboxPath(), output)

    return None


def _batchCorrectionSummaryReport(dataset, correctedDataset, output=None):
    """
    Generates a report post batch correction with pertinant figures (TIC, RSD etc.) before and after.
    """

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
    item['ReportType'] = 'feature summary'
    item['Nfeatures'] = dataset.intensityData.shape[1]
    item['Nsamples'] = dataset.intensityData.shape[0]
    item['SScount'] = str(sum(SSmask))
    item['SPcount'] = str(sum(SPmask))
    item['ERcount'] = str(sum(ERmask))
    item['LRcount'] = str(sum(LRmask))
    item['corrMethod'] = dataset.Attributes['corrMethod']

    ##
    # Report stats
    ##
    if output is not None:
        if not os.path.exists(output):
            os.makedirs(output)
        if not os.path.exists(os.path.join(output, 'graphics')):
            os.makedirs(os.path.join(output, 'graphics'))
        graphicsPath = os.path.join(output, 'graphics', 'report_batchCorrectionSummary')
        if not os.path.exists(graphicsPath):
            os.makedirs(graphicsPath)
    else:
        graphicsPath = None
        saveAs = None


    # Mean intensities of Study Pool samples (for future plotting segmented by intensity)
    meanIntensitiesSP = numpy.log(numpy.nanmean(dataset.intensityData[SPmask, :], axis=0))
    meanIntensitiesSP[numpy.mean(dataset.intensityData[SPmask, :], axis=0) == 0] = numpy.nan
    meanIntensitiesSP[numpy.isinf(meanIntensitiesSP)] = numpy.nan

    # Figure 1: Feature intensity histogram for all samples and all features in dataset (by sample type).

    # Pre-correction
    if output:
        item['FeatureIntensityFigurePRE'] = os.path.join(graphicsPath, item['Name'] + '_BCS1_meanIntesityFeaturePRE.' +
                                                         dataset.Attributes['figureFormat'])
        saveAs = item['FeatureIntensityFigurePRE']
    else:
        print('Figure 1: Feature intensity histogram for all samples and all features in dataset (by sample type).')
        print('Pre-correction.')

    _plotAbundanceBySampleType(dataset.intensityData, SSmask, SPmask, ERmask, saveAs, dataset)

    # Post-correction
    if output:
        item['FeatureIntensityFigurePOST'] = os.path.join(graphicsPath, item['Name'] + '_BCS1_meanIntesityFeaturePOST.' +
                                                          dataset.Attributes['figureFormat'])
        saveAs = item['FeatureIntensityFigurePOST']
    else:
        print('Post-correction.')

    _plotAbundanceBySampleType(correctedDataset.intensityData, SSmask, SPmask, ERmask, saveAs, correctedDataset)

    # Figure 2: TIC for all samples and features.

    # Pre-correction
    if output:
        item['TicPRE'] = os.path.join(graphicsPath, item['Name'] + '_BCS2_TicPRE.' + dataset.Attributes['figureFormat'])
        saveAs = item['TicPRE']
    else:
        print('Sample Total Ion Count (TIC) and distribtion (coloured by sample type).')
        print('Pre-correction.')

    plotTIC(dataset,
            addViolin=True,
            title='TIC Pre Batch-Correction',
            savePath=saveAs,
            figureFormat=dataset.Attributes['figureFormat'],
            dpi=dataset.Attributes['dpi'],
            figureSize=dataset.Attributes['figureSize'])

    # Post-correction
    if output:
        item['TicPOST'] = os.path.join(graphicsPath, item['Name'] + '_BCS2_TicPOST.' + dataset.Attributes['figureFormat'])
        saveAs = item['TicPOST']
    else:
        print('Post-correction.')

    plotTIC(correctedDataset,
            addViolin=True,
            title='TIC Post Batch-Correction',
            savePath=saveAs,
            figureFormat=dataset.Attributes['figureFormat'],
            dpi=dataset.Attributes['dpi'],
            figureSize=dataset.Attributes['figureSize'])

    # Figure 3: Histogram of RSD in study pool (SP) samples, segmented by abundance percentiles.

    # Pre-correction
    if output:
        item['RsdByPercFigurePRE'] = os.path.join(graphicsPath, item['Name'] + '_BCS3_rsdByPercPRE.' + dataset.Attributes[
            'figureFormat'])
        saveAs = item['RsdByPercFigurePRE']
    else:
        print(
            'Figure 3: Histogram of Residual Standard Deviation (RSD) in study pool (SP) samples, segmented by abundance percentiles.')
        print('Pre-correction.')

    histogram(dataset.rsdSP,
              xlabel='RSD',
              histBins=dataset.Attributes['histBins'],
              quantiles=dataset.Attributes['quantiles'],
              inclusionVector=numpy.exp(meanIntensitiesSP),
              logx=False,
              xlim=(0, 100),
              savePath=saveAs,
              figureFormat=dataset.Attributes['figureFormat'],
              dpi=dataset.Attributes['dpi'],
              figureSize=dataset.Attributes['figureSize'])

    # Post-correction
    if output:
        item['RsdByPercFigurePOST'] = os.path.join(graphicsPath, item['Name'] + '_BCS3_rsdByPercPOST.' + dataset.Attributes[
            'figureFormat'])
        saveAs = item['RsdByPercFigurePOST']
    else:
        print('Post-correction.')

    histogram(correctedDataset.rsdSP,
              xlabel='RSD',
              histBins=dataset.Attributes['histBins'],
              quantiles=dataset.Attributes['quantiles'],
              inclusionVector=numpy.exp(meanIntensitiesSP),
              logx=False,
              xlim=(0, 100),
              savePath=saveAs,
              figureFormat=dataset.Attributes['figureFormat'],
              dpi=dataset.Attributes['dpi'],
              figureSize=dataset.Attributes['figureSize'])

    # Figure 4: Residual Standard Deviation (RSD) distribution for all samples and all features in dataset (by sample type).

    # Pre-correction
    if output:
        item['RSDdistributionFigurePRE'] = os.path.join(graphicsPath, item['Name'] + '_BCS4_RSDdistributionFigurePRE.' +
                                                        dataset.Attributes['figureFormat'])
        saveAs = item['RSDdistributionFigurePRE']
    else:
        print('Figure 4: RSD distribution for all samples and all features in dataset (by sample type).')
        print('Pre-correction.')

    plotRSDs(dataset,
             ratio=False,
             logx=True,
             color='matchReport',
             savePath=saveAs,
             figureFormat=dataset.Attributes['figureFormat'],
             dpi=dataset.Attributes['dpi'],
             figureSize=dataset.Attributes['figureSize'])

    # Post-correction
    if output:
        item['RSDdistributionFigurePOST'] = os.path.join(graphicsPath, item['Name'] + '_BCS4_RSDdistributionFigurePOST.' +
                                                         dataset.Attributes['figureFormat'])
        saveAs = item['RSDdistributionFigurePOST']
    else:
        print('Post-correction.')

    plotRSDs(correctedDataset,
             ratio=False,
             logx=True,
             color='matchReport',
             savePath=saveAs,
             figureFormat=dataset.Attributes['figureFormat'],
             dpi=dataset.Attributes['dpi'],
             figureSize=dataset.Attributes['figureSize'])

    # Write HTML if saving
    ##
    if output:
        # Make paths for graphics local not absolute for use in the HTML.
        for key in item:
            if os.path.join(output, 'graphics') in str(item[key]):
                item[key] = re.sub('.*graphics', 'graphics', item[key])

        # Generate report
        from jinja2 import Environment, FileSystemLoader

        env = Environment(loader=FileSystemLoader(os.path.join(toolboxPath(), 'Templates')))
        template = env.get_template('MS_BatchCorrectionSummaryReport.html')
        filename = os.path.join(output, dataset.name + '_report_batchCorrectionSummary.html')

        f = open(filename, 'w')
        f.write(template.render(item=item,
                                attributes=dataset.Attributes,
                                version=version,
                                graphicsPath=graphicsPath))
        f.close()

        copyBackingFiles(toolboxPath(), output)

    return None


def _featureCorrelationToDilutionReport(dataset, output=None):
    """
    Generates a more detailed report on correlation to dilution, broken down by batch subset with TIC, detector voltage, a summary, and heatmap indicating potential saturation or other issues.
    """

    # Check inputs
    if not hasattr(dataset.sampleMetadata, 'Correction Batch'):
        raise ValueError("Correction Batch information missing, run addSampleInfo(descriptionFormat=\'Batches\')")


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
    item['ReportType'] = 'feature summary'
    item['Nfeatures'] = dataset.intensityData.shape[1]
    item['Nsamples'] = dataset.intensityData.shape[0]
    item['SScount'] = str(sum(SSmask))
    item['SPcount'] = str(sum(SPmask))
    item['ERcount'] = str(sum(ERmask))
    item['LRcount'] = str(sum(LRmask))
    item['corrMethod'] = dataset.Attributes['corrMethod']

    ##
    # Report stats
    ##
    if output is not None:
        if not os.path.exists(output):
            os.makedirs(output)
        if not os.path.exists(os.path.join(output, 'graphics')):
            os.makedirs(os.path.join(output, 'graphics'))
        graphicsPath = os.path.join(output, 'graphics', 'report_CorrelationToDilutionSummary')
        if not os.path.exists(graphicsPath):
            os.makedirs(graphicsPath)
    else:
        graphicsPath = None
        saveAs = None


    # Generate correlation to dilution for each batch subset - plot TIC and histogram of correlation to dilution

    # generate LRmask
    LRmask = generateLRmask(dataset)

    # instantiate dictionarys
    corLRbyBatch = {}  # to save correlations
    corLRsummary = {}  # summary of number of features with correlation above threshold
    corLRsummary['TotalOriginal'] = len(dataset.featureMask)

    if output:
        saveAs = graphicsPath
        figuresCorLRbyBatch = OrderedDict()  # To save figures
    else:
        figuresCorLRbyBatch = None

    for key in sorted(LRmask):
        corLRbyBatch[key] = _vcorrcoef(dataset.intensityData, dataset.sampleMetadata['Dilution'].values,
                                       method=dataset.Attributes['corrMethod'], sampleMask=LRmask[key])
        corLRsummary[key] = sum(corLRbyBatch[key] >= dataset.Attributes['corrThreshold'])
        figuresCorLRbyBatch = _localLRPlots(dataset,
                                            LRmask[key],
                                            corLRbyBatch[key],
                                            key,
                                            figures=figuresCorLRbyBatch,
                                            savePath=saveAs)

    # Calculate average (mean) correlation across all batch subsets
    corALL = numpy.zeros([len(corLRbyBatch), len(dataset.featureMask)])
    n = 0
    for key in corLRbyBatch:
        corALL[n, :] = corLRbyBatch[key]
        n = n + 1

    corLRbyBatch['MeanAllSubsets'] = numpy.mean(corALL, axis=0)
    corLRsummary['MeanAllSubsets'] = sum(corLRbyBatch['MeanAllSubsets'] >= dataset.Attributes['corrThreshold'])
    figuresCorLRbyBatch = _localLRPlots(dataset,
                                        (dataset.sampleMetadata['SampleType'].values == SampleType.StudyPool) & (
                                                    dataset.sampleMetadata[
                                                        'AssayRole'].values == AssayRole.LinearityReference),
                                        corLRbyBatch['MeanAllSubsets'],
                                        'MeanAllSubsets',
                                        figures=figuresCorLRbyBatch,
                                        savePath=saveAs)

    if figuresCorLRbyBatch is not None:
        # Make paths for graphics local not absolute for use in the HTML.
        for key in figuresCorLRbyBatch:
            if os.path.join(output, 'graphics') in str(figuresCorLRbyBatch[key]):
                figuresCorLRbyBatch[key] = re.sub('.*graphics', 'graphics', figuresCorLRbyBatch[key])
        # Save to item
        item['figuresCorLRbyBatch'] = figuresCorLRbyBatch

    # Summary table of number of features passing threshold with each subset
    temp = pandas.DataFrame(corLRsummary, index=range(1))
    temp['CurrentSettings'] = sum(dataset.correlationToDilution >= dataset.Attributes['corrThreshold'])
    temp = temp.T
    temp.rename(columns={0: 'N Features'}, inplace=True)

    item['NfeaturesSummary'] = temp
    item['corrThreshold'] = str(dataset.Attributes['corrThreshold'])
    item['corrMethod'] = dataset.Attributes['corrMethod']
    if sum(dataset.corrExclusions) != dataset.noSamples:
        item['corrExclusions'] = str(
            dataset.sampleMetadata.loc[dataset.corrExclusions == False, 'Sample File Name'].values)
    else:
        item['corrExclusions'] = 'none'

    if not output:
        print('Number of features exceeding correlation to dilution threshold (' + str(
            item['corrThreshold']) + ') for each LR sample subset/correlation to dilution method')
        display(temp)

        print('\nCurrent correlation settings:' +
              '\nCorrelation method: ' + item['corrMethod'] +
              '\nCorrelation exclusions: ' + item['corrExclusions'] +
              '\nCorrelation threshold: ' + item['corrThreshold'])

    # Assessment of potential saturation

    # Heatmap showing the proportion of features (across different intensities) where median
    # intensity at lower dilution factor >= that at higher dilution factor

    # calculate median feature intensity quantiles and feature masks
    medI = numpy.nanmedian(dataset.intensityData, axis=0)
    quantiles = numpy.percentile(medI, [25, 75])
    nf = dataset.intensityData.shape[1]
    lowImask = medI <= quantiles[0]
    midImask = (medI > quantiles[0]) & (medI <= quantiles[1])
    highImask = medI >= quantiles[1]

    # dilution factors present
    dilutions = (numpy.unique(
        dataset.sampleMetadata['Dilution'].values[~numpy.isnan(dataset.sampleMetadata['Dilution'].values)])).astype(int)
    dilutions.sort()

    # LR batch subsets
    LRbatchmask = generateLRmask(dataset)

    # median feature intensities for different dilution samples
    medItable = numpy.full([nf, len(dilutions) * len(LRbatchmask)], numpy.nan)
    i = 0
    for key in LRbatchmask:
        for d in dilutions:
            mask = (dataset.sampleMetadata['Dilution'].values == d) & (LRbatchmask[key])
            medItable[:, i] = numpy.nanmedian(dataset.intensityData[mask, :], axis=0)
            i = i + 1

    # dataframe for proportion of features with median intensity at lower dilution factor >= that at higher dilution factor
    i = 0
    for key in sorted(LRbatchmask):
        for d in numpy.arange(0, len(dilutions) - 1):
            if 'sat' not in locals():
                sat = pandas.DataFrame({'Average feature intensity': ['1. low ' + key],
                                        'LR': [str(d + 1) + '. ' + str(dilutions[d + 1]) + '<=' + str(dilutions[d])],
                                        'Proportion of features': [
                                            sum(medItable[lowImask, i + 1] <= medItable[lowImask, i]) / sum(
                                                lowImask) * 100]})
                sat = sat.append({'Average feature intensity': '2. medium ' + key,
                                  'LR': str(d + 1) + '. ' + str(dilutions[d + 1]) + '<=' + str(dilutions[d]),
                                  'Proportion of features': sum(
                                      medItable[midImask, i + 1] <= medItable[midImask, i]) / sum(midImask) * 100},
                                 ignore_index=True)
                sat = sat.append({'Average feature intensity': '3. high ' + key,
                                  'LR': str(d + 1) + '. ' + str(dilutions[d + 1]) + '<=' + str(dilutions[d]),
                                  'Proportion of features': sum(
                                      medItable[highImask, i + 1] <= medItable[highImask, i]) / sum(highImask) * 100},
                                 ignore_index=True)
            else:
                sat = sat.append({'Average feature intensity': '1. low ' + key,
                                  'LR': str(d + 1) + '. ' + str(dilutions[d + 1]) + '<=' + str(dilutions[d]),
                                  'Proportion of features': sum(
                                      medItable[lowImask, i + 1] <= medItable[lowImask, i]) / sum(lowImask) * 100},
                                 ignore_index=True)
                sat = sat.append({'Average feature intensity': '2. medium ' + key,
                                  'LR': str(d + 1) + '. ' + str(dilutions[d + 1]) + '<=' + str(dilutions[d]),
                                  'Proportion of features': sum(
                                      medItable[midImask, i + 1] <= medItable[midImask, i]) / sum(midImask) * 100},
                                 ignore_index=True)
                sat = sat.append({'Average feature intensity': '3. high ' + key,
                                  'LR': str(d + 1) + '. ' + str(dilutions[d + 1]) + '<=' + str(dilutions[d]),
                                  'Proportion of features': sum(
                                      medItable[highImask, i + 1] <= medItable[highImask, i]) / sum(highImask) * 100},
                                 ignore_index=True)
            i = i + 1
        i = i + 1

    satHeatmap = sat.pivot('Average feature intensity', 'LR', 'Proportion of features')
    satLineplot = sat.pivot('LR', 'Average feature intensity', 'Proportion of features')

    # plot heatmap
    with sns.axes_style("white"):
        fig = plt.figure(figsize=dataset.Attributes['figureSize'], dpi=dataset.Attributes['dpi'])
        gs = gridspec.GridSpec(1, 11)
        ax1 = plt.subplot(gs[0, :5])
        ax2 = plt.subplot(gs[0, -5:])
        ax1 = sns.heatmap(satHeatmap, ax=ax1, annot=True, fmt='.3g', vmin=0, vmax=100, cmap='Reds', cbar=False)
        ax2 = satLineplot.plot(kind='line', ax=ax2, ylim=[0, 100], colormap='jet')
        if output:
            item['SatFeaturesHeatmap'] = os.path.join(graphicsPath,
                                                      item['Name'] + '_satFeaturesHeatmap.' + dataset.Attributes[
                                                          'figureFormat'])
            plt.savefig(item['SatFeaturesHeatmap'], bbox_inches='tight', format=dataset.Attributes['figureFormat'],
                        dpi=dataset.Attributes['dpi'])
            plt.close()
        else:
            print('\n\nAssessment of potential saturation')
            print(
                '\nHeatmap/lineplot showing the proportion of features (in different intensity quantiles, low:0-25, medium:25-75, and high:75-100%) where the median intensity at lower dilution factors >= that at higher dilution factors')
            plt.show()

    # Write HTML if saving
    ##
    if output:
        # Make paths for graphics local not absolute for use in the HTML.
        for key in item:
            if os.path.join(output, 'graphics') in str(item[key]):
                item[key] = re.sub('.*graphics', 'graphics', item[key])

        # Generate report
        from jinja2 import Environment, FileSystemLoader

        env = Environment(loader=FileSystemLoader(os.path.join(toolboxPath(), 'Templates')))
        template = env.get_template('MS_CorrelationToDilutionReport.html')
        filename = os.path.join(output, dataset.name + '_report_correlationToDilutionSummary.html')

        f = open(filename, 'w')
        f.write(template.render(item=item,
                                attributes=dataset.Attributes,
                                version=version,
                                graphicsPath=graphicsPath))
        f.close()

        copyBackingFiles(toolboxPath(), output)

    return None


def _plotAbundanceBySampleType(intensityData, SSmask, SPmask, ERmask, saveAs, dataset):

    # Load toolbox wide color scheme
    if 'sampleTypeColours' in dataset.Attributes.keys():
        sTypeColourDict = copy.deepcopy(dataset.Attributes['sampleTypeColours'])
        for stype in SampleType:
            if stype.name in sTypeColourDict.keys():
                sTypeColourDict[stype] = sTypeColourDict.pop(stype.name)
    else:
        sTypeColourDict = {SampleType.StudySample: 'b', SampleType.StudyPool: 'g', SampleType.ExternalReference: 'r',
                            SampleType.MethodReference: 'm', SampleType.ProceduralBlank: 'c', 'Other': 'grey'}

    meanIntensities = OrderedDict()
    temp = numpy.nanmean(intensityData[SSmask,:], axis=0)
    temp[numpy.isinf(temp)] = numpy.nan
    meanIntensities['Study Sample'] = temp
    colour = [sTypeColourDict[SampleType.StudySample]]
    if sum(SPmask) != 0:
        temp = numpy.nanmean(intensityData[SPmask,:], axis=0)
        temp[numpy.isinf(temp)] = numpy.nan
        meanIntensities['Study Pool'] = temp
        colour.append(sTypeColourDict[SampleType.StudyPool])
    if sum(ERmask) != 0:
        temp = numpy.nanmean(intensityData[ERmask,:], axis=0)
        temp[numpy.isinf(temp)] = numpy.nan
        meanIntensities['External Reference'] = temp
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


def _localLRPlots(dataset, LRmask, corToLR, saveName, figures=None, savePath=None):
    # Plot TIC
    if savePath:
        saveTemp = saveName + ' LR Sample TIC (coloured by dilution)'
        figures[saveTemp] = os.path.join(savePath, saveTemp + '.' + dataset.Attributes['figureFormat'])
        saveAs = figures[saveTemp]
    else:
        print(saveName + ' LR Sample TIC (coloured by dilution)')
        saveAs = None;

    plotLRTIC(dataset,
              sampleMask=LRmask,
              savePath=saveAs,
              figureFormat=dataset.Attributes['figureFormat'],
              dpi=dataset.Attributes['dpi'],
              figureSize=dataset.Attributes['figureSize'])

    # Plot TIC detector voltage change
    if savePath:
        saveTemp = saveName + ' LR Sample TIC (coloured by change in detector voltage)'
        figures[saveTemp] = os.path.join(savePath, saveTemp + '.' + dataset.Attributes['figureFormat'])
        saveAs = figures[saveTemp]
    else:
        print(saveName + ' LR Sample TIC (coloured by change in detector voltage)')
        saveAs = None;

    plotLRTIC(dataset,
              sampleMask=LRmask,
              colourByDetectorVoltage=True,
              savePath=saveAs,
              figureFormat=dataset.Attributes['figureFormat'],
              dpi=dataset.Attributes['dpi'],
              figureSize=dataset.Attributes['figureSize'])

    # Plot histogram of correlation to dilution
    if savePath:
        saveTemp = saveName + ' Histogram of Correlation To Dilution'
        figures[saveTemp] = os.path.join(savePath, saveTemp + '.' + dataset.Attributes['figureFormat'])
        saveAs = figures[saveTemp]
    else:
        print(saveName + ' Histogram of Correlation To Dilution')
        saveAs = None

    histogram(corToLR,
              xlabel='Correlation to Dilution',
              histBins=dataset.Attributes['histBins'],
              savePath=saveAs,
              figureFormat=dataset.Attributes['figureFormat'],
              dpi=dataset.Attributes['dpi'],
              figureSize=dataset.Attributes['figureSize'])

    if figures is not None:
        return figures


def batchCorrectionTest(dataset, nFeatures=10, window=11):
    import copy
    import numpy
    import random
    from ..batchAndROCorrection._batchAndROCorrection import _batchCorrection

    # Samplemask
    SSmask = (dataset.sampleMetadata['SampleType'].values == SampleType.StudySample) & (
                dataset.sampleMetadata['AssayRole'].values == AssayRole.Assay)
    SPmask = (dataset.sampleMetadata['SampleType'].values == SampleType.StudyPool) & (
                dataset.sampleMetadata['AssayRole'].values == AssayRole.PrecisionReference)
    ERmask = (dataset.sampleMetadata['SampleType'].values == SampleType.ExternalReference) & (
                dataset.sampleMetadata['AssayRole'].values == AssayRole.PrecisionReference)
    LRmask = (dataset.sampleMetadata['SampleType'].values == SampleType.ExternalReference) & (
                dataset.sampleMetadata['AssayRole'].values == AssayRole.LinearityReference)
    sampleMask = (SSmask | SPmask | ERmask | LRmask) & (dataset.sampleMask == True).astype(bool)

    # Select subset of features (passing on correlation to dilution)

    # Correlation to dilution
    if not hasattr(dataset, 'corrMethod'):
        dataset.correlationToDilution

    # Exclude features failing correlation to dilution
    passMask = dataset.correlationToDilution >= dataset.Attributes['corrThreshold']

    # Exclude features with zero values
    zeroMask = sum(dataset.intensityData[sampleMask, :] == 0)
    zeroMask = zeroMask == 0

    passMask = passMask & zeroMask

    # Select subset of features on which to perform batch correction
    maskNum = [i for i, x in enumerate(passMask) if x]
    random.shuffle(maskNum)

    # Do batch correction
    featureList = []
    correctedData = numpy.zeros([dataset.intensityData.shape[0], nFeatures])
    fits = numpy.zeros([dataset.intensityData.shape[0], nFeatures])
    featureIX = 0
    parameters = dict()
    parameters['window'] = window
    parameters['method'] = 'LOWESS'
    parameters['align'] = 'median'

    for feature in maskNum:
        correctedP = _batchCorrection(dataset.intensityData[:, feature],
                                      dataset.sampleMetadata['Run Order'].values,
                                      SPmask,
                                      dataset.sampleMetadata['Correction Batch'].values,
                                      range(0, 1),  # All features
                                      parameters,
                                      0)

        if sum(numpy.isfinite(correctedP[0][1])) == dataset.intensityData.shape[0]:
            correctedData[:, featureIX] = correctedP[0][1]
            fits[:, featureIX] = correctedP[0][2]
            featureList.append(feature)
            featureIX = featureIX + 1

        if featureIX == nFeatures:
            break

    # Create copy of dataset and trim
    preData = copy.deepcopy(dataset)
    preData.intensityData = dataset.intensityData[:, featureList]
    preData.featureMetadata = dataset.featureMetadata.loc[featureList, :]
    preData.featureMetadata.reset_index(drop=True, inplace=True)

    # Run batch correction
    postData = copy.deepcopy(preData)
    postData.intensityData = correctedData
    postData.fit = fits

    # Return results
    return preData, postData, featureList
