import os
import re
from collections import OrderedDict

from ..plotting import plotScores, plotLoadings
from pyChemometrics.ChemometricsPCA import ChemometricsPCA


def generateBasicPCAReport(pcaModel, dataset, figureCounter=1, destinationPath=None, fileNamePrefix=''):
    """
	Visualise a PCA model by plotting scores plots coloured by sample type, loadings plots, 

	:param ChemometricsPCA pcaModel: ChemometricsPCA model of **dataset**
	:param Dataset dataset: Dataset object
	:param int figureCounter: Start numbering figures from this value
	:param destinationPath: If not ``None`` save to the path specified
	:type destinationPath: None or str
	:param str fileNamePrefix: Additional prefix to add to filenames if saving
	:returns: Dictionary of paths figures where saved to if destinationPath was not ``None``
	:rtype: dict
	"""
    if not isinstance(pcaModel, ChemometricsPCA):
        raise TypeError('pcaModel must be a ChemometricsPCA object')

    returnDict = dict()

    figuresQCscores = OrderedDict()

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

    ##
    # Scores Plot
    ##
    if destinationPath:
        saveAs = os.path.join(graphicsPath, dataset.name + '_PCAscoresPlot_')
    else:
        print('Figure %i: PCA scores plots coloured by sample class.' % (figureCounter))
        saveAs = None

    figuresQCscores = plotScores(pcaModel,
                                 classes=dataset.sampleMetadata['SampleClass'],
                                 colourType='categorical',
                                 title='SampleClass',
                                 colourDict=dataset.Attributes['sampleTypeColours'],
                                 markerDict=dataset.Attributes['sampleTypeMarkers'],
                                 savePath=saveAs,
                                 figures=figuresQCscores,
                                 figureFormat=dataset.Attributes['figureFormat'],
                                 dpi=dataset.Attributes['dpi'],
                                 figureSize=dataset.Attributes['figureSize'])

    for keyS in figuresQCscores:
        if 'graphics' in str(graphicsPath):  # expect graphics to have been already passed in the previous path
            figuresQCscores[keyS] = re.sub('.*graphics', 'graphics', figuresQCscores[keyS])

    returnDict['QCscores'] = figuresQCscores
    returnDict['PCAcount'] = figureCounter

    figuresLoadings = OrderedDict()

    ##
    # Loadings plot
    ##
    if destinationPath:
        saveAs = os.path.join(graphicsPath, dataset.name + '_PCAloadingsPlot_')
    else:
        print('\n\nFigure %i: PCA loadings plots.' % (figureCounter + 1))
        saveAs = None

    figuresLoadings = plotLoadings(pcaModel,
                                   dataset,
                                   title='',
                                   figures=figuresLoadings,
                                   savePath=saveAs,
                                   figureFormat=dataset.Attributes['figureFormat'],
                                   dpi=dataset.Attributes['dpi'],
                                   figureSize=dataset.Attributes['figureSize'])

    for keyL in figuresLoadings:
        if 'graphics' in str(graphicsPath):
            figuresLoadings[keyL] = re.sub('.*graphics', 'graphics', figuresLoadings[keyL])

    returnDict['loadings'] = figuresLoadings

    if destinationPath:
        return returnDict
    else:
        return None
