import os
import re
from collections import OrderedDict

from ..plotting import plotScores, plotLoadings
from pyChemometrics.ChemometricsPCA import ChemometricsPCA

def generateBasicPCAReport(pcaModel, dataset, figureCounter=1, output=None, fileNamePrefix=''):
	"""
	Visualise a PCA model by plotting scores plots coloured by sample type, loadings plots, 

	:param ChemometricsPCA pcaModel: ChemometricsPCA model of **dataset**
	:param Dataset dataset: Dataset object
	:param int figureCounter: Start numbering figures from this value
	:param output: If not ``None`` save to the path specified
	:type output: None or str
	:param str fileNamePrefix: Additional prefix to add to filenames if saving
	:returns: Dictionary of paths figures where saved to if output was not ``None``
	:rtype: dict
	"""
	if not isinstance(pcaModel, ChemometricsPCA):
		raise TypeError('pcaModel must be a ChemometricsPCA object')

	returnDict = dict()

	figuresQCscores = OrderedDict()

	##
	# Scores Plot
	##
	if output:
		saveAs = os.path.join(output, dataset.name + '_PCAscoresPlot_')
	else:
		print('Figure %i: PCA scores plots coloured by sample type.' % (figureCounter))
		saveAs = None

	figuresQCscores = plotScores(pcaModel,
								 classes=dataset.sampleMetadata['Plot Sample Type'],
								 classType='Plot Sample Type',
								 title='Sample Type',
								 savePath=saveAs,
								 figures=figuresQCscores,
								 figureFormat=dataset.Attributes['figureFormat'],
								 dpi=dataset.Attributes['dpi'],
								 figureSize=dataset.Attributes['figureSize'])

	for keyS in figuresQCscores:
		if 'graphics' in str(output): # expect graphics to have been already passed in the previous path
			if str(output) in str(figuresQCscores[keyS]):
				figuresQCscores[keyS] = re.sub('.*graphics', 'graphics', figuresQCscores[keyS])

	returnDict['QCscores'] = figuresQCscores
	returnDict['PCAcount'] = figureCounter

	figuresLoadings = OrderedDict()

	##
	# Loadings plot
	##
	if output:
		saveAs = os.path.join(output, dataset.name + '_PCAloadingsPlot_')
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
		if 'graphics' in str(output):
			if str(output) in str(figuresLoadings[keyL]):
				figuresLoadings[keyL] = re.sub('.*graphics', 'graphics', figuresLoadings[keyL])

	returnDict['loadings'] = figuresLoadings

	if output:
		return returnDict
	else:
		return None
