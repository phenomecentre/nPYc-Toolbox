import numpy as np
from ..plotting import histogram
from ..utilities import sampleClassMasks
from collections import OrderedDict

def plotAbundanceBySampleType(dataset, saveAs):

    # Define sample type masks for all samples in dataset
    sampleMasks = sampleClassMasks(dataset.sampleMetadata, on='SampleClass')

    # Save mean peakArea for histogram
    meanIntensities = OrderedDict()
    colour = []

    for sType in sampleMasks.keys():

        if sum(sampleMasks[sType]) != 0:
            temp = np.nanmean(dataset.intensityData[sampleMasks[sType],:], axis=0)
            temp[np.isinf(temp)] = np.nan
            meanIntensities[sType] = temp
            colour.append(dataset.Attributes['sampleTypeColours'][sType])

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