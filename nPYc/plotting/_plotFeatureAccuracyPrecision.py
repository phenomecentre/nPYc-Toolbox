import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import seaborn as sns
import numpy
import pandas
import copy
import warnings
from ..enumerations import VariableType, SampleType, AssayRole
from ._plotVariableScatter import plotVariableScatter


def plotAccuracyPrecision(tData, accuracy=True, percentRange=None, savePath=None, figureFormat='png', dpi=72, figureSize=(11, 7)):
    """
    Plot Accuracy or Precision for a TargetedDataset.

    Features at all present concentrations are shown on the y-axis, with accuracy or precision values on the x-axis.
    Accuracy are centered around 100%. If Precision values cover too wide a range, x-axis is log transformed.

    :param TargetedDataset tData: TargetedDataset object to plot
    :param bool accuracy: If ``True`` plot the Accuracy of each measurements, if ``False`` plot the Precision of measurements.
    :param percentRange: If float [0, inf], add a rectangle covering the range of acceptable percentage; for Accuracy 100 +/- percentage, for Precision 0 - percentage.
    :type percentRange: None or float
    :param savePath: If ``None`` plot interactively, otherwise save the figure to the path specified
    :type savePath: None or str
    :param str figureFormat: If saving the plot, use this format
    :param int dpi: Plot resolution
    :param figureSize: Dimensions of the figure
    :type figureSize: tuple(float, float)
    :raises ValueError: if targetedData does not satisfy to the TargetedDataset definition for QC
    :raises ValueError: if percentRange is not 'None' or float
    """

    # Check dataset is fit for plotting
    tmpTData = copy.deepcopy(tData)  # to not log validateObject
    validDataset = tmpTData.validateObject(verbose=False, raiseError=False, raiseWarning=False)
    if not validDataset['QC']:
        raise ValueError('Import Error: tData does not satisfy to the TargetedDataset definition for QC')
    if percentRange is not None:
        if not isinstance(percentRange, (int, float)):
            raise ValueError('Import Error: percentRange must be \'None\' or float')

    # Init
    accPrec = tData.accuracyPrecision()
    if accuracy:
        statistic = accPrec['Accuracy']
    else:
        statistic = accPrec['Precision']

    ## Prepare data (loop over features, append to an output df, remove all rows with NA, define a y-axis)
    # limit to sample types with existing data
    sType = []
    for skey in statistic.keys():
        if statistic[skey].shape[0] != 0:
            sType.append(skey)

    # define output df
    Conc = [str(i) for i in statistic['All Samples'].index]
    nConc = len(Conc)
    nFeat = tData.noFeatures
    featList = tData.featureMetadata['Feature Name'].tolist()
    cols = ['Feat', 'Conc']
    cols.extend(sType)
    statTable = pandas.DataFrame(columns=cols)

    # iterate over features
    horzLines = []
    for i in range(0, nFeat):
        featID = featList[i]
        tmpStatTable = pandas.DataFrame(columns=cols, index=statistic['All Samples'].index)
        # iterate over sample type columns
        for skey in sType:
            # iterate over each concentration applicable for this sample type table
            for measuredConc in statistic[skey].index:
                tmpStatTable.loc[measuredConc, skey] = statistic[skey].loc[measuredConc, featID]
        # remove empty rows and finish off table
        tmpStatTable = tmpStatTable.dropna(how='all')
        tmpStatTable['Feat'] = featID
        tmpStatTable['Conc'] = [str(i) for i in tmpStatTable.index]
        statTable = statTable.append(tmpStatTable, ignore_index=True)
        # list separation line between compounds
        horzLines = [sum([x, tmpStatTable.shape[0]]) for x in horzLines]  # increase y of all previous lines by the number of rows added
        horzLines.append(0.5)
    # Remove the bottom most line
    horzLines = horzLines[:-1]

    # Add a y-id, in reverse order, set a yName column for output
    statTable['yName'] = statTable['Feat'].astype(str) + ' - ' + statTable['Conc']
    statTable.drop('Feat', axis=1, inplace=True)
    statTable.drop('Conc', axis=1, inplace=True)

    ## Define plot
    # It is not possible to plot more than 35 rows clearly, extend the plot height
    extFactor = statTable.shape[0] / 35
    newHeight = figureSize[1] * extFactor
    figsize = (figureSize[0], newHeight)

    # Is there data to plot
    if statTable.shape[0] == 0:
        import warnings
        if accuracy:
            warnings.warn("Warning: no Accuracy values to plot.")
        else:
            warnings.warn("Warning: no Precision values to plot.")

    else:
        # define axis
        if accuracy:
            logx = False
            xL = 'Accuracy (%)'
            vL = [100.]
            hB = None
            if percentRange is not None:
                vB = [(100 - percentRange, 100 + percentRange)]
            else:
                vB = None
        else:
            if numpy.max(statTable.loc[:, sType].values.flatten()[numpy.isfinite(statTable.loc[:, sType].values.flatten().tolist())]) <= 100:
                logx = False
                xL = 'Precision (RSD %)'
            else:
                logx = True
                xL = 'Log Precision (RSD %)'
            vL = None
            hB = None
            if percentRange is not None:
                vB = [(0, percentRange)]
            else:
                vB = None
        yL = 'Feature'
        hL = horzLines

        plotVariableScatter(statTable, logX=logx, xLim=None, sampletypeColor=True, xLabel=xL, yLabel=yL, hLines=hL, hLineStyle='-', hBox=hB, vLines=vL, vLineStyle=':', vBox=vB, savePath=savePath, figureFormat=figureFormat, dpi=dpi, figureSize=figsize)
