import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import matplotlib.patches as mpatches
import seaborn as sns
import numpy
import pandas
import copy
from ..enumerations import VariableType, SampleType, AssayRole


def plotVariableScatter(inputTable, logX=False, xLim=None, xLabel='', yLabel='', sampletypeColor=False, hLines=None, hLineStyle='-', hBox=None, vLines=None, vLineStyle=':', vBox=None, savePath=None, figureFormat='png', dpi=72, figureSize=(11 ,7)):
    """
    Plot values on x-axis, with ordering on the y-axis.
    Entries as rows are placed on the x-axis, values of all columns are plotted on y-axis with different colors.
    If sampletypeColor=True, only columns named as SampleTypes will be plotted and colored according to other reports, otherwise all columns are plotted.
    Ordering of the rows is conserved, the first item is placed at the top of the y-axis and the last row is at the bottom.
    If a column ['yName'] is present, it is employed to label each y-axis entry.

    :param dataframe inputTable: DataFrame or accuracy or precision values, with features as rows and sample types as columns (['Study Sample', 'Study Pool', 'External Reference', 'All Samples', 'nan']). A 'yName' column can be present to display the feature name.
    :param bool logX: If ``True`` plot values on a log10 scaled x axis
    :param xLim: Tuple of (min, max) values to plot
    :type xLim: None or tuple(float, float)
    :param str xLabel: X-axis label
    :param str yLabel: Y-axis label
    :param bool sampletypeColor: If ``True`` only the sampleType columns are plotted with colors matching other reports
    :param hLines: None or list of y positions at which to plot an horizontal line. Features are positioned from 1 to nFeat
    :type hLines: None or list
    :param str hLineStyle: One of the axhline linestyle ('-', '--', '-.', ':')
    :param hBox: None or list of tuple of y positions defining horizontal box. Features are positioned from 1 to nFeat
    :type hBox: None or list
    :param vLines: None or list of v positions at which to plot an vertical line. Unit is the same as the v axis.
    :type vLines: None or list
    :param str vLineStyle: One of the axvline linestyle ('-', '--', '-.', ':')
    :param vBox: None or list of tuple of x positions defining horizontal box. Features are positioned from 1 to nFeat
    :type vBox: None or list
    :param color: Allows the default colour pallet to be overridden
    :type color: None or seaborn.palettes._ColorPalette
    :param savePath: If ``None`` plot interactively, otherwise save the figure to the path specified
    :type savePath: None or str
    """

    ## Checks
    if xLim is not None:
        if not isinstance(xLim, tuple):
            raise TypeError('xLim must be \'None\' or tuple(float, float)')
    if not isinstance(xLabel, str):
        raise TypeError('xLabel must be a str')
    if not isinstance(yLabel, str):
        raise TypeError('yLabel must be a str')
    if hLines is not None:
        if not isinstance(hLines, list):
            raise TypeError('hLines must be \'None\' or list')
    if vLines is not None:
        if not isinstance(vLines, list):
            raise TypeError('vLines must be \'None\' or list')
    if hLineStyle not in ['-', '--', '-.', ':']:
        raise ValueError('hLines must be one of the matplotlib axhline linestyle (\'-\', \'--\', \'-.\', \':\')')
    if vLineStyle not in ['-', '--', '-.', ':']:
        raise ValueError('vLines must be one of the matplotlib axvline linestyle (\'-\', \'--\', \'-.\', \':\')')
    if hBox is not None:
        if not isinstance(hBox, list):
            raise TypeError('hBox must be \'None\' or list')
        if not isinstance(hBox[0], tuple):
            raise TypeError('hBox must be a list of tuple')
    if vBox is not None:
        if not isinstance(vBox, list):
            raise TypeError('vBox must be \'None\' or list')
        if not isinstance(vBox[0], tuple):
            raise TypeError('vBox must be a list of tuple')

    ## Init
    sns.set_style("ticks", {'axes.linewidth': 0.75})
    fig = plt.figure(figsize=figureSize, dpi=dpi)
    ax = plt.subplot(1, 1, 1)
    current_palette = sns.color_palette()

    data = copy.deepcopy(inputTable)
    data.reset_index(drop=True, inplace=True)

    # reorder if needed and get a y position
    minY = 1
    maxY = data.shape[0]
    data['yPos'] = list(reversed(range(minY, maxY+ 1)))

    # Register +/-numpy.inf, all values for min/max
    infNegY = []
    infPosY = []
    allVal = []

    # color iterator and color scheme
    colorIdx = 1
    if sampletypeColor:
        ## Try loading toolbox wide color scheme
        # value just in case
        sTypeColourDict = {SampleType.StudySample: 'b', SampleType.StudyPool: 'g', SampleType.ExternalReference: 'r',
                           SampleType.MethodReference: 'm', SampleType.ProceduralBlank: 'c', 'Other': 'grey'}
        # load from the SOP as we do not have access to object
        try:
            from .._toolboxPath import toolboxPath
            import json
            import os

            with open(os.path.join(toolboxPath(), 'StudyDesigns', 'SOP', 'Generic.json')) as data_file:
                attributes = json.load(data_file)
            # convert key names to SampleType enum
            if 'sampleTypeColours' in attributes.keys():
                sTypeColourDict = copy.deepcopy(attributes['sampleTypeColours'])
                for stype in SampleType:
                    if stype.name in sTypeColourDict.keys():
                        sTypeColourDict[stype] = sTypeColourDict.pop(stype.name)
        except:
            pass
        # add cases specific to this plot
        sTypeColourDict['All Samples'] = 'black'
        sTypeColourDict['nan'] = 'grey'


    # Plot each columns of data
    # columns of interest
    if sampletypeColor:
        # match the subset of columns while imposing the sampleType ordering (can't be done with a set)
        expectedCols = pandas.DataFrame({'sType': [SampleType.StudySample, SampleType.ExternalReference, 'nan', 'All Samples', SampleType.StudyPool]})
        workingCols = expectedCols['sType'][expectedCols.sType.isin(inputTable.columns.tolist())].tolist()
    else:
        workingCols = list(set(inputTable.columns.tolist()) - set(['yName']))

    # plot each column, store +/-inf for later plot
    for wcol in workingCols:
        # change plot alpha and linewidth
        if sampletypeColor:
            if wcol == SampleType.StudyPool:
                lwPlot = 1
                alphaPlot = 1
            else:
                lwPlot = 1
                alphaPlot = 0.5
            currentColor = sTypeColourDict[wcol]
        else:
            lwPlot = 1
            alphaPlot = 1
            currentColor = current_palette[colorIdx]

        # only plot values, no inf or nan
        valueMask = numpy.isfinite(data[wcol].tolist()).tolist()
        tmpX = pandas.DataFrame({'x': data.loc[valueMask, wcol].tolist()})
        tmpY = pandas.DataFrame({'y': data.loc[valueMask, 'yPos'].tolist()})
        pt = ax.scatter(x=tmpX['x'], y=tmpY['y'], alpha=alphaPlot, lw=lwPlot, c=currentColor, label=wcol)
        colorIdx += 1

        # store position of inf
        infNegY.extend(data.loc[(data[wcol] == -numpy.inf).tolist(), 'yPos'])
        infPosY.extend(data.loc[(data[wcol] == numpy.inf).tolist(), 'yPos'])
        allVal.extend(data.loc[valueMask, wcol])

    # Plot a marker for -inf/+inf
    minX = min(allVal)
    maxX = max(allVal)
    infNegX = [minX] * len(infNegY)
    infPosX = [maxX] * len(infPosY)
    infX = infNegX + infPosX
    infY = infNegY + infPosY
    ax.scatter(x=infX, y=infY, marker='X', c='white', linewidth=1, edgecolor='black')

    # Vertical lines
    if vLines is not None:
        for vlY in vLines:
            vline = ax.axvline(x=vlY, ymin=0, ymax=1, linestyle=vLineStyle, color='grey')
            vline.set_zorder(0)

    # Horizontal lines
    if hLines is not None:
        for hlY in hLines:
            hline = ax.axhline(y=hlY, xmin=0, xmax=1, linestyle=hLineStyle, color='grey', lw=0.5, alpha=0.5)
            hline.set_zorder(0)

    # Horizontal Box
    if hBox is not None:
        for hB in hBox:
            p_rect = ax.add_patch(mpatches.Rectangle((minX, hB[0]), maxX, (hB[1] - hB[0]), facecolor='grey', alpha=0.15))  # Rectangle((x,y), width, height)
            p_rect.set_zorder(0)

    # Vertical Box
    if vBox is not None:
        for vB in vBox:
            p_rect = ax.add_patch(mpatches.Rectangle((vB[0], minY-1), (vB[1] - vB[0]), maxY+1, facecolor='grey', alpha=0.15))  # Rectangle((x,y), width, height)
            p_rect.set_zorder(0)


    # Set limits
    xpadding = (maxX - minX) / 100.0
    ypadding = (maxY - minY) / 100.0
    ypadding = numpy.floor(ypadding)

    if xLim:
        ax.set_xlim(xLim)
    else:
        ax.set_xlim((minX - xpadding, maxX + xpadding))

    ax.set_ylim((minY - 1 - ypadding, maxY + 1 + ypadding))

    # Log scale axis
    if logX:
        ax.set_xscale('symlog', nonposy='clip')
        for axis in [ax.xaxis, ax.yaxis]:
            axis.set_major_formatter(ScalarFormatter())

    # Axis and legend
    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)
    if 'yName' in inputTable.columns:
        plt.yticks(data['yPos'], data['yName'])
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    # Save or output
    if savePath:
        plt.savefig(savePath, bbox_inches='tight', format=figureFormat, dpi=dpi)
        plt.close()
    else:
        plt.show()
