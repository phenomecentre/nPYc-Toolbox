import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy
import pandas
import copy
from ..objects._msDataset import MSDataset
from ._violinPlot import _violinPlotHelper
from ..enumerations import AssayRole, SampleType
import matplotlib.dates as mdates
from matplotlib.dates import MO, TU, WE, TH, FR, SA, SU
from matplotlib.dates import WeekdayLocator, DateFormatter, AutoDateFormatter, AutoDateLocator
from matplotlib.ticker import AutoLocator, FormatStrFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Rectangle


# General function to be re-used in TIC and other plots
def _scatterplot(x, y, colourBy=None, ax=None, title='', savePath=None, figureFormat='png', dpi=72, figureSize=(11,7)):
    return None


def _boxplot(x, y, colourBy, ax=None, title='', savePath=None, figureFormat='png', dpi=72, figureSize=(11,7)):
    return None


def plotFeature(dataset, featureName, xAxis='Acquired Time', colourBy=None, colourMap=None, addBatchShading=False, logy=False, title='', withExclusions=True, savePath=None, figureFormat='png', dpi=72, figureSize=(11,7)):
    """
    Visualise Feature intensity across run order. Feature can be colored by continuous or discrete markers. if addBatchShading is True, plot is shaded by 'Batch'.

    :param MSDataset msData: Dataset object
    :param bool addViolin: If ``True`` adds violin plots of TIC distribution pre and post correction split by sample type
    :param bool addBatchShading: If ``True`` shades plot according to sample batch
    :param bool addLineAtGaps: If ``True`` adds line where acquisition time is greater than double the norm
    :param bool colourByDetectorVoltage: If ``True`` colours points by detector voltage, else colours by dilution
    :param bool logy: If ``True`` plot y on a log scale
    :param str title: Title for the plot
    :param bool withExclusions: If ``False``, discard masked features from the sum
    :param savePath: If ``None`` plot interactively, otherwise save the figure to the path specified
    :type savePath: None or str
    :param str figureFormat: If saving the plot, use this format
    :param int dpi: Plot resolution
    :param figureSize: Dimensions of the figure
    :type figureSize: tuple(float, float)
    """

    # Check inputs
    if addBatchShading & (xAxis == 'Batch'):
        raise ValueError('addBatchShading=True option incompatible with xAxis=\'Batch\'')

    featureIndex = numpy.where(dataset.featureMetadata['Feature Name'] == featureName)[0]
    featureIntensity = dataset.intensityData[:, featureIndex].squeeze()

    if title is None:
        title = "Feature Name: {0}".format(featureName)

    # Mask prepare excluded features
    if withExclusions:
        dataMask = dataset.sampleMask
    else:
        dataMask = numpy.ones(shape=dataset.sampleMask.shape, dtype=bool)

    # Handle missing data - separate masking behaviour for missing, or truncated
    # NA/Missing data is completely omited from plot
    dataMask[numpy.isnan(featureIntensity)] = False
    # lower and upper truncation have their specific masks
    lowerTruncatedMask = featureIntensity == -numpy.inf
    upperTruncatedMask = featureIntensity == numpy.inf

    # X axis options for plotting
    if xAxis == 'Acquired Time':
        xValues = dataset.sampleMetadata.loc[dataMask, 'Acquired Time']
        minX = numpy.min(xValues)
        maxX = numpy.max(xValues)
        delta = maxX - minX
        days = delta.days

        if days < 7 and days > 2:
            loc = WeekdayLocator(byweekday=(MO, TU, WE, TH, FR, SA, SU))
            formatter = DateFormatter('%d/%m/%y')

        elif days <= 2:
            loc = AutoDateLocator()
            formatter = AutoDateFormatter(loc)

        else:
            loc = WeekdayLocator(byweekday=(MO, SA))
            formatter = DateFormatter('%d/%m/%y')

    elif xAxis == 'Run Order':
        xValues = dataset.sampleMetadata.loc[dataMask, 'Run Order']
        minX = numpy.min(xValues)
        maxX = numpy.max(xValues)
        loc = AutoLocator()
        formatter = FormatStrFormatter('%d')


    elif xAxis == 'Batch':
        xValues = dataset.sampleMetadata.loc[dataMask, 'Correction Batch']

    # Load toolbox wide color scheme
    if 'sampleTypeColours' in dataset.Attributes.keys():
        sTypeColourDict = copy.deepcopy(dataset.Attributes['sampleTypeColours'])
        for stype in SampleType:
            if stype.name in sTypeColourDict.keys():
                sTypeColourDict[stype] = sTypeColourDict.pop(stype.name)
    else:
        sTypeColourDict = {SampleType.StudySample: 'b', SampleType.StudyPool: 'g', SampleType.ExternalReference: 'r',
                           SampleType.MethodReference: 'm', SampleType.ProceduralBlank: 'c', 'Other': 'grey'}

    # Define sample types
    SSmask = ((dataset.sampleMetadata['SampleType'].values == SampleType.StudySample) & (
                dataset.sampleMetadata['AssayRole'].values == AssayRole.Assay)) & dataMask
    SPmask = ((dataset.sampleMetadata['SampleType'].values == SampleType.StudyPool) & (
                dataset.sampleMetadata['AssayRole'].values == AssayRole.PrecisionReference)) & dataMask
    ERmask = ((dataset.sampleMetadata['SampleType'].values == SampleType.ExternalReference) & (
                dataset.sampleMetadata['AssayRole'].values == AssayRole.PrecisionReference)) & dataMask
    LRmask = ((dataset.sampleMetadata['SampleType'].values == SampleType.StudyPool) & (
                dataset.sampleMetadata['AssayRole'].values == AssayRole.LinearityReference)) & dataMask

    # Prepare the figure
    fig, ax = plt.subplots(1, figsize=figureSize, dpi=dpi)
    # If colouring by any variable
    if colourBy is not None:
        colorVector = dataset.sampleMetadata[colourBy]
        if colourMap is None:
            # Check if variable is continuous...
            if numpy.issubdtype(colorVector.dtype, numpy.number) and (len(numpy.unique(colorVector)) > 10):
                # Check if values are centered around 0 and pick a divergent colorscale...
                if sum(colorVector < 0) / colorVector.shape[0] > 0.05:
                    colourMap = plt.cm.get_cmap('RdBu')
                # ... or a standard continuous colorscale otherwise
                else:
                    colourMap = plt.cm.get_cmap('viridis')
                discrete = False
            # Or discrete
            else:
                if len(numpy.unique(colorVector)) <= 10:
                    colourMap = plt.cm.get_cmap('Set1')
                else:
                    colourMap = plt.cm.get_cmap('tab20')
                discrete = True
                # Plot with a colour by argument
        if discrete is False:
            sc = ax.scatter(xValues[dataMask], featureIntensity[dataMask], marker='o', c=colorVector[dataMask],
                            cmap=colourMap, vmin=numpy.min(colorVector), vmax=numpy.max(colorVector), edgecolors='grey')
        else:
            colors = [colourMap(i) for i in range(colorVector.unique().size)]
            names = colorVector.unique()
            for i, (name, color) in enumerate(zip(names, colors), 1):
                categoryMask = dataMask & (dataset.sampleMetadata[colourBy] == name)
                if len(colorVector.unique()) <= 20:
                    currentLabel = name
                else:
                    currentLabel = None
                sc = ax.scatter(xValues[categoryMask], featureIntensity[categoryMask], marker='o', color=color,
                                label=currentLabel, cmap=colourMap, edgecolors='grey')

        # Keep comments here for lodMasking
        # if lowerMask
        # ax.scatter(xValues[dataMask], featureIntensity[dataMask], marker='o', c=colourBy[dataMask], cmap=colourMap, vmin=numpy.min(colourBy), vmax=numpy.max(colourBy), edgecolors='grey')
        # if upperMask
        # ax.scatter(xValues[dataMask], featureIntensity[dataMask], marker='o', c=colourBy[dataMask], cmap=colourMap, vmin=numpy.min(colourBy), vmax=numpy.max(colourBy), edgecolors='grey')

        # Assume colorBy = None Colour by sample type
    else:
        discrete = True
        if sum(SSmask != 0):
            sc = ax.plot_date(xValues[SSmask], featureIntensity[SSmask], c=sTypeColourDict[SampleType.StudySample],
                              fmt='o', ms=6, label='Study Sample')  # c='y',
        if sum(SPmask != 0):
            sc = ax.plot_date(xValues[SPmask], featureIntensity[SPmask], c=sTypeColourDict[SampleType.StudyPool],
                              fmt='v', ms=8, label='Study Reference')  # c='m',
        if sum(ERmask != 0):
            sc = ax.plot_date(xValues[ERmask], featureIntensity[ERmask],
                              c=sTypeColourDict[SampleType.ExternalReference], fmt='^', ms=8,
                              label='Long-Term Reference')
        if sum(LRmask != 0):
            sc = ax.plot_date(xValues[LRmask], featureIntensity[LRmask], c=sTypeColourDict[SampleType.MethodReference],
                              fmt='s', ms=6, label='Serial Dilution')
        # if lowerMask
        # sc = ax.scatter(xValues[dataMask], featureIntensity[dataMask], marker='o', c=colourBy[dataMask], cmap=colourMap, vmin=numpy.min(colourBy), vmax=numpy.max(colourBy), edgecolors='grey')
        # if upperMask
        # sc = ax.scatter(xValues[dataMask], featureIntensity[dataMask], marker='o', c=colourBy[dataMask], cmap=colourMap, vmin=numpy.min(colourBy), vmax=numpy.max(colourBy), edgecolors='grey')

    logy = True
    if logy:
        ax.set_yscale('symlog')
    else:
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    # Shade by automatically defined batches - this is added to the main plot
    if addBatchShading:
        batches = (numpy.unique(dataset.sampleMetadata.loc[dataMask, 'Correction Batch'].values[~numpy.isnan(
            dataset.sampleMetadata.loc[dataMask, 'Correction Batch'].values)])).astype(int)

        # Define the colours (different for each batch) and get axis y-limits
        batchColourMap = plt.get_cmap('gnuplot')
        colors = [batchColourMap(i) for i in numpy.linspace(0, 1, len(batches) + 1)]
        ymin, ymax = ax.get_ylim()

        colIX = 1

        # Add shading for each batch
        for i in batches:
            if xAxis == 'Acquired Time':
                # Create rectangle x coordinates
                start = dataset.sampleMetadata[dataset.sampleMetadata['Run Order'] == min(
                    dataset.sampleMetadata[dataset.sampleMetadata['Correction Batch'].values == i]['Run Order'])][
                    'Acquired Time']
                end = dataset.sampleMetadata[dataset.sampleMetadata['Run Order'] == max(
                    dataset.sampleMetadata[dataset.sampleMetadata['Correction Batch'].values == i]['Run Order'])][
                    'Acquired Time']
                # Convert to matplotlib date representation
                start = mdates.date2num(start)
                end = mdates.date2num(end)
            elif xAxis == 'Run Order':
                # Create rectangle x coordinates
                batchMin = min(
                    dataset.sampleMetadata.loc[dataset.sampleMetadata['Correction Batch'].values == i, 'Run Order'])
                batchMax = max(
                    dataset.sampleMetadata.loc[dataset.sampleMetadata['Correction Batch'].values == i, 'Run Order'])
                start = dataset.sampleMetadata.loc[dataset.sampleMetadata['Run Order'] == batchMin, 'Run Order'].values
                end = dataset.sampleMetadata.loc[dataset.sampleMetadata['Run Order'] == batchMax, 'Run Order'].values

            # Plot rectangle
            if len(batches) <= 8:
                currentLabel = 'Batch %d' % (i)
            else:
                currentLabel = None
            rect = Rectangle((start, ymin), end - start, abs(ymin) + abs(ymax), color=colors[colIX], alpha=0.4,
                             label=currentLabel, zorder=0)
            ax.add_patch(rect)
            colIX = colIX + 1

    # Annotate figure
    ax.set_xlabel(xAxis)
    ax.set_ylabel(featureName)
    ax.set_xlim(minX, maxX)

    ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=45)
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(formatter)

    # Add colorbar and legend
    if colourBy is not None and discrete is False:
        divider = make_axes_locatable(ax)
        colorbar_ax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(sc, cax=colorbar_ax)
        cbar.set_label('Colored by {0}'.format(colourBy))
    # legend for discrete? Or discrete colorbar?
    # elif colourBy is not None and discrete is True:
    # ax.legend()

    # Only show legend for batches when the number is reasonable (less than 8)
    if addBatchShading and len(batches) < 8:
        # batchLegend = ax.legend(loc='upper left', bbox_to_anchor=(1.3, 1))
        batchLegend = ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1.015))

    fig.suptitle(title)


def plotFeatureInteractive(msData, featureName, xAxis='Acquired Time', colourBy=None, labelby='Run Order',
                       addBatchShading=False, logy=False, withExclusions=True, title=''):
    """
    Interactively visualise feature intensity with plotly. Data points can be coloured and labelled by any metadata field.

    Plots may be of two types:

    :param MSDataset msData: Dataset object
    :param str plottype: Select plot type, may be either ``Sample Type`` or ``Linearity Reference``
    :return: Data object to use with plotly
    """
    import plotly.graph_objs as go

    # Generate TIC
    tempFeatureMask = numpy.sum(numpy.isfinite(msData.intensityData), axis=0)
    tempFeatureMask = tempFeatureMask < msData.intensityData.shape[0]
    tic = numpy.sum(msData.intensityData[:, tempFeatureMask == False], axis=1)

    if withExclusions:
        tempSampleMask = msData.sampleMask
    else:
        tempSampleMask = numpy.ones(shape=msData.sampleMask.shape, dtype=bool)

    # Plot by 'Run Order' if 'Acquired Time' not available
    if ('Acquired Time' in msData.sampleMetadata.columns):
        plotby = 'Acquired Time'
    elif ('Run Order' in msData.sampleMetadata.columns):
        plotby = 'Run Order'
    else:
        print(
            'Acquired Time/Run Order data (columns in dataset.sampleMetadata) not available to plot')
        return

    if plottype == 'Sample Type':  # Plot TIC for SR samples coloured by batch

        SSmask = ((msData.sampleMetadata[
                       'SampleType'].values == SampleType.StudySample) & (
                              msData.sampleMetadata[
                                  'AssayRole'].values == AssayRole.Assay)) & tempSampleMask
        SPmask = ((msData.sampleMetadata[
                       'SampleType'].values == SampleType.StudyPool) & (
                              msData.sampleMetadata[
                                  'AssayRole'].values == AssayRole.PrecisionReference)) & tempSampleMask
        ERmask = ((msData.sampleMetadata[
                       'SampleType'].values == SampleType.ExternalReference) & (
                              msData.sampleMetadata[
                                  'AssayRole'].values == AssayRole.PrecisionReference)) & tempSampleMask

        SSplot = go.Scatter(
            x=msData.sampleMetadata[plotby][SSmask],
            y=tic[SSmask],
            mode='markers',
            marker=dict(
                colorscale='Portland',
                color=msData.sampleMetadata['Correction Batch'][SSmask],
                symbol='circle'
            ),
            name='Study Sample',
            text=msData.sampleMetadata[labelby][SSmask]
        )

        SRplot = go.Scatter(
            x=msData.sampleMetadata[plotby][SPmask],
            y=tic[SPmask],
            mode='markers',
            marker=dict(
                color='rgb(63, 158, 108)',
                symbol='cross'
            ),
            name='Study Reference',
            text=msData.sampleMetadata[labelby][SPmask]
        )

        LTRplot = go.Scatter(
            x=msData.sampleMetadata[plotby][ERmask],
            y=tic[ERmask],
            mode='markers',
            marker=dict(
                color='rgb(198, 83, 83)',
                symbol='cross'
            ),
            name='Long-Term Reference',
            text=msData.sampleMetadata[labelby][ERmask]
        )

        data = [SSplot, SRplot, LTRplot]
        Xlabel = plotby
        title = 'TIC by Sample Type Coloured by Batch'

    if plottype == 'Serial Dilution':  # Plot TIC for LR samples coloured by dilution

        LRmask = ((msData.sampleMetadata[
                       'SampleType'].values == SampleType.StudyPool) & (
                              msData.sampleMetadata[
                                  'AssayRole'].values == AssayRole.LinearityReference)) & tempSampleMask

        if hasattr(msData, 'corrExclusions'):

            if msData.corrExclusions is not None:
                SUBSETSmask = generateLRmask(msData)

                for element in msData.corrExclusions:
                    if element in SUBSETSmask:
                        LRmask[SUBSETSmask[element] == True] = False

        tic = tic[LRmask]
        runIX = numpy.argsort(msData.sampleMetadata['Run Order'][LRmask].values)
        runIX = numpy.argsort(runIX)
        labels = msData.sampleMetadata['Sample File Name'][LRmask].values

        LRplot = go.Scatter(
            x=runIX,
            y=tic,
            mode='markers',
            marker=dict(
                colorscale='Portland',
                color=msData.sampleMetadata['Dilution'][LRmask],
                symbol='circle'
            ),
            text=labels
        )

        data = [LRplot]
        Xlabel = 'Condensed Run Order'
        title = 'TIC of Dilution Series Samples Coloured by Dilution'

    # Add annotation
    layout = {
        'xaxis': dict(
            title=Xlabel,
        ),
        'yaxis': dict(
            title='TIC'
        ),
        'title': title,
        'hovermode': 'closest',
    }

    fig = {
        'data': data,
        'layout': layout,
    }

    return fig


def _featurePlotHelper(msData, featureName, xAxis='Acquired Time', colourBy=None, addViolin=True, addBatchShading=False,
                       logy=False, title='', withExclusions=True, savePath=None, figureFormat='png', dpi=72, figureSize=(11,7)):

    return colorBy, colormap, labelBy