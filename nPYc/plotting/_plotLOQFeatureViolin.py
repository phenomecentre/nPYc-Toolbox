import numpy
import pandas
import copy
import seaborn as sns
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from ..enumerations import AssayRole, SampleType, CalibrationMethod, QuantificationType
import warnings
from itertools import compress

def _featureLOQViolinPlotHelper(ax, tData, featID, splitByBatch=True, plotBatchLOQ=False, title=None, xLabel=None, xTick=True, yLabel=None, yTick=True, subplot=None, flipYLabel=False, logY=False, tightYLim=True, showLegend=False, onlyLegend=False):
    """
    Draw violin plots for a feature, with LOQ (LLOQ/ULOQ) lines, points for samples already out of LOQ. If `plotBatchLOQ`, samples which will be out of LOQ are also drawn with each batch LOQ line..
    Violins are separated by batch and then by SampleType if `splitByBatch`, otherwise only by SampleType
    Can focus the plot on LLOQ or ULOQ regions using `subplot`.
    Can return only the corresponding legend using `onlyLegend`

    :param axis ax: pointer to a pyplot axis handle to draw into
    :param TargetedDataset tData: TargetedDataset to plot
    :param int featID: integer ID of the feature to plot column position
    :param bool splitByBatch: if ``True`` separate samples by batch and SampleType, if ``False`` only separate by SampleType (x-axis).
    :param bool plotBatchLOQ: if ``True`` add a line at each batch LOQ and points for samples between batch LOQ and merged LOQ, if False only show merged LOQ lines and samples marked as out of LOQ.
    :param title: figure title
    :type title: None or str
    :param str xLabel: x-axis label
    :type xLabel: None or str
    :param bool xTick: if ``True`` show x-axis tick
    :param str ylabel: y-axis label
    :type yLabel: None or str
    :param bool yTick: if ``True`` show y-axis tick
    :param str subplot: ``None`` to plot all data range, 'LLOQ' for zoomed LLOQ region, 'ULOQ' for zoomed ULOQ region
    :param bool flipYlabel: if ``True``, place the y-axis label on the right-hand side
    :param bool logY: if ``True``, log scale the y-axis
    :param bool tightYLim: if ``True`` ylim are close to the points but can let LOQ lines outside, if ``False`` LOQ lines will be part of the plot. Only apply to `subplot=None`.
    :param bool showLegend: if ``True`` add a legend to the right
    :param bool onlyLegend: if ``True`` does not plot any features, only the legend (overriding all other options, only using 'ax', 'tData' for the SampleTypes to put, and 'plotBatchLOQ' for the required line markers)
    """
    if (subplot is not None) & (subplot != 'LLOQ') & (subplot != 'ULOQ'):
        raise ValueError('subplot expects \'None\', \'LLOQ\' or \'ULOQ\'')

    # Load toolbox wide color scheme
    if 'sampleTypeColours' in tData.Attributes.keys():
        sTypeColourDict = copy.deepcopy(tData.Attributes['sampleTypeColours'])
        for stype in SampleType:
            if stype.name in sTypeColourDict.keys():
                sTypeColourDict[stype] = sTypeColourDict.pop(stype.name)
    else:
        sTypeColourDict = {SampleType.StudySample: 'b', SampleType.StudyPool: 'g', SampleType.ExternalReference: 'r',
                           SampleType.MethodReference: 'm', SampleType.ProceduralBlank: 'c', 'Other': 'grey'}

    # SampleType masks
    SSmask = ((tData.sampleMetadata['SampleType'].values == SampleType.StudySample) & (tData.sampleMetadata['AssayRole'].values == AssayRole.Assay))
    SPmask = ((tData.sampleMetadata['SampleType'].values == SampleType.StudyPool) & (tData.sampleMetadata['AssayRole'].values == AssayRole.PrecisionReference))
    ERmask = ((tData.sampleMetadata['SampleType'].values == SampleType.ExternalReference) & (tData.sampleMetadata['AssayRole'].values == AssayRole.PrecisionReference))

    # Plot figures
    if not onlyLegend:
        # Init
        # find XLOQ_batchX, get batch ID, check agreement
        if plotBatchLOQ:
            col_LLOQ = sorted(tData.featureMetadata.columns[tData.featureMetadata.columns.to_series().str.contains('LLOQ_batch')].tolist())
            col_LLOQ_batch = sorted([int(i.replace('LLOQ_batch', '')) for i in col_LLOQ])
            col_ULOQ = sorted(tData.featureMetadata.columns[tData.featureMetadata.columns.to_series().str.contains('ULOQ_batch')].tolist())
            col_ULOQ_batch = sorted([int(i.replace('ULOQ_batch', '')) for i in col_ULOQ])
        batches = sorted((numpy.unique(tData.sampleMetadata.loc[:, 'Batch'].values[~numpy.isnan(tData.sampleMetadata.loc[:, 'Batch'].values)])).astype(int))
        number_of_batch = len(batches)
        # check batch LOQ info is present
        if plotBatchLOQ:
            if (col_LLOQ_batch != col_ULOQ_batch) | (col_LLOQ_batch != batches):
                warnings.warn('Batch LOQ information missing, unable to plotBatchLOQ')
                plotBatchLOQ = False

        # Sample without inf masks
        concentrationValues = tData._intensityData[:, featID]
        noInfMask = (abs(concentrationValues) != numpy.inf)

        # Current samples <LLOQ >ULOQ masks
        old_LLOQMask = (concentrationValues == -numpy.inf)
        old_ULOQMask = (concentrationValues == numpy.inf)

        # Samples which will be out <LLOQ >ULOQ masks
        new_LLOQ = tData.featureMetadata.loc[featID, 'LLOQ']
        new_ULOQ = tData.featureMetadata.loc[featID, 'ULOQ']
        with warnings.catch_warnings(): # Suppress RuntimeWarnings from NaN values
            warnings.simplefilter('ignore', RuntimeWarning)
            new_LLOQMask = (concentrationValues < new_LLOQ) & noInfMask
            new_ULOQMask = (concentrationValues > new_ULOQ) & noInfMask
        # Colour palette and hue_order specific to the samples present in this dataset
        # sTypeColourDict can be plugged directly as a palette, hue_order is still needed
        hue_order = []
        if sum(SSmask) > 0:
            hue_order.append(SampleType.StudySample)
        if sum(SPmask) > 0:
            hue_order.append(SampleType.StudyPool)
        if sum(ERmask) > 0:
            hue_order.append(SampleType.ExternalReference)
        # Need color for each sampleType otherwise swarmplot crash
        point_palette = {}
        for stype in hue_order:
            point_palette[stype] = 'white'

        # new LOQ shaded rectangle
        if (not numpy.isnan(new_LLOQ)) & (not numpy.isnan(new_ULOQ)):
            if splitByBatch:
                p_rect = ax.add_patch(mpatches.Rectangle((-0.5, new_LLOQ), max(batches), (new_ULOQ - new_LLOQ), facecolor='grey', alpha=0.15)) #Rectangle((x,y), width, height)
            else:
                p_rect = ax.add_patch(mpatches.Rectangle((0, new_LLOQ), 1, (new_ULOQ - new_LLOQ), facecolor='grey', alpha=0.15))
            p_rect.set_zorder(0)

        # Violin plot
        # (remove inf values, only keep plotted SampleTypes)
        violin_x = tData.sampleMetadata['Batch'][(SSmask | SPmask | ERmask) & noInfMask].values
        violin_y = concentrationValues[(SSmask | SPmask | ERmask) & noInfMask]
        violin_subgroup = tData.sampleMetadata['SampleType'][(SSmask | SPmask | ERmask) & noInfMask].values
        # make sure we have values to plot
        if len(violin_y) > 0:
            if splitByBatch:
                p_violin = sns.violinplot(x=violin_x, y=violin_y,
                                          #hue=violin_subgroup, palette=sTypeColourDict,
                                          hue_order=hue_order,
                                          ax=ax, density_norm='width', bw_method=.2, cut=0)
            else:
                # if not splitting by batch, the x-axis is the SampleType which is already recorder in _subgroup. Order is the same as hue_order. No hue used here
                p_violin = sns.violinplot(x=violin_subgroup, y=violin_y,
                                          palette=sTypeColourDict, order=hue_order,
                                          ax=ax, density_norm='width', bw_method=.2, cut=0)

        # merged LLOQ / ULOQ lines
        if not numpy.isnan(new_LLOQ):
            p_LLOQ_line = ax.axhline(y=new_LLOQ, xmin=0, xmax=1, linestyle='-.', color='grey')
        if not numpy.isnan(new_ULOQ):
            p_ULOQ_line = ax.axhline(y=new_ULOQ, xmin=0, xmax=1, linestyle='-.', color='grey')

        # Keep the xlims before batch plots
        xlims = ax.get_xlim()

        # if plotBatch: batch LOQ lines, already + future out of LOQ samples marked as points
        # if False: only already out of LOQ samples points
        pt_LOQ_x = []
        pt_LOQ_y = []
        pt_LOQ_subgroup = []
        if plotBatchLOQ:
            # Batch LLOQ / ULOQ, lines and points
            # Prepare batch points on LOQ (cannot be drawn in loop), draw batch LOQ lines
            x_increment = 1 / number_of_batch
            for i in range(number_of_batch):
                batchMask = (tData.sampleMetadata['Batch'].values == batches[i])
                ## LLOQ
                # get LLOQ_batchX column, keep old <LLOQ for given batch
                col_LLOQ_name = list(compress(col_LLOQ, (batches[i] == col_LLOQ_batch).tolist()))[0]
                old_LLOQ = tData.featureMetadata.loc[featID, col_LLOQ_name]
                old_LLOQ_x = tData.sampleMetadata.loc[(old_LLOQMask & batchMask), 'Batch'].values
                # append points
                pt_LOQ_x.extend(old_LLOQ_x)
                pt_LOQ_y.extend([old_LLOQ] * len(old_LLOQ_x))
                pt_LOQ_subgroup.extend(tData.sampleMetadata.loc[(old_LLOQMask & batchMask), 'SampleType'].values)
                ## ULOQ
                col_ULOQ_name = list(compress(col_ULOQ, (batches[i] == col_ULOQ_batch).tolist()))[0]
                old_ULOQ = tData.featureMetadata.loc[featID, col_ULOQ_name]
                old_ULOQ_x = tData.sampleMetadata.loc[(old_ULOQMask & batchMask), 'Batch'].values
                # append points
                pt_LOQ_x.extend(old_ULOQ_x)
                pt_LOQ_y.extend([old_ULOQ] * len(old_ULOQ_x))
                pt_LOQ_subgroup.extend(tData.sampleMetadata.loc[(old_ULOQMask & batchMask), 'SampleType'].values)
                # plot each batch LOQ lines
                if splitByBatch:
                    p_batchLLOQ_line = ax.axhline(y=old_LLOQ, xmin=i * x_increment, xmax=(i + 1) * x_increment, linestyle=':', color='orangered')
                    p_batchULOQ_line = ax.axhline(y=old_ULOQ, xmin=i * x_increment, xmax=(i + 1) * x_increment, linestyle=':', color='orangered')
                else:     # plotBatchLOQ=True with splitByBatch=False is not recommended for clarity of the plot
                    p_batchLLOQ_line = ax.axhline(y=old_LLOQ, xmin=0, xmax=1, linestyle=':', color='orangered')
                    p_batchULOQ_line = ax.axhline(y=old_ULOQ, xmin=0, xmax=1, linestyle=':', color='orangered')

            # future LOQ samples
            newLOQ_x = tData.sampleMetadata['Batch'][new_LLOQMask | new_ULOQMask].values
            pt_LOQ_x.extend(newLOQ_x)
            pt_LOQ_y.extend(concentrationValues[new_LLOQMask | new_ULOQMask])
            pt_LOQ_subgroup.extend(tData.sampleMetadata['SampleType'][new_LLOQMask | new_ULOQMask].values)

        else:
            # Only the points already out of LOQ (no batch specific values)
            ## LLOQ
            old_LLOQ = tData.featureMetadata.loc[featID, 'LLOQ']
            old_LLOQ_x = tData.sampleMetadata.loc[(old_LLOQMask), 'Batch'].values
            # append points
            pt_LOQ_x.extend(old_LLOQ_x)
            pt_LOQ_y.extend([old_LLOQ] * len(old_LLOQ_x))
            pt_LOQ_subgroup.extend(tData.sampleMetadata.loc[(old_LLOQMask), 'SampleType'].values)
            ## ULOQ
            old_ULOQ = tData.featureMetadata.loc[featID, 'ULOQ']
            old_ULOQ_x = tData.sampleMetadata.loc[(old_ULOQMask), 'Batch'].values
            # append points
            pt_LOQ_x.extend(old_ULOQ_x)
            pt_LOQ_y.extend([old_ULOQ] * len(old_ULOQ_x))
            pt_LOQ_subgroup.extend(tData.sampleMetadata.loc[(old_ULOQMask), 'SampleType'].values)

        # Plot all batch LOQ points
        if (len(pt_LOQ_x) != 0) and (not numpy.isnan(old_LLOQ)):
            if splitByBatch:
                p_LOQ_point = sns.swarmplot(x=pt_LOQ_x, y=pt_LOQ_y, hue=pt_LOQ_subgroup, marker='X', palette=point_palette, hue_order=hue_order, ax=ax, dodge=True, size=10, linewidth=1, edgecolor='black')
            else:
                p_LOQ_point = sns.swarmplot(x=pt_LOQ_subgroup, y=pt_LOQ_y, marker='X', palette=point_palette, order=hue_order, ax=ax, dodge=True, size=10, linewidth=1, edgecolor='black')

        # Modify the y-lim depending on the subplot
        # LLOQ
        if subplot == 'LLOQ':
            # if we don't plot batchLOQ, no batch LOQ columns expected
            if plotBatchLOQ:
                allLLOQ_col = copy.deepcopy(col_LLOQ)
            else:
                allLLOQ_col = []
            allLLOQ_col.append('LLOQ')
            # min and max LLOQ across all batch
            ymin = min(tData.featureMetadata.loc[featID, allLLOQ_col]) * 0.9
            ymax = max(tData.featureMetadata.loc[featID, allLLOQ_col]) * 1.1
            # case where LLOQ is nan, centre around min value
            if numpy.isnan(ymin) | numpy.isnan(ymax):
                ymin = min(violin_y.tolist()) * 0.9
                ymax = min(violin_y.tolist()) * 1.1
            ylims = (ymin, ymax)
        # ULOQ
        elif subplot == 'ULOQ':
            if plotBatchLOQ:
                allULOQ_col = copy.deepcopy(col_ULOQ)
            else:
                allULOQ_col = []
            allULOQ_col.append('ULOQ')
            # min and max ULOQ across all batch
            ymin = min(tData.featureMetadata.loc[featID, allULOQ_col]) * 0.9
            ymax = max(tData.featureMetadata.loc[featID, allULOQ_col]) * 1.1
            # case where ULOQ is nan, centre around max value
            if numpy.isnan(ymin) | numpy.isnan(ymax):
                ymin = max(violin_y.tolist()) * 0.9
                ymax = max(violin_y.tolist()) * 1.1
            ylims = (ymin, ymax)
        # All other plots
        else:
            ylims = ax.get_ylim()
            all_pts_y = violin_y.tolist() + pt_LOQ_y
            # if current (LOQ lines) too far off the data points, tighten ylim
            if tightYLim:
                # ULOQ more than twice the highest point
                if ylims[1] > 2 * max(all_pts_y):
                    ymax = max(all_pts_y) * 1.05
                else:
                    ymax = ylims[1]
                # LLOQ less than half of the lowest point [use the ULOQ scale to fit a bottom margin of correct size]
                if ylims[0] < 0.5 * min(all_pts_y):
                    bottomMargin = ymax * 0.05
                    ymin = min(all_pts_y) - bottomMargin
                else:
                    ymin = ylims[0]
                ylims = (ymin, ymax)
        # Apply new lims
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)

        # Title
        if title is not None:
            ax.set_title(title)
        # x label
        if xLabel is not None:
            ax.set_xlabel(xLabel)
        if not xTick:
            ax.get_xaxis().set_visible(False)
        # y label
        if yLabel is not None:
            ax.set_ylabel(yLabel)
        if not yTick:
            ax.get_yaxis().set_visible(False)
        if flipYLabel:
            ax.yaxis.set_label_position("right")

        # log y-axis
        if logY:
            # if log scale, make sure lower ylim is 0
            currentYlims = ax.get_ylim()
            if currentYlims[0] < 0.:
                ax.set_ylim(0, currentYlims[1])
            ax.set_yscale('symlog')
        else:
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    # onlyLegend: define the required colors but no data manipulation
    else:
        hue_order = []
        if sum(SSmask) > 0:
            hue_order.append(SampleType.StudySample)
        if sum(SPmask) > 0:
            hue_order.append(SampleType.StudyPool)
        if sum(ERmask) > 0:
            hue_order.append(SampleType.ExternalReference)

    # define legend
    legendText = []
    legendBox = []
    # Proxy legend
    # p_violin
    for stype in hue_order:
        legendText.append(str(stype))
        legendBox.append(mpatches.Patch(color=sTypeColourDict[stype]))
    # p_LLOQ_line, p_ULOQ_line
    legendText.append('LLOQ / ULOQ')
    legendBox.append(mlines.Line2D([], [], linestyle='-.', color='grey'))
    # p_rect
    legendText.append('LOQ Range')
    legendBox.append(mpatches.Patch(color='grey', alpha=0.15))
    # p_batchLLOQ_line, p_batchULOQ_line
    if plotBatchLOQ:
        legendText.append('Batch LLOQ / ULOQ')
        legendBox.append(mlines.Line2D([], [], linestyle=':', color='orangered'))
    # p_LOQ_point
    legendText.append('LLOQ / ULOQ Samples')
    legendBox.append(mlines.Line2D([], [], linestyle='', marker='X', markersize=10, markerfacecolor='white', markeredgecolor='black'))

    # add or remove legend
    if (not showLegend) & (not onlyLegend):
        # depending on the data present, a legend might not exist throwing an "AttributeError: 'NoneType' object has no attribute 'remove'"
        try:
            ax.legend_.remove()
        except AttributeError:
            pass
    elif onlyLegend:
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(legendBox, legendText, loc='upper left')
    else:
        ax.legend(legendBox, legendText, loc='upper left', bbox_to_anchor=(1, 1))
