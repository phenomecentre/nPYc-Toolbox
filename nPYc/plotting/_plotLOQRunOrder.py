import matplotlib.pyplot as plt
import seaborn as sns
import numpy
import pandas
import datetime
from nPYc.enumerations import AssayRole, SampleType, CalibrationMethod, QuantificationType
from matplotlib.dates import MO, TU, WE, TH, FR, SA, SU
from matplotlib.dates import WeekdayLocator, HourLocator, DateFormatter
from matplotlib import gridspec

def plotLOQRunOrder(targetedData, addCalibration=True, compareBatch=True, title='', savePath=None, figureFormat='png', dpi=72, figureSize=(11, 7)):
    """
    Visualise ratio of LLOQ and ULOQ by run order, separated by batch. Option to add barchart that summarises across batch

    :param TargetedDataset targetedData: TargetedDataset object
    :param bool addCalibration: If ``True`` add calibration samples
    :param bool compareBatch: If ``True`` add barchart across batch, separated by SampleType
    :param str title: Title for the plot
    :param savePath: If ``None`` plot interactively, otherwise save the figure to the path specified
    :type savePath: None or str
    :param str figureFormat: If saving the plot, use this format
    :param int dpi: Plot resolution
    :param figureSize: Dimensions of the figure
    :type figureSize: tuple(float, float)
    :raises ValueError: if targetedData does not satisfy to the TargetedDataset definition for QC
    :raises ValueError: if :py:attr:`calibration` does not match the number of batch
    """
    # Monitored features are plotted in a different color (own featureMask)
    # Count <LLOQ / normal / >ULOQ in features not monitored
    # Hatch SP and ER with sampleMask
    # If addCalibration, add single color bars (max height)
    # If more than one batch, create a subplot for each batch
    # If compareBatch, add subplot with batch average values for each subgroup. If not compare batch a single legend is plotted
    # Prepare all the data first, find the plotting parameters common to all subplots (x-axis, legend), then plot
    # x-axis width delta(min max), bar width and legend are shared across all subplots

    # Check dataset is fit for plotting
    validDataset = targetedData.validateObject(verbose=False, raiseError=False, raiseWarning=False)
    if not validDataset['QC']:
        raise ValueError('Import Error: targetedData does not satisfy to the TargetedDataset definition for QC')

    # Check addCalibration is possible
    if addCalibration:
        if isinstance(targetedData.calibration, dict):
            if 'Acquired Time' not in targetedData.calibration['calibSampleMetadata'].columns:
                addCalibration = False
        elif isinstance(targetedData.calibration, list):
            if 'Acquired Time' not in targetedData.calibration[0]['calibSampleMetadata'].columns:
                addCalibration = False

    # Init
    batches = sorted((numpy.unique(targetedData.sampleMetadata.loc[:, 'Batch'].values[~numpy.isnan(targetedData.sampleMetadata.loc[:, 'Batch'].values)])).astype(int))
    number_of_batch = len(batches)
    sns.set_style("ticks", {'axes.linewidth': 0.75})
    sns.set_color_codes(palette='deep')
    # If more than 1 batch, plot batch summary
    if (number_of_batch > 1) & compareBatch:
        gs = gridspec.GridSpec(number_of_batch + 1, 5)
        nbRow = number_of_batch + 1
    else:
        gs = gridspec.GridSpec(number_of_batch, 5)
        nbRow = number_of_batch
    # Set plot size
    nbRowByHeight = 3
    newHeight = int(numpy.ceil((figureSize[1]/nbRowByHeight) * nbRow))
    fig = plt.figure(figsize=(figureSize[0], newHeight), dpi=dpi)
    batch_data    = []
    batch_summary = {'SS': {'LLOQ': [], 'normal': [], 'ULOQ': []}, 'SP': {'LLOQ': [], 'normal': [], 'ULOQ': []}, 'ER': {'LLOQ': [], 'normal': [], 'ULOQ': []}}
    color_monitored = 'C7'
    color_LLOQ      = 'C3'
    color_normal    = 'C2'  # 'white'
    color_ULOQ      = 'C1'
    color_calib     = 'C0'

    # Prepare and store data for each batch
    for i in range(number_of_batch):
        out_data = dict()

        # Batch sample mask
        batchMask = (targetedData.sampleMetadata['Batch'].values == batches[i])

        # Define feature mask for Monitored and or not
        monitoredFeatureMask = (targetedData.featureMetadata['quantificationType'] == QuantificationType.Monitored).values
        quantifiedFeatureMask = (targetedData.featureMetadata['quantificationType'] != QuantificationType.Monitored).values

        # Define sample types
        SPMask = (targetedData.sampleMetadata['SampleType'].values == SampleType.StudyPool) & (targetedData.sampleMetadata['AssayRole'].values == AssayRole.PrecisionReference) & batchMask
        ERMask = (targetedData.sampleMetadata['SampleType'].values == SampleType.ExternalReference) & (targetedData.sampleMetadata['AssayRole'].values == AssayRole.PrecisionReference) & batchMask

        # x axis
        x    = targetedData.sampleMetadata.loc[batchMask, 'Acquired Time'].tolist()
        x_SP = targetedData.sampleMetadata.loc[SPMask, 'Acquired Time'].tolist()
        x_ER = targetedData.sampleMetadata.loc[ERMask, 'Acquired Time'].tolist()
        # make sure x_axis is datetime, if it's already a datetime it will trigger an AttributeError
        try:
            x    = [xtime.to_pydatetime() for xtime in x]
            x_SP = [xtime.to_pydatetime() for xtime in x_SP]
            x_ER = [xtime.to_pydatetime() for xtime in x_ER]
        except AttributeError:
            pass
        out_data['x']    = x
        out_data['x_SP'] = x_SP
        out_data['x_ER'] = x_ER

        if addCalibration:
            # list of calibration expected if more than 1 batch
            if isinstance(targetedData.calibration, list) & (number_of_batch != 1):
                x_calib = targetedData.calibration[i]['calibSampleMetadata']['Acquired Time'].tolist()
            # single calibration if only one batch
            elif isinstance(targetedData.calibration, dict) & (number_of_batch == 1):
                x_calib = targetedData.calibration['calibSampleMetadata']['Acquired Time'].tolist()
            else:
                raise ValueError('Calibration does not match the number of batch')
            out_data['x_calib'] = x_calib

        # y-axis (Subclass values)
        # Will plots all sample colors in y_, SP and ER are transparent and cover it
        y_monitored  = numpy.repeat(sum(monitoredFeatureMask), len(x))
        y_LLOQ       = (targetedData._intensityData[:, quantifiedFeatureMask][batchMask, :] == -numpy.inf).sum(1)
        y_normal     = ((targetedData._intensityData[:, quantifiedFeatureMask][batchMask, :] != -numpy.inf) & (targetedData._intensityData[:, quantifiedFeatureMask][batchMask, :] != numpy.inf)).sum(1)
        y_ULOQ       = (targetedData._intensityData[:, quantifiedFeatureMask][batchMask, :] == numpy.inf).sum(1)
        height_total = y_monitored + y_LLOQ + y_normal + y_ULOQ
        height_total = numpy.unique(height_total)
        if len(height_total) != 1:
            raise ValueError('Number of features do not match across samples')
        else:
            height_total = height_total[0]
        y_SP = numpy.repeat(height_total, len(x_SP))
        y_ER = numpy.repeat(height_total, len(x_ER))
        out_data['y_monitored'] = y_monitored
        out_data['y_LLOQ']      = y_LLOQ
        out_data['y_normal']    = y_normal
        out_data['y_ULOQ']      = y_ULOQ
        out_data['y_SP']        = y_SP
        out_data['y_ER']        = y_ER
        if addCalibration:
            y_calib = numpy.repeat(height_total, len(x_calib))
            out_data['y_calib'] = y_calib

        # Store values to compare across batch
        tmp_SPMask = (targetedData.sampleMetadata.loc[batchMask, 'SampleType'].values == SampleType.StudyPool) & (targetedData.sampleMetadata.loc[batchMask, 'AssayRole'].values == AssayRole.PrecisionReference)
        tmp_ERMask = (targetedData.sampleMetadata.loc[batchMask, 'SampleType'].values == SampleType.ExternalReference) & (targetedData.sampleMetadata.loc[batchMask, 'AssayRole'].values == AssayRole.PrecisionReference)
        tmp_SSMask = numpy.invert(tmp_SPMask | tmp_ERMask)
        # check some SS exist
        if sum(tmp_SSMask) != 0:
            batch_summary['SS']['LLOQ'].append(numpy.mean(y_LLOQ[tmp_SSMask]))
            batch_summary['SS']['normal'].append(numpy.mean(y_normal[tmp_SSMask]))
            batch_summary['SS']['ULOQ'].append(numpy.mean(y_ULOQ[tmp_SSMask]))
        else:
            batch_summary['SS']['LLOQ'].append(0)
            batch_summary['SS']['normal'].append(0)
            batch_summary['SS']['ULOQ'].append(0)
        # check some SP exist
        if sum(tmp_SPMask) != 0:
            batch_summary['SP']['LLOQ'].append(numpy.mean(y_LLOQ[tmp_SPMask]))
            batch_summary['SP']['normal'].append(numpy.mean(y_normal[tmp_SPMask]))
            batch_summary['SP']['ULOQ'].append(numpy.mean(y_ULOQ[tmp_SPMask]))
        else:
            batch_summary['SP']['LLOQ'].append(0)
            batch_summary['SP']['normal'].append(0)
            batch_summary['SP']['ULOQ'].append(0)
        # check some ER exist
        if sum(tmp_ERMask) != 0:
            batch_summary['ER']['LLOQ'].append(numpy.mean(y_LLOQ[tmp_ERMask]))
            batch_summary['ER']['normal'].append(numpy.mean(y_normal[tmp_ERMask]))
            batch_summary['ER']['ULOQ'].append(numpy.mean(y_ULOQ[tmp_ERMask]))
        else:
            batch_summary['ER']['LLOQ'].append(0)
            batch_summary['ER']['normal'].append(0)
            batch_summary['ER']['ULOQ'].append(0)

        # Axes parameters
        # Width of 1 sample (min difference between 2 samples). Diff is in second, matplotlib width use 1=1day
        try:
            timeList = sorted(x)
            timeDiff = [j - i for i, j in zip(timeList[:-1], timeList[1:])]
            acquisitionLength = min(timeDiff).seconds  # convert to seconds
        except ValueError:
            timeDiff = [datetime.timedelta(minutes=15)]
            acquisitionLength = 900                    # 15 min, if not enough points to calculate the diff
        interval = acquisitionLength / (24 * 60 * 60)  # acquisition length in day
        # width = 0.75*interval
        out_data['width'] = 0.6 * interval
        # Time range of this batch
        if addCalibration:
            x_full = x + x_calib
        else:
            x_full = x
        minX = min(x_full) - min(timeDiff)  # widen left and right so bars aren't cut
        maxX = max(x_full) + min(timeDiff)
        out_data['minX']  = minX
        out_data['maxX']  = maxX
        out_data['delta'] = maxX - minX
        # Legend parameters - check legends need it
        legend = {'has_SP': False, 'has_ER': False, 'has_calib': False}
        if len(x_SP) != 0:
            legend['has_SP'] = True
        if len(x_ER) != 0:
            legend['has_ER'] = True
        if addCalibration:
            if len(x_calib) != 0:
                legend['has_calib'] = True
        out_data['legend'] = legend

        # Store data
        batch_data.append(out_data)

    # Get common x-axis parameters
    common_width = min([i['width'] for i in batch_data])
    common_delta = max([i['delta'] for i in batch_data])
    delta = numpy.array([common_delta], dtype="timedelta64[us]")[0]  # convert from datetime.timedelta to numpy.timedelta64 #from numpy.timedelta64 "This will be fully compatible with the datetime class of the datetime module of Python only when using a time unit of microseconds"
    days = delta.astype('timedelta64[D]')
    days = days / numpy.timedelta64(1, 'D')
    # check width to add minor tick if <48hr. If >48hr do not tick the weekend
    if days < 2:
        detailedTick = True
    else:
        detailedTick = False
    # Get common legend parameters
    common_has_SP    = [i['legend']['has_SP'] for i in batch_data]
    common_has_ER    = [i['legend']['has_ER'] for i in batch_data]
    common_has_calib = [i['legend']['has_calib'] for i in batch_data]
    has_SP    = (True in common_has_SP)
    has_ER    = (True in common_has_ER)
    has_calib = (True in common_has_calib)

    # Plot each batch in its subplot
    for i in range(number_of_batch):
        # Allow for legend space if not compareBatch
        if (number_of_batch > 1) & compareBatch:
            ax = plt.subplot(gs[i, :])
        else:
            ax = plt.subplot(gs[i, :-1])

        # Plot
        height_cumulative = numpy.zeros(len(batch_data[i]['x']))

        # Monitored
        p_monitored = ax.bar(batch_data[i]['x'], batch_data[i]['y_monitored'], common_width, color=color_monitored)
        height_cumulative += batch_data[i]['y_monitored']
        # LLOQ
        p_LLOQ = ax.bar(batch_data[i]['x'], batch_data[i]['y_LLOQ'], common_width, bottom=height_cumulative, color=color_LLOQ)
        height_cumulative += batch_data[i]['y_LLOQ']
        # Normal
        p_normal = ax.bar(batch_data[i]['x'], batch_data[i]['y_normal'], common_width, bottom=height_cumulative, color=color_normal, alpha=0.3)  # , edgecolor='grey', linewidth=0.2)
        height_cumulative += batch_data[i]['y_normal']
        # ULOQ
        p_ULOQ = ax.bar(batch_data[i]['x'], batch_data[i]['y_ULOQ'], common_width, bottom=height_cumulative, color=color_ULOQ)
        height_cumulative += batch_data[i]['y_ULOQ']
        # SP
        p_SP = ax.bar(batch_data[i]['x_SP'], batch_data[i]['y_SP'], common_width, fill=False, edgecolor='black', hatch='/')
        # ER
        p_ER = ax.bar(batch_data[i]['x_ER'], batch_data[i]['y_ER'], common_width, fill=False, edgecolor='black', hatch='\\')
        # Calibration
        if addCalibration:
            p_calib = ax.bar(batch_data[i]['x_calib'], batch_data[i]['y_calib'], common_width, color=color_calib, edgecolor=color_calib)

        # Annotate Figure
        # Annotate axis
        if i == number_of_batch - 1:  # only x-tick for the last batch
            ax.set_xlabel('Acquisition Date')
        ax.set_ylabel('Number of features')
        ax.set_xlim(batch_data[i]['minX'], batch_data[i]['minX'] + common_delta)  # all batch cover the same range
        # ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=45)
        # x-axis tick mark
        if detailedTick:  # Major tick (day) + minor tick (6hr)
            majorLoc = WeekdayLocator(byweekday=(MO, TU, WE, TH, FR, SA, SU))
            majorFormatter = DateFormatter('%d/%m/%y')
            minorLoc = HourLocator(byhour=[6, 12, 18], interval=1)
            minorFormatter = DateFormatter('%H:%M')
            ax.xaxis.set_major_locator(majorLoc)
            ax.xaxis.set_major_formatter(majorFormatter)
            ax.xaxis.set_minor_locator(minorLoc)
            ax.xaxis.set_minor_formatter(minorFormatter)
        else:  # Major tick (weekday) only
            majorLoc = WeekdayLocator(byweekday=(MO, TU, WE, TH, FR))
            majorFormatter = DateFormatter('%d/%m/%y')
            ax.xaxis.set_major_locator(majorLoc)
            ax.xaxis.set_major_formatter(majorFormatter)

        # Legend
        legendBox = []
        legendText = []
        # basic color
        legendBox.extend([p_monitored[0], p_LLOQ[0], p_normal[0], p_ULOQ[0]])
        legendText.extend(['Monitored', '<LLOQ', 'Normal', '>ULOQ'])
        if has_SP:
            legendBox.append(p_SP[0])
            legendText.append('SP')
        if has_ER:
            legendBox.append(p_ER[0])
            legendText.append('ER')
        if has_calib:
            legendBox.append(p_calib[0])
            legendText.append('Calibration')
        # Legend to the right only if single batch, or multibatch without compareBatch. Title in multiBatch
        if (number_of_batch == 1):
            ax.legend(legendBox, legendText, loc='upper left', bbox_to_anchor=(1, 1))
        elif (i == 0) & (not compareBatch):
            ax.legend(legendBox, legendText, loc='upper left', bbox_to_anchor=(1, 1))
            ax.set_title('Batch ' + str(batches[i]))
        else:
            ax.set_title('Batch ' + str(batches[i]))
    # end of Batch subplots


    # Summary plot across batch
    # If more than 1 batch, plot batch summary
    if (number_of_batch > 1) & compareBatch:
        x_summary = batches
        empty_sampleType = {'LLOQ': numpy.repeat(0, number_of_batch).tolist(),
                            'normal': numpy.repeat(0, number_of_batch).tolist(),
                            'ULOQ': numpy.repeat(0, number_of_batch).tolist()}
        xi_pos = 0
        width = 0.9
        # SS
        if batch_summary['SS'] != empty_sampleType:
            axSS = plt.subplot(gs[number_of_batch, xi_pos])
            axSS.bar(x_summary, batch_summary['SS']['LLOQ'], width, color=color_LLOQ)
            axSS.bar(x_summary, batch_summary['SS']['normal'], width, color=color_normal, alpha=0.3, bottom=batch_summary['SS']['LLOQ'])
            axSS.bar(x_summary, batch_summary['SS']['ULOQ'], width, color=color_ULOQ, bottom=[sum(x) for x in zip(batch_summary['SS']['LLOQ'], batch_summary['SS']['normal'])])
            axSS.set_xlabel('Batch')
            axSS.set_ylabel('Number of features')
            axSS.set_title('SS')
            xi_pos += 1
        # SP
        if batch_summary['SP'] != empty_sampleType:
            axSP = plt.subplot(gs[number_of_batch, xi_pos])
            axSP.bar(x_summary, batch_summary['SP']['LLOQ'], width, color=color_LLOQ)
            axSP.bar(x_summary, batch_summary['SP']['normal'], width, color=color_normal, alpha=0.3, bottom=batch_summary['SP']['LLOQ'])
            axSP.bar(x_summary, batch_summary['SP']['ULOQ'], width, color=color_ULOQ, bottom=[sum(x) for x in zip(batch_summary['SP']['LLOQ'], batch_summary['SP']['normal'])])
            axSP.set_xlabel('Batch')
            axSP.set_title('SP')
            xi_pos += 1
        # ER
        if batch_summary['ER'] != empty_sampleType:
            axER = plt.subplot(gs[number_of_batch, xi_pos])
            axER.bar(x_summary, batch_summary['ER']['LLOQ'], width, color=color_LLOQ)
            axER.bar(x_summary, batch_summary['ER']['normal'], width, color=color_normal, alpha=0.3, bottom=batch_summary['ER']['LLOQ'])
            axER.bar(x_summary, batch_summary['ER']['ULOQ'], width, color=color_ULOQ, bottom=[sum(x) for x in zip(batch_summary['ER']['LLOQ'], batch_summary['ER']['normal'])])
            axER.set_xlabel('Batch')
            axER.set_title('ER')
            xi_pos += 1
        # Legend
        axLeg = plt.subplot(gs[number_of_batch, 4])
        axLeg.set_axis_off()
        axLeg.legend(legendBox, legendText, loc='center')

    # Title and layout
    fig.suptitle(title)
    fig.tight_layout()

    # Save or output
    if savePath:
        plt.savefig(savePath, bbox_inches='tight', format=figureFormat, dpi=dpi)
        plt.close()
    else:
        plt.show()
