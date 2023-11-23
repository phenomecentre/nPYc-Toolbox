# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 13:27:55 2016

@author: cs401
"""
import matplotlib
from matplotlib import gridspec
import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex
from matplotlib.collections import LineCollection
import matplotlib.dates as mdates
import os
import seaborn as sns
import plotly
import plotly.graph_objs as go
import pandas
import numpy
from ._violinPlot import _violinPlotHelper
from pyChemometrics.ChemometricsPCA import ChemometricsPCA
from ._plotDiscreteLoadings import plotDiscreteLoadings
from ._plotting import checkAndSetPlotAttributes
from ..objects import Dataset
from ..enumerations import SampleType
from ..utilities.generic import createDestinationPath
import copy
import datetime


def plotScree(R2, Q2=None, title='', xlabel='', ylabel='', savePath=None, figureFormat='png', dpi=72,
              figureSize=(11, 7)):
    """
	Plot a barchart of variance explained (R2) and predicted (Q2) (if available) for each PCA component.

	:param numpy.array R2: PCA R2 values
	:param numpy.array Q2: PCA Q2 values
	:param str title: Title for the plot
	:param str xlabel: Label for the x-axis
	:param str ylabel: Label for the y-axis
	"""

    fig, ax = plt.subplots(figsize=figureSize, dpi=dpi)

    ind = numpy.arange(len(R2))
    width = 0.35

    ax.bar(ind, R2, width, color='#3498db', alpha=.4, label='R2')

    if Q2 is not None:
        ax.bar(ind + width, Q2, width, color='#16a085', alpha=.4, label='Q2')

    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_xticks(ind + width)
    ax.set_xticklabels(numpy.arange(1, len(R2) + 1))
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    fig.suptitle(title)

    if savePath:
        plt.savefig(savePath, bbox_inches='tight', format=figureFormat, dpi=dpi)
        plt.close()
    else:
        plt.show()


def plotScores(pcaModel, classes=None, colourType=None,
               colourDict=None, markerDict=None, components=None,
               hotelling_alpha=0.05,
               plotAssociation=None, title='', xlabel='', figures=None,
               savePath=None, figureFormat='png', dpi=72,
               figureSize=(11, 7), opacity=.4):
    """
	Plot PCA scores for each pair of components in PCAmodel, coloured by values defined in classes, and with Hotelling's T2 ellipse (95%)

	:param ChemometricsPCA pcaModel: PCA model object (scikit-learn based)
	:param pandas.Series classes: Measurement/groupings associated with each sample, e.g., BMI/treatment status
	:param str colourType: either 'categorical' or 'continuous'
	:param dict colourDict:
	:param dict markerDict:
	:param components: If ``None`` plots all components in model, else plots those specified in components
	:type components: tuple (int, int)
	:param float hotelling_alpha: Significance value for plotting Hotellings ellipse
	:param bool plotAssociation: If ``True``, plots the association between each set of PCA scores and the metadata values
	:param numpy.array significance: Significance of association of scores from each component with values in classes from correlation or Kruskal-Wallis test for example (see multivariateReport.py)
	:param str title: Title for the plot
	:param str xlabel: Label for the x-axis
	:param dict figures: If not ``None``, saves location of each figure for output in html report (see multivariateReport.py)
	"""

    print("Plotting scores %s" % colourType)
    # Check inputs
    if not isinstance(pcaModel, ChemometricsPCA):
        raise TypeError('PCAmodel must be an instance of ChemometricsPCA')

    # Preparation
    values = pcaModel.scores
    ns, nc = values.shape

    if colourType is not None and colourType not in {'categorical', 'continuous', 'continuousCentered'}:
        raise ValueError('colourType must be == ' + str({'categorical', 'continuous', 'continuousCentered'}))

    if classes is not None and colourType is None:
        raise ValueError('If classes is specified, colourType must be')

    if classes is None:
        classes = pandas.Series('Study Sample' for i in range(ns))
        colourType = 'categorical'

    uniq = classes.unique()
    try:
        uniq.sort()
    except:
        pass
    if colourType == 'categorical':
        classes = classes.astype(str)

    # If colourDict check colour defined for every unique entry in class
    colourDict = checkAndSetPlotAttributes(uniqKeys=uniq, attribDict=colourDict, dictName="colourDict")
    markerDict = checkAndSetPlotAttributes(uniqKeys=uniq, attribDict=markerDict, dictName="markerDict", defaultVal="o")

    from matplotlib.patches import Ellipse

    if components is None:
        components = numpy.ones([nc]).astype(bool)
    components = numpy.where(components == True)
    components = components[0]
    # TODO: fix this so can plot if model only has one component
    if len(components) == 1:
        temp = numpy.arange(0, nc)
        components = numpy.append(components, min(temp[temp != components]))
    nc = len(components)

    if title != '':
        title = title.translate({ord(c): " " for c in "!@#$%^&*()[]{};:,./<>?\|`~-=_+"})
        plotTitle = title + ' '
        title = ''.join(title.split())
    else:
        plotTitle = ''

    # Calculate critical value for Hotelling's T2
    # Fval = f.ppf(0.95, 2, ns-2)
    # Plot scores for each pair of components
    for i in numpy.arange(0, nc, 2):

        if i + 1 >= nc:
            j = 0
        else:
            j = i + 1

        if plotAssociation is not None:
            fig = plt.figure(figsize=figureSize, dpi=dpi)
            gs = gridspec.GridSpec(2, 10)
            ax = plt.subplot(gs[:, 3:])
            ax1 = plt.subplot(gs[0, : 2])
            ax2 = plt.subplot(gs[1, : 2])
        else:
            fig, ax = plt.subplots(figsize=figureSize, dpi=dpi)

        # Add Hotelling's T2
        hotelling_ellipse = pcaModel.hotelling_T2(comps=numpy.array([components[i], components[j]]),
                                                  alpha=hotelling_alpha)

        # a = numpy.sqrt(numpy.var(values[:,components[i]])*Fval*2*((ns-1)/(ns-2)));
        # b = numpy.sqrt(numpy.var(values[:,components[j]])*Fval*2*((ns-1)/(ns-2)));
        ellipse = Ellipse(xy=(0, 0), width=hotelling_ellipse[0] * 2, height=hotelling_ellipse[1] * 2,
                          edgecolor='k', fc='None', lw=2)
        ax.add_patch(ellipse)

        xmin = numpy.minimum(min(values[:, components[i]]), -1 * hotelling_ellipse[0])
        xmax = numpy.maximum(max(values[:, components[i]]), hotelling_ellipse[0])
        ymin = numpy.minimum(min(values[:, components[j]]), -1 * hotelling_ellipse[1])
        ymax = numpy.maximum(max(values[:, components[j]]), hotelling_ellipse[1])

        ax.set_xlim([(xmin + (0.2 * xmin)), xmax + (0.2 * xmax)])
        ax.set_ylim([(ymin + (0.2 * ymin)), ymax + (0.2 * ymax)])

        if colourType == 'categorical':

            # Plot according to user defined colours if available
            if colourDict is not None:
                for u in uniq:
                    ax.scatter(values[classes.values == u, components[i]],
                               values[classes.values == u, components[j]],
                               c=colourDict[u], marker=markerDict[u],
                               label=u, alpha=opacity)

            else:
                colors_sns = {}

                # First plot any nans
                if any(u in {'nan', 'NaN', 'NaT', '', 'NA'} for u in uniq):
                    nans = [i for i, x in enumerate(classes) if x in {'nan', 'NaN', 'NaT', '', 'NA'}]
                    ax.scatter(values[nans, components[i]], values[nans, components[j]], c='#D3D3D3', label='NA')
                    nans = [i for i, x in enumerate(uniq) if x not in {'nan', 'NaN', 'NaT', '', 'NA'}]
                    uniqnonan = uniq[nans]
                    colors_sns['NA'] = '#D3D3D3'

                else:
                    uniqnonan = uniq

                # Then plot remaining classes using rainbow colourmap
                classIX = 0
                colors = iter(plt.cm.rainbow(numpy.linspace(0, 1, len(uniqnonan))))
                for u in uniqnonan:
                    c = rgb2hex(next(colors))
                    if classIX < 20:
                        ax.scatter(values[classes.values == u, components[i]],
                                   values[classes.values == u, components[j]], c=c, label=u,
                                   alpha=opacity)  # olors[classIX], label=u)
                    elif classIX == len(uniqnonan) - 1:
                        ax.scatter(values[classes.values == u, components[i]],
                                   values[classes.values == u, components[j]], c='0', alpha=0, label='...')
                        ax.scatter(values[classes.values == u, components[i]],
                                   values[classes.values == u, components[j]], c=c,
                                   label=u)  # colors[classIX], label=u)
                    else:
                        ax.scatter(values[classes.values == u, components[i]],
                                   values[classes.values == u, components[j]], c=c,
                                   label='_nolegend_', alpha=opacity)  # colors[classIX], label='_nolegend_')
                    classIX = classIX + 1
                    colors_sns[str(u)] = c

            if plotAssociation is not None:
                nonans = [i for i, x in enumerate(classes) if x in {'nan', 'NaN', 'NaT', '', 'NA'}]
                plotClasses = classes.copy()
                plotClasses[nonans] = 'NA'
                tempdata = {plotTitle: plotClasses,
                            'PC' + str(components[i] + 1): values[:, components[i]],
                            'PC' + str(components[j] + 1): values[:, components[j]]}
                tempdata = pandas.DataFrame(tempdata, columns=[plotTitle, 'PC' + str(components[i] + 1),
                                                               'PC' + str(components[j] + 1)])

                # Association for component[i]

                ax1 = sns.stripplot(x=plotTitle, y='PC' + str(components[i] + 1),
                                    data=tempdata, ax=ax1, palette=colors_sns)
                ax1.set(xticklabels=[])
                ax1.set(xlabel='')

                # Association for component[j]
                ax2 = sns.stripplot(x=plotTitle, y='PC' + str(components[j] + 1),
                                    data=tempdata, ax=ax2, palette=colors_sns)
                ax2.set(xticklabels=[])

            ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

        elif colourType == 'continuous':

            plotnans = classes.isnull().values
            if sum(plotnans != 0):
                ax.scatter(values[plotnans == True, components[i]], values[plotnans == True, components[j]],
                           c='#D3D3D3', label='NA')
                ax.legend()

            cb = ax.scatter(values[plotnans == False, components[i]], values[plotnans == False, components[j]],
                            c=classes[plotnans == False], cmap=plt.cm.rainbow, alpha=opacity)
            cbar = plt.colorbar(cb, ax=ax)
            cbar.set_label(title)

            if plotAssociation is not None:
                xvalnan = numpy.unique(classes[plotnans == False])
                xvalnan = xvalnan[-1] + xvalnan[-1] - xvalnan[-2]

                # Association for component[i]
                ax1.scatter(classes[plotnans == False], values[plotnans == False, components[i]],
                            c=classes[plotnans == False], cmap=plt.cm.rainbow, alpha=opacity)
                ax1.scatter(numpy.ones([sum(plotnans), 1]) * xvalnan, values[plotnans, components[i]], c='#D3D3D3')
                ax1.set_ylabel('PC' + str(components[i] + 1))
                ax1.set(xticklabels=[])

                # Association for component[j]
                ax2.scatter(classes[plotnans == False], values[plotnans == False, components[j]],
                            c=classes[plotnans == False], cmap=plt.cm.rainbow, alpha=opacity)
                ax2.scatter(numpy.ones([sum(plotnans), 1]) * xvalnan, values[plotnans, components[j]], c='#D3D3D3')
                ax2.set_xlabel(plotTitle)
                ax2.set_ylabel('PC' + str(components[j] + 1))

        # Annotate
        ylabel = 'PC' + str(components[j] + 1) + ' (' + '{0:.2f}'.format(
            pcaModel.modelParameters['VarExpRatio'][components[j]] * 100) + '%)'
        xlabel = 'PC' + str(components[i] + 1) + ' (' + '{0:.2f}'.format(
            pcaModel.modelParameters['VarExpRatio'][components[i]] * 100) + '%)'
        if plotAssociation is not None:
            ylabel = ylabel + ' significance: ' + '{0:.2f}'.format(plotAssociation[components[j]])
            xlabel = xlabel + ' significance: ' + '{0:.2f}'.format(plotAssociation[components[i]])
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        fig.suptitle(plotTitle + 'PC' + str(components[i] + 1) + ' vs PC' + str(components[j] + 1))

        # Save or show
        if savePath:

            if figures is not None:
                saveTemp = title + 'PC' + str(components[i] + 1) + 'vsPC' + str(components[j] + 1)
                figures[saveTemp] = savePath + saveTemp + '.' + figureFormat
            else:
                saveTemp = ''
            plt.savefig(savePath + saveTemp + '.' + figureFormat, bbox_inches='tight', format=figureFormat, dpi=dpi)
            plt.close()

        else:
            plt.show()

    # Return figures if saving for output in html report
    if figures is not None:
        return figures


def plotOutliers(values, runOrder, addViolin=False, sampleType=None,
                 colourDict=None, markerDict=None, abbrDict=None,
                 fCrit=None, fCritAlpha=None, pCritPercentile=None,
                 title='', xlabel='Run Order', ylabel='', savePath=None,
                 figureFormat='png', dpi=72, figureSize=(11, 7), opacity=.6):
    """
	Plot scatter plot of PCA outlier stats sumT (strong) or DmodX (moderate), with a line at [25, 50, 75, 95, 99] quantiles and at a critical value if specified

	:param numpy.array values: dModX or sum of scores, measure of 'fit' for each sample
	:param numpy.array runOrder: Order of sample acquisition (samples are plotted in this order)
	:param bool addViolin: If True adds a violin plot of distribution of values
	:param pandas.Series sampleType: Sample type of each sample, must be from 'Study Sample', 'Study Reference', 'Long-Term Reference', or 'Sample' (see multivariateReport.py)
	:param dict colourDict:
	:param dict markerDict:
	:param dict abbrDict:

	:param float fCrit: If not none, plots a line at Fcrit
	:param float fCritAlpha: Alpha value for Fcrit (for legend)
	:param float pCritPercentile: If not none, plots a line at this quantile
	:param str title: Title for the plot
	:param str xlabel: Label for the x-axis
	"""

    # Preparation
    if isinstance(sampleType, (str, type(None))):
        sampleType = pandas.Series(['Sample' for i in range(0, len(values))], name='sampleType')

    quantiles = [25, 50, 75, 95, 99]

    # Plot line at PcritPercentile in red if present
    if pCritPercentile is not None:
        if pCritPercentile in quantiles:
            quantiles.remove(pCritPercentile)

    quantilesVals = numpy.percentile(values, quantiles)

    uniq = sampleType.unique()

    colourDict = checkAndSetPlotAttributes(uniqKeys=uniq, attribDict=colourDict, dictName="colourDict",
                                           defaultVal="blue")
    markerDict = checkAndSetPlotAttributes(uniqKeys=uniq, attribDict=markerDict, dictName="markerDict", defaultVal="o")

    if abbrDict is not None:
        if not all(k in abbrDict.keys() for k in uniq):
            raise ValueError(
                'If abbrDict is specified every key should appear in the SampleClass column')
    else:
        abbrDict = {}
        for u in uniq:
            abbrDict[u] = u

    sns.set_color_codes(palette='deep')
    plt.figure(figsize=figureSize, dpi=dpi)
    gs = gridspec.GridSpec(1, 5)

    if addViolin == False:
        ax = plt.subplot(gs[0, :])
    else:
        ax = plt.subplot(gs[0, :-2])
        ax2 = plt.subplot(gs[0, -1])
    sampleMasks = []
    palette = {}

    print("colourDict %s" % colourDict)
    print("markerDict %s" % markerDict)
    # Plot data coloured by sample type
    # TODO: refactor this
    if any(sampleType == 'Study Sample'):
        x = 'Study Sample'
        ax.scatter(runOrder[sampleType.values == x],
                   values[sampleType.values == x],
                   c=colourDict[x],
                   marker=markerDict[x],
                   label=x, alpha=opacity)
        sampleMasks.append((abbrDict[x], sampleType.values == x))
        palette[abbrDict[x]] = colourDict[x]

    if any(sampleType == 'Study Reference'):
        x = 'Study Reference'
        ax.scatter(runOrder[sampleType.values == x],
                   values[sampleType.values == x],
                   c=colourDict[x],
                   marker=markerDict[x],
                   label=x, alpha=opacity)
        sampleMasks.append((abbrDict[x], sampleType.values == x))
        palette[abbrDict[x]] = colourDict[x]

    if any(sampleType == 'Long-Term Reference'):
        x = 'Long-Term Reference'
        ax.scatter(runOrder[sampleType.values == x],
                   values[sampleType.values == x],
                   c=colourDict[x],
                   marker=markerDict[x],
                   label=x, alpha=opacity)
        sampleMasks.append((abbrDict[x], sampleType.values == x))
        palette[abbrDict[x]] = colourDict[x]

    if any(sampleType == 'Sample'):
        x = 'Sample'
        ax.scatter(runOrder[sampleType.values == x],
                   values[sampleType.values == x],
                   label=x, alpha=opacity)
        sampleMasks.append((abbrDict[x], sampleType.values == x))
        palette[abbrDict[x]] = colourDict[x]

    xmin, xmax = ax.get_xlim()

    # TODO: DmodX from pyChemometrics, what about the other measure?

    # Plot lines at quantiles
    for q in numpy.arange(0, len(quantiles)):
        ax.plot([xmin, xmax], [quantilesVals[q], quantilesVals[q]], 'k--', label='Q' + str(quantiles[q]))

    # Add line at Fcrit critical value
    if fCrit:
        ax.plot([xmin, xmax], [fCrit, fCrit], 'c--', label='Fcrit (' + str(fCritAlpha) + ')')

    # Add line at PcritPercentage critical value
    if pCritPercentile:
        Pcrit = numpy.percentile(values, pCritPercentile)
        ax.plot([xmin, xmax], [Pcrit, Pcrit], 'r--', label='Q' + str(pCritPercentile))

    # Annotate
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim([xmin, xmax])

    # If required, violin plot of data distribution
    if addViolin == True:
        limits = ax.get_ylim()

        _violinPlotHelper(ax2, values, sampleMasks, None, 'Sample Type',
                          palette=palette, ylimits=limits, logy=False)

        ax2.yaxis.set_ticklabels([])

        sns.despine(trim=True, ax=ax2)

    # Save or show
    if savePath:
        plt.savefig(savePath, bbox_inches='tight', format=figureFormat, dpi=dpi)
        plt.close()
    else:
        plt.show()


def plotLoadings(pcaModel, msData, title='', figures=None, savePath=None, figureFormat='png', dpi=72,
                 figureSize=(11, 7)):
    """
	Plot PCA loadings for each component in PCAmodel. For NMR data plots the median spectrum coloured by the loading. For MS data plots an ion map (rt vs. mz) coloured by the loading.

	:param ChemometricsPCA pcaModel: PCA model object (scikit-learn based)
	:param Dataset msData: Dataset object
	:param str title: Title for the plot
	:param dict figures: If not ``None``, saves location of each figure for output in html report (see multivariateReport.py)
	"""

    nc = pcaModel.scores.shape[1]

    if ((msData.VariableType.name == 'Discrete') & (hasattr(msData.featureMetadata, 'Retention Time'))):

        Xvals = msData.featureMetadata['Retention Time']
        Xlabel = 'Retention Time'

        Yvals = msData.featureMetadata['m/z']
        Ylabel = 'm/z'

    elif ((msData.VariableType.name == 'Continuum') & (hasattr(msData.featureMetadata, 'ppm'))):

        Xvals = msData.featureMetadata['ppm']
        Xlabel = chr(948) + '1H'

        Yvals = numpy.median(msData.intensityData, axis=0)
        Ylabel = 'Median Intensity'

    elif msData.VariableType.name == 'Discrete':
        compStep = 3

        if savePath:
            saveTemp = title + 'PCAloadings'
            saveTo = savePath + saveTemp + '.' + figureFormat

            if figures is not None:
                figures[saveTemp] = saveTo
        else:
            saveTo = None
        plotDiscreteLoadings(msData, pcaModel, nbComponentPerRow=compStep, savePath=saveTo, figureFormat=figureFormat,
                             dpi=dpi)

        return figures

    else:
        print('add this functionality!!!')

    for i in numpy.arange(0, nc):

        cVect = pcaModel.loadings[i, :]
        orig_cmap = plt.cm.RdYlBu_r  # Red for high, Blue for negative, and we will have a very neutral yellow for 0
        maxval = numpy.max([numpy.abs(numpy.max(cVect)), numpy.abs(numpy.min(cVect))])
        maxcol = maxval  # numpy.max(cVect) # grab the maximum
        mincol = -maxval  # numpy.min(cVect) # Grab the minimum
        # name = 'new_%s' % i
        new_cmap = _shiftedColorMap(orig_cmap, start=0, midpoint=1 - maxcol / (maxcol + numpy.abs(mincol)), stop=1,
                                    name='new')

        fig, ax = plt.subplots(figsize=figureSize, dpi=dpi)

        if ((msData.VariableType.name == 'Discrete') & (hasattr(msData.featureMetadata, 'Retention Time'))):

            # To set the alpha of each point to be associated with the weight of the loading, generate an array where each row corresponds to a feature, the
            # first three columns to the colour of the point, and the last column to the alpha value
            from matplotlib.colors import Normalize
            import matplotlib.cm as cm

            # Return the colours for each feature
            norm = Normalize(vmin=mincol, vmax=maxcol)
            cb = cm.ScalarMappable(norm=norm, cmap=new_cmap)
            cVectAlphas = numpy.zeros((pcaModel.loadings.shape[1], 4))
            cIX = 0
            for c in cVect:
                cVectAlphas[cIX, :] = cb.to_rgba(cVect[cIX])
                cIX = cIX + 1

            # Set the alpha (min 0.2, max 1)
            cVectAlphas[:, 3] = (((abs(cVect) - numpy.min(abs(cVect))) * (1 - 0.2)) / (
                    numpy.max(abs(cVect)) - numpy.min(abs(cVect)))) + 0.2
            if any(cVectAlphas[:, 3] > 1):
                cVectAlphas[cVectAlphas[:, 3] > 1, 3] = 1

            # Plot
            ax.scatter(Xvals, Yvals, color=cVectAlphas)  # , edgecolors='k')
            cb.set_array(cVect)
            ax.set_xlim(
                [min(msData.featureMetadata['Retention Time']) - 1, max(msData.featureMetadata['Retention Time']) + 1])

        elif ((msData.VariableType.name == 'Continuum') & (hasattr(msData.featureMetadata, 'ppm'))):

            # The rasterized ... I don't think it made a big difference, but this was me trying to improve zooming/panning performance. We can compare again without it,
            # as I dont remember if my final conclusion was "rasterized is important therefore leave it " or "doesn't matter leave it"
            ax.set_rasterized(True)

            lvector = cVect
            points = numpy.array([Xvals, Yvals]).transpose().reshape(-1, 1, 2)
            segs = numpy.concatenate([points[:-1], points[1:]], axis=1)

            cb = LineCollection(segs, cmap=new_cmap)
            cb.set_array(lvector)
            plt.gca().add_collection(cb)  # add the collection to the plot
            plt.xlim(Xvals.min() - 0.4, Xvals.max() + 0.4)  # line collections don't auto-scale the plot
            plt.ylim(Yvals.min() * 1.2, Yvals.max() * 1.2)
            plt.gca().invert_xaxis()

        cbar = plt.colorbar(cb)
        cbar.set_label('Loadings')
        ax.set_xlabel(Xlabel)
        ax.set_ylabel(Ylabel)
        fig.suptitle('PCA Loadings for PC' + str(i + 1))

        if savePath:

            if figures is not None:
                saveTemp = title + 'PCAloadingsPC' + str(i + 1)
                figures[saveTemp] = savePath + saveTemp + '.' + figureFormat
            else:
                saveTemp = ''
            plt.savefig(savePath + saveTemp + '.' + figureFormat, bbox_inches='tight', format=figureFormat, dpi=dpi)
            plt.close()

        else:
            plt.show()

    if figures is not None:
        return figures


def plotScoresInteractive(dataset, pcaModel, colourBy, components=[1, 2], alpha=0.05, withExclusions=False,
                          destinationPath=None, autoOpen=True):
    """
	Interactively visualise PCA scores (coloured by a given sampleMetadata field, and for a given pair of components) with plotly, provides tooltips to allow identification of samples.

	:param Dataset dataset: Dataset
	:param PCA object pcaModel: PCA model object (scikit-learn based)
	:param str colourBy: **sampleMetadata** field name to of which values to colour samples by
	:param list components: List of two integers, components to plot
	:param float alpha: Significance value for plotting Hotellings ellipse
	:param bool withExclusions: If ``True``, only report on features and samples not masked by the sample and feature masks; must match between data and pcaModel
	:param str destinationPath: file path to save html version of plot
	:param bool autoOpen: If ``True``, opens html version of plot
	"""

    # Check inputs
    if not isinstance(dataset, Dataset):
        raise TypeError('dataset must be an instance of nPYc.Dataset')

    if not isinstance(pcaModel, ChemometricsPCA):
        raise TypeError('PCAmodel must be a ChemometricsPCA object')

    values = pcaModel.scores
    ns, nc = values.shape

    if colourBy not in dataset.sampleMetadata.columns:
        raise ValueError('colourBy must be a column in dataset.sampleMetadata')

    if (not all(isinstance(item, int) for item in components)) | (len(components) != 2):
        raise TypeError('components must be a list of two integer values')

    if numpy.min(components) < 1:
        raise ValueError('integer value in component can not be less than 1')

    if numpy.max(components) > nc:
        raise ValueError('integer value in component can not exceed the number of components in the model')

    # Reduce components by one (account for python indexing)
    components = [component - 1 for i, component in enumerate(components)]

    # Create destinationPath for saving outputs
    if destinationPath:
        createDestinationPath(destinationPath)

    # If withExclusions=True, apply masks
    dataMasked = copy.deepcopy(dataset)
    if withExclusions:
        dataMasked.applyMasks()

    # Check dimensions match
    if hasattr(pcaModel, '_npyc_dataset_shape'):
        if pcaModel._npyc_dataset_shape['NumberSamples'] != dataMasked.sampleMetadata.shape[0]:
            raise ValueError('Data dimension mismatch: Number of samples and features in the nPYc Dataset do not match'
                             'the numbers present when PCA was fitted. Verify if withExclusions argument is matching.')
    else:
        raise ValueError('Fit a PCA model beforehand using exploratoryAnalysisPCA.')

    # Data preparation
    classes = dataMasked.sampleMetadata[colourBy]
    hovertext = dataMasked.sampleMetadata['Sample File Name'].str.cat(classes.astype(str), sep='; ' + colourBy + ': ')
    plotnans = classes.isnull().values
    data = []

    # Ensure all values in column have the same type

    # list of all types in column; and set of unique types
    mylist = list(type(classes[i]) for i in range(ns))
    myset = set(mylist)

    # if time pass
    if any(my == pandas.Timestamp for my in myset) or any(my == datetime.datetime for my in myset):
        pass

    # else if mixed type convert to string
    elif len(myset) > 1:
        classes = classes.astype(str)

    # Plot NaN values in gray
    if sum(plotnans != 0):
        NaNplot = go.Scatter(
            x=values[plotnans == True, components[0]],
            y=values[plotnans == True, components[1]],
            mode='markers',
            marker=dict(
                color='rgb(180, 180, 180)',
                symbol='circle',
            ),
            text=hovertext[plotnans == True],
            name='NA',
            hoverinfo='text',
            showlegend=True
        )
        data.append(NaNplot)

    # Plot numeric values with a colorbar
    if classes.dtype in (int, float):
        CLASSplot = go.Scatter(
            x=values[plotnans == False, components[0]],
            y=values[plotnans == False, components[1]],
            mode='markers',
            marker=dict(
                colorscale='Portland',
                color=classes[plotnans == False],
                symbol='circle',
                showscale=True
            ),
            text=hovertext[plotnans == False],
            hoverinfo='text',
            showlegend=False
        )
        data.append(CLASSplot)

    # Plot categorical values by unique groups
    else:
        uniq = numpy.unique(classes[plotnans == False])
        for i in uniq:
            CLASSplot = go.Scatter(
                x=values[classes == i, components[0]],
                y=values[classes == i, components[1]],
                mode='markers',
                marker=dict(
                    colorscale='Portland',
                    symbol='circle',
                ),
                text=hovertext[classes == i],
                name=i,
                hoverinfo='text',
                showlegend=True
            )
            data.append(CLASSplot)

    hotelling_ellipse = pcaModel.hotelling_T2(comps=numpy.array([components[0], components[1]]), alpha=alpha)

    layout = {
        'shapes': [
            {
                'type': 'circle',
                'xref': 'x',
                'yref': 'y',
                'x0': 0 - hotelling_ellipse[0],
                'y0': 0 - hotelling_ellipse[1],
                'x1': 0 + hotelling_ellipse[0],
                'y1': 0 + hotelling_ellipse[1],
            }
        ],
        'xaxis': dict(
            title='PC' + str(components[0] + 1) + ' (' + '{0:.2f}'.format(
                pcaModel.modelParameters['VarExpRatio'][components[0]] * 100) + '%)'
        ),
        'yaxis': dict(
            title='PC' + str(components[1] + 1) + ' (' + '{0:.2f}'.format(
                pcaModel.modelParameters['VarExpRatio'][components[1]] * 100) + '%)'
        ),
        'title': 'Coloured by ' + colourBy,
        'legend': dict(
            yanchor='middle',
            xanchor='right'
        ),
        'hovermode': 'closest'
    }

    figure = go.Figure(data=data, layout=layout)

    # Save to destinationPath
    if destinationPath:
        saveTemp = dataset.name + '_PCAscoresPlot_' + colourBy + 'PC' + str(components[0] + 1) + 'vsPC' + str(
            components[1] + 1) + '.html'
        plotly.offline.plot(figure, filename=os.path.join(destinationPath, saveTemp), auto_open=autoOpen)

    return figure


def plotLoadingsInteractive(dataset, pcaModel, component=1, withExclusions=False, destinationPath=None, autoOpen=True):
    """
	Interactively visualise PCA loadings (for a given pair of components) with plotly, provides tooltips to allow identification of features.

	For MS data, plots RT vs. mz; for NMR plots ppm vs spectral intensity. Plots are coloured by the weight of the loadings.

	:param Dataset dataset: Dataset
	:param ChemometricsPCA pcaModel: PCA model object (scikit-learn based)
	:param int component: Component(s) to plot (one component (int) or list of two integers)
	:param bool withExclusions: If ``True``, only report on features and samples not masked by the sample and feature masks; must match between data and pcaModel
	"""

    # Separate one or two components plot
    if isinstance(component, list):
        multiPC = True
    else:
        multiPC = False

    # Check inputs
    if not isinstance(dataset, Dataset):
        raise TypeError('dataset must be an instance of nPYc.Dataset')

    if not isinstance(pcaModel, ChemometricsPCA):
        raise TypeError('PCAmodel must be a ChemometricsPCA object')

    if numpy.min(component) < 1:
        raise ValueError('integer value in component can not be less than 1')

    nc = pcaModel.scores.shape[1]
    if numpy.max(component) > nc:
        raise ValueError('integer value in component can not exceed the number of components in the model')

    # Create destinationPath for saving outputs
    if destinationPath:
        createDestinationPath(destinationPath)

    # If withExclusions=True, apply masks
    dataMasked = copy.deepcopy(dataset)
    if withExclusions:
        dataMasked.applyMasks()

    # NMR doesn't have Feature Name
    if (hasattr(dataMasked.featureMetadata, 'ppm')) and not (hasattr(dataMasked.featureMetadata, 'Feature Name')):
        dataMasked.featureMetadata['Feature Name'] = ["%.4f" % i for i in dataMasked.featureMetadata['ppm']]

    # Check dimensions match
    if hasattr(pcaModel, '_npyc_dataset_shape'):
        if pcaModel._npyc_dataset_shape['NumberFeatures'] != dataMasked.featureMetadata.shape[0]:
            raise ValueError('Data dimension mismatch: Number of samples and features in the nPYc Dataset do not match'
                             'the numbers present when PCA was fitted. Verify if withExclusions argument is matching.')
    else:
        raise ValueError('Fit a PCA model beforehand using exploratoryAnalysisPCA.')

    # check single PC
    if not multiPC:
        if not isinstance(component, int):
            raise TypeError('component must be a single integer value')

        component = component - 1  # Reduce component by one (account for python indexing)

        # Set up colour and tooltip values
        cVect = pcaModel.loadings[component, :]
        W_str = ["%.4f" % i for i in cVect]  # Format text for tooltips
        maxcol = numpy.max(abs(cVect))

    # check multi PC
    else:
        if ((not all(isinstance(item, int) for item in component)) | (len(component) != 2)):
            raise TypeError('component must be a list of two integer values')

        component = [cpt - 1 for i, cpt in
                     enumerate(component)]  # Reduce component by one (account for python indexing)

        # Set up colour and tooltip values
        cVectPC1 = pcaModel.loadings[component[0], :]
        cVectPC2 = pcaModel.loadings[component[1], :]
        PC1_id = [component[0] + 1] * cVectPC1.shape[0]
        PC2_id = [component[1] + 1] * cVectPC2.shape[0]
        WPC1_str = ["%.4f" % i for i in cVectPC1]  # Format text for tooltips first PC
        WPC2_str = ["%.4f" % i for i in cVectPC2]  # Format text for tooltips second PC

    # Set up
    data = []

    # Plot single PC
    if not multiPC:

        # For MS data
        if hasattr(dataMasked.featureMetadata, 'Retention Time'):

            hovertext = ["Feature: %s; W: %s" % i for i in zip(dataMasked.featureMetadata['Feature Name'], W_str)]

            # Convert cVect to a value between 0.1 and 1 - to set the alpha of each point relative to loading weight
            # alphas = (((abs(cVect) - numpy.min(abs(cVect))) * (1 - 0.2)) / (maxcol - numpy.min(abs(cVect)))) + 0.2

            alphas = numpy.fmax(numpy.abs(cVect) / numpy.max(numpy.abs(cVect)), 0.1)

            LOADSplot = go.Scattergl(
                x=dataMasked.featureMetadata['Retention Time'],
                y=dataMasked.featureMetadata['m/z'],
                mode='markers',
                marker=dict(
                    colorscale='RdBu_r',
                    cmin=-maxcol,
                    cmax=maxcol,
                    color=cVect,
                    opacity=alphas,
                    showscale=True,
                ),
                text=hovertext,
                hoverinfo='x, y, text',
                showlegend=False
            )
            data.append(LOADSplot)

            xReverse = True
            Xlabel = 'Retention Time'
            Ylabel = 'm/z'

        # For NMR data
        elif hasattr(dataMasked.featureMetadata, 'ppm'):

            hovertext = ["ppm: %.4f; W: %s" % i for i in zip(dataMasked.featureMetadata['ppm'], W_str)]

            # Bar starts at minimum spectral intensity
            LOADSmin = go.Bar(
                x=dataMasked.featureMetadata['ppm'],
                y=numpy.min(dataMasked.intensityData, axis=0),
                #			y = numpy.percentile(PCAmodel.intensityData, 1, axis=0),
                marker=dict(
                    color='white'
                ),
                hoverinfo='skip',
                showlegend=False
            )
            data.append(LOADSmin)

            # Bar ends at maximum spectral intensity, bar for each feature coloured by loadings weight
            LOADSmax = go.Bar(
                x=dataMasked.featureMetadata['ppm'],
                y=numpy.max(dataMasked.intensityData, axis=0),
                #			y = numpy.percentile(PCAmodel.intensityData, 99, axis=0),
                marker=dict(
                    colorscale='RdBu_r',
                    cmin=-maxcol,
                    cmax=maxcol,
                    color=cVect,
                    showscale=True,
                ),
                text=hovertext,
                hoverinfo='text',
                showlegend=False
            )
            data.append(LOADSmax)

            # Add line for median spectral intensity
            LOADSline = go.Scattergl(
                x=dataMasked.featureMetadata['ppm'],
                y=numpy.median(dataMasked.intensityData, axis=0),
                mode='lines',
                line=dict(
                    color='black',
                    width=1
                ),
                hoverinfo='skip',
                showlegend=False
            )
            data.append(LOADSline)

            xReverse = 'reversed'
            Xlabel = chr(948) + '1H'
            Ylabel = 'Intensity'

        # Other data, X axis is PC loading, Y axis is ordered features
        else:

            sortOrder = numpy.argsort(pcaModel.loadings[component, :])
            Yvals = list(range(pcaModel.loadings.shape[1], 0, -1))
            W_str = numpy.array(W_str)
            W_str = W_str[sortOrder]

            hovertext = ["Feature: %s; W: %s" % i for i in
                         zip(dataMasked.featureMetadata['Feature Name'][sortOrder], W_str)]

            LOADSplot = go.Scattergl(
                x=pcaModel.loadings[component, sortOrder],
                y=Yvals,
                mode='markers',
                text=hovertext,
                hoverinfo='text',
                showlegend=False
            )
            data.append(LOADSplot)

            xReverse = True
            Xlabel = 'Principal Component ' + str(component + 1)
            Ylabel = 'Feature'

        layout = {
            'xaxis': dict(
                title=Xlabel,
                autorange=xReverse
            ),
            'yaxis': dict(
                title=Ylabel
            ),
            'title': 'Loadings for PC ' + str(component + 1),
            'hovermode': 'closest',
            'bargap': 0,
            'barmode': 'stack'
        }

        saveTemp = dataMasked.name + '_PCAloadingsPlot_PC' + str(component + 1) + '.html'

    # Plot multi PC
    else:

        hovertext = ["Feature: %s; W PC%s: %s; W PC%s: %s" % i for i in
                     zip(dataMasked.featureMetadata['Feature Name'], PC1_id, WPC1_str, PC2_id, WPC2_str)]

        LOADSplot = go.Scattergl(
            x=pcaModel.loadings[component[0], :],
            y=pcaModel.loadings[component[1], :],
            mode='markers',
            text=hovertext,
            hoverinfo='text',
            showlegend=False
        )
        data.append(LOADSplot)

        layout = {
            'xaxis': dict(
                title='Principal Component ' + str(component[0] + 1)
            ),
            'yaxis': dict(
                title='Principal Component ' + str(component[1] + 1)
            ),
            'title': 'Loadings for PC ' + str(component[0] + 1) + ' vs. ' + str(component[1] + 1),
            'hovermode': 'closest',
            'bargap': 0,
            'barmode': 'stack'
        }

        saveTemp = dataMasked.name + '_PCAloadingsPlot_PC' + str(component[0] + 1) + 'vsPC' + str(
            component[1] + 1) + '.html'

    figure = go.Figure(data=data, layout=layout)

    # Save to destinationPath
    if destinationPath:
        plotly.offline.plot(figure, filename=os.path.join(destinationPath, saveTemp), auto_open=autoOpen)

    return figure


def plotMetadataDistribution(sampleMetadata, valueType, figures=None, savePath=None, figureFormat='png', dpi=72,
                             figureSize=(11, 7)):
    """
	Plot the distribution of a set of data, e.g., sampleMetadata fields. Plots a bar chart for categorical data, or a histogram for continuous data.

	:param sampleMetadata: Set of measurements/groupings associated with each sample, note can contain multiple columns, but they must be of one **valueType**
	:type sampleMetadata: dataset.sampleMetadata
	:param str valueType: Type of data contained in **sampleMetadata**, one of ``continuous``, ``categorical`` or ``date``
	:param dict figures: If not ``None``, saves location of each figure for output in html report (see multivariateReport.py)
	"""
    import math

    # Check inputs
    if not isinstance(valueType, str) & (valueType in {'categorical', 'continuous', 'date'}):
        raise ValueError('valueType must be == ' + str({'categorical', 'continuous', 'date'}))

    # Set up for plotting in subplot figures 1x2
    nax = 3  # number of axis per figure
    nv = sampleMetadata.shape[1]
    nf = math.ceil(nv / nax)
    plotNo = 0
    field = sampleMetadata.columns

    # Plot
    for figNo in range(nf):

        fig, axIXs = plt.subplots(1, nax, figsize=(figureSize[0], figureSize[1] / nax), dpi=dpi)

        for axNo in range(len(axIXs)):

            if plotNo >= nv:
                axIXs[axNo].axis('off')

            # Continuous plot histogram
            elif valueType == 'continuous':

                pandas.DataFrame.hist(sampleMetadata, column=field[plotNo], ax=axIXs[axNo])
                axIXs[axNo].set_ylabel('Count')
                axIXs[axNo].set_title(field[plotNo])

            # Categorical plot bar
            elif valueType == 'categorical':

                # Define colors (gray for NaNs, cycle through tab10 otherwise)
                classes = sampleMetadata[field[plotNo]].copy()
                nans = [i for i, x in enumerate(classes) if x in {'nan', 'NaN', 'NaT', '', 'NA'}]
                nonans = [i for i, x in enumerate(classes) if x not in {'nan', 'NaN', 'NaT', '', 'NA'}]
                colors = []
                labels = []
                counts = []
                if nans:
                    classes[nans] = 'NA'
                    colors.append('#D3D3D3')
                    labels.append('NA')
                    counts.append(len(nans))
                temp = classes[nonans].value_counts()
                temp.sort_index(inplace=True)

                ix = 0
                cmap = plt.cm.get_cmap('tab10')

                for i in temp.index:
                    colors.append(rgb2hex(cmap(ix)[:3]))
                    labels.append(i)
                    counts.append(temp[i])
                    ix += 1
                    if (ix % 10 == 0):
                        ix = 0

                # If 4 or less classes plot as pie chart
                if len(counts) <= 4:
                    axIXs[axNo].pie(counts, labels=labels, colors=colors, labeldistance=1.05)
                    x0, x1 = axIXs[axNo].get_xlim()
                    y0, y1 = axIXs[axNo].get_ylim()
                    axIXs[axNo].set_aspect(abs(x1 - x0) / abs(y1 - y0))
                    axIXs[axNo].set_ylabel('')

                # Else plot bar chart
                else:
                    axIXs[axNo].bar(numpy.arange(len(counts)), counts, align='center', color=colors, tick_label=labels)
                    axIXs[axNo].set_xticklabels(axIXs[axNo].xaxis.get_majorticklabels(), rotation=90)
                    axIXs[axNo].set_ylabel('Count')

                axIXs[axNo].set_title(field[plotNo])

            # Date
            elif valueType == 'date':

                try:
                    xtime = mdates.date2num(sampleMetadata[field[plotNo]].values)
                    axIXs[axNo].hist(xtime, bins=20)
                    locator = mdates.AutoDateLocator()
                    axIXs[axNo].xaxis.set_major_locator(locator)
                    axIXs[axNo].xaxis.set_major_formatter(
                        mdates.DateFormatter('%d/%m/%y %H:%M'))  # AutoDateFormatter(locator))
                    axIXs[axNo].xaxis.set_tick_params(rotation=90)
                    axIXs[axNo].grid()

                    axIXs[axNo].set_ylabel('Count')
                    axIXs[axNo].set_title(field[plotNo])
                except:
                    pass

            # Advance plotNo
            plotNo = plotNo + 1

        if savePath:
            if figures is not None:
                figures['metadataDistribution_' + valueType + str(
                    figNo)] = savePath + 'metadataDistribution_' + valueType + str(figNo) + '.' + figureFormat

            plt.savefig(savePath + 'metadataDistribution_' + valueType + str(figNo) + '.' + figureFormat,
                        bbox_inches='tight', format=figureFormat, dpi=dpi)
            plt.close()
        else:
            plt.show()

    if figures is not None:
        return figures


def _shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
	From Paul H at Stack Overflow
	http://stackoverflow.com/questions/7404116/defining-the-midpoint-of-a-colormap-in-matplotlib
	Function to offset the "center" of a colormap. Useful for
	data with a negative min and positive max and you want the
	middle of the colormap's dynamic range to be at zero

	Input
	-----
	  cmap : The matplotlib colormap to be altered
	  start : Offset from lowest point in the colormap's range.
		  Defaults to 0.0 (no lower ofset). Should be between
		  0.0 and `midpoint`.
	  midpoint : The new center of the colormap. Defaults to
		  0.5 (no shift). Should be between 0.0 and 1.0. In
		  general, this should be  1 - vmax/(vmax + abs(vmin))
		  For example if your data range from -15.0 to +5.0 and
		  you want the center of the colormap at 0.0, `midpoint`
		  should be set to  1 - 5/(5 + 15)) or 0.75
	  stop : Offset from highets point in the colormap's range.
		  Defaults to 1.0 (no upper ofset). Should be between
		  `midpoint` and 1.0.
	'''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = numpy.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = numpy.hstack([
        numpy.linspace(0.0, midpoint, 128, endpoint=False),
        numpy.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    matplotlib.colormaps.register(newcmap, force=True)
    # plt.register_cmap(cmap=newcmap)

    return newcmap
