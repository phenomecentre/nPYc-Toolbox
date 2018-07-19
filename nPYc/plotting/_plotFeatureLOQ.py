import sys
import os
import numpy
import pandas
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import warnings
from matplotlib import gridspec
from ..objects import TargetedDataset
from ._plotLOQFeatureViolin import _featureLOQViolinPlotHelper


def plotFeatureLOQ(tData, splitByBatch=True, plotBatchLOQ=False, zoomLOQ=False, logY=False, tightYLim=True, nbPlotPerRow=3, metabolitesPerPlot=5, withExclusions=True, savePath=None, figureFormat='png', dpi=72, figureSize=(11, 7)):
	"""
	Violin plot for each feature with line at LOQ concentrations. Option to split by batch, add each batch LOQs, split by SampleType.

	:param TargetedDataset tData: :py:class:`TargetedDataset`
	:param bool splitByBatch: If ``True`` separate each violin plot by batch
	:param bool plotBatchLOQ: If ``True`` add lines at LOQs (LLOQ/ULOQ) for each batch, and points for samples that will be out of LOQ
	:param bool zoomLOQ: If ``True`` plots a zoomed ULOQ plot on top, all data in the centre and a zoomed LLOQ plot at the bottom
	:param bool logY: If ``True`` log-scale the y-axis
	:param bool tightYLim: if ``True`` ylim are close to the points but can let LOQ lines outside, if ``False`` LOQ lines will be part of the plot.
	:param int nbPlotPerRow: Number of plots to place on each row
	:param int metabolitesPerPlot: Maximum numper of metabolites to plot in on single figure
	:param savePath: If ``None`` plot interactively, otherwise save the figure to the path specified
	:type savePath: None or str
	:param str figureFormat: If saving the plot, use this format
	:param int dpi: Plot resolution
	:param figureSize: Dimensions of the figure
	:type figureSize: tuple(float, float)
	:raises ValueError: if targetedData does not satisfy to the TargetedDataset definition for QC
	"""

	# Check dataset is fit for plotting
	tmpTData = copy.deepcopy(tData)	 # to not log validateObject
	validDataset = tmpTData.validateObject(verbose=False, raiseError=False, raiseWarning=False)
	if not validDataset['QC']:
		raise ValueError('Import Error: tData does not satisfy to the TargetedDataset definition for QC')

	if withExclusions:
		tmpTData.applyMasks()

	figurePaths = list()
	startPlot = 0

	with warnings.catch_warnings():
		warnings.simplefilter('ignore')
		while startPlot < tmpTData.noFeatures:
			evenMoreTmpTData = copy.deepcopy(tmpTData)

			endPlot = startPlot + metabolitesPerPlot
			if endPlot > evenMoreTmpTData.noFeatures:
				endPlot = evenMoreTmpTData.noFeatures

			evenMoreTmpTData.featureMask[:] = False
			evenMoreTmpTData.featureMask[startPlot:endPlot] = True
			evenMoreTmpTData.applyMasks()

			if savePath:
				saveTo = savePath + "_features_%i_to_%i" % (startPlot, endPlot)  + '.' + figureFormat
				figurePaths.append(saveTo)
			else:
				saveTo = None

			_plotFeatureLOQHelper(evenMoreTmpTData,
								 splitByBatch=splitByBatch,
								 plotBatchLOQ=plotBatchLOQ,
								 zoomLOQ=zoomLOQ,
								 logY=logY,
								 tightYLim=tightYLim,
								 nbPlotPerRow=nbPlotPerRow,
								 savePath=saveTo,
								 figureFormat=figureFormat,
								 dpi=dpi,
								 figureSize=figureSize)

			startPlot += metabolitesPerPlot

	return figurePaths


def _plotFeatureLOQHelper(tData, splitByBatch=True, plotBatchLOQ=False, zoomLOQ=False, logY=False, tightYLim=True, nbPlotPerRow=3, savePath=None, figureFormat='png', dpi=72, figureSize=(11, 7)):


	# Plot setup
	nbFeat = tData.noFeatures + 1				   # allow a spot for the legend box
	nbRow = int(numpy.ceil(nbFeat / nbPlotPerRow))
	newHeight = figureSize[1] * nbRow			   # extend the plot height to allow for the multiple rows
	# Allow vertical space for subplots
	if zoomLOQ:
		realNbRow = 7 * nbRow					   # 6 for plot (1 ULOQ, 4 all, 1 LLOQ) + 1 x-axis label and title
	else:
		realNbRow = nbRow

	sns.set_style("ticks", {'axes.linewidth': 0.75})
	sns.set_color_codes(palette='deep')
	fig = plt.figure(figsize=(figureSize[0], newHeight), dpi=72)
	gs = gridspec.GridSpec(realNbRow, nbPlotPerRow)

	# With LOQ subplots
	if zoomLOQ:
		# Loop over features (and a space to fit a legend), with zoomed subplots
		for featID in range(0, nbFeat):
			# Keep track of plot position
			plot_vert_pos = int(numpy.floor(featID / nbPlotPerRow) * 7)	  # jump 7 every time, 6 for plot + 1 for x-axis and title
			plot_horz_pos = featID - int(numpy.floor(featID / nbPlotPerRow) * nbPlotPerRow)

			# Init plot row and column position (ULOQ subplot = top 1/6th height, main plot 4/6th, LLOQ bottom 1/6th)
			if featID < (nbFeat - 1):
				ax_ULOQ = plt.subplot(gs[plot_vert_pos, plot_horz_pos])
				ax_all	= plt.subplot(gs[(plot_vert_pos + 1):(plot_vert_pos + 5), plot_horz_pos])
				ax_LLOQ = plt.subplot(gs[(plot_vert_pos + 5), plot_horz_pos])
			else:
				ax_leg = plt.subplot(gs[(plot_vert_pos):(plot_vert_pos + 7), plot_horz_pos])  # legend use all the height available

			# plot the features
			if featID < (nbFeat - 1):
				# Feature name and unit
				featName = tData.featureMetadata.loc[featID, 'Feature Name'] + " - (" + tData.featureMetadata.loc[featID, 'Unit'] + ")"

				# Detect if it's the first or last plot in the row, to add y-label
				if plot_horz_pos == 0:
					isFirstInRow = True
				else:
					isFirstInRow = False
				# Detect if it's the last in a line (or just last one), to put subplot names on the right hand side
				if (plot_horz_pos == (nbPlotPerRow - 1)) | (featID == (nbFeat - 1)):
					isLastInRow = True
				else:
					isLastInRow = False
				# x-axis label change depending on splitByBatch
				if splitByBatch:
					xLab = 'Batch'
				else:
					xLab = 'Sample Type'

				# Plot
				if isFirstInRow:  # y labels on the left
					_featureLOQViolinPlotHelper(ax=ax_all,	tData=tData, featID=featID, splitByBatch=splitByBatch, plotBatchLOQ=plotBatchLOQ, title=None,	  xLabel=None, xTick=False, yLabel='Concentration', yTick=True, subplot=None,	flipYLabel=False, logY=logY, tightYLim=tightYLim, showLegend=False, onlyLegend=False)
					_featureLOQViolinPlotHelper(ax=ax_ULOQ, tData=tData, featID=featID, splitByBatch=splitByBatch, plotBatchLOQ=plotBatchLOQ, title=featName, xLabel=None, xTick=False, yLabel=None,			yTick=True, subplot='ULOQ', flipYLabel=False, logY=logY, tightYLim=tightYLim, showLegend=False, onlyLegend=False)
					_featureLOQViolinPlotHelper(ax=ax_LLOQ, tData=tData, featID=featID, splitByBatch=splitByBatch, plotBatchLOQ=plotBatchLOQ, title=None,	  xLabel=xLab, xTick=True,	yLabel=None,			yTick=True, subplot='LLOQ', flipYLabel=False, logY=logY, tightYLim=tightYLim, showLegend=False, onlyLegend=False)
				elif isLastInRow:  # y labels on the right
					_featureLOQViolinPlotHelper(ax=ax_all,	tData=tData, featID=featID, splitByBatch=splitByBatch, plotBatchLOQ=plotBatchLOQ, title=None,	  xLabel=None, xTick=False, yLabel=None,   yTick=True, subplot=None,   flipYLabel=False, logY=logY, tightYLim=tightYLim, showLegend=False, onlyLegend=False)
					_featureLOQViolinPlotHelper(ax=ax_ULOQ, tData=tData, featID=featID, splitByBatch=splitByBatch, plotBatchLOQ=plotBatchLOQ, title=featName, xLabel=None, xTick=False, yLabel='ULOQ', yTick=True, subplot='ULOQ', flipYLabel=True,	 logY=logY, tightYLim=tightYLim, showLegend=False, onlyLegend=False)
					_featureLOQViolinPlotHelper(ax=ax_LLOQ, tData=tData, featID=featID, splitByBatch=splitByBatch, plotBatchLOQ=plotBatchLOQ, title=None,	  xLabel=xLab, xTick=True,	yLabel='LLOQ', yTick=True, subplot='LLOQ', flipYLabel=True,	 logY=logY, tightYLim=tightYLim, showLegend=False, onlyLegend=False)
				else:  # no y labels
					_featureLOQViolinPlotHelper(ax=ax_all,	tData=tData, featID=featID, splitByBatch=splitByBatch, plotBatchLOQ=plotBatchLOQ, title=None,	  xLabel=None, xTick=False, yLabel=None, yTick=True, subplot=None,	 flipYLabel=False, logY=logY, tightYLim=tightYLim, showLegend=False, onlyLegend=False)
					_featureLOQViolinPlotHelper(ax=ax_ULOQ, tData=tData, featID=featID, splitByBatch=splitByBatch, plotBatchLOQ=plotBatchLOQ, title=featName, xLabel=None, xTick=False, yLabel=None, yTick=True, subplot='ULOQ', flipYLabel=False, logY=logY, tightYLim=tightYLim, showLegend=False, onlyLegend=False)
					_featureLOQViolinPlotHelper(ax=ax_LLOQ, tData=tData, featID=featID, splitByBatch=splitByBatch, plotBatchLOQ=plotBatchLOQ, title=None,	  xLabel=xLab, xTick=True,	yLabel=None, yTick=True, subplot='LLOQ', flipYLabel=False, logY=logY, tightYLim=tightYLim, showLegend=False, onlyLegend=False)
					# The last is the Legend
			else:
				_featureLOQViolinPlotHelper(ax=ax_leg, tData=tData, featID=None, splitByBatch=splitByBatch, plotBatchLOQ=plotBatchLOQ, title=None, xLabel=None, xTick=False, yLabel=None, yTick=False, subplot=None, flipYLabel=False, logY=logY, tightYLim=tightYLim, showLegend=False, onlyLegend=True)
			# No tight layout with subplots

	# Without LOQ subplots
	else:
		# Loop over features (and a space to fit a legend), single plot
		for featID in range(0, nbFeat):
			# Keep track of plot position
			plot_vert_pos = int(numpy.floor(featID / nbPlotPerRow))
			plot_horz_pos = featID - int(numpy.floor(featID / nbPlotPerRow) * nbPlotPerRow)

			# Define ax
			ax_single = plt.subplot(gs[plot_vert_pos, plot_horz_pos])

			# plot the features
			if featID < (nbFeat - 1):
				# Feature name and unit
				featName = tData.featureMetadata.loc[featID, 'Feature Name'] + " - (" + tData.featureMetadata.loc[featID, 'Unit'] + ")"

				# Detect if it's the first or last plot in the row, to add y-label
				if plot_horz_pos == 0:
					isFirstInRow = True
				else:
					isFirstInRow = False
				# x-axis label change depending on splitByBatch
				if splitByBatch:
					xLab = 'Batch'
				else:
					xLab = 'Sample Type'

				# Plot
				if isFirstInRow:
					_featureLOQViolinPlotHelper(ax=ax_single, tData=tData, featID=featID, splitByBatch=splitByBatch, plotBatchLOQ=plotBatchLOQ, title=featName, xLabel=xLab, xTick=True, yLabel='Concentration', yTick=True, subplot=None, flipYLabel=False, logY=logY, tightYLim=tightYLim, showLegend=False, onlyLegend=False)
				else:
					_featureLOQViolinPlotHelper(ax=ax_single, tData=tData, featID=featID, splitByBatch=splitByBatch, plotBatchLOQ=plotBatchLOQ, title=featName, xLabel=xLab, xTick=True, yLabel=None, yTick=True, subplot=None, flipYLabel=False, logY=logY, tightYLim=tightYLim, showLegend=False, onlyLegend=False)
			# Plot the Legend
			else:
				_featureLOQViolinPlotHelper(ax=ax_single, tData=tData, featID=None, splitByBatch=splitByBatch, plotBatchLOQ=plotBatchLOQ, title=None, xLabel=None, xTick=False, yLabel=None, yTick=False, subplot=None, flipYLabel=False, logY=logY, tightYLim=tightYLim, showLegend=False, onlyLegend=True)
		# Tight layout
	fig.tight_layout()

	# Save or output
	if savePath:
		plt.savefig(savePath, bbox_inches='tight', format=figureFormat, dpi=dpi)
		plt.close()
	else:
		plt.show()
