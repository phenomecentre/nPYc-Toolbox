import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import numpy

from ..objects._dataset import Dataset
from ..utilities._checkInRange import checkInRange

def plotFeatureRanges(dataset, compounds, logx=False, histBins=20, savePath=None, figureFormat='png', dpi=72, figureSize=(4, 7)):
	"""
	plotFeatureRanges(dataset, compounds, logx=False, histBins=20, **kwargs)
	
	Plot distributions plots of the values listed in **compounds**, on to a set of axes with a linked x-axis.

	If reference ranges are specified in :py:attr:`~nPYc.objects.Dataset.featureMetadata`, a reference range will be drawn behind each plot. If reference ranges are available, distributions that for within the range will be shaded green, and those that fall outside red, where no reference range is available the distribution will be shaded blue.

	:param Dataset dataset: Dataset object to plot values from 
	:param list compounds: List of features to plot
	:param bool logx: Calculate and plot histograms on a log10 scale, if the minumn values is below 1, the histogram is calculated by adding one to all values
	:param int histBins: Number of bins for histograms
	"""

	if not isinstance(dataset, Dataset):
		raise TypeError('dataset must be an instance of Dataset.')

	with sns.axes_style("whitegrid", rc={'axes.grid': False}):

		width = figureSize[0]
		height = 0.6 * len(compounds)
		fig, ax = plt.subplots(len(compounds), 1, sharex=True, figsize=(width, height))

		globalMinV = numpy.finfo(numpy.float64).max
		globalMaxV = 0
		globalMax = 0

		if not isinstance(ax, numpy.ndarray):
			ax = [ax]

		for compound in compounds:
			if compound in dataset.featureMetadata['Feature Name'].values:
				featureIndex = dataset.featureMetadata.loc[dataset.featureMetadata['Feature Name'] == compound].index[0]

				values = dataset.intensityData[:,featureIndex]

				if numpy.min(values) < globalMinV:
					globalMinV = numpy.nanmin(values)
				if numpy.max(values) > globalMaxV:
					globalMaxV = numpy.nanmax(values)
	
		if globalMinV == globalMaxV:
			logx = False
			globalMaxV = globalMaxV + 1
	
		if logx == True:

			if globalMinV < 1:
				offset = 1

				globalMinV = globalMinV + offset
				globalMaxV = globalMaxV + offset

			else:
				offset = 0

			if globalMinV < 0:
				logx = False
				nbins = histBins
				xscale = 'linear'
			else:
				nbins = 10 ** numpy.linspace(numpy.log10(globalMinV), numpy.log10(globalMaxV), histBins)
				xscale = 'log'
		else:
			nbins = numpy.linspace(globalMinV, globalMaxV, histBins)
			xscale = 'linear'

		for i in range(len(compounds)):

			if compounds[i] in dataset.featureMetadata['Feature Name'].values:
				featureIndex = dataset.featureMetadata.loc[dataset.featureMetadata['Feature Name'] == compounds[i]].index[0]
				feature = dataset.featureMetadata.loc[dataset.featureMetadata['Feature Name'] == compounds[i]]

				values = dataset.intensityData[:,featureIndex]

				if logx:
					values = values + offset

				if {'Upper Reference Bound', 'Upper Reference Value', 'Lower Reference Bound', 'Lower Reference Value'}.issubset(feature.columns):

					minV = feature['Lower Reference Value'].values[0]
					maxV = feature['Upper Reference Value'].values[0]

					# Interpret '-' as no lower boud i.e. 0
					if minV == '-':
						minV = 0

					if numpy.isfinite(minV) and numpy.isfinite(maxV):
						plotRange = True

						if logx:
							minV = minV + offset
							maxV = maxV + offset

						lowerRange = (feature['Lower Reference Bound'].values[0], minV)
						upperRange = (feature['Upper Reference Bound'].values[0], maxV)
						if checkInRange(values, lowerRange, upperRange):
							plotColour = 'g'
						else:
							plotColour = 'r'
					else:
						plotRange = False
						plotColour = 'b'
				else:
					plotRange = False
					plotColour = 'b'
				if max(values) > globalMax:
					globalMax = max(values)

				ax[i].hist(values,
						bins=nbins,
						color=plotColour)

				if plotRange:
					ax[i].axvspan(minV, maxV, zorder=-100, color='#ebecfc')

				ax[i].spines['right'].set_visible(False)
				ax[i].spines['left'].set_visible(False)
				ax[i].spines['top'].set_visible(False)
				ax[i].set_ylabel(compounds[i], rotation='horizontal', fontsize=15, ha='right', va='center')
				ax[i].set_yticks([])

			else:
				ax[i].axis('off')

		if globalMaxV > globalMax:
			globalMax = globalMaxV

		ax[i].set_xlim([globalMinV, globalMaxV])
		ax[i].tick_params(labelsize=13)

		if logx:
			formatter = matplotlib.ticker.FormatStrFormatter('%.2g')
			ax[i].xaxis.set_major_formatter(formatter)

			ax[i].set_xscale('symlog')

			majorTickLocations = [globalMinV]
			position = 10
			while position < globalMaxV:
				majorTickLocations.append(position + offset)
				position = position * 10
			majorTickLocations.append(globalMaxV)

			majorTickLocations = numpy.asarray(majorTickLocations)

			ax[i].xaxis.set_major_formatter(ticker.FuncFormatter(lambda majorTickLocations, pos: '{0:g}'.format(majorTickLocations-offset)))
			ax[i].xaxis.set_major_locator(ticker.FixedLocator(majorTickLocations))
			for tick in ax[i].get_xticklabels():
				tick.set_rotation(-90)

			ax[i].xaxis.set_minor_formatter(ticker.NullFormatter())
			ax[i].tick_params(axis='x', which='both', length=4)

			minorTickLocations = ax[i].xaxis.get_minor_locator().tick_values(globalMinV, globalMaxV)
			ax[i].xaxis.set_minor_locator(ticker.FixedLocator(minorTickLocations - offset))

		else:
			ax[i].set_xticks([globalMinV, globalMaxV])

			ax[i].xaxis.set_minor_formatter(ticker.NullFormatter())
			ax[i].tick_params(axis='x', which='minor', length=4)

		if savePath:
			plt.savefig(savePath, bbox_inches='tight', format=figureFormat, dpi=dpi)
			plt.close()
		else:
			plt.show()

