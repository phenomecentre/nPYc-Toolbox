import numpy
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import matplotlib.ticker as ticker

from ._rangeFrameLocator import rangeFrameLocator


def blandAltman(data1, data2, interval=1.96, figureSize=(10,7), dpi=72, savePath=None, figureFormat='png'):
	"""
	blandAltman(data1, data2, interval=1.96, **kwargs)

	Generate a Bland-Altman [#]_ plot to compare two sets of measurements of the same value.

	:param data1: First measurement
	:type data1: list like
	:param data1: Second measurement
	:type data1: list like
	:param float interval: Multiple of the standard deviation to plot bounds at (defualt 1.96)

	.. [#] Altman, D. G., and J. M. Bland. “Measurement in Medicine: The Analysis of Method Comparison Studies.” Journal of the Royal Statistical Society. Series D (The Statistician), vol. 32, no. 3, 1983, pp. 307–317. `JSTOR <www.jstor.org/stable/2987937>`.
	"""
	fig, ax = plt.subplots(figsize=figureSize, dpi=dpi)

	mean = numpy.mean([data1, data2], axis=0)
	diff = data1 - data2
	md = numpy.mean(diff)
	sd = numpy.std(diff, axis=0)

	ax.scatter(mean, diff)
	ax.axhline(md, color='#6495ED', linestyle='--')
	ax.axhline(md + interval*sd, color='coral', linestyle='--')
	ax.axhline(md - interval*sd, color='coral', linestyle='--')

	trans = transforms.blended_transform_factory(
		ax.transAxes, ax.transData)

	intervalRange = (md + (interval * sd)) - (md - interval*sd)
	offset = (intervalRange / 100.0) * 1.5

	ax.text(0.98, md + offset, 'Mean', ha="right", va="bottom", transform=trans)
	ax.text(0.98, md - offset, '%.2f' % (interval), ha="right", va="top", transform=trans)

	ax.text(0.98, md + (interval * sd) + offset, '+%.2f SD' % (interval), ha="right", va="bottom", transform=trans)
	ax.text(0.98, md + (interval * sd) - offset, '%.2f' % (md + interval*sd), ha="right", va="top", transform=trans)

	ax.text(0.98, md - (interval * sd) - offset, '-%.2f SD'  % (interval), ha="right", va="top", transform=trans)
	ax.text(0.98, md - (interval * sd) + offset, '%.2f' % (md - interval*sd), ha="right", va="bottom", transform=trans)


	# Only draw spine between extent of the data
	ax.spines['left'].set_bounds(min(diff), max(diff))
	ax.spines['bottom'].set_bounds(min(mean), max(mean))

	# Hide the right and top spines
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)

	ax.set_ylabel('Difference between methods')
	ax.set_xlabel('Mean of methods')

	tickLocs = ax.xaxis.get_ticklocs()
	tickLocs = rangeFrameLocator(tickLocs, (min(mean), max(mean)))
	ax.xaxis.set_major_locator(ticker.FixedLocator(tickLocs))

	tickLocs = ax.yaxis.get_ticklocs()
	tickLocs = rangeFrameLocator(tickLocs, (min(diff), max(diff)))
	ax.yaxis.set_major_locator(ticker.FixedLocator(tickLocs))

	##
	# Save or draw
	##
	if savePath:
		fig.savefig(savePath, format=figureFormat, dpi=dpi)
		plt.close()
	else:
		plt.show()
