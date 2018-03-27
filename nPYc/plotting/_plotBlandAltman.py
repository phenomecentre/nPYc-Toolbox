import numpy
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import matplotlib.ticker as ticker
from scipy import stats

from ._rangeFrameLocator import rangeFrameLocator


def blandAltman(data1, data2, limitOfAgreement=1.96, confidenceInterval=None, figureSize=(10,7), dpi=72, savePath=None, figureFormat='png'):
	"""
	blandAltman(data1, data2, limitOfAgreement=1.96, confidenceInterval=None, **kwargs)

	Generate a Bland-Altman [#]_ [#]_ plot to compare two sets of measurements of the same value.

	:param data1: First measurement
	:type data1: list like
	:param data1: Second measurement
	:type data1: list like
	:param float limitOfAgreement: Multiple of the standard deviation to plot limit of agreement bounds at (defaults to 1.96)
	:param confidenceInterval: If not ``None``, plot the specified percentage confidence interval on the mean and limits of agreement 
	:type confidenceInterval: None or float

	.. [#] Altman, D. G., and Bland, J. M. “Measurement in Medicine: The Analysis of Method Comparison Studies” Journal of the Royal Statistical Society. Series D (The Statistician), vol. 32, no. 3, 1983, pp. 307–317. `JSTOR <https://www.jstor.org/stable/2987937>`_.
	.. [#] Altman, D. G., and Bland, J. M. “Measuring agreement in method comparison studies” Statistical Methods in Medical Research, vol. 8, no. 2, 1999, pp. 135–160. `DOI <https://doi.org/10.1177/096228029900800204>`_.
	"""

	if not limitOfAgreement > 0:
		raise ValueError('"limitOfAgreement" must be a number greater than zero.') 

	fig, ax = plt.subplots(figsize=figureSize, dpi=dpi)

	mean = numpy.mean([data1, data2], axis=0)
	diff = data1 - data2
	md = numpy.mean(diff)
	sd = numpy.std(diff, axis=0)

	if confidenceInterval:

		if (confidenceInterval > 99.9) | (confidenceInterval < 1):
			raise ValueError('"confidenceInterval" must be a number in the range 1 to 99.')

		n = len(diff)

		confidenceInterval = confidenceInterval / 100.
		confidenceInterval = stats.norm.interval(confidenceInterval, loc=md, scale=sd/numpy.sqrt(n))

		ax.axhspan(confidenceInterval[0],
				   confidenceInterval[1],
				   facecolor='#6495ED', alpha=0.2)

		ciLA = (1/n + ((limitOfAgreement**2 / (2 * (n -1))))) * sd**2

		ax.axhspan((md + limitOfAgreement*sd) - ciLA,
				   (md + limitOfAgreement*sd) + ciLA,
				   facecolor='coral', alpha=0.2)

		ax.axhspan((md - limitOfAgreement*sd) - ciLA,
				   (md - limitOfAgreement*sd) + ciLA,
				   facecolor='coral', alpha=0.2)

	ax.scatter(mean, diff)
	ax.axhline(md, color='#6495ED', linestyle='--')
	ax.axhline(md + limitOfAgreement*sd, color='coral', linestyle='--')
	ax.axhline(md - limitOfAgreement*sd, color='coral', linestyle='--')

	trans = transforms.blended_transform_factory(
		ax.transAxes, ax.transData)

	limitOfAgreementRange = (md + (limitOfAgreement * sd)) - (md - limitOfAgreement*sd)
	offset = (limitOfAgreementRange / 100.0) * 1.5

	ax.text(0.98, md + offset, 'Mean', ha="right", va="bottom", transform=trans)
	ax.text(0.98, md - offset, '%.2f' % (limitOfAgreement), ha="right", va="top", transform=trans)

	ax.text(0.98, md + (limitOfAgreement * sd) + offset, '+%.2f SD' % (limitOfAgreement), ha="right", va="bottom", transform=trans)
	ax.text(0.98, md + (limitOfAgreement * sd) - offset, '%.2f' % (md + limitOfAgreement*sd), ha="right", va="top", transform=trans)

	ax.text(0.98, md - (limitOfAgreement * sd) - offset, '-%.2f SD'  % (limitOfAgreement), ha="right", va="top", transform=trans)
	ax.text(0.98, md - (limitOfAgreement * sd) + offset, '%.2f' % (md - limitOfAgreement*sd), ha="right", va="bottom", transform=trans)

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
