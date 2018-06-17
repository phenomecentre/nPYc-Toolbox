import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy
from matplotlib import gridspec
import os

def jointplotRSDvCorrelation(rsd, correlation, histBins=100, savePath=None, figureFormat='png', dpi=72, figureSize=(11, 7)):
	"""
	Plot a 2D histogram of feature RSDs *vs* correlations to dilution, with marginal histograms.

	:param numpy.array rsd: Vector of feature relative standard deviations
	:param numpy.array correlation: Vector of correlation to dilution
	:param int histBins: Number of bins to break the histgram into
	:param savePath: If ``None``, plot interactively, otherwise attempt to save at this path
	:type savePath: None or str
	:param str figureFormat: If saving the plot, use this format
	:param int dpi: Plot resolution
	:param figureSize: Dimensions of the figure
	:type figureSize: tuple(float, float)
	"""
	# Sanitise and check inputs
	if rsd.shape != correlation.shape:
		raise ValueError("rsd and correlation must have the same dimensions")

	# Remove non-finite elements
	maskFinite = numpy.logical_and(numpy.isfinite(rsd), numpy.isfinite(correlation))
	maskFinite = (maskFinite) & (rsd != numpy.finfo(numpy.float64).max)
	
	rsd = rsd[maskFinite]
	correlation = correlation[maskFinite]

	# Set up the grid
	gs = gridspec.GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[1, 4])

	fig = plt.figure(figsize=figureSize, dpi=dpi)
	ax = plt.subplot(gs[1, 0])

	# Plot the scatter

	if min(rsd) <= 0:
		bins = None
		xscale = 'linear'
		nbins = histBins
	else:
		bins = 'log'
		xscale = 'log'
		nbins = 10 ** numpy.linspace(numpy.log10(min(rsd)), numpy.log10(max(rsd)), histBins)

	with sns.axes_style("white"):
		my_cmap = mpl.cm.get_cmap('BuPu')
		my_cmap.set_under('w')
		cax = ax.hexbin(rsd, correlation,
						bins=bins,
						xscale=xscale,
						cmap=my_cmap,
						vmin=.01)

		#Create Y-marginal (right)
		axr = plt.subplot(gs[1, 1],
						  sharey=ax,
						  xticks=[],
						  yticks=[],
						  frameon=False,
						  ylim=(-1, 1))
		axr.hist(correlation,
				 color='#5673E0',
				 orientation='horizontal',
				 bins=100)

		#Create X-marginal (top)
		axt = plt.subplot(gs[0,0],
						  sharex=ax,
						  frameon=False,
						  xticks=[],
						  yticks=[],
						  xlim=(min(rsd), max(rsd)))
		axt.hist(rsd,
				 color='#5673E0',
				 bins=nbins)

		#Bring the marginals closer to the scatter plot
		fig.tight_layout(pad = 1)

		# Format the axes
		cax.axes.set_xscale(xscale)
		cax.axes.set_xlabel('% RSD')
		cax.axes.set_ylabel('Correlation to Dilution')

		cax.axes.set_yticks([-1, -0.5, 0, 0.5, 1])

		cax.axes.tick_params(which='major',
							 bottom=True,
							 top=False,
							 right=False,
							 length=7,
							 width=1.5)
		cax.axes.tick_params(which='minor',
							 bottom=True,
							 top=False,
							 right=False,
							 length=4,
							 width=1)
		cax.axes.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())

		cax.axes.spines['top'].set_visible(False)
		cax.axes.spines['right'].set_visible(False)

		axt.axes.spines['bottom'].set_visible(True)
		axt.axes.spines['bottom'].set_color('k')
		axt.axes.tick_params(labelbottom=False)
		axt.axes.tick_params(which='major',
							 bottom=True,
							 top=False,
							 right=False,
							 length=7,
							 width=1.5)
		axt.axes.tick_params(which='minor',
							 bottom=True,
							 top=False,
							 right=False,
							 length=4,
							 width=1)

		axr.axes.spines['left'].set_visible(True)
		axr.axes.tick_params(left=True,
							 which='major',
							 bottom=False,
							 top=False,
							 right=False,
							 length=7,
							 width=1.5)
		axr.axes.tick_params(labelleft=False)

		if savePath:
			plt.savefig(savePath, bbox_inches='tight', format=figureFormat, dpi=dpi)
			plt.close()
		else:
			plt.show()
