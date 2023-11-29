import numpy
import pandas
import seaborn as sns

def _violinPlotHelper(ax, values, sampleMasks, xlabel, ylabel, palette=None, ylimits=None, logy=True):
	"""
	Draw a violin plot into the axis specified by *ax*.

	:param axis ax: pointer to a pyplot axis handle to draw into
	:param numpy.ndarray values: Values to genrerate lpots from
	:param list sampleMasks: List of ('Class Name', MembershipMask) tuples
	:param xlabel: Label for the x-axis
	:type xlabel: str or None
	:param ylabel: Label for the y-axis
	:type ylabel: str or None
	:param palette: Colours to use for plotting
	:type palette: palette name, list or dict	 
	:param ylimits: Tuple of (min, max) limits for the Y axis
	:type ylimits: None or tuple
	:param bool logy: If ``True`` plot values on a log axis
	"""
	
	localDFpre = numpy.full([len(values),len(sampleMasks)], numpy.nan)
	localDFpre = pandas.DataFrame(data=localDFpre, columns=[i[0] for i in sampleMasks])
	for key, mask in sampleMasks:
		localDFpre.loc[mask, key] = values[mask]

	# Replace infs and -inf used to represent LLOQ and ULOQ in targeted assays.
	#localDFpre.replace([numpy.inf, -numpy.inf], numpy.nan)
	localDFpre.dropna(axis='columns', how='all', inplace=True) # remove empty columns
	sns.set_color_codes(palette='deep')

	if palette is not None:
		sns.violinplot(data=localDFpre, density_norm='width', bw_method=.2, cut=0, ax=ax, palette=palette)
	else:
		sns.violinplot(data=localDFpre, density_norm='width', bw_method=.2, cut=0, ax=ax)

	# ax formatting
	if ylimits:
		ax.set_ylim(ylimits)
	if ylabel:
		ax.set_xlabel(ylabel)
	if logy:
		ax.set_yscale('symlog')
	else:
		ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
