import numpy
import seaborn as sns
import matplotlib.pyplot as plt
from pyChemometrics.ChemometricsPCA import ChemometricsPCA
from nPYc.objects import Dataset

def plotDiscreteLoadings(npycDataset, pcaModel, nbComponentPerRow=3, firstComponent=1, metadataColumn='Feature Name', sort=True, savePath=None, figureFormat='png', dpi=72, figureSize=(11, 7)):
	"""
	plotDiscreteLoadings(pcaModel, nbComponentPerRow=3, firstComponent=1, sort=True, **kwargs)

	Plot loadings for a linear model as a set of parallel vertical scatter plots.

	:param ChemometricsPCA pcaModel: Model to plot
	:param int nbComponentPerRow: Number of side-by-side loading plots to place per row
	:param int firstComponent: Start plotting components from this component
	:param bool sort: Plot variable in order of their magnitude in component one
	"""

	if not isinstance(npycDataset, Dataset):
		raise TypeError('npycDataset must be a Dataset object')

	if not isinstance(pcaModel, ChemometricsPCA):
		raise TypeError('pcaModel must be an instance of ChemometricsPCA.')

	if (firstComponent >= pcaModel.ncomps) or (firstComponent <= 0):
		raise ValueError(
			'firstComponent must be greater than zero and less than or equal to the number of components in the model.')

	if sort:
		sortOrder = numpy.argsort(pcaModel.loadings[0, :])
	else:
		sortOrder = numpy.arange(0, pcaModel.loadings.shape[1])

	# Define how many components to plot and how many rows
	firstComponent = firstComponent - 1
	lastComponent = pcaModel.ncomps - 1
	numberComponent = lastComponent - firstComponent + 1
	numberRows = int(numpy.ceil(numberComponent / nbComponentPerRow))

	# It is not possible to plot more than 30 rows clearly, extend the plot height
	extFactor = pcaModel.loadings.shape[1] / 30
	newHeight = figureSize[1] * extFactor
	# Extend by the number of rows
	newHeight = newHeight * numberRows
	figsize = (figureSize[0], newHeight)

	fig, axes = plt.subplots(numberRows, nbComponentPerRow, sharey=True, figsize=figsize, dpi=dpi)

	# Plot each component
	for i in range(firstComponent, lastComponent + 1):
		# grid position
		rowPos = int(numpy.floor((i - firstComponent) / nbComponentPerRow))
		colPos = (i - firstComponent) % nbComponentPerRow

		# different indexing of axes if only 1 row or multiple rows
		if nbComponentPerRow >= numberComponent:
			currentAxes = axes[colPos]
		else:
			currentAxes = axes[rowPos, colPos]

		currentAxes.scatter(pcaModel.loadings[i, sortOrder],
							numpy.arange(0, pcaModel.loadings.shape[1]),
							s=100,
							c=numpy.absolute(pcaModel.loadings[i, sortOrder]),
							linewidths=1,
							edgecolor='none',
							cmap=plt.get_cmap('plasma'),
							marker='o',
							zorder=10)

		currentAxes.axvline(x=0, zorder=1)
		currentAxes.set_title('PC %i' % (i + 1))
		currentAxes.set_xlabel('%.2f%%' % (pcaModel.modelParameters['VarExpRatio'][i] * 100))

		# Add y-label to first plot of row
		if rowPos == 0:
			currentAxes.axes.set_yticks(numpy.arange(0, pcaModel.loadings.shape[1]))
			currentAxes.axes.set_yticklabels(npycDataset.featureMetadata[metadataColumn].values[sortOrder])
			currentAxes.set_ylim((-0.5, pcaModel.loadings.shape[1] - 0.5))

	# Random 'ValueError: bottom cannot be >= top' from mpl which they cannot reliably correct
	try:
		plt.tight_layout()
	except ValueError:
		pass

	##
	# Save or draw
	##
	if savePath:
		fig.savefig(savePath, format=figureFormat, dpi=dpi)
		plt.close()
	else:
		plt.show()
