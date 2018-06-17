import os
import copy

from ..plotting import plotFeatureRanges

def generateFeatureDistributionReport(dataset, reportingGroups, logScale=False, filterUnits=None, destinationPath=None):
	"""
	Plots distributions of features. If reference ranges are present in :py:attr:`~nPYc.objects.Dataset.featureMetadata` they will be indicated on the plot.

	Plotted feature ranges will be grouped according to the dictionary specified in **reportingGroups**:
	
	.. code-block:: python
		{
			'Summary Group':{
				'Collection of features plotted on a common axis':['VLTG', 'IDTG', 'LDTG', 'HDTG'],
				'Second collection':['VLCH', 'IDCH', 'LDCH', 'HDCH'],
				'Feature plotted in isolation':['HDA1'],
			},
			'Second group':{
				'Free Cholesterol':['V1FC', 'V2FC', 'V3FC', 'V4FC', 'V5FC', 'V6FC'],
				'Cholesterol':['V1CH', 'V2CH', 'V3CH', 'V4CH', 'V5CH', 'V6CH'],
				'Phospholipids':['V1PL', 'V2PL', 'V3PL', 'V4PL', 'V5PL', 'V6PL'],
				'Triglycerides':['V1TG', 'V2TG', 'V3TG', 'V4TG', 'V5TG', 'V6TG']
			}
		}

	Plots will be grouped according to the top-level dictionaries, while individual features will be plotted onto common axes according to the names listed in the sub-dictionary.

	:param dict reportingGroups: Dictionary of dictionaries of lists of metabolites
	:param filterUnits: If specified only report on measutments with this unit
	:type filterUnits: None or str
	:param destinationPath: If ``None`` print interactively, otherwise a path to save destinationPath to
	:param bool logScale: If ``True`` plot distributions on a log scale 
	:param filterUnits: If not ``None``, only plot ranges for features with a unit that matches the value provided
	:type filterUnits: None or str 
	:type destinationPath: None or str
	:returns: Dictionary of plot paths
	:rtype: dict
	"""

	returnDict = dict()

	if destinationPath is not None and (not os.path.exists(os.path.join(destinationPath, 'graphics'))):
		os.makedirs(os.path.join(destinationPath, 'graphics'))

	if filterUnits:
		dataset = copy.deepcopy(dataset)

		unitMask = dataset.featureMetadata['Unit'].values == filterUnits

		dataset.intensityData = dataset.intensityData[:, unitMask]

		dataset.featureMetadata = dataset.featureMetadata.loc[unitMask, :]
		dataset.featureMetadata.reset_index(inplace=True, drop=True)

	featuresSet = set(dataset.featureMetadata['Feature Name'].values)

	for key in reportingGroups.keys():

		returnDict[key] = dict()

		if not destinationPath:
			print(key)

		for key2 in reportingGroups[key].keys():

			if set(reportingGroups[key][key2]) <= featuresSet:
				if destinationPath:
					figureName = 'FeatureDistribution_' + key + '-' + key2 + '.' + dataset.Attributes['figureFormat']
					saveAs = os.path.join(destinationPath, 'graphics', figureName)
					returnDict[key][key2] = saveAs

				else:
					print(key2)
					saveAs = None

				plotFeatureRanges(dataset,
								  reportingGroups[key][key2],
								  logx=logScale,
								  figureFormat=dataset.Attributes['figureFormat'],
								  dpi=dataset.Attributes['dpi'],
								  savePath=saveAs)

	return returnDict
