import json

def addReferenceRanges(featureMetadata, referencePath):
	"""
	Adds reference range information from a json file to the :py:attr:`~nPYc.objects.Dataset.featureMetadata` table.

	:param pandas.Dataframe featureMetadata: Pandas dataframe of feature metadata to add reference ranges to
	:param referencePath: Path to a json format file specifying reference ranges
	"""

	with open(referencePath, 'r') as data_file:
		referenceRanges = json.load(data_file)

	for analyte in referenceRanges.keys():
		if analyte in featureMetadata['Feature Name'].values:
			featureMetadata.loc[featureMetadata['Feature Name'] == analyte, 'Upper Reference Bound'] = \
			referenceRanges[analyte]['range'][1][0]
			featureMetadata.loc[featureMetadata['Feature Name'] == analyte, 'Upper Reference Value'] = \
			referenceRanges[analyte]['range'][1][1]

			featureMetadata.loc[featureMetadata['Feature Name'] == analyte, 'Lower Reference Bound'] = \
			referenceRanges[analyte]['range'][0][0]
			featureMetadata.loc[featureMetadata['Feature Name'] == analyte, 'Lower Reference Value'] = \
			referenceRanges[analyte]['range'][0][1]

			featureMetadata.loc[featureMetadata['Feature Name'] == analyte, 'Unit'] = referenceRanges[analyte]['unit']
