import pandas

def loadSampleManifest(path):
	"""
	Load the sample manifest out of an excel spreadsheet.

	:param str path: Path to the manifest file
	:returns: Flattened manifest
	:rtype: pandas.DataFrame
	"""

	table = dict()
	# Subjects
	table['subjects'] = pandas.read_excel(path, 'Subject Info', header=0)

	# Sampling events
	table['samplings'] = pandas.read_excel(path, 'Sampling Events', header=0)

	table = parseRelationships(table['subjects'], table['samplings'])

	return table


def parseRelationships(subjectsTable, samplingsTable):
	"""
	Parse sampling to subject relationships into a flat table.

	:param subjectsTable: Table of subject information and associated metadata.
	:type subjectsTable: pandas.DataFrame
	:param samplingsTable: Table of sampling events and associated metadata.
	:type samplingsTable: pandas.DataFrame
	:returns: Flattened manifest
	:rtype: pandas.DataFrame
	"""

	subjectsRequired = ['Subject ID']
	samplingRequired = ['Subject ID', 'Sampling ID']

	# Check for mapping columns
	for columnID in subjectsRequired:
		if columnID not in subjectsTable.columns:
			raise LookupError(columnID + ' not a column header in subjectsTable.')

	for columnID in samplingRequired:
		if columnID not in samplingsTable.columns:
			raise LookupError(columnID + ' not a column header in samplingsTable.')

	# Assemble the output table
	output = pandas.merge(samplingsTable, subjectsTable, how='left', on='Subject ID',suffixes=('_samplings', '_subjects'))

	return output
