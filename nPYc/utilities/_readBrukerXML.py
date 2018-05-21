from xml.etree import ElementTree
import pandas
import re
import numpy
import copy
import warnings

def importBrukerXML(filelist):
	"""
	Load Bruker quantification data from the xml files listed in *fileList*, and return as data matrices.

	Files that cannot be opened raise warnings, and are filtered from the returned matrices.

	TODO: Reconcile LODS, LOQS, and ranges while importing

	:param list filelist: List of paths to load data from
	:return: intensityData, sampleMetadata, and featureMetadata
	:rtype: tuple of (intensityData, sampleMetadata, featureMetadata)
	"""

	sampleMetadata = pandas.DataFrame([], columns=['Sample File Name', 'Sample Base Name', 'expno', 'Path', 'Acquired Time', 'Run Order'])
	sampleMetadata['Path'] = filelist
	sampleMetadata['Acquired Time'] = None
	sampleMetadata['Run Order'] = None
	intensityData = None
	featureMetadata = None

	importPass = numpy.ones(len(filelist), dtype=bool)

	nameParser = re.compile(r'^(.+?)_expno(\d+)\..+?$')

	for filename in filelist:
		try:
			sampleName, processingDate, quantList = readBrukerXML(filename)

			df = pandas.DataFrame.from_dict(quantList)

			if intensityData is None:
				intensityData = numpy.zeros((len(filelist), len(quantList)))
				featureMetadata = copy.deepcopy(df)
				featureMetadata.drop('value', inplace=True, axis=1)

			baseName = nameParser.match(sampleName).groups()

			sampleMetadata.loc[sampleMetadata['Path'] == filename, 'Sample Base Name'] = baseName[0] + '/' + baseName[1]
			#sampleMetadata.loc[sampleMetadata['Path'] == filename, 'Sample File Name'] = sampleName  # Sample File Name should match Base Name, instead of the Sample File Name hardcoded in the XML file
			sampleMetadata.loc[sampleMetadata['Path'] == filename, 'Sample File Name'] = baseName[0] + '/' + baseName[1]
			sampleMetadata.loc[sampleMetadata['Path'] == filename, 'expno'] = baseName[1]
			sampleMetadata.loc[sampleMetadata['Path'] == filename, 'Acquired Time'] = processingDate

			intensityData[sampleMetadata.loc[sampleMetadata['Path'] == filename].index.values, :] = df['value']
		except ElementTree.ParseError:
			warnings.warn('Error parsing xml in %s, skipping' % filename)

			importPass[sampleMetadata.loc[sampleMetadata['Path'] == filename].index.values] = False

	runOrder = sampleMetadata.sort_values(by='Acquired Time').index.values
	sampleMetadata['Run Order'] = numpy.argsort(runOrder)

	##
	# Drop failed imports
	##
	if intensityData is not None:
		intensityData = intensityData[importPass, :]
	sampleMetadata = sampleMetadata.loc[importPass, :]
	sampleMetadata.reset_index(drop=True, inplace=True)

	sampleMetadata['expno'] = pandas.to_numeric(sampleMetadata['expno'])
	sampleMetadata['Acquired Time'] = pandas.to_datetime(sampleMetadata['Acquired Time'])

	return (intensityData, sampleMetadata, featureMetadata)


def readBrukerXML(path):
	"""
	Extract Bruker quatification data from the XML file at *path* and return as a dict, with one element for each value.

	:param str path: Path to Bruker XML quantification report
	:returns: List of dicts
	:rtype: tuple of (filename, processing date, dict of measurements)
	"""

	tree = ElementTree.parse(path)
	root = tree.getroot()
	quantData = root.find('QUANTIFICATION')

	sampleName = root.find('SAMPLE').attrib['name']
	processingDate = root.find('SAMPLE').attrib['date']

	root.find('SAMPLE').attrib['name']

	quantList = list()

	for analyte in quantData:

		name = analyte.attrib['name']
		if 'comment' in analyte.attrib.keys():
			comment = analyte.attrib['comment']
		else:
			comment = ''

		qtype = analyte.attrib['type']

		for element in analyte.findall('VALUE'):
			reference = analyte.find('REFERENCE')

			refDict = dict()
			if reference is not None:
				if element.attrib['unit'] == reference.attrib['unit']:
					# Implicitly 2.5 and 97.5 in lipo xml
					refDict['Lower Reference Bound'] = 2.5
					refDict['Upper Reference Bound'] = 97.5

					refDict['Lower Reference Value'] = to_numeric(reference.attrib['vmin'])
					refDict['Upper Reference Value'] = to_numeric(reference.attrib['vmax'])

			item = {'Feature Name': name,
					'comment': comment,
					'type': qtype,
					'value': element.attrib['value'],
					'lod': to_numeric(element.attrib['lod']),
					'loq': to_numeric(element.attrib['loq']),
					'Unit': element.attrib['unit']}
			item = {**item, **refDict}
			quantList.append(item)

	return (sampleName, processingDate, quantList)


def to_numeric(string):
	try:
		return int(string)
	except ValueError:
		try:
			return float(string)
		except ValueError:
			return string
