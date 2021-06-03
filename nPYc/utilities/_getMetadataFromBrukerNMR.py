import os
import re
import codecs
import logging
from ._conditionalJoin import conditionalJoin
import warnings


def extractBrukerparams(path, queryItems, acqTimeRE):
	"""
	Read parameters defined in *queryItems* for Bruker data.

	:param filePath: Path to raw data folder
	:type filePath: str
	:param dict queryItems: names of parameters to extract values for
	:param str acqTimeRE: regular expression used to extract acquisition time
	:returns: Dictionary of extracted parameters
	:rtype: dict
	"""

	pathComponents = []
	path2 = path
	for i in range(5):
		path2, name = os.path.split(path2)
		pathComponents.append(name)

	results = dict()
	results['Warnings'] = ''

	results['File Path'] = path
	results['Sample File Name'] = pathComponents[4] + '/' + pathComponents[3]

	path = os.path.dirname(path)

	for inputFile in queryItems.keys():
		localPath = os.path.normpath(os.path.join(os.path.relpath(path), inputFile))
		try:
			f = codecs.open(localPath, 'r', encoding='latin-1')
			contents = f.read()

			logging.debug('Searching file: ' + os.path.join(localPath))

			# Loop over the search terms
			for findthis in queryItems[inputFile]:
				logging.debug('Looking for: ' + findthis)

				query = r'^' + re.escape(findthis) + '\W(.+?)\r?\n?^#'

				m = re.search(query, contents, re.MULTILINE|re.DOTALL)

				if m:
					foundLine = m.groups(0)[0]
					logging.debug('Found: ' + foundLine)
					results[findthis] = foundLine
				else:
					results[findthis] = ''
					results['Warnings'] = conditionalJoin(results['Warnings'], 'Parameter ' + findthis.strip() + ' not found.')
					warnings.warn('Parameter ' + findthis + ' not found in file: ' + os.path.join(localPath))

			f.close()
		except IOError:
			results['Warnings'] = conditionalJoin(results['Warnings'], 'Unable to open ' + localPath + ' for reading.')
			warnings.warn('Unable to open ' + localPath + ' for reading.')

	# Process parameters
	# Remove angle brackets
	trimAngleBrackets = ['##$PULPROG=', '##$AUNM=', '##$PROBHD=']
	for item in trimAngleBrackets:
		if item in results.keys():
			results[item] = results[item][1:-1]

	# Separate P
	if '##$P=' in results.keys():
		results['##$P='] = str.split(str.splitlines(results['##$P='])[1])[0]

	if '##OWNER=' in results.keys():
		lines = str.splitlines(results['##OWNER='])
		results['##OWNER='] = lines[0]
		match = acqTimeRE.match(lines[1])
		if match:
			results['Acquired Time'] = match.groups()[0]
			results['Acquired Time'] = results['Acquired Time'].split('(')[0].strip() # remove (UT+) info from XWIN
			results['Computer'] = match.groups()[1]

	cleanedresults = dict()
	for key in results.keys():
		if not key in ['##OWNER=', 'Sample File Name', 'File Path', 'Warnings', 'Acquired Time', 'Computer']:
			normalisedKey = key[3:-1]
			cleanedresults[normalisedKey] = results[key]
		else:
			if key == '##OWNER=':
				cleanedresults['OWNER'] = results['##OWNER=']
			else:
				cleanedresults[key] = results[key]

	return cleanedresults

