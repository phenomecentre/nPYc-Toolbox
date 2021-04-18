import warnings
import numpy
import re
from datetime import datetime
import os
import logging
import codecs
from ._conditionalJoin import conditionalJoin


def extractWatersRAWParams(filePath, queryItems):
	"""
	Read parameters defined in *queryItems* for Waters .RAW data.

	:param filePath: Path to .RAW folder
	:type filePath: str
	:param dict queryItems: names of parameters to extract values for
	:returns: Dictionary of extracted parameters
	:rtype: dict
	"""

	# Get filename
	filename = os.path.basename(filePath)
	results = dict()
	results['Warnings'] = ''

	results['File Path'] = filePath
	results['Sample File Name'] = os.path.splitext(filename)[0]

	for inputFile in queryItems.keys():
		localPath = os.path.join(filePath, inputFile)
		try:
			f = codecs.open(localPath, 'r', encoding='latin-1')
			contents = f.readlines()

			logging.debug('Searching file: ' + localPath)

			# Loop over the search terms
			for findthis in queryItems[inputFile]:
				logging.debug('Looking for: ' + findthis)
				indices = [i for i, s in enumerate(contents) if findthis in s]
				if indices:
					logging.debug('Found on line: ' + str(indices[0]))
					foundLine = contents[indices[0]]
					logging.debug('Line reads: ' + foundLine.rstrip())
					query = '(' + re.escape(findthis) + ')\W+(.+)\r'
					logging.debug('Regex is: ' + query)
					m = re.search(query, foundLine)
					logging.debug('Found this: ' + m.group(1) + ' and: ' + m.group(2))

					results[findthis.strip()] = m.group(2).strip()
				else:
					results['Warnings'] = conditionalJoin(results['Warnings'],
														  'Parameter ' + findthis.strip() + ' not found.')
					warnings.warn('Parameter ' + findthis + ' not found in file: ' + os.path.join(localPath))

			f.close()
		except IOError:
			for findthis in queryItems[inputFile]:
				results['Warnings'] = conditionalJoin(results['Warnings'],
													  'Unable to open ' + localPath + ' for reading.')
				warnings.warn('Unable to open ' + localPath + ' for reading.')

	return results
