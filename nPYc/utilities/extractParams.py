"""
:py:mod:`~nPYc.utilities.extractParams` contains several utility functions to read analytical parameters from raw data files.
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re
import logging
import os
import codecs
import pandas
import warnings
from ._conditionalJoin import *

def extractParams(filepath, filetype, pdata=1):
	"""
	Extract analytical parameters from raw data files

	:param filepath: Look for data in all the directories under this location.
	:type searchDirectory: string
	:param filetype: Search for this type of data
	:type filetype: string
	:return: Analytical parameters, indexed by file name.
	:rtype: pandas.Dataframe
	"""

	queryItems = dict()
	# Build our ID cirteria
	if filetype == 'Bruker':
		pattern = r'^1r$'
		pattern = re.compile(pattern)
		queryItems[os.path.join('..', '..', 'acqus')] = ['##OWNER=', '##$PULPROG=','##$RG=', '##$SW=','##$SFO1=', '##$TD=', '##$PROBHD=',
														 '##$BF1=', '##$O1=', '##$P=', '##$AUNM=', '##$NS=']
		queryItems['procs'] = ['##$OFFSET=', '##$SW_p=', '##$NC_proc=', '##$SF=', '##$SI=', '##$BYTORDP=', '##$XDIM=']

		# Assemble a list of files
		fileList = buildFileList(filepath, pattern)
		pdataPattern = re.compile(r'.+[/\\]\d+?[/\\]pdata[/\\]' + str(pdata) + r'[/\\]1r$')
		fileList = [x for x in fileList if pdataPattern.match(x)]

		query = r'^\$\$\W(.+?)\W+([\w-]+@[\w-]+)$'
		acqTimeRE = re.compile(query)

	elif filetype == 'Waters .raw':
		pattern = '.+?\.raw$'
		queryItems['_extern.inf'] = ['Resolution', 'Capillary (kV)','Sampling Cone', u'Source Temperature (°C)',
				   'Source Offset',u'Desolvation Temperature (°C)','Cone Gas Flow (L/Hr)','Desolvation Gas Flow (L/Hr)',
				   'LM Resolution','HM Resolution','Collision Energy', 'Polarity', 'Detector\t','Scan Time (sec)',
				   'Interscan Time (sec)','Start Mass','End Mass','Backing','Collision\t','TOF\t']
		queryItems['_HEADER.TXT'] = ['$$ Acquired Date:','$$ Acquired Time:', '$$ Instrument:']
		queryItems['_INLET.INF'] = ['ColumnType:', 'Column Serial Number:']

		# Assemble a list of files
		pattern = re.compile(pattern)
		fileList = buildFileList(filepath, pattern)

	# iterate over the list
	results = list()
	for filename in fileList:
		if filetype == 'Bruker':
			results.append(extractBrukerparams(filename, queryItems, acqTimeRE))
		elif filetype == 'Waters .raw':
			results.append(extractWatersRAWParams(filename, queryItems))

	resultsDF = pandas.DataFrame(results)
	resultsDF = resultsDF.apply(lambda x: pandas.to_numeric(x, errors='ignore'))

	return resultsDF


def buildFileList(filepath, pattern):
	"""
	Search for data files, by attempting to match to the file path regex *pattern*.

	:param filepath: Look for data in all the directories under this location
	:type searchDirectory: str
	:param pattern: Recognise experimental data by matching path to this compiled regex
	:type pattern: re.SRE_Pattern
	:return: A list of all paths below *searchDirectory* that matched *pattern*
	:rtype: list[str,]
	"""

	logging.debug('Searching in: ' + filepath)
	
	# Read in all folder names
	child_items = os.listdir(filepath)
	
	fileList = list()

	# Match against pattern
	for childItem in child_items:
		if pattern.match(childItem):
			logging.debug('Matched: ' + childItem)
			fileList.append(os.path.join(filepath, childItem))
		elif os.path.isdir(os.path.join(filepath, childItem)):
			# search again in this folder.
			logging.debug('Descending into: ' + childItem)
			fileList.extend(buildFileList(os.path.join(filepath, childItem), pattern))
		else:
			logging.debug('Discarding: ' + childItem)

	return fileList


def extractWatersRAWParams(filePath, queryItems):
	"""
	Read parameters defined in *queryItems* for Waters .RAW data.

	:param filePath: Path to .RAW folder
	:type filePath: str
	:returns: Dictionary of extracted parameters
	:rtype: dict
	"""

	# Get filename
	filename = os.path.basename(filePath)
	results = dict()
	results['Warnings'] = ''

	results['File Path'] = filePath
	results['Sample File Name'] = filename[:-4]

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
					results['Warnings'] = conditionalJoin(results['Warnings'] , 'Parameter ' + findthis.strip() + ' not found.')
					warnings.warn('Parameter ' + findthis + ' not found in file: ' + os.path.join(localPath))
			
			f.close()
		except IOError:
			for findthis in queryItems[inputFile]:
				results['Warnings'] = conditionalJoin(results['Warnings'], 'Unable to open ' + localPath + ' for reading.')
				warnings.warn('Unable to open ' + localPath + ' for reading.')

	return results



def extractBrukerparams(path, queryItems, acqTimeRE):
	"""

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
		localPath = os.path.join(path, inputFile)
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

	##
	# Process parameters
	##
	# Remove angle brackets
	trimAngleBrackets = ['##$PULPROG=', '##$AUNM=', '##$PROBHD=']
	for item in trimAngleBrackets:
		if item in results.keys():
			results[item] = results[item][1:-1]

	# Seperate P
	if '##$P=' in results.keys():
		results['##$P='] = str.split(str.splitlines(results['##$P='])[1])[0]

	if '##OWNER=' in results.keys():
		lines = str.splitlines(results['##OWNER='])
		results['##OWNER='] = lines[0]
		match = acqTimeRE.match(lines[1])
		if match:
			results['Acquired Time'] = match.groups()[0]
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


def main():
	import sys
	import argparse
	import logging
	
	parser = argparse.ArgumentParser(description='Parse experiment files for information.')
	parser.add_argument("-v", "--verbose", help="increase output verbosity.", action="store_true")
	parser.add_argument("-t", "--type", help="Filetype of data to parse.")
	parser.add_argument('source', type=str, help='Path containing.raw data to parse.')
	parser.add_argument('output', type=str, help='Destination for .csv output.')
	

	args = parser.parse_args()
	
	if args.verbose:
		logging.basicConfig(level=logging.DEBUG)
	
	"""Parse input file into output"""
	print('Working...')
	table = extractParams(args.source, 'Waters .raw')
	
	# Write output
	table.to_csv(args.output, encoding='utf-8', index=False)


if __name__=='__main__':
	main()

