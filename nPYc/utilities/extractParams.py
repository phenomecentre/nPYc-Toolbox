"""
:py:mod:`~nPYc.utilities.extractParams` contains several utility functions to read analytical parameters from raw data files.
"""

import re
import logging
import os
import pandas
import warnings
from datetime import datetime
import numpy
from ._getMetadataFrommzML import extractmzMLParamsRegex
from ._getMetadataFromWatersRaw import extractWatersRAWParams
from ._getMetadataFromBrukerNMR import extractBrukerparams


def extractParams(filepath, filetype, pdata=1, whichFiles=None):
    """
    Extract analytical parameters from raw data files for Bruker, Waters .RAW data and .mzML only.
    :param filepath: Look for data in all the directories under this location.
    :type searchDirectory: string
    :param filetype: Search for this type of data
    :type filetype: string
    :param int pdata: pdata folder for Bruker data
    :param list whichFiles: If a list of files is provided, only the files in it will be parsed
    :return: Analytical parameters, indexed by file name.
    :rtype: pandas.Dataframe
    """

    queryItems = dict()
    # Build our ID criteria
    if filetype == 'Bruker':
        pattern = r'^1r$'
        pattern = re.compile(pattern)
        queryItems[os.path.join('..', '..', 'acqus')] = ['##OWNER=', '##$PULPROG=', '##$RG=', '##$SW=', '##$SFO1=',
                                                            '##$TD=', '##$PROBHD=',
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
        queryItems['_extern.inf'] = ['Resolution', 'Capillary (kV)', 'Sampling Cone', u'Source Temperature (°C)',
                                     'Source Offset', u'Desolvation Temperature (°C)', 'Cone Gas Flow (L/Hr)',
                                     'Desolvation Gas Flow (L/Hr)',
                                     'LM Resolution', 'HM Resolution', 'Collision Energy', 'Polarity', 'Detector\t',
                                     'Scan Time (sec)',
                                     'Interscan Time (sec)', 'Start Mass', 'End Mass', 'Backing', 'Collision\t',
                                     'TOF\t']
        queryItems['_HEADER.TXT'] = ['$$ Acquired Date:', '$$ Acquired Time:', '$$ Instrument:']
        queryItems['_INLET.INF'] = ['ColumnType:', 'Column Serial Number:']

        # Assemble a list of files
        pattern = re.compile(pattern)
        fileList = buildFileList(filepath, pattern)

    elif filetype == '.mzML':
        pattern = '.+?\.mzML$'
        queryItems = ['startTimeStamp']
        pattern = re.compile(pattern)
        fileList = buildFileList(filepath, pattern)

    if whichFiles is not None:
        fileList = [x for x in fileList if x in whichFiles]

    # iterate over the list
    results = list()
    for filename in fileList:
        if filetype == 'Bruker':
            results.append(extractBrukerparams(filename, queryItems, acqTimeRE))
        elif filetype == 'Waters .raw':
            extractedWatersRaw = extractWatersRAWParams(filename, queryItems)
            try:
                extractedWatersRaw['Acquired Time'] = datetime.strptime(str(extractedWatersRaw['$$ Acquired Date:']) +
                " " + str(extractedWatersRaw['$$ Acquired Time:']), '%d-%b-%Y %H:%M:%S')
            except KeyError:
                extractedWatersRaw['Acquired Time'] = numpy.nan
            results.append(extractedWatersRaw)
        elif filetype == '.mzML':
            extractedmzML = extractmzMLParamsRegex(filename, queryItems)
            extractedmzML['Acquired Time'] = datetime.strptime(str(extractedmzML['startTimeStamp']), '%d-%b-%Y %H:%M:%S')
            results.append(extractmzMLParamsRegex(filename, queryItems))

    resultsDF = pandas.DataFrame(results)

    if resultsDF.shape[0] > 0:
        resultsDF['Sample File Name'] = resultsDF['Sample File Name'].str.strip()
        # Rename '$$ Acquired Time' and '$$ Acquired Date to avoid confusion
        resultsDF.rename(columns={'$$ Acquired Time:': 'Measurement Time'}, inplace=True)
        resultsDF.rename(columns={'$$ Acquired Date:': 'Measurement Date'}, inplace=True)

        duplicateSamples = resultsDF.loc[resultsDF['Sample File Name'].duplicated(keep=False)]
        if duplicateSamples.size > 0:
            warnings.warn('Duplicate raw data loaded, discarding duplicates.', UserWarning)
            # Drop duplicate files
            resultsDF = resultsDF.loc[resultsDF['Sample File Name'].duplicated(keep='first') is False]

        resultsDF = resultsDF.apply(lambda x: pandas.to_numeric(x, errors='ignore'))
        resultsDF['Acquired Time'] = pandas.to_datetime(resultsDF['Acquired Time'])
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


if __name__ == '__main__':
    main()

