import re
import logging
import copy
import os
import warnings
from nPYc.utilities._conditionalJoin import *


def extractmzMLParamsRegex(mzMLPath, queryItems=['startTimeStamp']):
    """
    Read parameters defined in *queryItems* from an mzML file. This implementation ignores the .xml structure, and reads
    the file in chunks of lines which are then parsed directly with regular expressions.
    :param mzMLPath: Path to the mzML file to be parsed.
    :param queryItems: list of fields to extract. These will be compiled into regex which search for values surrounded by
    "queryItem="" and "\""
    :returns: Dictionary of extracted parameters
    :rtype: dict
    """
    # Get filename
    filename = os.path.basename(mzMLPath)
    queryItems = copy.deepcopy(queryItems)
    results = dict()
    results['Warnings'] = ''

    results['File Path'] = mzMLPath
    results['Sample File Name'] = os.path.splitext(filename)[0]
    preparedRegex = [re.compile('(?<=' + x + '=")(.*?)(?=")') for x in queryItems]
    preparedRegex_value = [re.compile('(?<=name=\"' + x + '\" value=\")(.*?)(?=")') for x in queryItems]
    try:
        with open(mzMLPath, 'r') as xml_file:
            for line in xml_file:
                for idx, currRegex in enumerate(preparedRegex):
                    if currRegex.search(line):
                        results[queryItems[idx]] = currRegex.search(line).group(0)
                        del preparedRegex[idx]
                        del preparedRegex_value[idx]
                        del queryItems[idx]
                    elif preparedRegex_value[idx].search(line):
                        results[queryItems[idx]] = preparedRegex_value[idx].search(line).group(0)
                        del preparedRegex[idx]
                        del preparedRegex_value[idx]
                        del queryItems[idx]
                if len(preparedRegex) == 0:
                    break
        for notFoundParam in queryItems:
            results['Warnings'] = conditionalJoin(results['Warnings'],
                            'Parameter ' + notFoundParam + ' param not found.')
            warnings.warn('Parameter ' + notFoundParam + ' param not found in file: ' + results['Sample File Name'])

    except IOError:
        results['Warnings'] = conditionalJoin(results['Warnings'],
                                              'Unable to open ' + mzMLPath + ' for reading.')
        warnings.warn('Unable to open ' + mzMLPath + ' for reading.')

    return results