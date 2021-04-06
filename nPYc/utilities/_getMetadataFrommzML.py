import re
import logging
import os
import warnings
from xml.etree import ElementTree as ET
from nPYc.utilities._conditionalJoin import *


def extractmzMLParams(filePath, queryItems):
    """
    Read parameters defined in *queryItems* from an mzML file. This implementation uses the ElementTree
    .xml parser
    :param filePath: Path to mzML file
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

    try:
        xml_file = ET.parse(filePath)

        root_node = xml_file.getroot()

        logging.debug('Searching file: ' + filePath)
        # Loop over the search terms
        for currentTag in queryItems:

            tagValue = None
            for child in root_node.iter():
                if child.get(currentTag) is None:
                    pass
                else:
                    tagValue = child.get(currentTag)
                    results[currentTag] = tagValue

            logging.debug('Looking for: ' + currentTag)

            if tagValue is not None:
                logging.debug('Found Tag: ' + currentTag + ' with value: ' + tagValue)
            else:
                results['Warnings'] = conditionalJoin(results['Warnings'],
                                                       'Parameter ' + currentTag + ' param not found.')
                warnings.warn('Parameter ' + currentTag + ' param not found in file: ' + os.path.join(currentTag))

    except IOError:
        results['Warnings'] = conditionalJoin(results['Warnings'],
                                                'Unable to open ' + filePath + ' for reading.')
        warnings.warn('Unable to open ' + filePath + ' for reading.')
    except ET.ParseError:
        results['Warnings'] = conditionalJoin(results['Warnings'],
                                              'Error parsing the .XML structure in: ' + filePath)
        warnings.warn('Error parsing the .XML structure in: ' + filePath)

    return results


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