import warnings
import numpy
from ..utilities.extractParams import *
from datetime import datetime

def getSampleMetadataFromWatersRawFiles(rawDataPath):
	"""
	Get acquisition metadata from Waters RAW files and returns them as a dataframe

	:param str rawDataPath: Path to folder of raw data
	"""

	# Get the paramters as a table
	instrumentParams = extractParams(rawDataPath, 'Waters .raw')

	# Strip any whitespace from 'Sample File Name'
	instrumentParams['Sample File Name'] = instrumentParams['Sample File Name'].str.strip()

	# Parse acqustion times
	instrumentParams['Acquired Time'] = numpy.nan
	for i in range(instrumentParams.shape[0]):
		try:
			instrumentParams.loc[i, 'Acquired Time'] = datetime.strptime(str(instrumentParams.loc[i, '$$ Acquired Date:']) + " " + str(instrumentParams.loc[i,'$$ Acquired Time:']), '%d-%b-%Y %H:%M:%S')
		except ValueError:
			pass
		
	# Rename '$$ Acquired Time' and '$$ Acquired Date to avoid confusion
	instrumentParams.rename(columns={'$$ Acquired Time:': 'Measurement Time'}, inplace=True)
	instrumentParams.rename(columns={'$$ Acquired Date:': 'Measurement Date'}, inplace=True)
	
	
	##
	# Detect duplicate experiment filenames
	##
	duplicateSamples = instrumentParams.loc[instrumentParams['Sample File Name'].duplicated(keep=False)]
	if duplicateSamples.size > 0:
		warnings.warn('Duplicate raw data loaded, discarding duplicates.', UserWarning)
		# Drop duplicate files
		instrumentParams = instrumentParams.loc[instrumentParams['Sample File Name'].duplicated(keep='first')==False]

	return instrumentParams
