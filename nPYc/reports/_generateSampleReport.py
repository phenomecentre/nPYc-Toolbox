import numpy
import pandas
import copy
import os
import shutil
import sys
import inspect
from IPython.display import display
from .._toolboxPath import toolboxPath
from ..objects import Dataset
from ..utilities._internal import _copyBackingFiles as copyBackingFiles
from ..utilities.ms import generateTypeRoleMasks
from ..enumerations import AssayRole, SampleType
from ..__init__ import __version__ as version


def _generateSampleReport(dataTrue, withExclusions=False, destinationPath=None, returnOutput=False):
	"""
	Summarise samples in the dataset.

	Generate sample summary report, lists samples acquired, plus if possible, those missing as based on the expected sample manifest.

	:param Dataset dataTrue: Dataset to report on
	:param bool withExclusions: If ``True``, only report on features and samples not masked by the sample and feature masks
	:param destinationPath: If ``None``, run interactivly, else a str specifying the directory to save report into
	:type destinationPath: None or str
	:param bool returnOutput: If ``True``, returns a dictionary of all tables generated during run
	:return: Optional, dictionary of all tables generated during run
	"""

	# Check inputs
	if not isinstance(dataTrue, Dataset):
		raise TypeError('dataTrue must be an instance of nPYc.Dataset')
	if not isinstance(withExclusions, bool):
		raise TypeError('withExclusions must be a bool')
	if destinationPath is not None:
		if not isinstance(destinationPath, str):
			raise TypeError('destinationPath must be a string')
	if not isinstance(returnOutput, bool):
		raise TypeError('returnItem must be a bool')
	if 'Sample ID' not in dataTrue.sampleMetadata:
		raise ValueError('sampleMetadata must contain "Sample ID" column')

	# Create directory to save destinationPath	 # for now do nothing as sampleReport requires no files

	# Apply sample/feature masks if exclusions to be applied
	data = copy.deepcopy(dataTrue)
	if withExclusions:
		data.applyMasks()

	sampleSummary = dict()
	sampleSummary['Name'] = data.name

	# Sample type masks
	acquiredMasks = generateTypeRoleMasks(data.sampleMetadata)

	NotInCSVmask = data.sampleMetadata['Metadata Available'] == False

	# Samples marked for exclusion (either as marked as skipped or as False in sampleMask)

	try:
		markedToExclude = (data.sampleMetadata['Skipped'].values == True) | (data.sampleMask == False)
	except:
		markedToExclude = data.sampleMask == False

	# Summary table for samples acquired
	temp = numpy.zeros([7, 3], dtype=int)

	# Total numbers
	temp[:, 0] = [sum(acquiredMasks['ALLmask']),
					sum(acquiredMasks['SSmask']),
					sum(acquiredMasks['SPmask']),
					sum(acquiredMasks['ERmask']),
					sum(acquiredMasks['SRDmask']),
					sum(acquiredMasks['Blankmask']),
					sum(acquiredMasks['Unknownmask'])]

	# Numbers marked for exclusion (either skipped or in sampleMask)
	temp[:, 1] = [sum(markedToExclude & acquiredMasks['ALLmask']),
					sum(markedToExclude & acquiredMasks['SSmask']),
					sum(markedToExclude & acquiredMasks['SPmask']),
					sum(markedToExclude & acquiredMasks['ERmask']),
					sum(markedToExclude & acquiredMasks['SRDmask']),
					sum(markedToExclude & acquiredMasks['Blankmask']),
					sum(markedToExclude & acquiredMasks['Unknownmask'])]

	# Convert to dataframe
	sampleSummary['Acquired'] = pandas.DataFrame(data=temp,
		index=['All', 'Study Sample', 'Study Reference', 'Long-Term Reference', 'Serial Dilution', 'Blank', 'Unknown'],
		columns=['Total', 'Marked for Exclusion', 'Missing/Excluded'])

	# Marked for exclusion - details
	if (sum(markedToExclude) != 0):
		sampleSummary['MarkedToExclude Details'] = data.sampleMetadata[['Sample File Name', 'Exclusion Details']][markedToExclude]

	# Save details of samples of unknown type
	if (sum(NotInCSVmask) != 0):
		sampleSummary['NoMetadata Details'] = data.sampleMetadata[['Sample File Name']][NotInCSVmask]

	# Save details of samples of unknown type
	if (sum(acquiredMasks['Unknownmask']) != 0):
		sampleSummary['UnknownType Details'] = data.sampleMetadata[['Sample File Name']][acquiredMasks['Unknownmask']]

	# Finally - add column of samples already excluded or missing to sampleSummary
	ALL_exclusions = pandas.DataFrame(columns=['Sample File Name', 'Sample ID', 'Exclusion Details'])
	SS_exclusions = pandas.DataFrame(columns=['Sample File Name', 'Sample ID', 'Exclusion Details'])

	# Determine if samples have been excluded
	if hasattr(data, 'excludedFlag') and (len(data.excludedFlag) > 0):
		excludedIX = [i for i, x in enumerate(data.excludedFlag) if x == 'Samples']
		sampleMetadataExcluded = pandas.DataFrame(
			columns=['Sample File Name', 'Sample ID', 'SampleType', 'AssayRole', 'Exclusion Details'])

		# Stick info of all previously excluded samples together
		for i in excludedIX:
			temp = copy.deepcopy(data.sampleMetadataExcluded[i])

			sampleMetadataExcluded = sampleMetadataExcluded.append(temp[['Sample File Name', 'Sample ID',
																		 'SampleType', 'AssayRole',
																		 'Exclusion Details']], ignore_index=True)

		# Sample type masks, and only those marked as 'sample' or 'unknown' flagged
		excludedMasks = generateTypeRoleMasks(sampleMetadataExcluded)

		sampleSummary['Acquired'].loc['All', 'Missing/Excluded'] = sampleSummary['Acquired'].loc['All', 'Missing/Excluded'] + sum(excludedMasks['ALLmask'])
		sampleSummary['Acquired'].loc['Study Sample', 'Missing/Excluded'] = sampleSummary['Acquired'].loc['Study Sample', 'Missing/Excluded'] + sum(excludedMasks['SSmask'])
		sampleSummary['Acquired'].loc['Study Reference', 'Missing/Excluded'] = sampleSummary['Acquired'].loc['Study Reference', 'Missing/Excluded'] + sum(excludedMasks['SPmask'])
		sampleSummary['Acquired'].loc['Long-Term Reference', 'Missing/Excluded'] = sampleSummary['Acquired'].loc['Long-Term Reference', 'Missing/Excluded'] + sum(excludedMasks['ERmask'])
		sampleSummary['Acquired'].loc['Serial Dilution', 'Missing/Excluded'] = sampleSummary['Acquired'].loc['Serial Dilution', 'Missing/Excluded'] + sum(excludedMasks['SRDmask'])
		sampleSummary['Acquired'].loc['Blank', 'Missing/Excluded'] = sampleSummary['Acquired'].loc['Blank', 'Missing/Excluded'] + sum(excludedMasks['Blankmask'])
		sampleSummary['Acquired'].loc['Unknown', 'Missing/Excluded'] = sampleSummary['Acquired'].loc['Unknown', 'Missing/Excluded'] + sum(excludedMasks['Unknownmask'])

		ALL_exclusions = pandas.concat([ALL_exclusions,
			sampleMetadataExcluded[['Sample File Name', 'Sample ID', 'Exclusion Details']][excludedMasks['ALLmask']]],
			ignore_index=True)

		SS_exclusions = pandas.concat([SS_exclusions,
			sampleMetadataExcluded[['Sample File Name', 'Sample ID', 'Exclusion Details']][excludedMasks['SSmask']]],
			ignore_index=True)

	# Determine if any SS missing
	if hasattr(data, 'sampleAbsentMetadata'):

		# Standardise to 'Sample File Name' - when missing samples derived from data locations file
		if ('Assay data name' in data.sampleAbsentMetadata.columns) and ('Sample File Name' not in data.sampleAbsentMetadata.columns):
			data.sampleAbsentMetadata.rename(columns={"Assay data name": "Sample File Name"}, inplace=True)

		data.sampleAbsentMetadata['Exclusion Details'] = 'Missing/low volume'

		# Save sample details
		if 'Sample File Name' in data.sampleAbsentMetadata.columns:
			sampleSummary['NotAcquired'] = data.sampleAbsentMetadata[['Sample File Name', 'Sample ID']]
		else:
			sampleSummary['NotAcquired'] = data.sampleAbsentMetadata[
				['Sample ID', 'Assay data name', 'LIMS Marked Missing']]

		# Save SS details
		missingMasks = generateTypeRoleMasks(data.sampleAbsentMetadata)

		sampleSummary['Acquired'].loc['All', 'Missing/Excluded'] = sampleSummary['Acquired'].loc['All', 'Missing/Excluded'] + sum(missingMasks['ALLmask'])
		sampleSummary['Acquired'].loc['Study Sample', 'Missing/Excluded'] = sampleSummary['Acquired'].loc['Study Sample', 'Missing/Excluded'] + sum(missingMasks['SSmask'])
		sampleSummary['Acquired'].loc['Study Reference', 'Missing/Excluded'] = sampleSummary['Acquired'].loc['Study Reference', 'Missing/Excluded'] + sum(missingMasks['SPmask'])
		sampleSummary['Acquired'].loc['Long-Term Reference', 'Missing/Excluded'] = sampleSummary['Acquired'].loc['Long-Term Reference', 'Missing/Excluded'] + sum(missingMasks['ERmask'])
		sampleSummary['Acquired'].loc['Serial Dilution', 'Missing/Excluded'] = sampleSummary['Acquired'].loc['Serial Dilution', 'Missing/Excluded'] + sum(missingMasks['SRDmask'])
		sampleSummary['Acquired'].loc['Blank', 'Missing/Excluded'] = sampleSummary['Acquired'].loc['Blank', 'Missing/Excluded'] + sum(missingMasks['Blankmask'])
		sampleSummary['Acquired'].loc['Unknown', 'Missing/Excluded'] = sampleSummary['Acquired'].loc['Unknown', 'Missing/Excluded'] + sum(missingMasks['Unknownmask'])

		ALL_exclusions = pandas.concat([ALL_exclusions, data.sampleAbsentMetadata[
			['Sample File Name', 'Sample ID', 'Exclusion Details']][missingMasks['ALLmask']]], ignore_index=True)

		SS_exclusions = pandas.concat([SS_exclusions, data.sampleAbsentMetadata[
			['Sample File Name', 'Sample ID', 'Exclusion Details']][missingMasks['SSmask']]], ignore_index=True)

	ALL_exclusions.reset_index(inplace=True, drop=True)
	SS_exclusions.reset_index(inplace=True, drop=True)

	# Save only if excluded samples present
	if ALL_exclusions.shape[0] != 0:
		sampleSummary['Excluded Details'] = ALL_exclusions

	if SS_exclusions.shape[0] != 0:
		sampleSummary['StudySamples Exclusion Details'] = SS_exclusions

	# Drop rows where no samples present for that datatype
	sampleSummary['Acquired'].drop(sampleSummary['Acquired'].index[sampleSummary['Acquired']['Total'].values == 0], axis=0, inplace=True)

	# Update 'All', 'Missing/Excluded' to only reflect sample types present in data
	sampleSummary['Acquired'].loc['All', 'Missing/Excluded'] = sum(sampleSummary['Acquired']['Missing/Excluded'][1:])


	# Generate html report
	if destinationPath:
		# Set up template item and save required info

		from jinja2 import Environment, FileSystemLoader

		env = Environment(loader=FileSystemLoader(os.path.join(toolboxPath(), 'Templates')))
		template = env.get_template('generateSampleReport.html')
		filename = os.path.join(destinationPath, data.name + '_report_sampleSummary.html')
		# the jinja template expects item with sample summary inside so just create a field with everything inside
		sampleSummary['sampleSummary'] = sampleSummary
		f = open(filename,'w')
		f.write(template.render(item=sampleSummary, version=version, graphicsPath=os.path.join(destinationPath, 'graphics')))
		f.close()

		copyBackingFiles(toolboxPath(), os.path.join(destinationPath, 'graphics'))

		data.sampleSummary = sampleSummary

	# Return sampleSummary
	elif returnOutput:
		return sampleSummary

	# Output tables to command line
	else:

		print('Summary of Samples Acquired')
		display(sampleSummary['Acquired'])
		print('\n')

		if 'NotAcquired' in sampleSummary:
			print('Samples Missing from Acquisition/Import (i.e., present in metadata file but not acquired/imported)')
			display(sampleSummary['NotAcquired'])
			print('\n')

		if 'MarkedToExclude Details' in sampleSummary:
			print('Samples Marked for Exclusion')
			display(sampleSummary['MarkedToExclude Details'])
			print('\n')

		if 'Excluded Details' in sampleSummary:
			print('Samples Missing/Excluded')
			display(sampleSummary['Excluded Details'])
			print('\n')

		if 'UnknownType Details' in sampleSummary:
			print('Samples of Unknown Type')
			display(sampleSummary['UnknownType Details'])
			print('\n')
            
		if 'NoMetadata Details' in sampleSummary:
			print('Samples for which no Metadata was provided')
			display(sampleSummary['NoMetadata Details'])
			print('\n')
