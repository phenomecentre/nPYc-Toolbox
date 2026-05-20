import numpy as np
import pandas
import copy
import os
from IPython.display import display

from ..enumerations import SampleType, AssayRole
from .._toolboxPath import toolboxPath
from ..objects import Dataset
from ..utilities._internal import _copyBackingFiles as copyBackingFiles
from ..utilities.generic import sampleClassMasks, inferSampleClass
from ..__init__ import __version__ as version
#from ..utilities._errorHandling import npycToolboxError

def _generateSampleReport(dataTrue, withExclusions=False, destinationPath=None, returnOutput=False):
	"""
	Summarise samples in the dataset.

	Generate sample summary report, lists samples acquired, plus if possible, those missing as based on the expected sample manifest.

	:param Dataset dataTrue: Dataset to report on
	:param bool withExclusions: If ``True``, only report on features and samples not masked by the sample and feature masks
	:param destinationPath: If ``None``, run interactively, else a str specifying the directory to save report into
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

	# Apply sample/feature masks if exclusions to be applied
	data = copy.deepcopy(dataTrue)
	if withExclusions:
		data.applyMasks()

	sampleSummary = dict()
	sampleSummary['Name'] = data.name

	# Sample type masks
	sampleMasks = sampleClassMasks(data.sampleMetadata)

	# Masks for any study samples with missing metadata (this can be from basic CSV, LIMS, or Sample Manifest)
	NoMetadata = data.sampleMetadata['Metadata Available'] == False
	if 'LIMS Present' in data.sampleMetadata.columns:
		NoMetadata = (NoMetadata == True) | (data.sampleMetadata['LIMS Present'] == False)
	if 'SubjectInfoData' in data.sampleMetadata.columns:
		NoMetadata = (NoMetadata == True) | (data.sampleMetadata['SubjectInfoData'] == False)
	NoMetadata[data.sampleMetadata['SampleClass'] != 'Study Sample'] = False

	# Samples marked for exclusion (either as marked as skipped or as False in sampleMask)
	try:
		markedToExclude = (data.sampleMetadata['Skipped'].values == True) | (data.sampleMask == False)
	except:
		markedToExclude = data.sampleMask == False

	# Define columns for reporting missing/excuded/etc sample details
	cols = ['Sample File Name', 'Sample ID', 'SampleType', 'AssayRole', 'SampleClass', 'Exclusion Details']

	# Summary table for samples present in dataset
	sampleSummary['Dataset'] = pandas.DataFrame(np.zeros((len(sampleMasks), 3)).astype(int),
												 index=sampleMasks.keys(),
	                                             columns=['Present', 'Marked for Exclusion', 'Missing/Excluded'])

	for key in sampleMasks:
		# Total numbers present in dataset
		sampleSummary['Dataset'].loc[key, 'Present'] = sum(sampleMasks[key])

		# Numbers marked for exclusion (either skipped or in sampleMask)
		sampleSummary['Dataset'].loc[key, 'Marked for Exclusion'] = sum(markedToExclude & sampleMasks[key])

	# Marked for exclusion - details
	if sum(markedToExclude) != 0:
		sampleSummary['MarkedToExclude Details'] = data.sampleMetadata[cols][markedToExclude]
		sampleSummary['MarkedToExclude Details'].reset_index(drop=True, inplace=True)

	# Save details of samples with no associated metadata
	if sum(NoMetadata) != 0:
		sampleSummary['NoMetadata Details'] = data.sampleMetadata[cols][NoMetadata]
		sampleSummary['NoMetadata Details'].reset_index(drop=True, inplace=True)

	# Save details of samples of unknown type
	if hasattr(sampleMasks, 'Unknown') and (sum(sampleMasks['Unknown']) != 0):
		sampleSummary['UnknownType Details'] = data.sampleMetadata[cols][sampleMasks['Unknown']]
		sampleSummary['UnknownType Details'].reset_index(drop=True, inplace=True)

	# Save details of any samples already excluded or missing
	sampleSummary['Excluded Details'] = pandas.DataFrame(columns=cols)
	sampleSummary['Missing Details'] = pandas.DataFrame(columns=cols)

	if hasattr(data, 'excludedFlag') and ('Samples' in data.excludedFlag):

		# Create dataframe with columns required
		sampleMetadataExcluded = pandas.DataFrame(columns=cols)

		# Add info of all previously excluded samples (keep only columns required)
		excludedIX = [i for i, x in enumerate(data.excludedFlag) if x == 'Samples']
		for i in excludedIX:
			sampleMetadataExcluded = pandas.concat([sampleMetadataExcluded, data.sampleMetadataExcluded[i]],
			                                       ignore_index=True)

		# Generate sampleClass masks
		excludedMasks = sampleClassMasks(sampleMetadataExcluded)

		for key in sampleMasks:

			if (key in excludedMasks) and (sum(excludedMasks[key]) != 0):

				# Add numbers missing/excluded to sampleSummary['Dataset']
				sampleSummary['Dataset'].loc[key, 'Missing/Excluded'] = sum(excludedMasks[key])

		# Save details of all excluded samples
		sampleSummary['Excluded Details'] = pandas.concat([sampleSummary['Excluded Details'], sampleMetadataExcluded[sampleMetadataExcluded.columns.intersection(cols)]],
														  axis=0,
														  ignore_index=True)

	# Save details of any samples present in data locations but missing from data (and not already excluded)
	if hasattr(data, 'sampleAbsentMetadata'):

		# Standardise to 'Sample File Name' - when missing samples derived from data locations file we have 'Assay data name'
		if (hasattr(data.sampleAbsentMetadata, 'Assay data name')) and (not hasattr(data.sampleAbsentMetadata, 'Sample File Name')):
			data.sampleAbsentMetadata.rename(columns={"Assay data name": "Sample File Name"}, inplace=True)

		data.sampleAbsentMetadata['Exclusion Details'] = 'Missing/low volume'

		# Remove rows for any samples already acquired but already excluded (i.e., already in sampleSummary['Excluded Details'])
		data.sampleAbsentMetadata = data.sampleAbsentMetadata[~data.sampleAbsentMetadata['Sample File Name'].isin(sampleSummary['Excluded Details']['Sample File Name'].values)]
		data.sampleAbsentMetadata = data.sampleAbsentMetadata[~data.sampleAbsentMetadata['Sample ID'].isin(sampleSummary['Excluded Details']['Sample ID'].values)]

		# Infer SampleClass
		data.sampleAbsentMetadata = inferSampleClass(data.sampleAbsentMetadata)

		# Determine sample types of missing samples based on SampleClass
		missingMasks = sampleClassMasks(data.sampleAbsentMetadata, on='SampleClass')

		# Add numbers missing/excluded to sampleSummary['Dataset']
		for key in sampleMasks:
			if (key in missingMasks) and (sum(missingMasks[key]) != 0):
				sampleSummary['Dataset'].loc[key, 'Missing/Excluded'] = sampleSummary['Dataset'].loc[key, 'Missing/Excluded'] + sum(missingMasks[key])

		# Save sample details
		sampleSummary['Missing Details'] = pandas.concat([sampleSummary['Missing Details'], data.sampleAbsentMetadata],
		                                             axis=0,
		                                             ignore_index=True)


	# Save details of any samples present in sample manifest but missing from data locations
	if hasattr(data, 'subjectAbsentMetadata'):

		# Remove samples which are not the same biofluid as in the dataset
		unique_biofluid = pandas.unique(data.sampleMetadata['Biofluid'].values)
		biofluid_present = [True if str(x) in str(unique_biofluid) else False for x in
		                    data.subjectAbsentMetadata['Biofluid'].values]
		data.subjectAbsentMetadata = data.subjectAbsentMetadata[biofluid_present]

		# Standardise to 'Sample ID' - when missing samples derived from sample manifest we have 'Sampling ID'
		if (hasattr(data.subjectAbsentMetadata, 'Sampling ID')) and (not hasattr(data.subjectAbsentMetadata, 'Sample ID')):
			data.subjectAbsentMetadata.rename(columns={"Sampling ID": "Sample ID"}, inplace=True)

		data.subjectAbsentMetadata['Exclusion Details'] = 'Missing/low volume'

		# Remove rows for any samples already acquired but already excluded (i.e., already in sampleSummary['Excluded Details'])
		data.subjectAbsentMetadata = data.subjectAbsentMetadata[~data.subjectAbsentMetadata['Sample ID'].isin(sampleSummary['Excluded Details']['Sample ID'].values)]

		# Infer SampleClass (assume all samples in sample manifest are Study Samples)
		data.subjectAbsentMetadata['SampleType'] = SampleType.StudySample
		data.subjectAbsentMetadata['AssayRole'] = AssayRole.Assay
		data.subjectAbsentMetadata = inferSampleClass(data.subjectAbsentMetadata)

		# Determine sample types of missing samples
		missingMasks = sampleClassMasks(data.subjectAbsentMetadata)

		# Add numbers missing/excluded to sampleSummary['Dataset']
		for key in sampleMasks:
			if (key in missingMasks) and (sum(missingMasks[key]) != 0):
				sampleSummary['Dataset'].loc[key, 'Missing/Excluded'] = sampleSummary['Dataset'].loc[key, 'Missing/Excluded'] + sum(missingMasks[key])


		# Save sample details
		sampleSummary['Missing Details'] = pandas.concat([sampleSummary['Missing Details'], data.subjectAbsentMetadata],
													 axis=0,
													 ignore_index=True)

	# Save details for missing/excluded study samples only (i.e., only keep details of samples with Sample ID present)
	sampleSummary['Missing/excluded SS Details'] = pandas.concat([sampleSummary['Missing Details'], sampleSummary['Excluded Details']],
																 axis=0,
																 ignore_index=True)
	sampleSummary['Missing/excluded SS Details'] = sampleSummary['Missing/excluded SS Details'][sampleSummary['Missing/excluded SS Details']['Sample ID'].notna()]

	# Final formatting (including removal of dataframes if no corresponding samples)
	if len(sampleSummary['Excluded Details']) == 0:
		del sampleSummary['Excluded Details']
	else:
		sampleSummary['Excluded Details'].reset_index(inplace=True, drop=True)
		sampleSummary['Excluded Details'] = sampleSummary['Excluded Details'][
			sampleSummary['Excluded Details'].columns.intersection(cols)]

	if len(sampleSummary['Missing Details']) == 0:
		del sampleSummary['Missing Details']
	else:
		sampleSummary['Missing Details'].reset_index(inplace=True, drop=True)
		sampleSummary['Missing Details'] = sampleSummary['Missing Details'][
			sampleSummary['Missing Details'].columns.intersection(cols)]

	if len(sampleSummary['Missing/excluded SS Details']) == 0:
		del sampleSummary['Missing/excluded SS Details']

	# Add 'All Samples' sum to top row of sampleSummary['Dataset']
	sampleSummary['Dataset'] = pandas.concat([pandas.DataFrame(sampleSummary['Dataset'].sum(axis=0), columns=['All Samples']).transpose(), sampleSummary['Dataset']])

	# Remove 'Marked for Exclusion' column if no entries
	if sampleSummary['Dataset'].loc['All Samples', 'Marked for Exclusion'] == 0:
		sampleSummary['Dataset'].drop('Marked for Exclusion', axis=1, inplace=True)

	# Generate html report
	if destinationPath:

		# Set up template item and save required info
		from jinja2 import Environment, FileSystemLoader

		env = Environment(loader=FileSystemLoader(os.path.join(toolboxPath(), 'Templates')))
		template = env.get_template('generateSampleReport.html')
		filename = os.path.join(destinationPath, data.name + '_report_sampleSummary.html')
		# the jinja template expects item with sample summary inside so just create a field with everything inside
		sampleSummary['sampleSummary'] = sampleSummary
		f = open(filename, 'w')
		f.write(template.render(item=sampleSummary, version=version,
		                        graphicsPath=os.path.join(destinationPath, 'graphics')))
		f.close()

		copyBackingFiles(toolboxPath(), os.path.join(destinationPath, 'graphics'))

		data.sampleSummary = sampleSummary

	# Return sampleSummary
	elif returnOutput:
		return sampleSummary

	# Output tables to command line
	else:

		print('Sample Summary')
		display(sampleSummary['Dataset'])
		print('\n')

		if 'Missing Details' in sampleSummary:
			print('Samples Missing from Acquisition/Import (i.e., present in metadata file but not acquired/imported)')
			display(sampleSummary['Missing Details'])
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