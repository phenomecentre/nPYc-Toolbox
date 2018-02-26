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
from ..enumerations import AssayRole, SampleType
from ..__init__ import __version__ as version


def _generateSampleReport(dataTrue, withExclusions=False, output=None, returnOutput=False):
	"""
	Summarise samples in the dataset.

	Generate sample summary report, lists samples acquired, plus if possible, those missing as based on the expected sample manifest.

	:param Dataset dataTrue: Dataset to report on
	:param bool withExclusions: If ``True``, only report on features and samples not masked by the sample and feature masks
	:param output: If ``None``, run interactivly, else a str specifying the directory to save report into
	:type output: None or str
	:param bool returnOutput: If ``True``, returns a dictionary of all tables generated during run
	:return: Optional, dictionary of all tables generated during run
	"""

	# Check inputs
	if not isinstance(dataTrue, Dataset):
		raise TypeError('dataTrue must be an instance of nPYc.Dataset')
	if not isinstance(withExclusions, bool):
		raise TypeError('withExclusions must be a bool')
	if output is not None:
		if not isinstance(output, str):
			raise TypeError('output must be a string')
	if not isinstance(returnOutput, bool):
		raise TypeError('returnItem must be a bool')

	# Create directory to save output	
	if output:
		
		# If directory exists delete directory and contents
		if os.path.exists(os.path.join(output, 'graphics', 'report_sampleSummary')):
			shutil.rmtree(os.path.join(output, 'graphics', 'report_sampleSummary'))
		
		# Create directory to save output
		os.makedirs(os.path.join(output, 'graphics', 'report_sampleSummary'))

	# Apply sample/feature masks if exclusions to be applied
	data = copy.deepcopy(dataTrue)
	if withExclusions:
		data.applyMasks()

	sampleSummary = dict()
	sampleSummary['Name'] = data.name

	# Sample type masks
	try:
		SSmask = (data.sampleMetadata['SampleType'] == SampleType.StudySample) & (data.sampleMetadata['AssayRole'] == AssayRole.Assay)
		SPmask = (data.sampleMetadata['SampleType'] == SampleType.StudyPool) & (data.sampleMetadata['AssayRole'] == AssayRole.PrecisionReference)
		ERmask = (data.sampleMetadata['SampleType'] == SampleType.ExternalReference) & (data.sampleMetadata['AssayRole'] == AssayRole.PrecisionReference)
	except:
		SSmask = numpy.zeros(len(data.sampleMask)).astype(bool)
		SPmask = numpy.zeros(len(data.sampleMask)).astype(bool)
		ERmask = numpy.zeros(len(data.sampleMask)).astype(bool)
	UNDEFmask = pandas.isnull(data.sampleMetadata['Sample Base Name'])
	OTHERmask = (SSmask==False) & (SPmask==False) & (ERmask==False) & (UNDEFmask==False)
	# Samples marked for exclusion (either as marked as skipped or as False in sampleMask)
	try:
		markedToExclude = (data.sampleMetadata['Skipped'].values==True) | (data.sampleMask==False)
	except:
		markedToExclude = data.sampleMask==False

	# Samples already excluded

	# Determine if samples have been excluded
	try:
		excludedIX = [i for i, x in enumerate(data.excludedFlag) if x == 'Samples']
		sampleMetadataExcluded = pandas.DataFrame(columns=['Sample File Name', 'Sample Base Name', 'SampleType', 'AssayRole', 'Exclusion Details', 'Status'])
		excluded = len(excludedIX)
	except:
		excluded = 0

	if excluded != 0:

		# Stick info of all previously excluded samples together
		for i in excludedIX:
			temp = copy.deepcopy(data.sampleMetadataExcluded[i])
			sampleMetadataExcluded = sampleMetadataExcluded.append(temp.reindex(['Sample File Name', 'Sample Base Name', 'SampleType', 'AssayRole', 'Exclusion Details', 'Status'], axis=1), ignore_index=True)

		excluded = sampleMetadataExcluded.shape[0]

		# Amend for LIMS unavailable
		if not hasattr(sampleMetadataExcluded, 'Status'):
			sampleMetadataExcluded['Status'] = 'Unknown'

		# Sample type masks, and only those marked as 'sample' or 'unknown' flagged
		SSmaskEx = (sampleMetadataExcluded['SampleType'] == SampleType.StudySample) & (sampleMetadataExcluded['AssayRole'] == AssayRole.Assay)
		SPmaskEx = (sampleMetadataExcluded['SampleType'] == SampleType.StudyPool) & (sampleMetadataExcluded['AssayRole'] == AssayRole.PrecisionReference)
		ERmaskEx = (sampleMetadataExcluded['SampleType'] == SampleType.ExternalReference) & (sampleMetadataExcluded['AssayRole'] == AssayRole.PrecisionReference)
		UNDEFmaskEx = pandas.isnull(sampleMetadataExcluded['Sample Base Name'])
		OTHERmaskEx = (SSmaskEx==False) & (SPmaskEx==False) & (ERmaskEx==False) & (UNDEFmaskEx==False)

		sampleSummary['Excluded Details'] = sampleMetadataExcluded.set_index('Sample File Name')


	# Summary table for samples acquired
	temp = numpy.zeros([6,2], dtype=numpy.int)
	# Total numbers
	temp[:,0] = [data.sampleMetadata.shape[0], sum(SSmask), sum(SPmask), sum(ERmask), sum(OTHERmask), sum(UNDEFmask)]
	# Numbers marked for exclusion (either skipped or in sampleMask)
	temp[:,1] = [sum(markedToExclude), sum(markedToExclude & SSmask), sum(markedToExclude & SPmask),
		sum(markedToExclude & ERmask), sum(markedToExclude & OTHERmask), sum(markedToExclude & UNDEFmask)]

	# Convert to dataframe
	sampleSummary['Acquired'] = pandas.DataFrame(data = temp,
		index = ['All', 'Study Sample', 'Study Pool', 'External Reference', 'Other', 'Unknown'],
		columns = ['Total', 'Marked for Exclusion'])

	# Marked for exclusion - details
	if (sum(markedToExclude) != 0):
		sampleSummary['MarkedToExclude Details'] = data.sampleMetadata[['Sample File Name','Exclusion Details']][markedToExclude]

	# Save details of samples of unknown type
	if (sum(UNDEFmask) != 0):
		sampleSummary['UnknownType Details'] = data.sampleMetadata[['Sample File Name']][UNDEFmask]

	# Samples present/absent from LIMS file (if available)
	if hasattr(data, 'limsFile'):
		
		# Acquired table - number marked as missing in LIMS
		LIMSmissing = data.sampleMetadata['LIMS Marked Missing'].values
		sampleSummary['Acquired']['LIMS marked as missing'] = ['-', sum(numpy.logical_and(LIMSmissing, SSmask)), '-', '-', '-', '-']

		# Acquired table - number missing from LIMS
		noLIMS = data.sampleMetadata['LIMS Present'].values==False
		sampleSummary['Acquired']['Missing from LIMS'] = ['-', sum(numpy.logical_and(noLIMS, SSmask)), '-', '-', '-', '-']

		# Marked as missing in LIMS - details
		if (sum(numpy.logical_and(LIMSmissing, SSmask)) != 0):
			sampleSummary['LIMSmissing Details'] = data.sampleMetadata[['Sample File Name',
				'Sampling ID','Status','Exclusion Details']][numpy.logical_and(LIMSmissing, SSmask)]

		# Missing from LIMS - details
		if (sum(noLIMS & SSmask) != 0):
			sampleSummary['NoLIMS Details'] = data.sampleMetadata[['Sample File Name',
				'Sampling ID','Status','Exclusion Details']][numpy.logical_and(noLIMS, SSmask)]


		# Samples not available for acquisition (i.e., in LIMS but not acquired)
		if hasattr(data, 'sampleAbsentMetadata'):
			
			# Masks
			LIMSmissing = data.sampleAbsentMetadata['LIMS Marked Missing'].values==True
			LIMSpresent = data.sampleAbsentMetadata['LIMS Marked Missing'].values==False
					
			SSmaskAbs = (data.sampleAbsentMetadata['SampleType'] == SampleType.StudySample) & (data.sampleAbsentMetadata['AssayRole'] == AssayRole.Assay)
			SPmaskAbs = (data.sampleAbsentMetadata['SampleType'] == SampleType.StudyPool) & (data.sampleAbsentMetadata['AssayRole'] == AssayRole.PrecisionReference)
			ERmaskAbs = (data.sampleAbsentMetadata['SampleType'] == SampleType.ExternalReference) & (data.sampleAbsentMetadata['AssayRole'] == AssayRole.PrecisionReference)

			# Add info for samples already excluded and removed from dataset
			EXmask = numpy.zeros(data.sampleAbsentMetadata.shape[0], dtype=bool)

			if excluded != 0:

				# Prepare sampleMetadataExcluded
				sampleMetadataExcluded = sampleMetadataExcluded.loc[:,['Sample Base Name', 'Exclusion Details']]

				# Remove columns if previously matched
				if hasattr(data.sampleAbsentMetadata, 'Sample Base Name'):
					data.sampleAbsentMetadata.drop(['Sample Base Name'], axis=1, inplace=True)
				if hasattr(data.sampleAbsentMetadata, 'Exclusion Details'):
					data.sampleAbsentMetadata.drop(['Exclusion Details'], axis=1, inplace=True)
				# Drop duplicate rows (e.g., duplicate samples excluded twice)
				sampleMetadataExcluded = sampleMetadataExcluded.drop_duplicates(subset='Sample Base Name')

				# lower case for matching
				data.sampleAbsentMetadata.loc[:,'Assay data name Normalised'] = data.sampleAbsentMetadata['Assay data name'].str.lower()
				sampleMetadataExcluded.loc[:,'Sample Base Name'] = sampleMetadataExcluded['Sample Base Name'].str.lower()

				# Match to already excluded samples
				data.sampleAbsentMetadata = pandas.merge(data.sampleAbsentMetadata, sampleMetadataExcluded, left_on='Assay data name Normalised', right_on='Sample Base Name', how='left', sort=False)
				data.sampleAbsentMetadata.drop(labels=['Assay data name Normalised'], axis=1, inplace=True)

				# Update mask
				EXmask = data.sampleAbsentMetadata['Exclusion Details'].notnull()

			# Missing from acquisition table
			temp = numpy.zeros([4,4], dtype=numpy.int)
			# Totals
			temp[:,0] = [data.sampleAbsentMetadata.shape[0], sum(SSmaskAbs), sum(SPmaskAbs), sum(ERmaskAbs)]
			# Marked as missing in LIMS
			temp[:,1] = [sum(LIMSmissing), sum(LIMSmissing & SSmaskAbs), sum(LIMSmissing & SPmaskAbs), sum(LIMSmissing & ERmaskAbs)]
			# Marked as sample in LIMS
			temp[:,2] = [sum(LIMSpresent), sum(LIMSpresent & SSmaskAbs), sum(LIMSpresent & SPmaskAbs), sum(LIMSpresent & ERmaskAbs)]
			# Already excluded
			temp[:,3] = [sum(EXmask), sum(EXmask & SSmaskAbs), sum(EXmask & SPmaskAbs), sum(EXmask & ERmaskAbs)]
			# Convert to dataframe
			sampleSummary['NotAcquired'] = pandas.DataFrame(data = temp,
				index = ['All', 'Study Sample', 'Study Pool', 'External Reference'],
				columns = ['Total', 'Marked as Missing', 'Marked as Sample', 'Already Excluded'])

			# Missing from acquisition - details - only for those samples not already excluded
			if sum(EXmask==False) != 0:
				sampleSummary['NotAcquired Details'] = data.sampleAbsentMetadata[['Assay data name','Sampling ID','LIMS Marked Missing']][EXmask==False]


	# Samples present/absent from subject information file (if available)
	if hasattr(data, 'subjectInfo'):

		# Acquired table - number of SS acquired with no matching subjectInfo
		sampleSummary['Acquired']['Missing Subject Information'] = ['-', numpy.sum(data.sampleMetadata['Subject ID'].isnull() & SSmask),
			'-', '-' ,'-', '-']

		# Missing subject information - details
		if (sum(data.sampleMetadata['Subject ID'].isnull() & SSmask) != 0):
			sampleSummary['NoSubjectInfo Details'] = data.sampleMetadata[['Sample File Name','Status','Sampling ID', 'SubjectInfoData']][
				(data.sampleMetadata['Subject ID'].isnull() & SSmask)]


	# Finally - add column of samples already excluded to sampleSummary
	if excluded != 0:
		sampleSummary['Acquired']['Already Excluded'] = [excluded, sum(SSmaskEx), sum(SPmaskEx), sum(ERmaskEx), sum(OTHERmaskEx), sum(UNDEFmaskEx)]

	# Drop rows where no samples present for that datatype
	sampleSummary['Acquired'].drop(sampleSummary['Acquired'].index[sampleSummary['Acquired']['Total'].values == 0], axis=0, inplace=True)

	# Update 'All', 'Already Excluded' to only reflect sample types present in data
	if excluded != 0:
		sampleSummary['Acquired'].loc['All','Already Excluded'] = sum(sampleSummary['Acquired']['Already Excluded'][1:])


	# Generate html report
	if output:

		from jinja2 import Environment, FileSystemLoader

		env = Environment(loader=FileSystemLoader(os.path.join(toolboxPath(), 'Templates')))
		template = env.get_template('generateSampleReport.html')
		filename = os.path.join(output, data.name + '_report_sampleSummary.html')
		
		f = open(filename,'w')
		f.write(template.render(item=sampleSummary, version=version, graphicsPath='/report_sampleSummary'))
		f.close()

		copyBackingFiles(toolboxPath(), os.path.join(output, 'graphics', 'report_sampleSummary'))

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
			print('Summary of Samples Missing from Acquisition/Import (i.e., present in LIMS but not acquired/imported)')
			display(sampleSummary['NotAcquired'])
			print('\n')

		if 'MarkedToExclude Details' in sampleSummary:
			print('Details of Samples Marked for Exclusion')
			display(sampleSummary['MarkedToExclude Details'])
			print('\n')
		
		if 'UnknownType Details' in sampleSummary:
			print('Details of Samples with Unknown Type')
			display(sampleSummary['UnknownType Details'])
			print('\n')

		if 'LIMSmissing Details' in sampleSummary:
			print('Details of Samples Marked as Missing in LIMS (i.e., not expected)')
			display(sampleSummary['LIMSmissing Details'])
			print('\n')

		if 'NoLIMS Details' in sampleSummary:
			print('Details of Samples with no Corresponding LIMS Information')
			display(sampleSummary['NoLIMS Details'])
			print('\n')

		if 'NoSubjectInfo Details' in sampleSummary:
			print('Details of Samples with no Corresponding Subject Information')
			display(sampleSummary['NoSubjectInfo Details'])
			print('\n')

		if 'NotAcquired Details' in sampleSummary:
			print('Details of Samples Missing from Acquisition/Import (and not already excluded)')
			display(sampleSummary['NotAcquired Details'])
			print('\n')
