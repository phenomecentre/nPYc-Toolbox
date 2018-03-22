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
		SRDmask = (data.sampleMetadata['AssayRole'] == AssayRole.LinearityReference) & (data.sampleMetadata['SampleType'] == SampleType.StudyPool)
		Blankmask = data.sampleMetadata['SampleType'] == SampleType.ProceduralBlank

	except:
		SSmask = numpy.zeros(len(data.sampleMask)).astype(bool)
		SPmask = numpy.zeros(len(data.sampleMask)).astype(bool)
		ERmask = numpy.zeros(len(data.sampleMask)).astype(bool)
		SRDmask = numpy.zeros(len(data.sampleMask)).astype(bool)
		Blankmask = numpy.zeros(len(data.sampleMask)).astype(bool)

	NotInCSVmask = data.sampleMetadata['Metadata Available'] == False
	UnclearRolemask = (SSmask==False) & (SPmask==False) & (ERmask==False) & (NotInCSVmask==False) & (SRDmask == False) & (Blankmask==False)
	# Samples marked for exclusion (either as marked as skipped or as False in sampleMask)

	try:
		markedToExclude = (data.sampleMetadata['Skipped'].values==True) | (data.sampleMask==False)
	except:
		markedToExclude = data.sampleMask==False

	# Samples already excluded

	# Determine if samples have been excluded
	try:
		excludedIX = [i for i, x in enumerate(data.excludedFlag) if x == 'Samples']
		sampleMetadataExcluded = pandas.DataFrame(columns=['Sample File Name', 'Sample Base Name', 'SampleType', 'AssayRole', 'Exclusion Details', 'Metadata Available'])
		excluded = len(excludedIX)
	except:
		excluded = 0

	if excluded != 0:

		# Stick info of all previously excluded samples together
		for i in excludedIX:
			temp = copy.deepcopy(data.sampleMetadataExcluded[i])
			sampleMetadataExcluded = sampleMetadataExcluded.append(temp.reindex(['Sample File Name', 'Sample Base Name', 'SampleType', 'AssayRole', 'Exclusion Details', 'Metadata Available'], axis=1), ignore_index=True)

		excluded = sampleMetadataExcluded.shape[0]

		# Sample type masks, and only those marked as 'sample' or 'unknown' flagged
		SSmaskEx = (sampleMetadataExcluded['SampleType'] == SampleType.StudySample) & (sampleMetadataExcluded['AssayRole'] == AssayRole.Assay)
		SPmaskEx = (sampleMetadataExcluded['SampleType'] == SampleType.StudyPool) & (sampleMetadataExcluded['AssayRole'] == AssayRole.PrecisionReference)
		ERmaskEx = (sampleMetadataExcluded['SampleType'] == SampleType.ExternalReference) & (sampleMetadataExcluded['AssayRole'] == AssayRole.PrecisionReference)
		SRDmaskEx = (sampleMetadataExcluded['AssayRole'] == AssayRole.LinearityReference) & (sampleMetadataExcluded['SampleType'] == SampleType.StudyPool)
		BlankmaskEx = sampleMetadataExcluded['SampleType'] == SampleType.ProceduralBlank

		NotInCSVmaskEx = sampleMetadataExcluded['Metadata Available'] == False
		UnclearRolemaskEx = (SSmaskEx==False) & (SPmaskEx==False) & (ERmaskEx==False) & (NotInCSVmaskEx==False) & (BlankmaskEx == False) & (SRDmaskEx == False)

		sampleSummary['Excluded Details'] = sampleMetadataExcluded.set_index('Sample File Name')


	# Summary table for samples acquired
	temp = numpy.zeros([8,2], dtype=numpy.int)
	# Total numbers
	temp[:,0] = [data.sampleMetadata.shape[0], sum(SSmask), sum(SPmask), sum(ERmask), sum(SRDmask), sum(Blankmask), sum(NotInCSVmask), sum(UnclearRolemask)]
	# Numbers marked for exclusion (either skipped or in sampleMask)
	temp[:,1] = [sum(markedToExclude), sum(markedToExclude & SSmask), sum(markedToExclude & SPmask),
		sum(markedToExclude & ERmask), sum(markedToExclude & SRDmask), sum(markedToExclude & Blankmask), sum(markedToExclude & NotInCSVmask), sum(markedToExclude & UnclearRolemask)]

	# Convert to dataframe
	sampleSummary['Acquired'] = pandas.DataFrame(data = temp,
		index = ['All', 'Study Sample', 'Study Pool', 'External Reference', 'Serial Dilution', 'Blank Sample', 'No Metadata Available', 'Unspecified Sample Type or Assay Role'],
		columns = ['Total', 'Marked for Exclusion'])

	# Marked for exclusion - details
	if (sum(markedToExclude) != 0):
		sampleSummary['MarkedToExclude Details'] = data.sampleMetadata[['Sample File Name','Exclusion Details']][markedToExclude]

	# Save details of samples of unknown type
	if (sum(NotInCSVmask) != 0):
		sampleSummary['NoMetadata Details'] = data.sampleMetadata[['Sample File Name']][NotInCSVmask]

	# Save details of samples of unknown type
	if (sum(UnclearRolemask) != 0):
		sampleSummary['UnknownType Details'] = data.sampleMetadata[['Sample File Name']][UnclearRolemask]

	# Finally - add column of samples already excluded to sampleSummary
	if excluded != 0:
		sampleSummary['Acquired']['Already Excluded'] = [excluded, sum(SSmaskEx), sum(SPmaskEx), sum(ERmaskEx),
														 sum(SRDmaskEx), sum(BlankmaskEx), sum(NotInCSVmaskEx), sum(UnclearRolemaskEx)]
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

		if 'NoMetadata Details' in sampleSummary:
			print('Details of Samples for which no Metadata was provided')
			display(sampleSummary['NoMetadata Details'])
			print('\n')

		if 'UnknownType Details' in sampleSummary:
			print('Details of Samples with Unknown Type')
			display(sampleSummary['UnknownType Details'])
			print('\n')

		if 'NotAcquired Details' in sampleSummary:
			print('Details of Samples Missing from Acquisition/Import (and not already excluded)')
			display(sampleSummary['NotAcquired Details'])
			print('\n')

		if 'MarkedToExclude Details' in sampleSummary:
			print('Details of Samples Marked for Exclusion')
			display(sampleSummary['MarkedToExclude Details'])
			print('\n')

