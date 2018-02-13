import os
from datetime import datetime
import pandas
import numpy
import math
import numbers
from pathlib import PurePath
import re
from ..enumerations import VariableType, AssayRole, SampleType
import warnings

from ._dataset import Dataset
from ..utilities.nmr import interpolateSpectrum, baselinePWcalcs
from ..utilities.extractParams import buildFileList
from ..utilities._calibratePPMscale import calibratePPM
from ..utilities._lineWidth import lineWidth
from ..utilities._fitPeak import integrateResonance
from ..utilities import removeTrailingColumnNumbering
from .._toolboxPath import toolboxPath
from ..enumerations import VariableType, DatasetLevel, AssayRole, SampleType

class NMRDataset(Dataset):
	"""
	NMRDataset(datapath, fileType='Bruker', sop='GenericNMRurine', pulseprogram= 'noesygpp1d', **kwargs)

	:py:class:`NMRDataset` extends :py:class:`Dataset` to represent both spectral and peak-picked NMR datasets.

	Objects can be initialised from a variety of common data formats, including Bruker-format raw data, and BI-LISA targeted lipoprotein analysis.

	* Bruker
		When loading Bruker format raw spectra (:file:`1r` files), all directores below :file:`datapath` will be scanned for valid raw data, and those matching *pulseprogram* loaded and aligned onto a common scale as defined in *sop*.

	* BI-LISA
		BI-LISA data can be read from Excel workbooks, the name of the sheet containing the data to be loaded should be passed in the *pulseProgram* argument. Feature descriptors will be loaded from the 'Analytes' sheet, and file names converted back to the `ExperimentName/expno` format from `ExperimentName_EXPNO_expno`.

	:param str fileType: Type of data to be loaded
	:param str sheetname: Load data from the specifed sheet of the Excel workbook
	:param str pulseprogram: When loading raw data, only import spectra aquired with *pulseprogram*
	"""

	__importTypes = ['Bruker', 'BI-LISA'] # Raw data types we understand

	def __init__(self, datapath, fileType='Bruker', pulseProgram='noesygppr1d', sop='GenericNMRurine', dataType = 'Bruker', xmlFileName=r'.*?results\.xml$', pdata=1, **kwargs):
		"""
		NMRDataset(datapath, fileType='Bruker', sop='GenericNMRurine', pulseprogram='noesygpp1d', **kwargs)

		:py:class:`NMRDataset` extends :py:class:`Dataset` to represent both spectral and peak-picked NMR datasets.

		Objects can be initialised from a variety of common data formats, including Bruker-format raw data, and BI-LISA targeted lipoprotein analysis.

		* Bruker
			When loading Bruker format raw spectra (1r files), all directores below :file:`datapath` will be scanned for valid raw data, and those matching *pulseprogram* loaded and aligned onto a common scale as defined in *sop*.

		* BI-LISA
			BI-LISA data can be read from Excel workbooks, the name of the sheet containing the data to be loaded should be passed in the *pulseProgram* argument. Feature descriptors will be loaded from the 'Analytes' sheet, and file names converted back to the `ExperimentName/expno` format from `ExperimentName_EXPNO_expno`.

		:param str fileType: Type of data to be loaded
		:param str sheetname: Load data from the specifed sheet of the Excel workbook
		:param str pulseprogram: When loading raw data, only import spectra aquired with *pulseprogram*
		"""
		super().__init__(sop=sop, **kwargs)

		#assert fileType in self.__importTypes, "%s is not a filetype understood by NMRDataset." % (fileType)
		self.filePath, fileName = os.path.split(datapath)
		self.fileName, fileExtension = os.path.splitext(fileName)

		if fileType == 'Bruker':
			from ..utilities._importBrukerSpectrum import importBrukerSpectra

			self.Attributes['Feature Names'] = 'ppm'

			##
			# Some input validation
			##
			if not isinstance(self.Attributes['calibrateTo'], numbers.Number):
				raise TypeError("calibrateTo field in SOP should be a numerical value")
			if not isinstance(self.Attributes['bounds'], list):
				raise TypeError("bounds field in SOP should be a list on numeric values example [-1,10]")
			if not isinstance(self.Attributes['variableSize'], int):
				raise TypeError("variableSize field in SOP should be a integer.")

			self.VariableType = VariableType.Spectral
			self.Attributes['pulseProgram'] = pulseProgram

			##
			# Load data
			##
			(self._intensityData, ppm, self.sampleMetadata) = importBrukerSpectra(datapath,
																				  pulseProgram,
																				  pdata,
																				  self.Attributes)
			self.featureMetadata = pandas.DataFrame(ppm, columns=['ppm'])

			##
			# Set up additional metadata columns
			##
			self.sampleMetadata['Acquired Time'] = pandas.to_datetime(self.sampleMetadata['Acquired Time']).astype(datetime)
			self.sampleMetadata['AssayRole'] = AssayRole.Assay
			self.sampleMetadata['SampleType'] = SampleType.StudySample
			self.sampleMetadata['Dilution'] = 100
			self.sampleMetadata['Batch'] = numpy.nan
			self.sampleMetadata['Correction Batch'] = numpy.nan
			runOrder = self.sampleMetadata.sort_values(by='Acquired Time').index.values
			self.sampleMetadata['Run Order'] = numpy.argsort(runOrder)
			self.sampleMetadata['Sampling ID'] = numpy.nan
			self.sampleMetadata['Exclusion Details'] = self.sampleMetadata['Warnings']
			self.sampleMetadata.drop('Warnings', inplace=True, axis=1)

			self.initialiseMasks()
			self.sampleMask = (self.sampleMetadata['Exclusion Details'] == '').values

			self.addSampleInfo(descriptionFormat='Filenames')

			##
			# Do per-dataset QC work here
			##
			# TODO - refactor tp seperate these QC checks
			bounds = numpy.std(self.sampleMetadata['Delta PPM']) * 3
			meanVal = numpy.mean(self.sampleMetadata['Delta PPM'])
			self.sampleMetadata['calibrPass'] = numpy.logical_or((self.sampleMetadata['Delta PPM'] > meanVal - bounds),
																 (self.sampleMetadata['Delta PPM'] < meanVal + bounds))
			self._scale = self.featureMetadata['ppm'].values
			self._calcBLWP_PWandMerge()
			self.featureMask[:] = True
			self.sampleMask = self.sampleMetadata['overallFail'] == False
			self.Attributes['Log'].append([datetime.now(), 'Bruker format spectra loaded from %s' % (datapath)])

		elif fileType == 'BI-LISA':
			self._importBILISAData(datapath, pulseProgram)
			self.VariableType = VariableType.Discrete
			self.name = self.fileName

		elif fileType == 'empty':
			# Lets us build an empty object for testing &c
			pass
		else :
			raise NotImplementedError('%s is not a format understood by NMRDataset.' % (fileType))

		# Log init
		self.Attributes['Log'].append([datetime.now(), '%s instance initiated, with %d samples, %d features, from %s' % (self.__class__.__name__, self.noSamples, self.noFeatures, datapath)])


	def _importBILISAData(self, datapath, sheetname):
		"""
		find and load all the BILISA format data

		"""
		##
		# Load excel sheet
		##
		dataT = pandas.read_excel(datapath, sheet_name=sheetname)
		featureMappings = pandas.read_excel(datapath, sheet_name='Analytes')

		# Extract data
		self._intensityData = dataT.iloc[:,1:].as_matrix()

		##
		# Parse feature names into name + unit
		##
		varSpec = r"""
					^
					(?P<componentName>\w+?)
					\s
					in
					\s
					(?P<unit>.+)
					$
					"""
		# Get column headers and transpose
		varParser = re.compile(varSpec, re.VERBOSE)
		features = dataT.transpose().reset_index()['index']
		features.drop(0, axis=0, inplace=True)

		# Seperate unit and variable type
		featureMetadata = features.str.extract(varParser, expand=False)
		featureMetadata.reset_index(inplace=True)
		featureMetadata.drop('index', axis=1, inplace=True)

		# Rename Columns
		featureMetadata = featureMetadata.merge(featureMappings, on=None, left_on='componentName', right_on='Name', how='left')
		featureMetadata.drop('unit', axis=1, inplace=True)
		featureMetadata.drop('componentName', axis=1, inplace=True)
		featureMetadata.rename(columns={'Matrix': 'Component'}, inplace=True)
		featureMetadata.rename(columns={'Name': 'Feature Name'}, inplace=True)

		self.featureMetadata = featureMetadata

		##
		# Convert sample IDs back to folder/expno
		##
		filenameSpec = r"""
						^
						(?P<Rack>\w+?)
						_EXPNO_
						(?P<expno>.+)
						$
						"""
		fileNameParser = re.compile(filenameSpec, re.VERBOSE)
		sampleMetadata = dataT['SampleID'].str.extract(fileNameParser, expand=False)
		sampleMetadata['Sample File Name'] = sampleMetadata['Rack'].str.cat(sampleMetadata['expno'], sep='/')

		sampleMetadata['expno'] = pandas.to_numeric(sampleMetadata['expno'])

		sampleMetadata['Sample Base Name'] = sampleMetadata['Sample File Name']
		sampleMetadata['Exclusion Details'] = None
		sampleMetadata['AssayRole'] = numpy.nan
		sampleMetadata['SampleType'] = numpy.nan
		sampleMetadata['Dilution'] = numpy.nan
		sampleMetadata['Batch'] = numpy.nan
		sampleMetadata['Correction Batch'] = numpy.nan
		sampleMetadata['Run Order'] = numpy.nan
		sampleMetadata['Sampling ID'] = numpy.nan
		sampleMetadata['Acquired Time'] = numpy.nan

		self.sampleMetadata = sampleMetadata

		self.initialiseMasks()

		self.Attributes['Log'].append([datetime.now(), 'BI-LISA data loaded from %s' % (datapath)])


	def _calcBLWP_PWandMerge(self):#,scalePPM, intenData, start, stop, sampleType, filePathList, sf):

		"""
		calls the baselinePWcalcs function and works out fails and merges the dataframes saves as part of thenmrData object
		params:
			Input: nmrData object

		"""
		self.sampleMetadata['ImportFail'] = False
		if self.Attributes['pulseProgram'] in ['cpmgpr1d', 'noesygppr1d', 'noesypr1d']:#only for 1D data
			[rawDataDf, featureMask,  BL_lowRegionFrom, BL_highRegionTo, WP_lowRegionFrom, WP_highRegionTo] = baselinePWcalcs(self._scale,self._intensityData, -0.2, 0.2, None, self.sampleMetadata['File Path'], max(self.sampleMetadata['SF']),self.Attributes['pulseProgram'], self.Attributes['baseline_alpha'], self.Attributes['baseline_threshold'], self.Attributes['baselineLow_regionTo'], self.Attributes['baselineHigh_regionFrom'], self.Attributes['waterPeakCutRegionA'], self.Attributes['waterPeakCutRegionB'], self.Attributes['LWpeakRange'][0], self.Attributes['LWpeakRange'][1], self.featureMask)

#			stick these in as attributes
			self.Attributes['BL_lowRegionFrom']= BL_lowRegionFrom
			self.Attributes['BL_highRegionTo']= BL_highRegionTo
			self.Attributes['WP_lowRegionFrom']= WP_lowRegionFrom
			self.Attributes['WP_highRegionTo']= WP_highRegionTo

#			 merge
			self.sampleMetadata = pandas.merge(self.sampleMetadata, rawDataDf, on='File Path', how='left', sort=False)

			#create new column and mark as failed
			self.sampleMetadata['overallFail'] = True
			for i in range (len(self.sampleMetadata)):
				if self.sampleMetadata.ImportFail[i] ==False and self.sampleMetadata.loc[i, 'Line Width (Hz)'] >0 and self.sampleMetadata.loc[i, 'Line Width (Hz)']<self.Attributes['PWFailThreshold'] and self.sampleMetadata.BL_low_outliersFailArea[i] == False and self.sampleMetadata.BL_low_outliersFailNeg[i] == False and self.sampleMetadata.BL_high_outliersFailArea[i] == False and self.sampleMetadata.BL_high_outliersFailNeg[i] == False and self.sampleMetadata.WP_low_outliersFailArea[i] == False and self.sampleMetadata.WP_low_outliersFailNeg[i] == False and self.sampleMetadata.WP_high_outliersFailArea[i] == False and self.sampleMetadata.WP_high_outliersFailNeg[i] == False and self.sampleMetadata.calibrPass[i] == True:
					self.sampleMetadata.loc[i,('overallFail')] = False
				else:
					self.sampleMetadata.loc[i,('overallFail')] = True
			self.Attributes['Log'].append([datetime.now(), 'data merged Total samples %s, Failed samples %s' % (str(len(self.sampleMetadata)), str(len(self.sampleMetadata[self.sampleMetadata.overallFail ==True])))])
		else:
			self.Attributes['Log'].append([datetime.now(), 'Total samples %s', 'Failed samples %s' % (str(len(self.sampleMetadata)),(str(len(self.sampleMetadata[self.sampleMetadata.ImportFail ==False]))))])

		self.sampleMetadata['exceed90critical'] = False#create new df column
		for i in range (len(self.sampleMetadata)):
			if self.sampleMetadata.BL_low_outliersFailArea[i] == False and self.sampleMetadata.BL_low_outliersFailNeg[i] == False and self.sampleMetadata.BL_high_outliersFailArea[i] == False and self.sampleMetadata.BL_high_outliersFailNeg[i] == False and self.sampleMetadata.WP_low_outliersFailArea[i] == False and self.sampleMetadata.WP_low_outliersFailNeg[i] == False and self.sampleMetadata.WP_high_outliersFailArea[i] == False and self.sampleMetadata.WP_high_outliersFailNeg[i] == False:
				self.sampleMetadata.loc[i,('exceed90critical')] = False
			else:
				self.sampleMetadata.loc[i,('exceed90critical')] = True


	def addSampleInfo(self, descriptionFormat=None, filePath=None, filenameSpec=None, **kwargs):
		"""
		Load additional metadata and map it in to the :py:attr:`~Dataset.sampleMetadata` table.

		Possible options:

		* **'NPC LIMS'** NPC LIMS files mapping files names of raw analytical data to sample IDs
		* **'NPC Subject Info'** Map subject metadata from a NPC sample manifest file (format defined in 'PCSOP.082')
		* **'Raw Data'** Extract analytical parameters from raw data files
		* **'ISATAB'** ISATAB study designs
		* **'Filenames'** Parses sample information out of the filenames, based on the named capture groups in the regex passed in *filenamespec*
		* **'Basic CSV'** Joins the :py:attr:`sampleMetadata` table with the data in the ``csv`` file at *filePath=*, matching on the 'Sample File Name' column in both.

		:param str descriptionFormat: Format of metadata to be added
		:param str filePath: Path to the additional data to be added
		:param filenameSpec: Only used if *descriptionFormat* is 'Filenames'. A regular expression that extracts sample-type information into the following named capture groups: 'fileName', 'baseName', 'study', 'chromatography' 'ionisation', 'instrument', 'groupingKind' 'groupingNo', 'injectionKind', 'injectionNo', 'reference', 'exclusion' 'reruns', 'extraInjections', 'exclusion2'. if ``None`` is passed, use the *filenameSpec* key in *Attributes*, loaded from the SOP json
		:type filenameSpec: None or str
		:raises NotImplementedError: if the descriptionFormat is not understood
		"""

		if descriptionFormat.lower() == 'filenames':
			if filenameSpec is None: # Use spec from SOP
				filenameSpec = self.Attributes['filenameSpec']
			self._getSampleMetadataFromFilename(filenameSpec)
		else:
			super().addSampleInfo(descriptionFormat=descriptionFormat, filePath=filePath, filenameSpec=filenameSpec, **kwargs)


	def _matchDatasetToLIMS(self, pathToLIMSfile):
		"""
		Establish the `Sampling ID` by matching the `Sample Base Name` with the LIMS file information.

		:param str pathToLIMSfile: Path to LIMS file for map Sampling ID
		"""

		# Prepare input
		if 'Sample Base Name' not in self.sampleMetadata.columns:
			# if 'Sample Base Name' is missing, generate it
			from ..utilities.nmr import generateBaseName
			self.sampleMetadata.loc[:, 'Sample Base Name'], self.sampleMetadata.loc[:, 'expno'] = generateBaseName(self.sampleMetadata)

		# Merge LIMS file using Dataset method
		Dataset._matchDatasetToLIMS(self, pathToLIMSfile)

		# if rerun, keep the latest (highest expno)
		sampleMetadataSorted = self.sampleMetadata.sort_values('expno')
		rerunsMask = sampleMetadataSorted.duplicated(subset='Sample Base Name', keep='last')
		if any(rerunsMask):
			self.sampleMask[rerunsMask.sort_index() == True] = False
			warnings.warn(str(sum(rerunsMask)) + ' previous acquisitions masked, latest is kept.')
			self.Attributes['Log'].append([datetime.now(), 'Reacquistions detected, previous acquisition marked for exclusion.'])

		# Set SampleType and AssayRole
		self.sampleMetadata.loc[:, 'AssayRole'] = AssayRole.Assay
		self.sampleMetadata.loc[self.sampleMetadata.loc[:, 'Status'].str.match('Study Reference', na=False).astype(bool), 'AssayRole'] = AssayRole.PrecisionReference
		self.sampleMetadata.loc[self.sampleMetadata.loc[:, 'Status'].str.match('Long Term Reference', na=False).astype(bool), 'AssayRole'] = AssayRole.PrecisionReference
		self.sampleMetadata.loc[:, 'SampleType'] = SampleType.StudySample
		self.sampleMetadata.loc[self.sampleMetadata.loc[:, 'Status'].str.match('Study Reference', na=False).astype(bool), 'SampleType'] = SampleType.StudyPool
		self.sampleMetadata.loc[self.sampleMetadata.loc[:, 'Status'].str.match('Long Term Reference', na=False).astype(bool), 'SampleType'] = SampleType.ExternalReference

		# Update Sampling ID values using new 'SampleType', special case for Study Pool, External Reference and Procedural Blank
		self.sampleMetadata.loc[(((self.sampleMetadata['Sampling ID'] == 'Not specified') | (self.sampleMetadata['Sampling ID'] == 'Present but undefined in the LIMS file')) & (self.sampleMetadata['SampleType'] == SampleType.StudyPool)).tolist(), 'Sampling ID'] = 'Study Pool Sample'
		self.sampleMetadata.loc[(((self.sampleMetadata['Sampling ID'] == 'Not specified') | (self.sampleMetadata['Sampling ID'] == 'Present but undefined in the LIMS file')) & (self.sampleMetadata['SampleType'] == SampleType.ExternalReference)).tolist(), 'Sampling ID'] = 'External Reference Sample'
		self.sampleMetadata.loc[(((self.sampleMetadata['Sampling ID'] == 'Not specified') | (self.sampleMetadata['Sampling ID'] == 'Present but undefined in the LIMS file')) & (self.sampleMetadata['SampleType'] == SampleType.ProceduralBlank)).tolist(), 'Sampling ID'] = 'Procedural Blank Sample'

		# Neater output
		#self.sampleMetadata.loc[self.sampleMetadata['Sample position'] == 'nan', 'Sample position'] = ''


	def _getSampleMetadataFromFilename(self, filenameSpec):
		"""
		Infer sample acquisition metadata from standardised filename template.
		"""

		# Break filename down into constituent parts.
		baseNameParser = re.compile(filenameSpec, re.VERBOSE)
		fileNameParts = self.sampleMetadata['Sample File Name'].str.extract(baseNameParser, expand=False)

		self.sampleMetadata['Rack'] = fileNameParts['rack'].astype(int, errors='ignore')
		self.sampleMetadata['Study'] = fileNameParts['study']

		##
		# Generate a sample base name for the loaded dataset by rounding by 10
		# Only do this if 'Sample Base Name' is not defined already
		##
		if 'Sample Base Name' not in self.sampleMetadata.columns:
			from ..utilities.nmr import generateBaseName

			self.sampleMetadata.loc[:,'Sample Base Name'], self.sampleMetadata.loc[:,'expno'] = generateBaseName(self.sampleMetadata)

		self.Attributes['Log'].append([datetime.now(), 'Sample metadata parsed from filenames.'])


	def updateMasks(self, filterSamples=True, filterFeatures=True, sampleTypes=[SampleType.StudySample, SampleType.StudyPool], assayRoles=[AssayRole.Assay, AssayRole.PrecisionReference], exclusionRegions=None, **kwargs):
		"""
		Update :py:attr:`~Dataset.sampleMask` and :py:attr:`~Dataset.featureMask` according to parameters.

		:py:meth:`updateMasks` sets :py:attr:`~Dataset.sampleMask` or :py:attr:`~Dataset.featureMask` to ``False`` for those items failing analytical criteria.

		.. note:: To avoid reintroducing items manually excluded, this method only ever sets items to ``False``, therefore if you wish to move from more stringent criteria to a less stringent set, you will need to reset the mask to all ``True`` using :py:meth:`~Dataset.initialiseMasks`.

		:param bool filterSamples: If ``False`` don't modify sampleMask
		:param bool filterFeatures: If ``False`` don't modify featureMask
		:param sampleTypes: List of types of samples to retain
		:type sampleTypes: SampleType
		:param AssayRole sampleRoles: List of assays roles to retain
		:param exclusionRegions: If ``None`` Exclude ranges defined in :py:attr:`~Dataset.Attributes`['exclusionRegions']
		:type exclusionRegions: list of tuple
		"""

		# Feature exclusions
		if filterFeatures:
			if exclusionRegions is None and 'exclusionRegions' in self.Attributes.keys():
				exclusionRegions = self.Attributes['exclusionRegions']

			elif isinstance(exclusionRegions, list):
				pass

			elif isinstance(exclusionRegions, tuple):
				exclusionRegions = [exclusionRegions]

			if exclusionRegions is None:
				raise ValueError('No exclusion regions supplied')

			for region in exclusionRegions:
				low = region[0]
				high = region[1]

				if low == high:
					warnings.warn('Low (%.2f) and high (%.2f) bounds are identical, skipping region' % (low, high))
					continue

				elif low > high:
					low = high
					high = region[0]

				regionMask = numpy.logical_or(self.featureMetadata['ppm'].values < low,
											  self.featureMetadata['ppm'].values > high)

				self.featureMask = numpy.logical_and(self.featureMask,
													 regionMask)

		# Sample Exclusions
		if filterSamples:

			super().updateMasks(filterSamples=True,
								filterFeatures=False,
								sampleTypes=sampleTypes,
								assayRoles=assayRoles,
								**kwargs)

		self.Attributes['Log'].append([datetime.now(), "Dataset filtered with: filterSamples=%s, filterFeatures=%s, sampleClasses=%s, sampleRoles=%s, %s." % (
			filterSamples,
			filterFeatures,
			sampleTypes,
			assayRoles,
			', '.join("{!s}={!r}".format(key,val) for (key,val) in kwargs.items()))])


	def _exportISATAB(self, destinationPath, detailsDict):
		"""
		Export the dataset's metadata to the directory *destinationPath* as ISATAB

		:param str destinationPath: Path to a directory in which the output will be saved
		:param bool escapeDelimiters: Remove characters commonly used as delimiters in csv files from metadata
		:raises IOError: If writing one of the files fails
		"""
		import pandas as pd
		from isatools.model import Investigation, Study, Assay, OntologyAnnotation, OntologySource, Person,Publication,Protocol, Source
		from isatools.model import  Comment, Sample, Characteristic, Process, Material, DataFile, ParameterValue, plink


		from isatools import isatab
		import datetime
		#import numpy as np

		investigation = Investigation()

		investigation.identifier = detailsDict['investigation_identifier']
		investigation.title = detailsDict['investigation_title']
		investigation.description = detailsDict['investigation_description']
		investigation.submission_date = detailsDict['investigation_submission_date']#use today if not specified
		investigation.public_release_date = detailsDict['investigation_public_release_date']
		study = Study(filename='s_'+detailsDict['study_filename']+'.txt')
		study.identifier = detailsDict['study_identifier']
		study.title = detailsDict['study_title']
		study.description = detailsDict['study_description']
		study.submission_date = detailsDict['study_submission_date']
		study.public_release_date = detailsDict['study_public_release_date']
		investigation.studies.append(study)
		obi = OntologySource(name='OBI', description="Ontology for Biomedical Investigations")
		investigation.ontology_source_references.append(obi)
		intervention_design = OntologyAnnotation(term_source=obi)
		intervention_design.term = "intervention design"
		intervention_design.term_accession = "http://purl.obolibrary.org/obo/OBI_0000115"
		study.design_descriptors.append(intervention_design)

		# Other instance variables common to both Investigation and Study objects include 'contacts' and 'publications',
		# each with lists of corresponding Person and Publication objects.

		contact = Person(first_name=detailsDict['first_name'], last_name=detailsDict['last_name'], affiliation=detailsDict['affiliation'], roles=[OntologyAnnotation(term='submitter')])
		study.contacts.append(contact)
		publication = Publication(title="Experiments with Data", author_list="Auther 1, Author 2")
		publication.pubmed_id = "12345678"
		publication.status = OntologyAnnotation(term="published")
		study.publications.append(publication)

		# To create the study graph that corresponds to the contents of the study table file (the s_*.txt file), we need
		# to create a process sequence. To do this we use the Process class and attach it to the Study object's
		# 'process_sequence' list instance variable. Each process must be linked with a Protocol object that is attached to
		# a Study object's 'protocols' list instance variable. The sample collection Process object usually has as input
		# a Source material and as output a Sample material.

		sample_collection_protocol = Protocol(id_="sample collection",name="sample collection",protocol_type=OntologyAnnotation(term="sample collection"))
		aliquoting_protocol = Protocol(id_="aliquoting",name="aliquoting",protocol_type=OntologyAnnotation(term="aliquoting"))


		for index, row in self.sampleMetadata.iterrows():
		    src_name = row['Subject ID'] if row['Subject ID'] is not '' else row['Sampling ID']
		    source = Source(name=src_name)

		    source.comments.append(Comment(name='Study Name', value=row['Study']))
		    study.sources.append(source)

		    sample_name = row['Sampling ID'] if not pd.isnull(row['Sampling ID']) else row['Subject ID']
		    sample = Sample(name=sample_name, derives_from=[source])

		    characteristic_material_type = Characteristic(category=OntologyAnnotation(term="material type"), value=detailsDict['study_material_type'])
		    sample.characteristics.append(characteristic_material_type)

		    characteristic_material_role = Characteristic(category=OntologyAnnotation(term="material role"), value=row['AssayRole'])
		    sample.characteristics.append(characteristic_material_role)

		    # perhaps check if field exists first
		    characteristic_age = Characteristic(category=OntologyAnnotation(term="Age"), value=row['Age'],unit='Year')
		    sample.characteristics.append(characteristic_age)
		    # perhaps check if field exists first
		    characteristic_gender = Characteristic(category=OntologyAnnotation(term="Gender"), value=row['Gender'])
		    sample.characteristics.append(characteristic_gender)

		    ncbitaxon = OntologySource(name='NCBITaxon', description="NCBI Taxonomy")
		    characteristic_organism = Characteristic(category=OntologyAnnotation(term="Organism"),value=OntologyAnnotation(term="Homo Sapiens", term_source=ncbitaxon,term_accession="http://purl.bioontology.org/ontology/NCBITAXON/9606"))
		    sample.characteristics.append(characteristic_organism)

		    study.samples.append(sample)


		    sample_collection_process = Process(id_='sam_coll_proc',executes_protocol=sample_collection_protocol,date_=row['Sampling Date'])

		    aliquoting_process = Process(id_='sam_coll_proc',executes_protocol=aliquoting_protocol,date_=row['Sampling Date'])

		    sample_collection_process.inputs = [source]
		    aliquoting_process.outputs = [sample]

		    # links processes
		    plink(sample_collection_process, aliquoting_process)

		    study.process_sequence.append(sample_collection_process)
		    study.process_sequence.append(aliquoting_process)


		study.protocols.append(sample_collection_protocol)
		study.protocols.append(aliquoting_protocol)

		### Add NMR Assay ###
		nmr_assay = Assay(filename='a_'+detailsDict['assay_filename']+'.txt',measurement_type=OntologyAnnotation(term="metabolite profiling"),technology_type=OntologyAnnotation(term="nmr spectroscopy"))
		extraction_protocol = Protocol(name='extraction', protocol_type=OntologyAnnotation(term="material extraction"))

		study.protocols.append(extraction_protocol)
		nmr_protocol = Protocol(name='nmr spectroscopy', protocol_type=OntologyAnnotation(term="NMR Assay"))
		nmr_protocol.add_param('Run Order')
		nmr_protocol.add_param('Instrument')
		nmr_protocol.add_param('Sample Batch')
		nmr_protocol.add_param('Acquisition Batch')


		study.protocols.append(nmr_protocol)

		#for index, row in sampleMetadata.iterrows():
		for index, sample in enumerate(study.samples):
		    #print(sample.name)
		    row = self.sampleMetadata.loc[self.sampleMetadata['Sampling ID'].astype(str) == sample.name]
		    if row.empty:
		        row = self.sampleMetadata.loc[self.sampleMetadata['Subject ID'].astype(str) == sample.name]

		    # create an extraction process that executes the extraction protocol
		    extraction_process = Process(executes_protocol=extraction_protocol)

		    # extraction process takes as input a sample, and produces an extract material as output
		    sample_name = sample.name
		    sample = Sample(name=sample_name, derives_from=[source])
		    #print(row['Acquired Time'].values[0])

		    extraction_process.inputs.append(sample)
		    material = Material(name="extract-{}".format(index))
		    material.type = "Extract Name"
		    extraction_process.outputs.append(material)

		    # create a ms process that executes the nmr protocol
		    nmr_process = Process(executes_protocol=nmr_protocol,date_=datetime.datetime.isoformat(datetime.datetime.strptime(str(row['Acquired Time'].values[0]), '%Y-%m-%d %H:%M:%S')))

		    nmr_process.name = "assay-name-{}".format(index)
		    nmr_process.inputs.append(extraction_process.outputs[0])
		    # nmr process usually has an output data file
		    datafile = DataFile(filename=row['Assay data name'].values[0], label="NMR Assay Name", generated_from=[sample])
		    nmr_process.outputs.append(datafile)

		    #nmr_process.parameter_values.append(ParameterValue(category='Run Order',value=str(i)))
		    nmr_process.parameter_values = [ParameterValue(category=nmr_protocol.get_param('Run Order'),value=row['Run Order'].values[0])]
		    nmr_process.parameter_values.append(ParameterValue(category=nmr_protocol.get_param('Instrument'),value=row['Instrument'].values[0]))
		    nmr_process.parameter_values.append(ParameterValue(category=nmr_protocol.get_param('Sample Batch'),value=row['Sample batch'].values[0]))
		    nmr_process.parameter_values.append(ParameterValue(category=nmr_protocol.get_param('Acquisition Batch'),value=row['Batch'].values[0]))

		    # ensure Processes are linked forward and backward
		    plink(extraction_process, nmr_process)
		    # make sure the extract, data file, and the processes are attached to the assay
		    nmr_assay.samples.append(sample)
		    nmr_assay.data_files.append(datafile)
		    nmr_assay.other_material.append(material)
		    nmr_assay.process_sequence.append(extraction_process)
		    nmr_assay.process_sequence.append(nmr_process)
		    nmr_assay.measurement_type = OntologyAnnotation(term="metabolite profiling")
		    nmr_assay.technology_type = OntologyAnnotation(term="nmr spectroscopy")

		# attach the assay to the study
		study.assays.append(nmr_assay)


		_ = isatab.dump(isa_obj=investigation, output_path=destinationPath)



def main():
	pass

if __name__=='__main__':
	main()
