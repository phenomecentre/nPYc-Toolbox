import os
from datetime import datetime
import pandas
import numpy
import numbers
import re
import warnings

from ._dataset import Dataset
from ..utilities import removeTrailingColumnNumbering
from .._toolboxPath import toolboxPath
from ..enumerations import VariableType, AssayRole, SampleType
from ..utilities._nmr import qcCheckBaseline, qcCheckWaterPeak


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

	__importTypes = ['Bruker'] # Raw data types we understand

	def __init__(self, datapath, fileType='Bruker', pulseProgram='noesygppr1d', sop='GenericNMRurine', pdata=1, **kwargs):
		"""
		NMRDataset(datapath, fileType='Bruker', sop='GenericNMRurine', pulseprogram='noesygpp1d', **kwargs)

		:py:class:`NMRDataset` extends :py:class:`Dataset` to represent both spectral and peak-picked NMR datasets.

		Objects can be initialised from a variety of common data formats, including Bruker-format raw data, and BI-LISA targeted lipoprotein analysis.

		* Bruker
			When loading Bruker format raw spectra (1r files), all directores below :file:`datapath` will be scanned for valid raw data, and those matching *pulseprogram* loaded and aligned onto a common scale as defined in *sop*.

		:param str fileType: Type of data to be loaded
		:param str sheetname: Load data from the specifed sheet of the Excel workbook
		:param str pulseprogram: When loading raw data, only import spectra aquired with *pulseprogram*
		"""
		super().__init__(sop=sop, **kwargs)

		#assert fileType in self.__importTypes, "%s is not a filetype understood by NMRDataset." % (fileType)
		self.filePath, fileName = os.path.split(datapath)
		self.fileName, fileExtension = os.path.splitext(fileName)

		if fileType.lower() == 'bruker':
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
			self.sampleMetadata['Acquired Time'] = pandas.to_datetime(self.sampleMetadata['Acquired Time'], utc=True).dt.tz_localize(None)
			self.sampleMetadata['Acquired Time'] = self.sampleMetadata['Acquired Time'].dt.to_pydatetime()

			self.sampleMetadata['AssayRole'] = None#AssayRole.Assay
			self.sampleMetadata['SampleType'] = None#SampleType.StudySample
			self.sampleMetadata['Dilution'] = 100
			self.sampleMetadata['Batch'] = numpy.nan
			self.sampleMetadata['Correction Batch'] = numpy.nan
			runOrder = self.sampleMetadata.sort_values(by='Acquired Time').index.values
			self.sampleMetadata['Run Order'] = numpy.argsort(runOrder)
			self.sampleMetadata['Sampling ID'] = numpy.nan
			self.sampleMetadata['Exclusion Details'] = self.sampleMetadata['Warnings']
			self.sampleMetadata['Metadata Available'] = False
			self.sampleMetadata.drop('Warnings', inplace=True, axis=1)

			self.initialiseMasks()
			self.sampleMask = (self.sampleMetadata['Exclusion Details'] == '').values

			self.addSampleInfo(descriptionFormat='Filenames')

			self._scale = self.featureMetadata['ppm'].values
			self.featureMask[:] = True
			# Perform the quaality control checks to populate the class
			self._nmrQCChecks()

			self.Attributes['Log'].append([datetime.now(), 'Bruker format spectra loaded from %s' % (datapath)])
		elif fileType.lower() == 'csv export':
			(self.name, self.intensityData, self.featureMetadata, self.sampleMetadata) = self._initialiseFromCSV(datapath)
			self.VariableType = VariableType.Spectral
			self.initialiseMasks()
		elif fileType == 'empty':
			# Lets us build an empty object for testing &c
			pass
		else:
			raise NotImplementedError('%s is not a format understood by NMRDataset.' % (fileType))

		# Log init
		self.Attributes['Log'].append([datetime.now(), '%s instance initiated, with %d samples, %d features, from %s' % (self.__class__.__name__, self.noSamples, self.noFeatures, datapath)])


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
		elif descriptionFormat == 'ISATAB':
			super().addSampleInfo(descriptionFormat=descriptionFormat, filePath=filePath, **kwargs)
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
			from ..utilities._nmr import generateBaseName
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
			from ..utilities._nmr import generateBaseName

			self.sampleMetadata.loc[:,'Sample Base Name'], self.sampleMetadata.loc[:,'expno'] = generateBaseName(self.sampleMetadata)

		self.sampleMetadata['Metadata Available'] = True

		self.Attributes['Log'].append([datetime.now(), 'Sample metadata parsed from filenames.'])


	def updateMasks(self, filterSamples=True, filterFeatures=True,
					sampleTypes=list(SampleType),#[SampleType.StudySample, SampleType.StudyPool],
					assayRoles=list(AssayRole),#[AssayRole.Assay, AssayRole.PrecisionReference],
					exclusionRegions=None,
					sampleQCChecks=['LineWidthFail','CalibrationFail','BaselineFail','WaterPeakFail'],**kwargs):
		"""
		Update :py:attr:`~Dataset.sampleMask` and :py:attr:`~Dataset.featureMask` according to parameters.

		:py:meth:`updateMasks` sets :py:attr:`~Dataset.sampleMask` or :py:attr:`~Dataset.featureMask` to ``False`` for those items failing analytical criteria.

		.. note:: To avoid reintroducing items manually excluded, this method only ever sets items to ``False``, therefore if you wish to move from more stringent criteria to a less stringent set, you will need to reset the mask to all ``True`` using :py:meth:`~Dataset.initialiseMasks`.

		:param bool filterSamples: If ``False`` don't modify sampleMask
		:param bool filterFeatures: If ``False`` don't modify featureMask
		:param sampleTypes: List of types of samples to retain
		:type sampleTypes: SampleType
		:param AssayRole sampleRoles: List of assays roles to retain
		:param exclusionRegions: If ``None`` Exclude ranges defined in :py:attr:`~Dataset.Attributes`\['exclusionRegions'\]
		:param list sampleQCChecks: Which quality control metrics to use.
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

			self.excludeFeatures(exclusionRegions, on='ppm')
			# If features are modified, retrigger
			self._nmrQCChecks()

		# Sample Exclusions
		if filterSamples:

			# Retrigger QC checks before checking which samples need to be updated
			if not filterFeatures:
				self._nmrQCChecks()

			super().updateMasks(filterSamples=True,
								filterFeatures=False,
								sampleTypes=sampleTypes,
								assayRoles=assayRoles,
								**kwargs)
			columnNames = ['Sample File Name']
			columnNames.extend(sampleQCChecks)
			fail_summary = self.sampleMetadata.loc[:, columnNames]

			fail_summary = fail_summary[(fail_summary == True).any(1)]

			idxToMask = fail_summary.index

			for idx, row in fail_summary.iterrows():
				exclusion_message = ""
				for qc_check in range(1, len(columnNames)):
					if row[columnNames[qc_check]] == True:
						exclusion_message += columnNames[qc_check] + " + "
				self.sampleMetadata.loc[idx, 'Exclusion Details'] = exclusion_message.strip(" + ")

			self.sampleMask[idxToMask] = False

		self.Attributes['Log'].append([datetime.now(), "Dataset filtered with: filterSamples=%s, filterFeatures=%s, sampleClasses=%s, sampleRoles=%s, %s." % (
			filterSamples,
			filterFeatures,
			sampleTypes,
			assayRoles,
			', '.join("{!s}={!r}".format(key,val) for (key,val) in kwargs.items()))])


	def _exportISATAB(self, destinationPath, detailsDict):
		"""
		Export the dataset's metadata to the directory *destinationPath* as ISATAB
		detailsDict should have the format:
		detailsDict = {
		    'investigation_identifier' : "i1",
		    'investigation_title' : "Give it a title",
		    'investigation_description' : "Add a description",
		    'investigation_submission_date' : "2016-11-03",
		    'investigation_public_release_date' : "2016-11-03",
		    'first_name' : "Noureddin",
		    'last_name' : "Sadawi",
		    'affiliation' : "University",
		    'study_filename' : "my_ms_study",
		    'study_material_type' : "Serum",
		    'study_identifier' : "s1",
		    'study_title' : "Give the study a title",
		    'study_description' : "Add study description",
		    'study_submission_date' : "2016-11-03",
		    'study_public_release_date' : "2016-11-03",
		    'assay_filename' : "my_ms_assay"
		}

		:param str destinationPath: Path to a directory in which the output will be saved
		:param dict detailsDict: Contains several key, value pairs required to for ISATAB
		:raises IOError: If writing one of the files fails
		"""

		from isatools.model import Investigation, Study, Assay, OntologyAnnotation, OntologySource, Person,Publication,Protocol, Source
		from isatools.model import  Comment, Sample, Characteristic, Process, Material, DataFile, ParameterValue, plink
		from isatools import isatab
		import isaExplorer as ie

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
		    src_name = row['Sample File Name']
		    source = Source(name=src_name)

		    source.comments.append(Comment(name='Study Name', value=row['Study']))
		    study.sources.append(source)

		    sample_name = src_name
		    sample = Sample(name=sample_name, derives_from=[source])
		    # check if field exists first
		    status = row['Status'] if 'Status' in self.sampleMetadata.columns else 'N/A'
		    characteristic_material_type = Characteristic(category=OntologyAnnotation(term="material type"), value=status)
		    sample.characteristics.append(characteristic_material_type)

		    #characteristic_material_role = Characteristic(category=OntologyAnnotation(term="material role"), value=row['AssayRole'])
		    #sample.characteristics.append(characteristic_material_role)

		    # check if field exists first
		    age = row['Age'] if 'Age' in self.sampleMetadata.columns else 'N/A'
		    characteristic_age = Characteristic(category=OntologyAnnotation(term="Age"), value=age,unit='Year')
		    sample.characteristics.append(characteristic_age)
		    # check if field exists first
		    gender = row['Gender'] if 'Gender' in self.sampleMetadata.columns else 'N/A'
		    characteristic_gender = Characteristic(category=OntologyAnnotation(term="Gender"), value=gender)
		    sample.characteristics.append(characteristic_gender)

		    ncbitaxon = OntologySource(name='NCBITaxon', description="NCBI Taxonomy")
		    characteristic_organism = Characteristic(category=OntologyAnnotation(term="Organism"),value=OntologyAnnotation(term="Homo Sapiens", term_source=ncbitaxon,term_accession="http://purl.bioontology.org/ontology/NCBITAXON/9606"))
		    sample.characteristics.append(characteristic_organism)

		    study.samples.append(sample)

		    # check if field exists first
		    sampling_date = row['Sampling Date'] if not pandas.isnull(row['Sampling Date']) else None
		    sample_collection_process = Process(id_='sam_coll_proc',executes_protocol=sample_collection_protocol,date_=sampling_date)
		    aliquoting_process = Process(id_='sam_coll_proc',executes_protocol=aliquoting_protocol,date_=sampling_date)

		    sample_collection_process.inputs = [source]
		    aliquoting_process.outputs = [sample]

		    # links processes
		    plink(sample_collection_process, aliquoting_process)

		    study.process_sequence.append(sample_collection_process)
		    study.process_sequence.append(aliquoting_process)


		study.protocols.append(sample_collection_protocol)
		study.protocols.append(aliquoting_protocol)

		### Add NMR Assay ###
		nmr_assay = Assay(filename='a_'+detailsDict['assay_filename']+'.txt',measurement_type=OntologyAnnotation(term="metabolite profiling"),technology_type=OntologyAnnotation(term="NMR spectroscopy"))
		extraction_protocol = Protocol(name='extraction', protocol_type=OntologyAnnotation(term="material extraction"))

		study.protocols.append(extraction_protocol)
		nmr_protocol = Protocol(name='NMR spectroscopy', protocol_type=OntologyAnnotation(term="NMR Assay"))
		nmr_protocol.add_param('Run Order')
		#if 'Instrument' in self.sampleMetadata.columns:
		nmr_protocol.add_param('Instrument')
		#if 'Sample Batch' in self.sampleMetadata.columns:
		nmr_protocol.add_param('Sample Batch')
		nmr_protocol.add_param('Acquisition Batch')


		study.protocols.append(nmr_protocol)

		#for index, row in sampleMetadata.iterrows():
		for index, sample in enumerate(study.samples):
		    row = self.sampleMetadata.loc[self.sampleMetadata['Sample File Name'].astype(str) == sample.name]
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
		    nmr_process = Process(executes_protocol=nmr_protocol,date_=datetime.isoformat(datetime.strptime(str(row['Acquired Time'].values[0]), '%Y-%m-%d %H:%M:%S')))

		    nmr_process.name = "assay-name-{}".format(index)
		    nmr_process.inputs.append(extraction_process.outputs[0])
		    # nmr process usually has an output data file
		    # check if field exists first
		    assay_data_name = row['Assay data name'].values[0] if 'Assay data name' in self.sampleMetadata.columns else 'N/A'
		    datafile = DataFile(filename=assay_data_name, label="NMR Assay Name", generated_from=[sample])
		    nmr_process.outputs.append(datafile)

		    #nmr_process.parameter_values.append(ParameterValue(category='Run Order',value=str(i)))
		    nmr_process.parameter_values = [ParameterValue(category=nmr_protocol.get_param('Run Order'),value=row['Run Order'].values[0])]
		    # check if field exists first
		    instrument = row['Instrument'].values[0] if 'Instrument' in self.sampleMetadata.columns else 'N/A'
		    nmr_process.parameter_values.append(ParameterValue(category=nmr_protocol.get_param('Instrument'),value=instrument))
             # check if field exists first
		    sbatch = row['Sample batch'].values[0] if 'Sample batch' in self.sampleMetadata.columns else 'N/A'
		    nmr_process.parameter_values.append(ParameterValue(category=nmr_protocol.get_param('Sample Batch'),value=sbatch))
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
		    nmr_assay.technology_type = OntologyAnnotation(term="NMR spectroscopy")

		# attach the assay to the study
		study.assays.append(nmr_assay)

		if os.path.exists(os.path.join(destinationPath,'i_Investigation.txt')):
			ie.appendStudytoISA(study, destinationPath)
		else:
			isatab.dump(isa_obj=investigation, output_path=destinationPath)


	def _nmrQCChecks(self):
		"""

		Apply the quality control checks to the current dataset and update the sampleMetadata dataframe columns
		related to sample quality control.

		:return None:
		"""
		# Chemical shift calibration check
		bounds = numpy.std(self.sampleMetadata['Delta PPM']) * 3
		meanVal = numpy.mean(self.sampleMetadata['Delta PPM'])
		# QC metrics - keep the simple one here but we can remove for latter to feature summary
		self.sampleMetadata['CalibrationFail'] = ~numpy.logical_and(
			(self.sampleMetadata['Delta PPM'] > meanVal - bounds),
			(self.sampleMetadata['Delta PPM'] < meanVal + bounds))

		if 'PWFailThreshold' in self.Attributes.keys():
			# LineWidth quality check
			self.sampleMetadata['LineWidthFail'] = self.sampleMetadata['Line Width (Hz)'] >= self.Attributes[
				'PWFailThreshold']

		if 'baselineCheckRegion' in self.Attributes.keys():
			# Baseline check
			# Read attributes to derive regions
			ppmBaselineLow = tuple(self.Attributes['baselineCheckRegion'][0])
			ppmBaselineHigh = tuple(self.Attributes['baselineCheckRegion'][1])

			# Obtain the spectral regions - add sample Mask filter??here
			specsLowBaselineRegion = self.getFeatures(ppmBaselineLow)[1]
			specsHighBaselineRegion = self.getFeatures(ppmBaselineHigh)[1]

			isOutlierBaselineLow = qcCheckBaseline(specsLowBaselineRegion, self.Attributes['baseline_alpha'])
			isOutlierBaselineHigh = qcCheckBaseline(specsHighBaselineRegion, self.Attributes['baseline_alpha'])

			self.sampleMetadata['BaselineFail'] = isOutlierBaselineHigh | isOutlierBaselineLow

		if 'waterPeakCheckRegion' in self.Attributes.keys():
			# Water peak check
			ppmWaterLow = tuple(self.Attributes['waterPeakCheckRegion'][0])
			ppmWaterHigh = tuple(self.Attributes['waterPeakCheckRegion'][1])

			# Obtain the spectral regions - add sample Mask filter??here
			specsLowWaterPeakRegion = self.getFeatures(ppmWaterLow)[1]
			specsHighWaterPeakRegion = self.getFeatures(ppmWaterHigh)[1]

			isOutlierWaterPeakLow = qcCheckWaterPeak(specsLowWaterPeakRegion, self.Attributes['baseline_alpha'])
			isOutlierWaterPeakHigh = qcCheckWaterPeak(specsHighWaterPeakRegion, self.Attributes['baseline_alpha'])

			self.sampleMetadata['WaterPeakFail'] = isOutlierWaterPeakLow | isOutlierWaterPeakHigh

		return None

def main():
	pass

if __name__=='__main__':
	main()
