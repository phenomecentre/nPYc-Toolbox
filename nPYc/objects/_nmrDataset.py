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
from ..utilities._nmr import _qcCheckBaseline, _qcCheckWaterPeak


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

			self._scale = self.featureMetadata['ppm'].values
			self.featureMask[:] = True
			# Perform the quaality control checks to populate the class
			self.__nmrQCChecks()

			self.Attributes['Log'].append([datetime.now(), 'Bruker format spectra loaded from %s' % (datapath)])

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

		self.Attributes['Log'].append([datetime.now(), 'Sample metadata parsed from filenames.'])


	def updateMasks(self, filterSamples=True, filterFeatures=True,
					sampleTypes=[SampleType.StudySample, SampleType.StudyPool],
					assayRoles=[AssayRole.Assay, AssayRole.PrecisionReference], exclusionRegions=None,
					sampleQCChecks=['ImportFail','CalibrationFail','LineWidthFail','CalibrationFail','BaselineFail','WaterPeakFail'],**kwargs):
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

			# If features are modified, retrigger
			self.__nmrQCChecks()

		# Sample Exclusions
		if filterSamples:

			# Retrigger QC checks before checking which samples need to be updated
			if not filterFeatures:
				self.__nmrQCChecks()

			super().updateMasks(filterSamples=True,
								filterFeatures=False,
								sampleTypes=sampleTypes,
								assayRoles=assayRoles,
								**kwargs)
			columnNames = ['Sample File Name'].extend()
			fail_summary = self.sampleMetadata.loc[:, ['Sample File Name', 'ImportFail', 'LineWidthFail',
														  'CalibrationFail', 'BaselineFail', 'WaterPeakFail']]

			# Check this step!
			self.sampleMask &= ~fail_summary.any()

		self.Attributes['Log'].append([datetime.now(), "Dataset filtered with: filterSamples=%s, filterFeatures=%s, sampleClasses=%s, sampleRoles=%s, %s." % (
			filterSamples,
			filterFeatures,
			sampleTypes,
			assayRoles,
			', '.join("{!s}={!r}".format(key,val) for (key,val) in kwargs.items()))])

	def _exportISATAB(self, destinationPath, escapeDelimiters=False):
		"""
		Export the dataset's metadata to the directory *destinationPath* as ISATAB

		:param str destinationPath: Path to a directory in which the output will be saved
		:param bool escapeDelimiters: Remove characters commonly used as delimiters in csv files from metadata
		:raises IOError: If writing one of the files fails
		"""

		from distutils.dir_util import copy_tree
		#import re

		#copy the blank ISATAB to destinationPath so we can populate it
		copy_tree(os.path.join(toolboxPath(),'StudyDesigns','BlankISATAB','blank-nmr'), destinationPath)

		sampleMetadata = self.sampleMetadata.copy(deep=True)
		#featureMetadata = self.featureMetadata.copy(deep=True)

		#make sure this field is of type string
		#otherwise the escapeDelimters causes it to become empty
		sampleMetadata['Acquired Time'] = sampleMetadata['Acquired Time'].astype(str)
		sampleMetadata['SampleType'] = sampleMetadata['SampleType'].astype(str)
		sampleMetadata['AssayRole'] = sampleMetadata['AssayRole'].astype(str)

		# Columns required in ISATAB
		if 'Organism' not in sampleMetadata.columns:
			sampleMetadata['Organism'] = 'N/A'
		if 'Material Type' not in sampleMetadata.columns:
			sampleMetadata['Material Type'] = 'N/A'
		#if 'Detector Unit' not in sampleMetadata.columns:
		#	sampleMetadata['Detector Unit'] = 'Volt'
		#if 'Age Unit' not in sampleMetadata.columns:
		#	sampleMetadata['Age Unit'] = 'Year'

		sampleMetadata.to_csv(os.path.join(destinationPath + 'sampleMetadata.csv'), encoding='utf-8',index=False)

		if escapeDelimiters:
			# Remove any commas from metadata/feature tables - for subsequent import of resulting csv files to other software packages

			for column in sampleMetadata.columns:
				try:
					sampleMetadata[column] = sampleMetadata[column].str.replace(',', ';')
				except:
					pass

		studyPath = os.path.join(destinationPath,'s_NPC-Test-Study.txt')
		studyDF = pandas.read_csv(studyPath, sep='\t')
		#remove all rows in case the file is not empty
		studyDF.drop(studyDF.index, inplace=True)

		#populate study using sampleMetadata
		sampleMetadata['Source Name'] = sampleMetadata['Subject ID'].fillna(sampleMetadata['Sampling ID'])
		studyDF['Source Name'] = sampleMetadata['Source Name']

		studyDF['Characteristics[material role]'] = sampleMetadata['SampleType'].apply(lambda x: x.split('.')[1])
		studyDF['Comment[study name]'] = sampleMetadata['Study']
		studyDF['Characteristics[organism]'] = sampleMetadata['Organism']
		studyDF['Characteristics[material type]'] = sampleMetadata['Material Type']
		studyDF['Characteristics[gender]'] = sampleMetadata['Gender']
		studyDF['Characteristics[age]'] = sampleMetadata['Age']
		studyDF['Unit'] = 'Year' #sampleMetadata['Age Unit']

		studyDF['Protocol REF'] = 'sample collection'
		studyDF['Date'] = sampleMetadata['Sampling Date']
		studyDF['Comment[sample name]'] = ''

		studyDF['Protocol REF.1'] = 'aliquoting'

		sampleMetadata['Sample Name'] = sampleMetadata['Sampling ID'].fillna(sampleMetadata['Subject ID'])
		studyDF['Sample Name'] = sampleMetadata['Sample Name']

		#because ISATAB has several columns with the same name, pandas auto numbers them
		#here we remove the numbering at the end of field names
		studyDF.columns = removeTrailingColumnNumbering(studyDF.columns)

		#now we remove duplicate rows from study as we don't want the same sample to be declared more than once
		#removal is based on Source Name and Sample Name to preserve aliquoting
		studyDF.drop_duplicates(subset=['Source Name', 'Sample Name'], inplace=True)

		studyDF.to_csv(studyPath,sep='\t', encoding='utf-8',index=False)

		nmrAssayPath = os.path.join(destinationPath,'a_npc-test-study_metabolite_profiling_NMR_spectroscopy.txt')
		nmrAssay = pandas.read_csv(nmrAssayPath, sep='\t')

		#remove all rows in case the file is not empty
		nmrAssay.drop(nmrAssay.index, inplace=True)

		nmrAssay['Sample Name'] = sampleMetadata['Sample Name']

		nmrAssay['Protocol REF'] = 'extraction'
		nmrAssay['Protocol REF.1'] = 'labeling'
		nmrAssay['Protocol REF.2'] = 'NMR spectrometry'
		nmrAssay['Parameter Value[instrument]'] = sampleMetadata['Instrument']
		#nmrAssay['Parameter Value[detector voltage]'] = sampleMetadata['Detector']
		#nmrAssay['Unit.1'] = 'Volt'
		nmrAssay['Protocol REF.3'] = 'nmr assay'
		nmrAssay['Date'], nmrAssay['Comment[time]'] = sampleMetadata['Acquired Time'].str.split(' ', 1).str
		nmrAssay['Parameter Value[run order]'] = sampleMetadata['Run Order']

		nmrAssay['Parameter Value[sample batch]'] = sampleMetadata['Sample batch']

		if 'Acquisition batch' in sampleMetadata.columns:#rename field if necessary
			sampleMetadata.rename(columns={'Acquisition batch': 'Batch'}, inplace=True)

		nmrAssay['Parameter Value[acquisition batch]'] = sampleMetadata['Batch']

		nmrAssay['NMR Assay Name'] = sampleMetadata['Assay data name']

		nmrAssay['Protocol REF.4'] = 'data normalization'
		nmrAssay['Protocol REF.5'] = 'data transformation'

		nmrAssay.columns = removeTrailingColumnNumbering(nmrAssay.columns)
		#print(msAssay.columns)

		nmrAssay.to_csv(nmrAssayPath,sep='\t', encoding='utf-8',index=False)

	def __nmrQCChecks(self):
		"""

		Apply the quality control checks to the current dataset and apply the sampleMetadata dataframe.

		:return None:
		"""
		# Chemical shift calibration check
		bounds = numpy.std(self.sampleMetadata['Delta PPM']) * 3
		meanVal = numpy.mean(self.sampleMetadata['Delta PPM'])
		# QC metrics - keep the simple one here but we can remove for latter to feature summary
		self.sampleMetadata['calibrationFail'] = numpy.logical_or(
			(self.sampleMetadata['Delta PPM'] < meanVal - bounds),
			(self.sampleMetadata['Delta PPM'] > meanVal + bounds))

		# LineWidth quality check
		self.sampleMetadata['LineWidthFail'] = self.sampleMetadata['Line Width (Hz)'] >= self.Attributes[
			'PWFailThreshold']

		# Baseline check
		# Read attributes to derive regions
		ppmBaselineLow = tuple(self.Attributes['baselineCheckRegion'][0])
		ppmBaselineHigh = tuple(self.Attributes['baselineCheckRegion'][1])

		# Obtain the spectral regions - add sample Mask filter??here
		specsLowBaselineRegion = self.getFeatures(ppmBaselineLow)[1]
		specsHighBaselineRegion = self.getFeatures(ppmBaselineHigh)[1]

		isOutlierBaselineLow = _qcCheckBaseline(specsLowBaselineRegion)
		isOutlierBaselineHigh = _qcCheckBaseline(specsHighBaselineRegion)

		# Water peak check
		ppmWaterLow = tuple(self.Attributes['waterPeakCheckRegion'][0])
		ppmWaterHigh = tuple(self.Attributes['waterPeakCheckRegion'][1])

		# Obtain the spectral regions - add sample Mask filter??here
		specsLowWaterPeakRegion = self.getFeatures(ppmWaterLow)[1]
		specsHighWaterPeakRegion = self.getFeatures(ppmWaterHigh)[1]

		isOutlierWaterPeakLow = _qcCheckWaterPeak(specsLowWaterPeakRegion)
		isOutlierWaterPeakHigh = _qcCheckWaterPeak(specsHighWaterPeakRegion)

		self.sampleMetadata['WaterPeakFail'] = isOutlierWaterPeakLow | isOutlierWaterPeakHigh
		self.sampleMetadata['BaselineFail'] = isOutlierBaselineHigh | isOutlierBaselineLow

		return None

def main():
	pass

if __name__=='__main__':
	main()
