import scipy
import numpy
import pandas
import os
import json
import inspect
import re
from ..enumerations import VariableType, DatasetLevel, SampleType, AssayRole
from ..utilities.generic import removeDuplicateColumns
from .._toolboxPath import toolboxPath
from datetime import datetime
import copy
from ..utilities import removeDuplicateColumns
from ..utilities import normalisation
from ..utilities.normalisation._normaliserABC import Normaliser
import warnings


class Dataset:
	"""
	Base class for nPYc dataset objects.

	:param str sop: Load configuration parameters from specified SOP JSON file
	:param sopPath: By default SOPs are loaded from the :file:`nPYc/StudyDesigns/SOP/` directory, if not ``None`` the directory specified in *sopPath=* will be searched before the builtin SOP directory.
	"""

	"""
	Default timestamp format is :rfc:`3339`
	"""

	_timestampFormat = '%Y-%m-%dT%H:%M:%S'

	def __init__(self, sop='Generic', sopPath=None, **kwargs):
		"""
		Bare constructor.
		"""
		from .. import __version__

		self._intensityData = numpy.array(None)

		self.featureMetadata = pandas.DataFrame(None, columns=['Feature Name'])
		"""
		:math:`m` × :math:`q` pandas dataframe of feature identifiers and metadata

		The featureMetadata table can include any datatype that can be placed in a pandas cell, However the toolbox assumes certain prerequisites on the following columns in order to function:

		================ ========================================= ============
		Column           dtype                                     Usage
		================ ========================================= ============
		Feature Name     str or float                              ID of the :term:`feature` measured in this column. Each 'Feature Name' must be unique in the table. If 'Feature Name' is numeric, the columns should be sorted in ascending or descending order.
		================ ========================================= ============
		"""
		self.sampleMetadata = pandas.DataFrame(None,
											   columns=['Sample ID', 'AssayRole', 'SampleType', 'Sample File Name',
														'Sample Base Name', 'Dilution', 'Batch', 'Correction Batch',
														'Acquired Time', 'Run Order', 'Exclusion Details', 'Metadata Available'])
		"""
		:math:`n` × :math:`p` dataframe of sample identifiers and metadata.

		The sampleMetadata table can include any datatype that can be placed in a pandas cell, However the toolbox assumes certain prerequisites on the following columns in order to function:

		================== ========================================= ============
		Column             dtype                                     Usage
		================== ========================================= ============
		Sample ID          str                                       ID of the :term:`sampling event` generating this sample
		AssayRole          :py:class:`~nPYc.enumerations.AssayRole`  Defines the role of this assay
		SampleType         :py:class:`~nPYc.enumerations.SampleType` Defines the type of sample acquired
		Sample File Name   str                                       :term:`Unique file name<Sample File Name>` for the analytical data
		Sample Base Name   str                                       :term:`Common identifier<Sample Base Name>` that links analytical data to the *Sample ID*
		Dilution           float                                     Where *AssayRole* is :py:attr:`~nPYc.enumerations.AssayRole.LinearityReference`, the expected abundance is indicated here
		Batch              int                                       Acquisition batch
		Correction Batch   int                                       When detecting and correcting for :term:`batch<Batch Effects>` and :term:`Run-Order<Run-Order Effects>` effects, run-order effects are characterised within samples sharing the same *Correction Batch*, while batch effects are detected between distinct values
		Acquired Time      datetime.datetime                         Date and time of acquisition of raw data
		Run order          int                                       Order of sample acquisition
		Exclusion Details  str                                       Details of reasoning if marked for exclusion
		Metadata Available bool                                      Records which samples had metadata provided with the .addSampleInfo() method
		================== ========================================= ============
		"""
		self.featureMask = numpy.array(None, dtype=bool)
		""":math:`m` element vector, with ``True`` representing features to be included in analysis, and ``False`` those to be excluded"""
		self.sampleMask = numpy.array(None, dtype=bool)
		""":math:`p` element vector, with ``True`` representing samples to be included in analysis, and ``False`` those to be excluded"""

		self.Attributes = dict()
		"""
		Dictionary of object configuration attributes, including those loaded from :doc:`SOP files<configuration/builtinSOPs>`.

		Defined attributes are as follows\:

		================ ========================================= ============
		Key              dtype                                     Usage
		================ ========================================= ============
		'dpi'            positive int                              Raster resolution when plotting figures
		'figureSize'     positive (float, float)                   Size to plot figures
		'figureFormat'   str                                       Format to save figures in
		'histBins'       positive int                              Number of bins to use when drawing histograms
		'Feature Names'  Column in :py:attr:`featureMetadata`      ID of the primary feature name
		================ ========================================= ============
		"""

		self.VariableType = None
		self.AnalyticalPlatform = None
		""":py:class:`~nPYc.enumerations.VariableType` enum specifying the type of data represented."""

		self.Attributes['Log'] = list()
		self.Attributes['Log'].append([datetime.now(), 'nPYc Toolbox version %s.' % (__version__)])
		self._loadParameters(sop, sopPath)
		self._Normalisation = normalisation.NullNormaliser()

		# Allow SOP-loaded attributes to be overriden by kwargs
		self.Attributes = {**self.Attributes, **kwargs}

		self._name = self.__class__.__name__

	@property
	def intensityData(self):
		"""
		:math:`n` × :math:`m` numpy matrix of measurements
		"""
		return self.Normalisation.normalise(self._intensityData)

	@intensityData.setter
	def intensityData(self, X: numpy.ndarray):

		self._intensityData = X

	@property
	def noSamples(self) -> int:
		"""
		:return: Number of samples in the dataset (*n*)
		:rtype: int
		"""
		try:
			(noSamples, noFeatures) = self._intensityData.shape
		except:
			noSamples = 0
		return noSamples

	@property
	def noFeatures(self) -> int:
		"""
		:return: Number of features in the dataset (*m*)
		:rtype: int
		"""
		try:
			(noSamples, noFeatures) = self._intensityData.shape
		except:
			noFeatures = 0
		return noFeatures

	@property
	def log(self) -> str:
		"""
		Return log entries as a string.
		"""
		output = ""
		for (timestamp, item) in self.Attributes['Log']:
			output = output + timestamp.strftime(self._timestampFormat)
			output = output + "\t"
			output = output + item
			output = output + "\n"

		return output

	@property
	def name(self) -> str:
		"""
		Returns or sets the name of the dataset. *name* must be a string
		"""
		return self._name

	@name.setter
	def name(self, value: str):
		"""
		Validates *value* is valid for filenames
		"""
		if not isinstance(value, str):
			raise TypeError('Name must be a string.')
		self._name = value.strip()

	@property
	def Normalisation(self):
		"""
		:py:class:`~nPYc.utilities.normalisation._normaliserABC.Normaliser` object that transforms the measurements in :py:attr:`intensityData`.
		"""
		return self._Normalisation

	@Normalisation.setter
	def Normalisation(self, normaliser):
		if not isinstance(normaliser, Normaliser):
			raise TypeError('Normalisation must implement the Normaliser ABC!')
		else:
			self._Normalisation = normaliser

	def __repr__(self):
		"""
		Customise printing of instance description.
		"""
		return "<%s instance at %s, named %s, with %d samples, %d features>" % (
		self.__class__.__name__, id(self), self.name, self.noSamples, self.noFeatures)

	def validateObject(self, verbose=True, raiseError=False, raiseWarning=True):
		"""
		Checks that all the attributes specified in the class definition are present and of the required class and/or values.
		Checks for attributes existence and type.
		Check for mandatory columns existence, but does not check the column values (type or uniqueness).
		If 'sampleMetadataExcluded', 'intensityDataExcluded', 'featureMetadataExcluded' or 'excludedFlag' exist, the existence and number of exclusions (based on 'sampleMetadataExcluded') is checked
	
		:param verbose: if True the result of each check is printed (default True)
		:type verbose: bool
		:param raiseError: if True an error is raised when a check fails and the validation is interrupted (default False)
		:type raiseError: bool
		:param raiseWarning: if True a warning is raised when a check fails
		:type raiseWarning: bool
		:return: True if the Object conforms to basic :py:class:`Dataset`
		:rtype: bool
	
		:raises TypeError: if the Object class is wrong
		:raises AttributeError: if self.Attributes does not exist
		:raises TypeError: if self.Attributes is not a dict
		:raises AttributeError: if self.Attributes['Log'] does not exist
		:raises TypeError: if self.Attributes['Log'] is not a list
		:raises AttributeError: if self.Attributes['dpi'] does not exist
		:raises TypeError: if self.Attributes['dpi'] is not an int
		:raises AttributeError: if self.Attributes['figureSize'] does not exist
		:raises TypeError: if self.Attributes['figureSize'] is not a list
		:raises ValueError: if self.Attributes['figureSize'] is not of length 2
		:raises TypeError: if self.Attributes['figureSize'][0] is not a int or float
		:raises TypeError: if self.Attributes['figureSize'][1] is not a int or float
		:raises AttributeError: if self.Attributes['figureFormat'] does not exist
		:raises TypeError: if self.Attributes['figureFormat'] is not a str
		:raises AttributeError: if self.Attributes['histBins'] does not exist
		:raises TypeError: if self.Attributes['histBins'] is not an int
		:raises AttributeError: if self.Attributes['noFiles'] does not exist
		:raises TypeError: if self.Attributes['noFiles'] is not an int
		:raises AttributeError: if self.Attributes['quantiles'] does not exist
		:raises TypeError: if self.Attributes['quantiles'] is not a list
		:raises ValueError: if self.Attributes['quantiles'] is not of length 2
		:raises TypeError: if self.Attributes['quantiles'][0] is not a int or float
		:raises TypeError: if self.Attributes['quantiles'][1] is not a int or float
		:raises AttributeError: if self.Attributes['sampleMetadataNotExported'] does not exist
		:raises TypeError: if self.Attributes['sampleMetadataNotExported'] is not a list
		:raises AttributeError: if self.Attributes['featureMetadataNotExported'] does not exist
		:raises TypeError: if self.Attributes['featureMetadataNotExported'] is not a list
		:raises AttributeError: if self.Attributes['analyticalMeasurements'] does not exist
		:raises TypeError: if self.Attributes['analyticalMeasurements'] is not a dict
		:raises AttributeError: if self.Attributes['excludeFromPlotting'] does not exist
		:raises TypeError: if self.Attributes['excludeFromPlotting'] is not a list
		:raises AttributeError: if self.VariableType does not exist
		:raises AttributeError: if self._Normalisation does not exist
		:raises TypeError: if self._Normalisation is not the Normaliser ABC
		:raises AttributeError: if self._name does not exist
		:raises TypeError: if self._name is not a str
		:raises AttributeError: if self._intensityData does not exist
		:raises TypeError: if self._intensityData is not a numpy.ndarray
		:raises AttributeError: if self.sampleMetadata does not exist
		:raises TypeError: if self.sampleMetadata is not a pandas.DataFrame
		:raises LookupError: if self.sampleMetadata does not have a Sample File Name column
		:raises LookupError: if self.sampleMetadata does not have an AssayRole column
		:raises LookupError: if self.sampleMetadata does not have a SampleType column
		:raises LookupError: if self.sampleMetadata does not have a Dilution column
		:raises LookupError: if self.sampleMetadata does not have a Batch column
		:raises LookupError: if self.sampleMetadata does not have a Correction Batch column
		:raises LookupError: if self.sampleMetadata does not have a Run Order column
		:raises LookupError: if self.sampleMetadata does not have a Sample ID column
		:raises LookupError: if self.sampleMetadata does not have a Sample Base Name column
		:raises LookupError: if self.sampleMetadata does not have an Acquired Time column
		:raises LookupError: if self.sampleMetadata does not have an Exclusion Details column
		:raises AttributeError: if self.featureMetadata does not exist
		:raises TypeError: if self.featureMetadata is not a pandas.DataFrame
		:raises LookupError: if self.featureMetadata does not have a Feature Name column
		:raises AttributeError: if self.sampleMask does not exist
		:raises TypeError: if self.sampleMask is not a numpy.ndarray
		:raises ValueError: if self.sampleMask are not bool
		:raises AttributeError: if self.featureMask does not exist
		:raises TypeError: if self.featureMask is not a numpy.ndarray
		:raises ValueError: if self.featureMask are not bool
		:raises AttributeError: if self.sampleMetadataExcluded does not exist
		:raises TypeError: if self.sampleMetadataExcluded is not a list
		:raises AttributeError: if self.intensityDataExcluded does not exist
		:raises TypeError: if self.intensityDataExcluded is not a list
		:raises ValueError: if self.intensityDataExcluded does not have the same number of exclusions as self.sampleMetadataExcluded
		:raises AttributeError: if self.featureMetadataExcluded does not exist
		:raises TypeError: if self.featureMetadataExcluded is not a list
		:raises ValueError: if self.featureMetadataExcluded does not have the same number of exclusions as self.sampleMetadataExcluded
		:raises AttributeError: if self.excludedFlag does not exist
		:raises TypeError: if self.excludedFlag is not a list
		:raises ValueError: if self.excludedFlag does not have the same number of exclusions as self.sampleMetadataExcluded
		"""

		def conditionTest(successCond, successMsg, failureMsg, allFailures, verb, raiseErr, raiseWarn, exception):
			if not successCond:
				allFailures.append(failureMsg)
				msg = failureMsg
				if raiseWarn:
					warnings.warn(msg)
				if raiseErr:
					raise exception
			else:
				msg = successMsg
			if verb:
				print(msg)
			return (allFailures)

		## init
		failureList = []
		# reference number of exclusions in list, from sampleMetadataExcluded
		refNumExcluded = None

		## Check object class
		condition = isinstance(self, Dataset)
		success = 'Check Object class:\tOK'
		failure = 'Check Object class:\tFailure, not Dataset, but ' + str(type(self))
		failureList = conditionTest(condition, success, failure, failureList, verbose, raiseError, raiseWarning,
									exception=TypeError(failure))

		## self.Attributes
		# exist
		condition = hasattr(self, 'Attributes')
		success = 'Check self.Attributes exists:\tOK'
		failure = 'Check self.Attributes exists:\tFailure, no attribute \'self.Attributes\''
		failureList = conditionTest(condition, success, failure, failureList, verbose, raiseError, raiseWarning,
									exception=AttributeError(failure))
		if condition:
			# is a dict
			condition = isinstance(self.Attributes, dict)
			success = 'Check self.Attributes is a dict:\tOK'
			failure = 'Check self.Attributes is a dict:\tFailure, \'self.Attributes\' is' + str(type(self.Attributes))
			failureList = conditionTest(condition, success, failure, failureList, verbose, raiseError, raiseWarning,
										exception=TypeError(failure))
			if condition:
				## self.Attributes keys
				## Log
				# exist
				condition = 'Log' in self.Attributes
				success = 'Check self.Attributes[\'Log\'] exists:\tOK'
				failure = 'Check self.Attributes[\'Log\'] exists:\tFailure, no attribute \'self.Attributes[\'Log\']\''
				failureList = conditionTest(condition, success, failure, failureList, verbose, raiseError, raiseWarning,
											exception=AttributeError(failure))
				if condition:
					# is a list
					condition = isinstance(self.Attributes['Log'], list)
					success = 'Check self.Attributes[\'Log\'] is a list:\tOK'
					failure = 'Check self.Attributes[\'Log\'] is a list:\tFailure, \'self.Attributes[\'Log\']\' is ' + str(
						type(self.Attributes['Log']))
					failureList = conditionTest(condition, success, failure, failureList, verbose, raiseError,
												raiseWarning, exception=TypeError(failure))
				# end self.Attributes['Log']
				## dpi
				# exist
				condition = 'dpi' in self.Attributes
				success = 'Check self.Attributes[\'dpi\'] exists:\tOK'
				failure = 'Check self.Attributes[\'dpi\'] exists:\tFailure, no attribute \'self.Attributes[\'dpi\']\''
				failureList = conditionTest(condition, success, failure, failureList, verbose, raiseError, raiseWarning,
											exception=AttributeError(failure))
				if condition:
					# is an int
					condition = isinstance(self.Attributes['dpi'], (int, numpy.integer))
					success = 'Check self.Attributes[\'dpi\'] is an int:\tOK'
					failure = 'Check self.Attributes[\'dpi\'] is an int:\tFailure, \'self.Attributes[\'dpi\']\' is ' + str(
						type(self.Attributes['dpi']))
					failureList = conditionTest(condition, success, failure, failureList, verbose, raiseError,
												raiseWarning, exception=TypeError(failure))
				# end self.Attributes['dpi']
				## figureSize
				# exist
				condition = 'figureSize' in self.Attributes
				success = 'Check self.Attributes[\'figureSize\'] exists:\tOK'
				failure = 'Check self.Attributes[\'figureSize\'] exists:\tFailure, no attribute \'self.Attributes[\'figureSize\']\''
				failureList = conditionTest(condition, success, failure, failureList, verbose, raiseError, raiseWarning,
											exception=AttributeError(failure))
				if condition:
					# is a list
					condition = isinstance(self.Attributes['figureSize'], list)
					success = 'Check self.Attributes[\'figureSize\'] is a list:\tOK'
					failure = 'Check self.Attributes[\'figureSize\'] is a list:\tFailure, \'self.Attributes[\'figureSize\']\' is ' + str(
						type(self.Attributes['figureSize']))
					failureList = conditionTest(condition, success, failure, failureList, verbose, raiseError,
												raiseWarning, exception=TypeError(failure))
					if condition:
						# is of length 2
						condition = (len(self.Attributes['figureSize']) == 2)
						success = 'Check self.Attributes[\'figureSize\'] is of length 2:\tOK'
						failure = 'Check self.Attributes[\'figureSize\'] is of length 2:\tFailure, \'self.Attributes[\'figureSize\']\' is of length ' + str(
							len(self.Attributes['figureSize']))
						failureList = conditionTest(condition, success, failure, failureList, verbose, raiseError,
													raiseWarning, exception=ValueError(failure))
						if condition:
							# figureSize[] are int
							for i in range(2):
								condition = isinstance(self.Attributes['figureSize'][i],
													   (int, float, numpy.integer, numpy.floating))
								success = 'Check self.Attributes[\'figureSize\'][' + str(i) + '] is int or float:\tOK'
								failure = 'Check self.Attributes[\'figureSize\'][' + str(
									i) + '] is int or float:\tFailure, \'self.Attributes[\'figureSize\'][' + str(
									i) + '] is ' + str(type(self.Attributes['figureSize'][i]))
								failureList = conditionTest(condition, success, failure, failureList, verbose,
															raiseError, raiseWarning, exception=TypeError(failure))
					# end self.Attributes['figureSize'] length 2
				# end self.Attributes['figureSize] list
				# end self.Attributes['figureSize']
				## figureFormat
				# exist
				condition = 'figureFormat' in self.Attributes
				success = 'Check self.Attributes[\'figureFormat\'] exists:\tOK'
				failure = 'Check self.Attributes[\'figureFormat\'] exists:\tFailure, no attribute \'self.Attributes[\'figureFormat\']\''
				failureList = conditionTest(condition, success, failure, failureList, verbose, raiseError, raiseWarning,
											exception=AttributeError(failure))
				if condition:
					# is a str
					condition = isinstance(self.Attributes['figureFormat'], str)
					success = 'Check self.Attributes[\'figureFormat\'] is a str:\tOK'
					failure = 'Check self.Attributes[\'figureFormat\'] is a str:\tFailure, \'self.Attributes[\'figureFormat\']\' is ' + str(
						type(self.Attributes['figureFormat']))
					failureList = conditionTest(condition, success, failure, failureList, verbose, raiseError,
												raiseWarning, exception=TypeError(failure))
				# end self.Attributes['figureFormat']
				## histBins
				# exist
				condition = 'histBins' in self.Attributes
				success = 'Check self.Attributes[\'histBins\'] exists:\tOK'
				failure = 'Check self.Attributes[\'histBins\'] exists:\tFailure, no attribute \'self.Attributes[\'histBins\']\''
				failureList = conditionTest(condition, success, failure, failureList, verbose, raiseError, raiseWarning,
											exception=AttributeError(failure))
				if condition:
					# is an int
					condition = isinstance(self.Attributes['histBins'], (int, numpy.integer))
					success = 'Check self.Attributes[\'histBins\'] is an int:\tOK'
					failure = 'Check self.Attributes[\'histBins\'] is an int:\tFailure, \'self.Attributes[\'histBins\']\' is ' + str(
						type(self.Attributes['histBins']))
					failureList = conditionTest(condition, success, failure, failureList, verbose, raiseError,
												raiseWarning, exception=TypeError(failure))
				# end self.Attributes['histBins']
				## noFiles
				# exist
				condition = 'noFiles' in self.Attributes
				success = 'Check self.Attributes[\'noFiles\'] exists:\tOK'
				failure = 'Check self.Attributes[\'noFiles\'] exists:\tFailure, no attribute \'self.Attributes[\'noFiles\']\''
				failureList = conditionTest(condition, success, failure, failureList, verbose, raiseError, raiseWarning,
											exception=AttributeError(failure))
				if condition:
					# is an int
					condition = isinstance(self.Attributes['noFiles'], (int, numpy.integer))
					success = 'Check self.Attributes[\'noFiles\'] is an int:\tOK'
					failure = 'Check self.Attributes[\'noFiles\'] is an int:\tFailure, \'self.Attributes[\'noFiles\']\' is ' + str(
						type(self.Attributes['noFiles']))
					failureList = conditionTest(condition, success, failure, failureList, verbose, raiseError,
												raiseWarning, exception=TypeError(failure))
				# end self.Attributes['noFiles']
				## quantiles
				# exist
				condition = 'quantiles' in self.Attributes
				success = 'Check self.Attributes[\'quantiles\'] exists:\tOK'
				failure = 'Check self.Attributes[\'quantiles\'] exists:\tFailure, no attribute \'self.Attributes[\'quantiles\']\''
				failureList = conditionTest(condition, success, failure, failureList, verbose, raiseError, raiseWarning,
											exception=AttributeError(failure))
				if condition:
					# is a list
					condition = isinstance(self.Attributes['quantiles'], list)
					success = 'Check self.Attributes[\'quantiles\'] is a list:\tOK'
					failure = 'Check self.Attributes[\'quantiles\'] is a list:\tFailure, \'self.Attributes[\'quantiles\']\' is ' + str(
						type(self.Attributes['quantiles']))
					failureList = conditionTest(condition, success, failure, failureList, verbose, raiseError,
												raiseWarning, exception=TypeError(failure))
					if condition:
						# is of length 2
						condition = (len(self.Attributes['quantiles']) == 2)
						success = 'Check self.Attributes[\'quantiles\'] is of length 2:\tOK'
						failure = 'Check self.Attributes[\'quantiles\'] is of length 2:\tFailure, \'self.Attributes[\'quantiles\']\' is of length ' + str(
							len(self.Attributes['quantiles']))
						failureList = conditionTest(condition, success, failure, failureList, verbose, raiseError,
													raiseWarning, exception=ValueError(failure))
						if condition:
							# quantiles[] are int
							for i in range(2):
								condition = isinstance(self.Attributes['quantiles'][i],
													   (int, float, numpy.integer, numpy.floating))
								success = 'Check self.Attributes[\'quantiles\'][' + str(i) + '] is int or float:\tOK'
								failure = 'Check self.Attributes[\'quantiles\'][' + str(
									i) + '] is int or float:\tFailure, \'self.Attributes[\'quantiles\'][' + str(
									i) + '] is ' + str(type(self.Attributes['quantiles'][i]))
								failureList = conditionTest(condition, success, failure, failureList, verbose,
															raiseError, raiseWarning, exception=TypeError(failure))
					# end self.Attributes['quantiles'] length 2
				# end self.Attributes['quantiles'] list
				# end self.Attributes['quantiles']
				## sampleMetadataNotExported
				# exist
				condition = 'sampleMetadataNotExported' in self.Attributes
				success = 'Check self.Attributes[\'sampleMetadataNotExported\'] exists:\tOK'
				failure = 'Check self.Attributes[\'sampleMetadataNotExported\'] exists:\tFailure, no attribute \'self.Attributes[\'sampleMetadataNotExported\']\''
				failureList = conditionTest(condition, success, failure, failureList, verbose, raiseError, raiseWarning,
											exception=AttributeError(failure))
				if condition:
					# is a list
					condition = isinstance(self.Attributes['sampleMetadataNotExported'], list)
					success = 'Check self.Attributes[\'sampleMetadataNotExported\'] is a list:\tOK'
					failure = 'Check self.Attributes[\'sampleMetadataNotExported\'] is a list:\tFailure, \'self.Attributes[\'sampleMetadataNotExported\']\' is ' + str(
						type(self.Attributes['sampleMetadataNotExported']))
					failureList = conditionTest(condition, success, failure, failureList, verbose, raiseError,
												raiseWarning, exception=TypeError(failure))
				# end self.Attributes['sampleMetadataNotExported']
				## featureMetadataNotExported
				# exist
				condition = 'featureMetadataNotExported' in self.Attributes
				success = 'Check self.Attributes[\'featureMetadataNotExported\'] exists:\tOK'
				failure = 'Check self.Attributes[\'featureMetadataNotExported\'] exists:\tFailure, no attribute \'self.Attributes[\'featureMetadataNotExported\']\''
				failureList = conditionTest(condition, success, failure, failureList, verbose, raiseError, raiseWarning,
											exception=AttributeError(failure))
				if condition:
					# is a list
					condition = isinstance(self.Attributes['featureMetadataNotExported'], list)
					success = 'Check self.Attributes[\'featureMetadataNotExported\'] is a list:\tOK'
					failure = 'Check self.Attributes[\'featureMetadataNotExported\'] is a list:\tFailure, \'self.Attributes[\'featureMetadataNotExported\']\' is ' + str(
						type(self.Attributes['featureMetadataNotExported']))
					failureList = conditionTest(condition, success, failure, failureList, verbose, raiseError,
												raiseWarning, exception=TypeError(failure))
				# end self.Attributes['featureMetadataNotExported']
				## analyticalMeasurements
				# exist
				condition = 'analyticalMeasurements' in self.Attributes
				success = 'Check self.Attributes[\'analyticalMeasurements\'] exists:\tOK'
				failure = 'Check self.Attributes[\'analyticalMeasurements\'] exists:\tFailure, no attribute \'self.Attributes[\'analyticalMeasurements\']\''
				failureList = conditionTest(condition, success, failure, failureList, verbose, raiseError, raiseWarning,
											exception=AttributeError(failure))
				if condition:
					# is a dict
					condition = isinstance(self.Attributes['analyticalMeasurements'], dict)
					success = 'Check self.Attributes[\'analyticalMeasurements\'] is a dict:\tOK'
					failure = 'Check self.Attributes[\'analyticalMeasurements\'] is a dict:\tFailure, \'self.Attributes[\'analyticalMeasurements\']\' is ' + str(
						type(self.Attributes['analyticalMeasurements']))
					failureList = conditionTest(condition, success, failure, failureList, verbose, raiseError,
												raiseWarning, exception=TypeError(failure))
				# end self.Attributes['analyticalMeasurements']
				## excludeFromPlotting
				# exist
				condition = 'excludeFromPlotting' in self.Attributes
				success = 'Check self.Attributes[\'excludeFromPlotting\'] exists:\tOK'
				failure = 'Check self.Attributes[\'excludeFromPlotting\'] exists:\tFailure, no attribute \'self.Attributes[\'excludeFromPlotting\']\''
				failureList = conditionTest(condition, success, failure, failureList, verbose, raiseError, raiseWarning,
											exception=AttributeError(failure))
				if condition:
					# is a list
					condition = isinstance(self.Attributes['excludeFromPlotting'], list)
					success = 'Check self.Attributes[\'excludeFromPlotting\'] is a list:\tOK'
					failure = 'Check self.Attributes[\'excludeFromPlotting\'] is a list:\tFailure, \'self.Attributes[\'excludeFromPlotting\']\' is ' + str(
						type(self.Attributes['excludeFromPlotting']))
					failureList = conditionTest(condition, success, failure, failureList, verbose, raiseError,
												raiseWarning, exception=TypeError(failure))
			# end self.Attributes['excludeFromPlotting']
		# end self.Attributes dictionary
		# end self.Attributes

		## self.VariableType
		# exist
		condition = hasattr(self, 'VariableType')
		success = 'Check self.VariableType exists:\tOK'
		failure = 'Check self.VariableType exists:\tFailure, no attribute \'self.VariableType\''
		failureList = conditionTest(condition, success, failure, failureList, verbose, raiseError, raiseWarning,
									exception=AttributeError(failure))
		# end Variabletype

		## self._Normalisation
		# exist
		condition = hasattr(self, '_Normalisation')
		success = 'Check self._Normalisation exists:\tOK'
		failure = 'Check self._Normalisation exists:\tFailure, no attribute \'self._Normalisation\''
		failureList = conditionTest(condition, success, failure, failureList, verbose, raiseError, raiseWarning,
									exception=AttributeError(failure))
		if condition:
			# is Normaliser ABC
			condition = isinstance(self._Normalisation, Normaliser)
			success = 'Check self._Normalisation is Normaliser ABC:\tOK'
			failure = 'Check self._Normalisation is Normaliser ABC:\tFailure, \'self._Normalisation\' is ' + str(
				type(self._Normalisation))
			failureList = conditionTest(condition, success, failure, failureList, verbose, raiseError, raiseWarning,
										exception=TypeError(failure))
		# end self._Normalisation

		## self._name
		# exist
		condition = hasattr(self, '_name')
		success = 'Check self._name exists:\tOK'
		failure = 'Check self._name exists:\tFailure, no attribute \'self._name\''
		failureList = conditionTest(condition, success, failure, failureList, verbose, raiseError, raiseWarning,
									exception=AttributeError(failure))
		if condition:
			# is a str
			condition = isinstance(self._name, str)
			success = 'Check self._name is a str:\tOK'
			failure = 'Check self._name is a str:\tFailure, \'self._name\' is ' + str(type(self._name))
			failureList = conditionTest(condition, success, failure, failureList, verbose, raiseError, raiseWarning,
										exception=TypeError(failure))
		# end self._name

		## self._intensityData
		# exist
		condition = hasattr(self, '_intensityData')
		success = 'Check self._intensityData exists:\tOK'
		failure = 'Check self._intensityData exists:\tFailure, no attribute \'self._intensityData\''
		failureList = conditionTest(condition, success, failure, failureList, verbose, raiseError, raiseWarning,
									exception=AttributeError(failure))
		if condition:
			# is a numpy.ndarray
			condition = isinstance(self._intensityData, numpy.ndarray)
			success = 'Check self._intensityData is a numpy.ndarray:\tOK'
			failure = 'Check self._intensityData is a numpy.ndarray:\tFailure, \'self._intensityData\' is ' + str(
				type(self._intensityData))
			failureList = conditionTest(condition, success, failure, failureList, verbose, raiseError, raiseWarning,
										exception=TypeError(failure))
		# end self._intensityData numpy.ndarray
		# end self._intensityData

		## self.sampleMetadata
		# exist
		condition = hasattr(self, 'sampleMetadata')
		success = 'Check self.sampleMetadata exists:\tOK'
		failure = 'Check self.sampleMetadata exists:\tFailure, no attribute \'self.sampleMetadata\''
		failureList = conditionTest(condition, success, failure, failureList, verbose, raiseError, raiseWarning,
									exception=AttributeError(failure))
		if condition:
			# is a pandas.DataFrame
			condition = isinstance(self.sampleMetadata, pandas.DataFrame)
			success = 'Check self.sampleMetadata is a pandas.DataFrame:\tOK'
			failure = 'Check self.sampleMetadata is a pandas.DataFrame:\tFailure, \'self.sampleMetadata\' is ' + str(
				type(self.sampleMetadata))
			failureList = conditionTest(condition, success, failure, failureList, verbose, raiseError, raiseWarning,
										exception=TypeError(failure))
			if condition:
				# ['Sample File Name']
				condition = ('Sample File Name' in self.sampleMetadata.columns)
				success = 'Check self.sampleMetadata[\'Sample File Name\'] exists:\tOK'
				failure = 'Check self.sampleMetadata[\'Sample File Name\'] exists:\tFailure, \'self.sampleMetadata\' lacks a \'Sample File Name\' column'
				failureList = conditionTest(condition, success, failure, failureList, verbose, raiseError, raiseWarning,
											exception=LookupError(failure))
				# ['AssayRole']
				condition = ('AssayRole' in self.sampleMetadata.columns)
				success = 'Check self.sampleMetadata[\'AssayRole\'] exists:\tOK'
				failure = 'Check self.sampleMetadata[\'AssayRole\'] exists:\tFailure, \'self.sampleMetadata\' lacks an \'AssayRole\' column'
				failureList = conditionTest(condition, success, failure, failureList, verbose, raiseError, raiseWarning,
											exception=LookupError(failure))
				# ['SampleType']
				condition = ('SampleType' in self.sampleMetadata.columns)
				success = 'Check self.sampleMetadata[\'SampleType\'] exists:\tOK'
				failure = 'Check self.sampleMetadata[\'SampleType\'] exists:\tFailure, \'self.sampleMetadata\' lacks an \'SampleType\' column'
				failureList = conditionTest(condition, success, failure, failureList, verbose, raiseError, raiseWarning,
											exception=LookupError(failure))
				# ['Dilution']
				condition = ('Dilution' in self.sampleMetadata.columns)
				success = 'Check self.sampleMetadata[\'Dilution\'] exists:\tOK'
				failure = 'Check self.sampleMetadata[\'Dilution\'] exists:\tFailure, \'self.sampleMetadata\' lacks a \'Dilution\' column'
				failureList = conditionTest(condition, success, failure, failureList, verbose, raiseError, raiseWarning,
											exception=LookupError(failure))
				# ['Batch']
				condition = ('Batch' in self.sampleMetadata.columns)
				success = 'Check self.sampleMetadata[\'Batch\'] exists:\tOK'
				failure = 'Check self.sampleMetadata[\'Batch\'] exists:\tFailure, \'self.sampleMetadata\' lacks a \'Batch\' column'
				failureList = conditionTest(condition, success, failure, failureList, verbose, raiseError, raiseWarning,
											exception=LookupError(failure))
				# ['Correction Batch']
				condition = ('Correction Batch' in self.sampleMetadata.columns)
				success = 'Check self.sampleMetadata[\'Correction Batch\'] exists:\tOK'
				failure = 'Check self.sampleMetadata[\'Correction Batch\'] exists:\tFailure, \'self.sampleMetadata\' lacks a \'Correction Batch\' column'
				failureList = conditionTest(condition, success, failure, failureList, verbose, raiseError, raiseWarning,
											exception=LookupError(failure))
				# ['Run Order']
				condition = ('Run Order' in self.sampleMetadata.columns)
				success = 'Check self.sampleMetadata[\'Run Order\'] exists:\tOK'
				failure = 'Check self.sampleMetadata[\'Run Order\'] exists:\tFailure, \'self.sampleMetadata\' lacks a \'Run Order\' column'
				failureList = conditionTest(condition, success, failure, failureList, verbose, raiseError, raiseWarning,
											exception=LookupError(failure))
				# ['Sample ID']
				condition = ('Sample ID' in self.sampleMetadata.columns)
				success = 'Check self.sampleMetadata[\'Sample ID\'] exists:\tOK'
				failure = 'Check self.sampleMetadata[\'Sample ID\'] exists:\tFailure, \'self.sampleMetadata\' lacks a \'Sample ID\' column'
				failureList = conditionTest(condition, success, failure, failureList, verbose, raiseError, raiseWarning,
											exception=LookupError(failure))
				# ['Sample Base Name']
				condition = ('Sample Base Name' in self.sampleMetadata.columns)
				success = 'Check self.sampleMetadata[\'Sample Base Name\'] exists:\tOK'
				failure = 'Check self.sampleMetadata[\'Sample Base Name\'] exists:\tFailure, \'self.sampleMetadata\' lacks a \'Sample Base Name\' column'
				failureList = conditionTest(condition, success, failure, failureList, verbose, raiseError, raiseWarning,
											exception=LookupError(failure))
				# ['Acquired Time']
				condition = ('Acquired Time' in self.sampleMetadata.columns)
				success = 'Check self.sampleMetadata[\'Acquired Time\'] exists:\tOK'
				failure = 'Check self.sampleMetadata[\'Acquired Time\'] exists:\tFailure, \'self.sampleMetadata\' lacks a \'Acquired Time\' column'
				failureList = conditionTest(condition, success, failure, failureList, verbose, raiseError, raiseWarning,
											exception=LookupError(failure))
				# ['Exclusion Details']
				condition = ('Exclusion Details' in self.sampleMetadata.columns)
				success = 'Check self.sampleMetadata[\'Exclusion Details\'] exists:\tOK'
				failure = 'Check self.sampleMetadata[\'Exclusion Details\'] exists:\tFailure, \'self.sampleMetadata\' lacks a \'Exclusion Details\' column'
				failureList = conditionTest(condition, success, failure, failureList, verbose, raiseError, raiseWarning,
											exception=LookupError(failure))
		# end self.sampleMetadata pandas.DataFrame
		# end self.sampleMetadata

		## self.featureMetadata
		# exist
		condition = hasattr(self, 'featureMetadata')
		success = 'Check self.featureMetadata exists:\tOK'
		failure = 'Check self.featureMetadata exists:\tFailure, no attribute \'self.featureMetadata\''
		failureList = conditionTest(condition, success, failure, failureList, verbose, raiseError, raiseWarning,
									exception=AttributeError(failure))
		if condition:
			# is a pandas.DataFrame
			condition = isinstance(self.featureMetadata, pandas.DataFrame)
			success = 'Check self.featureMetadata is a pandas.DataFrame:\tOK'
			failure = 'Check self.featureMetadata is a pandas.DataFrame:\tFailure, \'self.featureMetadata\' is ' + str(
				type(self.featureMetadata))
			failureList = conditionTest(condition, success, failure, failureList, verbose, raiseError, raiseWarning,
										exception=TypeError(failure))
			if condition:
				# ['Feature Name']
				condition = ('Feature Name' in self.featureMetadata.columns)
				success = 'Check self.featureMetadata[\'Feature Name\'] exists:\tOK'
				failure = 'Check self.featureMetadata[\'Feature Name\'] exists:\tFailure, \'self.featureMetadata\' lacks a \'Feature Name\' column'
				failureList = conditionTest(condition, success, failure, failureList, verbose, raiseError, raiseWarning,
											exception=LookupError(failure))
			# end self.featureMetadata['Feature Name']
		# end self.featureMetadata pandas.DataFrame
		# end self.featureMetadata

		## self.sampleMask
		# exist
		condition = hasattr(self, 'sampleMask')
		success = 'Check self.sampleMask exists:\tOK'
		failure = 'Check self.sampleMask exists:\tFailure, no attribute \'self.sampleMask\''
		failureList = conditionTest(condition, success, failure, failureList, verbose, raiseError, raiseWarning,
									exception=AttributeError(failure))
		if condition:
			# is a numpy.ndarray
			condition = isinstance(self.sampleMask, numpy.ndarray)
			success = 'Check self.sampleMask is a numpy.ndarray:\tOK'
			failure = 'Check self.sampleMask is a numpy.ndarray:\tFailure, \'self.sampleMask\' is ' + str(
				type(self.sampleMask))
			failureList = conditionTest(condition, success, failure, failureList, verbose, raiseError, raiseWarning,
										exception=TypeError(failure))
			if condition:
				# if (self.sampleMask.all() != numpy.array(False, dtype=bool)):
				# self.sampleMask is bool
				condition = (self.sampleMask.dtype == numpy.dtype(bool))
				success = 'Check self.sampleMask is bool:\tOK'
				failure = 'Check self.sampleMask is bool:\tFailure, \'self.sampleMask\' is ' + str(
					self.sampleMask.dtype)
				failureList = conditionTest(condition, success, failure, failureList, verbose, raiseError, raiseWarning,
											exception=ValueError(failure))
		# end self.samplemask numpy.ndarray
		## end self.sampleMask

		## self.featureMask
		# exist
		condition = hasattr(self, 'featureMask')
		success = 'Check self.featureMask exists:\tOK'
		failure = 'Check self.featureMask exists:\tFailure, no attribute \'self.featureMask\''
		failureList = conditionTest(condition, success, failure, failureList, verbose, raiseError, raiseWarning,
									exception=AttributeError(failure))
		if condition:
			# is a numpy.ndarray
			condition = isinstance(self.featureMask, numpy.ndarray)
			success = 'Check self.featureMask is a numpy.ndarray:\tOK'
			failure = 'Check self.featureMask is a numpy.ndarray:\tFailure, \'self.featureMask\' is ' + str(
				type(self.featureMask))
			failureList = conditionTest(condition, success, failure, failureList, verbose, raiseError, raiseWarning,
										exception=TypeError(failure))
			if condition:
				# if (self.featureMask.all() != numpy.array(False, dtype=bool)):
				# self.featureMask is bool
				condition = (self.featureMask.dtype == numpy.dtype(bool))
				success = 'Check self.featureMask is bool:\tOK'
				failure = 'Check self.featureMask is bool:\tFailure, \'self.featureMask\' is ' + str(
					self.featureMask.dtype)
				failureList = conditionTest(condition, success, failure, failureList, verbose, raiseError, raiseWarning,
											exception=ValueError(failure))
		# end self.featureMask numpy.ndarray
		## end self.featureMask

		## Exclusion data
		# If any exclusion exists
		if (hasattr(self, 'sampleMetadataExcluded') | hasattr(self, 'intensityDataExcluded') | hasattr(self,
																									   'featureMetadataExcluded') | hasattr(
				self, 'excludedFlag')):
			if verbose:
				print('---- exclusion lists found, check exclusions ----')
			## sampleMetadataExcluded
			# exist
			condition = hasattr(self, 'sampleMetadataExcluded')
			success = 'Check self.sampleMetadataExcluded exists:\tOK'
			failure = 'Check self.sampleMetadataExcluded exists:\tFailure, no attribute \'self.sampleMetadataExcluded\''
			failureList = conditionTest(condition, success, failure, failureList, verbose, raiseError, raiseWarning,
										exception=AttributeError(failure))
			if condition:
				# is a list
				condition = isinstance(self.sampleMetadataExcluded, list)
				success = 'Check self.sampleMetadataExcluded is a list:\tOK'
				failure = 'Check self.sampleMetadataExcluded is a list:\tFailure, \'self.sampleMetadataExcluded\' is ' + str(
					type(self.sampleMetadataExcluded))
				failureList = conditionTest(condition, success, failure, failureList, verbose, raiseError, raiseWarning,
											exception=TypeError(failure))
				if condition:
					# Use sampleMetadataExcluded as reference number of exclusions
					refNumExcluded = len(self.sampleMetadataExcluded)
					if verbose:
						print('---- self.sampleMetadataExcluded used as reference number of exclusions ----')
						print('\t' + str(refNumExcluded) + ' exclusions')
			# end sampleMetadataExcluded is a list
			# end sampleMetadataExcluded
			## intensityDataExcluded
			# exist
			condition = hasattr(self, 'intensityDataExcluded')
			success = 'Check self.intensityDataExcluded exists:\tOK'
			failure = 'Check self.intensityDataExcluded exists:\tFailure, no attribute \'self.intensityDataExcluded\''
			failureList = conditionTest(condition, success, failure, failureList, verbose, raiseError, raiseWarning,
										exception=AttributeError(failure))
			if condition:
				# is a list
				condition = isinstance(self.intensityDataExcluded, list)
				success = 'Check self.intensityDataExcluded is a list:\tOK'
				failure = 'Check self.intensityDataExcluded is a list:\tFailure, \'self.intensityDataExcluded\' is ' + str(
					type(self.intensityDataExcluded))
				failureList = conditionTest(condition, success, failure, failureList, verbose, raiseError, raiseWarning,
											exception=TypeError(failure))
				if condition:
					# number of exclusions
					condition = (len(self.intensityDataExcluded) == refNumExcluded)
					success = 'Check self.intensityDataExcluded number of exclusions:\tOK'
					failure = 'Check self.intensityDataExcluded number of exclusions:\tFailure, \'self.intensityDataExcluded\' has ' + str(
						len(self.intensityDataExcluded)) + ' exclusions, ' + str(refNumExcluded) + ' expected'
					failureList = conditionTest(condition, success, failure, failureList, verbose, raiseError,
												raiseWarning, exception=ValueError(failure))
			# end intensityDataExcluded is a list
			# end intensityDataExclude
			## featureMetadataExcluded
			# exist
			condition = hasattr(self, 'featureMetadataExcluded')
			success = 'Check self.featureMetadataExcluded exists:\tOK'
			failure = 'Check self.featureMetadataExcluded exists:\tFailure, no attribute \'self.featureMetadataExcluded\''
			failureList = conditionTest(condition, success, failure, failureList, verbose, raiseError, raiseWarning,
										exception=AttributeError(failure))
			if condition:
				# is a list
				condition = isinstance(self.featureMetadataExcluded, list)
				success = 'Check self.featureMetadataExcluded is a list:\tOK'
				failure = 'Check self.featureMetadataExcluded is a list:\tFailure, \'self.featureMetadataExcluded\' is ' + str(
					type(self.featureMetadataExcluded))
				failureList = conditionTest(condition, success, failure, failureList, verbose, raiseError, raiseWarning,
											exception=TypeError(failure))
				if condition:
					# number of exclusions
					condition = (len(self.featureMetadataExcluded) == refNumExcluded)
					success = 'Check self.featureMetadataExcluded number of exclusions:\tOK'
					failure = 'Check self.featureMetadataExcluded number of exclusions:\tFailure, \'self.featureMetadataExcluded\' has ' + str(
						len(self.featureMetadataExcluded)) + ' exclusions, ' + str(refNumExcluded) + ' expected'
					failureList = conditionTest(condition, success, failure, failureList, verbose, raiseError,
												raiseWarning, exception=ValueError(failure))
			# end featureMetadataExcluded is a list
			# end featureMetadataExcluded
			## excludedFlag
			# exist
			condition = hasattr(self, 'excludedFlag')
			success = 'Check self.excludedFlag exists:\tOK'
			failure = 'Check self.excludedFlag exists:\tFailure, no attribute \'self.excludedFlag\''
			failureList = conditionTest(condition, success, failure, failureList, verbose, raiseError, raiseWarning,
										exception=AttributeError(failure))
			if condition:
				# is a list
				condition = isinstance(self.excludedFlag, list)
				success = 'Check self.excludedFlag is a list:\tOK'
				failure = 'Check self.excludedFlag is a list:\tFailure, \'self.excludedFlag\' is ' + str(
					type(self.excludedFlag))
				failureList = conditionTest(condition, success, failure, failureList, verbose, raiseError, raiseWarning,
											exception=TypeError(failure))
				if condition:
					# number of exclusions
					condition = (len(self.excludedFlag) == refNumExcluded)
					success = 'Check self.excludedFlag number of exclusions:\tOK'
					failure = 'Check self.excludedFlag number of exclusions:\tFailure, \'self.excludedFlag\' has ' + str(
						len(self.excludedFlag)) + ' exclusions, ' + str(refNumExcluded) + ' expected'
					failureList = conditionTest(condition, success, failure, failureList, verbose, raiseError,
												raiseWarning, exception=ValueError(failure))
			# end excludedFlag is a list
		# end excludedFlag
		# end exclusions are present
		else:
			if verbose:
				print('---- no exclusion lists found, no check ----')
		# end Exclusion Data

		## List additional attributes (print + log)
		expectedSet = set({'Attributes', 'VariableType', '_Normalisation', '_name', '_intensityData', 'sampleMetadata',
						   'featureMetadata', 'sampleMask', 'featureMask', 'sampleMetadataExcluded',
						   'intensityDataExcluded', 'featureMetadataExcluded', 'excludedFlag'})
		objectSet = set(self.__dict__.keys())
		additionalAttributes = objectSet - expectedSet
		if len(additionalAttributes) > 0:
			if verbose:
				print('--------')
				print(str(len(additionalAttributes)) + ' additional attributes in the object:')
				print('\t' + str(list(additionalAttributes)))
		else:
			if verbose:
				print('--------')
				print('No additional attributes in the object')

		## Log and final Output
		# Basic failure might compromise logging, failure of QC compromises sample meta
		if len(failureList) == 0:
			# Log
			self.Attributes['Log'].append([datetime.now(),
										   'Conforms to Dataset (0 errors), (%i samples and %i features), with %i additional attributes in the object: %s' % (
										   self.noSamples, self.noFeatures, len(additionalAttributes),
										   list(additionalAttributes))])
			# print results
			if verbose:
				print('--------')
				print('Conforms to Dataset:\t 0 errors found')
			return True

		# Try logging to something that might not have a log
		else:
			# try logging
			try:
				self.Attributes['Log'].append([datetime.now(),
											   'Failed Dataset validation, with the following %i issues: %s' % (
											   len(failureList), failureList)])
			except (AttributeError, KeyError, TypeError):
				if verbose:
					print('--------')
					print('Logging failed')
			# print results
			if verbose:
				print('--------')
				print('Does not conform to Dataset:\t %i errors found' % (len(failureList)))
			# output
			if raiseWarning:
				warnings.warn('Does not conform to Dataset:\t %i errors found' % (len(failureList)))
			return False

	def _loadParameters(self, sop, sopPath):
		"""
		Load assay parameters from JSON SOP files located in sopPath.

		SOP names should be unique (obviously), but this is not enforced. Duplicate SOP files may cause undefined behaviour.
		
		:param sop: the SOP name
		:type sop: string
		:param sopPath: the path to sop
		:type sopPath: string
		"""
		import json
		from collections import ChainMap
		from ..utilities.extractParams import buildFileList
		import re

		# Always load some generic values
		with open(os.path.join(toolboxPath(), 'StudyDesigns', 'SOP', 'Generic.json')) as data_file:
			attributes = json.load(data_file)
		self.Attributes = {**self.Attributes, **attributes}

		# But if SOP is Generic, skip
		if sop == 'Generic':
			return

		def splitext(path):
			return {os.path.splitext(os.path.basename(path))[0]: path}

		pattern = re.compile('.+?\.json$')

		builtinSOPS = os.path.join(toolboxPath(), 'StudyDesigns', 'SOP')
		sopPaths = buildFileList(builtinSOPS, pattern)

		if sopPath is not None:
			if not os.path.isdir(sopPath):
				raise ValueError("Path: %s must be a directory." % sopPath)
			sopPaths.extend(buildFileList(sopPath, pattern))

		# Remove empty entries from list
		sopPathList = [x for x in sopPaths if x != []]

		sopPaths = dict()
		for sopPATH in sopPathList:
			sopPaths.update(splitext(sopPATH))

		if not sop in sopPaths:
			raise ValueError("The SOP '%s' is not present in '%s', or '%s'." % (sop, builtinSOPS, sopPath))

		with open(sopPaths[sop]) as data_file:
			attributes = json.load(data_file)

		self.Attributes = {**self.Attributes, **attributes}

		self.Attributes['Log'].append([datetime.now(), 'SOP configuration %s loaded from %s.' % (sop, sopPaths[sop])])

	def initialiseMasks(self):
		"""
		Re-initialise :py:attr:`featureMask` and :py:attr:`sampleMask` to match the current dimensions of :py:attr:`intensityData`, and include all samples.
		"""

		self.featureMask = numpy.squeeze(numpy.ones([self.noFeatures, 1], dtype=bool), axis=1)
		"""*m* element vector, with ``True`` representing features to be included in analysis, and ``False`` those to be excluded"""
		self.sampleMask = numpy.squeeze(numpy.ones([self.noSamples, 1], dtype=bool), axis=1)
		"""*p* element vector, with ``True`` representing samples to be included in analysis, and ``False`` those to be excluded"""

		self.Attributes['Log'].append([datetime.now(), "Masks Initialised to True.\n"])

	def updateMasks(self, filterSamples=True, filterFeatures=True,
					sampleTypes=list(SampleType),
					assayRoles=list(AssayRole), **kwargs):
		"""
		Update :py:attr:`~Dataset.sampleMask` and :py:attr:`~Dataset.featureMask` according to parameters.

		:py:meth:`updateMasks` sets :py:attr:`~Dataset.sampleMask` or :py:attr:`~Dataset.featureMask` to ``False`` for those items failing analytical criteria.

		.. note:: To avoid reintroducing items manually excluded, this method only ever sets items to ``False``, therefore if you wish to move from more stringent criteria to a less stringent set, you will need to reset the mask to all ``True`` using :py:meth:`~Dataset.initialiseMasks`.

		:param bool filterSamples: If ``False`` don't modify sampleMask
		:param bool filterFeatures: If ``False`` don't modify featureMask
		:param sampleTypes: List of types of samples to retain
		:type sampleTypes: SampleType
		:param AssayRole sampleRoles: List of assays roles to retain
		"""

		if not isinstance(sampleTypes, list):
			raise TypeError('sampleTypes must be a list of SampleType enums')

		if not isinstance(assayRoles, list):
			raise TypeError('sampleTypes must be a list of AssayRole enums')

		if not all(isinstance(item, SampleType) for item in sampleTypes):
			raise TypeError('sampleTypes must be SampleType enums.')

		if not all(isinstance(item, AssayRole) for item in assayRoles):
			raise TypeError('assayRoles must be AssayRole enums.')

		# Feature exclusions
		if filterFeatures:
			raise NotImplementedError

		# Sample Exclusions
		if filterSamples:
			sampleMask = self.sampleMetadata['SampleType'].isin(sampleTypes)
			assayMask = self.sampleMetadata['AssayRole'].isin(assayRoles)

			sampleMask = numpy.logical_and(sampleMask, assayMask)

			self.sampleMask = numpy.logical_and(sampleMask, self.sampleMask)

		self.Attributes['Log'].append([datetime.now(),
									   "Dataset filtered with: filterSamples=%s, filterFeatures=%s, sampleClasses=%s, sampleRoles=%s, %s." % (
										   filterSamples,
										   filterFeatures,
										   sampleTypes,
										   assayRoles,
										   ', '.join("{!s}={!r}".format(key, val) for (key, val) in kwargs.items()))])

	def applyMasks(self):
		"""
		Permanently delete elements masked (those set to ``False``) in :py:attr:`sampleMask` and :py:attr:`featureMask`, from :py:attr:`featureMetadata`, :py:attr:`sampleMetadata`, and :py:attr:`intensityData`.
		"""

		# Only save to excluded if features or samples masked
		if (sum(self.sampleMask == False) > 0) | (sum(self.featureMask == False) > 0):

			# Instantiate lists if first application
			if not hasattr(self, 'sampleMetadataExcluded'):
				self.sampleMetadataExcluded = []
				self.intensityDataExcluded = []
				self.featureMetadataExcluded = []
				self.excludedFlag = []

			# Samples
			if sum(self.sampleMask) != len(self.sampleMask):

				# Account for if self.sampleMask is a pandas.series
				try:
					self.sampleMask = self.sampleMask.values
				except:
					pass

				# Save excluded samples
				self.sampleMetadataExcluded.append(self.sampleMetadata[:][self.sampleMask == False])
				self.intensityDataExcluded.append(self._intensityData[self.sampleMask == False, :])
				self.featureMetadataExcluded.append(self.featureMetadata)
				self.excludedFlag.append('Samples')

				# Delete excluded samples
				self.sampleMetadata = self.sampleMetadata.loc[self.sampleMask]
				self.sampleMetadata.reset_index(drop=True, inplace=True)
				self._intensityData = self._intensityData[self.sampleMask, :]

				if hasattr(self, 'fit'):
					self.fit = self.fit[self.sampleMask, :]

			# Features
			if sum(self.featureMask) != len(self.featureMask):

				# Save excluded features
				# Save excluded features
				self.featureMetadataExcluded.append(self.featureMetadata[:][self.featureMask == False])
				self.intensityDataExcluded.append(self._intensityData[:, self.featureMask == False])
				self.sampleMetadataExcluded.append(self.sampleMetadata)
				self.excludedFlag.append('Features')

				# Delete excluded features
				self.featureMetadata = self.featureMetadata.loc[self.featureMask]
				self.featureMetadata.reset_index(drop=True, inplace=True)
				self._intensityData = self._intensityData[:, self.featureMask]

			self.Attributes['Log'].append([datetime.now(), '%i samples and %i features removed from dataset.' % (
			sum(self.sampleMask == False), sum(self.featureMask == False))])

			# Build new masks
			self.initialiseMasks()

	def addSampleInfo(self, descriptionFormat=None, filePath=None, filetype=None, **kwargs):
		"""
		Load additional metadata and map it in to the :py:attr:`sampleMetadata` table.

		Possible options:

		* **'Basic CSV'** Joins the :py:attr:`sampleMetadata` table with the data in the ``csv`` file at *filePath=*, matching on the 'Sample File Name' column in both (see :doc:`samplemetadata`).
		* **'Filenames'** Parses sample information out of the filenames, based on the named capture groups in the regex passed in *filenamespec*
		* **'Raw Data'** Extract analytical parameters from raw data files

		:param str descriptionFormat: Format of metadata to be added
		:param str filePath: Path to the additional data to be added
		:raises NotImplementedError: if the descriptionFormat is not understood
		"""

		"""
		Extra options for internal NPC use:
		* **'NPC LIMS'** NPC LIMS files mapping files names of raw analytical data to sample IDs
		* **'NPC Subject Info'** Map subject metadata from a NPC sample manifest file (format defined in 'PCSOP.082')
		"""
		if descriptionFormat == 'Basic CSV':
			self._matchBasicCSV(filePath)
		elif descriptionFormat == 'NPC LIMS':
			self._matchDatasetToLIMS(filePath)
		elif descriptionFormat == 'NPC Subject Info':
			self._matchDatasetToSubjectInfo(filePath)
		elif descriptionFormat == 'Raw Data':
			self._getSampleMetadataFromRawData(filePath, filetype)
		elif descriptionFormat == 'Filenames':
			self._getSampleMetadataFromFilename(kwargs['filenameSpec'])
		else:
			raise NotImplementedError

	def addFeatureInfo(self, filePath=None, descriptionFormat=None, featureId=None, **kwargs):
		"""
		Load additional metadata and map it in to the :py:attr:`featureMetadata` table.

		Possible options:

		* **'Reference Ranges'** JSON file specifying upper and lower reference ranges for a feature.

		:param str filePath: Path to the additional data to be added
		:param str descriptionFormat:
		:param str featureId: Unique feature Id field in the metadata file provided to match with internal Feature Name
		:raises NotImplementedError: if the descriptionFormat is not understood
		"""
		if descriptionFormat is None:
			if featureId is None:
				raise ValueError('Please provide a valid featureId')

			# Read new data and copy the current state of featureMetadata
			csvData = pandas.read_csv(filePath)

			if not any(csvData[featureId].isin(self.featureMetadata['Feature Name'])):
				raise ValueError('No matching features found in csv file provided.')
			if any(csvData[featureId].duplicated()):
				raise ValueError('Duplicated features found in csv file provided')
			currentMetadata = self.featureMetadata.copy()

			# Overwrite previously existing columns
			columnsToRemove = csvData.columns
			if 'Feature Name' in columnsToRemove:
				columnsToRemove = columnsToRemove.drop(['Feature Name'])

			for column in columnsToRemove:
				if column in currentMetadata.columns:
					currentMetadata.drop(column, axis=1, inplace=True)

			currentMetadata = currentMetadata.merge(csvData, how='left', left_on='Feature Name',
													right_on=featureId, sort=False)

			# Avoid duplicating feature ID field
			if featureId != 'Feature Name':
				currentMetadata.drop(featureId, axis=1, inplace=True)

			self.featureMetadata = currentMetadata

		elif descriptionFormat.lower() == 'reference ranges':
			from ..utilities._addReferenceRanges import addReferenceRanges
			addReferenceRanges(self.featureMetadata, filePath)

	def _matchBasicCSV(self, filePath):
		"""
		Do a basic join of the data in the csv file at filePath to the :py:attr:`sampleMetadata` dataframe on the 'Sample File Name'.
		"""

		csvData = pandas.read_csv(filePath, dtype={'Sample File Name':str, 'Sample ID': str})
		currentMetadata = self.sampleMetadata.copy()

		if 'Sample File Name' not in csvData.columns:
			raise KeyError("No 'Sample File Name' column present, unable to join tables.")

		# Check if there are any duplicates in the csv file
		u_ids, u_counts = numpy.unique(csvData['Sample File Name'], return_counts=True)
		if any(u_counts > 1):
			warnings.warn('Check and remove duplicates in CSV file')
			return

		# Store previous AssayRole and SampleType in case they were parsed using from filename:
		#
		oldAssayRole = currentMetadata['AssayRole']
		oldSampleType = currentMetadata['SampleType']
		oldDilution = currentMetadata['Dilution']
		##
		# If colums exist in both csv data and dataset.sampleMetadata remove them from sampleMetadata
		##
		columnsToRemove = csvData.columns
		columnsToRemove = columnsToRemove.drop(['Sample File Name'])

		for column in columnsToRemove:
			if column in currentMetadata.columns:
				currentMetadata.drop(column, axis=1, inplace=True)

		# If AssayRole or SampleType columns are present parse strings into enums

		csvData['AssayRole'] = [(x.replace(" ", "")).lower() if type(x) is str else numpy.nan for x in csvData['AssayRole']]
		csvData['SampleType'] = [(x.replace(" ", "")).lower() if type(x) is str else numpy.nan for x in csvData['SampleType']]

		if 'AssayRole' in csvData.columns:
			for role in AssayRole:
				csvData.loc[csvData['AssayRole'].values == (str(role).replace(" ",  "")).lower(), 'AssayRole'] = role
		if 'SampleType' in csvData.columns:
			for stype in SampleType:
				csvData.loc[csvData['SampleType'].values == (str(stype).replace(" ", "")).lower(), 'SampleType'] = stype

		# If Acquired Time column is in the CSV file, reformat data to allow operations on timestamps and timedeltas,
		# which are used in some plotting functions
		if 'Acquired Time' in csvData:
			csv_datetime = pandas.to_datetime(csvData['Acquired Time'], errors='ignore')
			csv_datetime = csv_datetime.dt.strftime('%d-%b-%Y %H:%M:%S')
			csvData['Acquired Time'] = csv_datetime.apply(lambda x: datetime.strptime(x, '%d-%b-%Y %H:%M:%S')).astype('O')

		# Left join, without sort, so the intensityData matrix and the sample Masks are kept in order
		# Preserve information about sample mask alongside merge even on the case of samples missing from CSV file.

		# Is this required?? Masked field doesn't seem to be used anywhere else
		currentMetadata['Masked'] = False
		currentMetadata.loc[(self.sampleMask == False), 'Masked'] = True

		joinedTable = pandas.merge(currentMetadata, csvData, how='left', left_on='Sample File Name',
								   right_on='Sample File Name', sort=False)

		merged_samples = pandas.merge(currentMetadata, csvData, how='inner', left_on='Sample File Name',
								   right_on='Sample File Name', sort=False)

		merged_samples = merged_samples['Sample File Name']

		merged_indices = joinedTable[joinedTable['Sample File Name'].isin(merged_samples)].index

		# Samples in the CSV file but not acquired will go for sampleAbsentMetadata, for consistency with NPC Lims import
		csv_butnotacq = csvData.loc[csvData['Sample File Name'].isin(currentMetadata['Sample File Name']) == False, :]

		if csv_butnotacq.shape[0] != 0:
			sampleAbsentMetadata = csv_butnotacq.copy(deep=True)
			# Removed normalised index columns
			# Enum masks describing the data in each row
			sampleAbsentMetadata.loc[:, 'SampleType'] = SampleType.StudySample
			sampleAbsentMetadata.loc[sampleAbsentMetadata['SampleType'].str.match('StudyPool', na=False).astype(
				bool), 'SampleType'] = SampleType.StudyPool
			sampleAbsentMetadata.loc[sampleAbsentMetadata['SampleType'].str.match('ExternalReference', na=False).astype(
				bool), 'SampleType'] = SampleType.ExternalReference

			sampleAbsentMetadata.loc[:, 'AssayRole'] = AssayRole.Assay
			sampleAbsentMetadata.loc[sampleAbsentMetadata['AssayRole'].str.match('PrecisionReference', na=False).astype(
				bool), 'AssayRole'] = AssayRole.PrecisionReference
			sampleAbsentMetadata.loc[sampleAbsentMetadata['AssayRole'].str.match('LinearityReference', na=False).astype(
				bool), 'AssayRole'] = AssayRole.LinearityReference

			# Remove duplicate columns (these will be appended with _x or _y)
			cols = [c for c in sampleAbsentMetadata.columns if c[-2:] != '_y']
			sampleAbsentMetadata = sampleAbsentMetadata[cols]
			sampleAbsentMetadata.rename(columns=lambda x: x.replace('_x', ''), inplace=True)

			self.sampleAbsentMetadata = sampleAbsentMetadata

		# By default everything in the CSV has metadata available and samples mentioned there will not be masked
		# unless Include Sample field was == False
		joinedTable.loc[merged_indices, 'Metadata Available'] = True

		# Samples in the folder and processed but not mentioned in the CSV.
		acquired_butnotcsv = currentMetadata.loc[(currentMetadata['Sample File Name'].isin(csvData['Sample File Name']) == False), :]

		# Ensure that acquired but no csv only counts samples which 1 are not in CSV and 2 - also have no other kind of
		# AssayRole information provided (from parsing filenames for example)
		if acquired_butnotcsv.shape[0] != 0:

			noMetadataIndex = acquired_butnotcsv.index
			# Find samples where metadata was there previously and is not on the new CSV
			previousMetadataAvailable = currentMetadata.loc[(~oldSampleType.isnull()) & (~oldAssayRole.isnull())
															& ((currentMetadata['Sample File Name'].isin(csvData['Sample File Name']) == False)), :].index
			metadataNotAvailable = [x for x in noMetadataIndex if x not in previousMetadataAvailable]
			# Keep old AssayRoles and SampleTypes for cases not mentioned in CSV for which this information was previously
			# available
			joinedTable.loc[previousMetadataAvailable, 'AssayRole'] = oldAssayRole[previousMetadataAvailable]
			joinedTable.loc[previousMetadataAvailable, 'SampleType'] = oldSampleType[previousMetadataAvailable]
			joinedTable.loc[previousMetadataAvailable, 'Dilution'] = oldDilution[previousMetadataAvailable]
			
			#  If not in the new CSV, but previously there, keep it and don't mask
			if len(metadataNotAvailable) > 0:
				joinedTable.loc[metadataNotAvailable, 'Metadata Available'] = False
#				self.sampleMask[metadataNotAvailable] = False
#				joinedTable.loc[metadataNotAvailable, 'Exclusion Details'] = 'No Metadata in CSV'

		# 1) ACQ and in "include Sample" - drop and set mask to false
		#  Samples Not ACQ and in "include Sample" set to False - drop and ignore from the dataframe

		# Remove acquired samples where Include sample column equals false - does not remove, just masks the sample
		if 'Include Sample' in csvData.columns:
			which_to_drop = joinedTable[joinedTable['Include Sample'] == False].index
			#self.intensityData = numpy.delete(self.intensityData, which_to_drop, axis=0)
			#self.sampleMask = numpy.delete(self.sampleMask, which_to_drop)
			self.sampleMask[which_to_drop] = False
			#joinedTable.drop(which_to_drop, axis=0, inplace=True)
			joinedTable.drop('Include Sample', inplace=True, axis=1)

		previously_masked = joinedTable[joinedTable['Masked'] == True].index
		self.sampleMask[previously_masked] = False
		joinedTable.drop('Masked', inplace=True, axis=1)
		# Regenerate the dataframe index for joined table
		joinedTable.reset_index(inplace=True, drop=True)
		self.sampleMetadata = joinedTable

		# Commented out as we shouldn't need this here after removing the LIMS, but lets keep it
		# This should make it work - but its assuming the sample "NAME" is the same as File name as in LIMS.
		self.sampleMetadata['Sample Base Name'] = self.sampleMetadata['Sample File Name']

		# Ensure there is a batch column
		if 'Batch' not in self.sampleMetadata:
			self.sampleMetadata['Batch'] = 1

		self.Attributes['Log'].append([datetime.now(), 'Basic CSV matched from %s' % (filePath)])

	def _getSampleMetadataFromFilename(self, filenameSpec):
		"""
		Filename spec is not supported in the empty base class.
		"""
		raise NotImplementedError

	def _getSampleMetadataFromRawData(self, rawDataPath):
		"""
		Pull metadata out of raw experiment files.
		"""
		raise NotImplementedError

	def _matchDatasetToLIMS(self, pathToLIMSfile):
		"""
		Establish the `Sampling ID` by matching the `Sample Base Name` with the LIMS file information.

		:param str pathToLIMSfile: Path to LIMS file for map Sampling ID
		"""

		# Read in LIMS file
		self.limsFile = pandas.read_csv(pathToLIMSfile, converters={'Sample ID': str})

		if any(self.limsFile.columns.str.match('Sampling ID')) and any(self.limsFile.columns.str.match('Sample ID')):
			warnings.warn('The LIMS File contains both a Sample ID and Sampling ID Fields')

		# rename 'sample ID' to 'sampling ID' to match sampleMetadata format
		if any(self.limsFile.columns.str.match('Sampling ID')):
			self.limsFile.rename(columns={'Sampling ID': 'Sample ID'}, inplace=True)

		# Prepare data
		# Create normalised columns
		self.sampleMetadata.loc[:, 'Sample Base Name Normalised'] = self.sampleMetadata['Sample Base Name'].str.lower()
		# if is float, make it a string with 'Assay data location'
		if isinstance(self.limsFile.loc[0, 'Assay data name'], (int, float, numpy.integer, numpy.floating)):
			self.limsFile.loc[:, 'Assay data name'] = self.limsFile.loc[:, 'Assay data location'].str.cat(
				self.limsFile['Assay data name'].astype(str), sep='/')
		self.limsFile.loc[:, 'Assay data name Normalised'] = self.limsFile['Assay data name'].str.lower()

		# Match limsFile to sampleMetdata for samples with data PRESENT
		# Remove already present columns
		if 'Sampling ID' in self.sampleMetadata.columns: self.sampleMetadata.drop(['Sampling ID'], axis=1, inplace=True)
		if 'Sample ID' in self.sampleMetadata.columns: self.sampleMetadata.drop(['Sample ID'], axis=1, inplace=True)
		if 'Subject ID' in self.sampleMetadata.columns: self.sampleMetadata.drop(['Subject ID'], axis=1, inplace=True)

		merged_samples = pandas.merge(self.sampleMetadata, self.limsFile, how='inner',left_on='Sample Base Name Normalised',
									  right_on='Assay data name Normalised', sort=False)

		self.sampleMetadata = pandas.merge(self.sampleMetadata, self.limsFile, left_on='Sample Base Name Normalised',
										   right_on='Assay data name Normalised', how='left', sort=False)

		merged_samples = merged_samples['Sample File Name']

		merged_indices = self.sampleMetadata[self.sampleMetadata['Sample File Name'].isin(merged_samples)].index


		# Complete/create set of boolean columns describing the data in each row for sampleMetadata
		self.sampleMetadata.loc[:, 'Data Present'] = self.sampleMetadata['Sample File Name'].str.match('.+', na=False)
		self.sampleMetadata.loc[:, 'LIMS Present'] = self.sampleMetadata['Assay data name'].str.match('.+', na=False,
																									  case=False)
		self.sampleMetadata.loc[:, 'LIMS Marked Missing'] = self.sampleMetadata['Status'].str.match('Missing', na=False)

		# Remove duplicate columns (these will be appended with _x or _y)
		cols = [c for c in self.sampleMetadata.columns if c[-2:] != '_y']
		self.sampleMetadata = self.sampleMetadata[cols]
		self.sampleMetadata.rename(columns=lambda x: x.replace('_x', ''), inplace=True)

		# Find samples present in LIMS but not acquired
		lims_butnotacq = self.limsFile.loc[self.limsFile['Assay data name Normalised'].isin(
			self.sampleMetadata['Sample Base Name Normalised']) == False, :]

		# Removed normalised index coloumns
		self.sampleMetadata.drop(labels=['Sample Base Name Normalised', 'Assay data name Normalised'], axis=1,
								 inplace=True)
		self.limsFile.drop(labels=['Assay data name Normalised'], axis=1, inplace=True)

		# Enforce string type on matched data
		self.sampleMetadata['Assay data name'] = self.sampleMetadata['Assay data name'].astype(str)
		self.sampleMetadata['Assay data location'] = self.sampleMetadata['Assay data location'].astype(str)
		self.sampleMetadata['Sample ID'] = self.sampleMetadata['Sample ID'].astype(str)
		self.sampleMetadata['Status'] = self.sampleMetadata['Status'].astype(str)
		if hasattr(self.sampleMetadata, 'Sample batch'):
			self.sampleMetadata['Sample batch'] = self.sampleMetadata['Sample batch'].astype(str)
		if hasattr(self.sampleMetadata, 'Assay protocol'):
			self.sampleMetadata['Assay protocol'] = self.sampleMetadata['Assay protocol'].astype(str)
		if hasattr(self.sampleMetadata, 'Sample position'):
			self.sampleMetadata['Sample position'] = self.sampleMetadata['Sample position'].astype(str)

		if lims_butnotacq.shape[0] != 0:
			sampleAbsentMetadata = lims_butnotacq.copy(deep=True)

			# Enum masks describing the data in each row
			sampleAbsentMetadata.loc[:, 'SampleType'] = SampleType.StudySample
			sampleAbsentMetadata.loc[sampleAbsentMetadata['Status'].str.match('Study Reference', na=False).astype(
				bool), 'SampleType'] = SampleType.StudyPool
			sampleAbsentMetadata.loc[sampleAbsentMetadata['Status'].str.match('Long Term Reference', na=False).astype(
				bool), 'SampleType'] = SampleType.ExternalReference

			sampleAbsentMetadata.loc[:, 'AssayRole'] = AssayRole.Assay
			sampleAbsentMetadata.loc[sampleAbsentMetadata['Status'].str.match('Study Reference', na=False).astype(
				bool), 'AssayRole'] = AssayRole.PrecisionReference
			sampleAbsentMetadata.loc[sampleAbsentMetadata['Status'].str.match('Long Term Reference', na=False).astype(
				bool), 'AssayRole'] = AssayRole.PrecisionReference

			sampleAbsentMetadata.loc[:, 'LIMS Marked Missing'] = sampleAbsentMetadata['Status'].str.match('Missing',
																										  na=False).astype(
				bool)

			# Remove duplicate columns (these will be appended with _x or _y)
			cols = [c for c in sampleAbsentMetadata.columns if c[-2:] != '_y']
			sampleAbsentMetadata = sampleAbsentMetadata[cols]
			sampleAbsentMetadata.rename(columns=lambda x: x.replace('_x', ''), inplace=True)

			self.sampleAbsentMetadata = sampleAbsentMetadata

		# Rename values in Sample ID, special case for Study Pool, External Reference and Procedural Blank
		if 'SampleType' in self.sampleMetadata.columns:
			self.sampleMetadata.loc[(((self.sampleMetadata['Sample ID'] == 'nan') | (
						self.sampleMetadata['Sample ID'] == '')) & (self.sampleMetadata[
																		  'SampleType'] == SampleType.StudyPool)).tolist(), 'Sample ID'] = 'Study Pool Sample'
			self.sampleMetadata.loc[(((self.sampleMetadata['Sample ID'] == 'nan') | (
						self.sampleMetadata['Sample ID'] == '')) & (self.sampleMetadata[
																		  'SampleType'] == SampleType.ExternalReference)).tolist(), 'Sample ID'] = 'External Reference Sample'
			self.sampleMetadata.loc[(((self.sampleMetadata['Sample ID'] == 'nan') | (
						self.sampleMetadata['Sample ID'] == '')) & (self.sampleMetadata[
																		  'SampleType'] == SampleType.ProceduralBlank)).tolist(), 'Sample ID'] = 'Procedural Blank Sample'
		self.sampleMetadata.loc[(self.sampleMetadata['Sample ID'] == 'nan').tolist(), 'Sample ID'] = 'Not specified'
		self.sampleMetadata.loc[(self.sampleMetadata[
									 'Sample ID'] == '').tolist(), 'Sample ID'] = 'Present but undefined in the LIMS file'
		# Metadata Available field is set to True
		self.sampleMetadata.loc[merged_indices, 'Metadata Available'] = True

		# Log
		self.Attributes['Log'].append([datetime.now(), 'LIMS sample IDs matched from %s' % (pathToLIMSfile)])

	def _matchDatasetToSubjectInfo(self, pathToSubjectInfoFile):
		"""
		Match the Sample IDs in :py:attr:`sampleMetadata` to the subject information mapped in the sample manifest file found at *subjectInfoFile*.

		The column *Sample ID* in :py:attr:`sampleMetadata` is matched to *Sample ID* in the *Sampling Events* sheet

		:param str pathToSubjectInfoFile: path to subject information file, an Excel file with sheets 'Subject Info' and 'Sampling Events'
		"""
		self.subjectInfo = pandas.read_excel(pathToSubjectInfoFile, sheet_name='Subject Info',
											 converters={'Subject ID': str})
		cols = [c for c in self.subjectInfo.columns if c[:7] != 'Unnamed']
		self.subjectInfo = self.subjectInfo[cols]

		self.samplingEvents = pandas.read_excel(pathToSubjectInfoFile, sheet_name='Sampling Events',
												converters={'Subject ID': str, 'Sampling ID': str})
		cols = [c for c in self.samplingEvents.columns if c[:7] != 'Unnamed']
		self.samplingEvents = self.samplingEvents[cols]
		self.samplingEvents.rename(columns={'Sampling ID': 'Sample ID'}, inplace=True)

		# Create one overall samplingInfo sheet - combine subjectInfo and samplingEvents for samples present in samplingEvents
		self.samplingInfo = pandas.merge(self.samplingEvents, self.subjectInfo, left_on='Subject ID',
										 right_on='Subject ID', how='left', sort=False)

		self.samplingInfo.rename(columns={'Sampling ID': 'Sample ID'}, inplace=True)

		# Remove duplicate columns (these will be appended with _x or _y)
		self.samplingInfo = removeDuplicateColumns(self.samplingInfo)
		# Remove any rows which are just nans
		self.samplingInfo = self.samplingInfo.loc[self.samplingInfo['Sample ID'].values != 'nan', :]

		# Rename 'Sample Type' to 'Biofluid'
		if hasattr(self.samplingInfo, 'Sample Type'):
			self.samplingInfo.rename(columns={'Sample Type': 'Biofluid'}, inplace=True)

		# Check no duplicates in sampleInfo
		u_ids, u_counts = numpy.unique(self.samplingInfo['Sample ID'], return_counts=True)
		if any(u_counts > 1):
			warnings.warn('Check and remove (non-biofluid related) duplicates in sample manifest file')

		# Match subjectInfo to sampleMetadata for samples with data ABSENT (i.e., samples in sampleAbsentMetadata)
		if hasattr(self, 'sampleAbsentMetadata'):
			self.sampleAbsentMetadata = pandas.merge(self.sampleAbsentMetadata, self.samplingInfo,
													 left_on='Sample ID', right_on='Sample ID', how='left',
													 sort=False)

			# Remove duplicate columns (these will be appended with _x or _y)
			cols = [c for c in self.sampleAbsentMetadata.columns if c[-2:] != '_y']
			self.sampleAbsentMetadata = self.sampleAbsentMetadata[cols]
			self.sampleAbsentMetadata.rename(columns=lambda x: x.replace('_x', ''), inplace=True)

			self.sampleAbsentMetadata['SubjectInfoData'] = False
			self.sampleAbsentMetadata.loc[self.sampleAbsentMetadata['Subject ID'].notnull(), 'SubjectInfoData'] = True

		# Match subjectInfo to sampleMetdata for samples with data PRESENT
		self.sampleMetadata = pandas.merge(self.sampleMetadata, self.samplingInfo, left_on='Sample ID',
										   right_on='Sample ID', how='left', sort=False)

		# Remove duplicate columns (these will be appended with _x or _y)
		cols = [c for c in self.sampleMetadata.columns if c[-2:] != '_y']
		self.sampleMetadata = self.sampleMetadata[cols]
		self.sampleMetadata.rename(columns=lambda x: x.replace('_x', ''), inplace=True)

		self.sampleMetadata['SubjectInfoData'] = False
		self.sampleMetadata.loc[self.sampleMetadata['Subject ID'].notnull(), 'SubjectInfoData'] = True

		# Find samples present in sampleInfo but not in LIMS
		info_butnotlims = self.samplingInfo.loc[
						  self.samplingInfo['Sample ID'].isin(self.limsFile['Sample ID']) == False, :]

		if info_butnotlims.shape[0] != 0:
			self.subjectAbsentMetadata = info_butnotlims.copy(deep=True)

		self.Attributes['Log'].append([datetime.now(), 'Subject information matched from %s' % (pathToSubjectInfoFile)])

	def __validateColumns(self, df, assay):
		if (assay == 'NMR') and set(
				['Sample Name', 'NMR Assay Name', 'Date', 'Comment[time]', 'Parameter Value[run order]',
				 'Parameter Value[sample batch]',
				 'Parameter Value[acquisition batch]', 'Parameter Value[instrument]']).issubset(df.columns):
			return True
		elif (assay == 'MS') and set(
				['Sample Name', 'MS Assay Name', 'Date', 'Comment[time]', 'Parameter Value[run order]',
				 'Parameter Value[sample batch]',
				 'Parameter Value[instrument]']).issubset(df.columns):
			return True
		else:
			return False

	def _initialiseFromCSV(self, sampleMetadataPath):
		"""
		Initialise the object from the three csv outputs of :py:meth:`~nPYc.Dataset.exportDataset()`.

		NOTE: This function assumes that the saved dataset was well formed with all the expected columns in the metadata tables.

		:param str sampleMetadataPath: Path to the *Name_sampleMetadata.csv* table, the file names of the featureMetadata
		and intensityData tables are inferred from the provided filename.
		"""
		##
		# Determine object name and paths
		##
		(folderPath, fileName) = os.path.split(sampleMetadataPath)
		objectName = re.match('(.*?)_sampleMetadata.csv', fileName).groups()[0]

		intensityDataPath = os.path.join(folderPath, objectName + '_intensityData.csv')
		featureMetadataPath = os.path.join(folderPath, objectName + '_featureMetadata.csv')

		##
		# Load tables
		##
		intensityData = numpy.loadtxt(intensityDataPath, dtype=float, delimiter=',')

		featureMetadata = pandas.read_csv(featureMetadataPath, index_col=0)
		sampleMetadata = pandas.read_csv(sampleMetadataPath, index_col=0)

		##
		# Fix up types
		##
		featureMetadata['Feature Name'] = featureMetadata['Feature Name'].astype(str)
		sampleMetadata['Sample File Name'] = sampleMetadata['Sample File Name'].astype(str)

		sampleMetadata['Acquired Time'] = sampleMetadata['Acquired Time'].apply(pandas.to_datetime)
		sampleMetadata['Acquired Time'] = sampleMetadata['Acquired Time']

		# If AssayRole or SampleType columns are present parse strings into enums
		if 'AssayRole' in sampleMetadata.columns:
			for role in AssayRole:
				sampleMetadata.loc[sampleMetadata['AssayRole'].values == str(role), 'AssayRole'] = role
		if 'SampleType' in sampleMetadata.columns:
			for stype in SampleType:
				sampleMetadata.loc[sampleMetadata['SampleType'].values == str(stype), 'SampleType'] = stype

		return (objectName, intensityData, featureMetadata, sampleMetadata)




	def excludeSamples(self, sampleList, on='Sample File Name', message='User Excluded'):
		"""
		Sets the :py:attr:`sampleMask` for the samples listed in *sampleList* to ``False`` to mask them from the dataset.

		:param list sampleList: A list of sample IDs to be excluded
		:param str on: name of the column in :py:attr:`sampleMetadata` to match *sampleList* against, defaults to 'Sample File Name'
		:param str message: append this message to the 'Exclusion Details' field for each sample excluded, defaults to 'User Excluded'
		:return: a list of IDs passed in *sampleList* that could not be matched against the sample IDs present
		:rtype: list
		"""
		# Validate inputs
		if not on in self.sampleMetadata.keys():
			raise ValueError('%s is not a column in `sampleMetadata`' % on)
		if not isinstance(message, str):
			raise TypeError('`message` must be a string.')

		notFound = []

		if 'Exclusion Details' not in self.sampleMetadata:
			self.sampleMetadata['Exclusion Details'] = ''

		for sample in sampleList:
			if sample in self.sampleMetadata[on].unique():
				self.sampleMask[self.sampleMetadata[self.sampleMetadata[on] == sample].index] = False
				if (self.sampleMetadata.loc[self.sampleMetadata[on] == sample, 'Exclusion Details'].values in ['', None]):
					self.sampleMetadata.loc[self.sampleMetadata[on] == sample, 'Exclusion Details'] = message
				else:
					self.sampleMetadata.loc[self.sampleMetadata[on] == sample, 'Exclusion Details'] = \
					self.sampleMetadata.loc[self.sampleMetadata[on] == sample, 'Exclusion Details'] + ' AND ' + message
			else:
				# AMtched must be unique.
				notFound.append(sample)

		if any(notFound):
			return notFound

	def excludeFeatures(self, featureList, on='Feature Name', message='User Excluded'):
		"""
		Masks the features listed in *featureList* from the dataset.

		:param list featureList: A list of feature IDs to be excluded
		:param str on: name of the column in :py:attr:`featureMetadata` to match *featureList* against, defaults to 'Feature Name'
		:param str message: append this message to the 'Exclusion Details' field for each feature excluded, defaults to 'User Excluded'
		:return: A list of ID passed in *featureList* that could not be matched against the feature IDs present.
		:rtype: list
		"""

		# Validate inputs
		if not on in self.featureMetadata.keys():
			raise ValueError('%s is not a column in `featureMetadata`' % on)
		if not isinstance(message, str):
			raise TypeError('`message` must be a string.')

		notFound = []

		if 'Exclusion Details' not in self.featureMetadata:
			self.featureMetadata['Exclusion Details'] = ''

		if self.VariableType == VariableType.Discrete:
			for feature in featureList:
				if feature in self.featureMetadata[on].unique():
					self.featureMask[self.featureMetadata[self.featureMetadata[on] == feature].index] = False
					if (self.featureMetadata.loc[self.featureMetadata[on] == feature, 'Exclusion Details'].values == ''):
						self.featureMetadata.loc[self.featureMetadata[on] == feature, 'Exclusion Details'] = message
					else:
						self.featureMetadata.loc[self.featureMetadata[on] == feature, 'Exclusion Details'] = \
						self.featureMetadata.loc[
							self.featureMetadata[on] == feature, 'Exclusion Details'] + ' AND ' + message
				else:
					# AMtched must be unique.
					notFound.append(feature)

		elif self.VariableType == VariableType.Spectral:
			for chunk in featureList:
				start = min(chunk)
				stop = max(chunk)

				if start == stop:
					warnings.warn('Low (%.2f) and high (%.2f) bounds are identical, skipping region' % (start, stop))
					continue

				mask = numpy.logical_or(self.featureMetadata[on] < start,
										 self.featureMetadata[on] > stop)

				self.featureMask = numpy.logical_and(self.featureMask, mask)

				mask = numpy.logical_not(mask)
				self.featureMetadata.loc[mask, 'Exclusion Details'] = message

		else:
			raise ValueError('Unknown VariableType.')

		return notFound


	def exportDataset(self, destinationPath='.', saveFormat='CSV', withExclusions=True, escapeDelimiters=False, filterMetadata=True):
		"""
		Export dataset object in a variety of formats for import in other software, the export is named according to the :py:attr:`name` attribute of the Dataset object.

		Possible save formats are:

		* **CSV** Basic CSV output, :py:attr:`featureMetadata`, :py:attr:`sampleMetadata` and :py:attr:`intensityData` are written to three separate CSV files in *desitinationPath*
		* **UnifiedCSV** Exports :py:attr:`featureMetadata`, :py:attr:`sampleMetadata` and :py:attr:`intensityData` concatenated into a single CSV file

		:param str destinationPath: Save data into the directory specified here
		:param str format: File format for saved data, defaults to CSV.




		:param bool withExclusions: If ``True`` mask features and samples will be excluded
		:param bool escapeDelimiters: If ``True`` remove characters commonly used as delimiters in csv files from metadata
		:param bool filterMetadata: If ``True`` does not export the sampleMetadata and featureMetadata columns listed in self.Attributes['sampleMetadataNotExported'] and self.Attributes['featureMetadataNotExported']
		:raises ValueError: if *saveFormat* is not understood
		"""
		# Validate inputs
		if not isinstance(destinationPath, str):
			raise TypeError('`destinationPath` must be a string.')
		if not isinstance(withExclusions, bool):
			raise TypeError('`withExclusions` must be True or False')
		if not isinstance(filterMetadata, bool):
			raise TypeError('`filterMetadata` must be True or False')

		#  Create the fireacotry to save the data into.
		self.saveDir = destinationPath

		if not os.path.exists(self.saveDir):
			os.makedirs(self.saveDir)

		# make a deepcopy to allow .applyMasks() or filterMetadata
		exportDataset = copy.deepcopy(self)

		if withExclusions:
			exportDataset.applyMasks()

		# do not filter metadata if safe format is ISATAB
		if filterMetadata and (saveFormat in ['UnifiedCSV', 'CSV']):
			# sampleMetadata not exported
			sampleMetaColToRemove = list(set(exportDataset.sampleMetadata.columns.tolist()) & set(
				exportDataset.Attributes['sampleMetadataNotExported']))
			exportDataset.sampleMetadata.drop(sampleMetaColToRemove, axis=1, inplace=True)
			# sampleMetadata not exported
			featureMetaColToRemove = list(set(exportDataset.featureMetadata.columns.tolist()) & set(
				exportDataset.Attributes['featureMetadataNotExported']))
			exportDataset.featureMetadata.drop(featureMetaColToRemove, axis=1, inplace=True)

		if saveFormat == 'CSV':
			destinationPath = os.path.join(destinationPath, exportDataset.name)
			exportDataset._exportCSV(destinationPath, escapeDelimiters=escapeDelimiters)
		elif saveFormat == 'UnifiedCSV':
			destinationPath = os.path.join(destinationPath, exportDataset.name)
			exportDataset._exportUnifiedCSV(destinationPath, escapeDelimiters=escapeDelimiters)
		else:
			raise ValueError('Save format \'%s\' not understood.' % saveFormat)

		self.Attributes['Log'].append([datetime.now(), "%s format export made to %s\n" % (saveFormat, self.saveDir)])


	def _exportCSV(self, destinationPath, escapeDelimiters=False):
		"""
		Export the dataset to the directory *destinationPath* as a set of three CSV files:
			*destinationPath*_intensityData.csv
			*destinationPath*_sampleMetadata.csv
			*destinationPath*_featureMetadata.csv

		:param str destinationPath: Path to a directory in which the output will be saved
		:param bool escapeDelimiters: Remove characters commonly used as delimiters in csv files from metadata
		:raises IOError: If writing one of the files fails
		"""

		sampleMetadata = self.sampleMetadata.copy(deep=True)
		featureMetadata = self.featureMetadata.copy(deep=True)

		if escapeDelimiters:
			# Remove any commas from metadata/feature tables - for subsequent import of resulting csv files to other software packages

			for column in sampleMetadata.columns:
				try:
					if type(sampleMetadata[column][0]) is not datetime:
						sampleMetadata.loc[:, column] = sampleMetadata[column].str.replace(',', ';')
				except:
					pass

			for column in featureMetadata.columns:
				try:
					if type(featureMetadata[column][0]) is not datetime:
						featureMetadata.loc[:, column] = featureMetadata[column].str.replace(',', ';')
				except:
					pass

		# Export sample metadata
		sampleMetadata.to_csv(destinationPath + '_sampleMetadata.csv',
							  encoding='utf-8', date_format=self._timestampFormat)

		# Export feature metadata
		featureMetadata.to_csv(destinationPath + '_featureMetadata.csv',
							   encoding='utf-8')

		# Export intensity data
		numpy.savetxt(destinationPath + '_intensityData.csv',
					  self.intensityData, delimiter=",")



	def _exportUnifiedCSV(self, destinationPath, escapeDelimiters=True):
		"""
		Export the dataset to the directory *destinationPath* as a combined CSV file containing intensity data, and feature and sample metadata
			*destinationPath*_combinedData.csv.csv

		:param str destinationPath: Path to a directory in which the output will be saved
		:param bool escapeDelimiters: Remove characters commonly used as delimiters in csv files from metadata
		:raises IOError: If writing one of the files fails
		"""

		sampleMetadata = self.sampleMetadata.copy(deep=True)
		featureMetadata = self.featureMetadata.copy(deep=True)

		if escapeDelimiters:
			# Remove any commas from metadata/feature tables - for subsequent import of resulting csv files to other software packages

			for column in sampleMetadata.columns:
				try:
					if type(sampleMetadata[column][0]) is not datetime:
						sampleMetadata[column] = sampleMetadata[column].str.replace(',', ';')
				except:
					pass

			for column in featureMetadata.columns:
				try:
					if type(featureMetadata[column][0]) is not datetime:
						featureMetadata[column] = featureMetadata[column].str.replace(',', ';')
				except:
					pass

		# Export combined data in single file
		tmpXCombined = pandas.concat([featureMetadata.transpose(),
									  pandas.DataFrame(self.intensityData)], axis=0)

		with warnings.catch_warnings():
			# Seems no way to avoid pandas complaining here (v0.18.1)
			warnings.simplefilter("ignore")
			tmpCombined = pandas.concat([sampleMetadata, tmpXCombined], axis=1)

		# reorder rows to put metadata first
		tmpCombined = tmpCombined.reindex(tmpXCombined.index, axis=0)

		# Save
		tmpCombined.to_csv(os.path.join(destinationPath + '_combinedData.csv'),
						   encoding='utf-8', date_format=self._timestampFormat)


	def getFeatures(self, featureIDs, by=None, useMasks=True):
		"""
		Get a feature or list of features by name or ranges.

		If :py:attr:`VariableType` is :py:attr:`~nPYc.enumerations.VariableType.Discrete`, :py:meth:`getFeature` expects either a single or list of values, and matching features are returned.
		If :py:attr:`VariableType` is :py:attr:`~nPYc.enumerations.VariableType.Spectral`, pass either a single, or list of (min, max) tuples, the features returned will be a slice of the combined ranges. If the ranges passed overlap, the union will be returned.

		:param featureIDs: A single or list of feature IDs to return
		:type featureID: Same dtype as the :py:attr:`featureMetadata`\ **[by]** column
		:param by: Column in :py:attr:`featureMetadata` to search in, ``None`` use the column defined in :py:attr:`Attributes`\ ['Feature Names']
		:type by: None or str
		:returns: (featureMetadata, intensityData)
		:rtype: (pandas.Dataframe, numpy.ndarray)
		"""
		if not isinstance(featureIDs, list):
			featureIDs = [featureIDs]

		if by is None:
			by = self.Attributes['Feature Names']

		if by not in self.featureMetadata.keys():
			raise KeyError('"by": %s is not a key in featureMetadata' % (by))

		indexes = list()
		if self.VariableType == VariableType.Discrete:
			for feature in featureIDs:
				indexes.append(self.featureMetadata.loc[self.featureMetadata[by] == feature].index[0])

			if useMasks:
				indexes = [x for x in indexes if self.featureMask[x]]
				#indexes.remove(maskedVar)

			return self.featureMetadata.iloc[indexes], self.intensityData[:, indexes]

		elif self.VariableType == VariableType.Spectral:
			rangeMask = numpy.zeros_like(self.featureMask)
			for featureRange in featureIDs:
				if featureRange[0] > featureRange[1]:
					featureRange = tuple(reversed(featureRange))

				rangeMask[numpy.logical_and(self.featureMetadata[by].values >= featureRange[0],
											self.featureMetadata[by].values <= featureRange[1])] = True

			if useMasks:
				rangeMask &= self.featureMask

			return self.featureMetadata.loc[rangeMask], self.intensityData[:, rangeMask]
		else:
			raise TypeError('Dataset.VariableType type not understood!')

	def _exportHDF5(self, destinationPath):

		raise NotImplementedError


def main():
	print("Implementation of " + os.path.split(os.path.dirname(inspect.getfile(nPYc)))[1])


if __name__ == '__main__':
	pass
