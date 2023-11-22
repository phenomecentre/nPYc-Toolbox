import scipy
import pandas
import numpy
import sys
import unittest
import unittest.mock
import tempfile
import os
import io
import copy
import warnings
import random
import string
import plotly
from datetime import datetime, timedelta

from _pytest.outcomes import fail

sys.path.append("..")
import nPYc
from nPYc.enumerations import VariableType, AssayRole, SampleType, QuantificationType, CalibrationMethod
from generateTestDataset import generateTestDataset


class test_drawn_plotting(unittest.TestCase):

	def setUp(self):
		# Feature1 has lowest LLOQ and ULOQ in batch1, feature2 has lowest LLOQ and ULOQ in batch2
		# On feature1 and feature2, Sample1 will be <LLOQ, Sample2 >ULOQ, Sample3 same as input
		self.targetedDataset = nPYc.TargetedDataset('', fileType='empty')
		self.targetedDataset.name = 'unittest'
		self.targetedDataset.sampleMetadata = pandas.DataFrame({'Sample File Name': ['UnitTest_targeted_file_001',
																					 'UnitTest_targeted_file_002',
																					 'UnitTest_targeted_file_003'],
																'Sample Name': ['Sample1-B1', 'Sample2-B2',
																				'Sample3-B2'],
																'Sample Type': ['Analyte', 'Analyte', 'Analyte'],
																'Acqu Date': ['10-Sep-16', '10-Sep-16', '10-Sep-16'],
																'Acqu Time': ['03:23:02', '04:52:35', '05:46:40'],
																'Vial': ['1:A,2', '1:A,3', '1:A,4'],
																'Instrument': ['XEVO-TQS#UnitTest', 'XEVO-TQS#UnitTest',
																			   'XEVO-TQS#UnitTest'],
																'Acquired Time': [datetime(2016, 9, 10, 3, 23, 2),
																				  datetime(2016, 9, 10, 4, 52, 35),
																				  datetime(2016, 9, 10, 5, 46, 40)],
																'Run Order': [0, 1, 2], 'Batch': [1, 2, 2],
																'AssayRole': [AssayRole.Assay, AssayRole.Assay,
																			  AssayRole.Assay],
																'SampleType': [SampleType.StudySample,
																			   SampleType.StudySample,
																			   SampleType.StudySample],
																'Dilution': [numpy.nan, numpy.nan, numpy.nan],
																'Correction Batch': [numpy.nan, numpy.nan, numpy.nan],
																'Subject ID': ['', '', ''], 'Sample ID': ['', '', ''],
																'Sample Base Name': ['', '', ''],
																'Exclusion Details': ['', '', '']})
		self.targetedDataset.sampleMetadata['Acquired Time'] = self.targetedDataset.sampleMetadata[
			'Acquired Time'].dt.to_pydatetime()
		self.targetedDataset.featureMetadata = pandas.DataFrame(
			{'Feature Name': ['Feature1', 'Feature2'], 'TargetLynx Feature ID': [1, 2],
			 'calibrationMethod': [CalibrationMethod.backcalculatedIS, CalibrationMethod.backcalculatedIS],
			 'quantificationType': [QuantificationType.QuantAltLabeledAnalogue,
									QuantificationType.QuantAltLabeledAnalogue], 'unitCorrectionFactor': [1., 1],
			 'Unit': ['a Unit', 'pg/uL'], 'Cpd Info': ['info cpd1', 'info cpd2'], 'LLOQ_batch1': [5., 20.],
			 'LLOQ_batch2': [20., 5.], 'ULOQ_batch1': [80., 100.], 'ULOQ_batch2': [100., 80.], 'LLOQ': [20., 20.],
			 'ULOQ': [80., 80.], 'extID1': ['F1', 'F2'], 'extID2': ['ID1', 'ID2']})
		self.targetedDataset._intensityData = numpy.array([[10., 10.], [90., 90.], [25., 70.]])
		self.targetedDataset.expectedConcentration = pandas.DataFrame(numpy.array([[40., 60.], [40., 60.], [40., 60.]]),
																	  columns=self.targetedDataset.featureMetadata[
																		  'Feature Name'].values.tolist())
		self.targetedDataset.sampleMetadataExcluded = []
		self.targetedDataset.featureMetadataExcluded = []
		self.targetedDataset.intensityDataExcluded = []
		self.targetedDataset.expectedConcentrationExcluded = []
		self.targetedDataset.excludedFlag = []
		self.targetedDataset.calibration = dict()
		self.targetedDataset.calibration['calibIntensityData'] = numpy.ndarray((0, 2))
		self.targetedDataset.calibration['calibSampleMetadata'] = pandas.DataFrame()
		self.targetedDataset.calibration['calibFeatureMetadata'] = pandas.DataFrame(index=['Feature1', 'Feature2'],
																					columns=['Feature Name'])
		self.targetedDataset.calibration['calibExpectedConcentration'] = pandas.DataFrame(
			columns=['Feature1', 'Feature2'])
		self.targetedDataset.Attributes['methodName'] = 'unittest'
		self.targetedDataset.Attributes['externalID'] = ['extID1', 'extID2']
		self.targetedDataset.VariableType = VariableType.Continuum
		self.targetedDataset.initialiseMasks()

	def test_plottic_with_plot(self):
		noSamp = numpy.random.randint(50, high=100, size=None)
		noFeat = numpy.random.randint(200, high=400, size=None)
		msData = generateTestDataset(noSamp, noFeat)

		# three dicts used for display
		stc = msData.Attributes['sampleTypeColours']
		stm = msData.Attributes['sampleTypeMarkers']
		sta = msData.Attributes['sampleTypeAbbr']

		uniq = msData.sampleMetadata["SampleClass"].unique()
		print("uniq SampleClass %s" % uniq)
		print("unique values in colourdict are %s" % stc.keys())


		# continuous fails; categorical ok

		self.assertTrue(all(k in stc.keys() for k in uniq))

		nPYc.plotting.plotTIC(msData, colourBy="SampleClass", colourType='categorical',
							  colourDict=stc, markerDict=stm, abbrDict=sta)

	def test_plotScores_with_plot(self):
		noSamp = numpy.random.randint(50, high=100, size=None)
		noFeat = numpy.random.randint(200, high=400, size=None)
		dataset = generateTestDataset(noSamp, noFeat, dtype='MSDataset')
		try:
			pcaModel = nPYc.multivariate.exploratoryAnalysisPCA(dataset)
			ns, nc = pcaModel.scores.shape
			classes = pandas.Series('Study Sample' for i in range(ns))

			stc = dataset.Attributes['sampleTypeColours']
			stm = dataset.Attributes['sampleTypeMarkers']
			#print(dataset.sampleMetadata['SampleType'].unique())
			nPYc.plotting.plotScores(pcaModel, classes=classes, colourType="categorical")
									 #colourDict=stc, markerDict=stm)
		except ValueError as ve:
			fail("Error: %s" % ve)



