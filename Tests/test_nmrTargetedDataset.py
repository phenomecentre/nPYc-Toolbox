# -*- coding: utf-8 -*-
import pandas
import numpy
import io
import sys
import unittest
import unittest.mock
import tempfile
import os
import copy
import json, sys, logging
from datetime import datetime
sys.path.append("..")
import nPYc
import warnings
from nPYc.enumerations import VariableType, AssayRole, SampleType, QuantificationType, CalibrationMethod
from pandas.testing import assert_frame_equal
from numpy.testing import assert_array_almost_equal
class test_targeteddataset_full_brukerxml_load(unittest.TestCase):
	"""
	Test all steps of loadBrukerXMLDataset and parameters input: find and read Bruker XML files, filter features by units, format sample and feature metadata, initialise expectedConcentration and calibration, apply limits of quantification.
	Underlying functions tested independently
	Test BrukerQuant-UR until BrukerBI-LISA is definitive
	"""
	def setUp(self):
		self.datapathQuantPS = os.path.join('..','..','npc-standard-project','Raw_Data','nmr','UnitTest2')
		# 49 features, 9 samples, BrukerQuant-UR
		self.datapathQuantUR = os.path.join('..', '..', 'npc-standard-project', 'Raw_Data', 'nmr', 'UnitTest1')
		# Expected TargetedDataset
		# Do not check sampleMetadata['Path']
		
		self.expectedQuantPS = dict()
					
		plasma_sm = {
			    "Sample File Name": {
			        "0": "UnitTest2_Plasma_Rack1_SLL_051218/10",
			        "1": "UnitTest2_Plasma_Rack1_SLL_051218/20"
			    },
			    "Sample Base Name": {
			        "0": "UnitTest2_Plasma_Rack1_SLL_051218/10",
			        "1": "UnitTest2_Plasma_Rack1_SLL_051218/20"
			    },
			    "expno": {
			        "0": 10,
			        "1": 20
			    },
			    "Acquired Time": {
			        "1": pandas.Timestamp("2018-12-05 11:54:31"),
			        "0": pandas.Timestamp("2018-12-05 11:32:33")
			    },
			    "Run Order": {
			        "0": 0,
			        "1": 1
			    },
			    "AssayRole": {
			        "0": numpy.nan,
			        "1": numpy.nan
			    },
			    "SampleType": {
			        "0": numpy.nan,
			        "1": numpy.nan
			    },
			    "Dilution": {
			        "0": 100,
			        "1": 100
			    },
			    "Correction Batch": {
			        "0": numpy.nan,
			        "1": numpy.nan
			    },
			    "Sample ID": {
			        "0": numpy.nan,
			        "1": numpy.nan
			    },
			    "Exclusion Details": {
			        "0": None,
			        "1": None
			    },
			    "Batch": {
			        "0": 1,
			        "1": 1
			    },
			    "Metadata Available": {
			        "0": False,
			        "1": False
			    }
			}
		
		plasma_fm =	{
		    "Feature Name": {
		        "0": "Ethanol",
		        "1": "Trimethylamine-N-oxide",
		        "2": "2-Aminobutyric acid",
		        "3": "Alanine",
		        "4": "Asparagine"
		    },
		    "comment": {
		        "0": "",
		        "1": "",
		        "2": "",
		        "3": "",
		        "4": ""
		    },
		    "LOD": {
		        "0": 0.1,
		        "1": 0.08,
		        "2": 0.05,
		        "3": 0.02,
		        "4": 0.05
		    },
		    "LLOQ": {
		        "0": numpy.nan,
		        "1": numpy.nan,
		        "2": numpy.nan,
		        "3": numpy.nan,
		        "4": numpy.nan
		    },
		    "Unit": {
		        "0": "mmol/L",
		        "1": "mmol/L",
		        "2": "mmol/L",
		        "3": "mmol/L",
		        "4": "mmol/L"
		    },
		    "lodMask": {
		        "0": False,
		        "1": False,
		        "2": False,
		        "3": True,
		        "4": False
		    },
		    "Lower Reference Percentile": {
		        "0": 2.5,
		        "1": 2.5,
		        "2": 2.5,
		        "3": 2.5,
		        "4": 2.5
		    },
		    "Upper Reference Percentile": {
		        "0": 97.5,
		        "1": 97.5,
		        "2": 97.5,
		        "3": 97.5,
		        "4": 97.5
		    },
		    "Lower Reference Value": {
		        "0": "-",
		        "1": "-",
		        "2": "-",
		        "3": 0.29,
		        "4": "-"
		    },
		    "Upper Reference Value": {
		        "0": 0.82,
		        "1": 0.08,
		        "2": 0.1,
		        "3": 0.64,
		        "4": 0.08
		    },
		    "quantificationType": {
		        "0": QuantificationType.QuantOther,
		        "1": QuantificationType.QuantOther,
		        "2": QuantificationType.QuantOther,
		        "3": QuantificationType.QuantOther,
		        "4": QuantificationType.QuantOther
		    },
		    "calibrationMethod": {
		        "0": CalibrationMethod.otherCalibration,
		        "1": CalibrationMethod.otherCalibration,
		        "2": CalibrationMethod.otherCalibration,
		        "3": CalibrationMethod.otherCalibration,
		        "4": CalibrationMethod.otherCalibration,
		    },
		    "ULOQ": {
		        "0": numpy.nan,
		        "1": numpy.nan,
		        "2": numpy.nan,
		        "3": numpy.nan,
		        "4": numpy.nan
		    }
		}

		self.expectedQuantPS["featureMetadata"]  = pandas.DataFrame.from_dict(plasma_fm)
		self.expectedQuantPS["sampleMetadata"]  = pandas.DataFrame.from_dict(plasma_sm)
		self.expectedQuantPS["intensityData"] = numpy.array([[-numpy.inf, -numpy.inf, -numpy.inf, 0.517, -numpy.inf],
														[-numpy.inf, -numpy.inf, -numpy.inf, 0.479, -numpy.inf]])
		self.expectedQuantPS['expectedConcentration'] = pandas.DataFrame(None, index=list(self.expectedQuantPS['sampleMetadata'].index), columns=self.expectedQuantPS['featureMetadata']['Feature Name'].tolist())


		self.expectedQuantUR = dict()
		self.expectedQuantUR['sampleMetadata'] = pandas.DataFrame({'Sample File Name': ['UnitTest1_Urine_Rack1_SLL_270814/10',
																				 'UnitTest1_Urine_Rack1_SLL_270814/20',
																				 'UnitTest1_Urine_Rack1_SLL_270814/30',
																				 'UnitTest1_Urine_Rack1_SLL_270814/40',
																				 'UnitTest1_Urine_Rack1_SLL_270814/50',
																				 'UnitTest1_Urine_Rack1_SLL_270814/60',
																				 'UnitTest1_Urine_Rack1_SLL_270814/70',
																				 'UnitTest1_Urine_Rack1_SLL_270814/80',
																				 'UnitTest1_Urine_Rack1_SLL_270814/90'],
															'Sample Base Name': ['UnitTest1_Urine_Rack1_SLL_270814/10',
																				 'UnitTest1_Urine_Rack1_SLL_270814/20',
																				 'UnitTest1_Urine_Rack1_SLL_270814/30',
																				 'UnitTest1_Urine_Rack1_SLL_270814/40',
																				 'UnitTest1_Urine_Rack1_SLL_270814/50',
																				 'UnitTest1_Urine_Rack1_SLL_270814/60',
																				 'UnitTest1_Urine_Rack1_SLL_270814/70',
																				 'UnitTest1_Urine_Rack1_SLL_270814/80',
																				 'UnitTest1_Urine_Rack1_SLL_270814/90'],
															'expno': [10, 20, 30, 40, 50, 60, 70, 80, 90],
															'Acquired Time': [pandas.Timestamp(datetime(2017, 8, 23, 19, 39, 1)),
																			  pandas.Timestamp(datetime(2017, 8, 23, 19, 56, 55)),
																			  pandas.Timestamp(datetime(2017, 8, 23, 20, 14, 50)),
																			  pandas.Timestamp(datetime(2017, 8, 23, 20, 32, 35)),
																			  pandas.Timestamp(datetime(2017, 8, 23, 20, 50, 9)),
																			  pandas.Timestamp(datetime(2017, 8, 23, 21, 7, 48)),
																			  pandas.Timestamp(datetime(2017, 8, 23, 21, 25, 38)),
																			  pandas.Timestamp(datetime(2017, 8, 23, 21, 42, 57)),
																			  pandas.Timestamp(datetime(2017, 8, 23, 22, 0, 53))],
															'Run Order': [0, 1, 2, 3, 4, 5, 6, 7, 8],
															'AssayRole': [numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan],
															'SampleType': [numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan],
															'Dilution': [100, 100, 100, 100, 100, 100, 100, 100, 100],
															'Correction Batch': [numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan],
															'Sample ID': [numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan],
															'Exclusion Details': [None, None, None, None, None, None, None, None, None],
															'Batch': [1, 1, 1, 1, 1, 1, 1, 1, 1]})
		self.expectedQuantUR['sampleMetadata']['Metadata Available'] = False

		self.expectedQuantUR['featureMetadata'] = pandas.DataFrame({'Feature Name': 
																		['Dimethylamine',
																		 'Trimethylamine',
																		 '1-Methylhistidine',
																		 '2-Furoylglycine',
																		 '4-Aminobutyric acid',
																		 'Alanine', 'Arginine', 'Betaine',
																		 'Creatine', 'Glycine', 'Guanidinoacetic acid',
																		 'Methionine', 'N,N-Dimethylglycine',
																		 'Sarcosine', 'Taurine', 'Valine', 'Benzoic acid',
																		 'D-Mandelic acid', 'Hippuric acid', 'Acetic acid',
																		 'Citric acid', 'Formic acid', 'Fumaric acid',
																		 'Imidazole', 'Lactic acid', 'Proline betaine', 'Succinic acid',
																		 'Tartaric acid', 'Trigonelline', '2-Methylsuccinic acid', '2-Oxoglutaric acid',
																		 '3-Hydroxybutyric acid', 'Acetoacetic acid', 'Acetone', 'Oxaloacetic acid',
																		 'Pyruvic acid', '1-Methyladenosine', '1-Methylnicotinamide', 'Adenosine',
																		 'Allantoin', 'Allopurinol', 'Caffeine', 'Inosine', 'D-Galactose',
																		 'D-Glucose', 'D-Lactose', 'D-Mannitol', 'D-Mannose', 'Myo-Inositol'],
																	'LLOQ': [numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan,
																			 numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan,
																			 numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan,
																			 numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan,
																			 numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan,
																			 numpy.nan, numpy.nan, numpy.nan],
																	'LOD': [31.0, 2.0, 15.0, 39.0, 20.0, 10.0, 750.0, 7.0, 50.0,
																			34.0, 100.0, 18.0, 5.0, 2.0, 140.0, 2.0, 10.0, 2.0,
																			170.0, 5.0, 40.0, 10.0, 2.0, 48.0, 49.0, 25.0, 5.0,
																			5.0, 35.0, 48.0, 92.0, 100.0, 14.0, 2.0, 17.0, 9.0,
																			5.0, 32.0, 390.0, 17.0, 10.0, 45.0, 19.0, 43.0, 34.0,
																			96.0, 180.0, 6.0, 4400.0],
																	'Lower Reference Percentile': [2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5,
																								   2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5,
																								   2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5,
																								   2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5,
																								   2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5,
																								   2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5],
																	'Lower Reference Value': ['-', '-', '-', '-', '-', 11, '-', 9, '-', 38, '-', '-', '-', '-', '-', '-', '-', 2,
																							  '-', '-', '-', '-', '-', '-',  '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-',
																							  '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-'],
																	'ULOQ': [numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan,
																			 numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan,
																			 numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan,
																			 numpy.nan, numpy.nan, numpy.nan],
																	'Unit': ['mmol/mol Crea', 'mmol/mol Crea', 'mmol/mol Crea', 'mmol/mol Crea', 'mmol/mol Crea', 'mmol/mol Crea', 'mmol/mol Crea',
																			 'mmol/mol Crea', 'mmol/mol Crea', 'mmol/mol Crea', 'mmol/mol Crea', 'mmol/mol Crea', 'mmol/mol Crea', 'mmol/mol Crea',
																			 'mmol/mol Crea', 'mmol/mol Crea', 'mmol/mol Crea', 'mmol/mol Crea', 'mmol/mol Crea', 'mmol/mol Crea', 'mmol/mol Crea',
																			 'mmol/mol Crea', 'mmol/mol Crea', 'mmol/mol Crea', 'mmol/mol Crea', 'mmol/mol Crea', 'mmol/mol Crea', 'mmol/mol Crea',
																			 'mmol/mol Crea', 'mmol/mol Crea', 'mmol/mol Crea', 'mmol/mol Crea', 'mmol/mol Crea', 'mmol/mol Crea',  'mmol/mol Crea',
																			 'mmol/mol Crea', 'mmol/mol Crea', 'mmol/mol Crea', 'mmol/mol Crea', 'mmol/mol Crea', 'mmol/mol Crea',  'mmol/mol Crea',
																			 'mmol/mol Crea', 'mmol/mol Crea', 'mmol/mol Crea', 'mmol/mol Crea', 'mmol/mol Crea', 'mmol/mol Crea',  'mmol/mol Crea'],
																	'Upper Reference Percentile': [97.5, 97.5, 97.5, 97.5, 97.5, 97.5, 97.5, 97.5, 97.5, 97.5, 97.5, 97.5, 97.5, 97.5,
																								   97.5, 97.5, 97.5, 97.5, 97.5, 97.5, 97.5, 97.5, 97.5, 97.5, 97.5, 97.5, 97.5,
																								   97.5, 97.5, 97.5, 97.5, 97.5, 97.5, 97.5, 97.5, 97.5, 97.5, 97.5, 97.5, 97.5,
																								   97.5, 97.5, 97.5, 97.5, 97.5, 97.5, 97.5, 97.5, 97.5],
																	'Upper Reference Value': [54.0, 3.0, 15.0, 40.0, 20.0, 72.0, 750.0, 78.0, 280.0, 440.0, 140.0, 18.0, 15.0, 7.0,
																							  170.0, 7.0, 10.0, 17.0, 660.0, 51.0, 700.0, 43.0, 3.0, 48.0, 110.0, 280.0,
																							  39.0, 110.0, 67.0, 48.0, 92.0, 100.0, 30.0, 7.0, 66.0, 13.0, 5.0, 32.0, 390.0,
																							  47.0, 11.0, 61.0, 19.0, 44.0, 140.0, 96.0, 180.0, 8.0, 4400.0],
																	'calibrationMethod': [CalibrationMethod.otherCalibration, CalibrationMethod.otherCalibration, CalibrationMethod.otherCalibration, CalibrationMethod.otherCalibration,
																					  CalibrationMethod.otherCalibration, CalibrationMethod.otherCalibration, CalibrationMethod.otherCalibration, CalibrationMethod.otherCalibration,
																					  CalibrationMethod.otherCalibration, CalibrationMethod.otherCalibration, CalibrationMethod.otherCalibration,
																						  CalibrationMethod.otherCalibration, CalibrationMethod.otherCalibration, CalibrationMethod.otherCalibration,
																						  CalibrationMethod.otherCalibration, CalibrationMethod.otherCalibration, CalibrationMethod.otherCalibration, CalibrationMethod.otherCalibration,
																						  CalibrationMethod.otherCalibration, CalibrationMethod.otherCalibration, CalibrationMethod.otherCalibration, CalibrationMethod.otherCalibration, CalibrationMethod.otherCalibration,
																						  CalibrationMethod.otherCalibration, CalibrationMethod.otherCalibration, CalibrationMethod.otherCalibration, CalibrationMethod.otherCalibration, CalibrationMethod.otherCalibration,
																						  CalibrationMethod.otherCalibration, CalibrationMethod.otherCalibration, CalibrationMethod.otherCalibration, CalibrationMethod.otherCalibration, CalibrationMethod.otherCalibration, CalibrationMethod.otherCalibration,
																						  CalibrationMethod.otherCalibration, CalibrationMethod.otherCalibration, CalibrationMethod.otherCalibration, CalibrationMethod.otherCalibration, CalibrationMethod.otherCalibration, CalibrationMethod.otherCalibration, CalibrationMethod.otherCalibration,
																						  CalibrationMethod.otherCalibration, CalibrationMethod.otherCalibration, CalibrationMethod.otherCalibration, CalibrationMethod.otherCalibration, CalibrationMethod.otherCalibration, CalibrationMethod.otherCalibration,
																						  CalibrationMethod.otherCalibration, CalibrationMethod.otherCalibration],
																	'comment': ['', '', '', '', '', '', '', '', '', '', '', '', '', '',
																				'', '', '', '', '', '', '', '', '', '', '', '', '',
																				'', '', '', '', '', '', '', '', '', '', '', '', '',
																				'', '', '', '', '', '', '', '', ''],
																	'lodMask': [True, True, True, True, True, True, True, True, True, True, True, True,
																				True, True, True, True, True, True, True, True, True, True, True, True,
																				True, True, True, True, True, True, True, True, True, True, True, True,
																				True, True, True, True, True, True, True, True, True, True, True, True, True],
																	'quantificationType': [QuantificationType.QuantOther, QuantificationType.QuantOther, QuantificationType.QuantOther,
																						   QuantificationType.QuantOther, QuantificationType.QuantOther, QuantificationType.QuantOther,
																						   QuantificationType.QuantOther, QuantificationType.QuantOther, QuantificationType.QuantOther,
																						   QuantificationType.QuantOther, QuantificationType.QuantOther, QuantificationType.QuantOther,
																						   QuantificationType.QuantOther, QuantificationType.QuantOther, QuantificationType.QuantOther,
																						   QuantificationType.QuantOther, QuantificationType.QuantOther, QuantificationType.QuantOther,
																						   QuantificationType.QuantOther, QuantificationType.QuantOther, QuantificationType.QuantOther,
																						   QuantificationType.QuantOther, QuantificationType.QuantOther, QuantificationType.QuantOther,
																						   QuantificationType.QuantOther, QuantificationType.QuantOther, QuantificationType.QuantOther,
																						   QuantificationType.QuantOther, QuantificationType.QuantOther, QuantificationType.QuantOther,
																						   QuantificationType.QuantOther, QuantificationType.QuantOther, QuantificationType.QuantOther,
																						   QuantificationType.QuantOther, QuantificationType.QuantOther, QuantificationType.QuantOther,
																						   QuantificationType.QuantOther, QuantificationType.QuantOther, QuantificationType.QuantOther,
																						   QuantificationType.QuantOther, QuantificationType.QuantOther, QuantificationType.QuantOther,
																						   QuantificationType.QuantOther, QuantificationType.QuantOther, QuantificationType.QuantOther,
																						   QuantificationType.QuantOther, QuantificationType.QuantOther, QuantificationType.QuantOther,
																						   QuantificationType.QuantOther]})

		self.expectedQuantUR['intensityData'] = numpy.array([[53.00, 0.00, 0.00, 47.00, 0.00, 61.00, 0.00, 24.00, 2100.00, 250.00, 140.00, 0.00, 46.00, 5.00, 660.00, 12.00, 0.00, 0.00, 250.00, 25.00,
															  480.00, 49.00, 0.00, 0.00, 0.00, 39.00, 47.00, 16.00, 0.00, 0.00, 0.00, 0.00, 38.00, 7.00, 63.00, 14.00, 0.00, 0.00, 0.00, 0.00, 22.00,
															  0.00, 0.00, 0.00, 110.00, 0.00, 0.00, 0.00, 0.00],
															 [62.00, 0.00, 0.00, 0.00, 0.00, 69.00, 0.00, 41.00, 800.00, 210.00, 160.00, 0.00, 34.00, 2.00, 150.00, 8.00, 0.00, 0.00, 220.00, 11.00, 650.00,
															  36.00, 2.00, 0.00, 0.00, 100.00, 23.00, 710.00, 0.00, 0.00, 0.00, 0.00, 37.00, 7.00, 48.00, 15.00, 0.00, 0.00, 0.00, 25.00, 0.00, 66.00, 0.00,
															  0.00, 110.00, 0.00, 0.00, 0.00, 0.00],
															 [57.00, 0.00, 0.00, 0.00, 0.00, 65.00, 0.00, 54.00, 1700.00, 190.00, 210.00, 0.00, 37.00, 5.00, 180.00, 10.00, 0.00, 0.00, 330.00, 10.00,
															  350.00, 78.00, 0.00, 0.00, 0.00, 0.00, 6.00, 120.00, 0.00, 0.00, 0.00, 0.00, 15.00, 5.00, 0.00, 15.00, 0.00, 0.00, 0.00, 0.00, 0.00, 45.00,
															  0.00, 0.00, 110.00, 0.00, 0.00, 0.00, 0.00],
															 [61.00, 0.00, 0.00, 0.00, 0.00, 56.00, 0.00, 23.00, 1500.00, 150.00, 160.00, 0.00, 24.00, 3.00, 370.00, 7.00, 0.00, 0.00, 180.00, 18.00, 380.00,
															  65.00, 0.00, 0.00, 0.00, 0.00, 16.00, 0.00, 0.00, 0.00, 0.00, 0.00, 14.00, 4.00, 26.00, 14.00, 0.00, 0.00, 0.00, 0.00, 17.00, 56.00, 0.00, 0.00,
															  98.00, 0.00, 0.00, 0.00, 0.00],
															 [37.00, 0.00, 0.00, 0.00, 0.00, 52.00, 0.00, 0.00, 99.00, 160.00, 0.00, 0.00, 6.00, 0.00, 0.00, 3.00, 0.00, 0.00,210.00, 9.00, 620.00, 14.00,
															  0.00, 0.00, 0.00, 0.00, 14.00, 5.00, 0.00, 0.00, 0.00, 0.00, 18.00, 2.00, 24.00, 0.00, 0.00, 0.00, 0.00, 27.00, 0.00, 0.00, 0.00, 0.00, 84.00,
															  0.00, 0.00, 0.00, 0.00],
															 [32.00, 0.00, 0.00, 0.00, 0.00, 43.00, 0.00, 11.00, 460.00, 260.00, 120.00, 0.00, 0.00, 0.00, 190.00, 4.00, 0.00, 0.00, 0.00, 0.00, 790.00, 18.00,
															  0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 2.00, 0.00, 0.00, 0.00, 0.00, 0.00, 29.00, 0.00, 0.00, 0.00, 0.00, 100.00, 0.00,
															  0.00, 11.00, 0.00],
															 [36.00, 0.00, 0.00, 0.00, 0.00, 40.00, 0.00, 10.00, 270.00, 200.00, 130.00, 0.00, 0.00, 0.00, 210.00, 5.00, 0.00, 0.00, 0.00, 0.00, 730.00, 14.00,
															  0.00, 0.00, 0.00, 0.00, 6.00, 0.00, 0.00, 0.00, 0.00, 0.00, 25.00, 4.00, 38.00, 0.00, 0.00, 0.00, 0.00, 22.00, 10.00, 0.00, 0.00, 0.00, 85.00, 0.00,
															  190.00, 11.00, 0.00],
															 [33.00, 0.00, 0.00, 0.00, 0.00, 85.00, 0.00, 13.00, 1100.00, 350.00, 240.00, 0.00, 15.00, 0.00, 240.00, 4.00, 0.00, 0.00, 0.00, 20.00, 670.00, 47.00,
															  0.00, 0.00, 0.00, 0.00, 30.00, 250.00, 0.00, 0.00, 0.00, 0.00, 17.00, 3.00, 22.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 90.00, 0.00,
															  0.00, 0.00, 0.00],
															 [47.00, 4.00, 0.00, 0.00, 0.00, 79.00, 0.00, 21.00, 100.00, 470.00, 170.00, 0.00, 20.00, 3.00, 0.00, 3.00, 0.00, 0.00, 350.00, 15.00, 51.00, 52.00,
															  0.00, 0.00, 0.00, 0.00, 34.00, 17.00, 140.00, 0.00, 0.00, 0.00, 18.00, 3.00, 38.00, 0.00, 0.00, 0.00, 0.00, 24.00, 0.00, 54.00, 0.00, 0.00, 110.00,
															  0.00, 0.00, 0.00, 0.00]])

		self.expectedQuantUR['expectedConcentration'] = pandas.DataFrame(None, index=list(self.expectedQuantUR['sampleMetadata'].index), columns=self.expectedQuantUR['featureMetadata']['Feature Name'].tolist())

		# Calibration
		self.expectedQuantUR['calibIntensityData'] = numpy.ndarray((0, self.expectedQuantUR['featureMetadata'].shape[0]))
		self.expectedQuantUR['calibSampleMetadata'] = pandas.DataFrame(None, columns=self.expectedQuantUR['sampleMetadata'].columns)
		self.expectedQuantUR['calibSampleMetadata']['Metadata Available'] = False
		self.expectedQuantUR['calibFeatureMetadata'] = pandas.DataFrame({'Feature Name': self.expectedQuantUR['featureMetadata']['Feature Name'].tolist()})
		self.expectedQuantUR['calibExpectedConcentration'] = pandas.DataFrame(None, columns=self.expectedQuantUR['featureMetadata']['Feature Name'].tolist())
		# Excluded
		self.expectedQuantUR['sampleMetadataExcluded'] = []
		self.expectedQuantUR['featureMetadataExcluded'] = []
		self.expectedQuantUR['intensityDataExcluded'] = []
		self.expectedQuantUR['expectedConcentrationExcluded'] = []
		self.expectedQuantUR['excludedFlag'] = []
		# Attributes
		tmpDataset = nPYc.TargetedDataset('', fileType='empty')
		self.expectedQuantUR['Attributes'] = {'methodName':"Bruker Quant-UR Data",
										'dpi': tmpDataset.Attributes['dpi'],
										'rsdThreshold':30,
										'figureFormat': tmpDataset.Attributes['figureFormat'],
										'figureSize': tmpDataset.Attributes['figureSize'],
										'histBins': tmpDataset.Attributes['histBins'],
										'noFiles': tmpDataset.Attributes['noFiles'],
										'quantiles': tmpDataset.Attributes['quantiles'],
										'methodName': 'Bruker Quant-UR Data',
										'externalID': [],
										'sampleMetadataNotExported': ['Acqu Date', 'Acqu Time', 'Sample Type'],
										'featureMetadataNotExported': ['comment'],
										'analyticalMeasurements': {'Acquired Time': 'date', 'Acquisition batch': 'categorical', 'Assay data location': 'categorical',
																  'Assay data name': 'categorical', 'Assay protocol': 'categorical', 'AssayRole': 'categorical', 'Batch': 'categorical',
																  'Correction Batch': 'categorical', 'Dilution': 'continuous', 'Exclusion Details': 'categorical',
																  'Instrument': 'categorical', 'Matrix': 'categorical', 'Measurement Date': 'date', 'Measurement Time': 'date', 'Plate': 'categorical',
																  'Plot Sample Type': 'categorical', 'Re-Run': 'categorical', 'Run Order': 'continuous', 'Sample batch': 'categorical',
																  'Sample position': 'categorical', 'SampleType': 'categorical', 'Skipped': 'categorical', 'Study': 'categorical',
																  'Suplemental Injections': 'categorical', 'Well': 'categorical'},
										'excludeFromPlotting': ['Sample File Name', 'Sample Base Name', 'Batch Termini',
																'Study Reference', 'Long-Term Reference', 'Method Reference',
																'Dilution Series', 'Skipped', 'Study Sample', 'File Path',
																'Exclusion Details', 'Assay protocol', 'Status', 'Measurement Date',
																'Measurement Time', 'Data Present', 'LIMS Present', 'LIMS Marked Missing',
																'Assay data name', 'Assay data location', 'AssayRole',
																'SampleType', 'Sample ID', 'Plot Sample Type', 'SubjectInfoData',
																'Detector Unit', 'TargetLynx Sample ID', 'MassLynx Row ID'],
										'additionalQuantParamColumns': ['LOD',
																	   'Lower Reference Percentile',
																	   'Lower Reference Value',
																	   'Upper Reference Percentile',
																	   'Upper Reference Value'],
										"sampleTypeColours": {"StudySample": "b", "StudyPool": "g", "ExternalReference": "r", "MethodReference": "m", "ProceduralBlank": "c", "Other": "grey",
															  "Study Sample": "b", "Study Reference": "g", "Long-Term Reference": "r",
															  "Method Reference": "m", "Blank": "c", "Unspecified SampleType or AssayRole": "grey"}}

		# 112 features, 10 samples, BrukerBI-LISA
		self.datapathBILISA = os.path.join('..', '..', 'npc-standard-project', 'Raw_Data', 'nmr', 'UnitTest3')
		# Expected TargetedDataset
		# Do not check sampleMetadata['Path']
		self.expectedBILISA = dict()
		self.expectedBILISA['sampleMetadata'] = pandas.DataFrame({'Sample File Name': ['UnitTest3_Serum_Rack01_RCM_190116/10',
																					   'UnitTest3_Serum_Rack01_RCM_190116/100',
																					   'UnitTest3_Serum_Rack01_RCM_190116/110',
																					   'UnitTest3_Serum_Rack01_RCM_190116/120',
																					   'UnitTest3_Serum_Rack01_RCM_190116/130',
																					   'UnitTest3_Serum_Rack01_RCM_190116/140',
																					   'UnitTest3_Serum_Rack01_RCM_190116/150',
																					   'UnitTest3_Serum_Rack01_RCM_190116/160',
																					   'UnitTest3_Serum_Rack01_RCM_190116/170',
																					   'UnitTest3_Serum_Rack01_RCM_190116/180'],
																  'Sample Base Name': ['UnitTest3_Serum_Rack01_RCM_190116/10',
																					   'UnitTest3_Serum_Rack01_RCM_190116/100',
																					   'UnitTest3_Serum_Rack01_RCM_190116/110',
																					   'UnitTest3_Serum_Rack01_RCM_190116/120',
																					   'UnitTest3_Serum_Rack01_RCM_190116/130',
																					   'UnitTest3_Serum_Rack01_RCM_190116/140',
																					   'UnitTest3_Serum_Rack01_RCM_190116/150',
																					   'UnitTest3_Serum_Rack01_RCM_190116/160',
																					   'UnitTest3_Serum_Rack01_RCM_190116/170',
																					   'UnitTest3_Serum_Rack01_RCM_190116/180'],
																  'expno': [ 10, 100, 110, 120, 130, 140, 150, 160, 170, 180],
																  'Acquired Time': [pandas.Timestamp(datetime(2017, 5, 2, 12, 39, 12)),
																					pandas.Timestamp(datetime(2017, 5, 5, 21, 32, 37)),
																					pandas.Timestamp(datetime(2017, 5, 2, 16, 3, 59)),
																					pandas.Timestamp(datetime(2017, 5, 2, 16, 49, 39)),
																					pandas.Timestamp(datetime(2017, 5, 2, 17, 12, 42)),
																					pandas.Timestamp(datetime(2017, 5, 5, 21, 56, 7)),
																					pandas.Timestamp(datetime(2017, 5, 5, 22, 19, 12)),
																					pandas.Timestamp(datetime(2017, 5, 5, 22, 42, 32)),
																					pandas.Timestamp(datetime(2017, 5, 2, 18, 45, 32)),
																					pandas.Timestamp(datetime(2017, 5, 2, 19, 8, 37))],
															'Run Order': [0, 6, 1, 2, 3, 7, 8, 9, 4, 5],
															'AssayRole': [numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan],
															'SampleType': [numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan],
															'Dilution': [100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
															'Correction Batch': [numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan],
															'Sample ID': [numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan],
															'Exclusion Details': [None, None, None, None, None, None, None, None, None, None],
															'Batch': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]})
		self.expectedBILISA['sampleMetadata']['Metadata Available'] = False


		self.expectedBILISA['featureMetadata'] = pandas.DataFrame({'Feature Name': {0: 'TPTG', 1: 'TPCH', 2: 'LDCH', 3: 'HDCH', 4: 'TPA1', 5: 'TPA2', 
																					6: 'TPAB', 7: 'LDHD', 8: 'ABA1', 9: 'TBPN', 10: 'VLPN', 11: 'IDPN', 12: 'LDPN',
																					13: 'L1PN', 14: 'L2PN', 15: 'L3PN', 16: 'L4PN', 17: 'L5PN', 18: 'L6PN', 19: 'VLTG', 
																					20: 'IDTG', 21: 'LDTG', 22: 'HDTG', 23: 'VLCH', 24: 'IDCH', 25: 'VLFC', 26: 'IDFC',
																					27: 'LDFC', 28: 'HDFC', 29: 'VLPL', 30: 'IDPL', 31: 'LDPL', 32: 'HDPL', 33: 'HDA1',
																					34: 'HDA2', 35: 'VLAB', 36: 'IDAB', 37: 'LDAB', 38: 'V1TG', 39: 'V2TG', 40: 'V3TG', 
																					41: 'V4TG', 42: 'V5TG', 43: 'V1CH', 44: 'V2CH', 45: 'V3CH', 46: 'V4CH', 47: 'V5CH',
																					48: 'V1FC', 49: 'V2FC', 50: 'V3FC', 51: 'V4FC', 52: 'V5FC', 53: 'V1PL', 54: 'V2PL',
																					55: 'V3PL', 56: 'V4PL', 57: 'V5PL', 58: 'L1TG', 59: 'L2TG', 60: 'L3TG', 61: 'L4TG', 
																					62: 'L5TG', 63: 'L6TG', 64: 'L1CH', 65: 'L2CH', 66: 'L3CH', 67: 'L4CH', 68: 'L5CH',
																					69: 'L6CH', 70: 'L1FC', 71: 'L2FC', 72: 'L3FC', 73: 'L4FC', 74: 'L5FC', 75: 'L6FC',
																					76: 'L1PL', 77: 'L2PL', 78: 'L3PL', 79: 'L4PL', 80: 'L5PL', 81: 'L6PL', 82: 'L1AB', 
																					83: 'L2AB', 84: 'L3AB', 85: 'L4AB', 86: 'L5AB', 87: 'L6AB', 88: 'H1TG', 89: 'H2TG', 
																					90: 'H3TG', 91: 'H4TG', 92: 'H1CH', 93: 'H2CH', 94: 'H3CH', 95: 'H4CH', 96: 'H1FC', 
																					97: 'H2FC', 98: 'H3FC', 99: 'H4FC', 100: 'H1PL', 101: 'H2PL', 102: 'H3PL', 103: 'H4PL', 104: 'H1A1', 
																					105: 'H2A1', 106: 'H3A1', 107: 'H4A1', 108: 'H1A2', 109: 'H2A2', 110: 'H3A2', 111: 'H4A2'}, 
																   'comment': {0: 'Main Parameters, Triglycerides, TG', 1: 'Main Parameters, Cholesterol, Chol', 2: 'Main Parameters, LDL Cholesterol, LDL-Chol', 
																			   3: 'Main Parameters, HDL Cholesterol, HDL-Chol', 4: 'Main Parameters, Apo-A1, Apo-A1', 5: 'Main Parameters, Apo-A2, Apo-A2', 
																			   6: 'Main Parameters, Apo-B100, Apo-B100', 7: 'Calculated Figures, LDL Cholesterol / HDL Cholesterol, LDL-Chol/HDL-Chol', 
																			   8: 'Calculated Figures, Apo-A1 / Apo-B100, Apo-B100/Apo-A1', 9: 'Calculated Figures, Total ApoB Particle Number, Total Particle Number', 
																			   10: 'Calculated Figures, VLDL Particle Number, VLDL Particle Number', 11: 'Calculated Figures, IDL Particle Number, IDL Particle Number', 
																			   12: 'Calculated Figures, LDL Particle Number, LDL Particle Number', 13: 'Calculated Figures, LDL-1 Particle Number, LDL-1 Particle Number', 
																			   14: 'Calculated Figures, LDL-2 Particle Number, LDL-2 Particle Number', 15: 'Calculated Figures, LDL-3 Particle Number, LDL-3 Particle Number', 
																			   16: 'Calculated Figures, LDL-4 Particle Number, LDL-4 Particle Number', 17: 'Calculated Figures, LDL-5 Particle Number, LDL-5 Particle Number', 
																			   18: 'Calculated Figures, LDL-6 Particle Number, LDL-6 Particle Number', 19: 'Lipoprotein Main Fractions, Triglycerides, VLDL', 
																			   20: 'Lipoprotein Main Fractions, Triglycerides, IDL', 21: 'Lipoprotein Main Fractions, Triglycerides, LDL', 22: 'Lipoprotein Main Fractions, Triglycerides, HDL', 23: 'Lipoprotein Main Fractions, Cholesterol, VLDL', 24: 'Lipoprotein Main Fractions, Cholesterol, IDL', 
																			   25: 'Lipoprotein Main Fractions, Free Cholesterol, VLDL', 26: 'Lipoprotein Main Fractions, Free Cholesterol, IDL', 27: 'Lipoprotein Main Fractions, Free Cholesterol, LDL', 
																			   28: 'Lipoprotein Main Fractions, Free Cholesterol, HDL', 29: 'Lipoprotein Main Fractions, Phospholipids, VLDL', 30: 'Lipoprotein Main Fractions, Phospholipids, IDL', 
																			   31: 'Lipoprotein Main Fractions, Phospholipids, LDL', 32: 'Lipoprotein Main Fractions, Phospholipids, HDL', 33: 'Lipoprotein Main Fractions, Apo-A1, HDL', 
																			   34: 'Lipoprotein Main Fractions, Apo-A2, HDL', 35: 'Lipoprotein Main Fractions, Apo-B, VLDL', 36: 'Lipoprotein Main Fractions, Apo-B, IDL', 37: 'Lipoprotein Main Fractions, Apo-B, LDL', 
																			   38: 'VLDL Subfractions, Triglycerides, VLDL-1', 39: 'VLDL Subfractions, Triglycerides, VLDL-2', 40: 'VLDL Subfractions, Triglycerides, VLDL-3', 
																			   41: 'VLDL Subfractions, Triglycerides, VLDL-4', 42: 'VLDL Subfractions, Triglycerides, VLDL-5', 43: 'VLDL Subfractions, Cholesterol, VLDL-1', 44: 'VLDL Subfractions, Cholesterol, VLDL-2', 
																			   45: 'VLDL Subfractions, Cholesterol, VLDL-3', 46: 'VLDL Subfractions, Cholesterol, VLDL-4', 47: 'VLDL Subfractions, Cholesterol, VLDL-5', 
																			   48: 'VLDL Subfractions, Free Cholesterol, VLDL-1', 49: 'VLDL Subfractions, Free Cholesterol, VLDL-2', 50: 'VLDL Subfractions, Free Cholesterol, VLDL-3', 51: 'VLDL Subfractions, Free Cholesterol, VLDL-4', 
																			   52: 'VLDL Subfractions, Free Cholesterol, VLDL-5', 53: 'VLDL Subfractions, Phospholipids, VLDL-1', 54: 'VLDL Subfractions, Phospholipids, VLDL-2', 55: 'VLDL Subfractions, Phospholipids, VLDL-3', 
																			   56: 'VLDL Subfractions, Phospholipids, VLDL-4', 57: 'VLDL Subfractions, Phospholipids, VLDL-5', 58: 'LDL Subfractions, Triglycerides, LDL-1', 59: 'LDL Subfractions, Triglycerides, LDL-2',
																			   60: 'LDL Subfractions, Triglycerides, LDL-3', 61: 'LDL Subfractions, Triglycerides, LDL-4', 62: 'LDL Subfractions, Triglycerides, LDL-5', 63: 'LDL Subfractions, Triglycerides, LDL-6',
																			   64: 'LDL Subfractions, Cholesterol, LDL-1', 65: 'LDL Subfractions, Cholesterol, LDL-2', 66: 'LDL Subfractions, Cholesterol, LDL-3', 67: 'LDL Subfractions, Cholesterol, LDL-4', 
																			   68: 'LDL Subfractions, Cholesterol, LDL-5', 69: 'LDL Subfractions, Cholesterol, LDL-6', 70: 'LDL Subfractions, Free Cholesterol, LDL-1', 71: 'LDL Subfractions, Free Cholesterol, LDL-2', 
																			   72: 'LDL Subfractions, Free Cholesterol, LDL-3', 73: 'LDL Subfractions, Free Cholesterol, LDL-4', 74: 'LDL Subfractions, Free Cholesterol, LDL-5', 75: 'LDL Subfractions, Free Cholesterol, LDL-6', 
																			   76: 'LDL Subfractions, Phospholipids, LDL-1', 77: 'LDL Subfractions, Phospholipids, LDL-2', 78: 'LDL Subfractions, Phospholipids, LDL-3', 79: 'LDL Subfractions, Phospholipids, LDL-4', 
																			   80: 'LDL Subfractions, Phospholipids, LDL-5', 81: 'LDL Subfractions, Phospholipids, LDL-6', 82: 'LDL Subfractions, Apo-B, LDL-1', 83: 'LDL Subfractions, Apo-B, LDL-2', 
																			   84: 'LDL Subfractions, Apo-B, LDL-3', 85: 'LDL Subfractions, Apo-B, LDL-4', 86: 'LDL Subfractions, Apo-B, LDL-5', 87: 'LDL Subfractions, Apo-B, LDL-6', 88: 'HDL Subfractions, Triglycerides, HDL-1', 
																			   89: 'HDL Subfractions, Triglycerides, HDL-2', 90: 'HDL Subfractions, Triglycerides, HDL-3', 91: 'HDL Subfractions, Triglycerides, HDL-4', 92: 'HDL Subfractions, Cholesterol, HDL-1', 
																			   93: 'HDL Subfractions, Cholesterol, HDL-2', 94: 'HDL Subfractions, Cholesterol, HDL-3', 95: 'HDL Subfractions, Cholesterol, HDL-4', 96: 'HDL Subfractions, Free Cholesterol, HDL-1', 
																			   97: 'HDL Subfractions, Free Cholesterol, HDL-2', 98: 'HDL Subfractions, Free Cholesterol, HDL-3', 99: 'HDL Subfractions, Free Cholesterol, HDL-4', 100: 'HDL Subfractions, Phospholipids, HDL-1', 
																			   101: 'HDL Subfractions, Phospholipids, HDL-2', 102: 'HDL Subfractions, Phospholipids, HDL-3', 103: 'HDL Subfractions, Phospholipids, HDL-4', 104: 'HDL Subfractions, Apo-A1, HDL-1', 
																			   105: 'HDL Subfractions, Apo-A1, HDL-2', 106: 'HDL Subfractions, Apo-A1, HDL-3', 107: 'HDL Subfractions, Apo-A1, HDL-4', 108: 'HDL Subfractions, Apo-A2, HDL-1', 109: 'HDL Subfractions, Apo-A2, HDL-2', 
																			   110: 'HDL Subfractions, Apo-A2, HDL-3', 111: 'HDL Subfractions, Apo-A2, HDL-4'}, 
																   'LOD': {0: numpy.nan, 1: numpy.nan, 2: numpy.nan, 3: numpy.nan, 4: numpy.nan, 5: numpy.nan, 6: numpy.nan, 7: numpy.nan, 8: numpy.nan, 9: numpy.nan, 10: numpy.nan, 11: numpy.nan, 12: numpy.nan, 13: numpy.nan, 14: numpy.nan, 15: numpy.nan, 16: numpy.nan, 17: numpy.nan, 18: numpy.nan, 19: numpy.nan, 
																		   20: numpy.nan, 21: numpy.nan, 22: numpy.nan, 23: numpy.nan, 24: numpy.nan, 25: numpy.nan, 26: numpy.nan, 27: numpy.nan, 28: numpy.nan, 29: numpy.nan, 30: numpy.nan, 31: numpy.nan, 32: numpy.nan, 33: numpy.nan, 34: numpy.nan, 35: numpy.nan, 36: numpy.nan, 37: numpy.nan, 38: numpy.nan, 
																		   39: numpy.nan, 40: numpy.nan, 41: numpy.nan, 42: numpy.nan, 43: numpy.nan, 44: numpy.nan, 45: numpy.nan, 46: numpy.nan, 47: numpy.nan, 48: numpy.nan, 49: numpy.nan, 50: numpy.nan, 51: numpy.nan, 52: numpy.nan, 53: numpy.nan, 54: numpy.nan, 55: numpy.nan, 56: numpy.nan, 57: numpy.nan, 
																		   58: numpy.nan, 59: numpy.nan, 60: numpy.nan, 61: numpy.nan, 62: numpy.nan, 63: numpy.nan, 64: numpy.nan, 65: numpy.nan, 66: numpy.nan, 67: numpy.nan, 68: numpy.nan, 69: numpy.nan, 70: numpy.nan, 71: numpy.nan, 72: numpy.nan, 73: numpy.nan, 74: numpy.nan, 75: numpy.nan, 76: numpy.nan, 77: numpy.nan, 
																		   78: numpy.nan, 79: numpy.nan, 80: numpy.nan, 81: numpy.nan, 82: numpy.nan, 83: numpy.nan, 84: numpy.nan, 85: numpy.nan, 86: numpy.nan, 87: numpy.nan, 88: numpy.nan, 89: numpy.nan, 90: numpy.nan, 91: numpy.nan, 92: numpy.nan, 93: numpy.nan, 94: numpy.nan, 95: numpy.nan, 96: numpy.nan, 
																		   97: numpy.nan, 98: numpy.nan, 99: numpy.nan, 100: numpy.nan, 101: numpy.nan, 102: numpy.nan, 103: numpy.nan, 104: numpy.nan, 105: numpy.nan, 106: numpy.nan, 107: numpy.nan, 108: numpy.nan, 109: numpy.nan, 110: numpy.nan, 111: numpy.nan}, 
																   'LLOQ': {0: numpy.nan, 1: numpy.nan, 2: numpy.nan, 3: numpy.nan, 4: numpy.nan, 5: numpy.nan, 6: numpy.nan, 7: numpy.nan, 8: numpy.nan, 9: numpy.nan, 10: numpy.nan, 11: numpy.nan, 12: numpy.nan, 13: numpy.nan, 14: numpy.nan, 15: numpy.nan, 16: numpy.nan, 17: numpy.nan, 18: numpy.nan, 19: numpy.nan, 
																			20: numpy.nan, 21: numpy.nan, 22: numpy.nan, 23: numpy.nan, 24: numpy.nan, 25: numpy.nan, 26: numpy.nan, 27: numpy.nan, 28: numpy.nan, 29: numpy.nan, 30: numpy.nan, 31: numpy.nan, 32: numpy.nan, 33: numpy.nan, 34: numpy.nan, 35: numpy.nan, 36: numpy.nan, 37: numpy.nan, 38: numpy.nan,
																			39: numpy.nan, 40: numpy.nan, 41: numpy.nan, 42: numpy.nan, 43: numpy.nan, 44: numpy.nan, 45: numpy.nan, 46: numpy.nan, 47: numpy.nan, 48: numpy.nan, 49: numpy.nan, 50: numpy.nan, 51: numpy.nan, 52: numpy.nan, 53: numpy.nan, 54: numpy.nan, 55: numpy.nan, 56: numpy.nan, 57: numpy.nan, 
																			58: numpy.nan, 59: numpy.nan, 60: numpy.nan, 61: numpy.nan, 62: numpy.nan, 63: numpy.nan, 64: numpy.nan, 65: numpy.nan, 66: numpy.nan, 67: numpy.nan, 68: numpy.nan, 69: numpy.nan, 70: numpy.nan, 71: numpy.nan, 72: numpy.nan, 73: numpy.nan, 74: numpy.nan, 75: numpy.nan, 76: numpy.nan, 
																			77: numpy.nan, 78: numpy.nan, 79: numpy.nan, 80: numpy.nan, 81: numpy.nan, 82: numpy.nan, 83: numpy.nan, 84: numpy.nan, 85: numpy.nan, 86: numpy.nan, 87: numpy.nan, 88: numpy.nan, 89: numpy.nan, 90: numpy.nan, 91: numpy.nan, 92: numpy.nan, 93: numpy.nan, 94: numpy.nan, 95: numpy.nan, 
																			96: numpy.nan, 97: numpy.nan, 98: numpy.nan, 99: numpy.nan, 100: numpy.nan, 101: numpy.nan, 102: numpy.nan, 103: numpy.nan, 104: numpy.nan, 105: numpy.nan, 106: numpy.nan, 107: numpy.nan, 108: numpy.nan, 109: numpy.nan, 110: numpy.nan, 111: numpy.nan}, 
																   'Unit': {0: 'mg/dL', 1: 'mg/dL', 2: 'mg/dL', 3: 'mg/dL', 4: 'mg/dL', 5: 'mg/dL', 6: 'mg/dL', 7: '-/-', 8: '-/-', 9: 'nmol/L', 10: 'nmol/L', 11: 'nmol/L', 12: 'nmol/L', 13: 'nmol/L', 
																			14: 'nmol/L', 15: 'nmol/L', 16: 'nmol/L', 17: 'nmol/L', 18: 'nmol/L', 19: 'mg/dL', 20: 'mg/dL', 21: 'mg/dL', 22: 'mg/dL', 23: 'mg/dL', 24: 'mg/dL', 25: 'mg/dL', 26: 'mg/dL', 
																			27: 'mg/dL', 28: 'mg/dL', 29: 'mg/dL', 30: 'mg/dL', 31: 'mg/dL', 32: 'mg/dL', 33: 'mg/dL', 34: 'mg/dL', 35: 'mg/dL', 36: 'mg/dL', 37: 'mg/dL', 38: 'mg/dL', 39: 'mg/dL',
																			40: 'mg/dL', 41: 'mg/dL', 42: 'mg/dL', 43: 'mg/dL', 44: 'mg/dL', 45: 'mg/dL', 46: 'mg/dL', 47: 'mg/dL', 48: 'mg/dL', 49: 'mg/dL', 50: 'mg/dL', 51: 'mg/dL', 52: 'mg/dL',
																			53: 'mg/dL', 54: 'mg/dL', 55: 'mg/dL', 56: 'mg/dL', 57: 'mg/dL', 58: 'mg/dL', 59: 'mg/dL', 60: 'mg/dL', 61: 'mg/dL', 62: 'mg/dL', 63: 'mg/dL', 64: 'mg/dL', 65: 'mg/dL', 
																			66: 'mg/dL', 67: 'mg/dL', 68: 'mg/dL', 69: 'mg/dL', 70: 'mg/dL', 71: 'mg/dL', 72: 'mg/dL', 73: 'mg/dL', 74: 'mg/dL', 75: 'mg/dL', 76: 'mg/dL', 77: 'mg/dL', 78: 'mg/dL', 
																			79: 'mg/dL', 80: 'mg/dL', 81: 'mg/dL', 82: 'mg/dL', 83: 'mg/dL', 84: 'mg/dL', 85: 'mg/dL', 86: 'mg/dL', 87: 'mg/dL', 88: 'mg/dL', 89: 'mg/dL', 90: 'mg/dL', 91: 'mg/dL', 
																			92: 'mg/dL', 93: 'mg/dL', 94: 'mg/dL', 95: 'mg/dL', 96: 'mg/dL', 97: 'mg/dL', 98: 'mg/dL', 99: 'mg/dL', 100: 'mg/dL', 101: 'mg/dL', 102: 'mg/dL', 103: 'mg/dL', 104: 'mg/dL',
																			105: 'mg/dL', 106: 'mg/dL', 107: 'mg/dL', 108: 'mg/dL', 109: 'mg/dL', 110: 'mg/dL', 111: 'mg/dL'}, 
																   'lodMask': {0: True, 1: True, 2: True, 3: True, 4: True, 5: True, 6: True, 7: True, 8: True, 9: True, 10: True, 11: True, 12: True, 13: True, 14: True, 15: True, 16: True, 17: True, 
																			   18: True, 19: True, 20: True, 21: True, 22: True, 23: True, 24: True, 25: True, 26: True, 27: True, 28: True, 29: True, 30: True, 31: True, 32: True, 33: True, 34: True, 
																			   35: True, 36: True, 37: True, 38: True, 39: True, 40: True, 41: True, 42: True, 43: True, 44: True, 45: True, 46: True, 47: True, 48: True, 49: True, 50: True, 51: True, 
																			   52: True, 53: True, 54: True, 55: True, 56: True, 57: True, 58: True, 59: True, 60: True, 61: True, 62: True, 63: True, 64: True, 65: True, 66: True, 67: True, 68: True, 
																			   69: True, 70: True, 71: True, 72: True, 73: True, 74: True, 75: True, 76: True, 77: True, 78: True, 79: True, 80: True, 81: True, 82: True, 83: True, 84: True, 85: True, 
																			   86: True, 87: True, 88: True, 89: True, 90: True, 91: True, 92: True, 93: True, 94: True, 95: True, 96: True, 97: True, 98: True, 99: True, 100: True, 101: True, 102: True, 
																			   103: True, 104: True, 105: True, 106: True, 107: True, 108: True, 109: True, 110: True, 111: True},
																   'Lower Reference Percentile': {0: 2.5, 1: 2.5, 2: 2.5, 3: 2.5, 4: 2.5, 5: 2.5, 6: 2.5, 7: 2.5, 8: 2.5, 9: 2.5, 10: 2.5, 11: 2.5, 12: 2.5, 13: 2.5, 14: 2.5, 15: 2.5, 16: 2.5, 17: 2.5,
																								  18: 2.5, 19: 2.5, 20: 2.5, 21: 2.5, 22: 2.5, 23: 2.5, 24: 2.5, 25: 2.5, 26: 2.5, 27: 2.5, 28: 2.5, 29: 2.5, 30: 2.5, 31: 2.5, 32: 2.5, 33: 2.5, 34: 2.5, 
																								  35: 2.5, 36: 2.5, 37: 2.5, 38: 2.5, 39: 2.5, 40: 2.5, 41: 2.5, 42: 2.5, 43: 2.5, 44: 2.5, 45: 2.5, 46: 2.5, 47: 2.5, 48: 2.5, 49: 2.5, 50: 2.5, 51: 2.5,
																								  52: 2.5, 53: 2.5, 54: 2.5, 55: 2.5, 56: 2.5, 57: 2.5, 58: 2.5, 59: 2.5, 60: 2.5, 61: 2.5, 62: 2.5, 63: 2.5, 64: 2.5, 65: 2.5, 66: 2.5, 67: 2.5, 68: 2.5, 
																								  69: 2.5, 70: 2.5, 71: 2.5, 72: 2.5, 73: 2.5, 74: 2.5, 75: 2.5, 76: 2.5, 77: 2.5, 78: 2.5, 79: 2.5, 80: 2.5, 81: 2.5, 82: 2.5, 83: 2.5, 84: 2.5, 85: 2.5,
																								  86: 2.5, 87: 2.5, 88: 2.5, 89: 2.5, 90: 2.5, 91: 2.5, 92: 2.5, 93: 2.5, 94: 2.5, 95: 2.5, 96: 2.5, 97: 2.5, 98: 2.5, 99: 2.5, 100: 2.5, 101: 2.5, 
																								  102: 2.5, 103: 2.5, 104: 2.5, 105: 2.5, 106: 2.5, 107: 2.5, 108: 2.5, 109: 2.5, 110: 2.5, 111: 2.5}, 
																   'Upper Reference Percentile': {0: 97.5, 1: 97.5, 2: 97.5, 3: 97.5, 4: 97.5, 5: 97.5, 6: 97.5, 7: 97.5, 8: 97.5, 9: 97.5, 10: 97.5, 11: 97.5, 12: 97.5, 13: 97.5, 14: 97.5, 15: 97.5, 16: 97.5, 17: 97.5, 
																								  18: 97.5, 19: 97.5, 20: 97.5, 21: 97.5, 22: 97.5, 23: 97.5, 24: 97.5, 25: 97.5, 26: 97.5, 27: 97.5, 28: 97.5, 29: 97.5, 30: 97.5, 31: 97.5, 32: 97.5, 
																								  33: 97.5, 34: 97.5, 35: 97.5, 36: 97.5, 37: 97.5, 38: 97.5, 39: 97.5, 40: 97.5, 41: 97.5, 42: 97.5, 43: 97.5, 44: 97.5, 45: 97.5, 46: 97.5, 47: 97.5, 
																								  48: 97.5, 49: 97.5, 50: 97.5, 51: 97.5, 52: 97.5, 53: 97.5, 54: 97.5, 55: 97.5, 56: 97.5, 57: 97.5, 58: 97.5, 59: 97.5, 60: 97.5, 61: 97.5, 62: 97.5, 
																								  63: 97.5, 64: 97.5, 65: 97.5, 66: 97.5, 67: 97.5, 68: 97.5, 69: 97.5, 70: 97.5, 71: 97.5, 72: 97.5, 73: 97.5, 74: 97.5, 75: 97.5, 76: 97.5, 77: 97.5, 
																								  78: 97.5, 79: 97.5, 80: 97.5, 81: 97.5, 82: 97.5, 83: 97.5, 84: 97.5, 85: 97.5, 86: 97.5, 87: 97.5, 88: 97.5, 89: 97.5, 90: 97.5, 91: 97.5, 92: 97.5, 
																								  93: 97.5, 94: 97.5, 95: 97.5, 96: 97.5, 97: 97.5, 98: 97.5, 99: 97.5, 100: 97.5, 101: 97.5, 102: 97.5, 103: 97.5, 104: 97.5, 105: 97.5, 106: 97.5, 
																								  107: 97.5, 108: 97.5, 109: 97.5, 110: 97.5, 111: 97.5}, 
																   'Lower Reference Value': {0: 53.45, 1: 140.31, 2: 54.52, 3: 34.6, 4: 112.0, 5: 24.07, 6: 48.18, 7: 0.98, 8: 0.3, 9: 876.01, 10: 50.11, 11: 35.95, 12: 760.42, 13: 98.1, 14: 46.65, 
																							 15: 51.24, 16: 77.07, 17: 85.6, 18: 90.64, 19: 21.38, 20: 4.62, 21: 11.77, 22: 7.29, 23: 4.88, 24: 3.91, 25: 2.66, 26: 0.94, 27: 17.19, 28: 6.98, 29: 6.44, 
																							 30: 2.97, 31: 36.69, 32: 56.5, 33: 110.04, 34: 24.86, 35: 2.76, 36: 1.98, 37: 41.82, 38: 6.23, 39: 2.75, 40: 2.16, 41: 2.93, 42: 1.08, 43: 0.8, 44: 0.39, 
																							 45: 0.48, 46: 1.41, 47: 0.1, 48: 0.11, 49: 0.04, 50: 0.05, 51: 0.15, 52: 0.02, 53: 1.31, 54: 0.81, 55: 0.82, 56: 1.62, 57: 0.4, 58: 2.51, 59: 1.19, 60: 1.15, 
																							 61: 1.21, 62: 1.12, 63: 1.35, 64: 8.07, 65: 2.48, 66: 3.19, 67: 4.32, 68: 5.41, 69: 6.26, 70: 2.49, 71: 0.99, 72: 1.27, 73: 1.07, 74: 1.56, 75: 1.78, 76: 5.87, 
																							 77: 2.2, 78: 2.39, 79: 3.05, 80: 3.72, 81: 4.44, 82: 5.4, 83: 2.57, 84: 2.82, 85: 4.24, 86: 4.71, 87: 4.98, 88: 1.4, 89: 0.98, 90: 1.3, 91: 1.94, 92: 6.1, 
																							 93: 3.98, 94: 6.82, 95: 10.64, 96: 1.45, 97: 0.74, 98: 1.25, 99: 2.13, 100: 7.67, 101: 7.4, 102: 12.03, 103: 19.75, 104: 5.95, 105: 9.94, 106: 18.26, 
																							 107: 56.03, 108: 0.77, 109: 1.88, 110: 4.85, 111: 12.02}, 
																   'Upper Reference Value': {0: 489.81, 1: 341.43, 2: 226.6, 3: 96.25, 4: 216.85, 5: 47.7, 6: 159.94, 7: 4.08, 8: 1.07, 9: 2908.2, 10: 473.04, 11: 316.39, 12: 2559.51, 13: 567.23, 
																							 14: 426.74, 15: 499.0, 16: 577.04, 17: 614.86, 18: 815.44, 19: 335.6, 20: 100.01, 21: 45.22, 22: 28.58, 23: 76.99, 24: 49.96, 25: 33.09, 26: 13.89, 27: 63.35, 
																							 28: 27.05, 29: 67.58, 30: 32.53, 31: 120.76, 32: 135.93, 33: 222.02, 34: 47.99, 35: 26.02, 36: 17.4, 37: 140.77, 38: 212.19, 39: 66.92, 40: 48.88, 41: 28.43, 
																							 42: 7.26, 43: 35.13, 44: 15.36, 45: 16.23, 46: 15.11, 47: 3.93, 48: 12.89, 49: 6.6, 50: 8.09, 51: 7.21, 52: 2.2, 53: 32.41, 54: 15.43, 55: 13.53, 56: 12.87, 
																							 57: 5.06, 58: 14.03, 59: 6.4, 60: 5.72, 61: 8.43, 62: 8.98, 63: 13.16, 64: 58.72, 65: 48.09, 66: 45.64, 67: 48.86, 68: 48.69, 69: 54.19, 70: 16.94, 71: 14.35, 
																							 72: 13.29, 73: 12.49, 74: 12.71, 75: 12.18, 76: 29.8, 77: 24.82, 78: 24.21, 79: 25.06, 80: 25.29, 81: 27.91, 82: 31.2, 83: 23.47, 84: 27.44, 85: 31.74, 
																							 86: 33.82, 87: 44.85, 88: 11.96, 89: 5.47, 90: 5.49, 91: 8.49, 92: 46.07, 93: 15.58, 94: 18.79, 95: 30.26, 96: 11.96, 97: 4.59, 98: 5.27, 99: 8.54, 100: 57.17, 
																							 101: 26.82, 102: 31.53, 103: 43.53, 104: 75.4, 105: 36.22, 106: 47.09, 107: 110.49, 108: 8.31, 109: 7.78, 110: 11.84, 111: 29.58},
																   'quantificationType': {0: QuantificationType.Monitored, 1: QuantificationType.Monitored, 2: QuantificationType.Monitored, 3: QuantificationType.Monitored, 4: QuantificationType.Monitored, 5: QuantificationType.Monitored, 
																						  6: QuantificationType.Monitored, 7: QuantificationType.Monitored, 8: QuantificationType.Monitored, 9: QuantificationType.Monitored, 10: QuantificationType.Monitored, 
																						  11: QuantificationType.Monitored, 12: QuantificationType.Monitored, 13: QuantificationType.Monitored, 14: QuantificationType.Monitored, 15: QuantificationType.Monitored, 
																						  16: QuantificationType.Monitored, 17: QuantificationType.Monitored, 18: QuantificationType.Monitored, 19: QuantificationType.Monitored, 20: QuantificationType.Monitored, 
																						  21: QuantificationType.Monitored, 22: QuantificationType.Monitored, 23: QuantificationType.Monitored, 24: QuantificationType.Monitored, 25: QuantificationType.Monitored, 
																						  26: QuantificationType.Monitored, 27: QuantificationType.Monitored, 28: QuantificationType.Monitored, 29: QuantificationType.Monitored, 30: QuantificationType.Monitored, 
																						  31: QuantificationType.Monitored, 32: QuantificationType.Monitored, 33: QuantificationType.Monitored, 34: QuantificationType.Monitored, 35: QuantificationType.Monitored, 
																						  36: QuantificationType.Monitored, 37: QuantificationType.Monitored, 38: QuantificationType.Monitored, 39: QuantificationType.Monitored, 40: QuantificationType.Monitored, 
																						  41: QuantificationType.Monitored, 42: QuantificationType.Monitored, 43: QuantificationType.Monitored, 44: QuantificationType.Monitored, 45: QuantificationType.Monitored, 
																						  46: QuantificationType.Monitored, 47: QuantificationType.Monitored, 48: QuantificationType.Monitored, 49: QuantificationType.Monitored, 50: QuantificationType.Monitored, 
																						  51: QuantificationType.Monitored, 52: QuantificationType.Monitored, 53: QuantificationType.Monitored, 54: QuantificationType.Monitored, 55: QuantificationType.Monitored, 
																						  56: QuantificationType.Monitored, 57: QuantificationType.Monitored, 58: QuantificationType.Monitored, 59: QuantificationType.Monitored, 60: QuantificationType.Monitored, 
																						  61: QuantificationType.Monitored, 62: QuantificationType.Monitored, 63: QuantificationType.Monitored, 64: QuantificationType.Monitored, 65: QuantificationType.Monitored, 
																						  66: QuantificationType.Monitored, 67: QuantificationType.Monitored, 68: QuantificationType.Monitored, 69: QuantificationType.Monitored, 70: QuantificationType.Monitored, 
																						  71: QuantificationType.Monitored, 72: QuantificationType.Monitored, 73: QuantificationType.Monitored, 74: QuantificationType.Monitored, 75: QuantificationType.Monitored, 
																						  76: QuantificationType.Monitored, 77: QuantificationType.Monitored, 78: QuantificationType.Monitored, 79: QuantificationType.Monitored, 80: QuantificationType.Monitored, 
																						  81: QuantificationType.Monitored, 82: QuantificationType.Monitored, 83: QuantificationType.Monitored, 84: QuantificationType.Monitored, 85: QuantificationType.Monitored, 
																						  86: QuantificationType.Monitored, 87: QuantificationType.Monitored, 88: QuantificationType.Monitored, 89: QuantificationType.Monitored, 90: QuantificationType.Monitored, 
																						  91: QuantificationType.Monitored, 92: QuantificationType.Monitored, 93: QuantificationType.Monitored, 94: QuantificationType.Monitored, 95: QuantificationType.Monitored, 
																						  96: QuantificationType.Monitored, 97: QuantificationType.Monitored, 98: QuantificationType.Monitored, 99: QuantificationType.Monitored, 100: QuantificationType.Monitored, 
																						  101: QuantificationType.Monitored, 102: QuantificationType.Monitored, 103: QuantificationType.Monitored, 104: QuantificationType.Monitored, 105: QuantificationType.Monitored, 
																						  106: QuantificationType.Monitored, 107: QuantificationType.Monitored, 108: QuantificationType.Monitored, 109: QuantificationType.Monitored, 110: QuantificationType.Monitored, 
																						  111: QuantificationType.Monitored}, 
																   'calibrationMethod': {0: CalibrationMethod.noCalibration, 1: CalibrationMethod.noCalibration, 2: CalibrationMethod.noCalibration, 3: CalibrationMethod.noCalibration, 4: CalibrationMethod.noCalibration, 
																						 5: CalibrationMethod.noCalibration, 6: CalibrationMethod.noCalibration, 7: CalibrationMethod.noCalibration, 8: CalibrationMethod.noCalibration, 
																						 9: CalibrationMethod.noCalibration, 10: CalibrationMethod.noCalibration, 11: CalibrationMethod.noCalibration, 12: CalibrationMethod.noCalibration, 
																						 13: CalibrationMethod.noCalibration, 14: CalibrationMethod.noCalibration, 15: CalibrationMethod.noCalibration, 16: CalibrationMethod.noCalibration, 
																						 17: CalibrationMethod.noCalibration, 18: CalibrationMethod.noCalibration, 19: CalibrationMethod.noCalibration, 20: CalibrationMethod.noCalibration, 21: CalibrationMethod.noCalibration, 
																						 22: CalibrationMethod.noCalibration, 23: CalibrationMethod.noCalibration, 24: CalibrationMethod.noCalibration, 25: CalibrationMethod.noCalibration, 26: CalibrationMethod.noCalibration, 
																						 27: CalibrationMethod.noCalibration, 28: CalibrationMethod.noCalibration, 29: CalibrationMethod.noCalibration, 30: CalibrationMethod.noCalibration, 31: CalibrationMethod.noCalibration, 
																						 32: CalibrationMethod.noCalibration, 33: CalibrationMethod.noCalibration, 34: CalibrationMethod.noCalibration, 35: CalibrationMethod.noCalibration, 36: CalibrationMethod.noCalibration, 
																						 37: CalibrationMethod.noCalibration, 38: CalibrationMethod.noCalibration, 39: CalibrationMethod.noCalibration, 40: CalibrationMethod.noCalibration, 41: CalibrationMethod.noCalibration, 
																						 42: CalibrationMethod.noCalibration, 43: CalibrationMethod.noCalibration, 44: CalibrationMethod.noCalibration, 45: CalibrationMethod.noCalibration, 46: CalibrationMethod.noCalibration, 
																						 47: CalibrationMethod.noCalibration, 48: CalibrationMethod.noCalibration, 49: CalibrationMethod.noCalibration, 50: CalibrationMethod.noCalibration, 51: CalibrationMethod.noCalibration, 
																						 52: CalibrationMethod.noCalibration, 53: CalibrationMethod.noCalibration, 54: CalibrationMethod.noCalibration, 55: CalibrationMethod.noCalibration, 56: CalibrationMethod.noCalibration, 
																						 57: CalibrationMethod.noCalibration, 58: CalibrationMethod.noCalibration, 59: CalibrationMethod.noCalibration, 60: CalibrationMethod.noCalibration, 61: CalibrationMethod.noCalibration,
																						 62: CalibrationMethod.noCalibration, 63: CalibrationMethod.noCalibration, 64: CalibrationMethod.noCalibration, 65: CalibrationMethod.noCalibration, 66: CalibrationMethod.noCalibration, 
																						 67: CalibrationMethod.noCalibration, 68: CalibrationMethod.noCalibration, 69: CalibrationMethod.noCalibration, 70: CalibrationMethod.noCalibration, 71: CalibrationMethod.noCalibration, 
																						 72: CalibrationMethod.noCalibration, 73: CalibrationMethod.noCalibration, 74: CalibrationMethod.noCalibration, 75: CalibrationMethod.noCalibration, 76: CalibrationMethod.noCalibration, 
																						 77: CalibrationMethod.noCalibration, 78: CalibrationMethod.noCalibration, 79: CalibrationMethod.noCalibration, 80: CalibrationMethod.noCalibration, 81: CalibrationMethod.noCalibration,
																						 82: CalibrationMethod.noCalibration, 83: CalibrationMethod.noCalibration, 84: CalibrationMethod.noCalibration, 85: CalibrationMethod.noCalibration, 86: CalibrationMethod.noCalibration,
																						 87: CalibrationMethod.noCalibration, 88: CalibrationMethod.noCalibration, 89: CalibrationMethod.noCalibration, 90: CalibrationMethod.noCalibration, 91: CalibrationMethod.noCalibration, 
																						 92: CalibrationMethod.noCalibration, 93: CalibrationMethod.noCalibration, 94: CalibrationMethod.noCalibration, 95: CalibrationMethod.noCalibration, 96: CalibrationMethod.noCalibration,
																						 97: CalibrationMethod.noCalibration, 98: CalibrationMethod.noCalibration, 99: CalibrationMethod.noCalibration, 100: CalibrationMethod.noCalibration, 
																						 101: CalibrationMethod.noCalibration, 102: CalibrationMethod.noCalibration, 103: CalibrationMethod.noCalibration, 104: CalibrationMethod.noCalibration, 
																						 105: CalibrationMethod.noCalibration, 106: CalibrationMethod.noCalibration, 107: CalibrationMethod.noCalibration, 108: CalibrationMethod.noCalibration, 
																						 109: CalibrationMethod.noCalibration, 110: CalibrationMethod.noCalibration, 111: CalibrationMethod.noCalibration},
																   'ULOQ': {0: numpy.nan, 1: numpy.nan, 2: numpy.nan, 3: numpy.nan, 4: numpy.nan, 5: numpy.nan, 6: numpy.nan, 7: numpy.nan, 8: numpy.nan, 9: numpy.nan, 10: numpy.nan, 11: numpy.nan, 12: numpy.nan, 13: numpy.nan, 14: numpy.nan, 15: numpy.nan, 16: numpy.nan, 17: numpy.nan, 18: numpy.nan, 19: numpy.nan, 20: numpy.nan,
																			21: numpy.nan, 22: numpy.nan, 23: numpy.nan, 24: numpy.nan, 25: numpy.nan, 26: numpy.nan, 27: numpy.nan, 28: numpy.nan, 29: numpy.nan, 30: numpy.nan, 31: numpy.nan, 32: numpy.nan, 33: numpy.nan, 34: numpy.nan, 35: numpy.nan, 36: numpy.nan, 37: numpy.nan, 38: numpy.nan, 39: numpy.nan, 40: numpy.nan,
																			41: numpy.nan, 42: numpy.nan, 43: numpy.nan, 44: numpy.nan, 45: numpy.nan, 46: numpy.nan, 47: numpy.nan, 48: numpy.nan, 49: numpy.nan, 50: numpy.nan, 51: numpy.nan, 52: numpy.nan, 53: numpy.nan, 54: numpy.nan, 55: numpy.nan, 56: numpy.nan, 57: numpy.nan, 58: numpy.nan, 59: numpy.nan, 60: numpy.nan,
																			61: numpy.nan, 62: numpy.nan, 63: numpy.nan, 64: numpy.nan, 65: numpy.nan, 66: numpy.nan, 67: numpy.nan, 68: numpy.nan, 69: numpy.nan, 70: numpy.nan, 71: numpy.nan, 72: numpy.nan, 73: numpy.nan, 74: numpy.nan, 75: numpy.nan, 76: numpy.nan, 77: numpy.nan, 78: numpy.nan, 79: numpy.nan, 80: numpy.nan,
																			81: numpy.nan, 82: numpy.nan, 83: numpy.nan, 84: numpy.nan, 85: numpy.nan, 86: numpy.nan, 87: numpy.nan, 88: numpy.nan, 89: numpy.nan, 90: numpy.nan, 91: numpy.nan, 92: numpy.nan, 93: numpy.nan, 94: numpy.nan, 95: numpy.nan, 96: numpy.nan, 97: numpy.nan, 98: numpy.nan, 99: numpy.nan, 100: numpy.nan,
																			101: numpy.nan, 102: numpy.nan, 103: numpy.nan, 104: numpy.nan, 105: numpy.nan, 106: numpy.nan, 107: numpy.nan, 108: numpy.nan, 109: numpy.nan, 110: numpy.nan, 111: numpy.nan}})
		
		self.expectedBILISA['intensityData'] = numpy.array([[303.99, 278.77, 134.23, 51.30, 130.78, 28.58, 168.37, 2.62, 1.29, 3061.47, 600.04, 385.28, 2198.55, 541.55, 219.90, 160.24, 219.88, 421.21, 527.26,
															 188.71, 31.67, 70.47, 29.35, 69.77, 48.03, 28.61, 13.80, 37.52, 5.17, 58.43, 19.18, 86.89, 79.48, 134.42, 31.89, 33.00, 21.19, 120.91, 53.10, 36.78,
															 43.43, 42.21, 8.07, 10.19, 9.73, 15.17, 26.37, 5.00, 3.31, 4.81, 6.42, 11.15, 2.14, 8.83, 9.29, 14.47, 21.27, 4.98, 26.39, 8.74, 7.90, 10.56, 9.31,
															 10.95, 43.23, 13.58, 6.15, 10.22, 24.65, 32.38, 13.28, 4.93, 2.19, 2.93, 6.57, 5.57, 28.40, 9.52, 5.87, 8.10, 14.38, 18.09, 29.78, 12.09, 8.81, 12.09,
															 23.17, 29.00, 12.02, 5.00, 4.94, 7.62, 21.93, 7.90, 6.02, 12.53, 1.63, 1.07, 0.99, 2.51, 30.42, 15.35, 14.25, 20.86, 36.70, 22.30, 20.29, 56.70, 5.57,
															 5.21, 7.05, 13.36],
															[239.60, 269.23, 122.38, 105.63, 259.63, 36.61, 115.48, 1.16, 0.44, 2099.65, 313.87, 161.60, 1625.62, 403.79, 270.04, 223.23, 95.91, 178.89, 447.90,
															 155.48, 29.61, 40.00, 37.33, 33.08, 16.40, 17.22, 4.84, 36.79, 25.04, 39.47, 14.82, 75.92, 156.96, 268.55, 37.16, 17.26, 8.89, 89.40, 77.93, 28.63,
															 24.82, 18.14, 3.50, 9.10, 5.27, 7.51, 9.42, 2.12, 5.31, 2.52, 3.36, 3.69, 0.58, 12.31, 7.41, 8.44, 8.62, 2.23, 14.41, 5.78, 5.12, 3.80, 3.66, 8.98,
															 34.59, 24.20, 17.40, 5.48, 10.86, 29.79, 10.64, 7.39, 4.63, 2.70, 2.40, 6.56, 21.69, 14.12, 10.44, 4.62, 6.86, 17.80, 22.21, 14.85, 12.28, 5.27,
															 9.84, 24.63, 19.74, 6.48, 4.90, 5.76, 66.38, 15.38, 9.84, 13.85, 13.54, 2.76, 2.28, 1.89, 86.48, 27.48, 19.99, 23.69, 118.96, 44.48, 34.61, 75.47,
															 12.47, 7.49, 5.95, 10.85],
															[187.80, 255.49, 127.88, 87.24, 209.02, 34.79, 115.53, 1.47, 0.55, 2100.58, 292.37, 187.87, 1641.50, 374.48, 265.93, 208.41, 157.33, 244.85, 353.33,
															 108.04, 20.60, 39.78, 27.61, 31.11, 21.79, 14.55, 6.34, 35.79, 17.81, 31.84, 11.99, 77.41, 124.57, 215.80, 35.71, 16.08, 10.33, 90.28, 40.30, 18.27,
															 20.79, 21.25, 4.98, 4.77, 3.70, 6.28, 11.69, 3.19, 2.49, 1.78, 2.73, 4.65, 1.07, 6.21, 4.62, 7.02, 10.36, 3.24, 14.25, 5.69, 5.13, 4.95, 4.33, 6.56,
															 33.96, 24.26, 16.70, 11.64, 16.48, 23.20, 10.39, 7.47, 5.13, 3.90, 4.32, 5.19, 21.13, 14.06, 10.27, 7.61, 9.77, 14.18, 20.60, 14.63, 11.46, 8.65,
															 13.47, 19.43, 13.18, 5.01, 4.22, 5.34, 43.88, 14.40, 10.94, 15.83, 8.30, 2.34, 2.17, 2.36, 56.27, 23.89, 20.06, 24.51, 76.66, 34.29, 32.73, 71.17,
															 8.06, 6.14, 7.05, 13.24],
															[514.82, 166.43, 52.04, 47.19, 149.56, 40.44, 113.07, 1.10, 0.76, 2055.83, 794.53, 330.35, 1088.28, 145.20, 97.18, 0.00, 40.23, 424.38, 508.46, 396.61,
															 75.99, 45.70, 36.00, 85.59, 40.87, 42.57, 11.95, 5.25, 0.00, 102.07, 26.31, 37.72, 81.19, 162.85, 43.90, 43.70, 18.17, 59.85, 205.21, 66.13, 59.66,
															 56.03, 11.33, 23.74, 9.96, 15.55, 25.32, 6.38, 12.59, 5.49, 8.23, 11.16, 3.25, 30.90, 15.63, 19.77, 24.58, 7.05, 17.58, 3.61, 5.51, 5.04, 5.48, 9.57,
															 1.47, 1.78, 0.00, 0.00, 27.70, 30.82, 0.52, 0.00, 0.00, 0.00, 5.94, 5.75, 7.22, 1.29, 0.00, 0.49, 15.55, 17.43, 7.99, 5.34, 0.00, 2.21, 23.34, 27.96,
															 12.40, 6.12, 7.03, 11.29, 12.06, 7.94, 7.11, 17.50, 0.00, 0.10, 1.24, 3.14, 22.66, 18.04, 19.24, 29.38, 30.71, 27.67, 27.61, 74.92, 6.10, 7.30, 10.44,
															 20.76],
															[166.91, 199.47, 83.08, 90.03, 211.32, 32.01, 75.96, 0.92, 0.36, 1381.23, 216.69, 86.13, 993.41, 253.03, 176.79, 122.76, 58.32, 110.32, 259.39, 103.44,
															 20.53, 23.65, 28.13, 22.04, 8.67, 12.31, 2.60, 21.08, 18.48, 26.89, 8.63, 51.76, 124.88, 217.79, 33.11, 11.92, 4.74, 54.64, 55.80, 13.89, 13.13, 14.86,
															 4.79, 6.72, 1.58, 2.96, 6.38, 2.94, 3.35, 1.06, 1.66, 2.41, 0.76, 8.26, 3.44, 4.36, 6.56, 3.03, 8.63, 3.28, 3.34, 2.20, 1.74, 4.10, 23.59, 16.78, 10.28,
															 5.85, 7.94, 17.42, 6.49, 4.94, 3.86, 2.88, 2.18, 4.06, 15.05, 9.82, 6.56, 4.57, 4.94, 10.80, 13.92, 9.72, 6.75, 3.21, 6.07, 14.27, 13.51, 5.36, 4.30,
															 4.62, 45.29, 16.74, 12.14, 13.66, 7.83, 2.13, 1.80, 1.69, 57.46, 26.51, 20.05, 19.86, 78.34, 34.05, 35.28, 65.63, 7.30, 5.55, 6.61, 10.53],
															[204.69, 239.17, 130.86, 64.56, 167.89, 36.08, 125.49, 2.03, 0.75, 2281.80, 342.89, 174.71, 1757.31, 272.82, 218.27, 148.27, 173.50, 386.36, 529.62,
															 139.69, 22.96, 35.91, 22.03, 38.79, 20.55, 17.49, 6.31, 32.31, 9.92, 40.11, 12.37, 76.70, 92.02, 172.05, 37.72, 18.86, 9.61, 96.65, 56.77, 24.52, 26.77,
															 23.35, 4.81, 7.97, 5.51, 8.63, 12.70, 3.53, 3.69, 2.71, 3.72, 4.97, 1.20, 9.57, 6.55, 9.15, 11.13, 3.76, 11.17, 4.31, 4.10, 4.60, 5.13, 7.00, 23.72,
															 18.63, 10.87, 12.83, 28.48, 34.47, 7.15, 5.72, 3.32, 3.79, 6.68, 7.44, 15.16, 10.79, 7.23, 8.07, 15.70, 19.11, 15.00, 12.00, 8.15, 9.54, 21.25, 29.13,
															 8.30, 3.91, 4.13, 6.26, 21.90, 10.30, 9.98, 19.84, 3.20, 1.25, 1.79, 3.21, 28.01, 17.53, 18.52, 28.79, 35.95, 25.15, 29.62, 79.86, 4.43, 4.86, 7.49, 19.30],
															[251.29, 270.78, 132.76, 85.46, 238.14, 48.81, 118.49, 1.55, 0.50, 2154.51, 360.65, 220.68, 1552.59, 308.27, 238.96, 189.59, 262.05, 342.26, 274.46,
															 162.77, 33.72, 34.75, 33.10, 44.87, 27.25, 20.40, 7.89, 32.02, 16.56, 47.76, 15.16, 79.41, 138.49, 248.82, 49.04, 19.83, 12.14, 85.39, 72.38, 26.27,
															 27.50, 27.34, 6.96, 9.94, 4.90, 8.39, 14.57, 4.28, 4.83, 2.31, 3.45, 6.07, 1.99, 11.53, 6.60, 9.47, 13.42, 5.09, 11.78, 4.68, 5.03, 4.78, 4.02, 4.65,
															 28.70, 20.90, 15.38, 21.47, 25.66, 17.66, 8.56, 6.29, 3.62, 5.34, 5.80, 3.91, 18.60, 12.00, 9.81, 12.87, 14.66, 11.49, 16.95, 13.14, 10.43, 14.41,
															 18.82, 15.09, 12.65, 6.21, 6.34, 8.18, 30.84, 15.65, 15.68, 21.02, 5.59, 2.80, 3.70, 4.30, 45.40, 28.79, 30.18, 36.02, 61.25, 41.18, 49.26, 95.45,
															 7.92, 8.60, 11.86, 20.59],
															[91.57, 256.88, 147.04, 100.98, 222.79, 30.04, 108.71, 1.46, 0.49, 1976.65, 121.70, 112.51, 1729.08, 342.32, 294.64, 251.22, 176.77, 242.00, 369.01,
															 30.24, 6.39, 35.85, 25.78, 7.69, 10.73, 4.79, 2.96, 40.84, 24.12, 10.09, 6.56, 85.46, 130.87, 229.15, 30.97, 6.69, 6.19, 95.09, 7.14, 3.50, 6.05,
															 9.23, 3.24, 0.00, 0.27, 0.93, 4.80, 1.96, 0.00, 0.02, 0.29, 1.25, 0.06, 0.87, 0.93, 1.91, 4.68, 1.76, 11.06, 5.35, 4.93, 4.81, 4.03, 6.10, 33.38,
															 29.24, 22.75, 16.19, 18.35, 25.51, 9.90, 9.13, 7.73, 5.90, 5.23, 6.25, 20.04, 16.59, 13.25, 9.77, 10.60, 15.37, 18.83, 16.20, 13.82, 9.72, 13.31,
															 20.29, 13.82, 4.73, 3.45, 3.70, 55.30, 17.06, 11.57, 14.01, 11.73, 2.58, 2.04, 1.60, 67.00, 25.40, 17.66, 18.77, 94.23, 36.41, 31.75, 65.78, 8.51,
															 5.21, 5.51, 8.69],
															[108.74, 228.04, 105.26, 105.40, 216.12, 36.37, 85.19, 1.00, 0.39, 1548.99, 159.10, 110.49, 1276.59, 259.42, 236.69, 120.20, 22.71, 168.51, 430.07,
															 49.30, 9.98, 26.15, 22.15, 14.90, 10.59, 7.71, 3.22, 25.35, 20.77, 14.68, 7.44, 62.20, 138.39, 226.96, 37.08, 8.75, 6.08, 70.21, 14.11, 4.67, 8.49,
															 14.38, 4.73, 0.73, 0.52, 2.01, 7.11, 2.84, 0.71, 0.33, 0.99, 2.63, 0.65, 1.27, 1.31, 3.18, 6.74, 2.72, 8.42, 3.71, 3.88, 2.39, 2.50, 6.35, 24.33, 23.79,
															 10.81, 3.11, 12.16, 28.46, 7.11, 7.46, 4.29, 2.66, 3.44, 6.66, 15.16, 13.48, 6.86, 2.86, 6.98, 16.93, 14.27, 13.02, 6.61, 1.25, 9.27, 23.65, 11.80,
															 4.63, 3.38, 2.50, 51.33, 20.46, 14.84, 14.66, 10.00, 3.27, 2.35, 1.56, 63.58, 30.66, 23.48, 20.49, 87.62, 37.13, 37.54, 58.50, 8.56, 6.86, 7.99, 10.39],
															[162.57, 233.86, 132.04, 82.05, 197.44, 38.08, 104.29, 1.61, 0.53, 1896.32, 256.88, 126.96, 1501.86, 203.83, 237.30, 195.00, 219.55, 320.63, 350.75,
															 112.22, 18.11, 25.72, 22.26, 25.99, 13.68, 14.02, 4.07, 33.08, 15.57, 32.75, 10.04, 73.98, 110.39, 207.92, 38.79, 14.13, 6.98, 82.60, 48.53, 17.41,
															 19.12, 19.68, 4.84, 4.56, 2.59, 5.15, 8.73, 2.87, 2.80, 1.56, 2.26, 3.58, 0.98, 6.95, 4.46, 6.61, 8.92, 3.28, 6.75, 3.38, 3.68, 3.55, 3.18, 5.16, 18.81,
															 22.81, 17.84, 20.18, 25.38, 24.90, 5.29, 6.70, 5.03, 5.71, 6.13, 5.27, 12.04, 12.49, 10.17, 11.32, 13.76, 14.35, 11.21, 13.05, 10.72, 12.07, 17.63,
															 19.29, 9.48, 4.25, 3.74, 4.82, 31.83, 14.53, 11.85, 19.77, 6.16, 2.13, 2.37, 3.15, 41.12, 22.89, 19.96, 27.05, 58.15, 30.28, 34.20, 81.22, 6.13, 5.84,
															 7.66, 17.18]])
		self.expectedBILISA['expectedConcentration'] = pandas.DataFrame(None, index=list(self.expectedBILISA['sampleMetadata'].index), columns=self.expectedBILISA['featureMetadata']['Feature Name'].tolist())

		# Calibration
		self.expectedBILISA['calibIntensityData'] = numpy.ndarray((0, self.expectedBILISA['featureMetadata'].shape[0]))
		self.expectedBILISA['calibSampleMetadata'] = pandas.DataFrame(None, columns=self.expectedBILISA['sampleMetadata'].columns)
		self.expectedBILISA['calibSampleMetadata']['Metadata Available'] = False
		self.expectedBILISA['calibFeatureMetadata'] = pandas.DataFrame({'Feature Name': self.expectedBILISA['featureMetadata']['Feature Name'].tolist()})
		self.expectedBILISA['calibExpectedConcentration'] = pandas.DataFrame(None, columns=self.expectedBILISA['featureMetadata']['Feature Name'].tolist())
		# Excluded
		self.expectedBILISA['sampleMetadataExcluded'] = []
		self.expectedBILISA['featureMetadataExcluded'] = []
		self.expectedBILISA['intensityDataExcluded'] = []
		self.expectedBILISA['expectedConcentrationExcluded'] = []
		self.expectedBILISA['excludedFlag'] = []
		# Attributes
		tmpDataset = nPYc.TargetedDataset('', fileType='empty')
		self.expectedBILISA['Attributes'] = copy.deepcopy(self.expectedQuantUR['Attributes'])
		self.expectedBILISA['Attributes']['methodName'] = 'NMR Bruker - BI-LISA'

	def test_loadBrukerXMLDataset(self):

		with self.subTest(msg='Basic import BrukerQuant-UR with matching fileNamePattern'):
			expected = copy.deepcopy(self.expectedQuantUR)

			# Generate
			result = nPYc.NMRTargetedDataset(self.datapathQuantUR, fileType='Bruker Quantification', 
											sop='BrukerQuant-UR', fileNamePattern='.*?urine_quant_report_b\.xml$', 
											unit='mmol/mol Crea')
			# Remove path from sampleMetadata
			result.sampleMetadata.drop(['Path'], axis=1, inplace=True)
			result.calibration['calibSampleMetadata'].drop(['Path'], axis=1, inplace=True)

			# Need to sort samples as different OS have different path order
			result.sampleMetadata.sort_values('Sample Base Name', inplace=True)
			sortIndex = result.sampleMetadata.index.values
			result.intensityData = result.intensityData[sortIndex, :]
			result.expectedConcentration = result.expectedConcentration.loc[sortIndex,:]
			result.sampleMetadata = result.sampleMetadata.reset_index(drop=True)
			result.expectedConcentration = result.expectedConcentration.reset_index(drop=True)

			# Test
			assert_frame_equal(expected['sampleMetadata'].reindex(sorted(expected['sampleMetadata']), axis=1), result.sampleMetadata.reindex(sorted(result.sampleMetadata), axis=1))
			assert_frame_equal(expected['featureMetadata'].reindex(sorted(expected['featureMetadata']), axis=1), result.featureMetadata.reindex(sorted(result.featureMetadata), axis=1))
			numpy.testing.assert_array_almost_equal(expected['intensityData'], result._intensityData)
			assert_frame_equal(expected['expectedConcentration'].reindex(sorted(expected['expectedConcentration']), axis=1), result.expectedConcentration.reindex(sorted(result.expectedConcentration), axis=1))
			# Calibration
			assert_frame_equal(expected['calibSampleMetadata'].reindex(sorted(expected['calibSampleMetadata']), axis=1), result.calibration['calibSampleMetadata'].reindex(sorted(result.calibration['calibSampleMetadata']), axis=1))
			assert_frame_equal(expected['calibFeatureMetadata'].reindex(sorted(expected['calibFeatureMetadata']), axis=1), result.calibration['calibFeatureMetadata'].reindex(sorted(result.calibration['calibFeatureMetadata']), axis=1))
			numpy.testing.assert_array_almost_equal(expected['calibIntensityData'], result.calibration['calibIntensityData'])
			assert_frame_equal(expected['calibExpectedConcentration'].reindex(sorted(expected['calibExpectedConcentration']), axis=1), result.calibration['calibExpectedConcentration'].reindex(sorted(result.calibration['calibExpectedConcentration']), axis=1), check_index_type=False)
			# Attributes, no check of 'Log'
			self.assertEqual(len(expected['Attributes'].keys()), len(result.Attributes.keys()) - 1)
			for i in expected['Attributes']:
				self.assertEqual(expected['Attributes'][i], result.Attributes[i])

		with self.subTest(msg='Basic import BrukerQuant-UR with implicit fileNamePattern from SOP'):
			expected = copy.deepcopy(self.expectedQuantUR)
			# Generate
			
			result = nPYc.NMRTargetedDataset(self.datapathQuantUR, fileType='Bruker Quantification', sop='BrukerQuant-UR', unit='mmol/mol Crea')
			# Remove path from sampleMetadata
			result.sampleMetadata.drop(['Path'], axis=1, inplace=True)
			result.calibration['calibSampleMetadata'].drop(['Path'], axis=1, inplace=True)

			# Need to sort samples as different OS have different path order
			result.sampleMetadata.sort_values('Sample Base Name', inplace=True)
			sortIndex = result.sampleMetadata.index.values
			result.intensityData = result.intensityData[sortIndex, :]
			result.expectedConcentration = result.expectedConcentration.loc[sortIndex,:]
			result.sampleMetadata = result.sampleMetadata.reset_index(drop=True)
			result.expectedConcentration = result.expectedConcentration.reset_index(drop=True)

			# Test
			assert_frame_equal(expected['sampleMetadata'].reindex(sorted(expected['sampleMetadata']), axis=1), result.sampleMetadata.reindex(sorted(result.sampleMetadata), axis=1))
			assert_frame_equal(expected['featureMetadata'].reindex(sorted(expected['featureMetadata']), axis=1), result.featureMetadata.reindex(sorted(result.featureMetadata), axis=1))
			numpy.testing.assert_array_almost_equal(expected['intensityData'], result._intensityData)
			assert_frame_equal(expected['expectedConcentration'].reindex(sorted(expected['expectedConcentration']), axis=1), result.expectedConcentration.reindex(sorted(result.expectedConcentration), axis=1))
			# Calibration
			assert_frame_equal(expected['calibSampleMetadata'].reindex(sorted(expected['calibSampleMetadata']), axis=1), result.calibration['calibSampleMetadata'].reindex(sorted(result.calibration['calibSampleMetadata']), axis=1))
			assert_frame_equal(expected['calibFeatureMetadata'].reindex(sorted(expected['calibFeatureMetadata']), axis=1), result.calibration['calibFeatureMetadata'].reindex(sorted(result.calibration['calibFeatureMetadata']), axis=1))
			numpy.testing.assert_array_almost_equal(expected['calibIntensityData'], result.calibration['calibIntensityData'])
			assert_frame_equal(expected['calibExpectedConcentration'].reindex(sorted(expected['calibExpectedConcentration']), axis=1), result.calibration['calibExpectedConcentration'].reindex(sorted(result.calibration['calibExpectedConcentration']), axis=1), check_index_type=False)
			# Attributes, no check of 'Log'
			self.assertEqual(len(expected['Attributes'].keys()), len(result.Attributes.keys()) - 1)
			for i in expected['Attributes']:
				self.assertEqual(expected['Attributes'][i], result.Attributes[i])
	
	def test_loadBrukerBIQUANTXMLDataset(self):

		expected = copy.deepcopy(self.expectedQuantPS)
		expected["sampleMetadata"] = expected["sampleMetadata"].reset_index(drop=True)
		expected["featureMetadata"] = expected["featureMetadata"].reset_index(drop=True)
		expected["expectedConcentration"] = expected["expectedConcentration"].reset_index(drop=True)

		with self.subTest(msg='Basic import BrukerQuant-PS with matching fileNamePattern'):

			result = nPYc.NMRTargetedDataset(self.datapathQuantPS, fileType='Bruker Quantification', sop='BrukerBI-QUANT-PS', fileNamePattern='.*?plasma_quant_report\.xml$')

			result.sampleMetadata.drop(['Path'], axis=1, inplace=True)
			result.calibration['calibSampleMetadata'].drop(['Path'], axis=1, inplace=True)

			# Need to sort samples as different OS have different path order
			result.sampleMetadata.sort_values('Sample Base Name', inplace=True)
			sortIndex = result.sampleMetadata.index.values
			result.intensityData = result.intensityData[sortIndex, :]
			result.expectedConcentration = result.expectedConcentration.loc[sortIndex,:]
			result.sampleMetadata = result.sampleMetadata.reset_index(drop=True)
			result.expectedConcentration = result.expectedConcentration.reset_index(drop=True)

			assert_frame_equal(expected['sampleMetadata'].reindex(sorted(expected['sampleMetadata']), axis=1), result.sampleMetadata.reindex(sorted(result.sampleMetadata), axis=1))
			assert_frame_equal(expected['featureMetadata'].reindex(sorted(expected['featureMetadata']), axis=1), result.featureMetadata.reindex(sorted(result.featureMetadata), axis=1))
			numpy.testing.assert_array_almost_equal(expected['intensityData'], result._intensityData)
			assert_frame_equal(expected['expectedConcentration'].reindex(sorted(expected['expectedConcentration']), axis=1), result.expectedConcentration.reindex(sorted(result.expectedConcentration), axis=1))

		with self.subTest(msg='Basic import BrukerQuant-UR with implicit fileNamePattern from SOP'):

			result = nPYc.NMRTargetedDataset(self.datapathQuantPS, fileType='Bruker Quantification', sop='BrukerBI-QUANT-PS')
			result.sampleMetadata.drop(['Path'], axis=1, inplace=True)
			result.calibration['calibSampleMetadata'].drop(['Path'], axis=1, inplace=True)

			# Need to sort samples as different OS have different path order
			result.sampleMetadata.sort_values('Sample Base Name', inplace=True)
			sortIndex = result.sampleMetadata.index.values
			result.intensityData = result.intensityData[sortIndex, :]
			result.expectedConcentration = result.expectedConcentration.loc[sortIndex,:]
			result.sampleMetadata = result.sampleMetadata.reset_index(drop=True)
			result.expectedConcentration = result.expectedConcentration.reset_index(drop=True)

			assert_frame_equal(expected['sampleMetadata'].reindex(sorted(expected['sampleMetadata']), axis=1), result.sampleMetadata.reindex(sorted(result.sampleMetadata), axis=1))
			assert_frame_equal(expected['featureMetadata'].reindex(sorted(expected['featureMetadata']), axis=1), result.featureMetadata.reindex(sorted(result.featureMetadata), axis=1))
			numpy.testing.assert_array_almost_equal(expected['intensityData'], result._intensityData)
			assert_frame_equal(expected['expectedConcentration'].reindex(sorted(expected['expectedConcentration']), axis=1), result.expectedConcentration.reindex(sorted(result.expectedConcentration), axis=1))



	@unittest.mock.patch('sys.stdout', new_callable=io.StringIO)
	def test_loadBrukerXMLDataset_warnDuplicates(self, mock_stdout):

		with self.subTest(msg='Import duplicated features (BI-LISA), Raises warning if features are duplicated'):
			expected = copy.deepcopy(self.expectedBILISA)

			# Raise and check warning
			with warnings.catch_warnings(record=True) as w:
				# Cause all warnings to always be triggered.

				warnings.simplefilter("always")
				result = nPYc.NMRTargetedDataset(self.datapathBILISA, fileType='Bruker Quantification', sop='BrukerBI-LISA', fileNamePattern='.*?results\.xml$')
				# check each warning
				self.assertEqual(len(w), 1)
				assert issubclass(w[0].category, UserWarning)
				assert "The following features are present more than once, only the first occurence will be kept:" in str(w[0].message)

				# Check fuplicated features were filtered
				result.sampleMetadata.drop(['Path'], axis=1, inplace=True)
				result.calibration['calibSampleMetadata'].drop(['Path'], axis=1, inplace=True)

				# Need to sort samples as different OS have different path order
				result.sampleMetadata.sort_values('Sample Base Name', inplace=True)
				sortIndex = result.sampleMetadata.index.values
				result.intensityData = result.intensityData[sortIndex, :]
				result.expectedConcentration = result.expectedConcentration.loc[sortIndex, :]
				result.sampleMetadata = result.sampleMetadata.reset_index(drop=True)
				result.expectedConcentration = result.expectedConcentration.reset_index(drop=True)

				# Test
				assert_frame_equal(expected['sampleMetadata'].reindex(sorted(expected['sampleMetadata']), axis=1), result.sampleMetadata.reindex(sorted(result.sampleMetadata), axis=1))
				assert_frame_equal(expected['featureMetadata'].reindex(sorted(expected['featureMetadata']), axis=1), result.featureMetadata.reindex(sorted(result.featureMetadata), axis=1))
				numpy.testing.assert_array_almost_equal(expected['intensityData'], result._intensityData)
				assert_frame_equal(expected['expectedConcentration'].reindex(sorted(expected['expectedConcentration']), axis=1), result.expectedConcentration.reindex(sorted(result.expectedConcentration), axis=1))
				# Calibration
				assert_frame_equal(expected['calibSampleMetadata'].reindex(sorted(expected['calibSampleMetadata']), axis=1), result.calibration['calibSampleMetadata'].reindex(sorted(result.calibration['calibSampleMetadata']), axis=1))
				assert_frame_equal(expected['calibFeatureMetadata'].reindex(sorted(expected['calibFeatureMetadata']), axis=1), result.calibration['calibFeatureMetadata'].reindex(sorted(result.calibration['calibFeatureMetadata']), axis=1))
				numpy.testing.assert_array_almost_equal(expected['calibIntensityData'], result.calibration['calibIntensityData'])
				assert_frame_equal(expected['calibExpectedConcentration'].reindex(sorted(expected['calibExpectedConcentration']), axis=1), result.calibration['calibExpectedConcentration'].reindex(sorted(result.calibration['calibExpectedConcentration']), axis=1), check_index_type=False)
				# Attributes, no check of 'Log'
				self.assertEqual(len(expected['Attributes'].keys()), len(result.Attributes.keys()) - 1)
				for i in expected['Attributes']:
					self.assertEqual(expected['Attributes'][i], result.Attributes[i])


	def test_brukerXML_raises(self):

		with self.subTest(msg='Raises TypeError if `fileNamePattern` is not a str'):
			self.assertRaises(TypeError, lambda: nPYc.NMRTargetedDataset(self.datapathQuantUR, fileType='Bruker Quantification', sop='BrukerQuant-UR', fileNamePattern=5, unit='mmol/mol Crea'))

		with self.subTest(msg='Raises TypeError if `pdata` is not am int'):
			self.assertRaises(TypeError, lambda: nPYc.NMRTargetedDataset(self.datapathQuantUR, fileType='Bruker Quantification', sop='BrukerQuant-UR', pdata='notAnInt', fileNamePattern='.*?urine_quant_report_b\.xml$', unit='mmol/mol Crea'))
			self.assertRaises(TypeError, lambda: nPYc.NMRTargetedDataset(self.datapathQuantUR, fileType='Bruker Quantification', sop='BrukerQuant-UR', pdata=1.0, fileNamePattern='.*?urine_quant_report_b\.xml$', unit='mmol/mol Crea'))

		with self.subTest(msg='Raises TypeError if `unit` is not None or a str'):
			self.assertRaises(TypeError, lambda: nPYc.NMRTargetedDataset(self.datapathQuantUR, fileType='Bruker Quantification', sop='BrukerQuant-UR', unit=5, fileNamePattern='.*?urine_quant_report_b\.xml$'))

		with self.subTest(msg='Raises ValueError if `unit` is not one of the unit in the input data'):
			self.assertRaises(ValueError, lambda: nPYc.NMRTargetedDataset(self.datapathQuantUR, fileType='Bruker Quantification', sop='BrukerQuant-UR', unit='notAnExistingUnit', fileNamePattern='.*?urine_quant_report_b\.xml$'))


	@unittest.mock.patch('sys.stdout', new_callable=io.StringIO)
	def test_loadlims(self, mock_stdout):
		

			
		with self.subTest(msg='UnitTest1'):

			dataset = nPYc.NMRTargetedDataset(self.datapathQuantUR, fileType='Bruker Quantification', sop='BrukerQuant-UR', fileNamePattern='.*?urine_quant_report_b\.xml$', unit='mmol/mol Crea')

			limspath = os.path.join('..', '..', 'npc-standard-project', 'Derived_Worklists', 'UnitTest1_NMR_urine_PCSOP.011.csv')
			dataset.addSampleInfo(filePath=limspath, descriptionFormat='NPC LIMS')

			dataset.sampleMetadata.sort_values('Sample Base Name', inplace=True)
			sortIndex = dataset.sampleMetadata.index.values
			dataset.intensityData = dataset.intensityData[sortIndex, :]
			dataset.sampleMetadata = dataset.sampleMetadata.reset_index(drop=True)

			expected = pandas.Series(['UT1_S1_u1', 'UT1_S2_u1', 'UT1_S3_u1', 'UT1_S4_u1', 'UT1_S4_u2', 'UT1_S4_u3', 'UT1_S4_u4', 'External Reference Sample', 'Study Pool Sample'],
										name='Sample ID',
										dtype='str')

			pandas.testing.assert_series_equal(dataset.sampleMetadata['Sample ID'], expected)

			expected = pandas.Series(['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9'],
										name='Sample position',
										dtype='str')

			pandas.testing.assert_series_equal(dataset.sampleMetadata['Sample position'], expected)


		with self.subTest(msg='UnitTest3'):

			with warnings.catch_warnings():
				warnings.simplefilter('ignore', UserWarning)
				dataset = nPYc.NMRTargetedDataset(self.datapathBILISA, fileType='Bruker Quantification', sop='BrukerBI-LISA', fileNamePattern='.*?results\.xml$')

			limspath = os.path.join('..', '..', 'npc-standard-project', 'Derived_Worklists', 'UnitTest3_NMR_serum_PCSOP.012.csv')
			dataset.addSampleInfo(filePath=limspath, descriptionFormat='NPC LIMS')

			dataset.sampleMetadata.sort_values('Sample Base Name', inplace=True)
			sortIndex = dataset.sampleMetadata.index.values
			dataset.intensityData = dataset.intensityData[sortIndex, :]
			dataset.sampleMetadata = dataset.sampleMetadata.reset_index(drop=True)

			expected = pandas.Series(['UT3_S8', 'UT3_S7', 'UT3_S6', 'UT3_S5', 'UT3_S4',
										'UT3_S3', 'UT3_S2', 'External Reference Sample',
										'Study Pool Sample', 'UT3_S1'],
										name='Sample ID',
										dtype='str')

			pandas.testing.assert_series_equal(dataset.sampleMetadata['Sample ID'], expected)

			expected = pandas.Series(['A1', 'A2', 'A3', 'A4', 'A5',
									'A6', 'A7', 'A8', 'A9', 'A10'],
										name='Sample position',
										dtype='str')

			pandas.testing.assert_series_equal(dataset.sampleMetadata['Sample position'], expected)


class test_targeteddataset_exportdataset(unittest.TestCase):
	"""
	Test exportDataset, _exportCSV, _exportUnifiedCSV
	"""
	def setUp(self):
		self.targeted = nPYc.TargetedDataset('', fileType='empty')
		self.targeted.name = 'UnitTest'
		self.targeted.sampleMetadata = pandas.DataFrame({'Sample File Name': ['UnitTest_targeted_file_004','UnitTest_targeted_file_005','UnitTest_targeted_file_006', 'UnitTest_targeted_file_007', 'UnitTest_targeted_file_008', 'UnitTest_targeted_file_009'],'TargetLynx Sample ID': [4, 5, 6, 7, 8, 9], 'MassLynx Row ID': [4, 5, 6, 7, 8, 9], 'Sample Name': ['Sample-LLOQ', 'Sample-Fine', 'Sample-ULOQ', 'Blank', 'QC', 'Other'],'Sample Type': ['Analyte', 'Analyte', 'Analyte', 'Blank','QC', 'Solvent'], 'Acqu Date': ['10-Sep-16', '10-Sep-16', '10-Sep-16','10-Sep-16', '10-Sep-16', '10-Sep-16'], 'Acqu Time': ['05:46:40', '06:05:26', '07:26:32','08:25:53', '09:16:04', '10:50:46'], 'Vial': ['1:A,4', '1:A,5', '1:A,6', '1:A,7', '1:A,8','1:A,9'], 'Instrument': ['XEVO-TQS#UnitTest', 'XEVO-TQS#UnitTest', 'XEVO-TQS#UnitTest', 'XEVO-TQS#UnitTest', 'XEVO-TQS#UnitTest', 'XEVO-TQS#UnitTest'], 'Acquired Time': [datetime(2016, 9, 10, 5, 46, 40), datetime(2016, 9, 10, 6, 5, 26), datetime(2016, 9, 10, 7, 26, 32), datetime(2016, 9, 10, 8, 25, 53), datetime(2016, 9, 10, 9, 16, 4),datetime(2016, 9, 10, 10, 50, 46)], 'Run Order': [3, 4, 5, 6, 7, 8],'Batch': [1, 1, 1, 1, 1, 1],'Dilution': [100, 100, 100, 100, 100, 100]})
		self.targeted.sampleMetadata['Acquired Time'] = self.targeted.sampleMetadata['Acquired Time']
		self.targeted.featureMetadata = pandas.DataFrame({'Feature Name': ['Feature1-IS','Feature2-Monitored','Feature3-Unusable','Feature4-Unusable','Feature5-UnusableNoiseFilled','Feature6-UnusableNoiseFilled','Feature7-UnusableNoiseFilled','Feature8-axb','Feature9-logaxb','Feature10-ax'], 'TargetLynx Feature ID': [1,2,3,4,5,6,7,8,9,10], 'TargetLynx IS ID': ['1','1','1','1','1','1','1','1','1','1'], 'IS': [True, False, False, False, False, False, False, False, False, False], 'calibrationEquation': ['','','','','','','','((area * responseFactor)-b)/a','10**((numpy.log10(area * responseFactor)-b)/a)','area/a'], 'calibrationMethod': [CalibrationMethod.noIS, CalibrationMethod.noCalibration, CalibrationMethod.backcalculatedIS, CalibrationMethod.backcalculatedIS, CalibrationMethod.backcalculatedIS, CalibrationMethod.backcalculatedIS, CalibrationMethod.backcalculatedIS, CalibrationMethod.backcalculatedIS, CalibrationMethod.backcalculatedIS, CalibrationMethod.noIS], 'quantificationType': [QuantificationType.IS, QuantificationType.Monitored, QuantificationType.QuantOwnLabeledAnalogue, QuantificationType.QuantAltLabeledAnalogue, QuantificationType.QuantOwnLabeledAnalogue, QuantificationType.QuantOwnLabeledAnalogue, QuantificationType.QuantAltLabeledAnalogue,QuantificationType.QuantOwnLabeledAnalogue,QuantificationType.QuantAltLabeledAnalogue,QuantificationType.QuantOther], 'unitCorrectionFactor': [1.,1.,1.,1.,1.,1.,1.,1.,1.,1.], 'Unit': ['noUnit','pg/uL','uM','pg/uL','uM','pg/uL','uM','pg/uL','uM','pg/uL'], 'Cpd Info': ['info cpd1','info cpd2','info cpd3','info cpd4','info cpd5','info cpd6','info cpd7','info cpd8','info cpd9','info cpd10'], 'LLOQ': [100.,numpy.nan,numpy.nan,100.,100.,100.,100.,100.,100.,100.], 'ULOQ': [1000., numpy.nan, 1000, numpy.nan, 1000., 1000., 1000., 1000., 1000., 1000.], 'Noise (area)': [10.,numpy.nan,10.,10.,numpy.nan,10.,10.,10.,10.,10.], 'a': [2.,numpy.nan,2.,2.,2.,numpy.nan,2.,2.,2.,2.], 'another column': ['something 1','something 2','something 3','something 4','something 5','something 6','something 7','something 8','something 9','something 10'],'b': [1.,numpy.nan,1.,1.,1.,1.,numpy.nan,1.,1.,0.], 'r': [0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.99], 'r2': [0.995,0.995,0.995,0.995,0.995,0.995,0.995,0.995,0.995,0.995]})
		self.targeted.featureMetadata = self.targeted.featureMetadata.loc[[4, 5, 6, 7, 8, 9, 1], :]
		self.targeted.featureMetadata = self.targeted.featureMetadata.drop(['IS', 'TargetLynx IS ID'], axis=1)
		self.targeted.featureMetadata.reset_index(drop=True, inplace=True)
		self.targeted._intensityData = numpy.array([[-numpy.inf, -numpy.inf, -numpy.inf, -numpy.inf, -numpy.inf, -numpy.inf, 50], [500., 500., 500., 500., 500., 500., 500.], [numpy.inf, numpy.inf, numpy.inf, numpy.inf, numpy.inf, numpy.inf, 5000.], [500., 500., 500., 500., 500., 500., 500.], [500., 500., 500., 500., 500., 500., 500.], [500., 500., 500., 500., 500., 500., 500.]])
		calibFeatureMetadata = copy.deepcopy(self.targeted.featureMetadata)
		calibIntensityData = numpy.array([[250., 250., 250., 250., 250., 250., 250.], [500., 500., 500., 500., 500., 500., 500.],[750., 750., 750., 750., 750., 750., 750.]])
		calibPeakConcentrationDeviation = numpy.array([[0., 0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0., 0.]])
		calibPeakExpectedConcentration = numpy.array([[250., 250., 250., 250., 250., 250., 250.], [500., 500., 500., 500., 500., 500., 500.], [750., 750., 750., 750., 750., 750., 750.]])
		peakIntegrationFlag = pandas.DataFrame({'Feature1-IS': ['MM', 'MM', 'MM', 'MM', 'MM', 'MM', 'MM', 'MM', 'MM'], 'Feature2-Monitored': ['bb', 'MM', 'bb', 'bb', 'bb', 'bb', 'bb', 'bb', 'bb'], 'Feature3-Unusable': ['bb', 'bb', 'MM', 'bb', 'bb', 'bb', 'bb', 'bb', 'bb'], 'Feature4-Unusable': ['bb', 'bb', 'bb', 'MM', 'bb', 'bb', 'bb', 'bb', 'bb'], 'Feature5-UnusableNoiseFilled': ['bb', 'bb', 'bb', 'bb', 'MM', 'bb', 'bb', 'bb', 'bb'], 'Feature6-UnusableNoiseFilled': ['bb', 'bb', 'bb', 'bb', 'bb', 'MM', 'bb', 'bb', 'bb'], 'Feature7-UnusableNoiseFilled': ['bb', 'bb', 'bb', 'bb', 'bb', 'bb', 'MM', 'bb', 'bb'], 'Feature8-axb': ['bb', 'bb', 'bb', 'bb', 'bb', 'bb', 'bb', 'MM', 'bb'], 'Feature9-logaxb': ['bb', 'bb', 'bb', 'bb', 'bb', 'bb', 'bb', 'bb', 'MM'], 'Feature10-ax': ['MM', 'bb', 'bb', 'bb', 'bb', 'bb', 'bb', 'bb', 'bb']})
		peakIntegrationFlag = peakIntegrationFlag[['Feature1-IS', 'Feature2-Monitored', 'Feature3-Unusable', 'Feature4-Unusable', 'Feature5-UnusableNoiseFilled', 'Feature6-UnusableNoiseFilled', 'Feature7-UnusableNoiseFilled','Feature8-axb', 'Feature9-logaxb', 'Feature10-ax']]
		peakIntegrationFlag.index = [1, 2, 3, 4, 5, 6, 7, 8, 9]
		calibPeakIntegrationFlag = copy.deepcopy(peakIntegrationFlag).loc[[1, 2, 3], ['Feature5-UnusableNoiseFilled', 'Feature6-UnusableNoiseFilled', 'Feature7-UnusableNoiseFilled',	'Feature8-axb', 'Feature9-logaxb', 'Feature10-ax', 'Feature2-Monitored']]
		calibPeakIntegrationFlag.reset_index(drop=True, inplace=True)
		calibPeakResponse = numpy.array([[2100., 2100., 2100., 2100., 2100., 2100., 2100.], [2100., 2100., 2100., 2100., 2100., 2100., 2100.], [2100., 2100., 2100., 2100., 2100., 2100., 2100.]])
		calibPeakArea = numpy.array([[500., 500., 500., 500., 500., 500., 500.], [1000., 1000., 1000., 1000., 1000., 1000., 1000.], [1500., 1500., 1500., 1500., 1500., 1500., 1500.]])
		calibSampleMetadata = pandas.DataFrame({'Sample File Name': ['UnitTest_targeted_file_001', 'UnitTest_targeted_file_002', 'UnitTest_targeted_file_003','UnitTest_targeted_file_004', 'UnitTest_targeted_file_005', 'UnitTest_targeted_file_006','UnitTest_targeted_file_007', 'UnitTest_targeted_file_008', 'UnitTest_targeted_file_009'], 'TargetLynx Sample ID': [1, 2, 3, 4, 5, 6, 7, 8, 9], 'MassLynx Row ID': [1, 2, 3, 4, 5, 6, 7, 8, 9], 'Sample Name': ['Calib-Low', 'Calib-Mid', 'Calib-High', 'Sample-LLOQ', 'Sample-Fine','Sample-ULOQ', 'Blank', 'QC', 'Other'], 'Sample Type': ['Standard', 'Standard', 'Standard','Analyte', 'Analyte', 'Analyte', 'Blank', 'QC', 'Solvent'],'Acqu Date': ['10-Sep-16', '10-Sep-16', '10-Sep-16','10-Sep-16', '10-Sep-16', '10-Sep-16', '10-Sep-16', '10-Sep-16', '10-Sep-16'], 'Acqu Time': ['02:14:32', '03:23:02', '04:52:35', '05:46:40', '06:05:26', '07:26:32', '08:25:53', '09:16:04', '10:50:46'], 'Vial': ['1:A,1', '1:A,2', '1:A,3', '1:A,4', '1:A,5','1:A,6', '1:A,7', '1:A,8', '1:A,9'],'Instrument': ['XEVO-TQS#UnitTest', 'XEVO-TQS#UnitTest','XEVO-TQS#UnitTest', 'XEVO-TQS#UnitTest','XEVO-TQS#UnitTest', 'XEVO-TQS#UnitTest','XEVO-TQS#UnitTest', 'XEVO-TQS#UnitTest','XEVO-TQS#UnitTest'],'Calibrant': [True, True, True, False, False, False, False, False, False],'Study Sample': [False, False, False, True, True, True, False, False, False], 'Blank': [False, False, False, False, False, False, True, False, False], 'QC': [False, False, False, False, False, False, False,True, False],'Other': [False, False, False, False, False, False, False, False, True], 'Acquired Time': [pandas.Timestamp(datetime(2016, 9, 10, 2, 14, 32)), pandas.Timestamp(datetime(2016, 9, 10, 3, 23, 2)), pandas.Timestamp(datetime(2016, 9, 10, 4, 52, 35)), pandas.Timestamp(datetime(2016, 9, 10, 5, 46, 40)), pandas.Timestamp(datetime(2016, 9, 10, 6, 5, 26)), pandas.Timestamp(datetime(2016, 9, 10, 7, 26, 32)), pandas.Timestamp(datetime(2016, 9, 10, 8, 25, 53)), pandas.Timestamp(datetime(2016, 9, 10, 9, 16, 4)), pandas.Timestamp(datetime(2016, 9, 10, 10, 50, 46))],'Run Order': [0, 1, 2, 3, 4, 5, 6, 7, 8], 'Batch': [1, 1, 1, 1, 1, 1, 1, 1, 1]})
		calibSampleMetadata['Acquired Time'] = calibSampleMetadata['Acquired Time']
		calibSampleMetadata = calibSampleMetadata.loc[[0, 1, 2], :]
		calibSampleMetadata = calibSampleMetadata.drop(['Calibrant', 'Study Sample', 'Blank', 'QC', 'Other'], axis=1)
		self.targeted.calibration = dict({'calibSampleMetadata': calibSampleMetadata, 'calibFeatureMetadata': calibFeatureMetadata, 'calibIntensityData': calibIntensityData, 'calibPeakResponse': calibPeakResponse, 'calibPeakArea': calibPeakArea, 'calibPeakExpectedConcentration': calibPeakExpectedConcentration, 'calibPeakConcentrationDeviation': calibPeakConcentrationDeviation, 'calibPeakIntegrationFlag': calibPeakIntegrationFlag})
		self.targeted.initialiseMasks()


	def test_exportdataset_exportcsv(self):
		expectedSampleMetadata = copy.deepcopy(self.targeted.sampleMetadata)
		expectedSampleMetadata['Acquired Time'] = expectedSampleMetadata['Acquired Time']
		expectedFeatureMetadata = copy.deepcopy(self.targeted.featureMetadata)
		expectedFeatureMetadata['calibrationEquation'] = [numpy.nan, numpy.nan, numpy.nan, '((area * responseFactor)-b)/a', '10**((numpy.log10(area * responseFactor)-b)/a)', 'area/a', numpy.nan]
		expectedFeatureMetadata['calibrationMethod'] = ['Backcalculated with Internal Standard', 'Backcalculated with Internal Standard', 'Backcalculated with Internal Standard', 'Backcalculated with Internal Standard', 'Backcalculated with Internal Standard', 'No Internal Standard', 'No calibration']
		expectedFeatureMetadata['quantificationType'] = ['Quantified and validated with own labeled analogue', 'Quantified and validated with own labeled analogue', 'Quantified and validated with alternative labeled analogue', 'Quantified and validated with own labeled analogue', 'Quantified and validated with alternative labeled analogue', 'Other quantification', 'Monitored for relative information']
		expectedIntensityData = pandas.DataFrame([['<LLOQ', '<LLOQ', '<LLOQ', '<LLOQ', '<LLOQ', '<LLOQ', 50.], ['500.0', '500.0', '500.0', '500.0', '500.0', '500.0', 500.], ['>ULOQ', '>ULOQ', '>ULOQ', '>ULOQ', '>ULOQ', '>ULOQ', 5000.], ['500.0', '500.0', '500.0', '500.0', '500.0', '500.0', 500.], ['500.0', '500.0', '500.0', '500.0', '500.0', '500.0', 500.], ['500.0', '500.0', '500.0', '500.0', '500.0', '500.0', 500.]])

		with tempfile.TemporaryDirectory() as tmpdirname:
			targetFolder = os.path.join(tmpdirname)
			self.targeted.exportDataset(destinationPath=targetFolder, saveFormat='CSV')
			# Read
			exportedSampleMetadata  = pandas.read_csv(os.path.join(tmpdirname, self.targeted.name + '_sampleMetadata.csv'),  index_col=0, parse_dates=['Acquired Time'])
			exportedFeatureMetadata = pandas.read_csv(os.path.join(tmpdirname, self.targeted.name + '_featureMetadata.csv'), index_col=0)
			exportedIntensityData   = pandas.read_csv(os.path.join(tmpdirname, self.targeted.name + '_intensityData.csv'),   index_col=False, header=None)
			# Check
			assert_frame_equal(expectedSampleMetadata, exportedSampleMetadata, check_dtype=True)
			assert_frame_equal(expectedFeatureMetadata, exportedFeatureMetadata)
			assert_frame_equal(expectedIntensityData, exportedIntensityData)


	def test_exportdataset_exportunifiedcsv(self):
		expectedCombined = pandas.DataFrame({'Acqu Date': [numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, '10-Sep-16', '10-Sep-16', '10-Sep-16', '10-Sep-16', '10-Sep-16', '10-Sep-16'],'Acqu Time': [numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan,numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, '05:46:40', '06:05:26', '07:26:32', '08:25:53', '09:16:04', '10:50:46'],'Acquired Time': [numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, '2016-09-10 05:46:40','2016-09-10 06:05:26', '2016-09-10 07:26:32', '2016-09-10 08:25:53', '2016-09-10 09:16:04', '2016-09-10 10:50:46'], 'Batch': [numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, 1.0, 1.0, 1.0, 1.0, 1.0,  1.0], 'Dilution': [numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, 100.0, 100.0, 100.0, 100.0, 100.0,  100.0], 'Instrument': [numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan,numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan,numpy.nan, numpy.nan, numpy.nan, numpy.nan, 'XEVO-TQS#UnitTest','XEVO-TQS#UnitTest', 'XEVO-TQS#UnitTest', 'XEVO-TQS#UnitTest','XEVO-TQS#UnitTest', 'XEVO-TQS#UnitTest'], 'MassLynx Row ID': [numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, 4., 5., 6., 7.,8., 9.], 'Run Order': [numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, 3., 4., 5., 6., 7., 8.],'Sample File Name': [numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, 'UnitTest_targeted_file_004', 'UnitTest_targeted_file_005', 'UnitTest_targeted_file_006', 'UnitTest_targeted_file_007', 'UnitTest_targeted_file_008', 'UnitTest_targeted_file_009'],'Sample Name': [numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, 'Sample-LLOQ', 'Sample-Fine', 'Sample-ULOQ', 'Blank', 'QC', 'Other'], 'Sample Type': [numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, 'Analyte', 'Analyte','Analyte', 'Blank', 'QC', 'Solvent'],'TargetLynx Sample ID': [numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan,numpy.nan, 4., 5., 6., 7., 8., 9.],'Vial': [numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan,  numpy.nan, numpy.nan, '1:A,4', '1:A,5', '1:A,6', '1:A,7', '1:A,8', '1:A,9'],'0': ['info cpd5', 'Feature5-UnusableNoiseFilled', '100.0', numpy.nan, '5', '1000.0', 'uM', '2.0', 'something 5', '1.0', numpy.nan, 'Backcalculated with Internal Standard',  'Quantified and validated with own labeled analogue', '0.99', '0.995', '1.0', '<LLOQ', '500.0', '>ULOQ', '500.0', '500.0', '500.0'],'1': ['info cpd6', 'Feature6-UnusableNoiseFilled', '100.0', '10.0', '6', '1000.0', 'pg/uL', numpy.nan,'something 6', '1.0', numpy.nan, 'Backcalculated with Internal Standard', 'Quantified and validated with own labeled analogue', '0.99', '0.995', '1.0', '<LLOQ', '500.0', '>ULOQ', '500.0', '500.0', '500.0'],'2': ['info cpd7', 'Feature7-UnusableNoiseFilled', '100.0', '10.0', '7', '1000.0', 'uM', '2.0', 'something 7', numpy.nan, numpy.nan, 'Backcalculated with Internal Standard', 'Quantified and validated with alternative labeled analogue', '0.99', '0.995', '1.0', '<LLOQ', '500.0', '>ULOQ', '500.0', '500.0', '500.0'],'3': ['info cpd8', 'Feature8-axb', '100.0', '10.0', '8', '1000.0', 'pg/uL', '2.0', 'something 8', '1.0','((area * responseFactor)-b)/a', 'Backcalculated with Internal Standard', 'Quantified and validated with own labeled analogue', '0.99', '0.995', '1.0', '<LLOQ', '500.0', '>ULOQ', '500.0', '500.0', '500.0'],'4': ['info cpd9', 'Feature9-logaxb', '100.0', '10.0', '9', '1000.0', 'uM', '2.0', 'something 9', '1.0', '10**((numpy.log10(area * responseFactor)-b)/a)', 'Backcalculated with Internal Standard', 'Quantified and validated with alternative labeled analogue', '0.99', '0.995', '1.0', '<LLOQ', '500.0', '>ULOQ', '500.0', '500.0', '500.0'], '5': ['info cpd10', 'Feature10-ax', '100.0', '10.0', '10', '1000.0', 'pg/uL', '2.0', 'something 10', '0.0', 'area/a', 'No Internal Standard', 'Other quantification', '0.99', '0.995', '1.0', '<LLOQ', '500.0', '>ULOQ', '500.0', '500.0', '500.0'],'6': ['info cpd2', 'Feature2-Monitored', numpy.nan, numpy.nan, '2', numpy.nan, 'pg/uL', numpy.nan, 'something 2', numpy.nan, numpy.nan, 'No calibration', 'Monitored for relative information', '0.99', '0.995', '1.0', '50.0', '500.0', '5000.0', '500.0', '500.0', '500.0']})
		expectedCombined['Acqu Time'] = pandas.to_datetime(expectedCombined['Acqu Time'])
		expectedCombined['Acqu Date'] = pandas.to_datetime(expectedCombined['Acqu Date'])
		expectedCombined['Acquired Time'] = pandas.to_datetime(expectedCombined['Acquired Time'])

		expectedCombined.index = ['Cpd Info', 'Feature Name', 'LLOQ', 'Noise (area)', 'TargetLynx Feature ID', 'ULOQ', 'Unit', 'a', 'another column', 'b', 'calibrationEquation', 'calibrationMethod', 'quantificationType', 'r', 'r2', 'unitCorrectionFactor', '0', '1', '2', '3', '4', '5']
		expectedCombined = expectedCombined.loc[('Feature Name','TargetLynx Feature ID','calibrationEquation','calibrationMethod','quantificationType','unitCorrectionFactor','Unit','Cpd Info','LLOQ','ULOQ','Noise (area)','a','another column','b','r','r2','0','1','2','3','4','5'),:]

		with tempfile.TemporaryDirectory() as tmpdirname:
			targetFolder = os.path.join(tmpdirname)
			self.targeted.exportDataset(destinationPath=targetFolder, saveFormat='UnifiedCSV')
			# Read
			exportedCombined = pandas.read_csv(os.path.join(tmpdirname, self.targeted.name + '_combinedData.csv'), index_col=0, parse_dates=['Acqu Date', 'Acqu Time', 'Acquired Time'])
			# Check
			assert_frame_equal(expectedCombined.reindex(sorted(expectedCombined), axis=1), exportedCombined.reindex(sorted(exportedCombined), axis=1), check_dtype=False)


	def test_exportdataset_raise_warning(self):
		normalisationWarning = copy.deepcopy(self.targeted)
		normalisationWarning.intensityData[0, :] = [50., 50., 50., 50., 50., 50., 50.]
		normalisationWarning.intensityData[2, :] = [5000., 5000., 5000., 5000., 5000., 5000., 5000.]
		with tempfile.TemporaryDirectory() as tmpdirname:
			targetFolder = os.path.join(tmpdirname)

			with warnings.catch_warnings(record=True) as w:
				# Cause all warnings to always be triggered.
				warnings.simplefilter("always")
				# warning
				self.targeted.exportDataset(destinationPath=targetFolder, saveFormat='CSV')


if __name__ == '__main__':
	unittest.main()
