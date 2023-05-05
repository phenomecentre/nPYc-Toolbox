"""

"""

import scipy
import pandas
import numpy
import sys
import unittest
import os
import tempfile
import inspect
import copy
import warnings

sys.path.append("..")
import nPYc

from nPYc.utilities.generic import print_dict



class test_utilities_read_bruker_xml(unittest.TestCase):


	def test_utilities_readBrukerBIQUANT2_XML(self):
		
		from nPYc.utilities._readBrukerXML import readBrukerXML
		# show the whole diff on error
		self.maxDiff = None
		
		expected = ("UnitTest_Plasma_Rack1_SLL_051218_expno10.100000.17r","05-Dec-2018 11:32:33",
    				[
				        {
				            "Feature Name": "Ethanol",
				            "comment": "",
				            "type": "quantification",
				            "value": -numpy.inf,
				            "lod": 0.1,
				            "loq": "-",
				            "Unit": "mmol/L",
				            "lodMask": False,
				            "Lower Reference Bound": 2.5,
				            "Upper Reference Bound": 97.5,
				            "Lower Reference Value": "-",
				            "Upper Reference Value": 0.82
				        },
				        {
				            "Feature Name": "Trimethylamine-N-oxide",
				            "comment": "",
				            "type": "quantification",
				            "value": -numpy.inf,
				            "lod": 0.08,
				            "loq": "-",
				            "Unit": "mmol/L",
				            "lodMask": False,
				            "Lower Reference Bound": 2.5,
				            "Upper Reference Bound": 97.5,
				            "Lower Reference Value": "-",
				            "Upper Reference Value": 0.08
				        },
				        {
				            "Feature Name": "Alanine",
				            "comment": "",
				            "type": "quantification",
				            "value": 0.517,
				            "lod": 0.02,
				            "loq": "-",
				            "Unit": "mmol/L",
				            "lodMask": True,
				            "Lower Reference Bound": 2.5,
				            "Upper Reference Bound": 97.5,
				            "Lower Reference Value": 0.29,
				            "Upper Reference Value": 0.64
				        }
				    ]
			    	)
		actual = readBrukerXML(os.path.join('..', '..', 'npc-standard-project', 'Derived_Data','bruker_quant_v2_plasma.xml'))
		

		self.assertEqual(expected, actual)
		

	def test_utilities_importBrukerBIQUANT2_XML(self):

		from nPYc.utilities._readBrukerXML import importBrukerXML

		path1 = os.path.join('..', '..', 'npc-standard-project', 'Derived_Data','bruker_quant_v2_plasma.xml')
		
		expectedID = numpy.array([[ -numpy.inf,  -numpy.inf, 0.517],
 								[ -numpy.inf,  -numpy.inf, 0.517],
   								[ -numpy.inf,  -numpy.inf, 0.517]])
		
		expectedFM = pandas.DataFrame.from_dict({'Feature Name': {0: 'Ethanol', 1: 'Trimethylamine-N-oxide', 2: 'Alanine'}, 'comment': {0: '', 1: '', 2: ''}, 'type': {0: 'quantification', 1: 'quantification', 2: 'quantification'}, 'lodMask': {0: False, 1: False, 2: True}, 'lod': {0: 0.1, 1: 0.08, 2: 0.02}, 'loq': {0: '-', 1: '-', 2: '-'}, 'Unit': {0: 'mmol/L', 1: 'mmol/L', 2: 'mmol/L'}, 'Lower Reference Bound': {0: 2.5, 1: 2.5, 2: 2.5}, 'Upper Reference Bound': {0: 97.5, 1: 97.5, 2: 97.5}, 'Lower Reference Value': {0: '-', 1: '-', 2: 0.29}, 'Upper Reference Value': {0: 0.82, 1: 0.08, 2: 0.64}})
		
		expectedSM = pandas.DataFrame.from_dict({'Sample File Name': {0: 'UnitTest_Plasma_Rack1_SLL_051218/10', 1: 'UnitTest_Plasma_Rack1_SLL_051218/10', 2: 'UnitTest_Plasma_Rack1_SLL_051218/10'}, 'Sample Base Name': {0: 'UnitTest_Plasma_Rack1_SLL_051218/10', 1: 'UnitTest_Plasma_Rack1_SLL_051218/10', 2: 'UnitTest_Plasma_Rack1_SLL_051218/10'}, 'expno': {0: 10, 1: 10, 2: 10}, 'Path': {0: '../../npc-standard-project/Derived_Data/bruker_quant_v2_plasma.xml', 1: '../../npc-standard-project/Derived_Data/bruker_quant_v2_plasma.xml', 2: '../../npc-standard-project/Derived_Data/bruker_quant_v2_plasma.xml'}, 'Acquired Time': {0: pandas.Timestamp('2018-12-05 11:32:33'), 1: pandas.Timestamp('2018-12-05 11:32:33'), 2: pandas.Timestamp('2018-12-05 11:32:33')}, 'Run Order': {0: 0, 1: 1, 2: 2}})

		paths = [path1, path1, path1]
		
		(intensityData, sampleMetadata, featureMetadata) = importBrukerXML(paths)
		
		self.assertEqual(len(intensityData), len(paths))
		
		numpy.testing.assert_array_equal(intensityData, expectedID)
		
		pandas.testing.assert_frame_equal(sampleMetadata, expectedSM)
		
		pandas.testing.assert_frame_equal(featureMetadata, expectedFM, check_like=True)

		
	def test_utilities_readBrukerXML(self):
		
		from nPYc.utilities._readBrukerXML import readBrukerXML
		
		self.maxDiff = None
		
		with self.subTest(msg='BI-LISA type'):
			expected = ('UnitTest6_expno10.100000.11r',
						 '27-Aug-2015 09:59:47',
						 [{'Feature Name': 'TPTG',
						   'comment': 'Main Parameters, Triglycerides, TG',
						   'type': 'prediction',
						   'value': 134.65,
						   'lod': '-',
						   'loq': '-',
						   'Unit': 'mg/dL',
						   'lodMask': True,
							'Lower Reference Bound': 2.5,
							'Upper Reference Bound': 97.5,
							'Lower Reference Value': 53.45,
							'Upper Reference Value': 489.81,
						},
						  {'Feature Name': 'TPCH',
						   'comment': 'Main Parameters, Cholesterol, Chol',
						   'type': 'prediction',
						   'value': 183.95,
						   'lod': '-',
						   'loq': '-',
						   'Unit': 'mg/dL',
						   'lodMask': True,
						   'Lower Reference Bound': 2.5,
						   'Upper Reference Bound': 97.5,
						   'Lower Reference Value': 140.31,
						   'Upper Reference Value': 341.43},
						  {'Feature Name': 'LDCH',
						   'comment': 'Main Parameters, LDL Cholesterol, LDL-Chol',
						   'type': 'prediction',
						   'value': 94.57,
						   'lod': '-',
						   'loq': '-',
						   'Unit': 'mg/dL',
						   'lodMask': True,
							'Lower Reference Bound': 2.5,
							'Upper Reference Bound': 97.5,
							'Lower Reference Value': 54.52,
							'Upper Reference Value': 226.6},
						  {'Feature Name': 'HDCH',
						   'comment': 'Main Parameters, HDL Cholesterol, HDL-Chol',
						   'type': 'prediction',
						   'value': 52.22,
						   'lod': '-',
						   'loq': '-',
						   'lodMask': True,
						   'Unit': 'mg/dL',
						   'Lower Reference Bound': 2.5,
						   'Upper Reference Bound': 97.5,
						   'Lower Reference Value': 34.6,
						   'Upper Reference Value': 96.25}])
			actual = readBrukerXML(os.path.join('..', '..', 'npc-standard-project', 'Derived_Data','bruker_quant_BILISA.xml'))
			
			self.assertEqual(expected, actual)
			
		with self.subTest(msg='Urine Quant type'):
			expected = ('UnitTest5_expno840.100000.10r',
						 '15-Aug-2017 13:06:45',
						 [{'Feature Name': 'Creatinine',
						   'comment': '',
						   'type': 'quantification',
						   'value': 4.3,
						   'lod': '-',
						   'loq': '-',
						   'Unit': 'mmol/L',
						   'lodMask': True},
						  {'Feature Name': 'Dimethylamine',
						   'comment': '',
						   'type': 'quantification',
						   'value': 0.19,
						   'lod': 0.133,
						   'loq': '-',
						   'Unit': 'mmol/L',
						   'lodMask': True},
						  {'Feature Name': 'Dimethylamine',
						   'comment': '',
						   'type': 'quantification',
						   'value': 43,
						   'lod': 31,
						   'loq': '-',
						   'Unit': 'mmol/mol Crea',
						   'lodMask': True,
						   'Lower Reference Bound': 2.5,
						   'Upper Reference Bound': 97.5,
						   'Lower Reference Value': '-',
						   'Upper Reference Value': 54},
						  {'Feature Name': 'Trimethylamine',
						   'comment': '',
						   'type': 'quantification',
						   'value': 0,
						   'lod': 0.009,
						   'loq': '-',
						   'Unit': 'mmol/L',
						   'lodMask': False},
						  {'Feature Name': 'Trimethylamine',
						   'comment': '',
						   'type': 'quantification',
						   'value': 0,
						   'lod': 2,
						   'loq': '-',
						   'Unit': 'mmol/mol Crea',
						   'lodMask': True,
						   'Lower Reference Bound': 2.5,
						   'Upper Reference Bound': 97.5,
						   'Lower Reference Value': '-',
						   'Upper Reference Value': 3},
						  {'Feature Name': '1-Methylhistidine',
						   'comment': '',
						   'type': 'quantification',
						   'value': 0,
						   'lod': 0.065,
						   'loq': '-',
						   'Unit': 'mmol/L',
						   'lodMask': False},
						  {'Feature Name': '1-Methylhistidine',
						   'comment': '',
						   'type': 'quantification',
						   'value': 0,
						   'lod': 15,
						   'loq': '-',
						   'Unit': 'mmol/mol Crea',
						   'lodMask': True,
						   'Lower Reference Bound': 2.5,
						   'Upper Reference Bound': 97.5,
						   'Lower Reference Value': '-',
						   'Upper Reference Value': 15}])

			actual = readBrukerXML(os.path.join('..', '..', 'npc-standard-project', 'Derived_Data','bruker_quant_urine.xml'))

			self.assertEqual(expected, actual)


	def test_utilities_readBrukerXML_warns(self):

		from nPYc.utilities._readBrukerXML import importBrukerXML

		with tempfile.TemporaryDirectory() as tmpdirname:
			tmpfile = os.path.join(tmpdirname, 'malformedxml.xml')

			with open(tmpfile, 'w') as tmpf:
				tmpf.write('Most definitely not xml <as \n')

			self.assertWarnsRegex(UserWarning, 'Error parsing xml in .+?, skipping', importBrukerXML, [tmpfile])


	def test_utilities_importBrukerXML(self):

		from nPYc.utilities._readBrukerXML import importBrukerXML

		path = os.path.join('..', '..', 'npc-standard-project', 'Derived_Data','bruker_quant_urine.xml')
		
		paths = [path, path, path]
		
		expectedIntesityData = numpy.array([[4.3, 0.19, 43., 0., 0., 0., 0.],
											[4.3, 0.19, 43., 0., 0., 0., 0.],
											[4.3, 0.19, 43., 0., 0., 0., 0.]])

		expectedFeatureMetadata = pandas.DataFrame.from_dict({'Feature Name': {0: 'Creatinine', 1: 'Dimethylamine', 2: 'Dimethylamine', 3: 'Trimethylamine',
																			   4: 'Trimethylamine', 5: '1-Methylhistidine',6: '1-Methylhistidine'},
															  'Lower Reference Bound': {0: numpy.nan, 1: numpy.nan, 2: 2.5, 3: numpy.nan, 4: 2.5, 5: numpy.nan, 6: 2.5},
															  'Lower Reference Value': {0: numpy.nan, 1: numpy.nan, 2: '-', 3: numpy.nan, 4: '-', 5: numpy.nan, 6: '-'},
															  'Unit': {0: 'mmol/L', 1: 'mmol/L', 2: 'mmol/mol Crea', 3: 'mmol/L', 4: 'mmol/mol Crea',
																	   5: 'mmol/L', 6: 'mmol/mol Crea'},
															  'Upper Reference Bound': {0: numpy.nan, 1: numpy.nan, 2: 97.5, 3: numpy.nan, 4: 97.5, 5: numpy.nan, 6: 97.5},
															  'Upper Reference Value': {0: numpy.nan, 1: numpy.nan, 2: 54.0, 3: numpy.nan, 4: 3.0, 5: numpy.nan, 6: 15.0},
															  'comment': {0: '', 1: '', 2: '', 3: '', 4: '', 5: '', 6: ''},
															  'lod': {0: '-', 1: 0.133, 2: 31, 3: 0.009, 4: 2, 5: 0.065, 6: 15},
															  'lodMask': {0: True, 1: True, 2:True, 3: False, 4: True, 5: False, 6: True},
															  'loq': {0: '-', 1: '-', 2: '-', 3: '-', 4: '-', 5: '-', 6: '-'},
															  'type': {0: 'quantification', 1: 'quantification', 2: 'quantification',
																	   3: 'quantification', 4: 'quantification', 5: 'quantification',
																	   6: 'quantification'}})

		expectedSampleMetadata = pandas.DataFrame([], columns=['Sample File Name', 'Sample Base Name', 'expno', 'Path', 'Acquired Time', 'Run Order'])
		expectedSampleMetadata['Path'] = [path, path, path]
		#expectedSampleMetadata['Sample File Name'] = ['UnitTest5_expno840.100000.10r', 'UnitTest5_expno840.100000.10r', 'UnitTest5_expno840.100000.10r'] # change to Sample File Name matching Sample Base Name
		expectedSampleMetadata['Sample File Name'] = ['UnitTest5/840', 'UnitTest5/840', 'UnitTest5/840']
		expectedSampleMetadata['Sample Base Name'] = ['UnitTest5/840', 'UnitTest5/840', 'UnitTest5/840']
		expectedSampleMetadata['Acquired Time'] = [pandas.Timestamp('2017-08-15 13:06:45'), pandas.Timestamp('2017-08-15 13:06:45'), pandas.Timestamp('2017-08-15 13:06:45')]
		expectedSampleMetadata['Run Order'] = [0, 1, 2]
		expectedSampleMetadata['expno'] = [840, 840, 840]

		(intensityData, sampleMetadata, featureMetadata) = importBrukerXML(paths)

		numpy.testing.assert_array_equal(intensityData, expectedIntesityData)
		pandas.testing.assert_frame_equal(sampleMetadata, expectedSampleMetadata)
		pandas.testing.assert_frame_equal(featureMetadata, expectedFeatureMetadata, check_like=True)


	def test_utilities_importBrukerXML_fails(self):

		from nPYc.utilities._readBrukerXML import importBrukerXML

		path = os.path.join('..', '..', 'npc-standard-project', 'Derived_Data','bruker_quant_urine.xml')

		with tempfile.TemporaryDirectory() as tmpdirname:
			tmpfile = os.path.join(tmpdirname, 'malformedxml.xml')

			with open(tmpfile, 'w') as tmpf:
				tmpf.write('Most definitely not xml <as \n')

			paths = [path, path, tmpfile, path, tmpfile, tmpfile]

			with warnings.catch_warnings():
				warnings.simplefilter("ignore")
				(intensityData, sampleMetadata, featureMetadata) = importBrukerXML(paths)

		expectedSampleMetadata = pandas.DataFrame([], columns=['Sample File Name', 'Sample Base Name', 'expno', 'Path', 'Acquired Time', 'Run Order'])
		expectedSampleMetadata['Path'] = [path, path, path]
		#expectedSampleMetadata['Sample File Name'] = ['UnitTest5_expno840.100000.10r', 'UnitTest5_expno840.100000.10r', 'UnitTest5_expno840.100000.10r'] # change to Sample File Name matching Sample Base Name
		expectedSampleMetadata['Sample File Name'] = ['UnitTest5/840', 'UnitTest5/840', 'UnitTest5/840']
		expectedSampleMetadata['Sample Base Name'] = ['UnitTest5/840', 'UnitTest5/840', 'UnitTest5/840']
		expectedSampleMetadata['Acquired Time'] = [pandas.Timestamp('2017-08-15 13:06:45'), pandas.Timestamp('2017-08-15 13:06:45'), pandas.Timestamp('2017-08-15 13:06:45')]
		expectedSampleMetadata['Run Order'] = [0, 1, 2]
		expectedSampleMetadata['expno'] = [840, 840, 840]

		pandas.testing.assert_frame_equal(sampleMetadata, expectedSampleMetadata)
		self.assertEqual(intensityData.shape[0], 3)


