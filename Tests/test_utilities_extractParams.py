"""
Test parameter extraction from raw data files
"""

import scipy
import pandas
import numpy
import sys
import unittest
import os
import re
import tempfile
import inspect
from nPYc.utilities.extractParams import extractWatersRAWParams, extractBrukerparams, \
	extractmzMLParamsRegex, buildFileList
sys.path.append("..")
import nPYc


"""
Check extract params is working correctly.
"""


class test_utilities_extractParams(unittest.TestCase):

	def setUp(self):
		self.pathHeader = os.path.join('..', '..', 'npc-standard-project', 'Raw_Data')

	def test_extractParams_buildFileList(self):

		with self.subTest(msg='Waters Paths'):
			pathHeader = os.path.join(self.pathHeader, 'ms', 'parameters_data')

			expected = ['UnitTest1_LPOS_ToF02_B1E1_SR.raw', 'UnitTest1_LPOS_ToF02_B1E2_SR.raw', 'UnitTest1_LPOS_ToF02_B1E3_SR.raw',
						'UnitTest1_LPOS_ToF02_B1E4_SR.raw', 'UnitTest1_LPOS_ToF02_B1E5_SR.raw', 'UnitTest1_LPOS_ToF02_B1S1_SR.raw',
						'UnitTest1_LPOS_ToF02_B1S2_SR.raw', 'UnitTest1_LPOS_ToF02_B1S3_SR.raw', 'UnitTest1_LPOS_ToF02_B1S4_SR.raw',
						'UnitTest1_LPOS_ToF02_B1S5_SR.raw', 'UnitTest1_LPOS_ToF02_B1SRD01.raw', 'UnitTest1_LPOS_ToF02_B1SRD02.raw',
						'UnitTest1_LPOS_ToF02_B1SRD03.raw', 'UnitTest1_LPOS_ToF02_B1SRD04.raw', 'UnitTest1_LPOS_ToF02_B1SRD05.raw',
						'UnitTest1_LPOS_ToF02_B1SRD06.raw', 'UnitTest1_LPOS_ToF02_B1SRD07.raw', 'UnitTest1_LPOS_ToF02_B1SRD08.raw',
						'UnitTest1_LPOS_ToF02_B1SRD09.raw', 'UnitTest1_LPOS_ToF02_B1SRD10.raw', 'UnitTest1_LPOS_ToF02_B1SRD11.raw',
						'UnitTest1_LPOS_ToF02_B1SRD12.raw', 'UnitTest1_LPOS_ToF02_B1SRD13.raw', 'UnitTest1_LPOS_ToF02_B1SRD14.raw',
						'UnitTest1_LPOS_ToF02_B1SRD15.raw', 'UnitTest1_LPOS_ToF02_B1SRD16.raw', 'UnitTest1_LPOS_ToF02_B1SRD17.raw',
						'UnitTest1_LPOS_ToF02_B1SRD18.raw', 'UnitTest1_LPOS_ToF02_B1SRD19.raw', 'UnitTest1_LPOS_ToF02_B1SRD20.raw',
						'UnitTest1_LPOS_ToF02_B1SRD21.raw', 'UnitTest1_LPOS_ToF02_B1SRD22.raw', 'UnitTest1_LPOS_ToF02_B1SRD23.raw',
						'UnitTest1_LPOS_ToF02_B1SRD24.raw', 'UnitTest1_LPOS_ToF02_B1SRD25.raw', 'UnitTest1_LPOS_ToF02_B1SRD26.raw',
						'UnitTest1_LPOS_ToF02_B1SRD27.raw', 'UnitTest1_LPOS_ToF02_B1SRD28.raw', 'UnitTest1_LPOS_ToF02_B1SRD29.raw',
						'UnitTest1_LPOS_ToF02_B1SRD30.raw', 'UnitTest1_LPOS_ToF02_B1SRD31.raw', 'UnitTest1_LPOS_ToF02_B1SRD32.raw',
						'UnitTest1_LPOS_ToF02_B1SRD33.raw', 'UnitTest1_LPOS_ToF02_B1SRD34.raw', 'UnitTest1_LPOS_ToF02_B1SRD35.raw',
						'UnitTest1_LPOS_ToF02_B1SRD36.raw', 'UnitTest1_LPOS_ToF02_B1SRD37.raw', 'UnitTest1_LPOS_ToF02_B1SRD38.raw',
						'UnitTest1_LPOS_ToF02_B1SRD39.raw', 'UnitTest1_LPOS_ToF02_B1SRD40.raw', 'UnitTest1_LPOS_ToF02_B1SRD41.raw',
						'UnitTest1_LPOS_ToF02_B1SRD42.raw', 'UnitTest1_LPOS_ToF02_B1SRD43.raw', 'UnitTest1_LPOS_ToF02_B1SRD44.raw',
						'UnitTest1_LPOS_ToF02_B1SRD45.raw', 'UnitTest1_LPOS_ToF02_B1SRD46.raw', 'UnitTest1_LPOS_ToF02_B1SRD47.raw',
						'UnitTest1_LPOS_ToF02_B1SRD48.raw', 'UnitTest1_LPOS_ToF02_B1SRD49.raw', 'UnitTest1_LPOS_ToF02_B1SRD50.raw',
						'UnitTest1_LPOS_ToF02_B1SRD51.raw', 'UnitTest1_LPOS_ToF02_B1SRD52.raw', 'UnitTest1_LPOS_ToF02_B1SRD53.raw',
						'UnitTest1_LPOS_ToF02_B1SRD54.raw', 'UnitTest1_LPOS_ToF02_B1SRD55.raw', 'UnitTest1_LPOS_ToF02_B1SRD56.raw',
						'UnitTest1_LPOS_ToF02_B1SRD57.raw', 'UnitTest1_LPOS_ToF02_B1SRD58.raw', 'UnitTest1_LPOS_ToF02_B1SRD59.raw',
						'UnitTest1_LPOS_ToF02_B1SRD60.raw', 'UnitTest1_LPOS_ToF02_B1SRD61.raw', 'UnitTest1_LPOS_ToF02_B1SRD62.raw',
						'UnitTest1_LPOS_ToF02_B1SRD63.raw', 'UnitTest1_LPOS_ToF02_B1SRD64.raw', 'UnitTest1_LPOS_ToF02_B1SRD65.raw',
						'UnitTest1_LPOS_ToF02_B1SRD66.raw', 'UnitTest1_LPOS_ToF02_B1SRD67.raw', 'UnitTest1_LPOS_ToF02_B1SRD68.raw',
						'UnitTest1_LPOS_ToF02_B1SRD69.raw', 'UnitTest1_LPOS_ToF02_B1SRD70.raw', 'UnitTest1_LPOS_ToF02_B1SRD71.raw',
						'UnitTest1_LPOS_ToF02_B1SRD72.raw', 'UnitTest1_LPOS_ToF02_B1SRD73.raw', 'UnitTest1_LPOS_ToF02_B1SRD74.raw',
						'UnitTest1_LPOS_ToF02_B1SRD75.raw', 'UnitTest1_LPOS_ToF02_B1SRD76.raw', 'UnitTest1_LPOS_ToF02_B1SRD77.raw',
						'UnitTest1_LPOS_ToF02_B1SRD78.raw', 'UnitTest1_LPOS_ToF02_B1SRD79.raw', 'UnitTest1_LPOS_ToF02_B1SRD80.raw',
						'UnitTest1_LPOS_ToF02_B1SRD81.raw', 'UnitTest1_LPOS_ToF02_B1SRD82.raw', 'UnitTest1_LPOS_ToF02_B1SRD83.raw',
						'UnitTest1_LPOS_ToF02_B1SRD84.raw', 'UnitTest1_LPOS_ToF02_B1SRD85.raw', 'UnitTest1_LPOS_ToF02_B1SRD86.raw',
						'UnitTest1_LPOS_ToF02_B1SRD87.raw', 'UnitTest1_LPOS_ToF02_B1SRD88.raw', 'UnitTest1_LPOS_ToF02_B1SRD89.raw',
						'UnitTest1_LPOS_ToF02_B1SRD90.raw', 'UnitTest1_LPOS_ToF02_B1SRD91.raw', 'UnitTest1_LPOS_ToF02_B1SRD92.raw',
						'UnitTest1_LPOS_ToF02_Blank01.raw', 'UnitTest1_LPOS_ToF02_Blank02.raw', 'UnitTest1_LPOS_ToF02_S1W01.raw',
						'UnitTest1_LPOS_ToF02_S1W02.raw', 'UnitTest1_LPOS_ToF02_S1W03.raw', 'UnitTest1_LPOS_ToF02_S1W04.raw',
						'UnitTest1_LPOS_ToF02_S1W05.raw', 'UnitTest1_LPOS_ToF02_S1W06.raw', 'UnitTest1_LPOS_ToF02_S1W07.raw',
						'UnitTest1_LPOS_ToF02_S1W08_x.raw', 'UnitTest1_LPOS_ToF02_S1W09.raw', 'UnitTest1_LPOS_ToF02_S1W10.raw',
						'UnitTest1_LPOS_ToF02_S1W11_LTR.raw', 'UnitTest1_LPOS_ToF02_S1W12_SR.raw', 'UnitTest1_LPOS_ToF02_ERROR.raw']
			expected = [os.path.join(pathHeader, x) for x in expected]

			pattern = '.+?\.raw$'
			pattern = re.compile(pattern)

			obtained = buildFileList(pathHeader, pattern)
			# Sorting filelists to account for differences in directory transversing between operating system
			obtained.sort()
			expected.sort()

			self.assertEqual(obtained, expected)

		with self.subTest(msg='Bruker Paths'):
			pathHeader = os.path.join(self.pathHeader, 'nmr')

			expected = [os.path.join('UnitTest1', 'UnitTest1_Urine_Rack1_SLL_270814', '50', 'pdata', '1', '1r'),
						os.path.join('UnitTest1', 'UnitTest1_Urine_Rack1_SLL_270814', '60', 'pdata', '1', '1r'),
						os.path.join('UnitTest1', 'UnitTest1_Urine_Rack1_SLL_270814', '20', 'pdata', '1', '1r'),
						os.path.join('UnitTest1', 'UnitTest1_Urine_Rack1_SLL_270814', '80', 'pdata', '1', '1r'),
						os.path.join('UnitTest1', 'UnitTest1_Urine_Rack1_SLL_270814', '10', 'pdata', '1', '1r'),
						os.path.join('UnitTest1', 'UnitTest1_Urine_Rack1_SLL_270814', '30', 'pdata', '1', '1r'),
						os.path.join('UnitTest1', 'UnitTest1_Urine_Rack1_SLL_270814', '30', 'pdata', '2', '1r'),
						os.path.join('UnitTest1', 'UnitTest1_Urine_Rack1_SLL_270814', '90', 'pdata', '1', '1r'),
						os.path.join('UnitTest1', 'UnitTest1_Urine_Rack1_SLL_270814', '70', 'pdata', '1', '1r'),
						os.path.join('UnitTest1', 'UnitTest1_Urine_Rack1_SLL_270814', '40', 'pdata', '1', '1r'),
						os.path.join('UnitTest3', 'UnitTest3_Serum_Rack01_RCM_190116', '150', 'pdata', '1', '1r'),
						os.path.join('UnitTest3', 'UnitTest3_Serum_Rack01_RCM_190116', '161', 'pdata', '1', '1r'),
						os.path.join('UnitTest3', 'UnitTest3_Serum_Rack01_RCM_190116', '160', 'pdata', '1', '1r'),
						os.path.join('UnitTest3', 'UnitTest3_Serum_Rack01_RCM_190116', '151', 'pdata', '1', '1r'),
						os.path.join('UnitTest3', 'UnitTest3_Serum_Rack01_RCM_190116', '180', 'pdata', '1', '1r'),
						os.path.join('UnitTest3', 'UnitTest3_Serum_Rack01_RCM_190116', '11', 'pdata', '1', '1r'),
						os.path.join('UnitTest3', 'UnitTest3_Serum_Rack01_RCM_190116', '111', 'pdata', '1', '1r'),
						os.path.join('UnitTest3', 'UnitTest3_Serum_Rack01_RCM_190116', '120', 'pdata', '1', '1r'),
						os.path.join('UnitTest3', 'UnitTest3_Serum_Rack01_RCM_190116', '10', 'pdata', '1', '1r'),
						os.path.join('UnitTest3', 'UnitTest3_Serum_Rack01_RCM_190116', '181', 'pdata', '1', '1r'),
						os.path.join('UnitTest3', 'UnitTest3_Serum_Rack01_RCM_190116', '121', 'pdata', '1', '1r'),
						os.path.join('UnitTest3', 'UnitTest3_Serum_Rack01_RCM_190116', '110', 'pdata', '1', '1r'),
						os.path.join('UnitTest3', 'UnitTest3_Serum_Rack01_RCM_190116', '131', 'pdata', '1', '1r'),
						os.path.join('UnitTest3', 'UnitTest3_Serum_Rack01_RCM_190116', '100', 'pdata', '1', '1r'),
						os.path.join('UnitTest3', 'UnitTest3_Serum_Rack01_RCM_190116', '101', 'pdata', '1', '1r'),
						os.path.join('UnitTest3', 'UnitTest3_Serum_Rack01_RCM_190116', '130', 'pdata', '1', '1r'),
						os.path.join('UnitTest3', 'UnitTest3_Serum_Rack01_RCM_190116', '170', 'pdata', '1', '1r'),
						os.path.join('UnitTest3', 'UnitTest3_Serum_Rack01_RCM_190116', '141', 'pdata', '1', '1r'),
						os.path.join('UnitTest3', 'UnitTest3_Serum_Rack01_RCM_190116', '140', 'pdata', '1', '1r'),
						os.path.join('UnitTest3', 'UnitTest3_Serum_Rack01_RCM_190116', '171', 'pdata', '1', '1r')]
			expected = [os.path.join(pathHeader, x) for x in expected]

			pattern = '^1r$'
			pattern = re.compile(pattern)

			obtained = buildFileList(pathHeader, pattern)
			# Sorting filelists to account for differences in directory transversing between operating system
			obtained.sort()
			expected.sort()

			self.assertEqual(obtained, expected)

		with self.subTest(msg='with mzMLPaths'):
			pathHeader = os.path.join(self.pathHeader, 'ms', 'parameters_data')

			expected = ['UnitTest_RPOS_ToF10_U1W72_SR.mzML', 'UnitTest_RPOS_ToF10_U1W82_SR.mzML']

			expected = [os.path.join(pathHeader, x) for x in expected]

			pattern = '.+?\.mzML$'
			pattern = re.compile(pattern)

			obtained = buildFileList(pathHeader, pattern)
			# Sorting filelists to account for differences in directory transversing between operating system
			obtained.sort()
			expected.sort()

			self.assertEqual(obtained, expected)

	def test_extractParams_extractWatersRAWParams(self):

		pathHeader = os.path.join(self.pathHeader, 'ms', 'parameters_data')
		filePath = os.path.join(pathHeader, 'UnitTest1_LPOS_ToF02_S1W12_SR.raw')

		queryItems = dict()
		queryItems['_extern.inf'] = ['Resolution', 'Capillary (kV)','Sampling Cone', u'Source Temperature (°C)',
				   'Source Offset',u'Desolvation Temperature (°C)','Cone Gas Flow (L/Hr)','Desolvation Gas Flow (L/Hr)',
				   'LM Resolution','HM Resolution','Collision Energy', 'Polarity', 'Detector\t','Scan Time (sec)',
				   'Interscan Time (sec)','Start Mass','End Mass','Backing','Collision\t','TOF\t']
		queryItems['_HEADER.TXT'] = ['$$ Acquired Date:','$$ Acquired Time:', '$$ Instrument:']
		queryItems['_INLET.INF'] = ['ColumnType:', 'Column Serial Number:']

		expected = {'Column Serial Number:': '01573413615729',
					'$$ Acquired Date:': '27-Nov-2014',
					'$$ Acquired Time:': '13:29:48',
					'$$ Instrument:': 'XEVO-G2SQTOF#YDA121',
					'Backing': '3.52e0',
					'Capillary (kV)': '1.0000',
					'Collision': '1.23e-2',
					'Collision Energy': '6.000',
					'ColumnType:': 'ACQUITY UPLC® HSS T3 1.8µm',
					'Cone Gas Flow (L/Hr)': '150.0',
					'Desolvation Gas Flow (L/Hr)': '1000.0',
					'Desolvation Temperature (°C)': '600',
					'Detector': '3299',
					'End Mass': '1200.0',
					'File Path': os.path.join(pathHeader, 'UnitTest1_LPOS_ToF02_S1W12_SR.raw'),
					'HM Resolution': '15.0',
					'Interscan Time (sec)': '0.014',
					'LM Resolution': '10.0',
					'Polarity': 'ES-',
					'Resolution': '13000',
					'Sample File Name': 'UnitTest1_LPOS_ToF02_S1W12_SR',
					'Sampling Cone': '20.0000',
					'Scan Time (sec)': '0.150',
					'Source Offset': '80',
					'Source Temperature (°C)': '120',
					'Start Mass': '50.0',
					'TOF': '5.27e-7',
					'Warnings': ''}

		obtained = extractWatersRAWParams(filePath, queryItems)

		self.assertDictEqual(obtained, expected)

	def test_extractParams_extractWatersRAWParams_warns(self):

		pathHeader = os.path.join(self.pathHeader, 'ms', 'parameters_data')
		filePath = os.path.join(pathHeader, 'UnitTest1_LPOS_ToF02_S1W12_SR.raw')

		with self.subTest(msg='Missing parameter'):
			queryItems = dict()
			queryItems['_extern.inf'] = ['Resolution', 'Capillary (kV)','Sampling Cone', 'Unknown param']

			with self.assertWarnsRegex(UserWarning, 'Parameter Unknown param not found in file: '):
				obtained = extractWatersRAWParams(filePath, queryItems)

			self.assertEqual(obtained['Warnings'], 'Parameter Unknown param not found.')

		with self.subTest(msg='Missing file'):
			queryItems = dict()
			queryItems['unknown.file'] = ['Resolution', 'Capillary (kV)','Sampling Cone']

			with self.assertWarnsRegex(UserWarning, 'Unable to open '):
				obtained = extractWatersRAWParams(filePath, queryItems)

			self.assertEqual(obtained['Warnings'][:15], 'Unable to open ')

	def test_extractParams_extractBrukerparams(self):

		queryItems = dict()
		queryItems[os.path.join('..', '..', 'acqus')] = ['##OWNER=', '##$PULPROG=','##$RG=', '##$SW=','##$SFO1=', '##$TD=', '##$PROBHD=',
									 '##$BF1=', '##$O1=', '##$P=', '##$AUNM=']
		queryItems['procs'] = ['##$OFFSET=', '##$SW_p=', '##$NC_proc=', '##$SF=', '##$SI=', '##$BYTORDP=', '##$XDIM=']

		query = r'^\$\$\W(.+?)\W+([\w-]+@[\w-]+)$'
		acqTimeRE = re.compile(query)

		with self.subTest(msg='Topspin data'):
			pathHeader = os.path.join(self.pathHeader, 'nmr', 'UnitTest1', 'UnitTest1_Urine_Rack1_SLL_270814', '20')
			filePath = os.path.join(pathHeader, 'pdata', '1', '1r')

			expected = {'AUNM': 'au_ivdr_noesy',
						'Acquired Time': '2017-08-23 20:56:55.855 +0100',
						'BF1': '600.59',
						'BYTORDP': '0',
						'Computer': 'nmrsu@npc-nmr-01',
						'File Path': os.path.join(pathHeader, 'pdata', '1', '1r'),
						'NC_proc': '-7',
						'O1': '2823.78',
						'OFFSET': '14.79555',
						'OWNER': 'nmrsu',
						'P': '12.53',
						'PROBHD': 'Z814601_0070 (PA BBI 600S3 H-BB-D-05 Z)',
						'PULPROG': 'noesygppr1d',
						'RG': '84.66',
						'SF': '600.589947339331',
						'SFO1': '600.59282378',
						'SI': '131072',
						'SW': '20.0122783578804',
						'SW_p': '12019.2307692308',
						'Sample File Name': 'UnitTest1_Urine_Rack1_SLL_270814/20',
						'TD': '65536',
						'Warnings': '',
						'XDIM': '0'}

			obtained = extractBrukerparams(filePath, queryItems, acqTimeRE)

			self.assertDictEqual(obtained, expected)

		with self.subTest(msg='XWin-NMR procs'):
			pathHeader = os.path.join(self.pathHeader, 'nmr', 'UnitTest1', 'UnitTest1_Urine_Rack1_SLL_270814', '10')
			filePath = os.path.join(pathHeader, 'pdata', '1', '1r')

			expected = {'AUNM': 'best_au2',
						'Acquired Time': 'Tue Nov 19 17:28:15 2002 GMT',
						'BF1': '600.29',
						'BYTORDP': '1',
						'Computer': 'comet@bc-jkn-17',
						'File Path': os.path.join(pathHeader, 'pdata', '1', '1r'),
						'NC_proc': '-4',
						'O1': '2824.5',
						'OFFSET': '14.7869',
						'OWNER': 'comet',
						'P': '9',
						'PROBHD': '5mm FI TXB 1H-13C/15N-2H Z-GRD H8432/K0201 Z8432/0201',
						'PULPROG': 'noesypr1d',
						'RG': '128',
						'SF': '600.289936863629',
						'SFO1': '600.2928245',
						'SI': '32768',
						'SW': '20.022279592034',
						'SW_p': '12019.2307692308',
						'Sample File Name': 'UnitTest1_Urine_Rack1_SLL_270814/10',
						'TD': '32768',
						'Warnings': '',
						'XDIM': '8192'}

			obtained = extractBrukerparams(filePath, queryItems, acqTimeRE)

			self.assertDictEqual(obtained, expected)

	def test_extractParams_extractBrukerparams_warns(self):

		pathHeader = os.path.join(self.pathHeader, 'nmr', 'UnitTest3', 'UnitTest3_Serum_Rack01_RCM_190116', '10')
		filePath = os.path.join(pathHeader, 'pdata', '1', '1r')

		query = r'^\$\$\W(.+?)\W+([\w-]+@[\w-]+)$'
		acqTimeRE = re.compile(query)

		with self.subTest(msg='Missing parameter'):
			queryItems = dict()
			queryItems[os.path.join('..', '..', 'acqus')] = ['##$PULPROG=','##$RG=', '##$UNKNOWNPARAM=']

			with self.assertWarnsRegex(UserWarning, 'Parameter ##\$UNKNOWNPARAM= not found in file: '):
				obtained = extractBrukerparams(filePath, queryItems, acqTimeRE)

			self.assertEqual(obtained['Warnings'], 'Parameter ##$UNKNOWNPARAM= not found.')

		with self.subTest(msg='Missing file'):
			queryItems = dict()
			queryItems['MISSINGFILE'] = ['##$OFFSET=', '##$SW_p=', '##$NC_proc=', '##$SF=', '##$SI=', '##$BYTORDP=', '##$XDIM=']

			with self.assertWarnsRegex(UserWarning, 'Unable to open '):
				obtained = extractBrukerparams(filePath, queryItems, acqTimeRE)

			self.assertEqual(obtained['Warnings'][:15], 'Unable to open ')

	def test_extractParams_extractmzMLParams(self):

		pathHeader = os.path.join(self.pathHeader, 'ms', 'parameters_data')
		filePath = os.path.join(pathHeader, 'UnitTest_RPOS_ToF10_U1W72_SR.mzML')

		queryItems = ['startTimeStamp']
		expected = {'File Path': os.path.join(pathHeader, 'UnitTest_RPOS_ToF10_U1W72_SR.mzML'),
					'Sample File Name': 'UnitTest_RPOS_ToF10_U1W72_SR',
					'startTimeStamp': '2018-01-19T08:35:33Z',
					'Warnings': ''}

		obtained = extractmzMLParamsRegex(filePath, queryItems)

		self.assertDictEqual(obtained, expected)

	def test_extractParams_extractmzMLParams_warns(self):

		pathHeader = os.path.join(self.pathHeader, 'ms', 'parameters_data')
		filePath = os.path.join(pathHeader, 'UnitTest_RPOS_ToF10_U1W72_SR.mzML')

		with self.subTest(msg='Regex mzML parser - missing parameter'):
			queryItems = ['Unknown']
			with self.assertWarnsRegex(UserWarning, 'Parameter Unknown param not found in file: '):
				obtained = extractmzMLParamsRegex(filePath, queryItems)

			self.assertEqual(obtained['Warnings'], 'Parameter Unknown param not found.')

		from nPYc.utilities._getMetadataFrommzML import extractmzMLParams
		with self.subTest(msg='XML mzML parser - missing parameter'):
			queryItems = ['Unknown']

			with self.assertWarnsRegex(UserWarning, 'Parameter Unknown param not found in file: '):
				obtained = extractmzMLParams(filePath, queryItems)

			self.assertEqual(obtained['Warnings'], 'Parameter Unknown param not found.')

