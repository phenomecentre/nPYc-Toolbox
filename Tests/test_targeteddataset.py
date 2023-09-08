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

class test_targeteddataset_synthetic(unittest.TestCase):
	"""
	Test TargetedDataset object functions with synthetic data
	"""
	def setUp(self):

		# targetedDataset1 of 3 samples, 3 features
		# Batch are 1, 1 & 3
		# 1 excluded sample and 1 excluded feature (a copy of the first of each) [to test with targetedData2]
		# testcol_batch1 & testcol_batch3 test the column batch renaming when __add__
		# testcol_batch3 & specialcol_batch3 test the column batch detection
		self.targetedData1 = nPYc.TargetedDataset('', fileType='empty')
		self.targetedData1.sampleMetadata = pandas.DataFrame({'Sample File Name':['Unittest_targeted_file_001','Unittest_targeted_file_002','Unittest_targeted_file_003'],'Sample Base Name':['Unittest_targeted_file_001','Unittest_targeted_file_002','Unittest_targeted_file_003'],'AssayRole':[AssayRole.Assay,AssayRole.PrecisionReference,AssayRole.PrecisionReference],'SampleType':[SampleType.StudySample,SampleType.StudyPool,SampleType.ExternalReference],'MassLynx Row ID':[1,2,3],'Sample Name':['Sample1','Sample2','Sample3'],'Sample Type':['Analyte','Analyte','Analyte'],'Acqu Date':['26-May-17','26-May-17','26-May-17'],'Acqu Time':['16:42:57','16:58:49','17:14:41'],'Vial':['1:A,1','1:A,2','1:A,3'],'Instrument':['XEVO-TQS#UnitTest','XEVO-TQS#UnitTest','XEVO-TQS#UnitTest'],'Acquired Time':[pandas.Timestamp(datetime(2017,5,26,16,42,57)),pandas.Timestamp(datetime(2017,5,26,16,58,49)),pandas.Timestamp(datetime(2017,5,26,17,14,41))],'Run Order':[0,1,2],'Batch':[1,1,3],'Dilution':[50,100,100],'Correction Batch':[numpy.nan,numpy.nan,numpy.nan],'Sample ID':['Sample1','Sample2','Sample3'],'Subject ID': ['subject1', 'subject1', 'subject2'],'Exclusion Details':['','','']})
		self.targetedData1.featureMetadata = pandas.DataFrame({'Feature Name':['Feature1','Feature2','Feature3'],'TargetLynx Feature ID':[1,2,3],'calibrationEquation':['((area * responseFactor)-b)/a','10**((numpy.log10(area * responseFactor)-b)/a)',''],'calibrationMethod':[CalibrationMethod.backcalculatedIS, CalibrationMethod.backcalculatedIS, CalibrationMethod.noCalibration],'quantificationType':[QuantificationType.QuantOwnLabeledAnalogue, QuantificationType.QuantAltLabeledAnalogue, QuantificationType.Monitored],'unitCorrectionFactor':[10,1,1],'Unit':['\u00B5M','fg/\u00B5L','noUnit'],'Cpd Info_batch1':['Some info Feature1','Some info Feature2','Some info Feature3'],'Cpd Info_batch3':['Some info Feature1','Some info Feature2','Some info Feature3'],'Noise (area)_batch1':['','',''],'Noise (area)_batch3':['','',''],'LLOQ_batch1':[10,50,100],'LLOQ_batch3':[10,50,200],'ULOQ_batch1':[1000,5000,10000],'ULOQ_batch3':[1000,5000,25000],'a_batch1':['','',''],'a_batch3':['','',''],'b_batch1':['','',''],'b_batch3':['','',''],'r_batch1':['','',''],'r_batch3':['','',''],'r2_batch1':['','',''],'r2_batch3':['','',''],'testcol_batch1':['testcol_batch1','testcol_batch1','testcol_batch1'],'testcol_batch3':['testcol_batch3','testcol_batch3','testcol_batch3'],'specialcol_batch3':['specialcol_batch3','specialcol_batch3','specialcol_batch3'],'extID1': ['F1','F2','F3'],'extID2': ['ID1','ID2','ID3']})
		self.targetedData1._intensityData = numpy.array([[10, 20, 30], [10, 20, 30], [3, 15, 30]])
		self.targetedData1.expectedConcentration = pandas.DataFrame({'Feature1':[9, 22, 30], 'Feature2':[11, 18, 30], 'Feature3':[3, 15, 30]})
		self.targetedData1.Attributes['methodName'] = 'UnitTestMethod'
		self.targetedData1.Attributes['externalID'] = ['extID1', 'extID2']
		self.targetedData1.Attributes['sampleMetadataNotExported']  = []
		self.targetedData1.Attributes['featureMetadataNotExported'] = []
		# excluded data
		self.targetedData1.sampleMetadataExcluded = []
		self.targetedData1.intensityDataExcluded = []
		self.targetedData1.featureMetadataExcluded = []
		self.targetedData1.expectedConcentrationExcluded = []
		self.targetedData1.excludedFlag = []
		self.targetedData1.sampleMetadataExcluded.append(self.targetedData1.sampleMetadata[[True,False,False]])
		self.targetedData1.intensityDataExcluded.append(self.targetedData1._intensityData[0,:])
		self.targetedData1.featureMetadataExcluded.append(self.targetedData1.featureMetadata)
		self.targetedData1.expectedConcentrationExcluded.append(self.targetedData1.expectedConcentration.loc[[0], :])
		self.targetedData1.excludedFlag.append('Samples')
		self.targetedData1.featureMetadataExcluded.append(self.targetedData1.featureMetadata[[True,False,False]])
		self.targetedData1.intensityDataExcluded.append(self.targetedData1._intensityData[:,0])
		self.targetedData1.sampleMetadataExcluded.append(self.targetedData1.sampleMetadata)
		self.targetedData1.expectedConcentrationExcluded.append(self.targetedData1.expectedConcentration.iloc[:, [0]])
		self.targetedData1.excludedFlag.append('Features')
		# calibration import batch1
		td1b1_calibSampleMetadata = pandas.DataFrame({'Sample File Name': ['Unittest_targeted_file_009'], 'MassLynx Row ID': [9], 'Sample Name': ['calib1'],'Sample Type': ['Calibrant'], 'Acqu Date': ['26-May-17'], 'Acqu Time': ['17:29:57'], 'Vial': ['1:A,9'],'Instrument': ['XEVO-TQS#UnitTest'], 'Acquired Time': [pandas.Timestamp(datetime(2017, 5, 26, 17, 29, 57))],'Run Order': [3], 'Batch': [1]})
		td1b1_calibFeatureMetadata = self.targetedData1.featureMetadata
		td1b1_calibIntensityData = numpy.array([[10, 20, 30]])
		td1b1_calibExpectedConcentration = pandas.DataFrame({'Feature1': [1000.0], 'Feature2': [1000.0], 'Feature3': [1000.0]})
		td1b1_calibPeakResponse = pandas.DataFrame({'Feature1': [0.067860], 'Feature2': [0.024966], 'Feature3': [0.133737]})
		td1b1_calibPeakArea = pandas.DataFrame({'Feature1': [119.771404], 'Feature2': [74.349138], 'Feature3': [107.250678]})
		td1b1_calibPeakConcentrationDeviation = pandas.DataFrame({'Feature1': [972.044215], 'Feature2': [309.656566], 'Feature3': [6.642807]})
		td1b1_calibPeakIntegrationFlag = pandas.DataFrame({'Feature1': ['MMX'], 'Feature2': ['bdX'], 'Feature3': ['MM']})
		td1b1_calibPeakRT = pandas.DataFrame({'Feature1': [3.5], 'Feature2': [7.4], 'Feature3': [10.2]})
		td1b1_calibPeakInfo = {'peakResponse': td1b1_calibPeakResponse, 'peakArea': td1b1_calibPeakArea,'peakConcentrationDeviation': td1b1_calibPeakConcentrationDeviation,'peakIntegration': td1b1_calibPeakIntegrationFlag, 'peakRT': td1b1_calibPeakRT}
		calibB1 = dict({'calibSampleMetadata': td1b1_calibSampleMetadata, 'calibFeatureMetadata': td1b1_calibFeatureMetadata,'calibIntensityData': td1b1_calibIntensityData,'calibExpectedConcentration': td1b1_calibExpectedConcentration, 'calibPeakInfo': td1b1_calibPeakInfo})
		# calibration import batch 3
		td1b3_calibSampleMetadata = pandas.DataFrame({'Sample File Name': ['Unittest_targeted_file_010'], 'MassLynx Row ID': [10], 'Sample Name': ['calib2'],'Sample Type': ['Calibrant'], 'Acqu Date': ['26-May-17'], 'Acqu Time': ['17:45:49'], 'Vial': ['1:A,10'],'Instrument': ['XEVO-TQS#UnitTest'], 'Acquired Time': [pandas.Timestamp(datetime(2017, 5, 26, 17, 45, 49))],'Run Order': [4], 'Batch': [3]})
		td1b3_calibFeatureMetadata = self.targetedData1.featureMetadata
		td1b3_calibIntensityData = numpy.array([[3, 15, 30]])
		td1b3_calibExpectedConcentration = pandas.DataFrame({'Feature1': [2500.0], 'Feature2': [2500.0], 'Feature3': [2500.0]})
		td1b3_calibPeakResponse = pandas.DataFrame({'Feature1': [0.071964], 'Feature2': [0.046445], 'Feature3': [0.663232]})
		td1b3_calibPeakArea = pandas.DataFrame({'Feature1': [10.363023], 'Feature2': [6.977580], 'Feature3': [10.679743]})
		td1b3_calibPeakConcentrationDeviation = pandas.DataFrame({'Feature1': [369.509230], 'Feature2': [44.229980], 'Feature3': [-6.516648]})
		td1b3_calibPeakIntegrationFlag = pandas.DataFrame({'Feature1': ['bbX'], 'Feature2': ['bbX'], 'Feature3': ['bd']})
		td1b3_calibPeakRT = pandas.DataFrame({'Feature1': [3.6], 'Feature2': [7.1], 'Feature3': [10.9]})
		td1b3_calibPeakInfo = {'peakResponse': td1b3_calibPeakResponse, 'peakArea': td1b3_calibPeakArea,'peakConcentrationDeviation': td1b3_calibPeakConcentrationDeviation,'peakIntegration': td1b3_calibPeakIntegrationFlag, 'peakRT': td1b3_calibPeakRT}
		calibB3 = dict({'calibSampleMetadata': td1b3_calibSampleMetadata, 'calibFeatureMetadata': td1b3_calibFeatureMetadata,'calibIntensityData': td1b3_calibIntensityData, 'calibExpectedConcentration': td1b3_calibExpectedConcentration, 'calibPeakInfo': td1b3_calibPeakInfo})
		# calibration
		self.targetedData1.calibration = [calibB1, calibB3]
		self.targetedData1.VariableType = VariableType.Discrete
		self.targetedData1.initialiseMasks()

		# targetedDataset2 of 4 samples, 4 features (2 overlapping with targetedDataset1)
		# Batch are 2, 2, 5 & 5
		# 1 excluded sample and 1 excluded feature (a copy of the first of each)
		# testcol_batch2 & testcol_batch5 test column batch rename (with testcol_batch1 & testcol_batch3 in targetedDataset1)
		# specialcol_batch5 test column batch rename with specialcol_batch3 in targetedDataset1
		self.targetedData2 = nPYc.TargetedDataset('', fileType='empty')
		self.targetedData2.sampleMetadata = pandas.DataFrame({'Sample File Name': ['Unittest_targeted_file_004', 'Unittest_targeted_file_005', 'Unittest_targeted_file_006', 'Unittest_targeted_file_007'],'Sample Base Name': ['Unittest_targeted_file_004', 'Unittest_targeted_file_005', 'Unittest_targeted_file_006', 'Unittest_targeted_file_007'],'AssayRole':[AssayRole.Assay,AssayRole.Assay,AssayRole.PrecisionReference,AssayRole.PrecisionReference],'SampleType':[SampleType.StudySample,SampleType.StudySample,SampleType.StudyPool,SampleType.ExternalReference], 'MassLynx Row ID': [4, 5, 6, 7], 'Sample Name': ['Sample4', 'Sample5', 'Sample6', 'Sample7'], 'Sample Type': ['Analyte', 'Analyte', 'Analyte', 'Analyte'], 'Acqu Date': ['25-May-17', '25-May-17', '25-May-17', '25-May-17'], 'Acqu Time': ['12:42:57', '12:58:49', '13:14:41', '13:29:59'], 'Vial': ['1:A,4', '1:A,5', '1:A,6', '1:A,7'], 'Instrument': ['XEVO-TQS#UnitTest','XEVO-TQS#UnitTest','XEVO-TQS#UnitTest','XEVO-TQS#UnitTest'], 'Acquired Time': [datetime(2017, 5, 25, 12, 42, 57),datetime(2017, 5, 25, 12, 58, 49),datetime(2017, 5, 25, 13, 14, 41),datetime(2017, 5, 25, 13, 29, 59)], 'Run Order': [0, 1, 2, 3], 'Batch': [2, 2, 5, 5], 'Dilution':[100,90,110,100],'Correction Batch':[numpy.nan,numpy.nan,numpy.nan,numpy.nan],'Sample ID':['Sample4', 'Sample5', 'Sample6', 'Sample7'],'Subject ID':['subject1','subject3','subject3','subject3'],'Exclusion Details':['','','','']})
		self.targetedData2.featureMetadata = pandas.DataFrame({'Feature Name':['Feature2','Feature3','Feature4','Feature5'],'TargetLynx Feature ID':[2,3,4,5],'calibrationEquation':['10**((numpy.log10(area * responseFactor)-b)/a)','','((area * responseFactor)-b)/a','((area * responseFactor)-b)/a'],'calibrationMethod':[CalibrationMethod.backcalculatedIS,CalibrationMethod.noCalibration,CalibrationMethod.backcalculatedIS,CalibrationMethod.backcalculatedIS],'quantificationType':[QuantificationType.QuantAltLabeledAnalogue, QuantificationType.Monitored, QuantificationType.QuantOwnLabeledAnalogue, QuantificationType.QuantOwnLabeledAnalogue],'unitCorrectionFactor':[1,1,5,10],'Unit':['fg/\u00B5L','noUnit','\u00B5M','\u00B5M'],'Cpd Info_batch2':['Some info Feature2','Some info Feature3','Some info Feature4','Some info Feature5'],'Cpd Info_batch5':['Some info Feature2','Some info Feature3','Some info Feature4','Some info Feature5'],'Noise (area)_batch2':['','','',''],'Noise (area)_batch5':['','','',''],'LLOQ_batch2':[0.,0.,0.,0.],'LLOQ_batch5':[0.,0.,0.,0.],'ULOQ_batch2':[50000.,50000.,50000.,50000.],'ULOQ_batch5':[50000.,50000.,50000.,50000.],'a_batch2':['','','',''],'a_batch5':['','','',''],'b_batch2':['','','',''],'b_batch5':['','','',''],'r_batch2':['','','',''],'r_batch5':['','','',''],'r2_batch2':['','','',''],'r2_batch5':['','','',''],'testcol_batch2':['testcol_batch2','testcol_batch2','testcol_batch2','testcol_batch2'],'testcol_batch5':['testcol_batch5','testcol_batch5','testcol_batch5','testcol_batch5'],'specialcol_batch5':['specialcol_batch5','specialcol_batch5','specialcol_batch5','specialcol_batch5'],'extID1': ['F2','F3','F4','F5'],'extID2': ['ID2','ID3','ID4','ID5']})
		self.targetedData2._intensityData = numpy.array([[20, 30, 40, 50], [20, 30, 40, 50], [15, 30, 40, 50], [15, 30, 40, 50]])
		self.targetedData2.expectedConcentration = pandas.DataFrame({'Feature2': [19, 31, 39, 51], 'Feature3': [21, 29, 41, 49], 'Feature4': [16, 29, 41, 49], 'Feature5':[15, 30, 40, 50]})
		self.targetedData2.Attributes['methodName'] = 'UnitTestMethod'
		self.targetedData2.Attributes['externalID'] = ['extID1', 'extID2']
		self.targetedData2.Attributes['sampleMetadataNotExported']  = []
		self.targetedData2.Attributes['featureMetadataNotExported'] = []
		# excluded data
		self.targetedData2.sampleMetadataExcluded = []
		self.targetedData2.intensityDataExcluded = []
		self.targetedData2.featureMetadataExcluded = []
		self.targetedData2.expectedConcentrationExcluded = []
		self.targetedData2.excludedFlag = []
		self.targetedData2.sampleMetadataExcluded.append(self.targetedData2.sampleMetadata[[True,False,False,False]])
		self.targetedData2.intensityDataExcluded.append(self.targetedData2._intensityData[0,:])
		self.targetedData2.featureMetadataExcluded.append(self.targetedData2.featureMetadata)
		self.targetedData2.expectedConcentrationExcluded.append(self.targetedData2.expectedConcentration.loc[[0], :])
		self.targetedData2.excludedFlag.append('Samples')
		self.targetedData2.featureMetadataExcluded.append(self.targetedData2.featureMetadata[[True,False,False,False]])
		self.targetedData2.intensityDataExcluded.append(self.targetedData2._intensityData[:,0])
		self.targetedData2.sampleMetadataExcluded.append(self.targetedData2.sampleMetadata)
		self.targetedData2.expectedConcentrationExcluded.append(self.targetedData2.expectedConcentration.iloc[:, [0]])
		self.targetedData2.excludedFlag.append('Features')
		# calibration import batch 2
		td2b2_calibSampleMetadata = pandas.DataFrame({'Sample File Name': ['Unittest_targeted_file_011'], 'MassLynx Row ID': [11], 'Sample Name': ['calib3'],'Sample Type': ['Calibrant'], 'Acqu Date': ['25-May-17'], 'Acqu Time': ['14:00:23'], 'Vial': ['1:A,11'],'Instrument': ['XEVO-TQS#UnitTest'], 'Acquired Time': [datetime(2017, 5, 25, 14, 0, 23)], 'Run Order': [4], 'Batch': [2]})
		td2b2_calibFeatureMetadata = self.targetedData2.featureMetadata
		td2b2_calibIntensityData = numpy.array([[20, 30, 40, 50]])
		td2b2_calibExpectedConcentration = pandas.DataFrame({'Feature2': [1000.0], 'Feature3': [1000.0], 'Feature4': [1000.0], 'Feature5': [1000.0]})
		td2b2_calibPeakResponse = pandas.DataFrame({'Feature2': [10.363023], 'Feature3': [6.977580], 'Feature4': [10.679743], 'Feature5': [9.866470]})
		td2b2_calibPeakArea = pandas.DataFrame({'Feature2': [115276.820], 'Feature3': [28030.086], 'Feature4': [41121.988], 'Feature5': [78899.891]})
		td2b2_calibPeakConcentrationDeviation = pandas.DataFrame({'Feature2': [1.585510], 'Feature3': [2.100069], 'Feature4': [0.531491], 'Feature5': [0.035385]})
		td2b2_calibPeakIntegrationFlag = pandas.DataFrame({'Feature2': ['bb'], 'Feature3': ['bb'], 'Feature4': ['bb'], 'Feature5': ['bb']})
		td2b2_calibPeakRT = pandas.DataFrame({'Feature2': [1.5], 'Feature3': [8.4], 'Feature4': [10.9], 'Feature5':[15.3]})
		td2b2_calibPeakInfo = {'peakResponse': td2b2_calibPeakResponse, 'peakArea': td2b2_calibPeakArea, 'peakConcentrationDeviation': td2b2_calibPeakConcentrationDeviation, 'peakIntegration': td2b2_calibPeakIntegrationFlag, 'peakRT': td2b2_calibPeakRT}
		calibB2 = dict({'calibSampleMetadata': td2b2_calibSampleMetadata, 'calibFeatureMetadata': td2b2_calibFeatureMetadata, 'calibIntensityData': td2b2_calibIntensityData, 'calibExpectedConcentration': td2b2_calibExpectedConcentration, 'calibPeakInfo': td2b2_calibPeakInfo})
		# calibration import batch 5
		td2b5_calibSampleMetadata = pandas.DataFrame({'Sample File Name': ['Unittest_targeted_file_012'], 'MassLynx Row ID': [12], 'Sample Name': ['calib4'],'Sample Type': ['Calibrant'], 'Acqu Date': ['25-May-17'], 'Acqu Time': ['14:15:50'], 'Vial': ['1:A,12'], 'Instrument': ['XEVO-TQS#UnitTest'], 'Acquired Time': [datetime(2017, 5, 25, 14, 15, 50)], 'Run Order': [5], 'Batch': [5]})
		td2b5_calibFeatureMetadata = self.targetedData2.featureMetadata
		td2b5_calibIntensityData = numpy.array([[15, 30, 40, 50]])
		td2b5_calibExpectedConcentration = pandas.DataFrame({'Feature2': [2500.0], 'Feature3': [2500.0], 'Feature4': [2500.0], 'Feature5': [2500.0]})
		td2b5_calibPeakResponse = pandas.DataFrame({'Feature2': [119.771404], 'Feature3': [74.349138], 'Feature4': [107.250678], 'Feature5': [82.681753]})
		td2b5_calibPeakArea = pandas.DataFrame({'Feature2': [10606.862], 'Feature3': [15993.164], 'Feature4': [30870.994], 'Feature5': [3400.287]})
		td2b5_calibPeakConcentrationDeviation = pandas.DataFrame({'Feature2': [50000.0], 'Feature3': [50000.0], 'Feature4': [50000.0], 'Feature5': [50000.0]})
		td2b5_calibPeakIntegrationFlag = pandas.DataFrame({'Feature2': ['bb'], 'Feature3': ['bb'], 'Feature4': ['bb'], 'Feature5': ['bb']})
		td2b5_calibPeakRT = pandas.DataFrame({'Feature2': [0.85], 'Feature3': [9.2], 'Feature4': [9.9], 'Feature5': [14.9]})
		td2b5_calibPeakInfo = {'peakResponse': td2b5_calibPeakResponse, 'peakArea': td2b5_calibPeakArea, 'peakConcentrationDeviation': td2b5_calibPeakConcentrationDeviation, 'peakIntegration': td2b5_calibPeakIntegrationFlag, 'peakRT': td2b5_calibPeakRT}
		calibB5 = dict({'calibSampleMetadata': td2b5_calibSampleMetadata, 'calibFeatureMetadata': td2b5_calibFeatureMetadata, 'calibIntensityData': td2b5_calibIntensityData, 'calibExpectedConcentration': td2b5_calibExpectedConcentration, 'calibPeakInfo': td2b5_calibPeakInfo})
		# calibration
		self.targetedData2.calibration = [calibB2, calibB5]
		self.targetedData2.VariableType = VariableType.Discrete
		self.targetedData2.initialiseMasks()

		# targetedDataset3 of 1 sample 1 feature (1 overlapping with targetedDataset2)
		# testcol_batch1 has identical name as in targetedDataset1 to test column batch rename in __add__
		# some columns left without _batchX to test the first round of renaming
		self.targetedData3 = nPYc.TargetedDataset('', fileType='empty')
		self.targetedData3.sampleMetadata = pandas.DataFrame({'Sample File Name': ['Unittest_targeted_file_008'], 'Sample Base Name': ['Unittest_targeted_file_008'], 'MassLynx Row ID': [8], 'Sample Name': ['Sample8'],'AssayRole':[AssayRole.Assay],'SampleType':[SampleType.StudySample], 'Sample Type': ['Analyte'], 'Acqu Date': ['25-May-17'], 'Acqu Time': ['13:45:05'], 'Vial': ['1:A,8'], 'Instrument': ['XEVO-TQS#UnitTest'], 'Acquired Time': [datetime(2017, 5, 25, 13, 45, 5)], 'Run Order': [0], 'Batch': [1],'Dilution':[100],'Correction Batch':[numpy.nan],'Sample ID':['Sample8'],'Subject ID':['subject4'],'Exclusion Details':['']})
		self.targetedData3.featureMetadata = pandas.DataFrame({'Feature Name':['Feature5'],'TargetLynx Feature ID':[5],'calibrationEquation':['((area * responseFactor)-b)/a'],'calibrationMethod':[CalibrationMethod.backcalculatedIS],'quantificationType':[QuantificationType.QuantOwnLabeledAnalogue],'unitCorrectionFactor':[10],'Unit':['\u00B5M'],'Cpd Info':['Some info Feature5'],'Noise (area)':[''],'LLOQ':[0.],'ULOQ':[50000.],'a':[''],'b':[''],'r':[''],'r2':[''],'testcol_batch1':['testcol_batch1'],'extID1':['F5'],'extID2':['ID5']})
		self.targetedData3._intensityData = numpy.array([[50]])
		self.targetedData3.expectedConcentration = pandas.DataFrame({'Feature5': [75]})
		self.targetedData3.Attributes['methodName'] = 'UnitTestMethod'
		self.targetedData3.Attributes['externalID'] = ['extID1', 'extID2']
		self.targetedData3.Attributes['sampleMetadataNotExported']  = []
		self.targetedData3.Attributes['featureMetadataNotExported'] = []
		# calibration
		self.targetedData3.calibration = dict()
		self.targetedData3.calibration['calibSampleMetadata'] = pandas.DataFrame({'Sample File Name': ['Unittest_targeted_file_013'], 'MassLynx Row ID': [13], 'Sample Name': ['calib5'], 'Sample Type': ['Calibrant'], 'Acqu Date': ['25-May-17'], 'Acqu Time': ['14:30:17'], 'Vial': ['1:B,1'],	 'Instrument': ['XEVO-TQS#UnitTest'], 'Acquired Time': [datetime(2017, 5, 26, 14, 30, 17)],'Run Order': [1], 'Batch': [1]})
		self.targetedData3.calibration['calibFeatureMetadata'] = self.targetedData3.featureMetadata
		self.targetedData3.calibration['calibIntensityData'] = numpy.array([[1010.0]])
		self.targetedData3.calibration['calibExpectedConcentration'] = pandas.DataFrame({'Feature5': [75]})
		self.targetedData3.VariableType = VariableType.Discrete
		self.targetedData3.initialiseMasks()

		# expectedAddDataset which is targetedDataset1 + targetedDataset2 + targetedDataset3
		self.expectedAddDataset = nPYc.TargetedDataset('', fileType='empty')
		self.expectedAddDataset.sampleMetadata = pandas.DataFrame({'Sample File Name': ['Unittest_targeted_file_001','Unittest_targeted_file_002','Unittest_targeted_file_003','Unittest_targeted_file_004','Unittest_targeted_file_005','Unittest_targeted_file_006','Unittest_targeted_file_007','Unittest_targeted_file_008'],'Sample Base Name': ['Unittest_targeted_file_001','Unittest_targeted_file_002','Unittest_targeted_file_003','Unittest_targeted_file_004','Unittest_targeted_file_005','Unittest_targeted_file_006','Unittest_targeted_file_007','Unittest_targeted_file_008'],'AssayRole':[AssayRole.Assay, AssayRole.PrecisionReference, AssayRole.PrecisionReference, AssayRole.Assay, AssayRole.Assay, AssayRole.PrecisionReference, AssayRole.PrecisionReference, AssayRole.Assay],'SampleType':[SampleType.StudySample, SampleType.StudyPool, SampleType.ExternalReference, SampleType.StudySample, SampleType.StudySample, SampleType.StudyPool, SampleType.ExternalReference, SampleType.StudySample],'MassLynx Row ID': [1, 2, 3, 4, 5, 6, 7, 8],'Sample Name': ['Sample1', 'Sample2', 'Sample3', 'Sample4','Sample5', 'Sample6', 'Sample7', 'Sample8'],'Sample Type': ['Analyte', 'Analyte', 'Analyte', 'Analyte', 'Analyte', 'Analyte', 'Analyte', 'Analyte'],'Acqu Date': ['26-May-17', '26-May-17', '26-May-17','25-May-17', '25-May-17', '25-May-17','25-May-17', '25-May-17'],'Acqu Time': ['16:42:57', '16:58:49', '17:14:41', '12:42:57','12:58:49', '13:14:41', '13:29:59','13:45:05'],'Vial': ['1:A,1', '1:A,2', '1:A,3', '1:A,4', '1:A,5','1:A,6', '1:A,7', '1:A,8'],'Instrument': ['XEVO-TQS#UnitTest', 'XEVO-TQS#UnitTest','XEVO-TQS#UnitTest', 'XEVO-TQS#UnitTest','XEVO-TQS#UnitTest', 'XEVO-TQS#UnitTest','XEVO-TQS#UnitTest', 'XEVO-TQS#UnitTest'],'Acquired Time': [datetime(2017, 5, 26, 16, 42, 57), datetime(2017, 5, 26, 16, 58, 49), datetime(2017, 5, 26, 17, 14, 41), datetime(2017, 5, 25, 12, 42, 57), datetime(2017, 5, 25, 12, 58, 49), datetime(2017, 5, 25, 13, 14, 41), datetime(2017, 5, 25, 13, 29, 59), datetime(2017, 5, 25, 13, 45, 5)],'Run Order': [5, 6, 7, 0, 1, 2, 3, 4],'Batch': [1, 1, 2, 3, 3, 4, 4, 5],'Dilution':[50,100,100,100,90,110,100,100],'Correction Batch':[numpy.nan,numpy.nan,numpy.nan,numpy.nan,numpy.nan,numpy.nan,numpy.nan,numpy.nan],'Sample ID':['Sample1','Sample2','Sample3','Sample4','Sample5','Sample6','Sample7','Sample8'],'Subject ID':['subject1','subject1','subject2','subject1','subject3','subject3','subject3','subject4'],'Exclusion Details':['','','','','','','','']})
		self.expectedAddDataset.featureMetadata = pandas.DataFrame({'Feature Name': ['Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5'],'TargetLynx Feature ID_batch1': [1., 2., 3., numpy.nan, numpy.nan],'TargetLynx Feature ID_batch3': [numpy.nan, 2., 3., 4., 5.],'TargetLynx Feature ID_batch5': [numpy.nan,numpy.nan,numpy.nan,numpy.nan, 5.],'calibrationEquation_batch1': ['((area * responseFactor)-b)/a', '10**((numpy.log10(area * responseFactor)-b)/a)', '',numpy.nan,numpy.nan],'calibrationEquation_batch3': [numpy.nan, '10**((numpy.log10(area * responseFactor)-b)/a)', '','((area * responseFactor)-b)/a', '((area * responseFactor)-b)/a'],'calibrationEquation_batch5': [numpy.nan,numpy.nan,numpy.nan,numpy.nan, '((area * responseFactor)-b)/a'],'calibrationMethod': [CalibrationMethod.backcalculatedIS, CalibrationMethod.backcalculatedIS, CalibrationMethod.noCalibration, CalibrationMethod.backcalculatedIS, CalibrationMethod.backcalculatedIS], 'quantificationType': [QuantificationType.QuantOwnLabeledAnalogue, QuantificationType.QuantAltLabeledAnalogue, QuantificationType.Monitored, QuantificationType.QuantOwnLabeledAnalogue, QuantificationType.QuantOwnLabeledAnalogue],'unitCorrectionFactor_batch1': [10, 1, 1, numpy.nan,numpy.nan],'unitCorrectionFactor_batch3': [numpy.nan, 1, 1, 5, 10],'unitCorrectionFactor_batch5': [numpy.nan,numpy.nan,numpy.nan,numpy.nan, 10],'Unit': ['\u00B5M', 'fg/\u00B5L', 'noUnit', '\u00B5M', '\u00B5M'],'Cpd Info_batch1': ['Some info Feature1', 'Some info Feature2', 'Some info Feature3', numpy.nan, numpy.nan],'Cpd Info_batch2': ['Some info Feature1', 'Some info Feature2', 'Some info Feature3', numpy.nan, numpy.nan],'Cpd Info_batch3': [numpy.nan, 'Some info Feature2', 'Some info Feature3', 'Some info Feature4','Some info Feature5'],'Cpd Info_batch4': [numpy.nan, 'Some info Feature2', 'Some info Feature3', 'Some info Feature4', 'Some info Feature5'], 'Cpd Info_batch5': [numpy.nan, numpy.nan, numpy.nan, numpy.nan, 'Some info Feature5'], 'Noise (area)_batch1': ['', '', '', numpy.nan, numpy.nan],'Noise (area)_batch2': ['', '', '', numpy.nan, numpy.nan], 'Noise (area)_batch3': [numpy.nan, '', '', '', ''], 'Noise (area)_batch4': [numpy.nan, '', '', '', ''], 'Noise (area)_batch5': [numpy.nan, numpy.nan, numpy.nan, numpy.nan, ''], 'LLOQ_batch1': [10., 50., 100., numpy.nan, numpy.nan],'LLOQ_batch2': [10.,50.,200., numpy.nan, numpy.nan], 'LLOQ_batch3': [numpy.nan, 0.,0.,0.,0.],'LLOQ_batch4': [numpy.nan, 0.,0.,0.,0.], 'LLOQ_batch5': [numpy.nan, numpy.nan, numpy.nan, numpy.nan, 0.], 'ULOQ_batch1': [1000.,5000.,10000., numpy.nan, numpy.nan],'ULOQ_batch2': [1000.,5000.,25000., numpy.nan, numpy.nan], 'ULOQ_batch3': [numpy.nan, 50000.,50000.,50000.,50000.],'ULOQ_batch4': [numpy.nan, 50000.,50000.,50000.,50000.],'ULOQ_batch5': [numpy.nan, numpy.nan, numpy.nan, numpy.nan, 50000.],'a_batch1': ['', '', '', numpy.nan, numpy.nan], 'a_batch2': ['', '', '', numpy.nan, numpy.nan],'a_batch3': [numpy.nan, '', '', '', ''], 'a_batch4': [numpy.nan, '', '', '', ''], 'a_batch5': [numpy.nan, numpy.nan, numpy.nan, numpy.nan, ''],'b_batch1': ['', '', '', numpy.nan, numpy.nan], 'b_batch2': ['', '', '', numpy.nan, numpy.nan], 'b_batch3': [numpy.nan, '', '', '', ''], 'b_batch4': [numpy.nan, '', '', '', ''],'b_batch5': [numpy.nan, numpy.nan, numpy.nan, numpy.nan, ''], 'r_batch1': ['', '', '', numpy.nan, numpy.nan], 'r_batch2': ['', '', '', numpy.nan, numpy.nan], 'r_batch3': [numpy.nan, '', '', '', ''], 'r_batch4': [numpy.nan, '', '', '', ''],'r_batch5': [numpy.nan, numpy.nan, numpy.nan, numpy.nan, ''],'r2_batch1': ['', '', '', numpy.nan, numpy.nan], 'r2_batch2': ['', '', '', numpy.nan, numpy.nan],'r2_batch3': [numpy.nan, '', '', '', ''], 'r2_batch4': [numpy.nan, '', '', '', ''],'r2_batch5': [numpy.nan, numpy.nan, numpy.nan, numpy.nan, ''],'testcol_batch1': ['testcol_batch1', 'testcol_batch1', 'testcol_batch1', numpy.nan, numpy.nan],'testcol_batch2': ['testcol_batch3', 'testcol_batch3', 'testcol_batch3', numpy.nan, numpy.nan],'testcol_batch3': [numpy.nan, 'testcol_batch2', 'testcol_batch2', 'testcol_batch2', 'testcol_batch2'],'testcol_batch4': [numpy.nan, 'testcol_batch5', 'testcol_batch5', 'testcol_batch5', 'testcol_batch5'],'testcol_batch5': [numpy.nan, numpy.nan, numpy.nan, numpy.nan, 'testcol_batch1'],'specialcol_batch2': ['specialcol_batch3', 'specialcol_batch3', 'specialcol_batch3', numpy.nan, numpy.nan],'specialcol_batch4': [numpy.nan, 'specialcol_batch5', 'specialcol_batch5', 'specialcol_batch5','specialcol_batch5'], 'extID1': ['F1', 'F2', 'F3', 'F4', 'F5'], 'extID2': ['ID1', 'ID2', 'ID3', 'ID4', 'ID5']})
		self.expectedAddDataset._intensityData = numpy.array([[10, 20, 30, numpy.nan, numpy.nan], [10, 20, 30, numpy.nan, numpy.nan], [3, 15, 30, numpy.nan, numpy.nan],[numpy.nan, 20, 30, 40, 50], [numpy.nan, 20, 30, 40, 50], [numpy.nan, 15, 30, 40, 50],[numpy.nan, 15, 30, 40, 50], [numpy.nan, numpy.nan, numpy.nan, numpy.nan, 50]])
		self.expectedAddDataset.expectedConcentration = pandas.DataFrame({'Feature1': [9., 22., 30., numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan], 'Feature2': [11., 18., 30., 19., 31., 39., 51., numpy.nan], 'Feature3': [3., 15., 30., 21., 29., 41., 49., numpy.nan], 'Feature4': [numpy.nan, numpy.nan, numpy.nan, 16., 29., 41., 49., numpy.nan], 'Feature5': [numpy.nan, numpy.nan, numpy.nan, 15., 30., 40., 50., 75.]})
		self.expectedAddDataset.Attributes = copy.deepcopy(self.targetedData1.Attributes)
		self.expectedAddDataset.Attributes['methodName'] = 'UnitTestMethod'
		self.expectedAddDataset.Attributes['externalID'] = ['extID1', 'extID2']
		self.expectedAddDataset.Attributes['Log'] = self.targetedData1.Attributes['Log'] + [['Here first dataset validation import'],['Here first dataset, targeted validation import']] + self.targetedData2.Attributes['Log'] + [['Here second dataset validation import'],['Here second dataset, targeted validation import'],['Here merge Masks Initialised'], ['Here merge validated as dataset'],['Here merge validated as a targeted dataset'], ['Here end of first concatenation log'],['Here next start of _add_ dataset validation'],['Here next _add_ targeted validation']] + self.targetedData3.Attributes['Log'] + [['Here validation as dataset at start of second add'],['Here validation as targeted at start of second add'],['Here Masks Initialised'], ['Here validation as dataset at end of second _add_'],['Here validation as targeted at the end of second _add_'],['Here end of dataset concatenation log']]

		# excluded data
		self.expectedAddDataset.sampleMetadataExcluded = [self.targetedData1.sampleMetadataExcluded, self.targetedData2.sampleMetadataExcluded, []]
		self.expectedAddDataset.intensityDataExcluded = [self.targetedData1.intensityDataExcluded, self.targetedData2.intensityDataExcluded, []]
		self.expectedAddDataset.featureMetadataExcluded = [self.targetedData1.featureMetadataExcluded, self.targetedData2.featureMetadataExcluded, []]
		self.expectedAddDataset.expectedConcentrationExcluded = [self.targetedData1.expectedConcentrationExcluded, self.targetedData2.expectedConcentrationExcluded, []]
		self.expectedAddDataset.excludedFlag = [['Samples', 'Features'], ['Samples', 'Features'], []]
		self.expectedAddDataset.featureMask = numpy.array([1, 1, 1, 1, 1], dtype=bool)
		self.expectedAddDataset.sampleMask = numpy.array([1, 1, 1, 1, 1, 1, 1, 1], dtype=bool)
		# calibration
		self.expectedAddDataset.calibration = [calibB1,calibB3,calibB2,calibB5,self.targetedData3.calibration]
		self.expectedAddDataset.calibration[0]['calibSampleMetadata']['Batch'] = 1
		self.expectedAddDataset.calibration[1]['calibSampleMetadata']['Batch'] = 2
		self.expectedAddDataset.calibration[2]['calibSampleMetadata']['Batch'] = 3
		self.expectedAddDataset.calibration[3]['calibSampleMetadata']['Batch'] = 4
		self.expectedAddDataset.calibration[4]['calibSampleMetadata']['Batch'] = 5
		self.expectedAddDataset.VariableType = VariableType.Discrete

		# targetedData4, only sampleType and assayRole for updateMasks
		self.targetedData4 = nPYc.TargetedDataset('', fileType='empty')
		self.targetedData4._intensityData = numpy.zeros([18, 5], dtype=float)
		self.targetedData4.sampleMetadata['AssayRole'] = pandas.Series([AssayRole.LinearityReference,
																		AssayRole.LinearityReference,
																		AssayRole.LinearityReference,
																		AssayRole.LinearityReference,
																		AssayRole.LinearityReference,
																		AssayRole.PrecisionReference,
																		AssayRole.PrecisionReference,
																		AssayRole.PrecisionReference,
																		AssayRole.PrecisionReference,
																		AssayRole.PrecisionReference,
																		AssayRole.PrecisionReference,
																		AssayRole.Assay,
																		AssayRole.Assay,
																		AssayRole.Assay,
																		AssayRole.Assay,
																		AssayRole.Assay,
																		AssayRole.PrecisionReference,
																		AssayRole.PrecisionReference],
																		name='AssayRole',
																		dtype=object)
		self.targetedData4.sampleMetadata['SampleType'] = pandas.Series([SampleType.StudyPool,
																		SampleType.StudyPool,
																		SampleType.StudyPool,
																		SampleType.StudyPool,
																		SampleType.StudyPool,
																		SampleType.StudyPool,
																		SampleType.StudyPool,
																		SampleType.StudyPool,
																		SampleType.StudyPool,
																		SampleType.StudyPool,
																		SampleType.StudyPool,
																		SampleType.StudySample,
																		SampleType.StudySample,
																		SampleType.StudySample,
																		SampleType.StudySample,
																		SampleType.StudySample,
																		SampleType.ExternalReference,
																		SampleType.MethodReference],
																		name='SampleType',
																		dtype=object)

		self.targetedData4.featureMetadata['quantificationType'] = pandas.Series([QuantificationType.IS,
																				  QuantificationType.QuantOwnLabeledAnalogue,
																				  QuantificationType.QuantAltLabeledAnalogue,
																				  QuantificationType.QuantOther,
																				  QuantificationType.Monitored],
																				 name='quantificationType',
																				 dtype=object)
		self.targetedData4.featureMetadata['calibrationMethod'] = pandas.Series([CalibrationMethod.noIS,
																				CalibrationMethod.backcalculatedIS,
																				CalibrationMethod.backcalculatedIS,
																				CalibrationMethod.noIS,
																				CalibrationMethod.noCalibration],
																			   name='calibrationMethod',
																			   dtype=object)

		# targetedData5 for _getSampleMetadataFromFilename parsing
		self.targetedData5 = nPYc.TargetedDataset('', fileType='empty')
		self.targetedData5.sampleMetadata['Sample File Name'] = ['Test1_HPOS_ToF01_P1W02',
																 'Test2_RPOS_ToF02_U2W03',
																 'Test3_RNEG_ToF03_S3W04',
																 'Test4_LPOS_ToF04_P4W05_LTR',
																 'Test5_LNEG_ToF05_U5W06_SR',
																 'Test6_HPOS_ToF06_S4W05_MR',
																 'Test1_HPOS_ToF01_P1W02_x',
																 'Test2_RPOS_ToF02_U2W03_b',
																 'Test3_RNEG_ToF03_S3W04_2',
																 'Test4_RPOS_ToF04_B1S1_SR_q',
																 'Test5_LPOS_ToF05_B2E2_SR',
																 'Test6_LNEG_ToF06_B3SRD01_9',
																 'Test1_HPOS_ToF06_Blank01',
																 'Test1_HPOS_ToF06_IC02',
																 'Test1_HPOS_ToF06_EIC21']
		filenameSpec = "\n\t\t^(?P<fileName>\n\t\t\t(?P<baseName>\n\t\t\t\t(?P<study>\\w+?)\t\t\t\t\t\t\t\t\t\t# Study\n\t\t\t\t_\n\t\t\t\t(?P<chromatography>[HRL])(?P<ionisation>POS|NEG)\t# Chromatography and mode\n\t\t\t\t_\n\t\t\t\t(?P<instrument>\\w+?\\d\\d)\t\t\t\t\t\t\t# Instrument\n\t\t\t\t_\n\t\t\t\t(?P<groupingKind>Blank|E?IC|[A-Z]{1,2})(?P<groupingNo>\\d+?) # Sample grouping\n\t\t\t\t(?:\n\t\t\t\t(?P<injectionKind>[WSE]|SRD)(?P<injectionNo>\\d\\d?) # Subject ID\n\t\t\t\t)?\n\t\t\t\t(?:_(?P<reference>SR|LTR|MR))?\t\t\t\t\t  # Reference\n\t\t\t)\n\t\t\t(?:_(?P<exclusion>[xX]))?\t\t\t\t\t\t\t  # Exclusions\n\t\t\t(?:_(?P<reruns>[a-wyzA-WYZ]|[Rr]e[Rr]un\\d*?))?\t\t  # Reruns\n\t\t\t(?:_(?P<extraInjections>\\d+?))?\t\t\t\t\t\t  # Repeats\n\t\t\t(?:_(?P<exclusion2>[xX]))?\t\t\t\t\t\t\t  # badly ordered exclusions\n\t\t)$\n\t\t"
		self.targetedData5._getSampleMetadataFromFilename(filenameSpec)

		# targetedData6 for _fillBatches
		self.targetedData6 = nPYc.TargetedDataset('', fileType='empty')
		self.targetedData6.sampleMetadata['Sample File Name'] = ['Test_RPOS_ToF04_B1S1_SR',
																 'Test_RPOS_ToF04_B1S2_SR',
																 'Test_RPOS_ToF04_B1S3_SR',
																 'Test_RPOS_ToF04_B1S4_SR',
																 'Test_RPOS_ToF04_B1S5_SR',
																 'Test_RPOS_ToF04_P1W01',
																 'Test_RPOS_ToF04_P1W02_SR',
																 'Test_RPOS_ToF04_P1W03',
																 'Test_RPOS_ToF04_B1E1_SR',
																 'Test_RPOS_ToF04_B1E2_SR',
																 'Test_RPOS_ToF04_B1E3_SR',
																 'Test_RPOS_ToF04_B1E4_SR',
																 'Test_RPOS_ToF04_B1E5_SR',
																 'Test_RPOS_ToF04_B2S1_SR',
																 'Test_RPOS_ToF04_B2S2_SR',
																 'Test_RPOS_ToF04_B2S3_SR',
																 'Test_RPOS_ToF04_B2S4_SR',
																 'Test_RPOS_ToF04_B2S5_SR',
																 'Test_RPOS_ToF04_P2W01',
																 'Test_RPOS_ToF04_P2W02_SR',
																 'Test_RPOS_ToF04_P3W03',
																 'Test_RPOS_ToF04_B2S1_SR_2',
																 'Test_RPOS_ToF04_B2S2_SR_2',
																 'Test_RPOS_ToF04_B2S3_SR_2',
																 'Test_RPOS_ToF04_B2S4_SR_2',
																 'Test_RPOS_ToF04_B2S5_SR_2',
																 'Test_RPOS_ToF04_P3W03_b',
																 'Test_RPOS_ToF04_B2E1_SR',
																 'Test_RPOS_ToF04_B2E2_SR',
																 'Test_RPOS_ToF04_B2E3_SR',
																 'Test_RPOS_ToF04_B2E4_SR',
																 'Test_RPOS_ToF04_B2E5_SR',
																 'Test_RPOS_ToF04_B2SRD1']
		filenameSpec = "\n\t\t^(?P<fileName>\n\t\t\t(?P<baseName>\n\t\t\t\t(?P<study>\\w+?)\t\t\t\t\t\t\t\t\t\t# Study\n\t\t\t\t_\n\t\t\t\t(?P<chromatography>[HRL])(?P<ionisation>POS|NEG)\t# Chromatography and mode\n\t\t\t\t_\n\t\t\t\t(?P<instrument>\\w+?\\d\\d)\t\t\t\t\t\t\t# Instrument\n\t\t\t\t_\n\t\t\t\t(?P<groupingKind>Blank|E?IC|[A-Z]{1,2})(?P<groupingNo>\\d+?) # Sample grouping\n\t\t\t\t(?:\n\t\t\t\t(?P<injectionKind>[WSE]|SRD)(?P<injectionNo>\\d\\d?) # Subject ID\n\t\t\t\t)?\n\t\t\t\t(?:_(?P<reference>SR|LTR|MR))?\t\t\t\t\t  # Reference\n\t\t\t)\n\t\t\t(?:_(?P<exclusion>[xX]))?\t\t\t\t\t\t\t  # Exclusions\n\t\t\t(?:_(?P<reruns>[a-wyzA-WYZ]|[Rr]e[Rr]un\\d*?))?\t\t  # Reruns\n\t\t\t(?:_(?P<extraInjections>\\d+?))?\t\t\t\t\t\t  # Repeats\n\t\t\t(?:_(?P<exclusion2>[xX]))?\t\t\t\t\t\t\t  # badly ordered exclusions\n\t\t)$\n\t\t"
		self.targetedData6.addSampleInfo(descriptionFormat='Filenames', filenameSpec=filenameSpec)
		self.targetedData6.sampleMetadata['Run Order'] = self.targetedData6.sampleMetadata.index + 1
		

	def test_targeteddataset_deepcopy(self):
		copiedTargetedData = copy.deepcopy(self.targetedData1)

		self.assertEqual(vars(self.targetedData1).keys(), vars(copiedTargetedData).keys())
		for i in vars(self.targetedData1).keys():
			# need special case for numpy.ndarray
			if type(getattr(self.targetedData1, i)) is numpy.ndarray:
				numpy.testing.assert_array_equal(getattr(self.targetedData1,i), getattr(copiedTargetedData,i))
			elif type(getattr(self.targetedData1, i)) is pandas.core.frame.DataFrame:
				assert_frame_equal(getattr(self.targetedData1,i), getattr(copiedTargetedData,i))
			elif type(getattr(self.targetedData1, i)) is list:
				# the *Excluded are lists, need to match assert to type inside the list. Might not work with mixed type lists (expect the first type to match the whole list)
				if type(getattr(self.targetedData1, i)[0]) is pandas.core.frame.DataFrame:
					for j in range(len(getattr(self.targetedData1, i))):
						assert_frame_equal(getattr(self.targetedData1,i)[j], getattr(copiedTargetedData,i)[j])
				elif type(getattr(self.targetedData1, i)[0]) is numpy.ndarray:
					for k in range(len(getattr(self.targetedData1, i))):
						numpy.testing.assert_array_equal(getattr(self.targetedData1,i)[k], getattr(copiedTargetedData,i)[k])
				# list of self.calibration dict()
				elif type(getattr(self.targetedData1, i)[0]) is dict:
					for l in range(len(getattr(self.targetedData1, i))):
						# each key needs to be checked differently
						for m in getattr(self.targetedData1,i)[l].keys():
							if type(getattr(self.targetedData1,i)[l][m]) is pandas.core.frame.DataFrame:
								assert_frame_equal(getattr(self.targetedData1, i)[l][m], getattr(copiedTargetedData, i)[l][m])
							elif type(getattr(self.targetedData1,i)[l][m]) is numpy.ndarray:
								numpy.testing.assert_array_equal(getattr(self.targetedData1, i)[l][m], getattr(copiedTargetedData, i)[l][m])
							# dictionary in calibration is peakInfo
							elif type(getattr(self.targetedData1, i)[l][m]) is dict:
								for n in getattr(self.targetedData1, i)[l][m].keys():
									if type(getattr(self.targetedData1, i)[l][m][n]) is pandas.core.frame.DataFrame:
										assert_frame_equal(getattr(self.targetedData1, i)[l][m][n],getattr(copiedTargetedData, i)[l][m][n])
									elif type(getattr(self.targetedData1, i)[l][m][n]) is numpy.ndarray:
										numpy.testing.assert_array_equal(getattr(self.targetedData1, i)[l][m][n],getattr(copiedTargetedData, i)[l][m][n])
							else:
								self.assertEqual(getattr(self.targetedData1,i)[l][m], getattr(copiedTargetedData,i)[l][m])
				else:
					self.assertListEqual(getattr(self.targetedData1,i), getattr(copiedTargetedData,i))
			else:
				self.assertEqual(getattr(self.targetedData1,i), getattr(copiedTargetedData,i))


	def test_targeteddataset_validateObject(self):
		# Use targetedData3 for a basic (not merged) dataset, use targetedData1 for a merged dataset.
		# Required for calibration checks as code split between dict and list of dict

		with self.subTest(msg='validateObject successful on basic dataset'):
			goodDataset = copy.deepcopy(self.targetedData3)
			self.assertEqual(goodDataset.validateObject(verbose=False, raiseError=False, raiseWarning=True),{'Dataset': True, 'BasicTargetedDataset': True, 'QC': True, 'sampleMetadata': True})

		with self.subTest(msg='validateObject successful on merged dataset'):
			goodDataset = copy.deepcopy(self.targetedData1)
			self.assertEqual(goodDataset.validateObject(verbose=False, raiseError=False, raiseWarning=True),{'Dataset': True, 'BasicTargetedDataset': True, 'QC': True, 'sampleMetadata': True})

		with self.subTest(msg='BasicTargetedDataset fails on empty TargetedDataset'):
			badDataset = nPYc.TargetedDataset('', fileType='empty')
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicTargetedDataset':False ,'QC':False, 'sampleMetadata':False})

		with self.subTest(msg='check raise no warnings with raiseWarning=False'):
			badDataset = copy.deepcopy(self.targetedData3)
			del badDataset.Attributes['methodName']
			with warnings.catch_warnings(record=True) as w:
				# Cause all warnings to always be triggered.
				warnings.simplefilter("always")
				# warning
				result = badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False)
				# check it generally worked
				self.assertEqual(result, {'Dataset': True, 'BasicTargetedDataset': False, 'QC': False, 'sampleMetadata': False})
				# check each warning
				self.assertEqual(len(w), 0)

		with self.subTest(msg='check fail and raise warnings on bad Dataset'):
			badDataset = copy.deepcopy(self.targetedData1)
			delattr(badDataset, 'featureMetadata')
			with warnings.catch_warnings(record=True) as w:
				# Cause all warnings to always be triggered.
				warnings.simplefilter("always")
				# warning
				result = badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=True)
				# check it generally worked
				self.assertEqual(result, {'Dataset': False, 'BasicTargetedDataset': False, 'QC': False, 'sampleMetadata': False})
				# check each warning
				self.assertEqual(len(w), 5)
				assert issubclass(w[0].category, UserWarning)
				assert "Failure, no attribute 'self.featureMetadata'" in str(w[0].message)
				assert issubclass(w[1].category, UserWarning)
				assert "Does not conform to Dataset:" in str(w[1].message)
				assert issubclass(w[2].category, UserWarning)
				assert "Does not conform to basic TargetedDataset" in str(w[2].message)
				assert issubclass(w[3].category, UserWarning)
				assert "Does not have QC parameters" in str(w[3].message)
				assert issubclass(w[4].category, UserWarning)
				assert "Does not have sample metadata information" in str(w[4].message)

		with self.subTest(msg='check raise warnings BasicTargetedDataset'):
			badDataset = copy.deepcopy(self.targetedData3)
			del badDataset.Attributes['methodName']
			with warnings.catch_warnings(record=True) as w:
				# Cause all warnings to always be triggered.
				warnings.simplefilter("always")
				# warning
				result = badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=True)
				# check it generally worked
				self.assertEqual(result, {'Dataset': True, 'BasicTargetedDataset': False, 'QC': False, 'sampleMetadata': False})
				# check each warning
				self.assertEqual(len(w), 4)
				assert issubclass(w[0].category, UserWarning)
				assert "Failure, no attribute 'self.Attributes['methodName']" in str(w[0].message)
				assert issubclass(w[1].category, UserWarning)
				assert "Does not conform to basic TargetedDataset:" in str(w[1].message)
				assert issubclass(w[2].category, UserWarning)
				assert "Does not have QC parameters" in str(w[2].message)
				assert issubclass(w[3].category, UserWarning)
				assert "Does not have sample metadata information" in str(w[3].message)

		with self.subTest(msg='check raise warnings QC parameters'):
			badDataset = copy.deepcopy(self.targetedData3)
			badDataset.sampleMetadata['Batch'] = 'not an int or float'
			with warnings.catch_warnings(record=True) as w:
				# Cause all warnings to always be triggered.
				warnings.simplefilter("always")
				# warning
				result = badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=True)
				# check it generally worked
				self.assertEqual(result, {'Dataset': True, 'BasicTargetedDataset': True, 'QC': False, 'sampleMetadata': False})
				# check each warning
				self.assertEqual(len(w), 3)
				assert issubclass(w[0].category, UserWarning)
				assert "Failure, 'self.sampleMetadata['Batch']' is <class 'str'>" in str(w[0].message)
				assert issubclass(w[1].category, UserWarning)
				assert "Does not have QC parameters:" in str(w[1].message)
				assert issubclass(w[2].category, UserWarning)
				assert "Does not have sample metadata information:" in str(w[2].message)

		with self.subTest(msg='check raise warnings sampleMetadata'):
			badDataset = copy.deepcopy(self.targetedData3)
			badDataset.sampleMetadata.drop(['Subject ID'], axis=1, inplace=True)
			with warnings.catch_warnings(record=True) as w:
				# Cause all warnings to always be triggered.
				warnings.simplefilter("always")
				# warning
				result = badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=True)
				# check it generally worked
				self.assertEqual(result, {'Dataset': True, 'BasicTargetedDataset': True, 'QC': True, 'sampleMetadata': False})
				# check each warning
				self.assertEqual(len(w), 2)
				assert issubclass(w[0].category, UserWarning)
				assert "Failure, 'self.sampleMetadata' lacks a 'Subject ID' column" in str(w[0].message)
				assert issubclass(w[1].category, UserWarning)
				assert "Does not have sample metadata information:" in str(w[1].message)

		with self.subTest(msg='self.Attributes[\'methodName\'] does not exist'):
			badDataset = copy.deepcopy(self.targetedData3)
			del badDataset.Attributes['methodName']
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicTargetedDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(AttributeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.Attributes[\'methodName\'] is not a str'):
			badDataset = copy.deepcopy(self.targetedData3)
			badDataset.Attributes['methodName'] = pandas.DataFrame(None)
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicTargetedDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(TypeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='self.Attributes[\'externalID\'] does not exist'):
			badDataset = copy.deepcopy(self.targetedData3)
			del badDataset.Attributes['externalID']
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicTargetedDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(AttributeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.Attributes[\'externalID\'] is not a list'):
			badDataset = copy.deepcopy(self.targetedData3)
			badDataset.Attributes['externalID'] = 'Not a list'
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicTargetedDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(TypeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.VariableType is not an enum VariableType'):
			badDataset = copy.deepcopy(self.targetedData3)
			badDataset.VariableType = 'not an enum'
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicTargetedDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(TypeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='self.fileName does not exist'):
			badDataset = copy.deepcopy(self.targetedData3)
			delattr(badDataset, 'fileName')
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicTargetedDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(AttributeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.fileName is not a str or list'):
			badDataset = copy.deepcopy(self.targetedData3)
			badDataset.fileName = 5.
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicTargetedDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(TypeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='self.filePath does not exist'):
			badDataset = copy.deepcopy(self.targetedData3)
			delattr(badDataset, 'filePath')
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicTargetedDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(AttributeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.filePath is not a str or list'):
			badDataset = copy.deepcopy(self.targetedData3)
			badDataset.filePath = 5.
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicTargetedDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(TypeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.sampleMetadata does not have the same number of samples as self._intensityData'):
			badDataset = copy.deepcopy(self.targetedData3)
			badDataset.sampleMetadata.drop([0], axis=0, inplace=True)
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicTargetedDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(ValueError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.sampleMetadata[\'Sample File Name\'] is not str'):
			badDataset = copy.deepcopy(self.targetedData3)
			badDataset.sampleMetadata['Sample File Name'] = 5.
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicTargetedDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(TypeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.sampleMetadata[\'AssayRole\'] is not an enum \'AssayRole\''):
			badDataset = copy.deepcopy(self.targetedData3)
			badDataset.sampleMetadata['AssayRole'] = 5.
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicTargetedDataset': True, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(TypeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.sampleMetadata[\'SampleType\'] is not an enum \'SampleType\''):
			badDataset = copy.deepcopy(self.targetedData3)
			badDataset.sampleMetadata['SampleType'] = 5.
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicTargetedDataset': True, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(TypeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.sampleMetadata[\'Dilution\'] is not an int or float'):
			badDataset = copy.deepcopy(self.targetedData3)
			badDataset.sampleMetadata['Dilution'] = 'not an int'
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicTargetedDataset': True, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(TypeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.sampleMetadata[\'Batch\'] is not an int or float'):
			badDataset = copy.deepcopy(self.targetedData3)
			badDataset.sampleMetadata['Batch'] = 'not an int or float'
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicTargetedDataset': True, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(TypeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.sampleMetadata[\'Correction Batch\'] is not an int'):
			badDataset = copy.deepcopy(self.targetedData3)
			badDataset.sampleMetadata['Correction Batch'] = 'not an int or float'
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicTargetedDataset': True, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(TypeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.sampleMetadata[\'Run Order\'] is not an int'):
			badDataset = copy.deepcopy(self.targetedData3)
			badDataset.sampleMetadata['Run Order'] = 'not an int'
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicTargetedDataset': True, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(TypeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.sampleMetadata[\'Acquired Time\'] is not a datetime'):
			badDataset = copy.deepcopy(self.targetedData3)
			badDataset.sampleMetadata['Acquired Time'] = 5.
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicTargetedDataset': True, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(TypeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.sampleMetadata[\'Sample Base Name\'] is not str'):
			badDataset = copy.deepcopy(self.targetedData3)
			badDataset.sampleMetadata['Sample Base Name'] = 5.
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicTargetedDataset': True, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(TypeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.sampleMetadata does not have a Subject ID column'):
			badDataset = copy.deepcopy(self.targetedData3)
			badDataset.sampleMetadata.drop(['Subject ID'], axis=1, inplace=True)
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicTargetedDataset': True, 'QC': True, 'sampleMetadata': False})
			self.assertRaises(LookupError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.sampleMetadata[\'Subject ID\'] is not a str'):
			badDataset = copy.deepcopy(self.targetedData3)
			badDataset.sampleMetadata['Subject ID'] = 5.
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicTargetedDataset': True, 'QC': True, 'sampleMetadata': False})
			self.assertRaises(TypeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.sampleMetadata[\'Sample ID\'] is not a str'):
			badDataset = copy.deepcopy(self.targetedData3)
			badDataset.sampleMetadata['Sample ID'] = 5.
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicTargetedDataset': True, 'QC': True, 'sampleMetadata': False})
			self.assertRaises(TypeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.featureMetadata does not have the same number of samples as self._intensityData'):
			badDataset = copy.deepcopy(self.targetedData3)
			badDataset.featureMetadata.drop([0], axis=0, inplace=True)
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicTargetedDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(ValueError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.featureMetadata[\'Feature Name\'] is not a str'):
			badDataset = copy.deepcopy(self.targetedData3)
			badDataset.featureMetadata['Feature Name'] = 5.
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicTargetedDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(TypeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.featureMetadata[\'Feature Name\'] is not unique'):
			badDataset = copy.deepcopy(self.targetedData1)
			badDataset.featureMetadata['Feature Name'] = ['Feature1','Feature1','Feature1']
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicTargetedDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(ValueError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.featureMetadata does not have a calibrationMethod column'):
			badDataset = copy.deepcopy(self.targetedData3)
			badDataset.featureMetadata.drop(['calibrationMethod'], axis=1, inplace=True)
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicTargetedDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(LookupError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.featureMetadata[\'calibrationMethod\'] is not an enum \'CalibrationMethod\''):
			badDataset = copy.deepcopy(self.targetedData3)
			badDataset.featureMetadata['calibrationMethod'] = 5.
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicTargetedDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(TypeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.featureMetadata does not have a quantificationType column'):
			badDataset = copy.deepcopy(self.targetedData3)
			badDataset.featureMetadata.drop(['quantificationType'], axis=1, inplace=True)
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicTargetedDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(LookupError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.featureMetadata[\'quantificationType\'] is not an enum \'QuantificationType\''):
			badDataset = copy.deepcopy(self.targetedData3)
			badDataset.featureMetadata['quantificationType'] = 5.
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicTargetedDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(TypeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.featureMetadata does not have a Unit column'):
			badDataset = copy.deepcopy(self.targetedData3)
			badDataset.featureMetadata.drop(['Unit'], axis=1, inplace=True)
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicTargetedDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(LookupError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.featureMetadata[\'Unit\'] is not a str'):
			badDataset = copy.deepcopy(self.targetedData3)
			badDataset.featureMetadata['Unit'] = 5.
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicTargetedDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(TypeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.featureMetadata does not have an LLOQ or similar column'):
			badDataset = copy.deepcopy(self.targetedData3)
			badDataset.featureMetadata.drop(['LLOQ'], axis=1, inplace=True)
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicTargetedDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(LookupError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.featureMetadata[\'LLOQ\'] is not an int or float'):
			badDataset = copy.deepcopy(self.targetedData3)
			badDataset.featureMetadata['LLOQ'] = 'not an int of float'
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicTargetedDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(TypeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.featureMetadata does not have an ULOQ or similar column'):
			badDataset = copy.deepcopy(self.targetedData3)
			badDataset.featureMetadata.drop(['ULOQ'], axis=1, inplace=True)
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicTargetedDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(LookupError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.featureMetadata[\'ULOQ\'] is not an int or float'):
			badDataset = copy.deepcopy(self.targetedData3)
			badDataset.featureMetadata['ULOQ'] = 'not an int or float'
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicTargetedDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(TypeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.featureMetadata does not have the \'externalID\' as columns'):
			badDataset = copy.deepcopy(self.targetedData3)
			badDataset.featureMetadata.drop(['extID1'], axis=1, inplace=True)
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicTargetedDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(LookupError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.expectedConcentration does not exist'):
			badDataset = copy.deepcopy(self.targetedData3)
			delattr(badDataset, 'expectedConcentration')
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicTargetedDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(AttributeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.expectedConcentration is not a pandas.DataFrame'):
			badDataset = copy.deepcopy(self.targetedData3)
			badDataset.expectedConcentration = 5.
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicTargetedDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(TypeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.expectedConcentration does not have the same number of samples as self._intensityData'):
			badDataset = copy.deepcopy(self.targetedData3)
			badDataset.expectedConcentration = pandas.DataFrame(numpy.array([[1,1],[1,1]]))
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicTargetedDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(ValueError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.expectedConcentration does not have the same number of samples as self._intensityData'):
			badDataset = copy.deepcopy(self.targetedData3)
			badDataset.expectedConcentration = pandas.DataFrame(numpy.array([[1,1]]))
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicTargetedDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(ValueError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.expectedConcentration column name do not match self.featureMetadata[\'Feature Name\']'):
			badDataset = copy.deepcopy(self.targetedData3)
			badDataset.expectedConcentration = pandas.DataFrame(numpy.array([[1]]))
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicTargetedDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(ValueError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.sampleMask is not initialised'):
			badDataset = copy.deepcopy(self.targetedData3)
			badDataset.sampleMask = numpy.array(False, dtype=bool)
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicTargetedDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(ValueError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.sampleMask does not have the same number of samples as self._intensityData'):
			badDataset = copy.deepcopy(self.targetedData3)
			badDataset.sampleMask = numpy.squeeze(numpy.ones([5, 1], dtype=bool), axis=1)
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicTargetedDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(ValueError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.featureMask is not initialised'):
			badDataset = copy.deepcopy(self.targetedData3)
			badDataset.featureMask = numpy.array(False, dtype=bool)
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicTargetedDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(ValueError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.featureMask does not have the same number of samples as self._intensityData'):
			badDataset = copy.deepcopy(self.targetedData3)
			badDataset.featureMask = numpy.squeeze(numpy.ones([5, 1], dtype=bool), axis=1)
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicTargetedDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(ValueError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.calibration does not exist'):
			badDataset = copy.deepcopy(self.targetedData3)
			delattr(badDataset, 'calibration')
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicTargetedDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(AttributeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.calibration (basic) is not a dict'):
			badDataset = copy.deepcopy(self.targetedData3)
			badDataset.calibration = 'not a dict'
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicTargetedDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(TypeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.calibration (merged) is not a list of dict'):
			badDataset = copy.deepcopy(self.targetedData1)
			badDataset.calibration = ['not a dict in a list']
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicTargetedDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(TypeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.calibration[\'calibIntensityData\'] (basic) does not exist'):
			badDataset = copy.deepcopy(self.targetedData3)
			del badDataset.calibration['calibIntensityData']
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicTargetedDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(AttributeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.calibration[\'calibIntensityData\'] (merged) does not exist'):
			badDataset = copy.deepcopy(self.targetedData1)
			del badDataset.calibration[1]['calibIntensityData']
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicTargetedDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(AttributeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.calibration[\'calibIntensityData\'] (basic) is not a numpy.ndarray'):
			badDataset = copy.deepcopy(self.targetedData3)
			badDataset.calibration['calibIntensityData'] = 'not an array'
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicTargetedDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(TypeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.calibration[\'calibIntensityData\'] (merged) not a numpy.ndarray'):
			badDataset = copy.deepcopy(self.targetedData1)
			badDataset.calibration[1]['calibIntensityData'] = 'not an array'
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicTargetedDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(TypeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.calibration[\'calibIntensityData\'] (basic) does not have the same number of features as self._intensityData'):
			badDataset = copy.deepcopy(self.targetedData3)
			badDataset.calibration['calibIntensityData'] = numpy.array([[1,1,1],[1,1,1],[1,1,1]])
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicTargetedDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(ValueError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.calibration[\'calibSampleMetadata\'] (basic) does not exist'):
			badDataset = copy.deepcopy(self.targetedData3)
			del badDataset.calibration['calibSampleMetadata']
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicTargetedDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(AttributeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.calibration[\'calibSampleMetadata\'] (merged) does not exist'):
			badDataset = copy.deepcopy(self.targetedData1)
			del badDataset.calibration[1]['calibSampleMetadata']
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicTargetedDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(AttributeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.calibration[\'calibSampleMetadata\'] (basic) is not a pandas.DataFrame'):
			badDataset = copy.deepcopy(self.targetedData3)
			badDataset.calibration['calibSampleMetadata'] = 'not a DataFrame'
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicTargetedDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(TypeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.calibration[\'calibSampleMetadata\'] (merged) not a pandas.DataFrame'):
			badDataset = copy.deepcopy(self.targetedData1)
			badDataset.calibration[1]['calibSampleMetadata'] = 'not a DataFrame'
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicTargetedDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(TypeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.calibration[\'calibSampleMetadata\'] (basic) does not have the same number of samples as self.calibration[\'calibIntensityData\']'):
			badDataset = copy.deepcopy(self.targetedData3)
			badDataset.calibration['calibSampleMetadata'] = pandas.DataFrame(numpy.array([[1,1,1],[1,1,1],[1,1,1]]))
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicTargetedDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(ValueError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.calibration[\'calibSampleMetadata\'] (merged) does not have the same number of samples as self.calibration[\'calibIntensityData\']'):
			badDataset = copy.deepcopy(self.targetedData1)
			badDataset.calibration[1]['calibSampleMetadata'] = pandas.DataFrame(numpy.array([[1,1,1],[1,1,1],[1,1,1]]))
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicTargetedDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(ValueError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.calibration[\'calibFeatureMetadata\'] (basic) does not exist'):
			badDataset = copy.deepcopy(self.targetedData3)
			del badDataset.calibration['calibFeatureMetadata']
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicTargetedDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(AttributeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.calibration[\'calibFeatureMetadata\'] (merged) does not exist'):
			badDataset = copy.deepcopy(self.targetedData1)
			del badDataset.calibration[1]['calibFeatureMetadata']
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicTargetedDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(AttributeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.calibration[\'calibFeatureMetadata\'] (basic) is not a pandas.DataFrame'):
			badDataset = copy.deepcopy(self.targetedData3)
			badDataset.calibration['calibFeatureMetadata'] = 'not a DataFrame'
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicTargetedDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(TypeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.calibration[\'calibFeatureMetadata\'] (merged) not a pandas.DataFrame'):
			badDataset = copy.deepcopy(self.targetedData1)
			badDataset.calibration[1]['calibFeatureMetadata'] = 'not a DataFrame'
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicTargetedDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(TypeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.calibration[\'calibFeatureMetadata\'] (basic) does not have the same number of samples as self.calibration[\'calibIntensityData\']'):
			badDataset = copy.deepcopy(self.targetedData3)
			badDataset.calibration['calibFeatureMetadata'] = pandas.DataFrame(numpy.array([[1,1,1],[1,1,1],[1,1,1]]))
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicTargetedDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(ValueError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.calibration[\'calibFeatureMetadata\'] (merged) does not have the same number of samples as self.calibration[\'calibIntensityData\']'):
			badDataset = copy.deepcopy(self.targetedData1)
			badDataset.calibration[1]['calibFeatureMetadata'] = pandas.DataFrame(numpy.array([[1,1],[1,1]]))
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicTargetedDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(ValueError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.calibration[\'calibFeatureMetadata\'] (basic) does not have a [\'Feature Name\'] column'):
			badDataset = copy.deepcopy(self.targetedData3)
			badDataset.calibration['calibFeatureMetadata'] = pandas.DataFrame(numpy.array([[1]]))
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicTargetedDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(LookupError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.calibration[\'calibFeatureMetadata\'] (merged) does not have a [\'Feature Name\'] column'):
			badDataset = copy.deepcopy(self.targetedData1)
			badDataset.calibration[1]['calibFeatureMetadata'] = pandas.DataFrame(numpy.array([[1,1,1],[1,1,1],[1,1,1]]))
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicTargetedDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(LookupError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.calibration[\'calibExpectedConcentration\'] (basic) does not exist'):
			badDataset = copy.deepcopy(self.targetedData3)
			del badDataset.calibration['calibExpectedConcentration']
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicTargetedDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(AttributeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.calibration[\'calibExpectedConcentration\'] (merged) does not exist'):
			badDataset = copy.deepcopy(self.targetedData1)
			del badDataset.calibration[1]['calibExpectedConcentration']
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicTargetedDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(AttributeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.calibration[\'calibExpectedConcentration\'] (basic) is not a pandas.DataFrame'):
			badDataset = copy.deepcopy(self.targetedData3)
			badDataset.calibration['calibExpectedConcentration'] = 'not a DataFrame'
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicTargetedDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(TypeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.calibration[\'calibExpectedConcentration\'] (merged) not a pandas.DataFrame'):
			badDataset = copy.deepcopy(self.targetedData1)
			badDataset.calibration[1]['calibExpectedConcentration'] = 'not a DataFrame'
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicTargetedDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(TypeError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.calibration[\'calibExpectedConcentration\'] (basic) does not have the same number of samples as self.calibration[\'calibIntensityData\']'):
			badDataset = copy.deepcopy(self.targetedData3)
			badDataset.calibration['calibExpectedConcentration'] = pandas.DataFrame(numpy.array([[1,1,1],[1,1,1],[1,1,1]]))
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicTargetedDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(ValueError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.calibration[\'calibExpectedConcentration\'] (merged) does not have the same number of samples as self.calibration[\'calibIntensityData\']'):
			badDataset = copy.deepcopy(self.targetedData1)
			badDataset.calibration[1]['calibExpectedConcentration'] = pandas.DataFrame(numpy.array([[1,1],[1,1]]))
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicTargetedDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(ValueError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.calibration[\'calibExpectedConcentration\'] (basic) does not have the same number of features as self.calibration[\'calibIntensityData\']'):
			badDataset = copy.deepcopy(self.targetedData3)
			badDataset.calibration['calibExpectedConcentration'] = pandas.DataFrame(numpy.array([[1,1]]))
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicTargetedDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(ValueError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.calibration[\'calibExpectedConcentration\'] (merged) does not have the same number of features as self.calibration[\'calibIntensityData\']'):
			badDataset = copy.deepcopy(self.targetedData1)
			badDataset.calibration[1]['calibExpectedConcentration'] = pandas.DataFrame(numpy.array([[1,1]]))
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicTargetedDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(ValueError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.calibration[\'calibExpectedConcentration\'] (basic) does not have a [\'Feature Name\'] column'):
			badDataset = copy.deepcopy(self.targetedData3)
			badDataset.calibration['calibExpectedConcentration'] = pandas.DataFrame(numpy.array([[1]]))
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicTargetedDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(ValueError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)

		with self.subTest(msg='if self.calibration[\'calibExpectedConcentration\'] (merged) does not have a [\'Feature Name\'] column'):
			badDataset = copy.deepcopy(self.targetedData1)
			badDataset.calibration[1]['calibExpectedConcentration'] = pandas.DataFrame(numpy.array([[1,1,1]]))
			self.assertEqual(badDataset.validateObject(verbose=False, raiseError=False, raiseWarning=False), {'Dataset': True, 'BasicTargetedDataset': False, 'QC': False, 'sampleMetadata': False})
			self.assertRaises(ValueError, badDataset.validateObject, verbose=False, raiseError=True, raiseWarning=False)


	@unittest.mock.patch('sys.stdout', new_callable=io.StringIO)
	def test_targeteddataset_add_raise(self, mock_stdout):
		firstDataset = copy.deepcopy(self.targetedData1)
		# different targeted method
		otherMethodDataset = copy.deepcopy(self.targetedData2)
		otherMethodDataset.Attributes['methodName'] = 'anotherMethod'
		# duplicate feature
		duplicateFeatureDataset = copy.deepcopy(self.targetedData1)
		duplicateFeatureDataset.featureMetadata = pandas.concat([duplicateFeatureDataset.featureMetadata, duplicateFeatureDataset.featureMetadata[[True, False, False]]],ignore_index=True)

		with self.subTest(msg='Checking ValueError if targeted methods differ'):
			self.assertRaises(ValueError, firstDataset.__add__, other=otherMethodDataset)
		with self.subTest(msg='Checking ValueError if one input dataset doesnt pass validateObject()'):
			goodDataset   = copy.deepcopy(self.targetedData1)
			brokenDataset = copy.deepcopy(self.targetedData1)
			delattr(brokenDataset, 'sampleMetadata')
			self.assertRaises(ValueError, brokenDataset.__add__, other=goodDataset)
			self.assertRaises(ValueError, goodDataset.__add__, other=brokenDataset)
		with warnings.catch_warnings(record=True) as w:
			# Cause all warnings to always be triggered.
			warnings.simplefilter("always")
			# warning
			addedDataset = firstDataset + firstDataset
			# check
			assert issubclass(w[0].category, UserWarning)
			assert "are present more than once" in str(w[0].message)


	@unittest.mock.patch('sys.stdout', new_callable=io.StringIO)
	def test_targeteddataset_add(self, mock_stdout):

		with self.subTest(msg='Checking concatenation of datasets'):
			# Modify the sampleMetadataNotExported and featureMetadataNotExported here as excluding 'Feature Name' would compromise other tests (need a something that must not get _batchX appended)
			t1 = copy.deepcopy(self.targetedData1)
			t1.Attributes['sampleMetadataNotExported']  = ['testSampleMetaNotExported']
			t1.Attributes['featureMetadataNotExported'] = ['Feature Name', 'testFeatNotExported_batch1', 'testFeatNotExported_batch3']
			t2 = copy.deepcopy(self.targetedData2)
			t2.Attributes['sampleMetadataNotExported']  = ['testSampleMetaNotExported']
			t2.Attributes['featureMetadataNotExported'] = ['Feature Name', 'testFeatNotExported_batch2', 'testFeatNotExported_batch5']
			t3 = copy.deepcopy(self.targetedData3)
			t3.Attributes['sampleMetadataNotExported']  = ['testSampleMetaNotExported']
			t3.Attributes['featureMetadataNotExported'] = ['Feature Name', 'testFeatNotExported']

			with warnings.catch_warnings():
				warnings.simplefilter('ignore', UserWarning)
				concatenatedDataset = t1 + t2 + t3

			expectedDataset = copy.deepcopy(self.expectedAddDataset)
			expectedDataset.Attributes['sampleMetadataNotExported']  = ['testSampleMetaNotExported']
			expectedDataset.Attributes['featureMetadataNotExported'] = ['Feature Name', 'testFeatNotExported_batch1', 'testFeatNotExported_batch2', 'testFeatNotExported_batch3', 'testFeatNotExported_batch4', 'testFeatNotExported_batch5']

			# Checking class:
			self.assertEqual(type(concatenatedDataset),type(expectedDataset))
			# Checking sampleMetadata:
			assert_frame_equal(concatenatedDataset.sampleMetadata,expectedDataset.sampleMetadata)
			# Checking featureMetadata:
			assert_frame_equal(concatenatedDataset.featureMetadata.reindex(sorted(concatenatedDataset.featureMetadata),axis=1), expectedDataset.featureMetadata.reindex(sorted(expectedDataset.featureMetadata),axis=1))
			# Checking intensityData:
			numpy.testing.assert_array_equal(concatenatedDataset._intensityData, expectedDataset._intensityData)
			# Checking expectedConcentration:
			assert_frame_equal(concatenatedDataset.expectedConcentration.reindex(sorted(concatenatedDataset.expectedConcentration), axis=1),expectedDataset.expectedConcentration.reindex(sorted(expectedDataset.expectedConcentration), axis=1))
			# Checking Attributes:
			# same Attributes
			self.assertEqual(concatenatedDataset.Attributes.keys(), expectedDataset.Attributes.keys())
			for k in expectedDataset.Attributes.keys():
				# Log cannot be match 1 to 1 as we can't reproduce the datetime on run time of the __add__ and __init__ logs
				if k == 'Log':
					# only check columns without a datetime that can't be reproduced
					for i in [0, 1, 2, 5, 6, 7, 16, 17, 18]:
						self.assertListEqual(expectedDataset.Attributes[k][i], concatenatedDataset.Attributes[k][i])
				elif (k == 'sampleMetadataNotExported') | (k == 'featureMetadataNotExported'):
					self.assertListEqual(sorted(expectedDataset.Attributes[k]), sorted(concatenatedDataset.Attributes[k]))
				else:
					# will cover all cases unless Generic.json is changed
					if type(expectedDataset.Attributes[k]) is list:
						self.assertListEqual(expectedDataset.Attributes[k],concatenatedDataset.Attributes[k])
					else:
						self.assertEqual(expectedDataset.Attributes[k],concatenatedDataset.Attributes[k])
			# Checking sampleMetadataExcluded:
			assert_frame_equal(expectedDataset.sampleMetadataExcluded[0][0], concatenatedDataset.sampleMetadataExcluded[0][0])
			assert_frame_equal(expectedDataset.sampleMetadataExcluded[0][1], concatenatedDataset.sampleMetadataExcluded[0][1])
			assert_frame_equal(expectedDataset.sampleMetadataExcluded[1][0], concatenatedDataset.sampleMetadataExcluded[1][0])
			assert_frame_equal(expectedDataset.sampleMetadataExcluded[1][1], concatenatedDataset.sampleMetadataExcluded[1][1])
			self.assertListEqual(expectedDataset.sampleMetadataExcluded[2], concatenatedDataset.sampleMetadataExcluded[2])
			# Checking intensityMetadataExcluded:
			numpy.testing.assert_array_equal(expectedDataset.intensityDataExcluded[0], concatenatedDataset.intensityDataExcluded[0])
			numpy.testing.assert_array_equal(expectedDataset.intensityDataExcluded[1], concatenatedDataset.intensityDataExcluded[1])
			self.assertListEqual(expectedDataset.intensityDataExcluded[2], concatenatedDataset.intensityDataExcluded[2])
			# Checking featureMetadataExcluded:
			assert_frame_equal(expectedDataset.featureMetadataExcluded[0][0], concatenatedDataset.featureMetadataExcluded[0][0])
			assert_frame_equal(expectedDataset.featureMetadataExcluded[0][1], concatenatedDataset.featureMetadataExcluded[0][1])
			assert_frame_equal(expectedDataset.featureMetadataExcluded[1][0], concatenatedDataset.featureMetadataExcluded[1][0])
			assert_frame_equal(expectedDataset.featureMetadataExcluded[1][1], concatenatedDataset.featureMetadataExcluded[1][1])
			self.assertListEqual(expectedDataset.featureMetadataExcluded[2], concatenatedDataset.featureMetadataExcluded[2])
			# Checking expectedConcentrationExcluded:
			assert_frame_equal(expectedDataset.expectedConcentrationExcluded[0][0], concatenatedDataset.expectedConcentrationExcluded[0][0])
			assert_frame_equal(expectedDataset.expectedConcentrationExcluded[0][1], concatenatedDataset.expectedConcentrationExcluded[0][1])
			assert_frame_equal(expectedDataset.expectedConcentrationExcluded[1][0], concatenatedDataset.expectedConcentrationExcluded[1][0])
			assert_frame_equal(expectedDataset.expectedConcentrationExcluded[1][1], concatenatedDataset.expectedConcentrationExcluded[1][1])
			self.assertListEqual(expectedDataset.expectedConcentrationExcluded[2], concatenatedDataset.expectedConcentrationExcluded[2])
			# Checking excludedFlag:
			self.assertListEqual(expectedDataset.excludedFlag, concatenatedDataset.excludedFlag)
			# Checking featureMask:
			numpy.testing.assert_array_equal(expectedDataset.featureMask, concatenatedDataset.featureMask)
			# Checking sampleMask:
			numpy.testing.assert_array_equal(expectedDataset.sampleMask, concatenatedDataset.sampleMask)
			# Checking calibration:
			for i in range(len(expectedDataset.calibration)):
				assert_frame_equal(expectedDataset.calibration[i]['calibSampleMetadata'], concatenatedDataset.calibration[i]['calibSampleMetadata'])
				assert_frame_equal(expectedDataset.calibration[i]['calibFeatureMetadata'], concatenatedDataset.calibration[i]['calibFeatureMetadata'])
				numpy.testing.assert_array_equal(expectedDataset.calibration[i]['calibIntensityData'], concatenatedDataset.calibration[i]['calibIntensityData'])
				assert_frame_equal(expectedDataset.calibration[i]['calibExpectedConcentration'], concatenatedDataset.calibration[i]['calibExpectedConcentration'])

		with self.subTest(msg='Checking concatenation with unexpected attributes'):

			with warnings.catch_warnings():
				warnings.simplefilter('ignore', UserWarning)
				tempConcatDataset = self.targetedData1 + self.targetedData2
			otherConcatDataset = copy.deepcopy(self.targetedData3)

			# add variables
			tempConcatDataset.inCommon     = 'in common from self'
			tempConcatDataset.onlyInSelf   = 'only in self'
			otherConcatDataset.inCommon    = 'in common from other'
			otherConcatDataset.onlyInOther = 'only in other'

			# merge
			with warnings.catch_warnings():
				warnings.simplefilter('ignore', UserWarning)
				result = tempConcatDataset + otherConcatDataset

			# check
			self.assertEqual(result.inCommon[0], 'in common from self')
			self.assertEqual(result.inCommon[1], 'in common from other')
			self.assertEqual(result.onlyInSelf[0], 'only in self')
			self.assertEqual(result.onlyInSelf[1], None)
			self.assertEqual(result.onlyInOther[0], None)
			self.assertEqual(result.onlyInOther[1], 'only in other')

		with self.subTest(msg='Checking Warning mergeLimitsOfQuantification'):
			with warnings.catch_warnings(record=True) as w:
				#print(w)
				# Cause all warnings to always be triggered.
				warnings.simplefilter("always")
				# warning
				concatenatedDataset = self.targetedData1 + self.targetedData2 + self.targetedData3
				#check (2 sums, so 2 warnings)
				#assert len(w) == 2
				assert issubclass(w[-1].category, UserWarning)
				assert "Update the limits of quantification using" in str(w[-1].message)


	@unittest.mock.patch('sys.stdout', new_callable=io.StringIO)
	def test_targeteddataset_radd(self, mock_stdout):
		# sum([ X1, X2 ]) always tries to do 0 + X1 which fails, then reverts to __radd__ X1 + 0
		expectedDataset = copy.deepcopy(self.expectedAddDataset)
		with warnings.catch_warnings():
			warnings.simplefilter('ignore', UserWarning)
			concatenatedDataset = sum([self.targetedData1, self.targetedData2, self.targetedData3])

		with self.subTest(msg='Checking class'):
			self.assertEqual(type(concatenatedDataset),type(expectedDataset))
		with self.subTest(msg='Checking sampleMetadata'):
			assert_frame_equal(concatenatedDataset.sampleMetadata,expectedDataset.sampleMetadata)
		with self.subTest(msg='Checking featureMetadata'):
			assert_frame_equal(concatenatedDataset.featureMetadata.reindex(sorted(concatenatedDataset.featureMetadata),axis=1), expectedDataset.featureMetadata.reindex(sorted(expectedDataset.featureMetadata),axis=1))
		with self.subTest(msg='Checking intensityData'):
			numpy.testing.assert_array_equal(concatenatedDataset._intensityData, expectedDataset._intensityData)
		with self.subTest(msg='Checking expectedConcentration'):
			assert_frame_equal(concatenatedDataset.expectedConcentration.reindex(sorted(concatenatedDataset.expectedConcentration), axis=1),expectedDataset.expectedConcentration.reindex(sorted(expectedDataset.expectedConcentration), axis=1))
		with self.subTest(msg='Checking Attributes'):
			# same Attributes
			self.assertEqual(concatenatedDataset.Attributes.keys(), expectedDataset.Attributes.keys())
			for k in expectedDataset.Attributes.keys():
				# Log cannot be match 1 to 1 as we can't reproduce the datetime on run time of the __add__ and __init__ logs
				if k == 'Log':
					# only check columns without a datetime that can't be reproduced
					for i in [0, 1, 2, 5, 6, 7, 16, 17, 18]:
						self.assertListEqual(expectedDataset.Attributes[k][i], concatenatedDataset.Attributes[k][i])
				else:
					# will cover all cases unless Generic.json is changed
					if type(expectedDataset.Attributes[k]) is list:
						self.assertListEqual(expectedDataset.Attributes[k],concatenatedDataset.Attributes[k])
					else:
						self.assertEqual(expectedDataset.Attributes[k],concatenatedDataset.Attributes[k])
		with self.subTest(msg='Checking sampleMetadataExcluded'):
			assert_frame_equal(expectedDataset.sampleMetadataExcluded[0][0], concatenatedDataset.sampleMetadataExcluded[0][0])
			assert_frame_equal(expectedDataset.sampleMetadataExcluded[0][1], concatenatedDataset.sampleMetadataExcluded[0][1])
			assert_frame_equal(expectedDataset.sampleMetadataExcluded[1][0], concatenatedDataset.sampleMetadataExcluded[1][0])
			assert_frame_equal(expectedDataset.sampleMetadataExcluded[1][1], concatenatedDataset.sampleMetadataExcluded[1][1])
			self.assertListEqual(expectedDataset.sampleMetadataExcluded[2], concatenatedDataset.sampleMetadataExcluded[2])
		with self.subTest(msg='Checking intensityMetadataExcluded'):
			numpy.testing.assert_array_equal(expectedDataset.intensityDataExcluded[0], concatenatedDataset.intensityDataExcluded[0])
			numpy.testing.assert_array_equal(expectedDataset.intensityDataExcluded[1], concatenatedDataset.intensityDataExcluded[1])
			self.assertListEqual(expectedDataset.intensityDataExcluded[2], concatenatedDataset.intensityDataExcluded[2])
		with self.subTest(msg='Checking featureMetadataExcluded'):
			assert_frame_equal(expectedDataset.featureMetadataExcluded[0][0], concatenatedDataset.featureMetadataExcluded[0][0])
			assert_frame_equal(expectedDataset.featureMetadataExcluded[0][1], concatenatedDataset.featureMetadataExcluded[0][1])
			assert_frame_equal(expectedDataset.featureMetadataExcluded[1][0], concatenatedDataset.featureMetadataExcluded[1][0])
			assert_frame_equal(expectedDataset.featureMetadataExcluded[1][1], concatenatedDataset.featureMetadataExcluded[1][1])
			self.assertListEqual(expectedDataset.featureMetadataExcluded[2], concatenatedDataset.featureMetadataExcluded[2])
		with self.subTest(msg='Checking expectedConcentrationExcluded'):
			assert_frame_equal(expectedDataset.expectedConcentrationExcluded[0][0], concatenatedDataset.expectedConcentrationExcluded[0][0])
			assert_frame_equal(expectedDataset.expectedConcentrationExcluded[0][1], concatenatedDataset.expectedConcentrationExcluded[0][1])
			assert_frame_equal(expectedDataset.expectedConcentrationExcluded[1][0], concatenatedDataset.expectedConcentrationExcluded[1][0])
			assert_frame_equal(expectedDataset.expectedConcentrationExcluded[1][1], concatenatedDataset.expectedConcentrationExcluded[1][1])
			self.assertListEqual(expectedDataset.expectedConcentrationExcluded[2], concatenatedDataset.expectedConcentrationExcluded[2])
		with self.subTest(msg='Checking excludedFlag'):
			self.assertListEqual(expectedDataset.excludedFlag, concatenatedDataset.excludedFlag)
		with self.subTest(msg='Checking featureMask'):
			numpy.testing.assert_array_equal(expectedDataset.featureMask, concatenatedDataset.featureMask)
		with self.subTest(msg='Checking sampleMask'):
			numpy.testing.assert_array_equal(expectedDataset.sampleMask, concatenatedDataset.sampleMask)
		with self.subTest(msg='Checking calibration'):
			for i in range(len(expectedDataset.calibration)):
				assert_frame_equal(expectedDataset.calibration[i]['calibSampleMetadata'], concatenatedDataset.calibration[i]['calibSampleMetadata'])
				assert_frame_equal(expectedDataset.calibration[i]['calibFeatureMetadata'], concatenatedDataset.calibration[i]['calibFeatureMetadata'])
				numpy.testing.assert_array_equal(expectedDataset.calibration[i]['calibIntensityData'], concatenatedDataset.calibration[i]['calibIntensityData'])
				assert_frame_equal(expectedDataset.calibration[i]['calibExpectedConcentration'], concatenatedDataset.calibration[i]['calibExpectedConcentration'])


	def test_targeteddataset_applymasks(self):
		# remove feature3
		expectedDataset = copy.deepcopy(self.expectedAddDataset)
		# featureMask
		expectedDataset.featureMask = numpy.array([True,  True,  True,  True], dtype=bool)
		# featureMetadata
		expectedDataset.featureMetadata = expectedDataset.featureMetadata.loc[[True,True,False,True,True] ,:]
		expectedDataset.featureMetadata.reset_index(drop=True, inplace=True)
		# intensityData
		expectedDataset._intensityData = expectedDataset._intensityData[:,[True,True,False,True,True]]
		# expectedConcentration
		expectedDataset.expectedConcentration = expectedDataset.expectedConcentration.loc[:, [True, True, False, True, True]]
		# calibration
		# do not bother with calibPeakInfo as it's not modified
		expectedDataset.calibration[0]['calibFeatureMetadata'] = expectedDataset.calibration[0]['calibFeatureMetadata'].loc[[True,True,False] ,:]
		expectedDataset.calibration[0]['calibFeatureMetadata'].reset_index(drop=True, inplace=True)
		expectedDataset.calibration[0]['calibIntensityData'] = expectedDataset.calibration[0]['calibIntensityData'][:, [True,True,False]]
		expectedDataset.calibration[0]['calibExpectedConcentration'] = expectedDataset.calibration[0]['calibExpectedConcentration'].loc[:, [True, True, False]]
		expectedDataset.calibration[1]['calibFeatureMetadata'] = expectedDataset.calibration[1]['calibFeatureMetadata'].loc[[True, True, False], :]
		expectedDataset.calibration[1]['calibFeatureMetadata'].reset_index(drop=True, inplace=True)
		expectedDataset.calibration[1]['calibIntensityData'] = expectedDataset.calibration[1]['calibIntensityData'][:,[True, True, False]]
		expectedDataset.calibration[1]['calibExpectedConcentration'] = expectedDataset.calibration[1]['calibExpectedConcentration'].loc[:,[True, True, False]]
		expectedDataset.calibration[2]['calibFeatureMetadata'] = expectedDataset.calibration[2]['calibFeatureMetadata'].loc[[True, False, True, True], :]
		expectedDataset.calibration[2]['calibFeatureMetadata'].reset_index(drop=True, inplace=True)
		expectedDataset.calibration[2]['calibIntensityData'] = expectedDataset.calibration[2]['calibIntensityData'][:,[True, False, True, True]]
		expectedDataset.calibration[2]['calibExpectedConcentration'] = expectedDataset.calibration[2]['calibExpectedConcentration'].loc[:,[True, False, True, True]]
		expectedDataset.calibration[3]['calibFeatureMetadata'] = expectedDataset.calibration[3]['calibFeatureMetadata'].loc[[True, False, True, True], :]
		expectedDataset.calibration[3]['calibFeatureMetadata'].reset_index(drop=True, inplace=True)
		expectedDataset.calibration[3]['calibIntensityData'] = expectedDataset.calibration[3]['calibIntensityData'][:,[True, False, True, True]]
		expectedDataset.calibration[3]['calibExpectedConcentration'] = expectedDataset.calibration[3]['calibExpectedConcentration'].loc[:,[True, False, True, True]]
		# excludedFlag
		expectedDataset.excludedFlag.append('Features')

		maskedDataset = copy.deepcopy(self.expectedAddDataset)
		maskedDataset.featureMask[2] = False
		maskedDataset.applyMasks()

		with self.subTest(msg='Checking class'):
			self.assertEqual(type(maskedDataset),type(expectedDataset))
		with self.subTest(msg='Checking sampleMetadata'):
			assert_frame_equal(maskedDataset.sampleMetadata,expectedDataset.sampleMetadata)
		with self.subTest(msg='Checking featureMetadata'):
			assert_frame_equal(maskedDataset.featureMetadata.reindex(sorted(maskedDataset.featureMetadata),axis=1), expectedDataset.featureMetadata.reindex(sorted(expectedDataset.featureMetadata),axis=1))
		with self.subTest(msg='Checking _intensityData'):
			numpy.testing.assert_array_equal(maskedDataset._intensityData, expectedDataset._intensityData)
		with self.subTest(msg='Checking expectedConcentration'):
			assert_frame_equal(maskedDataset.expectedConcentration.reindex(sorted(maskedDataset.expectedConcentration), axis=1),expectedDataset.expectedConcentration.reindex(sorted(expectedDataset.expectedConcentration), axis=1))
		with self.subTest(msg='Checking Attributes'):
			# same Attributes
			self.assertEqual(maskedDataset.Attributes.keys(), expectedDataset.Attributes.keys())
			for k in expectedDataset.Attributes.keys():
				# Log cannot be match 1 to 1 as we can't reproduce the datetime on run time of the __add__ and __init__ logs
				if k == 'Log':
					# only check columns without a datetime that can't be reproduced
					for i in [0, 1, 2, 4, 5, 6, 12, 13, 14]:
						self.assertListEqual(expectedDataset.Attributes[k][i], maskedDataset.Attributes[k][i])
				else:
					# will cover all cases unless Generic.json is changed
					if type(expectedDataset.Attributes[k]) is list:
						self.assertListEqual(expectedDataset.Attributes[k], maskedDataset.Attributes[k])
					else:
						self.assertEqual(expectedDataset.Attributes[k], maskedDataset.Attributes[k])
		with self.subTest(msg='Checking sampleMetadataExcluded'):
			assert_frame_equal(expectedDataset.sampleMetadataExcluded[0][0], maskedDataset.sampleMetadataExcluded[0][0])
			assert_frame_equal(expectedDataset.sampleMetadataExcluded[0][1], maskedDataset.sampleMetadataExcluded[0][1])
			assert_frame_equal(expectedDataset.sampleMetadataExcluded[1][0], maskedDataset.sampleMetadataExcluded[1][0])
			assert_frame_equal(expectedDataset.sampleMetadataExcluded[1][1], maskedDataset.sampleMetadataExcluded[1][1])
			self.assertListEqual(expectedDataset.sampleMetadataExcluded[2], maskedDataset.sampleMetadataExcluded[2])
		with self.subTest(msg='Checking intensityMetadataExcluded'):
			numpy.testing.assert_array_equal(expectedDataset.intensityDataExcluded[0], maskedDataset.intensityDataExcluded[0])
			numpy.testing.assert_array_equal(expectedDataset.intensityDataExcluded[1], maskedDataset.intensityDataExcluded[1])
			self.assertListEqual(expectedDataset.intensityDataExcluded[2], maskedDataset.intensityDataExcluded[2])
		with self.subTest(msg='Checking featureMetadataExcluded'):
			assert_frame_equal(expectedDataset.featureMetadataExcluded[0][0], maskedDataset.featureMetadataExcluded[0][0])
			assert_frame_equal(expectedDataset.featureMetadataExcluded[0][1], maskedDataset.featureMetadataExcluded[0][1])
			assert_frame_equal(expectedDataset.featureMetadataExcluded[1][0], maskedDataset.featureMetadataExcluded[1][0])
			assert_frame_equal(expectedDataset.featureMetadataExcluded[1][1], maskedDataset.featureMetadataExcluded[1][1])
			self.assertListEqual(expectedDataset.featureMetadataExcluded[2], maskedDataset.featureMetadataExcluded[2])
		with self.subTest(msg='Checking expectedConcentrationExcluded'):
			assert_frame_equal(expectedDataset.expectedConcentrationExcluded[0][0], maskedDataset.expectedConcentrationExcluded[0][0])
			assert_frame_equal(expectedDataset.expectedConcentrationExcluded[0][1], maskedDataset.expectedConcentrationExcluded[0][1])
			assert_frame_equal(expectedDataset.expectedConcentrationExcluded[1][0], maskedDataset.expectedConcentrationExcluded[1][0])
			assert_frame_equal(expectedDataset.expectedConcentrationExcluded[1][1], maskedDataset.expectedConcentrationExcluded[1][1])
			self.assertListEqual(expectedDataset.expectedConcentrationExcluded[2],maskedDataset.expectedConcentrationExcluded[2])
		with self.subTest(msg='Checking excludedFlag'):
			self.assertListEqual(expectedDataset.excludedFlag, maskedDataset.excludedFlag)
		with self.subTest(msg='Checking featureMask'):
			numpy.testing.assert_array_equal(expectedDataset.featureMask, maskedDataset.featureMask)
		with self.subTest(msg='Checking sampleMask'):
			numpy.testing.assert_array_equal(expectedDataset.sampleMask, maskedDataset.sampleMask)
		with self.subTest(msg='Checking calibration'):
			for i in range(len(expectedDataset.calibration)):
				assert_frame_equal(expectedDataset.calibration[i]['calibSampleMetadata'], maskedDataset.calibration[i]['calibSampleMetadata'])
				assert_frame_equal(expectedDataset.calibration[i]['calibFeatureMetadata'], maskedDataset.calibration[i]['calibFeatureMetadata'])
				numpy.testing.assert_array_equal(expectedDataset.calibration[i]['calibIntensityData'], maskedDataset.calibration[i]['calibIntensityData'])
				assert_frame_equal(expectedDataset.calibration[i]['calibExpectedConcentration'], maskedDataset.calibration[i]['calibExpectedConcentration'])


	def test_targeteddataset_updatemasks_samples(self):

		with self.subTest(msg='Default Parameters'):
			targetedData4 = copy.deepcopy(self.targetedData4)
			expectedSampleMask = numpy.array([False, False, False, False, False, True, True, True, True, True, True, True, True, True, True,True, False, False], dtype=bool)

			targetedData4.initialiseMasks()
			targetedData4.updateMasks(filterFeatures=False)

			numpy.testing.assert_array_equal(expectedSampleMask, targetedData4.sampleMask)

		with self.subTest(msg='Export SP and ER'):
			targetedData4 = copy.deepcopy(self.targetedData4)
			expectedSampleMask = numpy.array([False, False, False, False, False, True, True, True, True, True, True, False, False, False, False, False, True, False], dtype=bool)

			targetedData4.initialiseMasks()
			targetedData4.updateMasks(filterFeatures=False, sampleTypes=[SampleType.StudyPool, SampleType.ExternalReference], assayRoles=[AssayRole.PrecisionReference])

			numpy.testing.assert_array_equal(expectedSampleMask, targetedData4.sampleMask)

		with self.subTest(msg='Export Dilution Samples only'):
			targetedData4 = copy.deepcopy(self.targetedData4)
			expectedSampleMask = numpy.array([True, True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False], dtype=bool)

			targetedData4.initialiseMasks()
			targetedData4.updateMasks(filterFeatures=False, sampleTypes=[SampleType.StudyPool], assayRoles=[AssayRole.LinearityReference])

			numpy.testing.assert_array_equal(expectedSampleMask, targetedData4.sampleMask)


	def test_targeteddataset_updatemasks_features(self):

		with self.subTest(msg='Default Parameters'):
			targetedData4 = copy.deepcopy(self.targetedData4)
			expectedFeatureMask = numpy.array([True, True, True, True, True], dtype=bool)

			targetedData4.initialiseMasks()
			targetedData4.updateMasks(filterSamples=False)

			numpy.testing.assert_array_equal(expectedFeatureMask, targetedData4.featureMask)

		with self.subTest(msg='No QuantificationType.IS'):
			targetedData4 = copy.deepcopy(self.targetedData4)
			qT = [QuantificationType.QuantOwnLabeledAnalogue, QuantificationType.QuantAltLabeledAnalogue, QuantificationType.QuantOther, QuantificationType.Monitored]
			cM = [CalibrationMethod.backcalculatedIS, CalibrationMethod.noIS, CalibrationMethod.noCalibration]

			expectedFeatureMask = numpy.array([False, True, True, True, True], dtype=bool)

			targetedData4.initialiseMasks()
			targetedData4.updateMasks(filterSamples=False, quantificationTypes=qT, calibrationMethods=cM)

			numpy.testing.assert_array_equal(expectedFeatureMask, targetedData4.featureMask)

		with self.subTest(msg='No QuantificationType.QuantOwnLabeledAnalogue'):
			targetedData4 = copy.deepcopy(self.targetedData4)
			qT = [QuantificationType.IS, QuantificationType.QuantAltLabeledAnalogue, QuantificationType.QuantOther, QuantificationType.Monitored]
			cM = [CalibrationMethod.backcalculatedIS, CalibrationMethod.noIS, CalibrationMethod.noCalibration]

			expectedFeatureMask = numpy.array([True, False, True, True, True], dtype=bool)

			targetedData4.initialiseMasks()
			targetedData4.updateMasks(filterSamples=False, quantificationTypes=qT, calibrationMethods=cM)

			numpy.testing.assert_array_equal(expectedFeatureMask, targetedData4.featureMask)

		with self.subTest(msg='No QuantificationType.QuantAltLabeledAnalogue'):
			targetedData4 = copy.deepcopy(self.targetedData4)
			qT = [QuantificationType.IS, QuantificationType.QuantOwnLabeledAnalogue, QuantificationType.QuantOther, QuantificationType.Monitored]
			cM = [CalibrationMethod.backcalculatedIS, CalibrationMethod.noIS, CalibrationMethod.noCalibration]

			expectedFeatureMask = numpy.array([True, True, False, True, True], dtype=bool)

			targetedData4.initialiseMasks()
			targetedData4.updateMasks(filterSamples=False, quantificationTypes=qT, calibrationMethods=cM)

			numpy.testing.assert_array_equal(expectedFeatureMask, targetedData4.featureMask)

		with self.subTest(msg='No QuantificationType.QuantOther'):
			targetedData4 = copy.deepcopy(self.targetedData4)
			qT = [QuantificationType.IS, QuantificationType.QuantOwnLabeledAnalogue, QuantificationType.QuantAltLabeledAnalogue, QuantificationType.Monitored]
			cM = [CalibrationMethod.backcalculatedIS, CalibrationMethod.noIS, CalibrationMethod.noCalibration]

			expectedFeatureMask = numpy.array([True, True, True, False, True], dtype=bool)

			targetedData4.initialiseMasks()
			targetedData4.updateMasks(filterSamples=False, quantificationTypes=qT, calibrationMethods=cM)

			numpy.testing.assert_array_equal(expectedFeatureMask, targetedData4.featureMask)

		with self.subTest(msg='No QuantificationType.Monitored'):
			targetedData4 = copy.deepcopy(self.targetedData4)
			qT = [QuantificationType.IS, QuantificationType.QuantOwnLabeledAnalogue, QuantificationType.QuantAltLabeledAnalogue, QuantificationType.QuantOther]
			cM = [CalibrationMethod.backcalculatedIS, CalibrationMethod.noIS, CalibrationMethod.noCalibration]

			expectedFeatureMask = numpy.array([True, True, True, True, False], dtype=bool)

			targetedData4.initialiseMasks()
			targetedData4.updateMasks(filterSamples=False, quantificationTypes=qT, calibrationMethods=cM)

			numpy.testing.assert_array_equal(expectedFeatureMask, targetedData4.featureMask)

		with self.subTest(msg='No CalibrationMethod.backcalculatedIS'):
			targetedData4 = copy.deepcopy(self.targetedData4)
			qT = [QuantificationType.IS, QuantificationType.QuantOwnLabeledAnalogue, QuantificationType.QuantAltLabeledAnalogue, QuantificationType.QuantOther, QuantificationType.Monitored]
			cM = [CalibrationMethod.noIS, CalibrationMethod.noCalibration]

			expectedFeatureMask = numpy.array([True, False, False, True, True], dtype=bool)

			targetedData4.initialiseMasks()
			targetedData4.updateMasks(filterSamples=False, quantificationTypes=qT, calibrationMethods=cM)

			numpy.testing.assert_array_equal(expectedFeatureMask, targetedData4.featureMask)

		with self.subTest(msg='No CalibrationMethod.noIS'):
			targetedData4 = copy.deepcopy(self.targetedData4)
			qT = [QuantificationType.IS, QuantificationType.QuantOwnLabeledAnalogue, QuantificationType.QuantAltLabeledAnalogue, QuantificationType.QuantOther, QuantificationType.Monitored]
			cM = [CalibrationMethod.backcalculatedIS, CalibrationMethod.noCalibration]

			expectedFeatureMask = numpy.array([False, True, True, False, True], dtype=bool)

			targetedData4.initialiseMasks()
			targetedData4.updateMasks(filterSamples=False, quantificationTypes=qT, calibrationMethods=cM)

			numpy.testing.assert_array_equal(expectedFeatureMask, targetedData4.featureMask)

		with self.subTest(msg='No CalibrationMethod.noCalibration'):
			targetedData4 = copy.deepcopy(self.targetedData4)
			qT = [QuantificationType.IS, QuantificationType.QuantOwnLabeledAnalogue, QuantificationType.QuantAltLabeledAnalogue, QuantificationType.QuantOther, QuantificationType.Monitored]
			cM = [CalibrationMethod.backcalculatedIS, CalibrationMethod.noIS]

			expectedFeatureMask = numpy.array([True, True, True, True, False], dtype=bool)

			targetedData4.initialiseMasks()
			targetedData4.updateMasks(filterSamples=False, quantificationTypes=qT, calibrationMethods=cM)

			numpy.testing.assert_array_equal(expectedFeatureMask, targetedData4.featureMask)

		with self.subTest(msg='No QuantificationType.IS, CalibrationMethod.noCalibration'):
			targetedData4 = copy.deepcopy(self.targetedData4)
			qT = [QuantificationType.QuantOwnLabeledAnalogue, QuantificationType.QuantAltLabeledAnalogue, QuantificationType.QuantOther, QuantificationType.Monitored]
			cM = [CalibrationMethod.backcalculatedIS, CalibrationMethod.noIS]

			expectedFeatureMask = numpy.array([False, True, True, True, False], dtype=bool)

			targetedData4.initialiseMasks()
			targetedData4.updateMasks(filterSamples=False, quantificationTypes=qT, calibrationMethods=cM)

			numpy.testing.assert_array_equal(expectedFeatureMask, targetedData4.featureMask)


	def test_targeteddataset_updateMasks_raises(self):
		targetedData4 = copy.deepcopy(self.targetedData4)

		with self.subTest(msg='Sample Types'):
			self.assertRaises(TypeError, targetedData4.updateMasks, sampleTypes=1)
			self.assertRaises(TypeError, targetedData4.updateMasks, sampleTypes=[1, 2, 4])

		with self.subTest(msg='Assay Roles'):
			self.assertRaises(TypeError, targetedData4.updateMasks, assayRoles=1)
			self.assertRaises(TypeError, targetedData4.updateMasks, assayRoles=[1, 2, 4])


	def test_targeteddataset_getsamplemetadatafromfilename(self):
		"""
		Test we are parsing NPC MS filenames correctly (PCSOP.081).
		"""
		with self.subTest(msg='Sample Base Name'):
			basename = pandas.Series(['Test1_HPOS_ToF01_P1W02',
									  'Test2_RPOS_ToF02_U2W03',
									  'Test3_RNEG_ToF03_S3W04',
									  'Test4_LPOS_ToF04_P4W05_LTR',
									  'Test5_LNEG_ToF05_U5W06_SR',
									  'Test6_HPOS_ToF06_S4W05_MR',
									  'Test1_HPOS_ToF01_P1W02',
									  'Test2_RPOS_ToF02_U2W03',
									  'Test3_RNEG_ToF03_S3W04',
									  'Test4_RPOS_ToF04_B1S1_SR',
									  'Test5_LPOS_ToF05_B2E2_SR',
									  'Test6_LNEG_ToF06_B3SRD01',
									  'Test1_HPOS_ToF06_Blank01',
									  'Test1_HPOS_ToF06_IC02',
									  'Test1_HPOS_ToF06_EIC21'],
									  name='Sample Base Name',
									  dtype=str)
			pandas.testing.assert_series_equal(self.targetedData5.sampleMetadata['Sample Base Name'], basename)

		with self.subTest(msg='Study'):
			study = pandas.Series(['Test1',
								   'Test2',
								   'Test3',
								   'Test4',
								   'Test5',
								   'Test6',
								   'Test1',
								   'Test2',
								   'Test3',
								   'Test4',
								   'Test5',
								   'Test6',
								   'Test1',
								   'Test1',
								   'Test1'],
								   name='Study',
								   dtype='str')
			pandas.testing.assert_series_equal(self.targetedData5.sampleMetadata['Study'], study)

		with self.subTest(msg='Chromatography'):
			chromatography = pandas.Series(['H',
											'R',
											'R',
											'L',
											'L',
											'H',
											'H',
											'R',
											'R',
											'R',
											'L',
											'L',
											'H',
											'H',
											'H'],
											name='Chromatography',
											dtype='str')
			pandas.testing.assert_series_equal(self.targetedData5.sampleMetadata['Chromatography'], chromatography)

		with self.subTest(msg='Ionisation'):
			ionisation = pandas.Series(['POS',
										'POS',
										'NEG',
										'POS',
										'NEG',
										'POS',
										'POS',
										'POS',
										'NEG',
										'POS',
										'POS',
										'NEG',
										'POS',
										'POS',
										'POS'],
										name='Ionisation',
										dtype='str')
			pandas.testing.assert_series_equal(self.targetedData5.sampleMetadata['Ionisation'], ionisation)

		with self.subTest(msg='Instrument'):
			instrument = pandas.Series(['ToF01',
										'ToF02',
										'ToF03',
										'ToF04',
										'ToF05',
										'ToF06',
										'ToF01',
										'ToF02',
										'ToF03',
										'ToF04',
										'ToF05',
										'ToF06',
										'ToF06',
										'ToF06',
										'ToF06'],
										name='Instrument',
										dtype='str')
			pandas.testing.assert_series_equal(self.targetedData5.sampleMetadata['Instrument'], instrument)

		with self.subTest(msg='Rerun'):
			reRun = pandas.Series(['',
								'',
								'',
								'',
								'',
								'',
								'',
								'b',
								'',
								'q',
								'',
								'',
								'',
								'',
								''],
								name='Re-Run',
								dtype='str')
			pandas.testing.assert_series_equal(self.targetedData5.sampleMetadata['Re-Run'], reRun)

		with self.subTest(msg='Suplemental Injections'):
			suplemental = pandas.Series(['',
										'',
										'',
										'',
										'',
										'',
										'',
										'',
										'2',
										'',
										'',
										'9',
										'',
										'',
										''],
										name='Suplemental Injections',
										dtype='str')
			pandas.testing.assert_series_equal(self.targetedData5.sampleMetadata['Suplemental Injections'], suplemental)

		with self.subTest(msg='Skipped'):
			skipped = pandas.Series([False,
									False,
									False,
									False,
									False,
									False,
									True,
									False,
									False,
									False,
									False,
									False,
									False,
									False,
									False],
									name='Skipped',
									dtype='bool')
			pandas.testing.assert_series_equal(self.targetedData5.sampleMetadata['Skipped'], skipped)

		with self.subTest(msg='Matrix'):
			matrix = pandas.Series(['P',
									'U',
									'S',
									'P',
									'U',
									'S',
									'P',
									'U',
									'S',
									'',
									'',
									'',
									'',
									'',
									''],
									name='Matrix',
									dtype='str')
			pandas.testing.assert_series_equal(self.targetedData5.sampleMetadata['Matrix'], matrix)

		with self.subTest(msg='Well'):
			well = pandas.Series([2,
								3,
								4,
								5,
								6,
								5,
								2,
								3,
								4,
								1,
								2,
								1,
								-1,
								-1,
								-1],
								name='Well',
								dtype='int')
			pandas.testing.assert_series_equal(self.targetedData5.sampleMetadata['Well'], well, check_dtype=False)
			self.assertEqual(self.targetedData5.sampleMetadata['Well'].dtype.kind, well.dtype.kind)

		with self.subTest(msg='Plate'):
			plate = pandas.Series([1,
									2,
									3,
									4,
									5,
									4,
									1,
									2,
									3,
									1,
									2,
									3,
									1,
									2,
									21],
									name='Plate',
									dtype='int')
			pandas.testing.assert_series_equal(self.targetedData5.sampleMetadata['Plate'], plate, check_dtype=False)
			self.assertEqual(self.targetedData5.sampleMetadata['Plate'].dtype.kind, well.dtype.kind)

		with self.subTest(msg='Batch'):
			batch = pandas.Series([numpy.nan,
								numpy.nan,
								numpy.nan,
								numpy.nan,
								numpy.nan,
								numpy.nan,
								numpy.nan,
								numpy.nan,
								numpy.nan,
								1.0,
								2.0,
								3.0,
								numpy.nan,
								numpy.nan,
								numpy.nan],
								name='Batch',
								dtype=float)
			pandas.testing.assert_series_equal(self.targetedData5.sampleMetadata['Batch'], batch)

		with self.subTest(msg='Dilution'):
			dilution = pandas.Series([numpy.nan,
									numpy.nan,
									numpy.nan,
									numpy.nan,
									numpy.nan,
									numpy.nan,
									numpy.nan,
									numpy.nan,
									numpy.nan,
									numpy.nan,
									numpy.nan,
									1.0,
									numpy.nan,
									numpy.nan,
									numpy.nan],
									name='Dilution',
									dtype='float')
			pandas.testing.assert_series_equal(self.targetedData5.sampleMetadata['Dilution'], dilution)

		with self.subTest(msg='AssayRole'):
			assayRole = pandas.Series([AssayRole.Assay,
									AssayRole.Assay,
									AssayRole.Assay,
									AssayRole.PrecisionReference,
									AssayRole.PrecisionReference,
									AssayRole.PrecisionReference,
									AssayRole.Assay,
									AssayRole.Assay,
									AssayRole.Assay,
									AssayRole.PrecisionReference,
									AssayRole.PrecisionReference,
									AssayRole.LinearityReference,
									AssayRole.LinearityReference,
									AssayRole.Assay,
									AssayRole.Assay],
									name='AssayRole',
									dtype=object)
			pandas.testing.assert_series_equal(self.targetedData5.sampleMetadata['AssayRole'], assayRole)

		with self.subTest(msg='SampleType'):
			sampleType = pandas.Series([SampleType.StudySample,
									SampleType.StudySample,
									SampleType.StudySample,
									SampleType.ExternalReference,
									SampleType.StudyPool,
									SampleType.MethodReference,
									SampleType.StudySample,
									SampleType.StudySample,
									SampleType.StudySample,
									SampleType.StudyPool,
									SampleType.StudyPool,
									SampleType.StudyPool,
									SampleType.ProceduralBlank,
									SampleType.StudyPool,
									SampleType.StudyPool],
									name='SampleType',
									dtype=object)
			pandas.testing.assert_series_equal(self.targetedData5.sampleMetadata['SampleType'], sampleType)


	def test_targeteddataset_fillbatch_correctionbatch(self):
		targetedData6 = copy.deepcopy(self.targetedData6)
		targetedData6._fillBatches()
		correctionBatch = pandas.Series([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
										 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0,
										 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, numpy.nan],
										name='Correction Batch',
										dtype=float)
		pandas.testing.assert_series_equal(targetedData6.sampleMetadata['Correction Batch'], correctionBatch)


	def test_targeteddataset_fillbatch_runorder_warns(self):
		targetedData6 = copy.deepcopy(self.targetedData6)
		targetedData6.sampleMetadata.drop('Run Order', axis=1, inplace=True)
		self.assertWarnsRegex(UserWarning, 'Unable to infer batches without run order, skipping\.', targetedData6._fillBatches)


	def test_targeteddataset_addsampleinfo_batches(self):
		# trigger fillBatch from addSampleInfo
		# same test as fillBatch
		targetedData6 = copy.deepcopy(self.targetedData6)
		targetedData6.addSampleInfo(descriptionFormat = 'Batches')
		correctionBatch = pandas.Series([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
										 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0,
										 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, numpy.nan],
										name='Correction Batch',
										dtype='float')
		# test
		pandas.testing.assert_series_equal(targetedData6.sampleMetadata['Correction Batch'], correctionBatch)


	def test_targeteddataset_addsampleinfo_filenames(self):
		# trigger _getsamplemetadatafromfilename from addSampleInfo
		# same test as _getsamplemetadatafromfilename
		targetedData5 = nPYc.TargetedDataset('', fileType='empty')
		targetedData5.sampleMetadata['Sample File Name'] = ['Test1_HPOS_ToF01_P1W02',
															'Test2_RPOS_ToF02_U2W03',
															'Test3_RNEG_ToF03_S3W04',
															'Test4_LPOS_ToF04_P4W05_LTR',
															'Test5_LNEG_ToF05_U5W06_SR',
															'Test6_HPOS_ToF06_S4W05_MR',
															'Test1_HPOS_ToF01_P1W02_x',
															'Test2_RPOS_ToF02_U2W03_b',
															'Test3_RNEG_ToF03_S3W04_2',
															'Test4_RPOS_ToF04_B1S1_SR_q',
															'Test5_LPOS_ToF05_B2E2_SR',
															'Test6_LNEG_ToF06_B3SRD01_9',
															'Test1_HPOS_ToF06_Blank01',
															'Test1_HPOS_ToF06_IC02',
															'Test1_HPOS_ToF06_EIC21']
		filenameSpec = "\n\t\t^(?P<fileName>\n\t\t\t(?P<baseName>\n\t\t\t\t(?P<study>\\w+?)\t\t\t\t\t\t\t\t\t\t# Study\n\t\t\t\t_\n\t\t\t\t(?P<chromatography>[HRL])(?P<ionisation>POS|NEG)\t# Chromatography and mode\n\t\t\t\t_\n\t\t\t\t(?P<instrument>\\w+?\\d\\d)\t\t\t\t\t\t\t# Instrument\n\t\t\t\t_\n\t\t\t\t(?P<groupingKind>Blank|E?IC|[A-Z]{1,2})(?P<groupingNo>\\d+?) # Sample grouping\n\t\t\t\t(?:\n\t\t\t\t(?P<injectionKind>[WSE]|SRD)(?P<injectionNo>\\d\\d?) # Subject ID\n\t\t\t\t)?\n\t\t\t\t(?:_(?P<reference>SR|LTR|MR))?\t\t\t\t\t  # Reference\n\t\t\t)\n\t\t\t(?:_(?P<exclusion>[xX]))?\t\t\t\t\t\t\t  # Exclusions\n\t\t\t(?:_(?P<reruns>[a-wyzA-WYZ]|[Rr]e[Rr]un\\d*?))?\t\t  # Reruns\n\t\t\t(?:_(?P<extraInjections>\\d+?))?\t\t\t\t\t\t  # Repeats\n\t\t\t(?:_(?P<exclusion2>[xX]))?\t\t\t\t\t\t\t  # badly ordered exclusions\n\t\t)$\n\t\t"
		targetedData5.addSampleInfo(descriptionFormat = 'Filenames', filenameSpec=filenameSpec)
		# expected
		basename = pandas.Series(['Test1_HPOS_ToF01_P1W02',
								  'Test2_RPOS_ToF02_U2W03',
								  'Test3_RNEG_ToF03_S3W04',
								  'Test4_LPOS_ToF04_P4W05_LTR',
								  'Test5_LNEG_ToF05_U5W06_SR',
								  'Test6_HPOS_ToF06_S4W05_MR',
								  'Test1_HPOS_ToF01_P1W02',
								  'Test2_RPOS_ToF02_U2W03',
								  'Test3_RNEG_ToF03_S3W04',
								  'Test4_RPOS_ToF04_B1S1_SR',
								  'Test5_LPOS_ToF05_B2E2_SR',
								  'Test6_LNEG_ToF06_B3SRD01',
								  'Test1_HPOS_ToF06_Blank01',
								  'Test1_HPOS_ToF06_IC02',
								  'Test1_HPOS_ToF06_EIC21'],
								 name='Sample Base Name',
								 dtype='str')
		# test
		pandas.testing.assert_series_equal(targetedData5.sampleMetadata['Sample Base Name'], basename)


	def test_targeteddataset_addsampleinfo_filenames_raise_attributeerror(self):
		# trigger AttributeError for addSampleInfo(descriptionFormat='Filenames') lacking a filenameSpec
		targetedData5 = nPYc.TargetedDataset('', fileType='empty')
		self.assertRaises(AttributeError, lambda: targetedData5.addSampleInfo(descriptionFormat='Filenames'))


	def test_targeteddataset_getsamplemetadatafromrawdata_raise_notimplemented(self):
		targetedData = nPYc.TargetedDataset('', fileType='empty')
		self.assertRaises(NotImplementedError, targetedData._getSampleMetadataFromRawData, rawDataPath='noPath')


	def test_targeteddataset_enumeration_quantificationtype_calibrationmethod(self):
		"""
		Ensures QuantificationType and CalibrationMethod __repr__ and __str__ return the expected values
		"""
		with self.subTest(msg='CalibrationMethod'):
			self.assertEqual(str(CalibrationMethod.noCalibration), 'No calibration')
			self.assertEqual(str(CalibrationMethod.noIS), 'No Internal Standard')
			self.assertEqual(str(CalibrationMethod.backcalculatedIS), 'Backcalculated with Internal Standard')

		with self.subTest(msg='QuantificationType'):
			self.assertEqual(str(QuantificationType.IS), 'Internal Standard')
			self.assertEqual(str(QuantificationType.QuantOwnLabeledAnalogue), 'Quantified and validated with own labeled analogue')
			self.assertEqual(str(QuantificationType.QuantAltLabeledAnalogue), 'Quantified and validated with alternative labeled analogue')
			self.assertEqual(str(QuantificationType.QuantOther), 'Other quantification')
			self.assertEqual(str(QuantificationType.Monitored), 'Monitored for relative information')


class test_targeteddataset_import_targetlynx_getdatasetfromxml(unittest.TestCase):
	"""
	Test import from TargetLynx XML file
	"""
	def test_targeteddataset_getdatasetfromxml(self):
		self.targetedData = nPYc.TargetedDataset('', fileType='empty')

		# 2 samples (1 calibration, 1 study sample)
		# 3 features (Feature1-IS, Feature2, Feature3)
		# Feature3 Study sample has analcon = '' to trigger Try/Except ValueError
		expectedSampleMetadata = pandas.DataFrame({'Sample File Name':['UnitTest4_targeted_file_001','UnitTest4_targeted_file_002'],'TargetLynx Sample ID':[1,2],'MassLynx Row ID':[1,2],'Sample Name':['Calibration 1','Study Sample 1'],'Sample Type':['Standard','Analyte'],'Acqu Date':['11-Sep-16','11-Sep-16'],'Acqu Time':['02:14:32','09:23:02'],'Vial':['1:A,1','1:A,2'],'Instrument':['XEVO-TQS#UnitTest','XEVO-TQS#UnitTest']})
		expectedSampleMetadata['Sample Base Name'] = expectedSampleMetadata['Sample File Name']
		expectedSampleMetadata['Metadata Available'] = False
		expectedFeatureMetadata = pandas.DataFrame({'Feature Name':['Feature1-IS','Feature2','Feature3'],'TargetLynx Feature ID':[1,2,3],'TargetLynx IS ID':['','1','1']})
		expectedIntensityData = numpy.array([[48.64601435, 48.7244571, 48.76854933],[20.60696312,273.85551508,359.219531]])
		expectedExpectedConcentration = pandas.DataFrame(numpy.array([[50., 50., 50.],[60.,numpy.nan,numpy.nan]]), columns=expectedFeatureMetadata['Feature Name'].values)
		expectedPeakResponse = pandas.DataFrame(numpy.array([[1.33416750e+05, 5.40558251e+00, 2.94037293e-01],[5.65167380e+04,2.97330493e+01,2.26275225e+01]]), columns=expectedFeatureMetadata['Feature Name'].values)
		expectedPeakArea = pandas.DataFrame(numpy.array([[133416.75, 14423.905, 784.59],[56516.738,28006.916,21313.896]]), columns=expectedFeatureMetadata['Feature Name'].values)
		expectedPeakConcentrationDeviation = pandas.DataFrame(numpy.array([[-2.70797131, -2.5510858,-2.46290134],[-65.65506146,numpy.nan,numpy.nan]]), columns=expectedFeatureMetadata['Feature Name'].values)
		expectedPeakIntegrationFlag = pandas.DataFrame({'1': ['bb', 'bb'], '2': ['bb', 'bb'], '3': ['MM', 'bb']})
		expectedPeakIntegrationFlag.columns = expectedFeatureMetadata['Feature Name'].values
		expectedPeakRT = pandas.DataFrame(numpy.array([[11.4263000488, 11.4921998978, 11.6340999603], [11.4306001663, 11.5010004044, 11.6407003403]]), columns=expectedFeatureMetadata['Feature Name'].values)

		# import
		datapath = os.path.join('..', '..', 'npc-standard-project', 'Derived_Data', 'UnitTest4_targeted.xml')
		sampleMetadata, featureMetadata, intensityData, expectedConcentration, peakResponse, peakArea, peakConcentrationDeviation, peakIntegrationFlag, peakRT = self.targetedData._TargetedDataset__getDatasetFromXML(datapath)

		# test
		assert_frame_equal(expectedSampleMetadata.reindex(sorted(expectedSampleMetadata),axis=1), sampleMetadata.reindex(sorted(sampleMetadata),axis=1))
		assert_frame_equal(expectedFeatureMetadata.reindex(sorted(expectedFeatureMetadata),axis=1), featureMetadata.reindex(sorted(featureMetadata),axis=1))
		numpy.testing.assert_array_almost_equal(expectedIntensityData, intensityData)
		assert_frame_equal(expectedExpectedConcentration.reindex(sorted(expectedExpectedConcentration),axis=1), expectedConcentration.reindex(sorted(expectedConcentration),axis=1))
		assert_frame_equal(expectedPeakResponse.reindex(sorted(expectedPeakResponse),axis=1), peakResponse.reindex(sorted(peakResponse),axis=1))
		assert_frame_equal(expectedPeakArea.reindex(sorted(expectedPeakArea),axis=1), peakArea.reindex(sorted(peakArea),axis=1))
		assert_frame_equal(expectedPeakConcentrationDeviation.reindex(sorted(expectedPeakConcentrationDeviation),axis=1), peakConcentrationDeviation.reindex(sorted(peakConcentrationDeviation),axis=1))
		assert_frame_equal(expectedPeakIntegrationFlag.reindex(sorted(expectedPeakIntegrationFlag),axis=1), peakIntegrationFlag.reindex(sorted(peakIntegrationFlag),axis=1))
		assert_frame_equal(expectedPeakRT.reindex(sorted(expectedPeakRT),axis=1), peakRT.reindex(sorted(peakRT),axis=1))


class test_targeteddataset_import_targetlynx_getcalibrationfromreport(unittest.TestCase):
	"""
	Test import of calibration report CSV file
	"""
	def setUp(self):
		self.targetedData = nPYc.TargetedDataset('', fileType='empty')

		self.calibrationReport = pandas.DataFrame({'Compound': ['Feature 1 - IS', 'Feature 2', 'Feature 3'], 'TargetLynx ID': [1, 2, 3],'Cpd Info': ['uM', 'fg/uL', 'noUnit'], 'Noise (area)': [38.95, 14.7, numpy.nan], 'LLOQ': [25, 10, 1], 'ULOQ': [1000, 2500, 2500], 'a': [0.997431, 1.04095, 1.0021], 'b': [-2.19262, numpy.nan, -0.901739], 'r': [0.997931, 0.999556, 0.999683],'r2': [0.995866, 0.999113, 0.999366], 'another column': [numpy.nan, 'something', 'something else!']})
		self.missingcolCalibrationReport = self.calibrationReport[['Compound', 'Cpd Info', 'LLOQ','ULOQ', 'another column']]


	def test_targeteddataset_getcalibrationfromreport(self):

		with tempfile.TemporaryDirectory() as tmpdirname:

			reportPath = os.path.join(tmpdirname, 'correctCalibrationReport.csv')
			self.calibrationReport.to_csv(reportPath, index=False)
			loadedReport = self.targetedData._TargetedDataset__getCalibrationFromReport(reportPath)

			assert_frame_equal(loadedReport.reindex(sorted(loadedReport), axis=1), self.calibrationReport.reindex(sorted(self.calibrationReport), axis=1))


	def test_targeteddataset_getcalibrationfromreport_raise(self):

		with tempfile.TemporaryDirectory() as tmpdirname:

			failReportPath = os.path.join(tmpdirname, 'wrongCalibrationReport.csv')
			self.missingcolCalibrationReport.to_csv(failReportPath, index=False)

			self.assertRaises(LookupError, self.targetedData._TargetedDataset__getCalibrationFromReport, path=failReportPath)


class test_targeteddataset_import_targetlynx_matchdatasettocalibrationreport(unittest.TestCase):
	"""
	Test matching of TargetLynx XML file and calibration report CSV file
	"""
	def setUp(self):
		self.targetedEmpty = nPYc.TargetedDataset('', fileType='empty')
		self.targetedEmpty.Attributes['externalID'] = []

		# ValueError on shape
		self.shapeError = dict()
		self.shapeError['goodIntensityData'] =  numpy.random.random((3, 3))
		self.shapeError['goodSampleMetadata'] = pandas.DataFrame(numpy.random.random((3, 3)))
		self.shapeError['badSampleMetadata'] = pandas.DataFrame(numpy.random.random((2, 2)))
		self.shapeError['goodFeatureMetadata'] = pandas.DataFrame(numpy.random.random((3, 3)))
		self.shapeError['badFeatureMetadata'] = pandas.DataFrame(numpy.random.random((2, 2)))
		self.shapeError['goodExpectedConcentration'] = pandas.DataFrame(numpy.random.random((3, 3)))
		self.shapeError['badExpectedConcentration'] = pandas.DataFrame(numpy.random.random((2, 2)))
		self.shapeError['goodPeakResponse'] = pandas.DataFrame(numpy.random.random((3, 3)))
		self.shapeError['badPeakResponse'] = pandas.DataFrame(numpy.random.random((2, 2)))
		self.shapeError['goodPeakArea'] = pandas.DataFrame(numpy.random.random((3, 3)))
		self.shapeError['badPeakArea'] = pandas.DataFrame(numpy.random.random((2, 2)))
		self.shapeError['goodPeakConcentrationDeviation'] = pandas.DataFrame(numpy.random.random((3, 3)))
		self.shapeError['badPeakConcentrationDeviation'] = pandas.DataFrame(numpy.random.random((2, 2)))
		self.shapeError['goodPeakIntegrationFlag'] = pandas.DataFrame(numpy.random.random((3, 3)))
		self.shapeError['badPeakIntegrationFlag'] = pandas.DataFrame(numpy.random.random((2, 2)))
		self.shapeError['goodPeakRT'] = pandas.DataFrame(numpy.random.random((3, 3)))
		self.shapeError['badPeakRT'] = pandas.DataFrame(numpy.random.random((2, 2)))

		# ValueError on JSON SOP
		self.targetedFailSOPIS = nPYc.TargetedDataset('', fileType='empty')
		self.targetedFailSOPIS.Attributes = dict()
		self.targetedFailSOPIS.Attributes['compoundID'] = ['1','2','3','4']
		self.targetedFailSOPIS.Attributes['compoundName'] = ['Feature1','Feature2','Feature3','Feature4']
		self.targetedFailSOPIS.Attributes['IS'] = ['False', 'False', 'True', 'True']
		self.targetedFailSOPIS.Attributes['unitFinal'] = ['uM', 'pg/uL', 'pg/uL', 'pg/uL']
		self.targetedFailSOPIS.Attributes['unitCorrectionFactor'] = ['1','1','1','1']
		self.targetedFailSOPIS.Attributes['calibrationMethod'] = ['backcalculatedIS','noIS','noIS','backcalculatedIS']
		self.targetedFailSOPIS.Attributes['calibrationEquation'] = ['((area * responseFactor)-b)/a','','','((area * responseFactor)-b)/a']
		self.targetedFailSOPIS.Attributes['quantificationType'] = ['','IS','IS','']
		self.targetedFailSOPIS.Attributes['externalID'] = []

		self.targetedFailSOPMonitored = nPYc.TargetedDataset('', fileType='empty')
		self.targetedFailSOPMonitored.Attributes = dict()
		self.targetedFailSOPMonitored.Attributes['compoundID'] = ['1', '2', '3', '4']
		self.targetedFailSOPMonitored.Attributes['compoundName'] = ['Feature1', 'Feature2', 'Feature3', 'Feature4']
		self.targetedFailSOPMonitored.Attributes['IS'] = ['True', 'False', 'False', 'False']
		self.targetedFailSOPMonitored.Attributes['unitFinal'] = ['uM', 'pg/uL', 'pg/uL', 'pg/uL']
		self.targetedFailSOPMonitored.Attributes['unitCorrectionFactor'] = ['1', '1', '1', '1']
		self.targetedFailSOPMonitored.Attributes['calibrationMethod'] = ['noIS', 'backcalculatedIS', 'noCalibration', 'noCalibration']
		self.targetedFailSOPMonitored.Attributes['calibrationEquation'] = ['', '', '','((area * responseFactor)-b)/a']
		self.targetedFailSOPMonitored.Attributes['quantificationType'] = ['IS', 'Monitored', 'Monitored', 'QuantOwnLabeledAnalogue']
		self.targetedFailSOPMonitored.Attributes['externalID'] = []

		# ValueError on calib /SOP match
		self.targetedFailSOPCalibSize = nPYc.TargetedDataset('', fileType='empty')
		self.targetedFailSOPCalibSize.Attributes['compoundID'] = ['1', '2', '3', '4']
		self.targetedFailSOPCalibSize.Attributes['compoundName'] = ['Feature1', 'Feature2', 'Feature3', 'Feature4']
		self.targetedFailSOPCalibSize.Attributes['IS'] = ['True', 'False', 'False', 'False']
		self.targetedFailSOPCalibSize.Attributes['unitFinal'] = ['noUnit', 'pg/uL', 'pg/uL', 'noUnit']
		self.targetedFailSOPCalibSize.Attributes['unitCorrectionFactor'] = ['1', '1', '1', '1']
		self.targetedFailSOPCalibSize.Attributes['calibrationMethod'] = ['noIS', 'backcalculatedIS', 'backcalculatedIS','noCalibration']
		self.targetedFailSOPCalibSize.Attributes['calibrationEquation'] = ['', '((area * responseFactor)-b)/a', '((area * responseFactor)-b)/a', '']
		self.targetedFailSOPCalibSize.Attributes['quantificationType'] = ['IS','QuantOwnLabeledAnalogue','QuantAltLabeledAnalogue','Monitored']
		self.targetedFailSOPCalibSize.Attributes['externalID'] = []

		self.targetedFailSOPCalibID = nPYc.TargetedDataset('', fileType='empty')
		self.targetedFailSOPCalibID.Attributes['compoundID'] = ['1', '2', '3', '4']
		self.targetedFailSOPCalibID.Attributes['compoundName'] = ['Feature1', 'Feature2', 'Feature3', 'Feature4']
		self.targetedFailSOPCalibID.Attributes['IS'] = ['True', 'False', 'False', 'False']
		self.targetedFailSOPCalibID.Attributes['unitFinal'] = ['noUnit', 'pg/uL', 'pg/uL', 'noUnit']
		self.targetedFailSOPCalibID.Attributes['unitCorrectionFactor'] = ['1', '1', '1', '1']
		self.targetedFailSOPCalibID.Attributes['calibrationMethod'] = ['noIS', 'backcalculatedIS', 'backcalculatedIS','noCalibration']
		self.targetedFailSOPCalibID.Attributes['calibrationEquation'] = ['', '((area * responseFactor)-b)/a', '((area * responseFactor)-b)/a', '']
		self.targetedFailSOPCalibID.Attributes['quantificationType'] = ['IS','QuantOwnLabeledAnalogue','QuantAltLabeledAnalogue','Monitored']
		self.targetedFailSOPCalibID.Attributes['externalID'] = []
		self.targetedFailSOPCalibIDReport = pandas.DataFrame({'Compound': ['Feature 2', 'Feature 3', 'Feature 4', 'Feature 5'], 'TargetLynx ID': [2,3,4,5],'Cpd Info': ['uM', 'fg/uL', 'noUnit',''], 'Noise (area)': [38.95, 14.7, numpy.nan,14.7],'LLOQ': [25, 10, 1,1], 'ULOQ': [1000, 2500, 2500, 2500], 'a': [0.997431, 1.04095, 1.0021,1.0021], 'b': [-2.19262, numpy.nan, -0.901739,-0.901739], 'r': [0.997931, 0.999556, 0.999683,0.999683],'r2': [0.995866, 0.999113, 0.999366,0.999366],'another column': [numpy.nan, 'something', 'something else!','also something else']})

		self.targetedFailSOPCalibName = self.targetedFailSOPCalibID
		self.targetedFailSOPCalibNameReport = pandas.DataFrame({'Compound': ['Feature-1', 'Feature2', 'Feature3', 'Feature4'], 'TargetLynx ID': [1,2,3,4],'Cpd Info': ['uM', 'fg/uL', 'noUnit',''], 'Noise (area)': [38.95, 14.7, numpy.nan,14.7],'LLOQ': [25, 10, 1,1], 'ULOQ': [1000, 2500, 2500, 2500], 'a': [0.997431, 1.04095, 1.0021,1.0021], 'b': [-2.19262, numpy.nan, -0.901739,-0.901739], 'r': [0.997931, 0.999556, 0.999683,0.999683],'r2': [0.995866, 0.999113, 0.999366,0.999366],'another column': [numpy.nan, 'something', 'something else!','also something else']})

		# Warning on calib/SOP TargetLynx mismatch
		# 3 Features TL
		self.targetedWarning = dict()
		self.targetedWarning['sampleMetadata'] = pandas.DataFrame({'Sample File Name': ['UnitTest4_targeted_file_001', 'UnitTest4_targeted_file_002'],'TargetLynx Sample ID': [1, 2], 'MassLynx Row ID': [1, 2], 'Sample Name': ['Calibration 1', 'Study Sample 1'],'Sample Type': ['Standard', 'Analyte'], 'Acqu Date': ['11-Sep-16', '11-Sep-16'],'Acqu Time': ['02:14:32', '09:23:02'], 'Vial': ['1:A,1', '1:A,2'],'Instrument': ['XEVO-TQS#UnitTest', 'XEVO-TQS#UnitTest']})
		self.targetedWarning['featureMetadata'] = pandas.DataFrame({'Feature Name': ['Feature1', 'Feature2', 'Feature3'], 'TargetLynx Feature ID': [1, 2, 3],'TargetLynx IS ID': ['', '1', '1']})
		self.targetedWarning['intensityData'] = numpy.array([[48.64601435, 48.7244571, 48.76854933], [20.60696312, 273.85551508, 0.]])
		self.targetedWarning['expectedConcentration'] = pandas.DataFrame(numpy.array([[50., 50., 50.], [60., numpy.nan, numpy.nan]]), columns=self.targetedWarning['featureMetadata']['Feature Name'].values.tolist())
		self.targetedWarning['peakResponse'] = pandas.DataFrame(numpy.array([[1.33416750e+05, 5.40558251e+00, 2.94037293e-01], [5.65167380e+04, 2.97330493e+01, 2.26275225e+01]]), columns=self.targetedWarning['featureMetadata']['Feature Name'].values.tolist())
		self.targetedWarning['peakArea'] = pandas.DataFrame(numpy.array([[133416.75, 14423.905, 784.59], [56516.738, 28006.916, 21313.896]]), columns=self.targetedWarning['featureMetadata']['Feature Name'].values.tolist())
		self.targetedWarning['peakConcentrationDeviation'] = pandas.DataFrame(numpy.array([[-2.70797131, -2.5510858, -2.46290134], [-65.65506146, numpy.nan, numpy.nan]]), columns=self.targetedWarning['featureMetadata']['Feature Name'].values.tolist())
		self.targetedWarning['peakIntegrationFlag'] = pandas.DataFrame({'1': ['bb', 'bb'], '2': ['bb', 'bb'], '3': ['MM', 'bb']})
		self.targetedWarning['peakIntegrationFlag'].index = [1, 2]
		self.targetedWarning['peakIntegrationFlag'].columns = self.targetedWarning['featureMetadata']['Feature Name']
		self.targetedWarning['peakRT'] = pandas.DataFrame(numpy.array([[10., 11., 12.], [13., 14., 15.]]), columns=self.targetedWarning['featureMetadata']['Feature Name'].values.tolist())
		# 2 Features SOP/Calib
		self.targetedWarningID = nPYc.TargetedDataset('', fileType='empty')
		self.targetedWarningID.Attributes['compoundID'] = ['1', '2']
		self.targetedWarningID.Attributes['compoundName'] = ['Feature1', 'Feature2']
		self.targetedWarningID.Attributes['IS'] = ['True', 'False']
		self.targetedWarningID.Attributes['unitFinal'] = ['noUnit', 'pg/uL']
		self.targetedWarningID.Attributes['unitCorrectionFactor'] = ['1', '1']
		self.targetedWarningID.Attributes['calibrationMethod'] = ['noIS', 'backcalculatedIS']
		self.targetedWarningID.Attributes['calibrationEquation'] = ['', '((area * responseFactor)-b)/a']
		self.targetedWarningID.Attributes['quantificationType'] = ['IS','QuantOwnLabeledAnalogue']
		self.targetedWarningID.Attributes['chromatography'] = 'R'
		self.targetedWarningID.Attributes['ionisation'] = 'NEG'
		self.targetedWarningID.Attributes['methodName'] = 'UnitTest'
		self.targetedWarningID.Attributes['externalID'] = []
		self.targetedWarningIDReport = pandas.DataFrame({'Compound': ['Feature1', 'Feature2'], 'TargetLynx ID': [1,2],'Cpd Info': ['uM', 'fg/uL'], 'Noise (area)': [38.95, 14.7], 'LLOQ': [25, 10], 'ULOQ': [1000, 2500], 'a': [0.997431, 1.04095],'b': [-2.19262, numpy.nan], 'r': [0.997931, 0.999556],'r2': [0.995866, 0.999113],'another column': [numpy.nan, 'something']})

		# Feature1 (TL) / Feature-1 (SOP/Calib)
		self.targetedWarningName = nPYc.TargetedDataset('', fileType='empty')
		self.targetedWarningName.Attributes['compoundID'] = ['1', '2','3']
		self.targetedWarningName.Attributes['compoundName'] = ['Feature-1', 'Feature2','Feature3']
		self.targetedWarningName.Attributes['IS'] = ['True', 'False','False']
		self.targetedWarningName.Attributes['unitFinal'] = ['noUnit', 'pg/uL','pg/uL']
		self.targetedWarningName.Attributes['unitCorrectionFactor'] = ['1', '1','1']
		self.targetedWarningName.Attributes['calibrationMethod'] = ['noIS', 'backcalculatedIS', 'backcalculatedIS']
		self.targetedWarningName.Attributes['calibrationEquation'] = ['', '((area * responseFactor)-b)/a', '((area * responseFactor)-b)/a']
		self.targetedWarningName.Attributes['quantificationType'] = ['IS','QuantOwnLabeledAnalogue','QuantOwnLabeledAnalogue']
		self.targetedWarningName.Attributes['chromatography'] = 'R'
		self.targetedWarningName.Attributes['ionisation'] = 'NEG'
		self.targetedWarningName.Attributes['methodName'] = 'UnitTest'
		self.targetedWarningName.Attributes['externalID'] = []
		self.targetedWarningNameReport = pandas.DataFrame({'Compound': ['Feature-1', 'Feature2', 'Feature3'], 'TargetLynx ID': [1, 2,3], 'Cpd Info': ['uM', 'fg/uL', 'fg/uL'],'Noise (area)': [38.95, 14.7, 15.89], 'LLOQ': [25, 10, 35], 'ULOQ': [1000, 2500, 2500], 'a': [0.997431, 1.04095, 0.982527], 'b': [-2.19262, numpy.nan, -1.15896], 'r': [0.997931, 0.999556, 0.998568], 'r2': [0.995866, 0.999113,0.989957], 'another column': [numpy.nan, 'something', 'something else']})

		# working version
		# 5 samples, 2 features, Feature2 excluded, Feature1 renamed to Feature-1
		self.targetedTargetLynxData = dict()
		self.targetedTargetLynxData['sampleMetadata'] = pandas.DataFrame({'Sample File Name': ['UnitTest4_targeted_file_001', 'UnitTest4_targeted_file_002','UnitTest4_targeted_file_003', 'UnitTest4_targeted_file_004','UnitTest4_targeted_file_005'],'TargetLynx Sample ID': [1, 2,3,4,5], 'MassLynx Row ID': [1, 2,3,4,5],'Sample Name': ['Calibration 1', 'Study Sample 1','Blank 1','QC 1','Other 1'], 'Sample Type': ['Standard', 'Analyte','Blank','QC','Solvent'],'Acqu Date': ['10-Sep-16', '10-Sep-16','12-Sep-16','10-Sep-16','10-Sep-16'], 'Acqu Time': ['02:14:32', '03:23:02', '13:52:35', '14:46:40', '15:05:26'], 'Vial': ['1:A,1', '1:A,2','1:A,3','1:A,4','1:A,5'],'Instrument': ['XEVO-TQS#UnitTest', 'XEVO-TQS#UnitTest','XEVO-TQS#UnitTest', 'XEVO-TQS#UnitTest','XEVO-TQS#UnitTest']})
		self.targetedTargetLynxData['featureMetadata'] = pandas.DataFrame({'Feature Name': ['Feature1', 'Feature2'], 'TargetLynx Feature ID': [1, 2], 'TargetLynx IS ID': ['1', '1']})
		self.targetedTargetLynxData['intensityData'] = numpy.array([[1.,2.], [3.,4.],[5.,6.],[7.,8.],[9.,10.]])
		self.targetedTargetLynxData['expectedConcentration'] = pandas.DataFrame(numpy.array([[31.,32.], [33.,34.],[35.,36.],[37.,38.],[39.,40.]]),columns=self.targetedTargetLynxData['featureMetadata']['Feature Name'].values.tolist())
		self.targetedTargetLynxData['peakResponse'] = pandas.DataFrame(numpy.array([[11.,12.], [13.,14.],[15.,16.],[17.,18.],[19.,20.]]), columns=self.targetedTargetLynxData['featureMetadata']['Feature Name'].values.tolist())
		self.targetedTargetLynxData['peakArea'] = pandas.DataFrame(numpy.array([[21.,22.], [23.,24.],[25.,26.],[27.,28.],[29.,30.]]), columns=self.targetedTargetLynxData['featureMetadata']['Feature Name'].values.tolist())
		self.targetedTargetLynxData['peakConcentrationDeviation'] = pandas.DataFrame(numpy.array([[41.,42.], [43.,44.], [45.,46.], [47.,48.],[49.,50.]]), columns=self.targetedTargetLynxData['featureMetadata']['Feature Name'].values.tolist())
		self.targetedTargetLynxData['peakIntegrationFlag'] = pandas.DataFrame({'1': ['bb', 'bb','MMX', 'dbX', 'bb'], '2': ['bb', 'bb', 'MM', 'bb','dd']})
		self.targetedTargetLynxData['peakIntegrationFlag'].index = [1, 2, 3, 4, 5]
		self.targetedTargetLynxData['peakIntegrationFlag'].columns = self.targetedTargetLynxData['featureMetadata']['Feature Name'].values.tolist()
		self.targetedTargetLynxData['peakRT'] = pandas.DataFrame(numpy.array([[51.,52],[53.,54.],[55.,56.],[57.,58.],[59.,60.]]), columns=self.targetedTargetLynxData['featureMetadata']['Feature Name'].values.tolist())

		self.targeted = nPYc.TargetedDataset('', fileType='empty')
		self.targeted.Attributes['compoundID'] = ['1']
		self.targeted.Attributes['compoundName'] = ['Feature-1']
		self.targeted.Attributes['IS'] = ['True']
		self.targeted.Attributes['unitFinal'] = ['noUnit']
		self.targeted.Attributes['unitCorrectionFactor'] = [1.]
		self.targeted.Attributes['calibrationMethod'] = ['noIS']
		self.targeted.Attributes['calibrationEquation'] = ['']
		self.targeted.Attributes['quantificationType'] = ['IS']
		self.targeted.Attributes['chromatography'] = 'R'
		self.targeted.Attributes['ionisation'] = 'NEG'
		self.targeted.Attributes['methodName'] = 'UnitTest'
		self.targeted.Attributes['externalID'] = ['extID1', 'extID2']
		self.targeted.Attributes['extID1'] = ['F1']
		self.targeted.Attributes['extID2'] = ['ID1']
		self.targetedReport = pandas.DataFrame({'Compound': ['Feature-1'], 'TargetLynx ID': [1],'Cpd Info': ['uM'], 'Noise (area)': [38.95], 'LLOQ': [25], 'ULOQ': [1000], 'a': [0.997431], 'b': [-2.19262], 'r': [0.997931], 'r2': [0.995866],'another column': ['something']})

		self.targetedExpected = dict()
		self.targetedExpected['sampleMetadata'] = pandas.DataFrame({'Sample File Name': ['UnitTest4_targeted_file_001', 'UnitTest4_targeted_file_002','UnitTest4_targeted_file_003', 'UnitTest4_targeted_file_004','UnitTest4_targeted_file_005'],'TargetLynx Sample ID': [1, 2,3,4,5], 'MassLynx Row ID': [1, 2,3,4,5],'Sample Name': ['Calibration 1', 'Study Sample 1','Blank 1','QC 1','Other 1'], 'Sample Type': ['Standard', 'Analyte','Blank','QC','Solvent'],'Acqu Date': ['10-Sep-16', '10-Sep-16','12-Sep-16','10-Sep-16','10-Sep-16'], 'Acqu Time': ['02:14:32', '03:23:02', '13:52:35', '14:46:40', '15:05:26'], 'Vial': ['1:A,1', '1:A,2','1:A,3','1:A,4','1:A,5'],'Instrument': ['XEVO-TQS#UnitTest', 'XEVO-TQS#UnitTest','XEVO-TQS#UnitTest', 'XEVO-TQS#UnitTest','XEVO-TQS#UnitTest'], 'Calibrant':[True,False,False,False,False], 'Study Sample': [False,True,False,False,False], 'Blank': [False,False,True,False,False], 'QC':[False,False,False,True,False], 'Other':[False,False,False,False,True], 'Acquired Time':[datetime(2016, 9, 10, 2, 14, 32), datetime(2016, 9, 10, 3, 23, 2), datetime(2016, 9, 12, 13, 52, 35), datetime(2016, 9, 10, 14, 46,40), datetime(2016, 9, 10, 15, 5, 26)], 'Run Order':[0,1,4,2,3], 'Batch':[1,1,1,1,1]})
		self.targetedExpected['sampleMetadata']['Acquired Time'] = self.targetedExpected['sampleMetadata']['Acquired Time']
		self.targetedExpected['featureMetadata'] = pandas.DataFrame({'Feature Name':['Feature-1'],'TargetLynx Feature ID':[1], 'TargetLynx IS ID':['1'], 'IS':[True], 'calibrationEquation':[''], 'calibrationMethod':[CalibrationMethod.noIS], 'quantificationType':[QuantificationType.IS], 'unitCorrectionFactor':[1.], 'Unit':['noUnit'], 'Cpd Info':['uM'], 'LLOQ':[25.0], 'Noise (area)':[38.95], 'ULOQ':[1000.0],'a': [0.997431],'another column': ['something'],  'b': [-2.19262], 'r': [0.997931], 'r2': [0.995866], 'extID1': ['F1'], 'extID2': ['ID1']})
		self.targetedExpected['featureMetadata']['IS'] = numpy.array([True], dtype=object)
		self.targetedExpected['featureMetadata'].index = [0]
		self.targetedExpected['intensityData'] = numpy.array([[1.],[3.],[5.],[7.],[9.]])
		self.targetedExpected['expectedConcentration'] = pandas.DataFrame(numpy.array([[31.],[33.],[35.],[37.],[39.]]), columns=self.targetedExpected['featureMetadata']['Feature Name'].values.tolist())
		self.targetedExpected['sampleMetadataExcluded'] = [self.targetedExpected['sampleMetadata'][['Sample File Name', 'TargetLynx Sample ID', 'MassLynx Row ID', 'Sample Name', 'Sample Type', 'Acqu Date', 'Acqu Time', 'Vial', 'Instrument']]]
		featureMetadataExcluded = pandas.DataFrame({'Feature Name':['Feature2'],'TargetLynx Feature ID':[2], 'TargetLynx IS ID':['1'], 'IS':[numpy.nan], 'calibrationEquation':[numpy.nan], 'calibrationMethod':[numpy.nan], 'quantificationType':[numpy.nan], 'compoundName':[numpy.nan], 'compoundID':[numpy.nan], 'unitCorrectionFactor':[numpy.nan], 'Unit':[numpy.nan], 'Cpd Info':[numpy.nan], 'LLOQ':[numpy.nan], 'Noise (area)':[numpy.nan], 'ULOQ':[numpy.nan],'a': [numpy.nan],'another column': [numpy.nan],  'b': [numpy.nan], 'r': [numpy.nan], 'r2': [numpy.nan], 'extID1': [numpy.nan], 'extID2': [numpy.nan]})
		featureMetadataExcluded.index = [1]
		featureMetadataExcluded['IS'] = numpy.array([numpy.nan], dtype=object)
		featureMetadataExcluded['calibrationEquation'] = numpy.array([numpy.nan], dtype=object)
		featureMetadataExcluded['calibrationMethod'] = numpy.array([numpy.nan], dtype=object)
		featureMetadataExcluded['compoundName'] = numpy.array([numpy.nan], dtype=object)
		featureMetadataExcluded['quantificationType'] = numpy.array([numpy.nan], dtype=object)
		featureMetadataExcluded['Unit'] = numpy.array([numpy.nan], dtype=object)
		featureMetadataExcluded['Cpd Info'] = numpy.array([numpy.nan], dtype=object)
		featureMetadataExcluded['another column'] = numpy.array([numpy.nan], dtype=object)
		featureMetadataExcluded['extID1'] = numpy.array([numpy.nan], dtype=object)
		featureMetadataExcluded['extID2'] = numpy.array([numpy.nan], dtype=object)
		self.targetedExpected['featureMetadataExcluded'] = [featureMetadataExcluded]
		self.targetedExpected['intensityDataExcluded'] = [numpy.array([[2.],[4.],[6.],[8.],[10.]])]
		self.targetedExpected['expectedConcentrationExcluded'] = [pandas.DataFrame({'Feature2':[32., 34., 36., 38., 40.]})]
		self.targetedExpected['excludedFlag'] = ['Features']
		self.targetedExpected['peakResponse'] = pandas.DataFrame(numpy.array([[11.],[13.],[15.],[17.],[19.]]), columns=self.targetedExpected['featureMetadata']['Feature Name'].values.tolist())
		self.targetedExpected['peakArea'] = pandas.DataFrame(numpy.array([[21.],[23.],[25.],[27.],[29.]]), columns=self.targetedExpected['featureMetadata']['Feature Name'].values.tolist())
		self.targetedExpected['peakConcentrationDeviation'] = pandas.DataFrame(numpy.array([[41.],[43.],[45.],[47.],[49.]]), columns=self.targetedExpected['featureMetadata']['Feature Name'].values.tolist())
		self.targetedExpected['peakIntegrationFlag'] = pandas.DataFrame({'Feature-1': ['bb','bb','MMX','dbX','bb']})
		self.targetedExpected['peakIntegrationFlag'].index = [1,2,3,4,5]
		self.targetedExpected['peakIntegrationFlag'].columns = self.targetedExpected['featureMetadata']['Feature Name'].values.tolist()
		self.targetedExpected['peakRT'] = pandas.DataFrame(numpy.array([[51.],[53.],[55.],[57.],[59.]]), columns=self.targetedExpected['featureMetadata']['Feature Name'].values.tolist())


		# 5 samples, 2 features, no exclusion, Feature1 renamed to Feature-1. Feature1 has unitCorrectionFactor of 10x
		self.targetedNoExclusion = nPYc.TargetedDataset('', fileType='empty')
		self.targetedNoExclusion.Attributes['compoundID'] = ['1', '2']
		self.targetedNoExclusion.Attributes['compoundName'] = ['Feature-1', 'Feature2']
		self.targetedNoExclusion.Attributes['IS'] = ['True', 'False']
		self.targetedNoExclusion.Attributes['unitFinal'] = ['noUnit', 'pg/uL']
		self.targetedNoExclusion.Attributes['unitCorrectionFactor'] = [10.,1.]
		self.targetedNoExclusion.Attributes['calibrationMethod'] = ['noIS', 'backcalculatedIS']
		self.targetedNoExclusion.Attributes['calibrationEquation'] = ['', '((area * responseFactor)-b)/a']
		self.targetedNoExclusion.Attributes['quantificationType'] = ['IS', 'QuantOwnLabeledAnalogue']
		self.targetedNoExclusion.Attributes['chromatography'] = 'R'
		self.targetedNoExclusion.Attributes['ionisation'] = 'NEG'
		self.targetedNoExclusion.Attributes['methodName'] = 'UnitTest'
		self.targetedNoExclusion.Attributes['externalID'] = ['extID1', 'extID2']
		self.targetedNoExclusion.Attributes['extID1'] = ['F1', 'F2']
		self.targetedNoExclusion.Attributes['extID2'] = ['ID1', 'ID2']
		self.targetedNoExclusionReport = pandas.DataFrame({'Compound': ['Feature-1', 'Feature2'], 'TargetLynx ID': [1, 2],'Cpd Info': ['uM', 'fg/uL'], 'Noise (area)': [38.95, 14.7], 'LLOQ': [25, 10], 'ULOQ': [1000, 2500], 'a': [0.997431, 1.04095], 'b': [-2.19262, numpy.nan], 'r': [0.997931, 0.999556], 'r2': [0.995866, 0.999113],'another column': ['something', 'something else']})

		self.targetedExpectedNoExclusion = dict()
		self.targetedExpectedNoExclusion['sampleMetadata'] = pandas.DataFrame({'Sample File Name': ['UnitTest4_targeted_file_001', 'UnitTest4_targeted_file_002', 'UnitTest4_targeted_file_003','UnitTest4_targeted_file_004', 'UnitTest4_targeted_file_005'], 'TargetLynx Sample ID': [1, 2, 3, 4, 5], 'MassLynx Row ID': [1, 2, 3, 4, 5], 'Sample Name': ['Calibration 1', 'Study Sample 1', 'Blank 1', 'QC 1', 'Other 1'], 'Sample Type': ['Standard', 'Analyte', 'Blank', 'QC', 'Solvent'],  'Acqu Date': ['10-Sep-16', '10-Sep-16','12-Sep-16', '10-Sep-16','10-Sep-16'], 'Acqu Time': ['02:14:32', '03:23:02', '13:52:35','14:46:40', '15:05:26'],  'Vial': ['1:A,1', '1:A,2', '1:A,3', '1:A,4', '1:A,5'], 'Instrument': ['XEVO-TQS#UnitTest','XEVO-TQS#UnitTest','XEVO-TQS#UnitTest', 'XEVO-TQS#UnitTest','XEVO-TQS#UnitTest'],'Calibrant': [True, False, False, False, False],'Study Sample': [False, True, False, False, False], 'Blank': [False, False, True, False, False], 'QC': [False, False, False, True, False], 'Other': [False, False, False, False, True], 'Acquired Time': [datetime(2016, 9, 10, 2, 14, 32), datetime(2016, 9, 10, 3, 23, 2), datetime(2016, 9, 12, 13, 52, 35),datetime(2016, 9, 10, 14, 46, 40), datetime(2016, 9, 10, 15, 5, 26)], 'Run Order': [0, 1, 4, 2, 3], 'Batch': [1, 1, 1, 1, 1]})
		self.targetedExpectedNoExclusion['sampleMetadata']['Acquired Time'] = self.targetedExpected['sampleMetadata']['Acquired Time']
		self.targetedExpectedNoExclusion['featureMetadata'] = pandas.DataFrame({'Feature Name': ['Feature-1','Feature2'], 'TargetLynx Feature ID': [1,2], 'TargetLynx IS ID': ['1','1'], 'IS': [True,False],'calibrationEquation': ['','((area * responseFactor)-b)/a'], 'calibrationMethod': [CalibrationMethod.noIS, CalibrationMethod.backcalculatedIS], 'quantificationType': [QuantificationType.IS, QuantificationType.QuantOwnLabeledAnalogue],'unitCorrectionFactor': [10.,1.], 'Unit': ['noUnit','pg/uL'], 'Cpd Info': ['uM','fg/uL'], 'Noise (area)': [38.95, 14.7], 'LLOQ': [250., 10.], 'ULOQ': [10000., 2500.], 'a': [0.997431, 1.04095], 'b': [-2.19262, numpy.nan], 'r': [0.997931, 0.999556], 'r2': [0.995866, 0.999113],'another column': ['something', 'something else'], 'extID1': ['F1', 'F2'], 'extID2': ['ID1', 'ID2']})
		self.targetedExpectedNoExclusion['featureMetadata'].index = [0,1]
		self.targetedExpectedNoExclusion['intensityData'] = numpy.array([[10.,2.], [30.,4.], [50.,6.], [70.,8.], [90.,10.]])
		self.targetedExpectedNoExclusion['expectedConcentration'] = pandas.DataFrame(numpy.array([[310.,32.], [330.,34.],[350.,36.],[370.,38.],[390.,40.]]), columns=self.targetedExpectedNoExclusion['featureMetadata']['Feature Name'].values.tolist())
		self.targetedExpectedNoExclusion['sampleMetadataExcluded'] = []
		self.targetedExpectedNoExclusion['featureMetadataExcluded'] = []
		self.targetedExpectedNoExclusion['intensityDataExcluded'] = []
		self.targetedExpectedNoExclusion['expectedConcentrationExcluded'] = []
		self.targetedExpectedNoExclusion['excludedFlag'] = []
		self.targetedExpectedNoExclusion['peakResponse'] = pandas.DataFrame(numpy.array([[11.,12.], [13.,14.],[15.,16.],[17.,18.],[19.,20.]]), columns=self.targetedExpectedNoExclusion['featureMetadata']['Feature Name'].values.tolist())
		self.targetedExpectedNoExclusion['peakArea'] = pandas.DataFrame(numpy.array([[21.,22.], [23.,24.],[25.,26.],[27.,28.],[29.,30.]]), columns=self.targetedExpectedNoExclusion['featureMetadata']['Feature Name'].values.tolist())
		self.targetedExpectedNoExclusion['peakConcentrationDeviation'] = pandas.DataFrame(numpy.array([[41.,42.], [43.,44.], [45.,46.], [47.,48.],[49.,50.]]), columns=self.targetedExpectedNoExclusion['featureMetadata']['Feature Name'].values.tolist())
		self.targetedExpectedNoExclusion['peakIntegrationFlag'] = pandas.DataFrame({'1': ['bb', 'bb','MMX', 'dbX', 'bb'], '2': ['bb', 'bb', 'MM', 'bb','dd']})
		self.targetedExpectedNoExclusion['peakIntegrationFlag'].index = [1, 2, 3, 4, 5]
		self.targetedExpectedNoExclusion['peakIntegrationFlag'].columns = self.targetedExpectedNoExclusion['featureMetadata']['Feature Name'].values.tolist()
		self.targetedExpectedNoExclusion['peakRT'] = pandas.DataFrame(numpy.array([[51.,52.],[53.,54.],[55.,56.],[57.,58.],[59.,60.]]),columns=self.targetedExpectedNoExclusion['featureMetadata']['Feature Name'].values.tolist())


	@unittest.mock.patch('sys.stdout', new_callable=io.StringIO)
	def test_targeteddataset_matchdatasettocalibrationreport(self, mock_stdout):

		with self.subTest(msg='Checking match with exclusion of Feature2 and renaming of Feature1 to Feature-1'):
			# Init
			targeted = copy.deepcopy(self.targeted)
			expected = copy.deepcopy(self.targetedExpected)
			result = dict()
			# Generate
			with warnings.catch_warnings():
				warnings.simplefilter('ignore', UserWarning)
				result['sampleMetadata'], result['featureMetadata'], result['intensityData'], result['expectedConcentration'], result['sampleMetadataExcluded'], result['featureMetadataExcluded'], result['intensityDataExcluded'], result['expectedConcentrationExcluded'], result['excludedFlag'], result['peakResponse'], result['peakArea'], result['peakConcentrationDeviation'], result['peakIntegrationFlag'], result['peakRT'] = targeted._TargetedDataset__matchDatasetToCalibrationReport(sampleMetadata=self.targetedTargetLynxData['sampleMetadata'],featureMetadata=self.targetedTargetLynxData['featureMetadata'],intensityData=self.targetedTargetLynxData['intensityData'],expectedConcentration=self.targetedTargetLynxData['expectedConcentration'],peakResponse=self.targetedTargetLynxData['peakResponse'], peakArea=self.targetedTargetLynxData['peakArea'],peakConcentrationDeviation=self.targetedTargetLynxData['peakConcentrationDeviation'],	peakIntegrationFlag=self.targetedTargetLynxData['peakIntegrationFlag'],peakRT=self.targetedTargetLynxData['peakRT'], calibReport=self.targetedReport)
			# Test
			assert_frame_equal(expected['sampleMetadata'].reindex(sorted(expected['sampleMetadata']), axis=1),result['sampleMetadata'].reindex(sorted(result['sampleMetadata']), axis=1))
			assert_frame_equal(expected['featureMetadata'].reindex(sorted(expected['featureMetadata']), axis=1),result['featureMetadata'].reindex(sorted(result['featureMetadata']), axis=1))
			numpy.testing.assert_array_equal(expected['intensityData'], result['intensityData'])
			assert_frame_equal(expected['expectedConcentration'].reindex(sorted(expected['expectedConcentration']),axis=1), result['expectedConcentration'].reindex(sorted(result['expectedConcentration']),axis=1))
			assert_frame_equal(expected['sampleMetadataExcluded'][0].reindex(sorted(expected['sampleMetadataExcluded'][0]), axis=1),result['sampleMetadataExcluded'][0].reindex(sorted(result['sampleMetadataExcluded'][0]), axis=1))
			assert_frame_equal(expected['featureMetadataExcluded'][0].reindex(sorted(expected['featureMetadataExcluded'][0]), axis=1),result['featureMetadataExcluded'][0].reindex(sorted(result['featureMetadataExcluded'][0]), axis=1))
			numpy.testing.assert_array_equal(expected['intensityDataExcluded'][0], result['intensityDataExcluded'][0])
			assert_frame_equal(expected['expectedConcentrationExcluded'][0].reindex(sorted(expected['expectedConcentrationExcluded'][0]), axis=1),result['expectedConcentrationExcluded'][0].reindex(sorted(result['expectedConcentrationExcluded'][0]), axis=1))
			self.assertEqual(expected['excludedFlag'], result['excludedFlag'])
			assert_frame_equal(expected['peakResponse'].reindex(sorted(expected['peakResponse']),axis=1), result['peakResponse'].reindex(sorted(result['peakResponse']),axis=1))
			assert_frame_equal(expected['peakArea'].reindex(sorted(expected['peakArea']),axis=1), result['peakArea'].reindex(sorted(result['peakArea']),axis=1))
			assert_frame_equal(expected['peakConcentrationDeviation'].reindex(sorted(expected['peakConcentrationDeviation']),axis=1), result['peakConcentrationDeviation'].reindex(sorted(result['peakConcentrationDeviation']),axis=1))
			assert_frame_equal(expected['peakIntegrationFlag'].reindex(sorted(expected['peakIntegrationFlag']),axis=1), result['peakIntegrationFlag'].reindex(sorted(result['peakIntegrationFlag']),axis=1))
			assert_frame_equal(expected['peakRT'].reindex(sorted(expected['peakRT']),axis=1), result['peakRT'].reindex(sorted(result['peakRT']),axis=1))

		with self.subTest(msg='Checking match without feature exclusion, with renaming of Feature1 to Feature-1'):
			# Init
			targetedNoExclusion = copy.deepcopy(self.targetedNoExclusion)
			expected = copy.deepcopy(self.targetedExpectedNoExclusion)
			result = dict()
			# Generate
			with warnings.catch_warnings():
				warnings.simplefilter('ignore', UserWarning)
				result['sampleMetadata'], result['featureMetadata'], result['intensityData'], result['expectedConcentration'], result['sampleMetadataExcluded'], result['featureMetadataExcluded'], result['intensityDataExcluded'], result['expectedConcentrationExcluded'], result['excludedFlag'], result['peakResponse'], result['peakArea'], result['peakConcentrationDeviation'], result['peakIntegrationFlag'], result['peakRT'] = targetedNoExclusion._TargetedDataset__matchDatasetToCalibrationReport(sampleMetadata=self.targetedTargetLynxData['sampleMetadata'],featureMetadata=self.targetedTargetLynxData['featureMetadata'],intensityData=self.targetedTargetLynxData['intensityData'],expectedConcentration=self.targetedTargetLynxData['expectedConcentration'],peakResponse=self.targetedTargetLynxData['peakResponse'],peakArea=self.targetedTargetLynxData['peakArea'],peakConcentrationDeviation=self.targetedTargetLynxData['peakConcentrationDeviation'],peakIntegrationFlag=self.targetedTargetLynxData['peakIntegrationFlag'],peakRT=self.targetedTargetLynxData['peakRT'], calibReport=self.targetedNoExclusionReport)
			# Test
			assert_frame_equal(expected['sampleMetadata'].reindex(sorted(expected['sampleMetadata']), axis=1),result['sampleMetadata'].reindex(sorted(result['sampleMetadata']), axis=1))
			assert_frame_equal(expected['featureMetadata'].reindex(sorted(expected['featureMetadata']), axis=1),result['featureMetadata'].reindex(sorted(result['featureMetadata']), axis=1))
			numpy.testing.assert_array_equal(expected['intensityData'], result['intensityData'])
			assert_frame_equal(expected['expectedConcentration'].reindex(sorted(expected['expectedConcentration']),axis=1), result['expectedConcentration'].reindex(sorted(result['expectedConcentration']),axis=1))
			self.assertListEqual(expected['sampleMetadataExcluded'], result['sampleMetadataExcluded'])
			self.assertListEqual(expected['featureMetadataExcluded'], result['featureMetadataExcluded'])
			self.assertListEqual(expected['intensityDataExcluded'], result['intensityDataExcluded'])
			self.assertListEqual(expected['expectedConcentrationExcluded'], result['expectedConcentrationExcluded'])
			self.assertEqual(expected['excludedFlag'], result['excludedFlag'])
			assert_frame_equal(expected['peakResponse'].reindex(sorted(expected['peakResponse']),axis=1), result['peakResponse'].reindex(sorted(result['peakResponse']),axis=1))
			assert_frame_equal(expected['peakArea'].reindex(sorted(expected['peakArea']),axis=1), result['peakArea'].reindex(sorted(result['peakArea']),axis=1))
			assert_frame_equal(expected['peakConcentrationDeviation'].reindex(sorted(expected['peakConcentrationDeviation']),axis=1), result['peakConcentrationDeviation'].reindex(sorted(result['peakConcentrationDeviation']),axis=1))
			assert_frame_equal(expected['peakIntegrationFlag'].reindex(sorted(expected['peakIntegrationFlag']),axis=1), result['peakIntegrationFlag'].reindex(sorted(result['peakIntegrationFlag']),axis=1))
			assert_frame_equal(expected['peakRT'].reindex(sorted(expected['peakRT']),axis=1), result['peakRT'].reindex(sorted(result['peakRT']),axis=1))


	@unittest.mock.patch('sys.stdout', new_callable=io.StringIO)
	def test_targeteddataset_matchdatasettocalibrationreport_raise(self, mock_stdout):

		with self.subTest(msg='Checking ValueError if input size do not match'):
			# sampleMetadata shape
			self.assertRaises(ValueError, self.targetedEmpty._TargetedDataset__matchDatasetToCalibrationReport, sampleMetadata=self.shapeError['badSampleMetadata'], featureMetadata=self.shapeError['goodFeatureMetadata'], intensityData=self.shapeError['goodIntensityData'], expectedConcentration=self.shapeError['goodExpectedConcentration'], peakResponse=self.shapeError['goodPeakResponse'], peakArea=self.shapeError['goodPeakArea'], peakConcentrationDeviation=self.shapeError['goodPeakConcentrationDeviation'], peakIntegrationFlag=self.shapeError['goodPeakIntegrationFlag'], peakRT=self.shapeError['goodPeakRT'], calibReport=pandas.DataFrame(numpy.random.random((3, 3))))
			# featureMetadata shape
			self.assertRaises(ValueError, self.targetedEmpty._TargetedDataset__matchDatasetToCalibrationReport, sampleMetadata=self.shapeError['goodSampleMetadata'], featureMetadata=self.shapeError['badFeatureMetadata'], intensityData=self.shapeError['goodIntensityData'], expectedConcentration=self.shapeError['goodExpectedConcentration'], peakResponse=self.shapeError['goodPeakResponse'], peakArea=self.shapeError['goodPeakArea'], peakConcentrationDeviation=self.shapeError['goodPeakConcentrationDeviation'], peakIntegrationFlag=self.shapeError['goodPeakIntegrationFlag'], peakRT=self.shapeError['goodPeakRT'], calibReport=pandas.DataFrame(numpy.random.random((3, 3))))
			# expectedConcentration
			self.assertRaises(ValueError, self.targetedEmpty._TargetedDataset__matchDatasetToCalibrationReport, sampleMetadata=self.shapeError['goodSampleMetadata'], featureMetadata=self.shapeError['goodFeatureMetadata'], intensityData=self.shapeError['goodIntensityData'], expectedConcentration=self.shapeError['badExpectedConcentration'], peakResponse=self.shapeError['goodPeakResponse'], peakArea=self.shapeError['goodPeakArea'], peakConcentrationDeviation=self.shapeError['goodPeakConcentrationDeviation'], peakIntegrationFlag=self.shapeError['goodPeakIntegrationFlag'], peakRT=self.shapeError['goodPeakRT'], calibReport=pandas.DataFrame(numpy.random.random((3, 3))))
			# peakResponse shape
			self.assertRaises(ValueError, self.targetedEmpty._TargetedDataset__matchDatasetToCalibrationReport, sampleMetadata=self.shapeError['goodSampleMetadata'], featureMetadata=self.shapeError['goodFeatureMetadata'], intensityData=self.shapeError['goodIntensityData'], expectedConcentration=self.shapeError['goodExpectedConcentration'], peakResponse=self.shapeError['badPeakResponse'], peakArea=self.shapeError['goodPeakArea'], peakConcentrationDeviation=self.shapeError['goodPeakConcentrationDeviation'], peakIntegrationFlag=self.shapeError['goodPeakIntegrationFlag'], peakRT=self.shapeError['goodPeakRT'], calibReport=pandas.DataFrame(numpy.random.random((3, 3))))
			# peakArea shape
			self.assertRaises(ValueError, self.targetedEmpty._TargetedDataset__matchDatasetToCalibrationReport, sampleMetadata=self.shapeError['goodSampleMetadata'], featureMetadata=self.shapeError['goodFeatureMetadata'], intensityData=self.shapeError['goodIntensityData'], expectedConcentration=self.shapeError['goodExpectedConcentration'], peakResponse=self.shapeError['goodPeakResponse'], peakArea=self.shapeError['badPeakArea'], peakConcentrationDeviation=self.shapeError['goodPeakConcentrationDeviation'], peakIntegrationFlag=self.shapeError['goodPeakIntegrationFlag'], peakRT=self.shapeError['goodPeakRT'], calibReport=pandas.DataFrame(numpy.random.random((3, 3))))
			# peakConcentrationDeviation shape
			self.assertRaises(ValueError, self.targetedEmpty._TargetedDataset__matchDatasetToCalibrationReport, sampleMetadata=self.shapeError['goodSampleMetadata'], featureMetadata=self.shapeError['goodFeatureMetadata'], intensityData=self.shapeError['goodIntensityData'], expectedConcentration=self.shapeError['goodExpectedConcentration'], peakResponse=self.shapeError['goodPeakResponse'], peakArea=self.shapeError['goodPeakArea'], peakConcentrationDeviation=self.shapeError['badPeakConcentrationDeviation'], peakIntegrationFlag=self.shapeError['goodPeakIntegrationFlag'], peakRT=self.shapeError['goodPeakRT'], calibReport=pandas.DataFrame(numpy.random.random((3, 3))))
			# peakIntegrationFlag shape
			self.assertRaises(ValueError, self.targetedEmpty._TargetedDataset__matchDatasetToCalibrationReport, sampleMetadata=self.shapeError['goodSampleMetadata'], featureMetadata=self.shapeError['goodFeatureMetadata'], intensityData=self.shapeError['goodIntensityData'], expectedConcentration=self.shapeError['goodExpectedConcentration'], peakResponse=self.shapeError['goodPeakResponse'], peakArea=self.shapeError['goodPeakArea'], peakConcentrationDeviation=self.shapeError['goodPeakConcentrationDeviation'], peakIntegrationFlag=self.shapeError['badPeakIntegrationFlag'], peakRT=self.shapeError['goodPeakRT'], calibReport=pandas.DataFrame(numpy.random.random((3, 3))))
			# peakRT shape
			self.assertRaises(ValueError, self.targetedEmpty._TargetedDataset__matchDatasetToCalibrationReport, sampleMetadata=self.shapeError['goodSampleMetadata'], featureMetadata=self.shapeError['goodFeatureMetadata'], intensityData=self.shapeError['goodIntensityData'], expectedConcentration=self.shapeError['goodExpectedConcentration'], peakResponse=self.shapeError['goodPeakResponse'], peakArea=self.shapeError['goodPeakArea'], peakConcentrationDeviation=self.shapeError['goodPeakConcentrationDeviation'], peakIntegrationFlag=self.shapeError['goodPeakIntegrationFlag'], peakRT=self.shapeError['badPeakRT'], calibReport=pandas.DataFrame(numpy.random.random((3, 3))))

		with self.subTest(msg='Checking ValueError if errors in SOP JSON'):
			# IS mismatch
			self.assertRaises(ValueError, self.targetedFailSOPIS._TargetedDataset__matchDatasetToCalibrationReport, sampleMetadata=self.shapeError['goodSampleMetadata'], featureMetadata=self.shapeError['goodFeatureMetadata'], intensityData=self.shapeError['goodIntensityData'], expectedConcentration=self.shapeError['goodExpectedConcentration'], peakResponse=self.shapeError['goodPeakResponse'], peakArea=self.shapeError['goodPeakArea'], peakConcentrationDeviation=self.shapeError['goodPeakConcentrationDeviation'], peakIntegrationFlag=self.shapeError['goodPeakIntegrationFlag'], peakRT=self.shapeError['goodPeakRT'], calibReport=pandas.DataFrame(numpy.random.random((3, 3))))
			# quantificationType mismatch
			self.assertRaises(ValueError, self.targetedFailSOPMonitored._TargetedDataset__matchDatasetToCalibrationReport, sampleMetadata=self.shapeError['goodSampleMetadata'], featureMetadata=self.shapeError['goodFeatureMetadata'], intensityData=self.shapeError['goodIntensityData'], expectedConcentration=self.shapeError['goodExpectedConcentration'], peakResponse=self.shapeError['goodPeakResponse'], peakArea=self.shapeError['goodPeakArea'], peakConcentrationDeviation=self.shapeError['goodPeakConcentrationDeviation'], peakIntegrationFlag=self.shapeError['goodPeakIntegrationFlag'], peakRT=self.shapeError['goodPeakRT'], calibReport=pandas.DataFrame(numpy.random.random((3, 3))))

		with self.subTest(msg='Checking ValueError if SOP and calibReport mismatch'):
			# 4 features in SOP, 3 in calibReport
			self.assertRaises(ValueError, self.targetedFailSOPCalibSize._TargetedDataset__matchDatasetToCalibrationReport, sampleMetadata=self.shapeError['goodSampleMetadata'], featureMetadata=self.shapeError['goodFeatureMetadata'], intensityData=self.shapeError['goodIntensityData'], expectedConcentration=self.shapeError['goodExpectedConcentration'], peakResponse=self.shapeError['goodPeakResponse'], peakArea=self.shapeError['goodPeakArea'], peakConcentrationDeviation=self.shapeError['goodPeakConcentrationDeviation'], peakIntegrationFlag=self.shapeError['goodPeakIntegrationFlag'], peakRT=self.shapeError['goodPeakRT'], calibReport=pandas.DataFrame(numpy.random.random((3, 3))))
			# different TargetLynx ID
			self.assertRaises(ValueError, self.targetedFailSOPCalibID._TargetedDataset__matchDatasetToCalibrationReport, sampleMetadata=self.shapeError['goodSampleMetadata'], featureMetadata=self.shapeError['goodFeatureMetadata'], intensityData=self.shapeError['goodIntensityData'], expectedConcentration=self.shapeError['goodExpectedConcentration'], peakResponse=self.shapeError['goodPeakResponse'], peakArea=self.shapeError['goodPeakArea'], peakConcentrationDeviation=self.shapeError['goodPeakConcentrationDeviation'], peakIntegrationFlag=self.shapeError['goodPeakIntegrationFlag'], peakRT=self.shapeError['goodPeakRT'], calibReport=self.targetedFailSOPCalibIDReport)
			# different compound names
			self.assertRaises(ValueError, self.targetedFailSOPCalibName._TargetedDataset__matchDatasetToCalibrationReport, sampleMetadata=self.shapeError['goodSampleMetadata'], featureMetadata=self.shapeError['goodFeatureMetadata'], intensityData=self.shapeError['goodIntensityData'], expectedConcentration=self.shapeError['goodExpectedConcentration'], peakResponse=self.shapeError['goodPeakResponse'], peakArea=self.shapeError['goodPeakArea'], peakConcentrationDeviation=self.shapeError['goodPeakConcentrationDeviation'], peakIntegrationFlag=self.shapeError['goodPeakIntegrationFlag'], peakRT=self.shapeError['badPeakRT'], calibReport=self.targetedFailSOPCalibNameReport)

		with self.subTest(msg='Checking Warning if SOP/calibReport and TargetLynx mismatch'):
			# mismatched number of features
			with warnings.catch_warnings(record=True) as w:
				# Cause all warnings to always be triggered.
				warnings.simplefilter("always")
				# warning
				self.targetedWarningID._TargetedDataset__matchDatasetToCalibrationReport(sampleMetadata=self.targetedWarning['sampleMetadata'], featureMetadata=self.targetedWarning['featureMetadata'], intensityData=self.targetedWarning['intensityData'], expectedConcentration=self.targetedWarning['expectedConcentration'], peakResponse=self.targetedWarning['peakResponse'], peakArea=self.targetedWarning['peakArea'], peakConcentrationDeviation=self.targetedWarning['peakConcentrationDeviation'], peakIntegrationFlag=self.targetedWarning['peakIntegrationFlag'], peakRT=self.targetedWarning['peakRT'], calibReport=self.targetedWarningIDReport)
				#check
				assert len(w) == 1
				assert issubclass(w[-1].category, UserWarning)
				assert "features shared across" in str(w[-1].message)

			# mismatch feature names
			with warnings.catch_warnings(record=True) as w:
				# Cause all warnings to always be triggered.
				warnings.simplefilter("always")
				# warning
				self.targetedWarningName._TargetedDataset__matchDatasetToCalibrationReport(sampleMetadata=self.targetedWarning['sampleMetadata'], featureMetadata=self.targetedWarning['featureMetadata'], intensityData=self.targetedWarning['intensityData'], expectedConcentration=self.targetedWarning['expectedConcentration'], peakResponse=self.targetedWarning['peakResponse'], peakArea=self.targetedWarning['peakArea'], peakConcentrationDeviation=self.targetedWarning['peakConcentrationDeviation'], peakIntegrationFlag=self.targetedWarning['peakIntegrationFlag'], peakRT=self.targetedWarning['peakRT'], calibReport=self.targetedWarningNameReport)
				#check
				assert len(w) == 1
				assert issubclass(w[-1].category, UserWarning)
				assert "TargetLynx feature names & SOP/Calibration Report compounds names differ" in str(w[-1].message)


class test_targeteddataset_import_targetlynx_filtertargetlynx(unittest.TestCase):
	"""
	Test filtering of samples types and generation of :py:attr:'calibration', and filtering of IS features
	"""
	def setUp(self):
		## Basic input after _readTargetLynxDataset
		# 5 samples (1 calib, 1 Study Sample, 1 Blank, 1 QC, 1 Other)
		# 2 features, (1 IS, 1 normal feature) a fake exclusion
		self.targetedInFilterSample = nPYc.TargetedDataset('', fileType='empty')
		self.targetedInFilterSample.sampleMetadata = pandas.DataFrame({'Sample File Name': ['UnitTest_targeted_file_001','UnitTest_targeted_file_002', 'UnitTest_targeted_file_003','UnitTest_targeted_file_004','UnitTest_targeted_file_005'],
																'TargetLynx Sample ID': [1, 2, 3, 4, 5],
																'MassLynx Row ID': [1, 2, 3, 4, 5],
																'Sample Name': ['Calib', 'Sample', 'Blank', 'QC', 'Other'],
																'Sample Type': ['Standard', 'Analyte', 'Blank', 'QC', 'Solvent'],
																'Acqu Date': ['10-Sep-16', '10-Sep-16', '10-Sep-16','10-Sep-16', '10-Sep-16'],
																'Acqu Time': ['02:14:32', '03:23:02', '04:52:35', '05:46:40','06:05:26'],
																'Vial': ['1:A,1', '1:A,2', '1:A,3', '1:A,4', '1:A,5'],
																'Instrument': ['XEVO-TQS#UnitTest', 'XEVO-TQS#UnitTest','XEVO-TQS#UnitTest', 'XEVO-TQS#UnitTest','XEVO-TQS#UnitTest'],
																'Calibrant': [True, False, False, False, False],
																'Study Sample': [False, True, False, False, False],
																'Blank': [False, False, True, False, False],
																'QC': [False, False, False, True, False],
																'Other': [False, False, False, False, True],
																'Acquired Time': [datetime(2016, 9, 10, 2, 14, 32), datetime(2016, 9, 10, 3, 23, 2), datetime(2016, 9, 10, 4, 52, 35), datetime(2016, 9, 10, 5, 46, 40), datetime(2016, 9, 10, 6, 5, 26)],
																'Run Order': [0, 1, 2, 3, 4],
																'Batch': [1, 1, 1, 1, 1]})
		self.targetedInFilterSample.sampleMetadata['Acquired Time'] = self.targetedInFilterSample.sampleMetadata['Acquired Time']
		self.targetedInFilterSample.featureMetadata = pandas.DataFrame({'Feature Name': ['Feature1', 'Feature2'], 'TargetLynx Feature ID': [1, 2], 'TargetLynx IS ID': ['1', '1'], 'IS': [True, False], 'calibrationEquation': ['((area * responseFactor)-b)/a', '10**((numpy.log10(area * responseFactor)-b)/a)'], 'calibrationMethod': [CalibrationMethod.noIS, CalibrationMethod.backcalculatedIS], 'quantificationType': [QuantificationType.IS, QuantificationType.QuantAltLabeledAnalogue], 'unitCorrectionFactor': [1., 1.], 'Unit': ['pg/uL', 'pg/uL'], 'Cpd Info': ['info cpd1', 'info cpd2'], 'LLOQ': [100., 100.], 'ULOQ': [1000., 1000.], 'another column': ['something 1', 'something 2']})
		self.targetedInFilterSample._intensityData = numpy.array([[100., 250.], [250., 250.], [250., 100.], [500., 500.], [500., 500.]])
		self.targetedInFilterSample.expectedConcentration = pandas.DataFrame( numpy.array([[1., 2.], [3., 4.], [5., 6.], [7., 8.], [9., 10.]]), columns=self.targetedInFilterSample.featureMetadata['Feature Name'].values.tolist())
		self.targetedInFilterSample.sampleMetadataExcluded = [pandas.DataFrame(numpy.random.random((5, 17)))]
		self.targetedInFilterSample.featureMetadataExcluded = [pandas.DataFrame(numpy.random.random((2, 13)))]
		self.targetedInFilterSample.intensityDataExcluded = [numpy.random.random((5, 2))]
		self.targetedInFilterSample.expectedConcentrationExcluded = [pandas.DataFrame(numpy.random.random((5, 2)))]
		self.targetedInFilterSample.excludedFlag = ['Samples']
		inPeakResponse = pandas.DataFrame( numpy.array([[2100., 2100.], [2100., 2100.], [2100., 2100.], [2100., 2100.], [2100., 2100.]]), columns=self.targetedInFilterSample.featureMetadata['Feature Name'].values.tolist())
		inPeakArea = pandas.DataFrame(numpy.array([[200., 500.], [200., 1000.], [200., 1500.], [200., 100.], [200., 1000.]]), columns=self.targetedInFilterSample.featureMetadata['Feature Name'].values.tolist())
		inPeakConcentrationDeviation = pandas.DataFrame(numpy.array([[0., 0., ], [0., 0.], [0., 0.], [0., 0.], [0., 0., ]]), columns=self.targetedInFilterSample.featureMetadata[ 'Feature Name'].values.tolist())
		inPeakIntegrationFlag = pandas.DataFrame({'1': ['MM', 'MM', 'MM', 'MM', 'MM'], '2': ['bb', 'MM', 'bb', 'bb', 'bb']})
		inPeakIntegrationFlag.index = [0,1,2,3,4]
		inPeakIntegrationFlag.columns = self.targetedInFilterSample.featureMetadata['Feature Name'].values.tolist()
		inPeakRT = pandas.DataFrame(numpy.array([[10., 11.], [12., 13.], [14., 15.], [16., 17.], [18., 19.]]),columns=self.targetedInFilterSample.featureMetadata['Feature Name'].values.tolist())
		self.targetedInFilterSample.peakInfo = {'peakResponse': inPeakResponse, 'peakArea': inPeakArea, 'peakConcentrationDeviation': inPeakConcentrationDeviation, 'peakIntegrationFlag': inPeakIntegrationFlag, 'peakRT': inPeakRT}

		## Basic filterSample output
		# sampleTypeToProcess=['Study Sample','QC','Blank','Other'], with everything also excluded (filter when using)
		self.targetedOutFilterSample = copy.deepcopy(self.targetedInFilterSample)
		# remove calibration sample
		toKeep  = [False, True, True, True, True]
		notKeep = [True, False, False, False, False]
		# calibration
		calibPeakInfo = {'peakResponse': self.targetedOutFilterSample.peakInfo['peakResponse'].loc[notKeep, :], 'peakArea': self.targetedOutFilterSample.peakInfo['peakArea'].loc[notKeep, :], 'peakConcentrationDeviation': self.targetedOutFilterSample.peakInfo['peakConcentrationDeviation'].loc[notKeep, :], 'peakIntegrationFlag': self.targetedOutFilterSample.peakInfo['peakIntegrationFlag'].loc[notKeep, :], 'peakRT': self.targetedOutFilterSample.peakInfo['peakRT'].loc[notKeep, :]}
		self.targetedOutFilterSample.calibration = {'calibSampleMetadata': self.targetedOutFilterSample.sampleMetadata.loc[notKeep, :],'calibFeatureMetadata': self.targetedOutFilterSample.featureMetadata,'calibIntensityData': self.targetedOutFilterSample.intensityData[notKeep, :],'calibExpectedConcentration': self.targetedOutFilterSample.expectedConcentration.loc[notKeep, :],'calibPeakInfo': calibPeakInfo}
		# remove calibration
		self.targetedOutFilterSample.sampleMetadata = self.targetedOutFilterSample.sampleMetadata.loc[toKeep, :]
		self.targetedOutFilterSample._intensityData = self.targetedOutFilterSample._intensityData[toKeep, :]
		self.targetedOutFilterSample.expectedConcentration = self.targetedOutFilterSample.expectedConcentration.loc[toKeep, :]
		self.targetedOutFilterSample.expectedConcentration.reset_index(drop=True, inplace=True)
		tmpOutPeakInfo = {'peakResponse': self.targetedOutFilterSample.peakInfo['peakResponse'].loc[toKeep, :], 'peakArea': self.targetedOutFilterSample.peakInfo['peakArea'].loc[toKeep, :], 'peakConcentrationDeviation': self.targetedOutFilterSample.peakInfo['peakConcentrationDeviation'].loc[toKeep, :], 'peakIntegrationFlag': self.targetedOutFilterSample.peakInfo['peakIntegrationFlag'].loc[toKeep, :], 'peakRT': self.targetedOutFilterSample.peakInfo['peakRT'].loc[toKeep, :]}
		self.targetedOutFilterSample.peakInfo = tmpOutPeakInfo
		# exclusions
		self.targetedOutFilterSample.sampleMetadataExcluded.append(copy.deepcopy(self.targetedOutFilterSample.sampleMetadata))
		self.targetedOutFilterSample.featureMetadataExcluded.append(self.targetedOutFilterSample.featureMetadata)
		self.targetedOutFilterSample.intensityDataExcluded.append(self.targetedOutFilterSample.intensityData)
		self.targetedOutFilterSample.expectedConcentrationExcluded.append(self.targetedOutFilterSample.expectedConcentration)
		self.targetedOutFilterSample.sampleMetadata.reset_index(drop=True, inplace=True)
		self.targetedOutFilterSample.sampleMetadata.drop(['Calibrant', 'Study Sample', 'Blank', 'QC', 'Other'], axis=1, inplace=True)

		## Filter IS output
		# `_filterTargetLynxSamples(sampleTypeToProcess=['Study Sample', 'QC', 'Blank', 'Other'])` with IS filtered out
		self.targetedOutFilterIS = copy.deepcopy(self.targetedOutFilterSample)
		# remove previous sample exclusion (no sample filter exclusion)
		tmpExcludedImportSampleMetadata = [self.targetedOutFilterIS.sampleMetadataExcluded[0]]
		tmpExcludedImportFeatureMetadata = [self.targetedOutFilterIS.featureMetadataExcluded[0]]
		tmpExcludedImportIntensityData = [self.targetedOutFilterIS.intensityDataExcluded[0]]
		tmpExcludedImportExpectedConcentration = [self.targetedOutFilterIS.expectedConcentrationExcluded[0]]
		# add excluded IS
		tmpExcludedImportSampleMetadata.append(self.targetedOutFilterIS.sampleMetadata)
		tmpExcludedImportFeatureMetadata.append(self.targetedOutFilterIS.featureMetadata.loc[[True, False], :])
		tmpExcludedImportIntensityData.append(self.targetedOutFilterIS._intensityData[:, [True, False]])
		tmpExcludedImportExpectedConcentration.append(self.targetedOutFilterIS.expectedConcentration.loc[:, [True, False]])
		self.targetedOutFilterIS.sampleMetadataExcluded = tmpExcludedImportSampleMetadata
		self.targetedOutFilterIS.featureMetadataExcluded = tmpExcludedImportFeatureMetadata
		self.targetedOutFilterIS.intensityDataExcluded = tmpExcludedImportIntensityData
		self.targetedOutFilterIS.expectedConcentrationExcluded = tmpExcludedImportExpectedConcentration
		# remove IS feature
		self.targetedOutFilterIS.peakInfo['peakResponse'] = self.targetedOutFilterIS.peakInfo['peakResponse'].loc[:, [False,True]]
		self.targetedOutFilterIS.peakInfo['peakArea'] = self.targetedOutFilterIS.peakInfo['peakArea'].loc[:, [False,True]]
		self.targetedOutFilterIS.peakInfo['peakConcentrationDeviation'] = self.targetedOutFilterIS.peakInfo['peakConcentrationDeviation'].loc[:, [False,True]]
		self.targetedOutFilterIS.peakInfo['peakIntegrationFlag'] = self.targetedOutFilterIS.peakInfo['peakIntegrationFlag'].loc[:, [False,True]]
		self.targetedOutFilterIS.peakInfo['peakRT'] = self.targetedOutFilterIS.peakInfo['peakRT'].loc[:, [False,True]]
		self.targetedOutFilterIS.calibration['calibFeatureMetadata'] = self.targetedOutFilterIS.calibration['calibFeatureMetadata'].loc[[False,True],:]
		self.targetedOutFilterIS.calibration['calibFeatureMetadata'].reset_index(drop=True, inplace=True)
		self.targetedOutFilterIS.calibration['calibIntensityData'] = self.targetedOutFilterIS.calibration['calibIntensityData'][:,[False, True]]
		self.targetedOutFilterIS.calibration['calibExpectedConcentration'] = self.targetedOutFilterIS.calibration['calibExpectedConcentration'].loc[:,[False, True]]
		self.targetedOutFilterIS.calibration['calibPeakInfo']['peakResponse'] = self.targetedOutFilterIS.calibration['calibPeakInfo']['peakResponse'].loc[:,[False,True]]
		self.targetedOutFilterIS.calibration['calibPeakInfo']['peakArea'] = self.targetedOutFilterIS.calibration['calibPeakInfo']['peakArea'].loc[:,[False,True]]
		self.targetedOutFilterIS.calibration['calibPeakInfo']['peakConcentrationDeviation'] = self.targetedOutFilterIS.calibration['calibPeakInfo']['peakConcentrationDeviation'].loc[:,[False,True]]
		self.targetedOutFilterIS.calibration['calibPeakInfo']['peakIntegrationFlag'] = self.targetedOutFilterIS.calibration['calibPeakInfo']['peakIntegrationFlag'].loc[:,[False,True]]
		self.targetedOutFilterIS.calibration['calibPeakInfo']['peakRT'] = self.targetedOutFilterIS.calibration['calibPeakInfo']['peakRT'].loc[:,[False,True]]
		tmpFeatureMetadata = copy.deepcopy(self.targetedOutFilterIS.featureMetadata.loc[[False,True], :])
		tmpFeatureMetadata.reset_index(drop=True, inplace=True)
		tmpFeatureMetadata = tmpFeatureMetadata.drop(['IS', 'TargetLynx IS ID'], axis=1)
		self.targetedOutFilterIS.featureMetadata = tmpFeatureMetadata
		self.targetedOutFilterIS._intensityData = self.targetedOutFilterIS._intensityData[:, [False,True]]
		self.targetedOutFilterIS.expectedConcentration = self.targetedOutFilterIS.expectedConcentration.loc[:, [False,True]]


	@unittest.mock.patch('sys.stdout', new_callable=io.StringIO)
	def test_targeteddataset_filtertargetlynxsamples(self, mock_stdout):

		with self.subTest(msg='Checking filterTargetLynxSamples, sampleTypeToProcess = [\'Study Sample\',\'QC\',\'Blank\',\'Other\']'):
			# Expected
			expected = copy.deepcopy(self.targetedOutFilterSample)
			# No added exclusions (every samples kept)
			expected.sampleMetadataExcluded = [expected.sampleMetadataExcluded[0]]
			expected.featureMetadataExcluded = [expected.featureMetadataExcluded[0]]
			expected.intensityDataExcluded = [expected.intensityDataExcluded[0]]
			expected.expectedConcentrationExcluded = [expected.expectedConcentrationExcluded[0]]
			expected.excludedFlag = ['Samples']
			# Result
			result = copy.deepcopy(self.targetedInFilterSample)
			result._filterTargetLynxSamples(sampleTypeToProcess=['Study Sample', 'QC', 'Blank', 'Other'])

			# Class
			self.assertEqual(type(result), type(expected))
			# sampleMetadata
			assert_frame_equal(result.sampleMetadata, expected.sampleMetadata)
			# featureMetadata
			assert_frame_equal(result.featureMetadata.reindex(sorted(result.featureMetadata), axis=1),	expected.featureMetadata.reindex(sorted(expected.featureMetadata), axis=1))
			# intensityData
			numpy.testing.assert_array_equal(result._intensityData, expected._intensityData)
			# expectedConcentration
			assert_frame_equal(result.expectedConcentration.reindex(sorted(result.expectedConcentration), axis=1), expected.expectedConcentration.reindex(sorted(expected.expectedConcentration), axis=1))
			# Calibration
			assert_frame_equal(result.calibration['calibSampleMetadata'], expected.calibration['calibSampleMetadata'])
			assert_frame_equal(result.calibration['calibFeatureMetadata'], expected.calibration['calibFeatureMetadata'])
			numpy.testing.assert_array_equal(result.calibration['calibIntensityData'], expected.calibration['calibIntensityData'])
			assert_frame_equal(result.calibration['calibExpectedConcentration'], expected.calibration['calibExpectedConcentration'])
			assert_frame_equal(result.calibration['calibPeakInfo']['peakResponse'], expected.calibration['calibPeakInfo']['peakResponse'])
			assert_frame_equal(result.calibration['calibPeakInfo']['peakArea'], expected.calibration['calibPeakInfo']['peakArea'])
			assert_frame_equal(result.calibration['calibPeakInfo']['peakConcentrationDeviation'], expected.calibration['calibPeakInfo']['peakConcentrationDeviation'])
			assert_frame_equal(result.calibration['calibPeakInfo']['peakIntegrationFlag'], expected.calibration['calibPeakInfo']['peakIntegrationFlag'])
			assert_frame_equal(result.calibration['calibPeakInfo']['peakRT'], expected.calibration['calibPeakInfo']['peakRT'])
			# Import exclusion
			self.assertEqual(len(result.sampleMetadataExcluded), len(expected.sampleMetadataExcluded))
			for i in range(len(result.sampleMetadataExcluded)):
				assert_frame_equal(result.sampleMetadataExcluded[i], expected.sampleMetadataExcluded[i])
			self.assertEqual(len(result.featureMetadataExcluded), len(expected.featureMetadataExcluded))
			for j in range(len(result.featureMetadataExcluded)):
				assert_frame_equal(result.featureMetadataExcluded[j], expected.featureMetadataExcluded[j])
			self.assertEqual(len(result.intensityDataExcluded), len(expected.intensityDataExcluded))
			for k in range(len(result.intensityDataExcluded)):
				numpy.testing.assert_array_equal(result.intensityDataExcluded[k], expected.intensityDataExcluded[k])
			self.assertEqual(len(result.expectedConcentrationExcluded), len(expected.expectedConcentrationExcluded))
			for l in range(len(result.expectedConcentrationExcluded)):
				assert_frame_equal(result.expectedConcentrationExcluded[l], expected.expectedConcentrationExcluded[l])
			self.assertEqual(result.excludedFlag, expected.excludedFlag)
			# peakInfo
			assert_frame_equal(result.peakInfo['peakResponse'], expected.peakInfo['peakResponse'])
			assert_frame_equal(result.peakInfo['peakArea'], expected.peakInfo['peakArea'])
			assert_frame_equal(result.peakInfo['peakConcentrationDeviation'], expected.peakInfo['peakConcentrationDeviation'])
			assert_frame_equal(result.peakInfo['peakIntegrationFlag'], expected.peakInfo['peakIntegrationFlag'])
			assert_frame_equal(result.peakInfo['peakRT'], expected.peakInfo['peakRT'])

		with self.subTest(msg='Checking filterTargetLynxSamples, sampleTypeToProcess = [\'Study Sample\',\'QC\',\'Blank\']'):
			# Expected
			expected = copy.deepcopy(self.targetedOutFilterSample)
			expected.sampleMetadata = expected.sampleMetadata.loc[[True, True, True, False], :]
			expected._intensityData = expected._intensityData[[True, True, True, False], :]
			expected.expectedConcentration = expected.expectedConcentration.loc[[True, True, True, False], :]
			expected.sampleMetadataExcluded[1] = expected.sampleMetadataExcluded[1].loc[[False, False, False, True],:]
			expected.intensityDataExcluded[1] = expected.intensityDataExcluded[1][[False, False, False, True],:]
			expected.expectedConcentrationExcluded[1] = expected.expectedConcentrationExcluded[1].loc[[False, False, False, True],:]
			expected.expectedConcentrationExcluded[1].index = [4]
			expected.excludedFlag = ['Samples','Samples']
			expected.peakInfo['peakResponse'] = expected.peakInfo['peakResponse'].loc[[True, True, True, False], :]
			expected.peakInfo['peakArea'] = expected.peakInfo['peakArea'].loc[[True, True, True, False], :]
			expected.peakInfo['peakConcentrationDeviation'] = expected.peakInfo['peakConcentrationDeviation'].loc[[True, True, True, False], :]
			expected.peakInfo['peakIntegrationFlag'] = expected.peakInfo['peakIntegrationFlag'].loc[[True, True, True, False], :]
			expected.peakInfo['peakRT'] = expected.peakInfo['peakRT'].loc[[True, True, True, False], :]

			# Result
			result = copy.deepcopy(self.targetedInFilterSample)
			result._filterTargetLynxSamples(sampleTypeToProcess=['Study Sample', 'QC', 'Blank'])

			# Class
			self.assertEqual(type(result), type(expected))
			# sampleMetadata
			assert_frame_equal(result.sampleMetadata, expected.sampleMetadata)
			# featureMetadata
			assert_frame_equal(result.featureMetadata.reindex(sorted(result.featureMetadata), axis=1),	expected.featureMetadata.reindex(sorted(expected.featureMetadata), axis=1))
			# intensityData
			numpy.testing.assert_array_equal(result._intensityData, expected._intensityData)
			# expectedConcentration
			assert_frame_equal(result.expectedConcentration.reindex(sorted(result.expectedConcentration), axis=1), expected.expectedConcentration.reindex(sorted(expected.expectedConcentration), axis=1))
			# Calibration
			assert_frame_equal(result.calibration['calibSampleMetadata'], expected.calibration['calibSampleMetadata'])
			assert_frame_equal(result.calibration['calibFeatureMetadata'], expected.calibration['calibFeatureMetadata'])
			numpy.testing.assert_array_equal(result.calibration['calibIntensityData'], expected.calibration['calibIntensityData'])
			assert_frame_equal(result.calibration['calibExpectedConcentration'], expected.calibration['calibExpectedConcentration'])
			assert_frame_equal(result.calibration['calibPeakInfo']['peakResponse'], expected.calibration['calibPeakInfo']['peakResponse'])
			assert_frame_equal(result.calibration['calibPeakInfo']['peakArea'], expected.calibration['calibPeakInfo']['peakArea'])
			assert_frame_equal(result.calibration['calibPeakInfo']['peakConcentrationDeviation'], expected.calibration['calibPeakInfo']['peakConcentrationDeviation'])
			assert_frame_equal(result.calibration['calibPeakInfo']['peakIntegrationFlag'], expected.calibration['calibPeakInfo']['peakIntegrationFlag'])
			assert_frame_equal(result.calibration['calibPeakInfo']['peakRT'], expected.calibration['calibPeakInfo']['peakRT'])
			# Import exclusion
			self.assertEqual(len(result.sampleMetadataExcluded), len(expected.sampleMetadataExcluded))
			for i in range(len(result.sampleMetadataExcluded)):
				assert_frame_equal(result.sampleMetadataExcluded[i], expected.sampleMetadataExcluded[i])
			self.assertEqual(len(result.featureMetadataExcluded), len(expected.featureMetadataExcluded))
			for j in range(len(result.featureMetadataExcluded)):
				assert_frame_equal(result.featureMetadataExcluded[j], expected.featureMetadataExcluded[j])
			self.assertEqual(len(result.intensityDataExcluded), len(expected.intensityDataExcluded))
			for k in range(len(result.intensityDataExcluded)):
				numpy.testing.assert_array_equal(result.intensityDataExcluded[k], expected.intensityDataExcluded[k])
			self.assertEqual(len(result.expectedConcentrationExcluded), len(expected.expectedConcentrationExcluded))
			for l in range(len(result.expectedConcentrationExcluded)):
				assert_frame_equal(result.expectedConcentrationExcluded[l], expected.expectedConcentrationExcluded[l])
			self.assertEqual(result.excludedFlag, expected.excludedFlag)
			# peakInfo
			assert_frame_equal(result.peakInfo['peakResponse'], expected.peakInfo['peakResponse'])
			assert_frame_equal(result.peakInfo['peakArea'], expected.peakInfo['peakArea'])
			assert_frame_equal(result.peakInfo['peakConcentrationDeviation'], expected.peakInfo['peakConcentrationDeviation'])
			assert_frame_equal(result.peakInfo['peakIntegrationFlag'], expected.peakInfo['peakIntegrationFlag'])
			assert_frame_equal(result.peakInfo['peakRT'], expected.peakInfo['peakRT'])

		with self.subTest(msg='Checking filterTargetLynxSamples, sampleTypeToProcess = [\'Study Sample\',\'QC\',\'Other\']'):
			# Expected
			expected = copy.deepcopy(self.targetedOutFilterSample)
			expected.sampleMetadata = expected.sampleMetadata.loc[[True, False, True, True], :]
			expected.sampleMetadata.reset_index(drop=True, inplace=True)
			expected._intensityData = expected._intensityData[[True, False, True, True], :]
			expected.expectedConcentration = expected.expectedConcentration.loc[[True, False, True, True], :]
			expected.expectedConcentration.reset_index(drop=True, inplace=True)
			expected.sampleMetadataExcluded[1] = expected.sampleMetadataExcluded[1].loc[[False, True, False, False],:]
			expected.intensityDataExcluded[1] = expected.intensityDataExcluded[1][[False, True, False, False],:]
			expected.expectedConcentrationExcluded[1] = expected.expectedConcentrationExcluded[1].loc[[False, True, False, False],:]
			expected.expectedConcentrationExcluded[1].index = [2]
			expected.excludedFlag = ['Samples','Samples']
			expected.peakInfo['peakResponse'] = expected.peakInfo['peakResponse'].loc[[True, False, True, True], :]
			expected.peakInfo['peakArea'] = expected.peakInfo['peakArea'].loc[[True, False, True, True], :]
			expected.peakInfo['peakConcentrationDeviation'] = expected.peakInfo['peakConcentrationDeviation'].loc[[True, False, True, True], :]
			expected.peakInfo['peakIntegrationFlag'] = expected.peakInfo['peakIntegrationFlag'].loc[[True, False, True, True], :]
			expected.peakInfo['peakRT'] = expected.peakInfo['peakRT'].loc[[True, False, True, True], :]

			# Result
			result = copy.deepcopy(self.targetedInFilterSample)
			result._filterTargetLynxSamples(sampleTypeToProcess=['Study Sample', 'QC', 'Other'])

			# Class
			self.assertEqual(type(result), type(expected))
			# sampleMetadata
			assert_frame_equal(result.sampleMetadata, expected.sampleMetadata)
			# featureMetadata
			assert_frame_equal(result.featureMetadata.reindex(sorted(result.featureMetadata), axis=1),	expected.featureMetadata.reindex(sorted(expected.featureMetadata), axis=1))
			# intensityData
			numpy.testing.assert_array_equal(result._intensityData, expected._intensityData)
			# expectedConcentration
			assert_frame_equal(result.expectedConcentration.reindex(sorted(result.expectedConcentration), axis=1), expected.expectedConcentration.reindex(sorted(expected.expectedConcentration), axis=1))
			# Calibration
			assert_frame_equal(result.calibration['calibSampleMetadata'], expected.calibration['calibSampleMetadata'])
			assert_frame_equal(result.calibration['calibFeatureMetadata'], expected.calibration['calibFeatureMetadata'])
			numpy.testing.assert_array_equal(result.calibration['calibIntensityData'], expected.calibration['calibIntensityData'])
			assert_frame_equal(result.calibration['calibExpectedConcentration'], expected.calibration['calibExpectedConcentration'])
			assert_frame_equal(result.calibration['calibPeakInfo']['peakResponse'], expected.calibration['calibPeakInfo']['peakResponse'])
			assert_frame_equal(result.calibration['calibPeakInfo']['peakArea'], expected.calibration['calibPeakInfo']['peakArea'])
			assert_frame_equal(result.calibration['calibPeakInfo']['peakConcentrationDeviation'], expected.calibration['calibPeakInfo']['peakConcentrationDeviation'])
			assert_frame_equal(result.calibration['calibPeakInfo']['peakIntegrationFlag'], expected.calibration['calibPeakInfo']['peakIntegrationFlag'])
			assert_frame_equal(result.calibration['calibPeakInfo']['peakRT'], expected.calibration['calibPeakInfo']['peakRT'])
			# Import exclusion
			self.assertEqual(len(result.sampleMetadataExcluded), len(expected.sampleMetadataExcluded))
			for i in range(len(result.sampleMetadataExcluded)):
				assert_frame_equal(result.sampleMetadataExcluded[i], expected.sampleMetadataExcluded[i])
			self.assertEqual(len(result.featureMetadataExcluded), len(expected.featureMetadataExcluded))
			for j in range(len(result.featureMetadataExcluded)):
				assert_frame_equal(result.featureMetadataExcluded[j], expected.featureMetadataExcluded[j])
			self.assertEqual(len(result.intensityDataExcluded), len(expected.intensityDataExcluded))
			for k in range(len(result.intensityDataExcluded)):
				numpy.testing.assert_array_equal(result.intensityDataExcluded[k], expected.intensityDataExcluded[k])
			self.assertEqual(len(result.expectedConcentrationExcluded), len(expected.expectedConcentrationExcluded))
			for l in range(len(result.expectedConcentrationExcluded)):
				assert_frame_equal(result.expectedConcentrationExcluded[l], expected.expectedConcentrationExcluded[l])
			self.assertEqual(result.excludedFlag, expected.excludedFlag)
			# peakInfo
			assert_frame_equal(result.peakInfo['peakResponse'], expected.peakInfo['peakResponse'])
			assert_frame_equal(result.peakInfo['peakArea'], expected.peakInfo['peakArea'])
			assert_frame_equal(result.peakInfo['peakConcentrationDeviation'], expected.peakInfo['peakConcentrationDeviation'])
			assert_frame_equal(result.peakInfo['peakIntegrationFlag'], expected.peakInfo['peakIntegrationFlag'])
			assert_frame_equal(result.peakInfo['peakRT'], expected.peakInfo['peakRT'])

		with self.subTest(msg='Checking filterTargetLynxSamples, sampleTypeToProcess = [\'Study Sample\']'):
			# Expected
			expected = copy.deepcopy(self.targetedOutFilterSample)
			expected.sampleMetadata = expected.sampleMetadata.loc[[True, False, False, False], :]
			expected.sampleMetadata.reset_index(drop=True, inplace=True)
			expected._intensityData = expected._intensityData[[True, False, False, False], :]
			expected.expectedConcentration = expected.expectedConcentration.loc[[True, False, False, False], :]
			expected.sampleMetadataExcluded[1] = expected.sampleMetadataExcluded[1].loc[[False, True, True, True],:]
			expected.intensityDataExcluded[1] = expected.intensityDataExcluded[1][[False, True, True, True],:]
			expected.expectedConcentrationExcluded[1] = expected.expectedConcentrationExcluded[1].loc[[False, True, True, True],:]
			expected.expectedConcentrationExcluded[1].index = [2,3,4]
			expected.excludedFlag = ['Samples','Samples']
			expected.peakInfo['peakResponse'] = expected.peakInfo['peakResponse'].loc[[True, False, False, False], :]
			expected.peakInfo['peakArea'] = expected.peakInfo['peakArea'].loc[[True, False, False, False], :]
			expected.peakInfo['peakConcentrationDeviation'] = expected.peakInfo['peakConcentrationDeviation'].loc[[True, False, False, False], :]
			expected.peakInfo['peakIntegrationFlag'] = expected.peakInfo['peakIntegrationFlag'].loc[[True, False, False, False], :]
			expected.peakInfo['peakRT'] = expected.peakInfo['peakRT'].loc[[True, False, False, False], :]

			# Result
			result = copy.deepcopy(self.targetedInFilterSample)
			result._filterTargetLynxSamples(sampleTypeToProcess=['Study Sample'])

			# Class
			self.assertEqual(type(result), type(expected))
			# sampleMetadata
			assert_frame_equal(result.sampleMetadata, expected.sampleMetadata)
			# featureMetadata
			assert_frame_equal(result.featureMetadata.reindex(sorted(result.featureMetadata), axis=1),	expected.featureMetadata.reindex(sorted(expected.featureMetadata), axis=1))
			# intensityData
			numpy.testing.assert_array_equal(result._intensityData, expected._intensityData)
			# expectedConcentration
			assert_frame_equal(result.expectedConcentration.reindex(sorted(result.expectedConcentration), axis=1), expected.expectedConcentration.reindex(sorted(expected.expectedConcentration), axis=1))
			# Calibration
			assert_frame_equal(result.calibration['calibSampleMetadata'], expected.calibration['calibSampleMetadata'])
			assert_frame_equal(result.calibration['calibFeatureMetadata'], expected.calibration['calibFeatureMetadata'])
			numpy.testing.assert_array_equal(result.calibration['calibIntensityData'], expected.calibration['calibIntensityData'])
			assert_frame_equal(result.calibration['calibExpectedConcentration'], expected.calibration['calibExpectedConcentration'])
			assert_frame_equal(result.calibration['calibPeakInfo']['peakResponse'], expected.calibration['calibPeakInfo']['peakResponse'])
			assert_frame_equal(result.calibration['calibPeakInfo']['peakArea'], expected.calibration['calibPeakInfo']['peakArea'])
			assert_frame_equal(result.calibration['calibPeakInfo']['peakConcentrationDeviation'], expected.calibration['calibPeakInfo']['peakConcentrationDeviation'])
			assert_frame_equal(result.calibration['calibPeakInfo']['peakIntegrationFlag'], expected.calibration['calibPeakInfo']['peakIntegrationFlag'])
			assert_frame_equal(result.calibration['calibPeakInfo']['peakRT'], expected.calibration['calibPeakInfo']['peakRT'])
			# Import exclusion
			self.assertEqual(len(result.sampleMetadataExcluded), len(expected.sampleMetadataExcluded))
			for i in range(len(result.sampleMetadataExcluded)):
				assert_frame_equal(result.sampleMetadataExcluded[i], expected.sampleMetadataExcluded[i])
			self.assertEqual(len(result.featureMetadataExcluded), len(expected.featureMetadataExcluded))
			for j in range(len(result.featureMetadataExcluded)):
				assert_frame_equal(result.featureMetadataExcluded[j], expected.featureMetadataExcluded[j])
			self.assertEqual(len(result.intensityDataExcluded), len(expected.intensityDataExcluded))
			for k in range(len(result.intensityDataExcluded)):
				numpy.testing.assert_array_equal(result.intensityDataExcluded[k], expected.intensityDataExcluded[k])
			self.assertEqual(len(result.expectedConcentrationExcluded), len(expected.expectedConcentrationExcluded))
			for l in range(len(result.expectedConcentrationExcluded)):
				assert_frame_equal(result.expectedConcentrationExcluded[l], expected.expectedConcentrationExcluded[l])
			self.assertEqual(result.excludedFlag, expected.excludedFlag)
			# peakInfo
			assert_frame_equal(result.peakInfo['peakResponse'], expected.peakInfo['peakResponse'])
			assert_frame_equal(result.peakInfo['peakArea'], expected.peakInfo['peakArea'])
			assert_frame_equal(result.peakInfo['peakConcentrationDeviation'], expected.peakInfo['peakConcentrationDeviation'])
			assert_frame_equal(result.peakInfo['peakIntegrationFlag'], expected.peakInfo['peakIntegrationFlag'])
			assert_frame_equal(result.peakInfo['peakRT'], expected.peakInfo['peakRT'])

		with self.subTest(msg='Checking filterTargetLynxSamples, default parameter, sampleTypeToProcess = [\'Study Sample\',\'QC\']'):
			# Expected
			expected = copy.deepcopy(self.targetedOutFilterSample)
			expected.sampleMetadata = expected.sampleMetadata.loc[[True, False, True, False], :]
			expected.sampleMetadata.reset_index(drop=True, inplace=True)
			expected._intensityData = expected._intensityData[[True, False, True, False], :]
			expected.expectedConcentration = expected.expectedConcentration.loc[[True, False, True, False], :]
			expected.expectedConcentration.reset_index(drop=True, inplace=True)
			expected.sampleMetadataExcluded[1] = expected.sampleMetadataExcluded[1].loc[[False, True, False, True],:]
			expected.intensityDataExcluded[1] = expected.intensityDataExcluded[1][[False, True, False, True],:]
			expected.expectedConcentrationExcluded[1] = expected.expectedConcentrationExcluded[1].loc[[False, True, False, True],:]
			expected.expectedConcentrationExcluded[1].index = [2,4]
			expected.excludedFlag = ['Samples','Samples']
			expected.peakInfo['peakResponse'] = expected.peakInfo['peakResponse'].loc[[True, False, True, False], :]
			expected.peakInfo['peakArea'] = expected.peakInfo['peakArea'].loc[[True, False, True, False], :]
			expected.peakInfo['peakConcentrationDeviation'] = expected.peakInfo['peakConcentrationDeviation'].loc[[True, False, True, False], :]
			expected.peakInfo['peakIntegrationFlag'] = expected.peakInfo['peakIntegrationFlag'].loc[[True, False, True, False], :]
			expected.peakInfo['peakRT'] = expected.peakInfo['peakRT'].loc[[True, False, True, False], :]

			# Result
			result = copy.deepcopy(self.targetedInFilterSample)
			result._filterTargetLynxSamples(sampleTypeToProcess=['Study Sample', 'QC'])

			# Class
			self.assertEqual(type(result), type(expected))
			# sampleMetadata
			assert_frame_equal(result.sampleMetadata, expected.sampleMetadata)
			# featureMetadata
			assert_frame_equal(result.featureMetadata.reindex(sorted(result.featureMetadata), axis=1),	expected.featureMetadata.reindex(sorted(expected.featureMetadata), axis=1))
			# intensityData
			numpy.testing.assert_array_equal(result._intensityData, expected._intensityData)
			# expectedConcentration
			assert_frame_equal(result.expectedConcentration.reindex(sorted(result.expectedConcentration), axis=1), expected.expectedConcentration.reindex(sorted(expected.expectedConcentration), axis=1))
			# Calibration
			assert_frame_equal(result.calibration['calibSampleMetadata'], expected.calibration['calibSampleMetadata'])
			assert_frame_equal(result.calibration['calibFeatureMetadata'], expected.calibration['calibFeatureMetadata'])
			numpy.testing.assert_array_equal(result.calibration['calibIntensityData'], expected.calibration['calibIntensityData'])
			assert_frame_equal(result.calibration['calibExpectedConcentration'], expected.calibration['calibExpectedConcentration'])
			assert_frame_equal(result.calibration['calibPeakInfo']['peakResponse'], expected.calibration['calibPeakInfo']['peakResponse'])
			assert_frame_equal(result.calibration['calibPeakInfo']['peakArea'], expected.calibration['calibPeakInfo']['peakArea'])
			assert_frame_equal(result.calibration['calibPeakInfo']['peakConcentrationDeviation'], expected.calibration['calibPeakInfo']['peakConcentrationDeviation'])
			assert_frame_equal(result.calibration['calibPeakInfo']['peakIntegrationFlag'], expected.calibration['calibPeakInfo']['peakIntegrationFlag'])
			assert_frame_equal(result.calibration['calibPeakInfo']['peakRT'], expected.calibration['calibPeakInfo']['peakRT'])
			# Import exclusion
			self.assertEqual(len(result.sampleMetadataExcluded), len(expected.sampleMetadataExcluded))
			for i in range(len(result.sampleMetadataExcluded)):
				assert_frame_equal(result.sampleMetadataExcluded[i], expected.sampleMetadataExcluded[i])
			self.assertEqual(len(result.featureMetadataExcluded), len(expected.featureMetadataExcluded))
			for j in range(len(result.featureMetadataExcluded)):
				assert_frame_equal(result.featureMetadataExcluded[j], expected.featureMetadataExcluded[j])
			self.assertEqual(len(result.intensityDataExcluded), len(expected.intensityDataExcluded))
			for k in range(len(result.intensityDataExcluded)):
				numpy.testing.assert_array_equal(result.intensityDataExcluded[k], expected.intensityDataExcluded[k])
			self.assertEqual(len(result.expectedConcentrationExcluded), len(expected.expectedConcentrationExcluded))
			for l in range(len(result.expectedConcentrationExcluded)):
				assert_frame_equal(result.expectedConcentrationExcluded[l], expected.expectedConcentrationExcluded[l])
			self.assertEqual(result.excludedFlag, expected.excludedFlag)
			# peakInfo
			assert_frame_equal(result.peakInfo['peakResponse'], expected.peakInfo['peakResponse'])
			assert_frame_equal(result.peakInfo['peakArea'], expected.peakInfo['peakArea'])
			assert_frame_equal(result.peakInfo['peakConcentrationDeviation'], expected.peakInfo['peakConcentrationDeviation'])
			assert_frame_equal(result.peakInfo['peakIntegrationFlag'], expected.peakInfo['peakIntegrationFlag'])
			assert_frame_equal(result.peakInfo['peakRT'], expected.peakInfo['peakRT'])

		with self.subTest(msg='Checking filterTargetLynxSamples, default parameter no previous exclusions, sampleTypeToProcess = [\'Study Sample\',\'QC\']'):
			# Expected
			expected = copy.deepcopy(self.targetedOutFilterSample)
			expected.sampleMetadata = expected.sampleMetadata.loc[[True, False, True, False], :]
			expected.sampleMetadata.reset_index(drop=True, inplace=True)
			expected._intensityData = expected._intensityData[[True, False, True, False], :]
			expected.expectedConcentration = expected.expectedConcentration.loc[[True, False, True, False], :]
			expected.expectedConcentration.reset_index(drop=True, inplace=True)
			expected.sampleMetadataExcluded = [expected.sampleMetadataExcluded[1].loc[[False, True, False, True],:]]
			expected.intensityDataExcluded = [expected.intensityDataExcluded[1][[False, True, False, True],:]]
			expected.expectedConcentrationExcluded = [expected.expectedConcentrationExcluded[1].loc[[False, True, False, True],:]]
			expected.expectedConcentrationExcluded[0].index = [2, 4]
			expected.excludedFlag = ['Samples','Samples']
			expected.peakInfo['peakResponse'] = expected.peakInfo['peakResponse'].loc[[True, False, True, False], :]
			expected.peakInfo['peakArea'] = expected.peakInfo['peakArea'].loc[[True, False, True, False], :]
			expected.peakInfo['peakConcentrationDeviation'] = expected.peakInfo['peakConcentrationDeviation'].loc[[True, False, True, False], :]
			expected.peakInfo['peakIntegrationFlag'] = expected.peakInfo['peakIntegrationFlag'].loc[[True, False, True, False], :]
			expected.peakInfo['peakRT'] = expected.peakInfo['peakRT'].loc[[True, False, True, False], :]

			# Result
			result = copy.deepcopy(self.targetedInFilterSample)
			result.sampleMetadataExcluded = []
			result.intensityDataExcluded = []
			result.expectedConcentrationExcluded = []
			result._filterTargetLynxSamples(sampleTypeToProcess=['Study Sample', 'QC'])

			# Class
			self.assertEqual(type(result), type(expected))
			# sampleMetadata
			assert_frame_equal(result.sampleMetadata, expected.sampleMetadata)
			# featureMetadata
			assert_frame_equal(result.featureMetadata.reindex(sorted(result.featureMetadata), axis=1),	expected.featureMetadata.reindex(sorted(expected.featureMetadata), axis=1))
			# intensityData
			numpy.testing.assert_array_equal(result._intensityData, expected._intensityData)
			# expectedConcentration
			assert_frame_equal(result.expectedConcentration.reindex(sorted(result.expectedConcentration), axis=1), expected.expectedConcentration.reindex(sorted(expected.expectedConcentration), axis=1))
			# Calibration
			assert_frame_equal(result.calibration['calibSampleMetadata'], expected.calibration['calibSampleMetadata'])
			assert_frame_equal(result.calibration['calibFeatureMetadata'], expected.calibration['calibFeatureMetadata'])
			numpy.testing.assert_array_equal(result.calibration['calibIntensityData'], expected.calibration['calibIntensityData'])
			assert_frame_equal(result.calibration['calibExpectedConcentration'], expected.calibration['calibExpectedConcentration'])
			assert_frame_equal(result.calibration['calibPeakInfo']['peakResponse'], expected.calibration['calibPeakInfo']['peakResponse'])
			assert_frame_equal(result.calibration['calibPeakInfo']['peakArea'], expected.calibration['calibPeakInfo']['peakArea'])
			assert_frame_equal(result.calibration['calibPeakInfo']['peakConcentrationDeviation'], expected.calibration['calibPeakInfo']['peakConcentrationDeviation'])
			assert_frame_equal(result.calibration['calibPeakInfo']['peakIntegrationFlag'], expected.calibration['calibPeakInfo']['peakIntegrationFlag'])
			assert_frame_equal(result.calibration['calibPeakInfo']['peakRT'], expected.calibration['calibPeakInfo']['peakRT'])
			# Import exclusion
			self.assertEqual(len(result.sampleMetadataExcluded), len(expected.sampleMetadataExcluded))
			for i in range(len(result.sampleMetadataExcluded)):
				assert_frame_equal(result.sampleMetadataExcluded[i], expected.sampleMetadataExcluded[i])
			self.assertEqual(len(result.featureMetadataExcluded), len(expected.featureMetadataExcluded))
			for j in range(len(result.featureMetadataExcluded)):
				assert_frame_equal(result.featureMetadataExcluded[j], expected.featureMetadataExcluded[j])
			self.assertEqual(len(result.intensityDataExcluded), len(expected.intensityDataExcluded))
			for k in range(len(result.intensityDataExcluded)):
				numpy.testing.assert_array_equal(result.intensityDataExcluded[k], expected.intensityDataExcluded[k])
			self.assertEqual(len(result.expectedConcentrationExcluded), len(expected.expectedConcentrationExcluded))
			for l in range(len(result.expectedConcentrationExcluded)):
				assert_frame_equal(result.expectedConcentrationExcluded[l], expected.expectedConcentrationExcluded[l])
			self.assertEqual(result.excludedFlag, expected.excludedFlag)
			# peakInfo
			assert_frame_equal(result.peakInfo['peakResponse'], expected.peakInfo['peakResponse'])
			assert_frame_equal(result.peakInfo['peakArea'], expected.peakInfo['peakArea'])
			assert_frame_equal(result.peakInfo['peakConcentrationDeviation'], expected.peakInfo['peakConcentrationDeviation'])
			assert_frame_equal(result.peakInfo['peakIntegrationFlag'], expected.peakInfo['peakIntegrationFlag'])
			assert_frame_equal(result.peakInfo['peakRT'], expected.peakInfo['peakRT'])


	def test_targeteddataset_filtertargetlynxsamples_raise(self):

		with self.subTest(msg='Checking ValueError if sampleTypeToProcess does not exist'):
			self.assertRaises(ValueError, lambda: self.targetedInFilterSample._filterTargetLynxSamples(sampleTypeToProcess=['Study Sample', 'Unknown Type']))

		with self.subTest(msg='Checking AttributeError if sampleMetadataExcluded does not exist'):
			missingExcludedImportSampleMetadata = copy.deepcopy(self.targetedInFilterSample)
			delattr(missingExcludedImportSampleMetadata, 'sampleMetadataExcluded')
			self.assertRaises(AttributeError, lambda: missingExcludedImportSampleMetadata._filterTargetLynxSamples(sampleTypeToProcess=['Study Sample', 'QC']))

		with self.subTest(msg='Checking AttributeError if featureMetadataExcluded does not exist'):
			missingExcludedImportFeatureMetadata = copy.deepcopy(self.targetedInFilterSample)
			delattr(missingExcludedImportFeatureMetadata, 'featureMetadataExcluded')
			self.assertRaises(AttributeError, lambda: missingExcludedImportFeatureMetadata._filterTargetLynxSamples(sampleTypeToProcess=['Study Sample', 'QC']))

		with self.subTest(msg='Checking AttributeError if intensityDataExcluded does not exist'):
			missingExcludedImportIntensityData = copy.deepcopy(self.targetedInFilterSample)
			delattr(missingExcludedImportIntensityData, 'intensityDataExcluded')
			self.assertRaises(AttributeError, lambda: missingExcludedImportIntensityData._filterTargetLynxSamples(sampleTypeToProcess=['Study Sample', 'QC']))

		with self.subTest(msg='Checking AttributeError if expectedConcentrationExcluded does not exist'):
			missingExcludedImportExpectedConcentration = copy.deepcopy(self.targetedInFilterSample)
			delattr(missingExcludedImportExpectedConcentration, 'expectedConcentrationExcluded')
			self.assertRaises(AttributeError, lambda: missingExcludedImportExpectedConcentration._filterTargetLynxSamples(sampleTypeToProcess=['Study Sample', 'QC']))

		with self.subTest(msg='Checking AttributeError if excludedFlag does not exist'):
			missingExcludedImportExcludedFlag = copy.deepcopy(self.targetedInFilterSample)
			delattr(missingExcludedImportExcludedFlag, 'excludedFlag')
			self.assertRaises(AttributeError, lambda: missingExcludedImportExcludedFlag._filterTargetLynxSamples(sampleTypeToProcess=['Study Sample', 'QC']))


	@unittest.mock.patch('sys.stdout', new_callable=io.StringIO)
	def test_targeteddataset_filtertargetlynxis(self, mock_stdout):
		# Use a modified output of `_filterTargetLynxSamples(sampleTypeToProcess=['Study Sample', 'QC', 'Blank', 'Other'])` as input
		# Expected
		expected = copy.deepcopy(self.targetedOutFilterIS)
		expected.excludedFlag = ['Samples', 'Features']
		# Result
		result = copy.deepcopy(self.targetedOutFilterSample)
		# no exclusion from Filtering Samples
		result.sampleMetadataExcluded = [expected.sampleMetadataExcluded[0]]
		result.featureMetadataExcluded = [expected.featureMetadataExcluded[0]]
		result.intensityDataExcluded = [expected.intensityDataExcluded[0]]
		result.expectedConcentrationExcluded = [expected.expectedConcentrationExcluded[0]]
		result._filterTargetLynxIS()

		# Class
		self.assertEqual(type(result), type(expected))
		# sampleMetadata
		assert_frame_equal(result.sampleMetadata, expected.sampleMetadata)
		# featureMetadata
		assert_frame_equal(result.featureMetadata.reindex(sorted(result.featureMetadata), axis=1), expected.featureMetadata.reindex(sorted(expected.featureMetadata), axis=1))
		# intensityData
		numpy.testing.assert_array_equal(result._intensityData, expected._intensityData)
		# expectedConcentration
		assert_frame_equal(result.expectedConcentration.reindex(sorted(result.expectedConcentration), axis=1),expected.expectedConcentration.reindex(sorted(expected.expectedConcentration), axis=1))
		# Calibration
		assert_frame_equal(result.calibration['calibSampleMetadata'], expected.calibration['calibSampleMetadata'])
		assert_frame_equal(result.calibration['calibFeatureMetadata'], expected.calibration['calibFeatureMetadata'])
		numpy.testing.assert_array_equal(result.calibration['calibIntensityData'], expected.calibration['calibIntensityData'])
		assert_frame_equal(result.calibration['calibExpectedConcentration'], expected.calibration['calibExpectedConcentration'])
		assert_frame_equal(result.calibration['calibPeakInfo']['peakResponse'], expected.calibration['calibPeakInfo']['peakResponse'])
		assert_frame_equal(result.calibration['calibPeakInfo']['peakArea'], expected.calibration['calibPeakInfo']['peakArea'])
		assert_frame_equal(result.calibration['calibPeakInfo']['peakConcentrationDeviation'], expected.calibration['calibPeakInfo']['peakConcentrationDeviation'])
		assert_frame_equal(result.calibration['calibPeakInfo']['peakIntegrationFlag'], expected.calibration['calibPeakInfo']['peakIntegrationFlag'])
		assert_frame_equal(result.calibration['calibPeakInfo']['peakRT'], expected.calibration['calibPeakInfo']['peakRT'])
		# Import exclusion
		self.assertEqual(len(result.sampleMetadataExcluded), len(expected.sampleMetadataExcluded))
		for i in range(len(result.sampleMetadataExcluded)):
			assert_frame_equal(result.sampleMetadataExcluded[i], expected.sampleMetadataExcluded[i])
		self.assertEqual(len(result.featureMetadataExcluded), len(expected.featureMetadataExcluded))
		for j in range(len(result.featureMetadataExcluded)):
			assert_frame_equal(result.featureMetadataExcluded[j], expected.featureMetadataExcluded[j])
		self.assertEqual(len(result.intensityDataExcluded), len(expected.intensityDataExcluded))
		for k in range(len(result.intensityDataExcluded)):
			numpy.testing.assert_array_equal(result.intensityDataExcluded[k], expected.intensityDataExcluded[k])
		self.assertEqual(len(result.expectedConcentrationExcluded), len(expected.expectedConcentrationExcluded))
		for l in range(len(result.expectedConcentrationExcluded)):
			assert_frame_equal(result.expectedConcentrationExcluded[l], expected.expectedConcentrationExcluded[l])
		self.assertEqual(result.excludedFlag, expected.excludedFlag)
		# peakInfo
		assert_frame_equal(result.peakInfo['peakResponse'], expected.peakInfo['peakResponse'])
		assert_frame_equal(result.peakInfo['peakArea'], expected.peakInfo['peakArea'])
		assert_frame_equal(result.peakInfo['peakConcentrationDeviation'], expected.peakInfo['peakConcentrationDeviation'])
		assert_frame_equal(result.peakInfo['peakIntegrationFlag'], expected.peakInfo['peakIntegrationFlag'])
		assert_frame_equal(result.peakInfo['peakRT'], expected.peakInfo['peakRT'])

		expectedStdOut = '1 feature are kept for processing, 1 IS removed\n-----\n'
		self.assertEqual(mock_stdout.getvalue(), expectedStdOut)


	def test_targeteddataset_filtertargetlynxis_raise(self):

		with self.subTest(msg='Checking AttributeError if calibration does not exist'):
			missingCalibration = copy.deepcopy(self.targetedOutFilterSample)
			delattr(missingCalibration, 'calibration')
			self.assertRaises(AttributeError, lambda: missingCalibration._filterTargetLynxIS())

		with self.subTest(msg='Checking AttributeError if sampleMetadataExcluded does not exist'):
			missingExcludedImportSampleMetadata = copy.deepcopy(self.targetedOutFilterSample)
			delattr(missingExcludedImportSampleMetadata, 'sampleMetadataExcluded')
			self.assertRaises(AttributeError, lambda: missingExcludedImportSampleMetadata._filterTargetLynxIS())

		with self.subTest(msg='Checking AttributeError if featureMetadataExcluded does not exist'):
			missingExcludedImportFeatureMetadata = copy.deepcopy(self.targetedOutFilterSample)
			delattr(missingExcludedImportFeatureMetadata, 'featureMetadataExcluded')
			self.assertRaises(AttributeError, lambda: missingExcludedImportFeatureMetadata._filterTargetLynxIS())

		with self.subTest(msg='Checking AttributeError if intensityDataExcluded does not exist'):
			missingExcludedImportIntensityData = copy.deepcopy(self.targetedOutFilterSample)
			delattr(missingExcludedImportIntensityData, 'intensityDataExcluded')
			self.assertRaises(AttributeError, lambda: missingExcludedImportIntensityData._filterTargetLynxIS())

		with self.subTest(msg='Checking AttributeError if expectedConcentrationExcluded does not exist'):
			missingExcludedImportExpectedConcentration = copy.deepcopy(self.targetedOutFilterSample)
			delattr(missingExcludedImportExpectedConcentration, 'expectedConcentrationExcluded')
			self.assertRaises(AttributeError, lambda: missingExcludedImportExpectedConcentration._filterTargetLynxIS())

		with self.subTest(msg='Checking AttributeError if excludedFlag does not exist'):
			missingExcludedImportExcludedFlag = copy.deepcopy(self.targetedInFilterSample)
			delattr(missingExcludedImportExcludedFlag, 'excludedFlag')
			self.assertRaises(AttributeError, lambda: missingExcludedImportExcludedFlag._filterTargetLynxIS())


class test_targeteddataset_read_data_from_targetlynx(unittest.TestCase):
	"""
	Test import from TargetLynx: XML file, calibration report CSV file, and matching of both
	Underlying functions have already been independently tested
	"""
	def setUp(self):
		# 3 Features, 2 samples in TL .xml
		# 2 Features in SOP/Calib, 'Feature 1 - IS' renamed to 'Feature1'. 'Feature3' dropped
		self.targeted = nPYc.TargetedDataset('', fileType='empty')

		# Calibration report .csv
		self.calibrationReport = pandas.DataFrame({'Compound': ['Feature1', 'Feature2'], 'TargetLynx ID': [1, 2], 'Cpd Info': ['uM', 'fg/uL'], 'Noise (area)': [38.95, 14.7], 'LLOQ': [25, 10],'ULOQ': [1000, 2500], 'a': [0.997431, 1.04095], 'b': [-2.19262, numpy.nan], 'r': [0.997931, 0.999556], 'r2': [0.995866, 0.999113], 'another column': [numpy.nan, 'something']})

		# SOP JSON
		self.targeted.Attributes['compoundID'] = ['1', '2']
		self.targeted.Attributes['compoundName'] = ['Feature1', 'Feature2']
		self.targeted.Attributes['IS'] = ['True', 'False']
		self.targeted.Attributes['unitFinal'] = ['noUnit', 'pg/uL']
		self.targeted.Attributes['unitCorrectionFactor'] = [1., 1.]
		self.targeted.Attributes['calibrationMethod'] = ['noIS', 'backcalculatedIS']
		self.targeted.Attributes['calibrationEquation'] = ['', '((area * responseFactor)-b)/a']
		self.targeted.Attributes['quantificationType'] = ['IS', 'QuantOwnLabeledAnalogue']
		self.targeted.Attributes['chromatography'] = 'R'
		self.targeted.Attributes['ionisation'] = 'NEG'
		self.targeted.Attributes['methodName'] = 'UnitTest'
		self.targeted.Attributes['externalID'] = []

		# Expected TargetedDataset
		self.expected = dict()
		self.expected['sampleMetadata'] = pandas.DataFrame({'Sample File Name': ['UnitTest4_targeted_file_001', 'UnitTest4_targeted_file_002'], 'TargetLynx Sample ID': [1, 2], 'MassLynx Row ID': [1, 2],'Sample Name': ['Calibration 1', 'Study Sample 1'], 'Sample Type': ['Standard', 'Analyte'],'Acqu Date': ['11-Sep-16', '11-Sep-16'], 'Acqu Time': ['02:14:32', '09:23:02'], 'Vial': ['1:A,1', '1:A,2'],'Instrument': ['XEVO-TQS#UnitTest', 'XEVO-TQS#UnitTest'], 'Calibrant': [True, False],'Study Sample': [False, True], 'Blank': [False, False], 'QC': [False, False], 'Other': [False, False],'Acquired Time': [datetime(2016, 9, 11, 2, 14, 32),datetime(2016, 9, 11, 9, 23, 2)], 'Run Order': [0, 1],'Batch': [1, 1],'AssayRole': [numpy.nan,numpy.nan],'SampleType': [numpy.nan,numpy.nan],'Dilution': [numpy.nan,numpy.nan],'Correction Batch': [numpy.nan,numpy.nan],'Sample ID': [numpy.nan,numpy.nan],'Exclusion Details': [numpy.nan,numpy.nan]})
		self.expected['sampleMetadata']['Sample Base Name'] = self.expected['sampleMetadata']['Sample File Name']
		self.expected['sampleMetadata']['Acquired Time'] = self.expected['sampleMetadata']['Acquired Time']
		self.expected['sampleMetadata']['Metadata Available'] = False
		self.expected['featureMetadata'] = pandas.DataFrame({'Feature Name': ['Feature1', 'Feature2'], 'TargetLynx Feature ID':[1, 2], 'TargetLynx IS ID': ['', '1'],'IS':[True, False], 'calibrationEquation':['', '((area * responseFactor)-b)/a'],'calibrationMethod': [CalibrationMethod.noIS, CalibrationMethod.backcalculatedIS], 'quantificationType': [QuantificationType.IS, QuantificationType.QuantOwnLabeledAnalogue], 'unitCorrectionFactor': [1.,1.], 'Unit': ['noUnit','pg/uL'], 'Cpd Info': ['uM','fg/uL'], 'LLOQ': [25., 10.], 'Noise (area)': [38.95, 14.7], 'ULOQ': [1000., 2500.], 'a': [0.997431, 1.04095], 'another column': [numpy.nan, 'something'],'b': [-2.19262, numpy.nan], 'r': [0.997931, 0.999556], 'r2': [0.995866, 0.999113]})
		self.expected['featureMetadata']['IS'] = numpy.array([True, False], dtype=object)
		self.expected['intensityData'] = numpy.array([[48.64601435, 48.7244571], [20.60696312, 273.85551508]])
		self.expected['expectedConcentration'] = pandas.DataFrame(numpy.array([[50., 50.], [60., numpy.nan]]), columns=self.expected['featureMetadata']['Feature Name'].values)
		# peak Info
		self.expected['peakResponse'] = pandas.DataFrame(numpy.array([[1.33416750e+05, 5.40558251e+00], [5.65167380e+04, 2.97330493e+01]]), columns=self.expected['featureMetadata']['Feature Name'].values)
		self.expected['peakArea'] = pandas.DataFrame(numpy.array([[133416.75, 14423.905], [56516.738, 28006.916]]), columns=self.expected['featureMetadata']['Feature Name'].values)
		self.expected['peakConcentrationDeviation'] = pandas.DataFrame(numpy.array([[-2.70797131, -2.5510858], [-65.65506146, numpy.nan]]), columns=self.expected['featureMetadata']['Feature Name'].values)
		self.expected['peakIntegrationFlag'] = pandas.DataFrame({'1': ['bb', 'bb'], '2': ['bb', 'bb']})
		self.expected['peakIntegrationFlag'].columns = self.expected['featureMetadata']['Feature Name'].values
		self.expected['peakRT'] = pandas.DataFrame(numpy.array([[11.4263000488, 11.4921998978], [11.4306001663, 11.5010004044]]), columns=self.expected['featureMetadata']['Feature Name'].values)
		# Excluded
		self.expected['sampleMetadataExcluded'] = [self.expected['sampleMetadata'][['Sample File Name', 'Sample Base Name', 'TargetLynx Sample ID', 'MassLynx Row ID', 'Sample Name', 'Sample Type', 'Acqu Date', 'Acqu Time', 'Vial', 'Instrument']]]
		self.expected['sampleMetadataExcluded'][0].loc[:, 'Metadata Available'] = False
		featureMetadataExcluded = pandas.DataFrame({'Feature Name': ['Feature3'], 'TargetLynx Feature ID': [3], 'TargetLynx IS ID': ['1'], 'IS': [numpy.nan],'calibrationEquation': [numpy.nan], 'calibrationMethod': [numpy.nan], 'compoundID': [numpy.nan], 'compoundName': [numpy.nan], 'quantificationType': [numpy.nan],'unitCorrectionFactor': [numpy.nan], 'Unit': [numpy.nan], 'Cpd Info': [numpy.nan],'LLOQ': [numpy.nan], 'Noise (area)': [numpy.nan], 'ULOQ': [numpy.nan], 'a': [numpy.nan],'another column': [numpy.nan], 'b': [numpy.nan], 'r': [numpy.nan], 'r2': [numpy.nan]})
		featureMetadataExcluded.index = [2]
		featureMetadataExcluded['IS'] = numpy.array([numpy.nan], dtype=object)
		featureMetadataExcluded['calibrationEquation'] = numpy.array([numpy.nan], dtype=object)
		featureMetadataExcluded['calibrationMethod'] = numpy.array([numpy.nan], dtype=object)
		featureMetadataExcluded['compoundName'] = numpy.array([numpy.nan], dtype=object)
		featureMetadataExcluded['quantificationType'] = numpy.array([numpy.nan], dtype=object)
		featureMetadataExcluded['Unit'] = numpy.array([numpy.nan], dtype=object)
		featureMetadataExcluded['Cpd Info'] = numpy.array([numpy.nan], dtype=object)
		featureMetadataExcluded['another column'] = numpy.array([numpy.nan], dtype=object)
		self.expected['featureMetadataExcluded'] = [featureMetadataExcluded]
		self.expected['intensityDataExcluded'] = [numpy.array([[48.76854933], [359.219531]])]
		self.expected['expectedConcentrationExcluded'] = [pandas.DataFrame({'Feature3':[50., numpy.nan]})]
		self.expected['excludedFlag'] = ['Features']
		# Attributes
		self.expected['Attributes'] = copy.deepcopy(self.targeted.Attributes)
		for k in ['compoundID', 'compoundName', 'IS', 'unitFinal', 'unitCorrectionFactor', 'calibrationMethod', 'calibrationEquation', 'quantificationType', 'Log']:
			del self.expected['Attributes'][k]


	@unittest.mock.patch('sys.stdout', new_callable=io.StringIO)
	def test_targeteddataset_readtargetlynxdataset(self, mock_stdout):

		with tempfile.TemporaryDirectory() as tmpdirname:
			# Init
			reportPath = os.path.join(tmpdirname, 'calibrationReport.csv')
			self.calibrationReport.to_csv(reportPath, index=False)
			XMLpath = os.path.join('..', '..', 'npc-standard-project', 'Derived_Data', 'UnitTest4_targeted.xml')
			result = copy.deepcopy(self.targeted)
			# Generate
			with warnings.catch_warnings():
				warnings.simplefilter('ignore', UserWarning)
				result._readTargetLynxDataset(datapath=XMLpath, calibrationReportPath=reportPath)
			# Test
			assert_frame_equal(self.expected['sampleMetadata'].reindex(sorted(self.expected['sampleMetadata']), axis=1), result.sampleMetadata.reindex(sorted(result.sampleMetadata), axis=1))
			assert_frame_equal(self.expected['featureMetadata'].reindex(sorted(self.expected['featureMetadata']), axis=1), result.featureMetadata.reindex(sorted(result.featureMetadata), axis=1))
			numpy.testing.assert_array_almost_equal(self.expected['intensityData'], result._intensityData)
			assert_frame_equal(self.expected['expectedConcentration'].reindex(sorted(self.expected['expectedConcentration']), axis=1), result.expectedConcentration.reindex(sorted(result.expectedConcentration), axis=1))
			assert_frame_equal(self.expected['sampleMetadataExcluded'][0].reindex(sorted(self.expected['sampleMetadataExcluded'][0]), axis=1), result.sampleMetadataExcluded[0].reindex(sorted(result.sampleMetadataExcluded[0]), axis=1))
			assert_frame_equal(self.expected['featureMetadataExcluded'][0].reindex(sorted(self.expected['featureMetadataExcluded'][0]), axis=1), result.featureMetadataExcluded[0].reindex(sorted(result.featureMetadataExcluded[0]), axis=1))
			numpy.testing.assert_array_almost_equal(self.expected['intensityDataExcluded'][0], result.intensityDataExcluded[0])
			assert_frame_equal(self.expected['expectedConcentrationExcluded'][0].reindex(sorted(self.expected['expectedConcentrationExcluded'][0]), axis=1), result.expectedConcentrationExcluded[0].reindex(sorted(result.expectedConcentrationExcluded[0]), axis=1))
			self.assertEqual(self.expected['excludedFlag'], result.excludedFlag)
			assert_frame_equal(self.expected['peakResponse'].reindex(sorted(self.expected['peakResponse']), axis=1), result.peakInfo['peakResponse'].reindex(sorted(result.peakInfo['peakResponse']), axis=1))
			assert_frame_equal(self.expected['peakArea'].reindex(sorted(self.expected['peakArea']), axis=1), result.peakInfo['peakArea'].reindex(sorted(result.peakInfo['peakArea']), axis=1))
			assert_frame_equal(self.expected['peakConcentrationDeviation'].reindex(sorted(self.expected['peakConcentrationDeviation']), axis=1), result.peakInfo['peakConcentrationDeviation'].reindex(sorted(result.peakInfo['peakConcentrationDeviation']), axis=1))
			assert_frame_equal(self.expected['peakIntegrationFlag'].reindex(sorted(self.expected['peakIntegrationFlag']), axis=1), result.peakInfo['peakIntegrationFlag'].reindex(sorted(result.peakInfo['peakIntegrationFlag']), axis=1))
			assert_frame_equal(self.expected['peakRT'].reindex(sorted(self.expected['peakRT']), axis=1), result.peakInfo['peakRT'].reindex(sorted(result.peakInfo['peakRT']), axis=1))
			self.assertEqual(len(self.expected['Attributes'].keys()), len(result.Attributes.keys())-1)
			for i in self.expected['Attributes']:
				self.assertEqual(self.expected['Attributes'][i], result.Attributes[i])


class test_targeteddataset_limitsofquantification(unittest.TestCase):
	"""
	Test apply limits of quantification, here on imported dataset from TargetLynx
	"""
	def setUp(self):
		# 3 samples (1<LLOQ, 1 ok, 1>ULOQ)
		# 2 features, (1 Monitored, 2 normal (make them fail LLOQ/ULOQ later)) a fake exclusion
		self.targetedDataset = nPYc.TargetedDataset('', fileType='empty')
		self.targetedDataset.sampleMetadata = pandas.DataFrame({'Sample File Name': ['UnitTest_targeted_file_002', 'UnitTest_targeted_file_003','UnitTest_targeted_file_004'], 'Sample Name': ['Sample1-LLOQ', 'Sample2', 'Sample3-ULOQ'], 'Sample Type': ['Analyte', 'Analyte', 'Analyte'], 'Acqu Date': ['10-Sep-16', '10-Sep-16','10-Sep-16'], 'Acqu Time': ['03:23:02', '04:52:35', '05:46:40'], 'Vial': ['1:A,2', '1:A,3', '1:A,4'], 'Instrument': ['XEVO-TQS#UnitTest', 'XEVO-TQS#UnitTest','XEVO-TQS#UnitTest'], 'Acquired Time': [datetime(2016, 9, 10, 3, 23, 2), datetime(2016, 9, 10, 4, 52, 35), datetime(2016, 9, 10, 5, 46, 40)],'Run Order': [0, 1, 2], 'Batch': [1, 1, 1]})
		self.targetedDataset.sampleMetadata['Acquired Time'] = self.targetedDataset.sampleMetadata['Acquired Time']
		self.targetedDataset.featureMetadata = pandas.DataFrame({'Feature Name': ['Feature1-Monitored', 'Feature2','Feature3'], 'TargetLynx Feature ID': [1, 2,3], 'calibrationEquation': ['', '10**((numpy.log10(area * responseFactor)-b)/a)','10**((numpy.log10(area * responseFactor)-b)/a)'], 'calibrationMethod': [CalibrationMethod.noCalibration, CalibrationMethod.backcalculatedIS, CalibrationMethod.backcalculatedIS], 'quantificationType': [QuantificationType.Monitored, QuantificationType.QuantAltLabeledAnalogue, QuantificationType.QuantAltLabeledAnalogue], 'unitCorrectionFactor': [numpy.nan, 1.,1.], 'Unit': ['noUnit', 'pg/uL','another Unit'], 'Cpd Info': ['info cpd1', 'info cpd2', 'indo cpd3'], 'LLOQ': [numpy.nan, 100., 100.], 'ULOQ': [numpy.nan, 1000.,1000.], 'another column': ['something 1', 'something 2', 'something 3']})
		self.targetedDataset._intensityData = numpy.array([[50., 50., 50.], [500., 500., 500.], [5000., 5000., 5000.]])
		self.targetedDataset.expectedConcentration = pandas.DataFrame(numpy.array([[1.,2.,3.], [4., 5.,6.], [7., 8., 9.]]), columns=self.targetedDataset.featureMetadata['Feature Name'].values.tolist())
		self.targetedDataset.sampleMetadataExcluded = [pandas.DataFrame(numpy.random.random((5, 17)))]
		self.targetedDataset.featureMetadataExcluded = [pandas.DataFrame(numpy.random.random((2, 13)))]
		self.targetedDataset.intensityDataExcluded = [numpy.random.random((5, 2))]
		self.targetedDataset.expectedConcentrationExcluded = [pandas.DataFrame(numpy.random.random((5, 2)))]
		self.targetedDataset.excludedFlag = ['Samples']
		calibSampleMetadata = pandas.DataFrame({'Sample File Name': ['UnitTest_targeted_file_001'],'Sample Name': ['Calib'],'Sample Type': ['Standard'],'Acqu Date': ['10-Sep-16'],'Acqu Time': ['02:14:32'],'Vial': ['1:A,1'],'Instrument': ['XEVO-TQS#UnitTest'],'Acquired Time': [datetime(2016, 9, 10, 2, 14, 32)],'Run Order': [0],'Batch': [1]})
		calibSampleMetadata['Acquired Time'] = calibSampleMetadata['Acquired Time']
		calibFeatureMetadata = pandas.DataFrame({'Feature Name': ['Feature1-Monitored', 'Feature2','Feature3'], 'TargetLynx Feature ID': [1, 2,3], 'calibrationEquation': ['', '10**((numpy.log10(area * responseFactor)-b)/a)','10**((numpy.log10(area * responseFactor)-b)/a)'], 'calibrationMethod': [CalibrationMethod.noCalibration, CalibrationMethod.backcalculatedIS, CalibrationMethod.backcalculatedIS], 'quantificationType': [QuantificationType.Monitored, QuantificationType.QuantAltLabeledAnalogue, QuantificationType.QuantAltLabeledAnalogue], 'unitCorrectionFactor': [numpy.nan, 1.,1.], 'Unit': ['noUnit', 'pg/uL','another Unit'], 'Cpd Info': ['info cpd1', 'info cpd2', 'indo cpd3'], 'LLOQ': [numpy.nan, 100., 100.], 'ULOQ': [numpy.nan, 1000.,1000.], 'another column': ['something 1', 'something 2', 'something 3']})
		calibIntensityData = numpy.array([[100., 200., 300.]])
		calibExpectedConcentration = pandas.DataFrame(numpy.array([[100.,200.,300.]]), columns=self.targetedDataset.featureMetadata['Feature Name'].values.tolist())
		self.targetedDataset.calibration = {'calibSampleMetadata': calibSampleMetadata ,'calibFeatureMetadata': calibFeatureMetadata,'calibIntensityData': calibIntensityData,'calibExpectedConcentration': calibExpectedConcentration}


	@unittest.mock.patch('sys.stdout', new_callable=io.StringIO)
	def test_applylimitsofquantification(self, mock_stdout):

		with self.subTest(msg='Check Basic ApplyLOQ with onlyLLOQ=False, previous exclusion'):
			# Expected
			expected = copy.deepcopy(self.targetedDataset)
			# filtered values and reordered columns
			expected._intensityData = numpy.array([[-numpy.inf, -numpy.inf, 50.], [500., 500., 500.], [numpy.inf, numpy.inf, 5000.]])
			expected.featureMetadata = expected.featureMetadata.iloc[[1,2,0],:]
			expected.featureMetadata.reset_index(drop=True, inplace=True)
			expected.expectedConcentration = expected.expectedConcentration.iloc[:, [1, 2, 0]]
			expected.calibration['calibIntensityData'] = expected.calibration['calibIntensityData'][:,[1, 2, 0]]
			expected.calibration['calibFeatureMetadata'] = expected.calibration['calibFeatureMetadata'].iloc[[1, 2, 0], :]
			expected.calibration['calibFeatureMetadata'].reset_index(drop=True, inplace=True)
			expected.calibration['calibExpectedConcentration'] = expected.calibration['calibExpectedConcentration'].iloc[:,[1, 2, 0]]
			# Result
			result = copy.deepcopy(self.targetedDataset)
			result._applyLimitsOfQuantification(onlyLLOQ=False)

			# Class
			self.assertEqual(type(result), type(expected))
			# sampleMetadata
			assert_frame_equal(result.sampleMetadata, expected.sampleMetadata)
			# featureMetadata
			assert_frame_equal(result.featureMetadata, expected.featureMetadata)
			# intensityData
			numpy.testing.assert_array_equal(result._intensityData, expected._intensityData)
			# expectedConcentration
			assert_frame_equal(result.expectedConcentration, expected.expectedConcentration)
			# Calibration
			assert_frame_equal(result.calibration['calibSampleMetadata'], expected.calibration['calibSampleMetadata'])
			assert_frame_equal(result.calibration['calibFeatureMetadata'], expected.calibration['calibFeatureMetadata'])
			numpy.testing.assert_array_equal(result.calibration['calibIntensityData'], expected.calibration['calibIntensityData'])
			assert_frame_equal(result.calibration['calibExpectedConcentration'], expected.calibration['calibExpectedConcentration'])
			# Exclusions
			self.assertEqual(len(result.sampleMetadataExcluded), len(expected.sampleMetadataExcluded))
			for i in range(len(result.sampleMetadataExcluded)):
				assert_frame_equal(result.sampleMetadataExcluded[i], expected.sampleMetadataExcluded[i])
			self.assertEqual(len(result.featureMetadataExcluded), len(expected.featureMetadataExcluded))
			for j in range(len(result.featureMetadataExcluded)):
				assert_frame_equal(result.featureMetadataExcluded[j], expected.featureMetadataExcluded[j])
			self.assertEqual(len(result.intensityDataExcluded), len(expected.intensityDataExcluded))
			for k in range(len(result.intensityDataExcluded)):
				numpy.testing.assert_array_equal(result.intensityDataExcluded[k], expected.intensityDataExcluded[k])
			self.assertEqual(len(result.expectedConcentrationExcluded), len(expected.expectedConcentrationExcluded))
			for l in range(len(result.expectedConcentrationExcluded)):
				assert_frame_equal(result.expectedConcentrationExcluded[l], expected.expectedConcentrationExcluded[l])
			self.assertEqual(result.excludedFlag, expected.excludedFlag)

		with self.subTest(msg='Check ApplyLOQ with onlyLLOQ=True, previous exclusion'):
			# Expected
			expected = copy.deepcopy(self.targetedDataset)
			# filtered values and reordered columns
			expected._intensityData = numpy.array([[-numpy.inf, -numpy.inf, 50.], [500., 500., 500.], [5000., 5000., 5000.]])
			expected.featureMetadata = expected.featureMetadata.iloc[[1,2,0],:]
			expected.featureMetadata.reset_index(drop=True, inplace=True)
			expected.expectedConcentration = expected.expectedConcentration.iloc[:, [1, 2, 0]]
			expected.calibration['calibIntensityData'] = expected.calibration['calibIntensityData'][:,[1, 2, 0]]
			expected.calibration['calibFeatureMetadata'] = expected.calibration['calibFeatureMetadata'].iloc[[1, 2, 0], :]
			expected.calibration['calibFeatureMetadata'].reset_index(drop=True, inplace=True)
			expected.calibration['calibExpectedConcentration'] = expected.calibration['calibExpectedConcentration'].iloc[:,[1, 2, 0]]
			# Result
			result = copy.deepcopy(self.targetedDataset)
			result._applyLimitsOfQuantification(onlyLLOQ=True)

			# Class
			self.assertEqual(type(result), type(expected))
			# sampleMetadata
			assert_frame_equal(result.sampleMetadata, expected.sampleMetadata)
			# featureMetadata
			assert_frame_equal(result.featureMetadata, expected.featureMetadata)
			# intensityData
			numpy.testing.assert_array_equal(result._intensityData, expected._intensityData)
			# expectedConcentration
			assert_frame_equal(result.expectedConcentration, expected.expectedConcentration)
			# Calibration
			assert_frame_equal(result.calibration['calibSampleMetadata'], expected.calibration['calibSampleMetadata'])
			assert_frame_equal(result.calibration['calibFeatureMetadata'], expected.calibration['calibFeatureMetadata'])
			numpy.testing.assert_array_equal(result.calibration['calibIntensityData'], expected.calibration['calibIntensityData'])
			assert_frame_equal(result.calibration['calibExpectedConcentration'], expected.calibration['calibExpectedConcentration'])
			# Exclusions
			self.assertEqual(len(result.sampleMetadataExcluded), len(expected.sampleMetadataExcluded))
			for i in range(len(result.sampleMetadataExcluded)):
				assert_frame_equal(result.sampleMetadataExcluded[i], expected.sampleMetadataExcluded[i])
			self.assertEqual(len(result.featureMetadataExcluded), len(expected.featureMetadataExcluded))
			for j in range(len(result.featureMetadataExcluded)):
				assert_frame_equal(result.featureMetadataExcluded[j], expected.featureMetadataExcluded[j])
			self.assertEqual(len(result.intensityDataExcluded), len(expected.intensityDataExcluded))
			for k in range(len(result.intensityDataExcluded)):
				numpy.testing.assert_array_equal(result.intensityDataExcluded[k], expected.intensityDataExcluded[k])
			self.assertEqual(len(result.expectedConcentrationExcluded), len(expected.expectedConcentrationExcluded))
			for l in range(len(result.expectedConcentrationExcluded)):
				assert_frame_equal(result.expectedConcentrationExcluded[l], expected.expectedConcentrationExcluded[l])
			self.assertEqual(result.excludedFlag, expected.excludedFlag)

		with self.subTest(msg='Check ApplyLOQ missing \'LLOQ\' Feature2, previous exclusion'):
			# Expected
			expected = copy.deepcopy(self.targetedDataset)
			# filtered values and reordered columns
			expected.sampleMetadataExcluded.append(expected.sampleMetadata)
			expected.intensityDataExcluded.append(numpy.array([[50.], [500.], [5000.]]))
			expected.featureMetadataExcluded.append(expected.featureMetadata.iloc[[1],:])
			expected.featureMetadataExcluded[1]['LLOQ'] = numpy.nan
			expected.expectedConcentrationExcluded.append(expected.expectedConcentration.iloc[:, [1]])
			expected.excludedFlag.append('Features')
			expected._intensityData = numpy.array([[-numpy.inf, 50.], [500., 500.], [numpy.inf, 5000.]])
			expected.featureMetadata = expected.featureMetadata.iloc[[2,0],:]
			expected.featureMetadata.reset_index(drop=True, inplace=True)
			expected.expectedConcentration = expected.expectedConcentration.iloc[:, [2, 0]]
			expected.calibration['calibIntensityData'] = expected.calibration['calibIntensityData'][:,[2, 0]]
			expected.calibration['calibFeatureMetadata'] = expected.calibration['calibFeatureMetadata'].iloc[[2, 0], :]
			expected.calibration['calibFeatureMetadata'].reset_index(drop=True, inplace=True)
			expected.calibration['calibExpectedConcentration'] = expected.calibration['calibExpectedConcentration'].iloc[:, [2, 0]]
			# Result
			result = copy.deepcopy(self.targetedDataset)
			result.featureMetadata['LLOQ'] = [numpy.nan, numpy.nan, 100.]
			result._applyLimitsOfQuantification(onlyLLOQ=False)

			# Class
			self.assertEqual(type(result), type(expected))
			# sampleMetadata
			assert_frame_equal(result.sampleMetadata, expected.sampleMetadata)
			# featureMetadata
			assert_frame_equal(result.featureMetadata, expected.featureMetadata)
			# intensityData
			numpy.testing.assert_array_equal(result._intensityData, expected._intensityData)
			# expectedConcentration
			assert_frame_equal(result.expectedConcentration, expected.expectedConcentration)
			# Calibration
			assert_frame_equal(result.calibration['calibSampleMetadata'], expected.calibration['calibSampleMetadata'])
			assert_frame_equal(result.calibration['calibFeatureMetadata'], expected.calibration['calibFeatureMetadata'])
			numpy.testing.assert_array_equal(result.calibration['calibIntensityData'], expected.calibration['calibIntensityData'])
			assert_frame_equal(result.calibration['calibExpectedConcentration'], expected.calibration['calibExpectedConcentration'])
			# Exclusions
			self.assertEqual(len(result.sampleMetadataExcluded), len(expected.sampleMetadataExcluded))
			for i in range(len(result.sampleMetadataExcluded)):
				assert_frame_equal(result.sampleMetadataExcluded[i], expected.sampleMetadataExcluded[i])
			self.assertEqual(len(result.featureMetadataExcluded), len(expected.featureMetadataExcluded))
			for j in range(len(result.featureMetadataExcluded)):
				assert_frame_equal(result.featureMetadataExcluded[j], expected.featureMetadataExcluded[j])
			self.assertEqual(len(result.intensityDataExcluded), len(expected.intensityDataExcluded))
			for k in range(len(result.intensityDataExcluded)):
				numpy.testing.assert_array_equal(result.intensityDataExcluded[k], expected.intensityDataExcluded[k])
			self.assertEqual(len(result.expectedConcentrationExcluded), len(expected.expectedConcentrationExcluded))
			for l in range(len(result.expectedConcentrationExcluded)):
				assert_frame_equal(result.expectedConcentrationExcluded[l], expected.expectedConcentrationExcluded[l])
			self.assertEqual(result.excludedFlag, expected.excludedFlag)

		with self.subTest(msg='Check ApplyLOQ missing \'ULOQ\' Feature2, previous exclusion'):
			# Expected
			expected = copy.deepcopy(self.targetedDataset)
			# filtered values and reordered columns
			expected.sampleMetadataExcluded.append(expected.sampleMetadata)
			expected.intensityDataExcluded.append(numpy.array([[50.], [500.], [5000.]]))
			expected.featureMetadataExcluded.append(expected.featureMetadata.iloc[[1],:])
			expected.featureMetadataExcluded[1]['ULOQ'] = numpy.nan
			expected.expectedConcentrationExcluded.append(expected.expectedConcentration.iloc[:, [1]])
			expected.excludedFlag.append('Features')
			expected._intensityData = numpy.array([[-numpy.inf, 50.], [500., 500.], [numpy.inf, 5000.]])
			expected.featureMetadata = expected.featureMetadata.iloc[[2,0],:]
			expected.featureMetadata.reset_index(drop=True, inplace=True)
			expected.expectedConcentration = expected.expectedConcentration.iloc[:, [2, 0]]
			expected.calibration['calibIntensityData'] = expected.calibration['calibIntensityData'][:,[2, 0]]
			expected.calibration['calibFeatureMetadata'] = expected.calibration['calibFeatureMetadata'].iloc[[2, 0], :]
			expected.calibration['calibFeatureMetadata'].reset_index(drop=True, inplace=True)
			expected.calibration['calibExpectedConcentration'] = expected.calibration['calibExpectedConcentration'].iloc[:, [2, 0]]
			# Result
			result = copy.deepcopy(self.targetedDataset)
			result.featureMetadata['ULOQ'] = [numpy.nan, numpy.nan, 1000.]
			result._applyLimitsOfQuantification(onlyLLOQ=False)

			# Class
			self.assertEqual(type(result), type(expected))
			# sampleMetadata
			assert_frame_equal(result.sampleMetadata, expected.sampleMetadata)
			# featureMetadata
			assert_frame_equal(result.featureMetadata, expected.featureMetadata)
			# intensityData
			numpy.testing.assert_array_equal(result._intensityData, expected._intensityData)
			# expectedConcentration
			assert_frame_equal(result.expectedConcentration, expected.expectedConcentration)
			# Calibration
			assert_frame_equal(result.calibration['calibSampleMetadata'], expected.calibration['calibSampleMetadata'])
			assert_frame_equal(result.calibration['calibFeatureMetadata'], expected.calibration['calibFeatureMetadata'])
			numpy.testing.assert_array_equal(result.calibration['calibIntensityData'], expected.calibration['calibIntensityData'])
			assert_frame_equal(result.calibration['calibExpectedConcentration'], expected.calibration['calibExpectedConcentration'])
			# Exclusions
			self.assertEqual(len(result.sampleMetadataExcluded), len(expected.sampleMetadataExcluded))
			for i in range(len(result.sampleMetadataExcluded)):
				assert_frame_equal(result.sampleMetadataExcluded[i], expected.sampleMetadataExcluded[i])
			self.assertEqual(len(result.featureMetadataExcluded), len(expected.featureMetadataExcluded))
			for j in range(len(result.featureMetadataExcluded)):
				assert_frame_equal(result.featureMetadataExcluded[j], expected.featureMetadataExcluded[j])
			self.assertEqual(len(result.intensityDataExcluded), len(expected.intensityDataExcluded))
			for k in range(len(result.intensityDataExcluded)):
				numpy.testing.assert_array_equal(result.intensityDataExcluded[k], expected.intensityDataExcluded[k])
			self.assertEqual(len(result.expectedConcentrationExcluded), len(expected.expectedConcentrationExcluded))
			for l in range(len(result.expectedConcentrationExcluded)):
				assert_frame_equal(result.expectedConcentrationExcluded[l], expected.expectedConcentrationExcluded[l])
			self.assertEqual(result.excludedFlag, expected.excludedFlag)

		with self.subTest(msg='Check ApplyLOQ missing \'LLOQ\' Feature2, No previous exclusion'):
			# Expected
			expected = copy.deepcopy(self.targetedDataset)
			# filtered values and reordered columns
			expected.sampleMetadataExcluded[0] = expected.sampleMetadata
			expected.intensityDataExcluded[0] = numpy.array([[50.], [500.], [5000.]])
			expected.featureMetadataExcluded[0] = expected.featureMetadata.iloc[[1],:]
			expected.featureMetadataExcluded[0]['LLOQ'] = numpy.nan
			expected.expectedConcentrationExcluded[0] = expected.expectedConcentration.iloc[:, [1]]
			expected.excludedFlag[0] = 'Features'
			expected._intensityData = numpy.array([[-numpy.inf, 50.], [500., 500.], [numpy.inf, 5000.]])
			expected.featureMetadata = expected.featureMetadata.iloc[[2,0],:]
			expected.featureMetadata.reset_index(drop=True, inplace=True)
			expected.expectedConcentration = expected.expectedConcentration.iloc[:, [2, 0]]
			expected.calibration['calibIntensityData'] = expected.calibration['calibIntensityData'][:,[2, 0]]
			expected.calibration['calibFeatureMetadata'] = expected.calibration['calibFeatureMetadata'].iloc[[2, 0], :]
			expected.calibration['calibFeatureMetadata'].reset_index(drop=True, inplace=True)
			expected.calibration['calibExpectedConcentration'] = expected.calibration['calibExpectedConcentration'].iloc[:, [2, 0]]
			# Result
			result = copy.deepcopy(self.targetedDataset)
			result.featureMetadata['LLOQ'] = [numpy.nan, numpy.nan, 100.]
			delattr(result, 'intensityDataExcluded')
			delattr(result, 'sampleMetadataExcluded')
			delattr(result, 'featureMetadataExcluded')
			delattr(result, 'expectedConcentrationExcluded')
			delattr(result, 'excludedFlag')
			result._applyLimitsOfQuantification(onlyLLOQ=False)

			# Class
			self.assertEqual(type(result), type(expected))
			# sampleMetadata
			assert_frame_equal(result.sampleMetadata, expected.sampleMetadata)
			# featureMetadata
			assert_frame_equal(result.featureMetadata, expected.featureMetadata)
			# intensityData
			numpy.testing.assert_array_equal(result._intensityData, expected._intensityData)
			# expectedConcentration
			assert_frame_equal(result.expectedConcentration, expected.expectedConcentration)
			# Calibration
			assert_frame_equal(result.calibration['calibSampleMetadata'], expected.calibration['calibSampleMetadata'])
			assert_frame_equal(result.calibration['calibFeatureMetadata'], expected.calibration['calibFeatureMetadata'])
			numpy.testing.assert_array_equal(result.calibration['calibIntensityData'], expected.calibration['calibIntensityData'])
			assert_frame_equal(result.calibration['calibExpectedConcentration'], expected.calibration['calibExpectedConcentration'])
			# Exclusions
			self.assertEqual(len(result.sampleMetadataExcluded), len(expected.sampleMetadataExcluded))
			for i in range(len(result.sampleMetadataExcluded)):
				assert_frame_equal(result.sampleMetadataExcluded[i], expected.sampleMetadataExcluded[i])
			self.assertEqual(len(result.featureMetadataExcluded), len(expected.featureMetadataExcluded))
			for j in range(len(result.featureMetadataExcluded)):
				assert_frame_equal(result.featureMetadataExcluded[j], expected.featureMetadataExcluded[j])
			self.assertEqual(len(result.intensityDataExcluded), len(expected.intensityDataExcluded))
			for k in range(len(result.intensityDataExcluded)):
				numpy.testing.assert_array_equal(result.intensityDataExcluded[k], expected.intensityDataExcluded[k])
			self.assertEqual(len(result.expectedConcentrationExcluded), len(expected.expectedConcentrationExcluded))
			for l in range(len(result.expectedConcentrationExcluded)):
				assert_frame_equal(result.expectedConcentrationExcluded[l], expected.expectedConcentrationExcluded[l])
			self.assertEqual(result.excludedFlag, expected.excludedFlag)

		with self.subTest(msg='Check ApplyLOQ with QuantificationType.QuantOther and ULOQ numpy.nan, onlyLLOQ=False, previous exclusion'):
			# Feature 1 is QuantificationType.QuantOther, ULOQ is nan, feature is retained and no LOQ applied where nan
			# Expected
			expected = copy.deepcopy(self.targetedDataset)
			# filtered values and reordered columns
			expected._intensityData = numpy.array([[-numpy.inf, -numpy.inf, 50.], [500., 500., 500.], [5000., numpy.inf, 5000.]])
			expected.featureMetadata.loc[1, 'quantificationType'] = QuantificationType.QuantOther
			expected.featureMetadata.loc[1, 'calibrationMethod'] = CalibrationMethod.otherCalibration
			expected.featureMetadata.loc[1, 'ULOQ'] = numpy.nan
			expected.featureMetadata = expected.featureMetadata.iloc[[1,2,0],:]
			expected.featureMetadata.reset_index(drop=True, inplace=True)
			expected.expectedConcentration = expected.expectedConcentration.iloc[:, [1, 2, 0]]
			expected.calibration['calibIntensityData'] = expected.calibration['calibIntensityData'][:,[1, 2, 0]]
			expected.calibration['calibFeatureMetadata'] = expected.calibration['calibFeatureMetadata'].iloc[[1, 2, 0], :]
			expected.calibration['calibFeatureMetadata'].reset_index(drop=True, inplace=True)
			expected.calibration['calibExpectedConcentration'] = expected.calibration['calibExpectedConcentration'].iloc[:,[1, 2, 0]]
			# Result
			result = copy.deepcopy(self.targetedDataset)
			result.featureMetadata.loc[1, 'quantificationType'] = QuantificationType.QuantOther
			result.featureMetadata.loc[1, 'calibrationMethod'] = CalibrationMethod.otherCalibration
			result.featureMetadata.loc[1, 'ULOQ'] = numpy.nan
			result._applyLimitsOfQuantification(onlyLLOQ=False)

			# Class
			self.assertEqual(type(result), type(expected))
			# sampleMetadata
			assert_frame_equal(result.sampleMetadata, expected.sampleMetadata)
			# featureMetadata
			assert_frame_equal(result.featureMetadata, expected.featureMetadata)
			# intensityData
			numpy.testing.assert_array_equal(result._intensityData, expected._intensityData)
			# expectedConcentration
			assert_frame_equal(result.expectedConcentration, expected.expectedConcentration)
			# Calibration
			assert_frame_equal(result.calibration['calibSampleMetadata'], expected.calibration['calibSampleMetadata'])
			assert_frame_equal(result.calibration['calibFeatureMetadata'], expected.calibration['calibFeatureMetadata'])
			numpy.testing.assert_array_equal(result.calibration['calibIntensityData'], expected.calibration['calibIntensityData'])
			assert_frame_equal(result.calibration['calibExpectedConcentration'], expected.calibration['calibExpectedConcentration'])
			# Exclusions
			self.assertEqual(len(result.sampleMetadataExcluded), len(expected.sampleMetadataExcluded))
			for i in range(len(result.sampleMetadataExcluded)):
				assert_frame_equal(result.sampleMetadataExcluded[i], expected.sampleMetadataExcluded[i])
			self.assertEqual(len(result.featureMetadataExcluded), len(expected.featureMetadataExcluded))
			for j in range(len(result.featureMetadataExcluded)):
				assert_frame_equal(result.featureMetadataExcluded[j], expected.featureMetadataExcluded[j])
			self.assertEqual(len(result.intensityDataExcluded), len(expected.intensityDataExcluded))
			for k in range(len(result.intensityDataExcluded)):
				numpy.testing.assert_array_equal(result.intensityDataExcluded[k], expected.intensityDataExcluded[k])
			self.assertEqual(len(result.expectedConcentrationExcluded), len(expected.expectedConcentrationExcluded))
			for l in range(len(result.expectedConcentrationExcluded)):
				assert_frame_equal(result.expectedConcentrationExcluded[l], expected.expectedConcentrationExcluded[l])
			self.assertEqual(result.excludedFlag, expected.excludedFlag)


	def test_applylimitsofquantification_raise(self):

		with self.subTest(msg='Checking AttributeError if \'LLOQ\' missing'):
			failLLOQ = copy.deepcopy(self.targetedDataset)
			failLLOQ.featureMetadata = failLLOQ.featureMetadata.drop(['LLOQ'], axis=1)
			self.assertRaises(AttributeError, lambda: failLLOQ._applyLimitsOfQuantification(onlyLLOQ=False))

		with self.subTest(msg='Checking AttributeError if \'ULOQ\' missing'):
			failULOQ = copy.deepcopy(self.targetedDataset)
			failULOQ.featureMetadata = failULOQ.featureMetadata.drop(['ULOQ'], axis=1)
			self.assertRaises(AttributeError, lambda: failULOQ._applyLimitsOfQuantification(onlyLLOQ=False))


class test_targeteddataset_mergelimitsofquantification(unittest.TestCase):
	"""
	Test merging of limits of quantification after __add__
	"""
	def setUp(self):
		# Feature1 has lowest LLOQ and ULOQ in batch1, feature2 has lowest LLOQ and ULOQ in batch2, feature3 has numpy.nan in batch1 LLOQ and batch2 ULOQ
		# On feature1 and feature2, Sample1 will be <LLOQ, Sample2 >ULOQ, Sample3 same as input. Feature3 removed in applyLOQ to NA
		self.targetedDataset = nPYc.TargetedDataset('', fileType='empty')
		self.targetedDataset.sampleMetadata = pandas.DataFrame({'Sample File Name': ['UnitTest_targeted_file_001', 'UnitTest_targeted_file_002', 'UnitTest_targeted_file_003'],
																'Sample Name': ['Sample1-B1', 'Sample2-B2', 'Sample3-B2'],
																'Sample Type': ['Analyte', 'Analyte', 'Analyte'],
																'Acqu Date': ['10-Sep-16', '10-Sep-16', '10-Sep-16'],
																'Acqu Time': ['03:23:02', '04:52:35', '05:46:40'],
																'Vial': ['1:A,2', '1:A,3', '1:A,4'],
																'Instrument': ['XEVO-TQS#UnitTest', 'XEVO-TQS#UnitTest', 'XEVO-TQS#UnitTest'],
																'Acquired Time': [datetime(2016, 9, 10, 3, 23, 2), datetime(2016, 9, 10, 4, 52, 35), datetime(2016, 9, 10, 5, 46, 40)],
																'Run Order': [0, 1, 2], 'Batch': [1, 2, 2],
																'AssayRole': [AssayRole.Assay, AssayRole.Assay, AssayRole.Assay],
																'SampleType': [SampleType.StudySample, SampleType.StudySample, SampleType.StudySample],
																'Dilution': [numpy.nan, numpy.nan, numpy.nan],
																'Correction Batch': [numpy.nan, numpy.nan, numpy.nan],
																'Subject ID': ['', '', ''], 'Sample ID': ['', '', ''],
																'Sample Base Name': ['', '', ''],
																'Exclusion Details': ['', '', '']})
		self.targetedDataset.sampleMetadata['Acquired Time'] = self.targetedDataset.sampleMetadata['Acquired Time']
		self.targetedDataset.featureMetadata = pandas.DataFrame({'Feature Name': ['Feature1', 'Feature2', 'Feature3'],
																 'TargetLynx Feature ID': [1, 2, 3],
																 'calibrationMethod': [CalibrationMethod.backcalculatedIS, CalibrationMethod.backcalculatedIS, CalibrationMethod.backcalculatedIS],
																 'quantificationType': [QuantificationType.QuantAltLabeledAnalogue, QuantificationType.QuantAltLabeledAnalogue, QuantificationType.QuantAltLabeledAnalogue],
																 'unitCorrectionFactor': [1., 1., 1.],
																 'Unit': ['a Unit', 'pg/uL', 'another Unit'],
																 'Cpd Info': ['info cpd1', 'info cpd2', 'info cpd3'],
																 'LLOQ_batch1': [5., 20., numpy.nan],
																 'LLOQ_batch2': [20., 5., 5.],
																 'ULOQ_batch1': [80., 100., 80.],
																 'ULOQ_batch2': [100., 80., numpy.nan]})
		self.targetedDataset._intensityData = numpy.array([[10., 10., 10.], [90., 90., 90.], [25., 70., 70.]])
		self.targetedDataset.expectedConcentration = pandas.DataFrame(numpy.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]), columns=self.targetedDataset.featureMetadata['Feature Name'].values.tolist())
		self.targetedDataset.sampleMetadataExcluded = []
		self.targetedDataset.featureMetadataExcluded = []
		self.targetedDataset.intensityDataExcluded = []
		self.targetedDataset.expectedConcentrationExcluded = []
		self.targetedDataset.excludedFlag = []
		self.targetedDataset.calibration = []
		self.targetedDataset.Attributes['methodName'] = 'unittest'
		self.targetedDataset.Attributes['externalID'] = []
		self.targetedDataset.VariableType = VariableType.Discrete
		self.targetedDataset.initialiseMasks()


	@unittest.mock.patch('sys.stdout', new_callable=io.StringIO)
	def test_mergelimitsofquantification(self, mock_stdout):
		# On feature1 and feature2, Sample1 will be <LLOQ, Sample2 >ULOQ, Sample3 same as input. Feature3 removed in applyLOQ due to the NA
		# No check of exclusions as it's tested in applyLOQ

		with self.subTest(msg='Check mergeLOQ with onlyLLOQ=False, keepBatchLOQ=False'):
			# Expected
			expected = copy.deepcopy(self.targetedDataset)
			# filtered values and reordered columns
			expected._intensityData = numpy.array([[-numpy.inf, -numpy.inf], [numpy.inf, numpy.inf], [25., 70.]])
			expected.featureMetadata = expected.featureMetadata.iloc[[0, 1], :]
			expected.featureMetadata['LLOQ'] = [20., 20.]
			expected.featureMetadata['ULOQ'] = [80., 80.]
			expected.featureMetadata.drop(['LLOQ_batch1','LLOQ_batch2','ULOQ_batch1','ULOQ_batch2'], inplace=True, axis=1)
			#expected.featureMetadata.reset_index(drop=True, inplace=True)
			expected.expectedConcentration = expected.expectedConcentration.iloc[:, [0, 1]]
			# Result
			result = copy.deepcopy(self.targetedDataset)
			result.mergeLimitsOfQuantification(onlyLLOQ=False, keepBatchLOQ=False)

			# Class
			self.assertEqual(type(result), type(expected))
			# sampleMetadata
			assert_frame_equal(result.sampleMetadata, expected.sampleMetadata)
			# featureMetadata
			assert_frame_equal(result.featureMetadata, expected.featureMetadata)
			# intensityData
			numpy.testing.assert_array_equal(result._intensityData, expected._intensityData)
			# expectedConcentration
			assert_frame_equal(result.expectedConcentration, expected.expectedConcentration)
			# Calibration
			self.assertEqual(result.calibration, expected.calibration)

		with self.subTest(msg='Check mergeLOQ with onlyLLOQ=False, keepBatchLOQ=True'):
			# Expected
			expected = copy.deepcopy(self.targetedDataset)
			# filtered values and reordered columns
			expected._intensityData = numpy.array([[-numpy.inf, -numpy.inf], [numpy.inf, numpy.inf], [25., 70.]])
			expected.featureMetadata = expected.featureMetadata.iloc[[0, 1], :]
			expected.featureMetadata['LLOQ'] = [20., 20.]
			expected.featureMetadata['ULOQ'] = [80., 80.]
			#expected.featureMetadata.reset_index(drop=True, inplace=True)
			expected.expectedConcentration = expected.expectedConcentration.iloc[:, [0, 1]]
			# Result
			result = copy.deepcopy(self.targetedDataset)
			result.mergeLimitsOfQuantification(onlyLLOQ=False, keepBatchLOQ=True)

			# Class
			self.assertEqual(type(result), type(expected))
			# sampleMetadata
			assert_frame_equal(result.sampleMetadata, expected.sampleMetadata)
			# featureMetadata
			assert_frame_equal(result.featureMetadata, expected.featureMetadata)
			# intensityData
			numpy.testing.assert_array_equal(result._intensityData, expected._intensityData)
			# expectedConcentration
			assert_frame_equal(result.expectedConcentration, expected.expectedConcentration)
			# Calibration
			self.assertEqual(result.calibration, expected.calibration)

		with self.subTest(msg='Check mergeLOQ with onlyLLOQ=True, keepBatchLOQ=False'):
			# Expected
			expected = copy.deepcopy(self.targetedDataset)
			# filtered values and reordered columns
			expected._intensityData = numpy.array([[-numpy.inf, -numpy.inf], [90., 90.], [25., 70.]])
			expected.featureMetadata = expected.featureMetadata.iloc[[0, 1], :]
			expected.featureMetadata['LLOQ'] = [20., 20.]
			expected.featureMetadata['ULOQ'] = [80., 80.]
			expected.featureMetadata.drop(['LLOQ_batch1', 'LLOQ_batch2', 'ULOQ_batch1', 'ULOQ_batch2'], inplace=True, axis=1)
			# expected.featureMetadata.reset_index(drop=True, inplace=True)
			expected.expectedConcentration = expected.expectedConcentration.iloc[:, [0, 1]]
			# Result
			result = copy.deepcopy(self.targetedDataset)
			result.mergeLimitsOfQuantification(onlyLLOQ=True, keepBatchLOQ=False)

			# Class
			self.assertEqual(type(result), type(expected))
			# sampleMetadata
			assert_frame_equal(result.sampleMetadata, expected.sampleMetadata)
			# featureMetadata
			assert_frame_equal(result.featureMetadata, expected.featureMetadata)
			# intensityData
			numpy.testing.assert_array_equal(result._intensityData, expected._intensityData)
			# expectedConcentration
			assert_frame_equal(result.expectedConcentration, expected.expectedConcentration)
			# Calibration
			self.assertEqual(result.calibration, expected.calibration)


	@unittest.mock.patch('sys.stdout', new_callable=io.StringIO)
	def test_mergelimitsofquantification_raise(self, mock_stdout):

		with self.subTest(msg='Check ValueError if input does not satisfy to BasicTargetedDataset'):
			failmergeLOQ = copy.deepcopy(self.targetedDataset)
			failmergeLOQ.sampleMetadata.drop(['AssayRole'], axis=1, inplace=True)
			self.assertRaises(ValueError, lambda: failmergeLOQ.mergeLimitsOfQuantification(onlyLLOQ=True, keepBatchLOQ=False))

		with self.subTest(msg='Check ValueError if number of batch and LLOQ_batchX or ULOQ_batchX do not match'):
			failmergeLOQ = copy.deepcopy(self.targetedDataset)
			failmergeLOQ.featureMetadata.drop(['LLOQ_batch1'], axis=1, inplace=True)
			self.assertRaises(ValueError, lambda: failmergeLOQ.mergeLimitsOfQuantification(onlyLLOQ=True, keepBatchLOQ=False))

			failmergeLOQ = copy.deepcopy(self.targetedDataset)
			failmergeLOQ.featureMetadata.drop(['ULOQ_batch2'], axis=1, inplace=True)
			self.assertRaises(ValueError, lambda: failmergeLOQ.mergeLimitsOfQuantification(onlyLLOQ=True, keepBatchLOQ=False))

		with self.subTest(msg='Checking Warning if LLOQ or ULOQ already exist'):
			triggerWarning = copy.deepcopy(self.targetedDataset)
			triggerWarning.featureMetadata['LLOQ'] = [numpy.nan,numpy.nan,numpy.nan]

			with self.assertWarnsRegex(UserWarning, 'values will be overwritten'):

				triggerWarning.mergeLimitsOfQuantification(onlyLLOQ=True, keepBatchLOQ=False)


class test_targeteddataset_targetlynxlimitsofquantificationnoisefilled(unittest.TestCase):
	"""
	Test apply limits of quantification noise filled, only works on imported dataset from TargetLynx
	"""
	def setUp(self):
		# 9 samples (3 calib, 3 samples, 1 blank, 1 QC, 1 Other)
		# 10 features (1 IS, 1 Monitored, 2 unusable (missing LLOQ ULOQ), 3 unusableNoiseFilled (noise area, a, b), 3 different equations)
		self.targetedIn = nPYc.TargetedDataset('', fileType='empty')
		self.targetedIn.sampleMetadata = pandas.DataFrame({'Sample File Name': ['UnitTest_targeted_file_001','UnitTest_targeted_file_002','UnitTest_targeted_file_003'],
															 'Sample Name': ['Sample-LLOQ', 'Sample-Fine','Sample-ULOQ'],
															 'Sample Type': ['Analyte', 'Analyte', 'Analyte'],
															 'Acqu Date': ['10-Sep-16', '10-Sep-16', '10-Sep-16'],
															 'Acqu Time': ['02:14:32', '03:23:02', '04:52:35'],
															 'Vial': ['1:A,1', '1:A,2', '1:A,3'],
															 'Instrument': ['XEVO-TQS#UnitTest', 'XEVO-TQS#UnitTest','XEVO-TQS#UnitTest'],
															 'Acquired Time': [datetime(2016, 9, 10, 2, 14, 32),datetime(2016, 9, 10, 3, 23, 2),datetime(2016, 9, 10, 4, 52, 35)],
															 'Run Order': [0, 1, 2], 'Batch': [1, 1, 1]})
		self.targetedIn.sampleMetadata['Acquired Time'] = self.targetedIn.sampleMetadata['Acquired Time']
		self.targetedIn.featureMetadata = pandas.DataFrame({'Feature Name': ['Feature1-Monitored','Feature2-UnusableLLOQ','Feature3-UnusableULOQ','Feature4-UnusableNoiseFilled-Noise','Feature5-UnusableNoiseFilled-a','Feature6-UnusableNoiseFilled-b','Feature7-axb', 'Feature8-logaxb','Feature9-ax'],
															  'TargetLynx Feature ID': [1, 2, 3, 4, 5, 6, 7, 8, 9],
															  'calibrationEquation': ['', '', 'area/a', '', '', '','((area * responseFactor)-b)/a','10**((numpy.log10(area * responseFactor)-b)/a)','area/a'],
															  'calibrationMethod': [CalibrationMethod.noCalibration, CalibrationMethod.backcalculatedIS, CalibrationMethod.backcalculatedIS, CalibrationMethod.backcalculatedIS, CalibrationMethod.backcalculatedIS, CalibrationMethod.backcalculatedIS, CalibrationMethod.backcalculatedIS, CalibrationMethod.backcalculatedIS, CalibrationMethod.noIS],
															  'quantificationType': [QuantificationType.Monitored, QuantificationType.QuantOwnLabeledAnalogue, QuantificationType.QuantAltLabeledAnalogue, QuantificationType.QuantOwnLabeledAnalogue, QuantificationType.QuantOwnLabeledAnalogue, QuantificationType.QuantAltLabeledAnalogue, QuantificationType.QuantOwnLabeledAnalogue, QuantificationType.QuantAltLabeledAnalogue, QuantificationType.QuantOwnLabeledAnalogue],
															  'unitCorrectionFactor': [1., 1., 1., 1., 1., 1., 1., 1.,1.],
															  'Unit': ['pg/uL', 'uM', 'pg/uL', 'uM', 'pg/uL', 'uM','pg/uL', 'uM', 'pg/uL'],
															  'Cpd Info': ['info cpd1', 'info cpd2', 'info cpd3','info cpd4', 'info cpd5', 'info cpd6','info cpd7', 'info cpd8', 'info cpd9'],
															  'LLOQ': [numpy.nan, numpy.nan, 100., 100., 100.,100., 100., 100., 100.],
															  'ULOQ': [numpy.nan, 1000, numpy.nan, 1000., 1000.,1000., 1000., 1000., 1000.],
															  'Noise (area)': [numpy.nan, 10., 10., numpy.nan, 10., 10., 10., 10., 10.],
															  'a': [numpy.nan, 2., 2., 2., numpy.nan, 2., 2., 2., 2.],
															  'another column': ['something 1', 'something 2','something 3', 'something 4','something 5', 'something 6','something 7', 'something 8','something 9'],
															  'b': [numpy.nan, 1., 1., 1., 1., numpy.nan, 1., 1., 0.],
															  'r': [0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99,0.99],
															  'r2': [0.995, 0.995, 0.995, 0.995, 0.995, 0.995, 0.995,0.995, 0.995]})
		self.targetedIn._intensityData = numpy.array([[50., 50., 50., 50., 50., 50., 50., 50., 50.], [500., 500., 500., 500., 500., 500., 500., 500., 500.],[5000., 5000., 5000., 5000., 5000., 5000., 5000., 5000., 5000.]])
		self.targetedIn.expectedConcentration = pandas.DataFrame(numpy.array([[1., 2., 3., 4., 5., 6., 7., 8., 9.], [11., 12., 13., 14., 15., 16., 17., 18., 19.],[21., 22., 23., 24., 25., 26., 27., 28., 29.]]), columns=self.targetedIn.featureMetadata['Feature Name'].values.tolist())
		self.targetedIn.sampleMetadataExcluded = [pandas.DataFrame(numpy.random.random((3, 17)))]
		self.targetedIn.featureMetadataExcluded = [pandas.DataFrame(numpy.random.random((9, 13)))]
		self.targetedIn.intensityDataExcluded = [numpy.random.random((3, 9))]
		self.targetedIn.expectedConcentrationExcluded = [pandas.DataFrame(numpy.random.random((3, 9)))]
		self.targetedIn.excludedFlag = ['Samples']
		calibSampleMetadata = pandas.DataFrame({'Sample File Name': ['UnitTest_targeted_file_004','UnitTest_targeted_file_005','UnitTest_targeted_file_006'],
												'Sample Name': ['Calib-Low', 'Calib-Mid', 'Calib-High'],
												'Sample Type': ['Standard', 'Standard', 'Standard'],
												'Acqu Date': ['10-Sep-16', '10-Sep-16', '10-Sep-16'],
												'Acqu Time': ['05:46:40', '06:05:26', '07:26:32'],
												'Vial': ['1:A,4', '1:A,5', '1:A,6'],
												'Instrument': ['XEVO-TQS#UnitTest', 'XEVO-TQS#UnitTest','XEVO-TQS#UnitTest'],
												'Acquired Time': [datetime(2016, 9, 10, 5, 46, 40), datetime(2016, 9, 10, 6, 5, 26), datetime(2016, 9, 10, 7, 26, 32)],
												'Run Order': [3, 4, 5], 'Batch': [1, 1, 1]})
		calibSampleMetadata['Acquired Time'] = calibSampleMetadata['Acquired Time']
		calibFeatureMetadata = copy.deepcopy(self.targetedIn.featureMetadata)
		calibIntensityData = numpy.array([[250., 250., 250., 250., 250., 250., 250., 250., 250.],[500., 500., 500., 500., 500., 500., 500., 500., 500.],[750., 750., 750., 750., 750., 750., 750., 750., 750.]])
		calibExpectedConcentration = copy.deepcopy(self.targetedIn.expectedConcentration)
		calibPeakArea = pandas.DataFrame(numpy.array([[500., 500., 500., 500., 500., 500., 500., 500., 500.],[1000., 1000., 1000., 1000., 1000., 1000., 1000., 1000., 1000.],[1500., 1500., 1500., 1500., 1500., 1500., 1500., 1500., 1500.]]), columns=calibFeatureMetadata['Feature Name'].values.tolist(), index = calibSampleMetadata.index.values.tolist())
		calibPeakResponse = pandas.DataFrame(numpy.array([[2100., 2100., 2100., 2100., 2100., 2100., 2100., 2100., 2100.],[2100., 2100., 2100., 2100., 2100., 2100., 2100., 2100., 2100.],[2100., 2100., 2100., 2100., 2100., 2100., 2100., 2100., 2100.]]), columns=calibFeatureMetadata['Feature Name'].values.tolist(), index = calibSampleMetadata.index.values.tolist())
		calibPeakConcentrationDeviation = pandas.DataFrame(numpy.random.random((3, 9)), columns=calibFeatureMetadata['Feature Name'].values.tolist(), index = calibSampleMetadata.index.values.tolist())
		calibPeakIntegrationFlag = pandas.DataFrame(numpy.random.random((3, 9)), columns=calibFeatureMetadata['Feature Name'].values.tolist(), index=calibSampleMetadata.index.values.tolist())
		calibPeakRT = pandas.DataFrame(numpy.random.random((3, 9)), columns=calibFeatureMetadata['Feature Name'].values.tolist(), index=calibSampleMetadata.index.values.tolist())
		calibPeakInfo = {'peakResponse': calibPeakResponse, 'peakArea': calibPeakArea, 'peakConcentrationDeviation': calibPeakConcentrationDeviation, 'peakIntegrationFlag': calibPeakIntegrationFlag, 'peakRT': calibPeakRT}
		self.targetedIn.calibration = {'calibSampleMetadata': calibSampleMetadata, 'calibFeatureMetadata': calibFeatureMetadata, 'calibIntensityData': calibIntensityData, 'calibExpectedConcentration': calibExpectedConcentration, 'calibPeakInfo': calibPeakInfo}


	@unittest.mock.patch('sys.stdout', new_callable=io.StringIO)
	def test_targetlynxlimitsofquantificationnoisefilled(self, mock_stdout):

		with self.subTest(msg='Check Basic ApplyLLOQ Noise Filled with onlyLLOQ=True, responseReference=None auto-ref, previous exclusion'):
			# Expected
			expected = copy.deepcopy(self.targetedIn)
			# Exclusions
			expected.sampleMetadataExcluded.append(expected.sampleMetadata)
			expected.featureMetadataExcluded.append(expected.featureMetadata.iloc[[1,3,4,5],:])
			expected.intensityDataExcluded.append(expected._intensityData[:,[1,3,4,5]])
			expected.expectedConcentrationExcluded.append(expected.expectedConcentration.iloc[:,[1,3,4,5]])
			expected.excludedFlag.append('Features')
			# filtered values and reordered columns
			expected._intensityData = numpy.array([[5., 10., 1.4491376746189439, 5., 50.], [500., 500., 500., 500., 500.], [5000., 5000., 5000., 5000., 5000.]])
			expected.featureMetadata = expected.featureMetadata.iloc[[2,6,7,8,0],:]
			expected.featureMetadata.reset_index(drop=True, inplace=True)
			expected.featureMetadata['noiseConcentration'] = [5., 10., 1.4491376746189439, 5., numpy.nan]
			expected.featureMetadata['responseFactor'] = [2.1, 2.1, 2.1, 2.1, numpy.nan]
			expected.expectedConcentration = expected.expectedConcentration.iloc[:, [2,6,7,8,0]]
			expected.calibration['calibIntensityData'] = expected.calibration['calibIntensityData'][:,[2,6,7,8,0]]
			expected.calibration['calibFeatureMetadata'] = expected.calibration['calibFeatureMetadata'].iloc[[2,6,7,8,0], :]
			expected.calibration['calibFeatureMetadata'].reset_index(drop=True, inplace=True)
			expected.calibration['calibExpectedConcentration'] = expected.calibration['calibExpectedConcentration'].iloc[:,[2,6,7,8,0]]
			expected.calibration['calibPeakInfo']['peakArea'] = expected.calibration['calibPeakInfo']['peakArea'].iloc[:,[2,6,7,8,0]]
			expected.calibration['calibPeakInfo']['peakResponse'] = expected.calibration['calibPeakInfo']['peakResponse'].iloc[:,[2,6,7,8,0]]
			expected.calibration['calibPeakInfo']['peakConcentrationDeviation'] = expected.calibration['calibPeakInfo']['peakConcentrationDeviation'].iloc[:,[2,6,7,8,0]]
			expected.calibration['calibPeakInfo']['peakIntegrationFlag'] = expected.calibration['calibPeakInfo']['peakIntegrationFlag'].iloc[:,[2,6,7,8,0]]
			expected.calibration['calibPeakInfo']['peakRT'] = expected.calibration['calibPeakInfo']['peakRT'].iloc[:,[2,6,7,8,0]]
			# Result
			result = copy.deepcopy(self.targetedIn)
			with warnings.catch_warnings():
				warnings.simplefilter("ignore", UserWarning)
				result._targetLynxApplyLimitsOfQuantificationNoiseFilled(onlyLLOQ=True, responseReference=None)

			# Class
			self.assertEqual(type(result), type(expected))
			# sampleMetadata
			assert_frame_equal(result.sampleMetadata, expected.sampleMetadata)
			# featureMetadata
			assert_frame_equal(result.featureMetadata.reindex(sorted(result.featureMetadata), axis=1),	expected.featureMetadata.reindex(sorted(expected.featureMetadata), axis=1))
			# intensityData
			numpy.testing.assert_array_equal(result._intensityData, expected._intensityData)
			# expectedConcentration
			assert_frame_equal(result.expectedConcentration, expected.expectedConcentration)
			# Calibration
			assert_frame_equal(result.calibration['calibSampleMetadata'], expected.calibration['calibSampleMetadata'])
			assert_frame_equal(result.calibration['calibFeatureMetadata'], expected.calibration['calibFeatureMetadata'])
			numpy.testing.assert_array_equal(result.calibration['calibIntensityData'], expected.calibration['calibIntensityData'])
			assert_frame_equal(result.calibration['calibExpectedConcentration'], expected.calibration['calibExpectedConcentration'])
			assert_frame_equal(result.calibration['calibPeakInfo']['peakArea'], expected.calibration['calibPeakInfo']['peakArea'])
			assert_frame_equal(result.calibration['calibPeakInfo']['peakResponse'], expected.calibration['calibPeakInfo']['peakResponse'])
			assert_frame_equal(result.calibration['calibPeakInfo']['peakConcentrationDeviation'], expected.calibration['calibPeakInfo']['peakConcentrationDeviation'])
			assert_frame_equal(result.calibration['calibPeakInfo']['peakIntegrationFlag'], expected.calibration['calibPeakInfo']['peakIntegrationFlag'])
			assert_frame_equal(result.calibration['calibPeakInfo']['peakRT'], expected.calibration['calibPeakInfo']['peakRT'])
			# Exclusions
			self.assertEqual(len(result.sampleMetadataExcluded), len(expected.sampleMetadataExcluded))
			for i in range(len(result.sampleMetadataExcluded)):
				assert_frame_equal(result.sampleMetadataExcluded[i], expected.sampleMetadataExcluded[i])
			self.assertEqual(len(result.featureMetadataExcluded), len(expected.featureMetadataExcluded))
			for j in range(len(result.featureMetadataExcluded)):
				assert_frame_equal(result.featureMetadataExcluded[j], expected.featureMetadataExcluded[j])
			self.assertEqual(len(result.intensityDataExcluded), len(expected.intensityDataExcluded))
			for k in range(len(result.intensityDataExcluded)):
				numpy.testing.assert_array_equal(result.intensityDataExcluded[k], expected.intensityDataExcluded[k])
			self.assertEqual(len(result.expectedConcentrationExcluded), len(expected.expectedConcentrationExcluded))
			for l in range(len(result.expectedConcentrationExcluded)):
				assert_frame_equal(result.expectedConcentrationExcluded[l], expected.expectedConcentrationExcluded[l])
			self.assertEqual(result.excludedFlag, expected.excludedFlag)

		with self.subTest(msg='Check ApplyLLOQ Noise Filled with onlyLLOQ=False, responseReference=None auto-ref, previous exclusion'):
			# Expected
			expected = copy.deepcopy(self.targetedIn)
			# Exclusions
			expected.sampleMetadataExcluded.append(expected.sampleMetadata)
			expected.featureMetadataExcluded.append(expected.featureMetadata.iloc[[1, 2, 3, 4, 5], :])
			expected.intensityDataExcluded.append(expected._intensityData[:, [1, 2, 3, 4, 5]])
			expected.expectedConcentrationExcluded.append(expected.expectedConcentration.iloc[:, [1, 2, 3, 4, 5]])
			expected.excludedFlag.append('Features')
			# filtered values and reordered columns
			expected._intensityData = numpy.array([[10., 1.4491376746189439, 5., 50.], [500., 500., 500., 500.],[numpy.inf, numpy.inf, numpy.inf, 5000.]])
			expected.featureMetadata = expected.featureMetadata.iloc[[6, 7, 8, 0], :]
			expected.featureMetadata.reset_index(drop=True, inplace=True)
			expected.featureMetadata['noiseConcentration'] = [10., 1.4491376746189439, 5., numpy.nan]
			expected.featureMetadata['responseFactor'] = [2.1, 2.1, 2.1, numpy.nan]
			expected.expectedConcentration = expected.expectedConcentration.iloc[:, [6, 7, 8, 0]]
			expected.calibration['calibIntensityData'] = expected.calibration['calibIntensityData'][:, [6, 7, 8, 0]]
			expected.calibration['calibFeatureMetadata'] = expected.calibration['calibFeatureMetadata'].iloc[[6, 7, 8, 0], :]
			expected.calibration['calibFeatureMetadata'].reset_index(drop=True, inplace=True)
			expected.calibration['calibExpectedConcentration'] = expected.calibration['calibExpectedConcentration'].iloc[:,[6, 7, 8, 0]]
			expected.calibration['calibPeakInfo']['peakArea'] = expected.calibration['calibPeakInfo']['peakArea'].iloc[:, [6, 7, 8, 0]]
			expected.calibration['calibPeakInfo']['peakResponse'] = expected.calibration['calibPeakInfo']['peakResponse'].iloc[:, [6, 7, 8, 0]]
			expected.calibration['calibPeakInfo']['peakConcentrationDeviation'] = expected.calibration['calibPeakInfo']['peakConcentrationDeviation'].iloc[:, [6, 7, 8, 0]]
			expected.calibration['calibPeakInfo']['peakIntegrationFlag'] = expected.calibration['calibPeakInfo']['peakIntegrationFlag'].iloc[:,[6, 7, 8, 0]]
			expected.calibration['calibPeakInfo']['peakRT'] = expected.calibration['calibPeakInfo']['peakRT'].iloc[:,[6, 7, 8, 0]]
			# Result
			result = copy.deepcopy(self.targetedIn)
			with warnings.catch_warnings():
				warnings.simplefilter("ignore", UserWarning)
				result._targetLynxApplyLimitsOfQuantificationNoiseFilled(onlyLLOQ=False, responseReference=None)

			# Class
			self.assertEqual(type(result), type(expected))
			# sampleMetadata
			assert_frame_equal(result.sampleMetadata, expected.sampleMetadata)
			# featureMetadata
			assert_frame_equal(result.featureMetadata.reindex(sorted(result.featureMetadata), axis=1),expected.featureMetadata.reindex(sorted(expected.featureMetadata), axis=1))
			# intensityData
			numpy.testing.assert_array_equal(result._intensityData, expected._intensityData)
			# expectedConcentration
			assert_frame_equal(result.expectedConcentration, expected.expectedConcentration)
			# Calibration
			assert_frame_equal(result.calibration['calibSampleMetadata'], expected.calibration['calibSampleMetadata'])
			assert_frame_equal(result.calibration['calibFeatureMetadata'], expected.calibration['calibFeatureMetadata'])
			numpy.testing.assert_array_equal(result.calibration['calibIntensityData'], expected.calibration['calibIntensityData'])
			assert_frame_equal(result.calibration['calibExpectedConcentration'], expected.calibration['calibExpectedConcentration'])
			assert_frame_equal(result.calibration['calibPeakInfo']['peakArea'], expected.calibration['calibPeakInfo']['peakArea'])
			assert_frame_equal(result.calibration['calibPeakInfo']['peakResponse'], expected.calibration['calibPeakInfo']['peakResponse'])
			assert_frame_equal(result.calibration['calibPeakInfo']['peakConcentrationDeviation'], expected.calibration['calibPeakInfo']['peakConcentrationDeviation'])
			assert_frame_equal(result.calibration['calibPeakInfo']['peakIntegrationFlag'], expected.calibration['calibPeakInfo']['peakIntegrationFlag'])
			assert_frame_equal(result.calibration['calibPeakInfo']['peakRT'], expected.calibration['calibPeakInfo']['peakRT'])
			# Exclusions
			self.assertEqual(len(result.sampleMetadataExcluded), len(expected.sampleMetadataExcluded))
			for i in range(len(result.sampleMetadataExcluded)):
				assert_frame_equal(result.sampleMetadataExcluded[i], expected.sampleMetadataExcluded[i])
			self.assertEqual(len(result.featureMetadataExcluded), len(expected.featureMetadataExcluded))
			for j in range(len(result.featureMetadataExcluded)):
				assert_frame_equal(result.featureMetadataExcluded[j], expected.featureMetadataExcluded[j])
			self.assertEqual(len(result.intensityDataExcluded), len(expected.intensityDataExcluded))
			for k in range(len(result.intensityDataExcluded)):
				numpy.testing.assert_array_equal(result.intensityDataExcluded[k], expected.intensityDataExcluded[k])
			self.assertEqual(len(result.expectedConcentrationExcluded), len(expected.expectedConcentrationExcluded))
			for l in range(len(result.expectedConcentrationExcluded)):
				assert_frame_equal(result.expectedConcentrationExcluded[l], expected.expectedConcentrationExcluded[l])
			self.assertEqual(result.excludedFlag, expected.excludedFlag)

		with self.subTest(msg='Check ApplyLLOQ Noise Filled with onlyLLOQ=False, responseReference=None auto-ref, no previous exclusion'):
			# Expected
			expected = copy.deepcopy(self.targetedIn)
			# Exclusions
			expected.sampleMetadataExcluded = [expected.sampleMetadata]
			expected.featureMetadataExcluded = [expected.featureMetadata.iloc[[1, 2, 3, 4, 5], :]]
			expected.intensityDataExcluded = [expected._intensityData[:, [1, 2, 3, 4, 5]]]
			expected.expectedConcentrationExcluded = [expected.expectedConcentration.iloc[:, [1, 2, 3, 4, 5]]]
			expected.excludedFlag = ['Features']
			# filtered values and reordered columns
			expected._intensityData = numpy.array([[10., 1.4491376746189439, 5., 50.], [500., 500., 500., 500.],[numpy.inf, numpy.inf, numpy.inf, 5000.]])
			expected.featureMetadata = expected.featureMetadata.iloc[[6, 7, 8, 0], :]
			expected.featureMetadata.reset_index(drop=True, inplace=True)
			expected.featureMetadata['noiseConcentration'] = [10., 1.4491376746189439, 5., numpy.nan]
			expected.featureMetadata['responseFactor'] = [2.1, 2.1, 2.1, numpy.nan]
			expected.expectedConcentration = expected.expectedConcentration.iloc[:, [6, 7, 8, 0]]
			expected.calibration['calibIntensityData'] = expected.calibration['calibIntensityData'][:, [6, 7, 8, 0]]
			expected.calibration['calibFeatureMetadata'] = expected.calibration['calibFeatureMetadata'].iloc[[6, 7, 8, 0], :]
			expected.calibration['calibFeatureMetadata'].reset_index(drop=True, inplace=True)
			expected.calibration['calibExpectedConcentration'] = expected.calibration['calibExpectedConcentration'].iloc[:,[6, 7, 8, 0]]
			expected.calibration['calibPeakInfo']['peakArea'] = expected.calibration['calibPeakInfo']['peakArea'].iloc[:, [6, 7, 8, 0]]
			expected.calibration['calibPeakInfo']['peakResponse'] = expected.calibration['calibPeakInfo']['peakResponse'].iloc[:, [6, 7, 8, 0]]
			expected.calibration['calibPeakInfo']['peakConcentrationDeviation'] = expected.calibration['calibPeakInfo']['peakConcentrationDeviation'].iloc[:, [6, 7, 8, 0]]
			expected.calibration['calibPeakInfo']['peakIntegrationFlag'] = expected.calibration['calibPeakInfo']['peakIntegrationFlag'].iloc[:,[6, 7, 8, 0]]
			expected.calibration['calibPeakInfo']['peakRT'] = expected.calibration['calibPeakInfo']['peakRT'].iloc[:,[6, 7, 8, 0]]
			# Result
			result = copy.deepcopy(self.targetedIn)
			delattr(result, 'sampleMetadataExcluded')
			delattr(result, 'featureMetadataExcluded')
			delattr(result, 'intensityDataExcluded')
			delattr(result, 'expectedConcentrationExcluded')
			delattr(result, 'excludedFlag')

			with warnings.catch_warnings():
				warnings.simplefilter("ignore")
				result._targetLynxApplyLimitsOfQuantificationNoiseFilled(onlyLLOQ=False, responseReference=None)

			# Class
			self.assertEqual(type(result), type(expected))
			# sampleMetadata
			assert_frame_equal(result.sampleMetadata, expected.sampleMetadata)
			# featureMetadata
			assert_frame_equal(result.featureMetadata.reindex(sorted(result.featureMetadata), axis=1),expected.featureMetadata.reindex(sorted(expected.featureMetadata), axis=1))
			# intensityData
			numpy.testing.assert_array_equal(result._intensityData, expected._intensityData)
			# expectedConcentration
			assert_frame_equal(result.expectedConcentration, expected.expectedConcentration)
			# Calibration
			assert_frame_equal(result.calibration['calibSampleMetadata'], expected.calibration['calibSampleMetadata'])
			assert_frame_equal(result.calibration['calibFeatureMetadata'], expected.calibration['calibFeatureMetadata'])
			numpy.testing.assert_array_equal(result.calibration['calibIntensityData'], expected.calibration['calibIntensityData'])
			assert_frame_equal(result.calibration['calibExpectedConcentration'], expected.calibration['calibExpectedConcentration'])
			assert_frame_equal(result.calibration['calibPeakInfo']['peakArea'], expected.calibration['calibPeakInfo']['peakArea'])
			assert_frame_equal(result.calibration['calibPeakInfo']['peakResponse'], expected.calibration['calibPeakInfo']['peakResponse'])
			assert_frame_equal(result.calibration['calibPeakInfo']['peakConcentrationDeviation'], expected.calibration['calibPeakInfo']['peakConcentrationDeviation'])
			assert_frame_equal(result.calibration['calibPeakInfo']['peakIntegrationFlag'], expected.calibration['calibPeakInfo']['peakIntegrationFlag'])
			assert_frame_equal(result.calibration['calibPeakInfo']['peakRT'], expected.calibration['calibPeakInfo']['peakRT'])
			# Exclusions
			self.assertEqual(len(result.sampleMetadataExcluded), len(expected.sampleMetadataExcluded))
			for i in range(len(result.sampleMetadataExcluded)):
				assert_frame_equal(result.sampleMetadataExcluded[i], expected.sampleMetadataExcluded[i])
			self.assertEqual(len(result.featureMetadataExcluded), len(expected.featureMetadataExcluded))
			for j in range(len(result.featureMetadataExcluded)):
				assert_frame_equal(result.featureMetadataExcluded[j], expected.featureMetadataExcluded[j])
			self.assertEqual(len(result.intensityDataExcluded), len(expected.intensityDataExcluded))
			for k in range(len(result.intensityDataExcluded)):
				numpy.testing.assert_array_equal(result.intensityDataExcluded[k], expected.intensityDataExcluded[k])
			self.assertEqual(len(result.expectedConcentrationExcluded), len(expected.expectedConcentrationExcluded))
			for l in range(len(result.expectedConcentrationExcluded)):
				assert_frame_equal(result.expectedConcentrationExcluded[l], expected.expectedConcentrationExcluded[l])
			self.assertEqual(result.excludedFlag, expected.excludedFlag)

		with self.subTest(msg='Check ApplyLLOQ Noise Filled with onlyLLOQ=False, responseReference=\'UnitTest_targeted_file_006\', previous exclusion'):
			# Expected
			expected = copy.deepcopy(self.targetedIn)
			# Exclusions
			expected.sampleMetadataExcluded.append(expected.sampleMetadata)
			expected.featureMetadataExcluded.append(expected.featureMetadata.iloc[[1, 2, 3, 4, 5], :])
			expected.intensityDataExcluded.append(expected._intensityData[:, [1, 2, 3, 4, 5]])
			expected.expectedConcentrationExcluded.append(expected.expectedConcentration.iloc[:, [1, 2, 3, 4, 5]])
			expected.excludedFlag.append('Features')
			# filtered values and reordered columns
			expected._intensityData = numpy.array([[6.5, 1.1832159566199232, 5., 50.], [500., 500., 500., 500.], [numpy.inf, numpy.inf, numpy.inf, 5000.]])
			expected.featureMetadata = expected.featureMetadata.iloc[[6, 7, 8, 0], :]
			expected.featureMetadata.reset_index(drop=True, inplace=True)
			expected.featureMetadata['noiseConcentration'] = [6.5, 1.1832159566199232, 5., numpy.nan]
			expected.featureMetadata['responseFactor'] = [1.4, 1.4, 1.4, numpy.nan]
			expected.expectedConcentration = expected.expectedConcentration.iloc[:, [6, 7, 8, 0]]
			expected.calibration['calibIntensityData'] = expected.calibration['calibIntensityData'][:, [6, 7, 8, 0]]
			expected.calibration['calibFeatureMetadata'] = expected.calibration['calibFeatureMetadata'].iloc[[6, 7, 8, 0], :]
			expected.calibration['calibFeatureMetadata'].reset_index(drop=True, inplace=True)
			expected.calibration['calibExpectedConcentration'] = expected.calibration['calibExpectedConcentration'].iloc[:, [6, 7, 8, 0]]
			expected.calibration['calibPeakInfo']['peakArea'] = expected.calibration['calibPeakInfo']['peakArea'].iloc[:, [6, 7, 8, 0]]
			expected.calibration['calibPeakInfo']['peakResponse'] = expected.calibration['calibPeakInfo']['peakResponse'].iloc[:, [6, 7, 8, 0]]
			expected.calibration['calibPeakInfo']['peakConcentrationDeviation'] = expected.calibration['calibPeakInfo']['peakConcentrationDeviation'].iloc[:, [6, 7, 8, 0]]
			expected.calibration['calibPeakInfo']['peakIntegrationFlag'] = expected.calibration['calibPeakInfo']['peakIntegrationFlag'].iloc[:,[6, 7, 8, 0]]
			expected.calibration['calibPeakInfo']['peakRT'] = expected.calibration['calibPeakInfo']['peakRT'].iloc[:, [6, 7, 8, 0]]
			# Result
			result = copy.deepcopy(self.targetedIn)
			result._targetLynxApplyLimitsOfQuantificationNoiseFilled(onlyLLOQ=False, responseReference='UnitTest_targeted_file_006')

			# Class
			self.assertEqual(type(result), type(expected))
			# sampleMetadata
			assert_frame_equal(result.sampleMetadata, expected.sampleMetadata)
			# featureMetadata
			assert_frame_equal(result.featureMetadata.reindex(sorted(result.featureMetadata), axis=1), expected.featureMetadata.reindex(sorted(expected.featureMetadata), axis=1))
			# intensityData
			numpy.testing.assert_array_equal(result._intensityData, expected._intensityData)
			# expectedConcentration
			assert_frame_equal(result.expectedConcentration, expected.expectedConcentration)
			# Calibration
			assert_frame_equal(result.calibration['calibSampleMetadata'], expected.calibration['calibSampleMetadata'])
			assert_frame_equal(result.calibration['calibFeatureMetadata'], expected.calibration['calibFeatureMetadata'])
			numpy.testing.assert_array_equal(result.calibration['calibIntensityData'], expected.calibration['calibIntensityData'])
			assert_frame_equal(result.calibration['calibExpectedConcentration'], expected.calibration['calibExpectedConcentration'])
			assert_frame_equal(result.calibration['calibPeakInfo']['peakArea'], expected.calibration['calibPeakInfo']['peakArea'])
			assert_frame_equal(result.calibration['calibPeakInfo']['peakResponse'], expected.calibration['calibPeakInfo']['peakResponse'])
			assert_frame_equal(result.calibration['calibPeakInfo']['peakConcentrationDeviation'], expected.calibration['calibPeakInfo']['peakConcentrationDeviation'])
			assert_frame_equal(result.calibration['calibPeakInfo']['peakIntegrationFlag'], expected.calibration['calibPeakInfo']['peakIntegrationFlag'])
			assert_frame_equal(result.calibration['calibPeakInfo']['peakRT'], expected.calibration['calibPeakInfo']['peakRT'])
			# Exclusions
			self.assertEqual(len(result.sampleMetadataExcluded), len(expected.sampleMetadataExcluded))
			for i in range(len(result.sampleMetadataExcluded)):
				assert_frame_equal(result.sampleMetadataExcluded[i], expected.sampleMetadataExcluded[i])
			self.assertEqual(len(result.featureMetadataExcluded), len(expected.featureMetadataExcluded))
			for j in range(len(result.featureMetadataExcluded)):
				assert_frame_equal(result.featureMetadataExcluded[j], expected.featureMetadataExcluded[j])
			self.assertEqual(len(result.intensityDataExcluded), len(expected.intensityDataExcluded))
			for k in range(len(result.intensityDataExcluded)):
				numpy.testing.assert_array_equal(result.intensityDataExcluded[k], expected.intensityDataExcluded[k])
			self.assertEqual(len(result.expectedConcentrationExcluded), len(expected.expectedConcentrationExcluded))
			for l in range(len(result.expectedConcentrationExcluded)):
				assert_frame_equal(result.expectedConcentrationExcluded[l], expected.expectedConcentrationExcluded[l])
			self.assertEqual(result.excludedFlag, expected.excludedFlag)

		with self.subTest(msg='Check ApplyLLOQ Noise Filled with onlyLLOQ=False, responseReference=[\'UnitTest_targeted_file_005\',\'UnitTest_targeted_file_006\',\'UnitTest_targeted_file_005\'], previous exclusion'):
			# Expected
			expected = copy.deepcopy(self.targetedIn)
			# Exclusions
			expected.sampleMetadataExcluded.append(expected.sampleMetadata)
			expected.featureMetadataExcluded.append(expected.featureMetadata.iloc[[1, 2, 3, 4, 5], :])
			expected.intensityDataExcluded.append(expected._intensityData[:, [1, 2, 3, 4, 5]])
			expected.expectedConcentrationExcluded.append(expected.expectedConcentration.iloc[:, [1, 2, 3, 4, 5]])
			expected.excludedFlag.append('Features')
			# filtered values and reordered columns
			expected._intensityData = numpy.array([[10., 1.1832159566199232, 5., 50.], [500., 500., 500., 500.], [numpy.inf, numpy.inf, numpy.inf, 5000.]])
			expected.featureMetadata = expected.featureMetadata.iloc[[6, 7, 8, 0], :]
			expected.featureMetadata.reset_index(drop=True, inplace=True)
			expected.featureMetadata['noiseConcentration'] = [10., 1.1832159566199232, 5., numpy.nan]
			expected.featureMetadata['responseFactor'] = [2.1, 1.4, 2.1, numpy.nan]
			expected.expectedConcentration = expected.expectedConcentration.iloc[:, [6, 7, 8, 0]]
			expected.calibration['calibIntensityData'] = expected.calibration['calibIntensityData'][:, [6, 7, 8, 0]]
			expected.calibration['calibFeatureMetadata'] = expected.calibration['calibFeatureMetadata'].iloc[[6, 7, 8, 0], :]
			expected.calibration['calibFeatureMetadata'].reset_index(drop=True, inplace=True)
			expected.calibration['calibExpectedConcentration'] = expected.calibration['calibExpectedConcentration'].iloc[:, [6, 7, 8, 0]]
			expected.calibration['calibPeakInfo']['peakArea'] = expected.calibration['calibPeakInfo']['peakArea'].iloc[:, [6, 7, 8, 0]]
			expected.calibration['calibPeakInfo']['peakResponse'] = expected.calibration['calibPeakInfo']['peakResponse'].iloc[:, [6, 7, 8, 0]]
			expected.calibration['calibPeakInfo']['peakConcentrationDeviation'] = expected.calibration['calibPeakInfo']['peakConcentrationDeviation'].iloc[:, [6, 7, 8, 0]]
			expected.calibration['calibPeakInfo']['peakIntegrationFlag'] = expected.calibration['calibPeakInfo']['peakIntegrationFlag'].iloc[:, [6, 7, 8, 0]]
			expected.calibration['calibPeakInfo']['peakRT'] = expected.calibration['calibPeakInfo']['peakRT'].iloc[:, [6, 7, 8, 0]]
			# Result
			result = copy.deepcopy(self.targetedIn)
			with warnings.catch_warnings():
				warnings.simplefilter("ignore")
				result._targetLynxApplyLimitsOfQuantificationNoiseFilled(onlyLLOQ=False, responseReference=['UnitTest_targeted_file_005','UnitTest_targeted_file_006','UnitTest_targeted_file_005'])

			# Class
			self.assertEqual(type(result), type(expected))
			# sampleMetadata
			assert_frame_equal(result.sampleMetadata, expected.sampleMetadata)
			# featureMetadata
			assert_frame_equal(result.featureMetadata.reindex(sorted(result.featureMetadata), axis=1), expected.featureMetadata.reindex(sorted(expected.featureMetadata), axis=1))
			# intensityData
			numpy.testing.assert_array_equal(result._intensityData, expected._intensityData)
			# expectedConcentration
			assert_frame_equal(result.expectedConcentration, expected.expectedConcentration)
			# Calibration
			assert_frame_equal(result.calibration['calibSampleMetadata'], expected.calibration['calibSampleMetadata'])
			assert_frame_equal(result.calibration['calibFeatureMetadata'], expected.calibration['calibFeatureMetadata'])
			numpy.testing.assert_array_equal(result.calibration['calibIntensityData'], expected.calibration['calibIntensityData'])
			assert_frame_equal(result.calibration['calibExpectedConcentration'], expected.calibration['calibExpectedConcentration'])
			assert_frame_equal(result.calibration['calibPeakInfo']['peakArea'], expected.calibration['calibPeakInfo']['peakArea'])
			assert_frame_equal(result.calibration['calibPeakInfo']['peakResponse'], expected.calibration['calibPeakInfo']['peakResponse'])
			assert_frame_equal(result.calibration['calibPeakInfo']['peakConcentrationDeviation'], expected.calibration['calibPeakInfo']['peakConcentrationDeviation'])
			assert_frame_equal(result.calibration['calibPeakInfo']['peakIntegrationFlag'], expected.calibration['calibPeakInfo']['peakIntegrationFlag'])
			assert_frame_equal(result.calibration['calibPeakInfo']['peakRT'], expected.calibration['calibPeakInfo']['peakRT'])
			# Exclusions
			self.assertEqual(len(result.sampleMetadataExcluded), len(expected.sampleMetadataExcluded))
			for i in range(len(result.sampleMetadataExcluded)):
				assert_frame_equal(result.sampleMetadataExcluded[i], expected.sampleMetadataExcluded[i])
			self.assertEqual(len(result.featureMetadataExcluded), len(expected.featureMetadataExcluded))
			for j in range(len(result.featureMetadataExcluded)):
				assert_frame_equal(result.featureMetadataExcluded[j], expected.featureMetadataExcluded[j])
			self.assertEqual(len(result.intensityDataExcluded), len(expected.intensityDataExcluded))
			for k in range(len(result.intensityDataExcluded)):
				numpy.testing.assert_array_equal(result.intensityDataExcluded[k], expected.intensityDataExcluded[k])
			self.assertEqual(len(result.expectedConcentrationExcluded), len(expected.expectedConcentrationExcluded))
			for l in range(len(result.expectedConcentrationExcluded)):
				assert_frame_equal(result.expectedConcentrationExcluded[l], expected.expectedConcentrationExcluded[l])
			self.assertEqual(result.excludedFlag, expected.excludedFlag)


	@unittest.mock.patch('sys.stdout', new_callable=io.StringIO)
	def test_targetlynxlimitsofquantificationnoisefilled_raise(self, mock_stdout):

		with self.subTest(msg='Checking Warning if responseReference was not provided'):
			triggerWarning = copy.deepcopy(self.targetedIn)
			with warnings.catch_warnings(record=True) as w:
				# Cause all warnings to always be triggered.
				warnings.simplefilter("always")
				# warning
				triggerWarning._targetLynxApplyLimitsOfQuantificationNoiseFilled(onlyLLOQ=False, responseReference=None)
				# check
				assert len(w) == 1
				assert issubclass(w[-1].category, UserWarning)
				assert "No responseReference provided, sample in the middle of" in str(w[-1].message)

		with self.subTest(msg='Checking AttributeError if \'LLOQ\' missing'):
			failLLOQ = copy.deepcopy(self.targetedIn)
			failLLOQ.featureMetadata = failLLOQ.featureMetadata.drop(['LLOQ'], axis=1)
			self.assertRaises(AttributeError, lambda: failLLOQ._targetLynxApplyLimitsOfQuantificationNoiseFilled(onlyLLOQ=False))

		with self.subTest(msg='Checking AttributeError if \'ULOQ\' missing'):
			failULOQ = copy.deepcopy(self.targetedIn)
			failULOQ.featureMetadata = failULOQ.featureMetadata.drop(['ULOQ'], axis=1)
			self.assertRaises(AttributeError, lambda: failULOQ._applyLimitsOfQuantification(onlyLLOQ=False))

		with self.subTest(msg='Checking AttributeError if \'calibrationEquation\' missing'):
			failCalibEquation = copy.deepcopy(self.targetedIn)
			failCalibEquation.featureMetadata = failCalibEquation.featureMetadata.drop(['calibrationEquation'], axis=1)
			self.assertRaises(AttributeError, lambda: failCalibEquation._targetLynxApplyLimitsOfQuantificationNoiseFilled(onlyLLOQ=False))

		with self.subTest(msg='Checking AttributeError if \'unitCorrectionFactor\' missing'):
			failUnitCorrectionFactor = copy.deepcopy(self.targetedIn)
			failUnitCorrectionFactor.featureMetadata = failUnitCorrectionFactor.featureMetadata.drop(['unitCorrectionFactor'], axis=1)
			self.assertRaises(AttributeError, lambda: failUnitCorrectionFactor._targetLynxApplyLimitsOfQuantificationNoiseFilled(onlyLLOQ=False))

		with self.subTest(msg='Checking AttributeError if \'Noise (area)\' missing'):
			failNoiseArea = copy.deepcopy(self.targetedIn)
			failNoiseArea.featureMetadata = failNoiseArea.featureMetadata.drop(['Noise (area)'], axis=1)
			self.assertRaises(AttributeError, lambda: failNoiseArea._targetLynxApplyLimitsOfQuantificationNoiseFilled(onlyLLOQ=False))

		with self.subTest(msg='Checking AttributeError if \'a\' missing'):
			failA = copy.deepcopy(self.targetedIn)
			failA.featureMetadata = failA.featureMetadata.drop(['a'], axis=1)
			self.assertRaises(AttributeError, lambda: failA._targetLynxApplyLimitsOfQuantificationNoiseFilled(onlyLLOQ=False))

		with self.subTest(msg='Checking AttributeError if \'b\' missing'):
			failB = copy.deepcopy(self.targetedIn)
			failB.featureMetadata = failB.featureMetadata.drop(['b'], axis=1)
			self.assertRaises(AttributeError, lambda: failB._targetLynxApplyLimitsOfQuantificationNoiseFilled(onlyLLOQ=False))

		with self.subTest(msg='Checking AttributeError if calibration[\'calibPeakInfo\'] missing'):
			failCalibPeakInfo = copy.deepcopy(self.targetedIn)
			del failCalibPeakInfo.calibration['calibPeakInfo']
			self.assertRaises(AttributeError, lambda: failCalibPeakInfo._targetLynxApplyLimitsOfQuantificationNoiseFilled(onlyLLOQ=False))

		with self.subTest(msg='Checking ValueError if calibration[\'calibPeakInfo\'][\'peakArea\'] has wrong number of samples or features'):
			# wrong number of samples
			failCalibPeakInfoAreaS = copy.deepcopy(self.targetedIn)
			failCalibPeakInfoAreaS.calibration['calibPeakInfo']['peakArea'] = failCalibPeakInfoAreaS.calibration['calibPeakInfo']['peakArea'].iloc[[0,1],:]
			self.assertRaises(ValueError, lambda: failCalibPeakInfoAreaS._targetLynxApplyLimitsOfQuantificationNoiseFilled(onlyLLOQ=False))

			# wrong number of features
			failCalibPeakInfoAreaF = copy.deepcopy(self.targetedIn)
			failCalibPeakInfoAreaF.calibration['calibPeakInfo']['peakArea'] = failCalibPeakInfoAreaF.calibration['calibPeakInfo']['peakArea'].iloc[:,[0,1]]
			self.assertRaises(ValueError, lambda: failCalibPeakInfoAreaF._targetLynxApplyLimitsOfQuantificationNoiseFilled(onlyLLOQ=False))

		with self.subTest(msg='Checking ValueError if calibration[\'calibPeakInfo\'][\'peakResponse\'] has wrong number of samples or features'):
			# wrong number of samples
			failCalibPeakInfoResponseS = copy.deepcopy(self.targetedIn)
			failCalibPeakInfoResponseS.calibration['calibPeakInfo']['peakResponse'] = failCalibPeakInfoResponseS.calibration['calibPeakInfo']['peakResponse'].iloc[[0,1],:]
			self.assertRaises(ValueError, lambda: failCalibPeakInfoResponseS._targetLynxApplyLimitsOfQuantificationNoiseFilled(onlyLLOQ=False))

			# wrong number of features
			failCalibPeakInfoResponseF = copy.deepcopy(self.targetedIn)
			failCalibPeakInfoResponseF.calibration['calibPeakInfo']['peakResponse'] = failCalibPeakInfoResponseF.calibration['calibPeakInfo']['peakResponse'].iloc[:,[0,1]]
			self.assertRaises(ValueError, lambda: failCalibPeakInfoResponseF._targetLynxApplyLimitsOfQuantificationNoiseFilled(onlyLLOQ=False))

		with self.subTest(msg='Checking ValueError if responseReference is not recognised or wrong length'):
			# unknown name
			self.assertRaises(ValueError, lambda: self.targetedIn._targetLynxApplyLimitsOfQuantificationNoiseFilled(onlyLLOQ=False, responseReference='Random_file_name'))
			# wrong list length
			self.assertRaises(ValueError, lambda: self.targetedIn._targetLynxApplyLimitsOfQuantificationNoiseFilled(onlyLLOQ=False, responseReference=['Random_file_name1', 'Random_file_name2']))
			# unknown name in list
			self.assertRaises(ValueError, lambda: self.targetedIn._targetLynxApplyLimitsOfQuantificationNoiseFilled(onlyLLOQ=False, responseReference=['Random_file_name1','UnitTest_targeted_file_005','UnitTest_targeted_file_006']))
			# unknown type
			self.assertRaises(ValueError, lambda: self.targetedIn._targetLynxApplyLimitsOfQuantificationNoiseFilled(onlyLLOQ=False, responseReference=dict()))

		with self.subTest(msg='Checking ValueError if calibrationEquation fails'):
			with warnings.catch_warnings():
				warnings.simplefilter("ignore", UserWarning)
				failCalibrationEquation = copy.deepcopy(self.targetedIn)
				failCalibrationEquation.featureMetadata['calibrationEquation'] = ['','','','','','','((area * responseFactor)-b)/a', '10**((numpy.log10(area * responseFactor)-b)/a)', 'myVariable/a']
				self.assertRaises(ValueError, lambda: failCalibrationEquation._targetLynxApplyLimitsOfQuantificationNoiseFilled(onlyLLOQ=False, responseReference=None))


class test_targeteddataset_full_targetlynx_load(unittest.TestCase):
	"""
	Test all steps of loadTargetLynxDataset and parameters input: read and match TL files, filter calib (set SampleTypeToProcess), keep IS, noiseFilled or not (set onlyLLOQ, responseReference), keepPeakInfo or not, keepExcluded or not.
	Underlying function tested independently
	"""
	def setUp(self):
		# 3 Features, 2 samples in TL .xml
		# 3 Features in SOP/Calib

		# Calibration report .csv
		self.calibrationReport = pandas.DataFrame({'Compound': ['Feature1IS', 'Feature2', 'Feature3'], 'TargetLynx ID': [1, 2, 3], 'Cpd Info': ['uM', 'fg/uL', 'fg/uL'], 'Noise (area)': [numpy.nan, 14.7, 25.6], 'LLOQ': [numpy.nan, 300, 25.],'ULOQ': [numpy.nan, 2500, 350.], 'a': [numpy.nan, 1.04095,1.19658], 'b': [numpy.nan, -1.78935, -1.5875],'r': [0.997931, 0.999556, 0.999], 'r2': [0.995866, 0.999113, 0.98995],'another column': [numpy.nan, 'something','something else']})
		# SOP JSON
		self.SOP = {'compoundID': ['1', '2','3'],'compoundName': ['Feature1IS', 'Feature2','Feature3'], 'IS': ['True', 'False','False'], 'unitFinal': ['noUnit', 'pg/uL','pg/uL'],'unitCorrectionFactor': [1.,1.,1.], 'calibrationMethod': ['noIS','backcalculatedIS','backcalculatedIS'], 'calibrationEquation': ['','((area * responseFactor)-b)/a','((area * responseFactor)-b)/a'], 'quantificationType': ['IS','QuantOwnLabeledAnalogue','QuantOwnLabeledAnalogue'], 'chromatography': 'R','ionisation': 'NEG', 'methodName': 'UnitTest', 'externalID': ['extID1','extID2'], 'extID1': ['F1','F2','F3'],'extID2': ['ID1','ID2','ID3'], 'sampleMetadataNotExported': ['test not exported sampleMetadata'], 'featureMetadataNotExported': ['test not exported featureMetadata']}

		# Expected TargetedDataset
		self.expected = dict()
		self.expected['sampleMetadata'] = pandas.DataFrame({'Sample File Name': ['UnitTest4_targeted_file_002'],'TargetLynx Sample ID': [2], 'MassLynx Row ID': [2],'Sample Name': ['Study Sample 1'], 'Sample Type': ['Analyte'],'Acqu Date': ['11-Sep-16'], 'Acqu Time': ['09:23:02'], 'Vial': ['1:A,2'],'Instrument': ['XEVO-TQS#UnitTest'], 'Acquired Time': [pandas.Timestamp(datetime(2016, 9, 11, 9, 23, 2))], 'Run Order': [1],'Batch': [1],'AssayRole': [numpy.nan],'SampleType': [numpy.nan],'Dilution': [numpy.nan],'Correction Batch': [numpy.nan],'Sample ID': [numpy.nan],'Exclusion Details': [numpy.nan]})
		self.expected['sampleMetadata']['Metadata Available'] = False
		self.expected['sampleMetadata']['Sample Base Name'] = self.expected['sampleMetadata']['Sample File Name']
		self.expected['sampleMetadata']['Acquired Time'] = pandas.to_datetime(self.expected['sampleMetadata'].loc[:, 'Acquired Time'])
		self.expected['featureMetadata'] = pandas.DataFrame({'Feature Name': ['Feature2','Feature3'], 'TargetLynx Feature ID':[2, 3], 'calibrationEquation':['((area * responseFactor)-b)/a', '((area * responseFactor)-b)/a'],'calibrationMethod': [CalibrationMethod.backcalculatedIS, CalibrationMethod.backcalculatedIS], 'quantificationType': [QuantificationType.QuantOwnLabeledAnalogue, QuantificationType.QuantOwnLabeledAnalogue], 'unitCorrectionFactor': [1.,1.], 'Unit': ['pg/uL','pg/uL'], 'Cpd Info': ['fg/uL','fg/uL'], 'LLOQ': [300., 25.], 'Noise (area)': [14.7, 25.6], 'ULOQ': [2500., 350.], 'a': [1.04095, 1.19658], 'another column': ['something', 'something else'],'b': [-1.78935, -1.5875], 'r': [0.999556, 0.999], 'r2': [0.999113, 0.98995], 'extID1': ['F2','F3'],'extID2': ['ID2','ID3']})
		#self.expected['intensityData'] = numpy.array([[-numpy.inf,  359.219531]]) # value for onlyLLOQ=True
		self.expected['intensityData'] = numpy.array([[-numpy.inf, numpy.inf]])
		self.expected['expectedConcentration'] = pandas.DataFrame(numpy.array([[numpy.nan, numpy.nan]]), columns=self.expected['featureMetadata']['Feature Name'].values)
		# Calibration
		self.expected['calibSampleMetadata'] = pandas.DataFrame({'Sample File Name': ['UnitTest4_targeted_file_001'], 'TargetLynx Sample ID': [1], 'MassLynx Row ID': [1], 'Sample Name': ['Calibration 1'], 'Sample Type': ['Standard'], 'Acqu Date': ['11-Sep-16'], 'Acqu Time': ['02:14:32'], 'Vial': ['1:A,1'], 'Instrument': ['XEVO-TQS#UnitTest'], 'Calibrant': [True], 'Study Sample': [False], 'Blank': [False], 'QC': [False], 'Other': [False], 'Acquired Time': [datetime(2016, 9, 11, 2, 14, 32)], 'Run Order': [0], 'Batch': [1], 'AssayRole': [numpy.nan], 'SampleType': [numpy.nan], 'Dilution': [numpy.nan], 'Correction Batch': [numpy.nan], 'Sample ID': [numpy.nan], 'Exclusion Details': [numpy.nan]})
		self.expected['calibSampleMetadata']['Sample Base Name'] = self.expected['calibSampleMetadata']['Sample File Name']
		self.expected['calibSampleMetadata']['Acquired Time'] = self.expected['calibSampleMetadata']['Acquired Time']
		self.expected['calibSampleMetadata']['Metadata Available'] = False
		self.expected['calibFeatureMetadata'] = copy.deepcopy(self.expected['featureMetadata'])
		self.expected['calibFeatureMetadata']['IS'] = numpy.array([False, False])
		self.expected['calibFeatureMetadata']['TargetLynx IS ID'] = ['1','1']
		self.expected['calibIntensityData'] = numpy.array([[48.7244571,  48.76854933]])
		self.expected['calibExpectedConcentration'] = pandas.DataFrame(numpy.array([[50., 50.]]), columns=self.expected['calibFeatureMetadata']['Feature Name'].values)
		self.expected['calibPeakResponse'] = pandas.DataFrame(numpy.array([[5.4055825074, 0.2940372929]]), columns=self.expected['calibFeatureMetadata']['Feature Name'].values)
		self.expected['calibPeakArea'] = pandas.DataFrame(numpy.array([[14423.905, 784.59]]), columns=self.expected['calibFeatureMetadata']['Feature Name'].values)
		self.expected['calibPeakConcentrationDeviation'] = pandas.DataFrame(numpy.array([[-2.5510858032, -2.4629013366]]), columns=self.expected['calibFeatureMetadata']['Feature Name'].values)
		self.expected['calibPeakIntegrationFlag'] = pandas.DataFrame({'Feature2': ['bb'], 'Feature3': ['MM']})
		self.expected['calibPeakRT'] = pandas.DataFrame(numpy.array([[11.4921998978, 11.63409996]]), columns=self.expected['calibFeatureMetadata']['Feature Name'].values)
		# Excluded
		self.expected['sampleMetadataExcluded'] = [copy.deepcopy(self.expected['sampleMetadata'])]#[['Sample File Name', 'Sample Base Name', 'TargetLynx Sample ID', 'MassLynx Row ID', 'Sample Name', 'Sample Type', 'Acqu Date', 'Acqu Time', 'Vial', 'Instrument', ]]]
		#self.expected['sampleMetadataExcluded'][0]['Metadata Available'] = False
		self.expected['featureMetadataExcluded'] = [pandas.DataFrame({'Feature Name': ['Feature1IS'], 'TargetLynx Feature ID':[1], 'TargetLynx IS ID':[''], 'IS': True, 'calibrationEquation':[''],'calibrationMethod': [CalibrationMethod.noIS], 'quantificationType': [QuantificationType.IS], 'unitCorrectionFactor': [1.], 'Unit': ['noUnit'], 'Cpd Info': ['uM'], 'LLOQ': [numpy.nan], 'Noise (area)': [numpy.nan], 'ULOQ': [numpy.nan], 'a': [numpy.nan], 'b': [numpy.nan], 'r': [0.997931], 'r2': [0.995866], 'extID1': ['F1'], 'extID2': ['ID1']})]
		self.expected['featureMetadataExcluded'][0]['another column'] =  numpy.array([numpy.nan], dtype=object)
		self.expected['intensityDataExcluded'] = [numpy.array([[20.60696312]])]
		self.expected['expectedConcentrationExcluded'] = [pandas.DataFrame(numpy.array([[60.]]), columns=['Feature1IS'])]
		self.expected['excludedFlag'] = ['Features']
		# Attributes
		tmpDataset = nPYc.TargetedDataset('', fileType='empty')
		self.expected['Attributes'] = {'calibrationReportPath': '', 'chromatography': 'R', 'ionisation': 'NEG', 'methodName': 'UnitTest', 'dpi': tmpDataset.Attributes['dpi'], 'figureFormat': tmpDataset.Attributes['figureFormat'], 'figureSize': tmpDataset.Attributes['figureSize'], 'histBins': tmpDataset.Attributes['histBins'], 'noFiles': tmpDataset.Attributes['noFiles'], 'quantiles': tmpDataset.Attributes['quantiles'], 'externalID': ['extID1','extID2'], 'sampleMetadataNotExported': ['test not exported sampleMetadata'], 'featureMetadataNotExported': ['test not exported featureMetadata'], "analyticalMeasurements": {}, "excludeFromPlotting": [], "sampleTypeColours": {"StudySample": "b", "StudyPool": "g", "ExternalReference": "r", "MethodReference": "m", "ProceduralBlank": "c", "Other": "grey", "Study Sample": "b",
																																																																																																																																																																								"Study Reference": "g", "Long-Term Reference": "r", "Method Reference": "m", "Blank": "c", "Unspecified SampleType or AssayRole": "grey"}}


	@unittest.mock.patch('sys.stdout', new_callable=io.StringIO)
	def test_targeteddataset_loadtargetlynxdataset(self, mock_stdout):

		with self.subTest(msg='Basic import, keepIS=False, noiseFilled=False, keepPeakInfo=False, keepExcluded=False, no additional changes'):
			with tempfile.TemporaryDirectory() as tmpdirname:
				# Init
				# Create temp SOP file
				with open(os.path.join(tmpdirname, 'targetedSOP.json'), 'w') as outfile:
					json.dump(self.SOP, outfile)
				# Create temp .csv calibReport
				reportPath = os.path.join(tmpdirname, 'calibrationReport.csv')
				self.calibrationReport.to_csv(reportPath, index=False)
				XMLpath = os.path.join('..', '..', 'npc-standard-project', 'Derived_Data','UnitTest4_targeted.xml')
				expected = copy.deepcopy(self.expected)
				expected['Attributes']['calibrationReportPath'] = reportPath
				# Generate
				with warnings.catch_warnings():
					warnings.simplefilter("ignore", UserWarning)
					result = nPYc.TargetedDataset(XMLpath, fileType='TargetLynx', sop='targetedSOP', sopPath=tmpdirname, calibrationReportPath=reportPath, keepIS=False, noiseFilled=False, keepPeakInfo=False, keepExcluded=False)

				# Test
				assert_frame_equal(expected['sampleMetadata'].reindex(sorted(expected['sampleMetadata']), axis=1), result.sampleMetadata.reindex(sorted(result.sampleMetadata), axis=1))
				assert_frame_equal(expected['featureMetadata'].reindex(sorted(expected['featureMetadata']), axis=1), result.featureMetadata.reindex(sorted(result.featureMetadata), axis=1))
				numpy.testing.assert_array_almost_equal(expected['intensityData'], result._intensityData)
				assert_frame_equal(expected['expectedConcentration'].reindex(sorted(expected['expectedConcentration']), axis=1), result.expectedConcentration.reindex(sorted(result.expectedConcentration), axis=1))
				# Calibration
				assert_frame_equal(expected['calibSampleMetadata'].reindex(sorted(expected['calibSampleMetadata']), axis=1), result.calibration['calibSampleMetadata'].reindex(sorted(result.calibration['calibSampleMetadata']), axis=1))
				assert_frame_equal(expected['calibFeatureMetadata'].reindex(sorted(expected['calibFeatureMetadata']), axis=1), result.calibration['calibFeatureMetadata'].reindex(sorted(result.calibration['calibFeatureMetadata']), axis=1))
				numpy.testing.assert_array_almost_equal(expected['calibIntensityData'], result.calibration['calibIntensityData'])
				assert_frame_equal(expected['calibExpectedConcentration'].reindex(sorted(expected['calibExpectedConcentration']), axis=1), result.calibration['calibExpectedConcentration'].reindex(sorted(result.calibration['calibExpectedConcentration']), axis=1))
				# No peakInfo
				# No exclusions
				# Attributes
				self.assertEqual(len(expected['Attributes'].keys()), len(result.Attributes.keys()) - 1)
				for i in expected['Attributes']:
					self.assertEqual(expected['Attributes'][i], result.Attributes[i])

		with self.subTest(msg='Modify sampleTypeToProcess, trigger an error, keepIS=False, noiseFilled=False, keepPeakInfo=False, keepExcluded=False'):
			with tempfile.TemporaryDirectory() as tmpdirname:
				# Init
				# Create temp SOP file
				with open(os.path.join(tmpdirname, 'targetedSOP.json'), 'w') as outfile:
					json.dump(self.SOP, outfile)
				# Create temp .csv calibReport
				reportPath = os.path.join(tmpdirname, 'calibrationReport.csv')
				self.calibrationReport.to_csv(reportPath, index=False)
				XMLpath = os.path.join('..', '..', 'npc-standard-project', 'Derived_Data','UnitTest4_targeted.xml')

				# Trigger an error when sampleTypeToProcess is wrongly altered
				with warnings.catch_warnings():
					warnings.simplefilter('ignore', UserWarning)
					self.assertRaises(ValueError, lambda: nPYc.TargetedDataset(XMLpath, fileType='TargetLynx', sop='targetedSOP', sopPath=tmpdirname, calibrationReportPath=reportPath, sampleTypeToProcess=['Study Sample', 'Unknown Type'], keepIS=False, noiseFilled=False, keepPeakInfo=False, keepExcluded=False))

		with self.subTest(msg='Change keepIS, keepIS=True, noiseFilled=False, keepPeakInfo=False, keepExcluded=False, no additional changes'):
			# IS feature is kept, however doesn't have LLOQ/ULOQ so is excluded in applyLLOQ
			with tempfile.TemporaryDirectory() as tmpdirname:
				# Init
				# Create temp SOP file
				with open(os.path.join(tmpdirname, 'targetedSOP.json'), 'w') as outfile:
					json.dump(self.SOP, outfile)
				# Create temp .csv calibReport
				reportPath = os.path.join(tmpdirname, 'calibrationReport.csv')
				self.calibrationReport.to_csv(reportPath, index=False)
				XMLpath = os.path.join('..', '..', 'npc-standard-project', 'Derived_Data','UnitTest4_targeted.xml')
				expected = copy.deepcopy(self.expected)
				expected['Attributes']['calibrationReportPath'] = reportPath
				expected['featureMetadata']['IS'] = numpy.array([False, False])
				expected['featureMetadata']['TargetLynx IS ID'] = ['1', '1']
				# Generate
				with warnings.catch_warnings():
					warnings.simplefilter("ignore", UserWarning)
					result = nPYc.TargetedDataset(XMLpath, fileType='TargetLynx', sop='targetedSOP', sopPath=tmpdirname, calibrationReportPath=reportPath, keepIS=True, noiseFilled=False, keepPeakInfo=False, keepExcluded=False)

				# Test
				assert_frame_equal(expected['sampleMetadata'].reindex(sorted(expected['sampleMetadata']), axis=1), result.sampleMetadata.reindex(sorted(result.sampleMetadata), axis=1))
				assert_frame_equal(expected['featureMetadata'].reindex(sorted(expected['featureMetadata']), axis=1), result.featureMetadata.reindex(sorted(result.featureMetadata), axis=1))
				numpy.testing.assert_array_almost_equal(expected['intensityData'], result._intensityData)
				assert_frame_equal(expected['expectedConcentration'].reindex(sorted(expected['expectedConcentration']), axis=1), result.expectedConcentration.reindex(sorted(result.expectedConcentration), axis=1))
				# Calibration
				assert_frame_equal(expected['calibSampleMetadata'].reindex(sorted(expected['calibSampleMetadata']), axis=1), result.calibration['calibSampleMetadata'].reindex(sorted(result.calibration['calibSampleMetadata']), axis=1))
				assert_frame_equal(expected['calibFeatureMetadata'].reindex(sorted(expected['calibFeatureMetadata']), axis=1), result.calibration['calibFeatureMetadata'].reindex(sorted(result.calibration['calibFeatureMetadata']), axis=1))
				numpy.testing.assert_array_almost_equal(expected['calibIntensityData'], result.calibration['calibIntensityData'])
				assert_frame_equal(expected['calibExpectedConcentration'].reindex(sorted(expected['calibExpectedConcentration']), axis=1), result.calibration['calibExpectedConcentration'].reindex(sorted(result.calibration['calibExpectedConcentration']), axis=1))
				# No peakInfo
				# No Exclusions
				# Attributes
				self.assertEqual(len(expected['Attributes'].keys()), len(result.Attributes.keys()) - 1)
				for i in expected['Attributes']:
					self.assertEqual(expected['Attributes'][i], result.Attributes[i])

		with self.subTest(msg='Modify onlyLLOQ, Basic import (noiseFilled=False) + onlyLLOQ=True, keepIS=False, noiseFilled=False, keepPeakInfo=False, keepExcluded=False, onlyLLOQ=False'):
			with tempfile.TemporaryDirectory() as tmpdirname:
				# Init
				# Create temp SOP file
				with open(os.path.join(tmpdirname, 'targetedSOP.json'), 'w') as outfile:
					json.dump(self.SOP, outfile)
				# Create temp .csv calibReport
				reportPath = os.path.join(tmpdirname, 'calibrationReport.csv')
				self.calibrationReport.to_csv(reportPath, index=False)
				XMLpath = os.path.join('..', '..', 'npc-standard-project', 'Derived_Data','UnitTest4_targeted.xml')
				expected = copy.deepcopy(self.expected)
				expected['Attributes']['calibrationReportPath'] = reportPath
				expected['intensityData'] = numpy.array([[-numpy.inf, 359.219531]])
				# Generate
				with warnings.catch_warnings():
					warnings.simplefilter("ignore", UserWarning)
					result = nPYc.TargetedDataset(XMLpath, fileType='TargetLynx', sop='targetedSOP', sopPath=tmpdirname, calibrationReportPath=reportPath, onlyLLOQ=True, keepIS=False, noiseFilled=False, keepPeakInfo=False, keepExcluded=False)

				# Test
				assert_frame_equal(expected['sampleMetadata'].reindex(sorted(expected['sampleMetadata']), axis=1), result.sampleMetadata.reindex(sorted(result.sampleMetadata), axis=1))
				assert_frame_equal(expected['featureMetadata'].reindex(sorted(expected['featureMetadata']), axis=1), result.featureMetadata.reindex(sorted(result.featureMetadata), axis=1))
				numpy.testing.assert_array_almost_equal(expected['intensityData'], result._intensityData)
				assert_frame_equal(expected['expectedConcentration'].reindex(sorted(expected['expectedConcentration']), axis=1), result.expectedConcentration.reindex(sorted(result.expectedConcentration), axis=1))
				# Calibration
				assert_frame_equal(expected['calibSampleMetadata'].reindex(sorted(expected['calibSampleMetadata']), axis=1), result.calibration['calibSampleMetadata'].reindex(sorted(result.calibration['calibSampleMetadata']), axis=1))
				assert_frame_equal(expected['calibFeatureMetadata'].reindex(sorted(expected['calibFeatureMetadata']), axis=1), result.calibration['calibFeatureMetadata'].reindex(sorted(result.calibration['calibFeatureMetadata']), axis=1))
				numpy.testing.assert_array_almost_equal(expected['calibIntensityData'], result.calibration['calibIntensityData'])
				assert_frame_equal(expected['calibExpectedConcentration'].reindex(sorted(expected['calibExpectedConcentration']), axis=1), result.calibration['calibExpectedConcentration'].reindex(sorted(result.calibration['calibExpectedConcentration']), axis=1))
				# No peakInfo
				# No exclusions
				# Attributes
				self.assertEqual(len(expected['Attributes'].keys()), len(result.Attributes.keys()) - 1)
				for i in expected['Attributes']:
					self.assertEqual(expected['Attributes'][i], result.Attributes[i])

		with self.subTest(msg='Change noiseFilled, keepIS=False, noiseFilled=True, keepPeakInfo=False, keepExcluded=False, no additional changes'):
			with tempfile.TemporaryDirectory() as tmpdirname:
				# Init
				# Create temp SOP file
				with open(os.path.join(tmpdirname, 'targetedSOP.json'), 'w') as outfile:
					json.dump(self.SOP, outfile)
				# Create temp .csv calibReport
				reportPath = os.path.join(tmpdirname, 'calibrationReport.csv')
				self.calibrationReport.to_csv(reportPath, index=False)
				XMLpath = os.path.join('..', '..', 'npc-standard-project', 'Derived_Data','UnitTest4_targeted.xml')
				expected = copy.deepcopy(self.expected)
				expected['Attributes']['calibrationReportPath'] = reportPath
				expected['featureMetadata']['responseFactor'] = [0.000374765537308, 0.000374765537287]
				expected['featureMetadata']['noiseConcentration'] = [1.724251, 1.334716]
				expected['intensityData'][0,0] = 1.7242509759339286
				# Generate
				with warnings.catch_warnings():
					warnings.simplefilter('ignore', UserWarning)
					result = nPYc.TargetedDataset(XMLpath, fileType='TargetLynx', sop='targetedSOP', sopPath=tmpdirname, calibrationReportPath=reportPath, keepIS=False, noiseFilled=True, keepPeakInfo=False, keepExcluded=False)

				# Test
				assert_frame_equal(expected['sampleMetadata'].reindex(sorted(expected['sampleMetadata']), axis=1), result.sampleMetadata.reindex(sorted(result.sampleMetadata), axis=1))
				assert_frame_equal(expected['featureMetadata'].reindex(sorted(expected['featureMetadata']), axis=1), result.featureMetadata.reindex(sorted(result.featureMetadata), axis=1))
				numpy.testing.assert_array_almost_equal(expected['intensityData'], result._intensityData)
				assert_frame_equal(expected['expectedConcentration'].reindex(sorted(expected['expectedConcentration']), axis=1), result.expectedConcentration.reindex(sorted(result.expectedConcentration), axis=1))
				# Calibration
				assert_frame_equal(expected['calibSampleMetadata'].reindex(sorted(expected['calibSampleMetadata']), axis=1), result.calibration['calibSampleMetadata'].reindex(sorted(result.calibration['calibSampleMetadata']), axis=1))
				assert_frame_equal(expected['calibFeatureMetadata'].reindex(sorted(expected['calibFeatureMetadata']), axis=1), result.calibration['calibFeatureMetadata'].reindex(sorted(result.calibration['calibFeatureMetadata']), axis=1))
				numpy.testing.assert_array_almost_equal(expected['calibIntensityData'], result.calibration['calibIntensityData'])
				assert_frame_equal(expected['calibExpectedConcentration'].reindex(sorted(expected['calibExpectedConcentration']), axis=1), result.calibration['calibExpectedConcentration'].reindex(sorted(result.calibration['calibExpectedConcentration']), axis=1))
				# No peakInfo
				# No Exclusions
				# Attributes
				self.assertEqual(len(expected['Attributes'].keys()), len(result.Attributes.keys()) - 1)
				for i in expected['Attributes']:
					self.assertEqual(expected['Attributes'][i], result.Attributes[i])

		with self.subTest(msg='Modify onlyLLOQ with noiseFilled, keepIS=False, noiseFilled=True, keepPeakInfo=False, keepExcluded=False, onlyLLOQ=False'):
			with tempfile.TemporaryDirectory() as tmpdirname:
				# Init
				# Create temp SOP file
				with open(os.path.join(tmpdirname, 'targetedSOP.json'), 'w') as outfile:
					json.dump(self.SOP, outfile)
				# Create temp .csv calibReport
				reportPath = os.path.join(tmpdirname, 'calibrationReport.csv')
				self.calibrationReport.to_csv(reportPath, index=False)
				XMLpath = os.path.join('..', '..', 'npc-standard-project', 'Derived_Data','UnitTest4_targeted.xml')
				expected = copy.deepcopy(self.expected)
				expected['Attributes']['calibrationReportPath'] = reportPath
				expected['featureMetadata']['responseFactor'] = [0.000374765537308, 0.000374765537287]
				expected['featureMetadata']['noiseConcentration'] = [1.724251, 1.334716]
				expected['intensityData'] = [[1.7242509759339286, numpy.inf]]
				# Generate
				with warnings.catch_warnings():
					warnings.simplefilter("ignore", UserWarning)
					result = nPYc.TargetedDataset(XMLpath, fileType='TargetLynx', sop='targetedSOP', sopPath=tmpdirname, calibrationReportPath=reportPath, onlyLLOQ=False, keepIS=False, noiseFilled=True, keepPeakInfo=False, keepExcluded=False)

				# Test
				assert_frame_equal(expected['sampleMetadata'].reindex(sorted(expected['sampleMetadata']), axis=1), result.sampleMetadata.reindex(sorted(result.sampleMetadata), axis=1))
				assert_frame_equal(expected['featureMetadata'].reindex(sorted(expected['featureMetadata']), axis=1), result.featureMetadata.reindex(sorted(result.featureMetadata), axis=1))
				numpy.testing.assert_array_almost_equal(expected['intensityData'], result._intensityData)
				assert_frame_equal(expected['expectedConcentration'].reindex(sorted(expected['expectedConcentration']), axis=1), result.expectedConcentration.reindex(sorted(result.expectedConcentration), axis=1))
				# Calibration
				assert_frame_equal(expected['calibSampleMetadata'].reindex(sorted(expected['calibSampleMetadata']), axis=1), result.calibration['calibSampleMetadata'].reindex(sorted(result.calibration['calibSampleMetadata']), axis=1))
				assert_frame_equal(expected['calibFeatureMetadata'].reindex(sorted(expected['calibFeatureMetadata']), axis=1), result.calibration['calibFeatureMetadata'].reindex(sorted(result.calibration['calibFeatureMetadata']), axis=1))
				numpy.testing.assert_array_almost_equal(expected['calibIntensityData'], result.calibration['calibIntensityData'])
				assert_frame_equal(expected['calibExpectedConcentration'].reindex(sorted(expected['calibExpectedConcentration']), axis=1), result.calibration['calibExpectedConcentration'].reindex(sorted(result.calibration['calibExpectedConcentration']), axis=1))
				# No peakInfo
				# No Exclusions
				# Attributes
				self.assertEqual(len(expected['Attributes'].keys()), len(result.Attributes.keys()) - 1)
				for i in expected['Attributes']:
					self.assertEqual(expected['Attributes'][i], result.Attributes[i])

		with self.subTest(msg='Modify responseReference with noiseFilled, trigger an error, keepIS=False, noiseFilled=True, keepPeakInfo=False, keepExcluded=False'):
			with tempfile.TemporaryDirectory() as tmpdirname:
				# Init
				# Create temp SOP file
				with open(os.path.join(tmpdirname, 'targetedSOP.json'), 'w') as outfile:
					json.dump(self.SOP, outfile)
				# Create temp .csv calibReport
				reportPath = os.path.join(tmpdirname, 'calibrationReport.csv')
				self.calibrationReport.to_csv(reportPath, index=False)
				XMLpath = os.path.join('..', '..', 'npc-standard-project', 'Derived_Data','UnitTest4_targeted.xml')

				# Trigger an error when responseReference is wrongly altered
				with warnings.catch_warnings():
					warnings.simplefilter("ignore", UserWarning)
					self.assertRaises(ValueError, lambda: nPYc.TargetedDataset(XMLpath, fileType='TargetLynx', sop='targetedSOP', sopPath=tmpdirname, calibrationReportPath=reportPath, responseReference=5., keepIS=False, noiseFilled=True, keepPeakInfo=False, keepExcluded=False))

		with self.subTest(msg='Change keepPeakInfo, keepIS=False, noiseFilled=False, keepPeakInfo=True, keepExcluded=False, no additional changes'):
			with tempfile.TemporaryDirectory() as tmpdirname:
				# Init
				# Create temp SOP file
				with open(os.path.join(tmpdirname, 'targetedSOP.json'), 'w') as outfile:
					json.dump(self.SOP, outfile)
				# Create temp .csv calibReport
				reportPath = os.path.join(tmpdirname, 'calibrationReport.csv')
				self.calibrationReport.to_csv(reportPath, index=False)
				XMLpath = os.path.join('..', '..', 'npc-standard-project', 'Derived_Data','UnitTest4_targeted.xml')
				expected = copy.deepcopy(self.expected)
				expected['Attributes']['calibrationReportPath'] = reportPath
				expected['peakArea'] = pandas.DataFrame(numpy.array([[28006.916, 21313.896]]), index=[1], columns=expected['featureMetadata']['Feature Name'].values)
				expected['peakResponse'] = pandas.DataFrame(numpy.array([[29.73305, 22.62752]]), index=[1], columns=expected['featureMetadata']['Feature Name'].values)
				expected['peakConcentrationDeviation'] = pandas.DataFrame(numpy.array([[numpy.nan, numpy.nan]]), index=[1], columns=expected['featureMetadata']['Feature Name'].values)
				expected['peakIntegrationFlag'] = pandas.DataFrame({'Feature2': ['bb'], 'Feature3': ['bb']}, index=[1])
				expected['peakRT'] = pandas.DataFrame(numpy.array([[11.501, 11.6407]]), index=[1], columns=expected['featureMetadata']['Feature Name'].values)
				# Generate
				with warnings.catch_warnings():
					warnings.simplefilter("ignore", UserWarning)
					result = nPYc.TargetedDataset(XMLpath, fileType='TargetLynx', sop='targetedSOP', sopPath=tmpdirname, calibrationReportPath=reportPath, keepIS=False, noiseFilled=False, keepPeakInfo=True, keepExcluded=False)

				# Test
				assert_frame_equal(expected['sampleMetadata'].reindex(sorted(expected['sampleMetadata']), axis=1), result.sampleMetadata.reindex(sorted(result.sampleMetadata), axis=1))
				assert_frame_equal(expected['featureMetadata'].reindex(sorted(expected['featureMetadata']), axis=1), result.featureMetadata.reindex(sorted(result.featureMetadata), axis=1))
				numpy.testing.assert_array_almost_equal(expected['intensityData'], result._intensityData)
				assert_frame_equal(expected['expectedConcentration'].reindex(sorted(expected['expectedConcentration']), axis=1), result.expectedConcentration.reindex(sorted(result.expectedConcentration), axis=1))
				# Calibration
				assert_frame_equal(expected['calibSampleMetadata'].reindex(sorted(expected['calibSampleMetadata']), axis=1), result.calibration['calibSampleMetadata'].reindex(sorted(result.calibration['calibSampleMetadata']), axis=1))
				assert_frame_equal(expected['calibFeatureMetadata'].reindex(sorted(expected['calibFeatureMetadata']), axis=1), result.calibration['calibFeatureMetadata'].reindex(sorted(result.calibration['calibFeatureMetadata']), axis=1))
				numpy.testing.assert_array_almost_equal(expected['calibIntensityData'], result.calibration['calibIntensityData'])
				assert_frame_equal(expected['calibExpectedConcentration'].reindex(sorted(expected['calibExpectedConcentration']), axis=1), result.calibration['calibExpectedConcentration'].reindex(sorted(result.calibration['calibExpectedConcentration']), axis=1))
				# Calibration peakInfo
				assert_frame_equal(expected['calibPeakArea'], result.calibration['calibPeakInfo']['peakArea'])
				assert_frame_equal(expected['calibPeakResponse'], result.calibration['calibPeakInfo']['peakResponse'])
				assert_frame_equal(expected['calibPeakConcentrationDeviation'], result.calibration['calibPeakInfo']['peakConcentrationDeviation'])
				assert_frame_equal(expected['calibPeakIntegrationFlag'], result.calibration['calibPeakInfo']['peakIntegrationFlag'])
				assert_frame_equal(expected['calibPeakRT'], result.calibration['calibPeakInfo']['peakRT'])
				# peakInfo
				assert_frame_equal(expected['peakArea'], result.peakInfo['peakArea'])
				assert_frame_equal(expected['peakResponse'], result.peakInfo['peakResponse'])
				assert_frame_equal(expected['peakConcentrationDeviation'], result.peakInfo['peakConcentrationDeviation'])
				assert_frame_equal(expected['peakIntegrationFlag'], result.peakInfo['peakIntegrationFlag'])
				assert_frame_equal(expected['peakRT'], result.peakInfo['peakRT'])
				# No exclusions
				# Attributes
				self.assertEqual(len(expected['Attributes'].keys()), len(result.Attributes.keys()) - 1)
				for i in expected['Attributes']:
					self.assertEqual(expected['Attributes'][i], result.Attributes[i])

		with self.subTest(msg='Change keepExcluded, keepIS=False, noiseFilled=False, keepPeakInfo=False, keepExcluded=True, no additional changes'):
			with tempfile.TemporaryDirectory() as tmpdirname:
				# Init
				# Create temp SOP file
				with open(os.path.join(tmpdirname, 'targetedSOP.json'), 'w') as outfile:
					json.dump(self.SOP, outfile)
				# Create temp .csv calibReport
				reportPath = os.path.join(tmpdirname, 'calibrationReport.csv')
				self.calibrationReport.to_csv(reportPath, index=False)
				XMLpath = os.path.join('..', '..', 'npc-standard-project', 'Derived_Data','UnitTest4_targeted.xml')
				expected = copy.deepcopy(self.expected)
				expected['Attributes']['calibrationReportPath'] = reportPath
				# Generate
				with warnings.catch_warnings():
					warnings.simplefilter("ignore", UserWarning)
					result = nPYc.TargetedDataset(XMLpath, fileType='TargetLynx', sop='targetedSOP', sopPath=tmpdirname, calibrationReportPath=reportPath, keepIS=False, noiseFilled=False, keepPeakInfo=False, keepExcluded=True)

				# Test
				assert_frame_equal(expected['sampleMetadata'].reindex(sorted(expected['sampleMetadata']), axis=1), result.sampleMetadata.reindex(sorted(result.sampleMetadata), axis=1))
				assert_frame_equal(expected['featureMetadata'].reindex(sorted(expected['featureMetadata']), axis=1), result.featureMetadata.reindex(sorted(result.featureMetadata), axis=1))
				numpy.testing.assert_array_almost_equal(expected['intensityData'], result._intensityData)
				assert_frame_equal(expected['expectedConcentration'].reindex(sorted(expected['expectedConcentration']), axis=1), result.expectedConcentration.reindex(sorted(result.expectedConcentration), axis=1))
				# Calibration
				assert_frame_equal(expected['calibSampleMetadata'].reindex(sorted(expected['calibSampleMetadata']), axis=1), result.calibration['calibSampleMetadata'].reindex(sorted(result.calibration['calibSampleMetadata']), axis=1))
				assert_frame_equal(expected['calibFeatureMetadata'].reindex(sorted(expected['calibFeatureMetadata']), axis=1), result.calibration['calibFeatureMetadata'].reindex(sorted(result.calibration['calibFeatureMetadata']), axis=1))
				numpy.testing.assert_array_almost_equal(expected['calibIntensityData'], result.calibration['calibIntensityData'])
				assert_frame_equal(expected['calibExpectedConcentration'].reindex(sorted(expected['calibExpectedConcentration']), axis=1), result.calibration['calibExpectedConcentration'].reindex(sorted(result.calibration['calibExpectedConcentration']), axis=1))
				# No peakInfo
				# Exclusions
				self.assertEqual(len(expected['sampleMetadataExcluded']), len(result.sampleMetadataExcluded))
				for i in range(len(result.sampleMetadataExcluded)):
					assert_frame_equal(expected['sampleMetadataExcluded'][i].reindex(sorted(expected['sampleMetadataExcluded'][i]), axis=1), result.sampleMetadataExcluded[i].reindex(sorted(result.sampleMetadataExcluded[i]), axis=1))
				self.assertEqual(len(expected['featureMetadataExcluded']), len(result.featureMetadataExcluded))
				for j in range(len(result.featureMetadataExcluded)):
					assert_frame_equal(expected['featureMetadataExcluded'][j].reindex(sorted(expected['featureMetadataExcluded'][j]), axis=1), result.featureMetadataExcluded[j].reindex(sorted(result.featureMetadataExcluded[j]), axis=1))
				self.assertEqual(len(expected['intensityDataExcluded']), len(result.intensityDataExcluded))
				for k in range(len(result.intensityDataExcluded)):
					numpy.testing.assert_array_almost_equal(expected['intensityDataExcluded'][k], result.intensityDataExcluded[k])
				self.assertEqual(len(expected['expectedConcentrationExcluded']), len(result.expectedConcentrationExcluded))
				for l in range(len(result.expectedConcentrationExcluded)):
					assert_frame_equal(expected['expectedConcentrationExcluded'][l].reindex(sorted(expected['expectedConcentrationExcluded'][l]), axis=1), result.expectedConcentrationExcluded[l].reindex(sorted(result.expectedConcentrationExcluded[l]), axis=1))
				self.assertEqual(result.excludedFlag, expected['excludedFlag'])
				# Attributes
				self.assertEqual(len(expected['Attributes'].keys()), len(result.Attributes.keys()) - 1)
				for i in expected['Attributes']:
					self.assertEqual(expected['Attributes'][i], result.Attributes[i])


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



class test_targeteddataset_import_undefined(unittest.TestCase):
	"""
	Test an error is raised when passing an unknown fileType
	"""
	def setUp(self):
		self.targetedData = nPYc.TargetedDataset('', fileType='empty')


	def test_targeteddataset_import_raise_notimplemented(self):
		self.assertRaises(NotImplementedError, nPYc.TargetedDataset, os.path.join('nopath'), fileType=None)


class test_targeteddataset_accuracy_precision(unittest.TestCase):
	"""
	Test the calculation of Accuracy and Precision
	"""
	def test_accuracyPrecision(self):
		from generateTestDataset import generateTestDataset

		noSamp = numpy.random.randint(100, high=200, size=None)
		noFeat = numpy.random.randint(20, high=80, size=None)

		tmpData = generateTestDataset(noSamp, noFeat)
		data = nPYc.TargetedDataset('', fileType='empty', sop='Generic')
		data.Attributes = tmpData.Attributes
		data._intensityData = tmpData._intensityData
		data.sampleMetadata = tmpData.sampleMetadata
		data.featureMetadata = tmpData.featureMetadata
		# expectedConcentration values between 1 and 10
		data.expectedConcentration = pandas.DataFrame(numpy.reshape(numpy.random.choice(numpy.arange(1, 11), noSamp * noFeat), newshape=(noSamp, noFeat)), index=data.sampleMetadata.index, columns=data.featureMetadata['Feature Name'])

		with self.subTest(msg="Default, all samples"):
			result = data.accuracyPrecision(onlyPrecisionReferences=False)
			self.assertTrue('Accuracy' in result.keys())
			self.assertTrue(SampleType.ExternalReference in result['Accuracy'].keys())
			self.assertTrue(SampleType.StudyPool in result['Accuracy'].keys())
			self.assertTrue(SampleType.StudySample in result['Accuracy'].keys())
			self.assertTrue(result['Accuracy'][SampleType.StudySample].shape == (len(numpy.unique(data.expectedConcentration)), noFeat))
			self.assertTrue('Precision' in result.keys())
			self.assertTrue(SampleType.ExternalReference in result['Precision'].keys())
			self.assertTrue(SampleType.StudyPool in result['Precision'].keys())
			self.assertTrue(SampleType.StudySample in result['Precision'].keys())
			self.assertTrue(result['Precision'][SampleType.StudySample].shape == (len(numpy.unique(data.expectedConcentration)), noFeat))

		with self.subTest(msg="Only Precision Reference"):
			# test the masking of precisionReference replacement of numpy.nan in Study Sample
			result = data.accuracyPrecision(onlyPrecisionReferences=True)
			self.assertTrue('Accuracy' in result.keys())
			self.assertTrue(SampleType.ExternalReference in result['Accuracy'].keys())
			self.assertTrue(SampleType.StudyPool in result['Accuracy'].keys())
			self.assertTrue(SampleType.StudySample in result['Accuracy'].keys())
			self.assertTrue(result['Accuracy'][SampleType.StudySample].shape == (0, noFeat))
			self.assertTrue('Precision' in result.keys())
			self.assertTrue(SampleType.ExternalReference in result['Precision'].keys())
			self.assertTrue(SampleType.StudyPool in result['Precision'].keys())
			self.assertTrue(SampleType.StudySample in result['Precision'].keys())
			self.assertTrue(result['Precision'][SampleType.StudySample].shape == (0, noFeat))


if __name__ == '__main__':
	unittest.main()
