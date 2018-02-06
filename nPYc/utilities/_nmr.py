import numpy
import pandas
import math
import os
from pathlib import PurePath

from ..utilities._nmr import cutSec
from ..utilities._baseline import baseline


def interpolateSpectrum(spectrum, originalScale, targetScale):
	"""
	Interpolate spectra onto *targetScale* using python interp1d

	:param spectrum: the raw spectrum
	:type spectrum: array
	:param originalScale: the ppm
	:type originalScale:array
	:param targetScale: the scale
	:type targetScale: array
	:returns: interpolatedSpectra
	:rtype: array
	"""

	from scipy.interpolate import interp1d

	if spectrum.ndim == 2:
		noSpectra = spectrum.shape[0]
	elif spectrum.ndim == 1:
		noSpectra = 1
	else:
		raise ValueError("Interpolation is only supported for either a single or 2d array of 1D spectra.")

	if noSpectra == 1:
		interpolatedSpectrum = interp1d(originalScale,spectrum)(targetScale)

	else:
		interpolatedSpectrum = numpy.zeros(shape=(noSpectra, len(targetScale)))

		for i in range(noSpectra):
			interpolatedSpectrum[i,:] = interp1d(originalScale,spectrum[i,:], assume_sorted=True)(targetScale)

	return interpolatedSpectrum


def baselinePWcalcs(scalePPM, intenData, start, stop, filePathList, alpha, threshold, baselineLow_regionTo,baselineHigh_regionFrom, WPcutRegionA, WPcutRegionB, featureMask):
	"""
	calculate the baseline high/low and water peak high/low and the peak width in Hz for chosen peak (TSP or lactate)
	
	output a merged dataframe of all the calculated data

	:param scalePPM:
	:type scalePPM: array
	:param intenData: 
	:type intenData:array
	:param start:
	:type start: int
	:param stop:
	:type start: int
	:param sampleType:
	:type sampleType: string
	:param filePathList:
	:type filePathList: list
	:param sf: spectrometer frequency
	:type sf: int
	:param alpha:
	:type alpha: 
	:param threshold
	:type threshold: int
	:param baselineLow_regionTo
	:param: baselineHigh_regionFrom
	:param: WPcutRegionA
	:param: WPcutRegionB
	:param: LWpeakRangeFrom
	:param: LWpeakRangeTo
	:param pulProg:
	:type pulProg: string	
	:param mergedDF: final output merge all the data in to one dataframe
	:type mergedDF: Datafrane
	"""

	#if the sample size is less than 80 use gold standard data as spectra for baseline calcs
	#-----need to add the gold standard data loading here, need to ask Jake where to put the data(.npy files)----Ans: Jake wants data stored in json files so will neeed to extract and store for eg any arrays as text in json etc
	#ive saved as text files for now due to it being an array couldnt get it to load json format; saved using numpy.savetxt(r'D:\npyc-toolbox\nPYc\StudyDesigns\gold1_data.out', gold1_data, delimiter=',')  
	intenData_merged = intenData
	sizeintenData=numpy.size(intenData,0)
	
	#cutsec
	(ppmAfterCutSec, X, featureMask) = cutSec(scalePPM, intenData_merged, start, stop, featureMask)#flip ppm----inteData_merged is merged with gold standard data else same as as using intenData as does not get merged above remains original

	#water peak

	#	add the regions as variables for ppm not taken from attributes so we can return these values and save as attributes allowing us to use them later in plots and displaying in reports if need be
	BL_lowRegionFrom=ppmAfterCutSec.min()
	BL_highRegionTo=ppmAfterCutSec.max()
	WP_lowRegionFrom=WPcutRegionA-0.1
	WP_highRegionTo=WPcutRegionB+0.1
	
	[WPppmAfterCutSec, WP_X, featureMask]= cutSec(ppmAfterCutSec, X, WPcutRegionA, WPcutRegionB, featureMask)
	#calculate baseline

	df_outliers = baseline(filePathList, X, ppmAfterCutSec, BL_lowRegionFrom,baselineLow_regionTo,'BL_low_', alpha, threshold)
   #Baseline fluctuations between 9.5 and max(ppm) (High)
	baselineDF_High = baseline(filePathList, X, ppmAfterCutSec, baselineHigh_regionFrom,BL_highRegionTo, 'BL_high_', alpha, threshold)
                
  #now for water peak low region
	WPbaselineDF = baseline(filePathList, WP_X, WPppmAfterCutSec, WP_lowRegionFrom,WPcutRegionA, 'WP_low_', alpha, threshold)
  #waterPeak fluctuations between 9.5 and max(ppm) (High)
	WPbaselineDF_High = baseline(filePathList, WP_X, WPppmAfterCutSec, WPcutRegionB,WP_highRegionTo, 'WP_high_', alpha, threshold)
	#merge all data
	mergedDF=pandas.merge(df_outliers, baselineDF_High, on='File Path', how='left').fillna(0)# had tp merge like this and fill values with blank that were missing else was throwing away all values that were missing from baselinedf
	mergedDF=pandas.merge(mergedDF, WPbaselineDF, on='File Path', how='left').fillna(0)# had tp merge like this and fill values with blank that were missing else was throwing away all values that were missing from baselinedf
	mergedDF=pandas.merge(mergedDF, WPbaselineDF_High, on='File Path', how='left').fillna(0)# had tp merge like this and fill values with blank that were missing else was throwing away all values that were missing from baselinedf
	del df_outliers, baselineDF_High, WPbaselineDF, WPbaselineDF_High

	return mergedDF, featureMask, BL_lowRegionFrom, BL_highRegionTo, WP_lowRegionFrom, WP_highRegionTo


def generateBaseName(sampleMetadata):
	"""
	Generate a sample base name for the loaded dataset by rounding by 10
	:param sampleMetadata:
	"""
	# Spliting expno from 'Sample File Name'
	def splitExpno(sampleFileName):
		output = sampleFileName.split(sep='/')
		return output[-1]

	# Rounding expno down by 10s
	def roundExpno(expno):
		return int(math.floor(expno / 10.0) * 10)

	def splitRack(sampleFileName):
		output = sampleFileName.split(sep='/')
		return output[-2]

	if not 'expno' in sampleMetadata.columns:
		expno = sampleMetadata.loc[:, 'Sample File Name'].apply(splitExpno)
		expno = pandas.to_numeric(expno)

	else:
		expno = sampleMetadata['expno']

	roundedExpno = expno.map(roundExpno)
	roundedExpno = roundedExpno.astype(str)

	return sampleMetadata.loc[:,'Sample File Name'].apply(splitRack).str.cat(roundedExpno, sep='/').values, expno.values


def cutSec(ppm, X, start, stop, featureMask):
	"""
	Temove defined regions from NMR spectra data
	input/output as per matlab version of code:

	% ppm (1,nv) = ppm scale for nv variables
	% X (ns,nv)  = NMR spectral data for ns samples
	% start (1,1) = ppm value for start of region to remove
	% stop (1,1) = ppm value for end of region to remove
	%
	% OUTPUT:
	% ppm (1,nr) = ppm scale with region from start:stop removed
	% X (ns,nr)  = NMR spectral with region from start:stop removed

	"""

	flip = 0
	if ppm[0] > ppm[-1]:
		flip = 1
		ppm = ppm[::-1]
		X = X[:, ::-1]

	# find first entry in ppm with >='start' valu
	start = (ppm >= start).nonzero()
	start = start[0][0]  # first entry
	stop = (ppm <= stop).nonzero()
	stop = stop[0][-1]  # last entry

	# currently setting featureMask will get rid of peaks in start:stop region BUT it also marks as excluded so have removed as inaccurately marking for exclusion when all we want to do is remove from intensityData not mark as exluded
	try:
		featureMask[0,
		start:stop] = False  # this may only occur on unit test data, not sure need to check but either way was causing issue
	except:
		featureMask[start:stop] = False
	if flip == 1:
		ppm = ppm[::-1]
		X = X[:, ::-1]
	return ppm, X, featureMask
	pass

## FUNCTION ON PROBATION HERE - cannot work with self 
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

