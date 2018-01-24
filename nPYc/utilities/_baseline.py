# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 14:08:25 2016

@author: aahmed1
"""
import numpy as np
from scipy.stats.mstats import mquantiles
import pandas as pd
def baseline(filePathList, X, ppm, regionFrom, regionTo, BLorWP, alpha, threshold):
	"""
	% Function to determine baseline differences from either end of the 
	% (removed) presaturated water peak for a set of spectra, and plot results
	
	% Samples are defined as outliers if either:
	%   1. 'threshold' percent of the area (0.1 ppm) either side of the removed
	%      water region exceeds the critical value 'alpha'
	%   2. 'threshold' percent of the signal (0.1 ppm) either side of the 
	%      removed water region is negative
	%
	% Arguments:
	%   X (ns,nv) = spectral data (ns samples, nv variables)
	%   ppm (1,nv) = ppm scale 
	%
	% Optional arguments (in pairs):
	%   'alpha', value = critical value for defining the rejection
	%                    region (comprises alpha*100% of the sampling 
	%                    distribution) (default: alpha = 0.05)
	%   'threshold', value = percentage of samples in region allowed to exceed 
	%                        the 'alpha' value before sample is defined as an
	%                        outlier (default: threshold = 90)
	%   'savedir', char = path to directory in which to save figures
	%   'savename', char = name to which to save figure
	%
	% Return values:
	%   outliers (ns,1 dataset) = logical vector indicating outlying samples
	%   output (struct) = variables generated during run
	"""
	#reverse data if necessary
	if ppm[0]>ppm[1]:
		ppm = ppm[::-1]
		X = X[:, ::-1]
           
	ns=np.size(X,0)#number of samples
        
        #define region to investigate
        #find first entry in ppm with >='start' value
	minR = (ppm>=regionFrom).nonzero()
	minR = minR[0][0]#first entry
	maxR = (ppm>=regionTo).nonzero()
	maxR = maxR[0][0]#first entry
        #integrate area under peaks for each sample; first need to calculate area under curve: scipy library has trapezoid method but this gives actual area unlike the code here: hence translated matlab code instead
	area = areaTrap(X, ppm, ppm[minR], ppm[maxR])

	areaAbs=np.absolute(area)#absolute value / or distance from 0 [basically removes all (-) from negative numbers
        
        #proportion of points that exceed critical value for each sample
        
        #critical value
	areaCrit = mquantiles(areaAbs, 1 - alpha, axis=0)
        
        #proportion of points exceeding critical value
	failAreaCalc = areaAbs>np.tile(areaCrit,[ns, 1])
	failArea = [0] * np.size(failAreaCalc,0)
	for i in range(np.size(failAreaCalc,0)):# the for loop and failArea calc here is the quivalent of the fist part of the failArea equation in matlab code line 82 ie sum(areaAbs > repmat(areaCrit, ns, 1), 2)
		failAreaCalc1=sum(failAreaCalc[i])
		failArea[i]=failAreaCalc1
          
        #failArea = (areaAbs>np.tile(areaCrit,[ns, 1]),2)   / np.size(area,1)*100
	failArea = np.asarray(failArea)#first convert to array as cannot do division etc on list
	failArea = failArea / float(np.size(area,1))*100

        #proportion of points failing negativity test
	failNeg = [0] * np.size(failAreaCalc,0)
	for i in range(np.size(failAreaCalc,0)):# the for loop and failNeg calc here is the quivalent of the fist part of the failNeg equation in matlab code
		failNegCalc=sum(area[i]<0)/float(np.size(area,1))*100
		failNeg[i]=failNegCalc
	failNeg = np.asarray(failNeg)#convert to array 
        #this needs to be set so that if number of points exceed critical value in percentage then the sample gets classed as fail---previously got this confused fpor the final WP graph but this is number of points in 1 sample NOT individual sample
	outliersFailArea = failArea>threshold#outliers weather its high low BL or WP
	outliersFailNeg = failNeg>threshold#outliers weather its high low BL or WP
	index = range(np.size(failAreaCalc,0))
	outliersFailAreaDF = pd.DataFrame(outliersFailArea, index=index)
	outliersFailAreaDF = outliersFailAreaDF.join(filePathList)
	outliersFailNegDF = pd.DataFrame(outliersFailNeg, index=index)
	outliersFailNegDF = outliersFailNegDF.join(filePathList)
	outliersFailNegDF=outliersFailNegDF.dropna(axis=0)#delete all NaN valued rows which appear when we do gold standard data merge before entering this function from main class when we have smaller sample sizes
	outliersDF = pd.merge(outliersFailAreaDF, outliersFailNegDF, on='File Path', how='right').fillna(method='bfill')#for some r4eason on some data (when we have failures) it was causing duplicates on merge
	outliersDF.columns =[BLorWP+'outliersFailArea', 'File Path', BLorWP+'outliersFailNeg']#rename the column names
	del outliersFailAreaDF, outliersFailNegDF
        #add other info to DF
	outliersDF[BLorWP+'failArea'] = failArea[0:len(outliersDF)]#this is so ignores data that is from gold_standard data if sample sixe less than 80
	outliersDF[BLorWP+'failNeg'] = failNeg[0:len(outliersDF)]
	del regionFrom,regionTo,minR,maxR,area,alpha,threshold,areaCrit,failArea,failNeg
	return outliersDF

    ########## Area function - directly translated from matlab code, calculates area under curve using trapeze method #############
def areaTrap(X, ppm, start, stop):

        #also single calculations for a single sample differs to that of a batch at the batch takes the overall percentage of the X matrix
	start = (ppm>=start).nonzero()
	start = start[0][0]#first entry
	stop = (ppm<=stop).nonzero()
	stop = stop[0][-1]#last entry
	step=ppm[1]-ppm[0]
           
        #create empty (0's) array

	area = np.zeros([np.size(X,0), stop-start-1],dtype=object)#add dtype=object to avoid errors when writing strings etc to array

            
        #populate the array
	for i in range(np.size(X,0)):
		for j in range(start, stop-1):
                
			area[i, j-start] = (X[i,j]+X[i,j+1])/2*step
	return area
def main():
	pass

if __name__=='__main__':
	main()