# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 14:18:31 2016

@author: aahmed1
"""
#class _cutSec():
def cutSec(ppm, X, start, stop, featureMask):	
	"""
	Its purpose is to remove defined regions from NMR spectra data
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
	flip=0
	if ppm[0]>ppm[-1]:
		flip=1
		ppm = ppm[::-1]
		X = X[:, ::-1]
            
        #find first entry in ppm with >='start' valu
	start = (ppm>=start).nonzero()
	start = start[0][0]#first entry
	stop = (ppm<=stop).nonzero()
	stop = stop[0][-1]#last entry

#currently setting featureMask will get rid of peaks in start:stop region BUT it also marks as excluded so have removed as inaccurately marking for exclusion when all we want to do is remove from intensityData not mark as exluded
	try:
		featureMask[0,start:stop]=False # this may only occur on unit test data, not sure need to check but either way was causing issue
	except:
		featureMask[start:stop]=False
	if flip==1:
		ppm = ppm[::-1]
		X = X[:, ::-1]
	return ppm, X, featureMask
	pass

