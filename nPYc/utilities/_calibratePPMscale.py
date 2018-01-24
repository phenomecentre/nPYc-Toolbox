import numpy

def calibratePPM(calibrationType, calibrateTo, ppmSearchRange, ppm, spectrum, spectrumType='1D', align='circularShift'):
	"""
	Calibrate *spectrum* against the *ppm* scale.

	The calibration type may use the following methods to detect the resonance to be calibrated to:
	- *singlet*: The highest point within *ppmSearchRange* is found
	- *doublet*: The midpoint of the two highest peaks in *ppmSearchRange* is found, the technique in Pearce *et al.*[#]_ is used to account for broad baseline effects

	Spectra may be aligned by:
	- *circularShift*: The spectrum is moved pointwise to the left or right, points that fall off the end of the spectrum are wrapped back to the opposite end

	:param str calibrationType: Either 'singlet' or 'doublet' 
	:param float calibrateTo: Set the PPM value of the detected target to this value
	:param ppmSearchRange: Tuple of (low cutoff, high cutoff) indicating the range to search in
	:type ppmSearchRange: (float, float)
	:param numpy.array ppm: The PPM scale of the spectrum to be aligned 
	:param numpy.array spectrum: The spectrum to be aligned
	:param str spectrumType: Type of spectrum supplied
	:param str align: Method of alignment to use
	:returns: Tuple of (*aligned spectrum*, *aligned ppm scale*, distance between target peak and *calibrateTo* in PPM)
	:rtype: (numpy.array, numpy.array, float)

	.. [#] Jake T. M. Pearce, Toby J. Athersuch, Timothy M. D. Ebbels, John C. Lindon, Jeremy K. Nicholson, and Hector C. Keun, Robust Algorithms for Automated Chemical Shift Calibration of 1D 1H NMR Spectra of Blood Serum, Analytical Chemistry 2008 80 (18), 7158-7162, `DOI: 10.1021/ac8011494 <http://dx.doi.org/10.1021/ac8011494>`_
	"""

	# If the ppm scale is descending, flip the data matrixes l<>r
	descending = False
	if ppm[0] > ppm[1]:
		ppm = ppm[::-1]
		spectrum = spectrum[::-1]
		descending = True

	if spectrumType.lower() == 'j-res':
		# Not implemented
		raise NotImplementedError('Calibration of J-Res spectra not implemented')

	elif spectrumType.lower() == '1d':

		if calibrationType.lower() == 'doublet':

			deltaPPM = referenceToResolvedMultiplet(spectrum, ppm, ppmSearchRange, 2)

			deltaPPM = int(round(numpy.mean(deltaPPM)))

		elif calibrationType.lower() == 'singlet':

			deltaPPM = referenceToSinglet(spectrum, ppm, ppmSearchRange)

		else:
			raise NotImplementedError('Unknown calibration type')

		targetIndex = numpy.size(numpy.where(ppm<=calibrateTo))
		deltaPPM = deltaPPM - targetIndex

		if align.lower() == 'circularshift':
			spectrum = numpy.roll(spectrum, -deltaPPM)

		elif align.lower() == 'interpolate':
			raise NotImplementedError('Alignment by interpolation not implemented')

	else:
		raise ValueError('"%s" is not a recognised spectrum type' % (spectrumType))

	if align.lower() == 'circularshift':
		ppm = ppm - (ppm[targetIndex] -calibrateTo)

	#if we flipped the spectrum , now flip it back
	if descending == True:
		ppm = ppm[::-1]
		spectrum = spectrum[::-1]

	return spectrum, ppm, ppm[targetIndex + deltaPPM] - ppm[targetIndex]


def referenceToSinglet(spectrum, ppm, peakRange):
	"""
	Find the highest point in the region indicated by the tuple of (*low bound*, *high bound*).

	Where all values in peakRange are identical behaviour is undefined.

	:param numpy.array spectrum: Vector of values representing the real part of the spectrum
	:param numpy.array ppm: Chemical shift scale corresponding to *spectrum*, must be ordered ascending
	:param peakRange: Tuple of low and high ppm vales demarcating the range in which to search.
	:type peakRange: (float, float)
	:return: The location of the highest point in the selected region
	"""

	# Mask is False in the region we will exlude
	regionMask = (ppm < peakRange[0]) | (ppm > peakRange[1])

	maskedSpectrum = numpy.ma.masked_array(spectrum, mask=regionMask)

	deltaPPM = numpy.argmax(maskedSpectrum)

	return deltaPPM


def referenceToResolvedMultiplet(spectrum, ppm, peakRange, multiplicity, peakMaskWidth=0.004):
	"""
	Find the *multiplicity* sharpest and most intense peaks in *peakRange*.


	:param numpy.array spectrum: Vector of values representing the real part of the spectrum
	:param numpy.array ppm: Chemical shift scale corresponding to *spectrum*, must be ordered ascending
	:param peakRange: Tuple of low and high ppm vales demarcating the range in which to search.
	:type peakRange: (float, float)
	:param int multiplicity: 
	:return: The location of the centre of the dominant doublet in the selected region
	"""

	# Mask is False in the region we will exlude
	regionMask = (ppm > peakRange[0]) & (ppm < peakRange[1])

	maskedSpectrum = numpy.ma.masked_array(spectrum, mask=~regionMask)

	peakLocations = []

	for i in range(multiplicity):
		# Take the approximate second derivative
		diffedSpectrum = numpy.diff(maskedSpectrum, 2)

		# Find the two lowest points, corresponding to the top of the two sharpest peaks.
		peakIndex = numpy.argmin(diffedSpectrum)
		peakPPM = ppm[peakIndex]

		#Having found the first peak, flatten it so we can locate the second.
		peakMask = (ppm < peakPPM - peakMaskWidth) | (ppm > peakPPM + peakMaskWidth)

		regionMask = numpy.logical_and(regionMask, peakMask)

		maskedSpectrum = numpy.ma.masked_array(spectrum, mask=~regionMask)

		peakLocations.append(peakIndex + 1)

	return peakLocations
