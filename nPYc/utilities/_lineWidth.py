import numpy

from ._fitPeak import fitPeak

def lineWidth(X, ppm, sf, peakRange, multiplicity='singlet', parameters=dict(), shiftTollerance=0.003, peakIntesityFraction=None):
	"""
	Calculates the line width of a resonance in Hz, by fitting Pseudo Voigt peaks to the resonance.

	The method attempts to guess good fitting parameters based on the multiplicity and *peakRange* specified, but calculation can be significantly improved and or sped-up by providing overriding these defaults with the *parameters* dict.

	If *peakIntesityFraction* is not ``None``, the percentage ratio of the baseline component to the peak is calculated, and if it falls below the threshold provided, `NaN` is returned.

	:param numpy.array spectrum:
	:param numpy.array ppm:
	:param float sf: Spectrometer frequency
	:param peakRange: Tuple of (low bound, high bound), to search for the peak top in
	:type peakRange: (float, float)
	:param dict parameters: Dictionary of peak parameters used to overide defaults
	:param float shiftTollerance:
	:param peakIntesityFraction: Ratio of the baseline component to peak area
	:type peakIntesityFraction: None or float
	:returns: Width of the peak found in *peakRange* in Hz
	:rtype: float
	"""

	maxLW = 6 / sf
	estLW = 1 / sf

	fit = fitPeak(X, ppm, peakRange, multiplicity, parameters=parameters, maxLW=maxLW, estLW=estLW, shiftTollerance=shiftTollerance)

	if not fit.result.success:
		lw = numpy.nan
	elif peakIntesityFraction and (fit.eval_components(x=ppm)['p1_'].sum()/numpy.absolute(fit.eval_components(x=ppm)['baseline'].sum())) * 100 < peakIntesityFraction:
		lw = numpy.nan
	else:
		lw = fit.params['p1_fwhm'].value * sf

	return lw
