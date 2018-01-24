import numpy
import lmfit

from ._calibratePPMscale import referenceToResolvedMultiplet

def integrateResonance(spectrum, ppm, centrePoint, parameters=dict()):
	"""
	Calculates the intergral of the resonances at *centrePoint*.

	TODO: Extend for higher multiplicity resonances

	:param numpy.array spectrum:
	:param numpy.array ppm:
	:param float centrePoint: Centre of the resonances to be modeled
	:type peakRange: (float, float)
	:param dict parameters: Dictionary of peak parameters used to overide defaults
	:rtype: float
	"""

	defaultParams = dict()
	defaultParams['p1_fraction'] = {'vary':True, 'expr':'', 'min':0, 'max':1.1}
	defaultParams['p1_amplitude'] = {'vary':True, 'expr':'', 'min':None, 'max':numpy.inf}
	defaultParams['p1_center'] = {'value':centrePoint, 'vary':True, 'expr':'', 'min':centrePoint-0.05, 'max':centrePoint+0.03}

	parameters = {**defaultParams, **parameters}

	fit = fitPeak(spectrum, ppm, ((centrePoint - 0.1), (centrePoint + 0.1)), 'singlet', parameters=parameters)

	##
	# Return the intergral based on the entire ppm scale to avaiod narrow search windows biasing the result
	##
	return fit.eval_components(x=ppm)['p1_'].sum()


def fitPeak(X, ppm, peakRange, multiplicity, parameters=dict(), maxLW=None, estLW=None, shiftTollerance=0.003):
	"""
	Calculates the line width of a resonance in Hz, by fitting Pseudo Voigt peaks to the resonance.

	The method attempts to guess good fitting parameters based on the multiplicity and *peakRange* specified, but calculation can be significantly improved and or sped-up by providing overriding these defaults with the *parameters* dict.

	If *peakIntesityFraction* is not ``None``, the percentage ratio of the baseline component to the peak is calculated, and if it falls below the threshold provided, `NaN` is returned.

	:param numpy.array spectrum:
	:param numpy.array ppm:
	:param float sf: Spectrometer frequency
	:param peakRange: Tuple of (low bound, high bound), toe search for the peak top in
	:type peakRange: (float, float)
	:param dict parameters: Dictionary of peak parameters used to overide defaults
	:param float shiftTollerance:
	:returns: Width of the peak found in *peakRange* in Hz
	:rtype: float
	"""

	if ppm[0] > ppm[1]:
		ppm = ppm[::-1]
		X = X[::-1]

	peakMask = numpy.where((ppm >= peakRange[0]) & (ppm <= peakRange[1]))[0]
	spec = X[peakMask]
	localPPM = ppm[peakMask]

	if multiplicity.lower() == 'singlet':

		centrePoint = numpy.mean(peakRange)

		defaultParams = dict()
		defaultParams['p1_fraction'] = {'vary':True, 'expr':'',  'min':0, 'max':1.1}
		defaultParams['p1_amplitude'] = {'vary':True, 'expr':'',  'min':0, 'max':numpy.inf}
		defaultParams['p1_sigma'] = {'value':estLW, 'vary':True, 'expr':'', 'min':0, 'max':maxLW}
		defaultParams['p1_center'] = {'value':centrePoint, 'vary':True, 'expr':'', 'min':centrePoint-0.05, 'max':centrePoint+0.03}

		parameters = {**defaultParams, **parameters}

		peak = lmfit.models.PseudoVoigtModel(prefix='p1_')
		baseline = lmfit.models.LinearModel(prefix='baseline')

		pars = peak.guess(spec, x=localPPM)
		pars += baseline.guess(spec, x=localPPM)
		pars['p1_fraction'].set(vary=parameters['p1_fraction']['vary'],
								expr=parameters['p1_fraction']['expr'],
								min=parameters['p1_fraction']['min'],
								max=parameters['p1_fraction']['max'])
		pars['p1_amplitude'].set(vary=parameters['p1_amplitude']['vary'],
								expr=parameters['p1_amplitude']['expr'],
								min=parameters['p1_amplitude']['min'],
								max=parameters['p1_amplitude']['max'])
		pars['p1_sigma'].set(value=parameters['p1_sigma']['value'],
							 vary=parameters['p1_sigma']['vary'],
							 expr=parameters['p1_sigma']['expr'],
							 min=parameters['p1_sigma']['min'],
							 max=parameters['p1_sigma']['max'])
		pars['p1_center'].set(value=parameters['p1_center']['value'],
							  vary=parameters['p1_center']['vary'],
							  expr=parameters['p1_center']['expr'],
							  min=parameters['p1_center']['min'],
							  max=parameters['p1_center']['max'])

		peak += baseline

	elif multiplicity.lower() == 'doublet':

		peakPositions = referenceToResolvedMultiplet(spec, localPPM, peakRange, 2)

		defaultParams = dict()
		defaultParams['p1_fraction'] = {'vary':True, 'expr':'p2_fraction', 'min':None, 'max':None}
		defaultParams['p1_amplitude'] = {'vary':True, 'expr':'',  'min':0, 'max':numpy.inf}
		defaultParams['p1_center'] = {'value':localPPM[peakPositions[0]],
									  'vary':True, 'expr':'',
									  'min':localPPM[peakPositions[0]]-shiftTollerance,
									  'max':localPPM[peakPositions[0]]+shiftTollerance}
		defaultParams['p1_sigma'] = {'value':estLW, 'vary':True, 'expr':'', 'min':0, 'max':maxLW}

		defaultParams['p2_fraction'] = {'vary':True, 'expr':'', 'min':0, 'max':1.1}
		defaultParams['p2_amplitude'] = {'vary':True, 'expr':'p1_amplitude', 'min':0, 'max':numpy.inf}
		defaultParams['p2_sigma'] = {'vary':True, 'expr':'p1_sigma', 'min':None, 'max':None}
		defaultParams['p2_center'] = {'value':localPPM[peakPositions[1]],
									  'vary':True, 'expr':'',
									  'min':localPPM[peakPositions[1]]-shiftTollerance,
									  'max':localPPM[peakPositions[1]]+shiftTollerance}

		parameters = {**defaultParams, **parameters}

		peak = lmfit.models.PseudoVoigtModel(prefix='p1_')
		peak2 = lmfit.models.PseudoVoigtModel(prefix='p2_')
		baseline = lmfit.models.LinearModel(prefix='baseline')

		pars = peak.guess(spec, x=localPPM)
		pars += peak2.guess(spec, x=localPPM)
		pars += baseline.guess(spec, x=localPPM)

		pars['p1_fraction'].set(vary=parameters['p1_fraction']['vary'],
								expr=parameters['p1_fraction']['expr'],
								min=parameters['p1_fraction']['min'],
								max=parameters['p1_fraction']['max'])
		pars['p1_amplitude'].set(vary=parameters['p1_amplitude']['vary'],
								expr=parameters['p1_amplitude']['expr'],
								min=parameters['p1_amplitude']['min'],
								max=parameters['p1_amplitude']['max'])
		pars['p1_sigma'].set(value=parameters['p1_sigma']['value'],
							 vary=parameters['p1_sigma']['vary'],
							 expr=parameters['p1_sigma']['expr'],
							 min=parameters['p1_sigma']['min'],
							 max=parameters['p1_sigma']['max'])
		pars['p1_center'].set(value=parameters['p1_center']['value'],
							  vary=parameters['p1_center']['vary'],
							  expr=parameters['p1_center']['expr'],
							  max=parameters['p1_center']['max'])


		pars['p2_fraction'].set(vary=parameters['p2_fraction']['vary'],
								expr=parameters['p2_fraction']['expr'],
								min=parameters['p2_fraction']['min'],
								max=parameters['p2_fraction']['max'])
		pars['p2_amplitude'].set(vary=parameters['p2_amplitude']['vary'],
								expr=parameters['p2_amplitude']['expr'],
								min=parameters['p2_amplitude']['min'],
								max=parameters['p2_amplitude']['max'])
		pars['p2_sigma'].set(vary=parameters['p2_sigma']['vary'],
							 expr=parameters['p2_sigma']['expr'],
							 min=parameters['p2_sigma']['min'],
							 max=parameters['p2_sigma']['max'])
		pars['p2_center'].set(value=parameters['p2_center']['value'],
							  vary=parameters['p2_center']['vary'],
							  expr=parameters['p2_center']['expr'],
							  min=parameters['p2_center']['min'])

		peak += peak2
		peak += baseline

	elif multiplicity.lower() == 'quartet':

		peakPositions = referenceToResolvedMultiplet(spec, localPPM, peakRange, 2)

		if localPPM[peakPositions[0]] < localPPM[peakPositions[1]]:
			old = peakPositions
			peakPositions = [old[1], old[0]]

		coupling = localPPM[peakPositions[0]] - localPPM[peakPositions[1]]

		defaultParams = dict()
		defaultParams['p2_fraction'] = {'vary':True, 'expr':'p1_fraction', 'min':None, 'max':None}
		defaultParams['p2_amplitude'] = {'vary':True, 'expr':'p1_amplitude / 3', 'min':None, 'max':numpy.inf}
		defaultParams['p2_sigma'] = {'vary':True, 'expr':'p1_sigma', 'min':0, 'max':maxLW}
		posEst = localPPM[peakPositions[0]] + coupling
		defaultParams['p2_center'] = {'value':posEst,
							  'vary':True, 'expr':'',
							  'min':posEst-shiftTollerance,
							  'max':posEst+shiftTollerance}

		defaultParams['p1_fraction'] = {'vary':True, 'expr':'', 'min':0, 'max':1.1}
		defaultParams['p1_amplitude'] = {'vary':True, 'expr':'', 'min':0, 'max':numpy.inf}
		defaultParams['p1_sigma'] = {'value':estLW, 'vary':True, 'expr':'', 'min':None, 'max':None}
		defaultParams['p1_center'] = {'value':localPPM[peakPositions[0]],
									  'vary':True, 'expr':'',
									  'min':localPPM[peakPositions[0]]-shiftTollerance,
									  'max':localPPM[peakPositions[0]]+shiftTollerance}

		defaultParams['p3_fraction'] = {'vary':True, 'expr':'p1_fraction', 'min':None, 'max':None}
		defaultParams['p3_amplitude'] = {'vary':True, 'expr':'p1_amplitude', 'min':None, 'max':numpy.inf}
		defaultParams['p3_sigma'] = {'vary':True, 'expr':'p1_sigma', 'min':None, 'max':None}
		defaultParams['p3_center'] = {'value':localPPM[peakPositions[1]],
									  'vary':True, 'expr':'',
									  'min':localPPM[peakPositions[1]]-shiftTollerance,
									  'max':localPPM[peakPositions[1]]+shiftTollerance}

		defaultParams['p4_fraction'] = {'vary':True, 'expr':'p1_fraction', 'min':None, 'max':None}
		defaultParams['p4_amplitude'] = {'vary':True, 'expr':'p1_amplitude / 3', 'min':None, 'max':numpy.inf}
		defaultParams['p4_sigma'] = {'vary':True, 'expr':'p1_sigma', 'min':None, 'max':None}
		posEst = localPPM[peakPositions[1]] - coupling
		defaultParams['p4_center'] = {'value':posEst,
							  'vary':True, 'expr':'',
							  'min':posEst-shiftTollerance,
							  'max':posEst+shiftTollerance}

		parameters = {**defaultParams, **parameters}

		peak = lmfit.models.PseudoVoigtModel(prefix='p1_')
		peak2 = lmfit.models.PseudoVoigtModel(prefix='p2_')
		peak3 = lmfit.models.PseudoVoigtModel(prefix='p3_')
		peak4 = lmfit.models.PseudoVoigtModel(prefix='p4_')
		baseline = lmfit.models.LinearModel(prefix='baseline')

		pars = peak.guess(spec, x=localPPM)
		pars += peak2.guess(spec, x=localPPM)
		pars += peak3.guess(spec, x=localPPM)
		pars += peak4.guess(spec, x=localPPM)
		pars += baseline.guess(spec, x=localPPM)

		pars['p1_fraction'].set(vary=parameters['p1_fraction']['vary'],
								expr=parameters['p1_fraction']['expr'],
								min=parameters['p1_fraction']['min'],
								max=parameters['p1_fraction']['max'])
		pars['p1_amplitude'].set(vary=parameters['p1_amplitude']['vary'],
								expr=parameters['p1_amplitude']['expr'],
								min=parameters['p1_amplitude']['min'],
								max=parameters['p1_amplitude']['max'])
		pars['p1_sigma'].set(value=parameters['p1_sigma']['value'],
							 vary=parameters['p1_sigma']['vary'],
							 expr=parameters['p1_sigma']['expr'],
							 min=parameters['p1_sigma']['min'],
							 max=parameters['p1_sigma']['max'])
		pars['p1_center'].set(value=parameters['p1_center']['value'],
							  vary=parameters['p1_center']['vary'],
							  expr=parameters['p1_center']['expr'],
							  max=parameters['p1_center']['max'])

		pars['p2_fraction'].set(vary=parameters['p2_fraction']['vary'],
								expr=parameters['p2_fraction']['expr'],
								min=parameters['p2_fraction']['min'],
								max=parameters['p2_fraction']['max'])
		pars['p2_amplitude'].set(vary=parameters['p2_amplitude']['vary'],
								expr=parameters['p2_amplitude']['expr'],
								min=parameters['p2_amplitude']['min'],
								max=parameters['p2_amplitude']['max'])
		pars['p2_sigma'].set( vary=parameters['p2_sigma']['vary'],
							 expr=parameters['p2_sigma']['expr'],
							 min=parameters['p2_sigma']['min'],
							 max=parameters['p2_sigma']['max'])
		pars['p2_center'].set(value=parameters['p2_center']['value'],
							  vary=parameters['p2_center']['vary'],
							  expr=parameters['p2_center']['expr'],
							  min=parameters['p2_center']['min'])

		pars['p3_fraction'].set(vary=parameters['p3_fraction']['vary'],
								expr=parameters['p3_fraction']['expr'],
								min=parameters['p3_fraction']['min'],
								max=parameters['p3_fraction']['max'])
		pars['p3_amplitude'].set(vary=parameters['p3_amplitude']['vary'],
								expr=parameters['p3_amplitude']['expr'],
								min=parameters['p3_amplitude']['min'],
								max=parameters['p3_amplitude']['max'])
		pars['p3_sigma'].set(vary=parameters['p3_sigma']['vary'],
							 expr=parameters['p3_sigma']['expr'],
							 min=parameters['p3_sigma']['min'],
							 max=parameters['p3_sigma']['max'])
		pars['p3_center'].set(value=parameters['p3_center']['value'],
							  vary=parameters['p3_center']['vary'],
							  expr=parameters['p3_center']['expr'],
							  min=parameters['p3_center']['min'])

		pars['p4_fraction'].set(vary=parameters['p4_fraction']['vary'],
								expr=parameters['p4_fraction']['expr'],
								min=parameters['p4_fraction']['min'],
								max=parameters['p4_fraction']['max'])
		pars['p4_amplitude'].set(vary=parameters['p4_amplitude']['vary'],
								expr=parameters['p4_amplitude']['expr'],
								min=parameters['p4_amplitude']['min'],
								max=parameters['p4_amplitude']['max'])
		pars['p4_sigma'].set(vary=parameters['p4_sigma']['vary'],
							 expr=parameters['p4_sigma']['expr'],
							 min=parameters['p4_sigma']['min'],
							 max=parameters['p4_sigma']['max'])
		pars['p4_center'].set(value=parameters['p4_center']['value'],
							  vary=parameters['p4_center']['vary'],
							  expr=parameters['p4_center']['expr'],
							  min=parameters['p4_center']['min'])
		peak += peak2
		peak += peak3
		peak += peak4
		peak += baseline

	else:
		raise ValueError('"%s" is not an understood multiplicity.' % (multiplicity))

	fit = peak.fit(spec, pars, x=localPPM)

	return fit
