import os
import numpy
import warnings
from xml.etree import ElementTree

from ..utilities._calibratePPMscale import calibratePPM
from ..utilities._lineWidth import lineWidth
from ..utilities._fitPeak import integrateResonance
from ..utilities._nmr import interpolateSpectrum
from ..utilities import extractParams

def importBrukerSpectra(path, pulseProgram, pdata, Attributes):
	"""
	Load processed Bruker spectra found under *path*, with a pulse program that matches *pulseProgram*.

	Import can be configured by passing values in the *Attributes* dictionary as follows:

	====================== ===== =========================
	Key                    dtype Usage
	====================== ===== =========================
	variableSize           int   Number of points in the each returned spectrum
	alignTo                str   Method to reference the ppm scale, see :py:func:`~nPYc.utilities.calibratePPM` for options
	calibrateTo            float Calibration target will be set to this shift
	ppmSearchRange         tuple Search for calibration target in this (low, high) window
	LWpeakMultiplicity     str   Multiplicity of peak used to calculate line widths, see :py:func:`~nPYc.utilities.lineWidth` for options
	LWpeakRange            tuple Search for LW target in this (low, high) window
	LWpeakIntesityFraction float The integrated LW peak must exceed the fractional baseline intergral by this percentage fraction
	====================== ===== =========================

	:param str path: Find all matching spectra under this directory tree
	:param str pulseProgram: Only load spectra acquired with a matching pulse program
	:param int pdata: Load processed data fromt the specified pdata
	:param dict Attributes: Dictionary of configuration parameters
	:returns: Tuple of (spectra, ppm, metadata)
	:rtype: (numpy.array, numpy.array, pandas.DataFrame)
	"""

	metadata = extractParams(path, 'Bruker', pdata=pdata)

	if metadata.shape[0] == 0:
		raise ValueError("No Bruker format spectra found in '%s'." % (path))

	##
	# Trim spectra that do not match pulseProgram
	##
	metadata = metadata.loc[metadata['PULPROG'] == pulseProgram]
	metadata.reset_index(inplace=True)

	if metadata.shape[0] == 0:
		raise ValueError("No Bruker format spectra acquired with the '%s' pulse program found." % (pulseProgram))

	intensityData = numpy.zeros((metadata.shape[0], Attributes['variableSize']))
	skipERETIC = False

	metadata['Delta PPM'] = numpy.nan
	metadata['ERETIC Intergral'] = numpy.nan
	metadata['ERETIC Concentration (mM)'] = numpy.nan
	metadata['Line Width (Hz)'] = numpy.nan
	metadata['Warnings'] = ''
	ppm = numpy.linspace(Attributes['bounds'][1], Attributes['bounds'][0], Attributes['variableSize'])
	for row in metadata.iterrows():
		try:
			##
			# Load spectral data
			##
			spectrum, localPPM = importBrukerSpectrum(row[1]['File Path'],
													row[1]['OFFSET'],
													row[1]['SW_p'],
													row[1]['NC_proc'],
													row[1]['SF'],
													row[1]['SI'],
													row[1]['BYTORDP'])

			##
			# Do per-spectrum QC work here
			##
			##
			# If QuantFactorSample.xml exists, intergrate ERETIC siginal
			##
			quantFilePath = os.path.dirname(row[1]['File Path'])
			quantFilePath = os.path.join(quantFilePath, '..', '..', 'QuantFactorSample.xml')

			if os.path.isfile(quantFilePath):
				try:
					position, erLineWidth, concentration = parseQuantFactorSample(quantFilePath)

					metadata.loc[row[0], 'ERETIC Concentration (mM)'] = concentration
					metadata.loc[row[0], 'ERETIC Intergral'] = integrateResonance(spectrum, localPPM, position)
				except:
					metadata.loc[row[0], 'Warnings'] = 'Error calculating ERETIC intergral'
					warnings.warn('Error parsing `QuantFactorSample`.\nSkipping integration of ERETIC signal for %s.' % (row[1]['File Path']))

			##
			# Calibrate PPM scale
			##
			spectrum, localPPM, deltaPPM = calibratePPM(Attributes['alignTo'], Attributes['calibrateTo'], Attributes['ppmSearchRange'], localPPM, spectrum)
			metadata.loc[row[0], 'Delta PPM'] = deltaPPM

			lwHz = lineWidth(spectrum, localPPM, metadata.loc[row[0], 'SF'], Attributes['LWpeakRange'],
							multiplicity=Attributes['LWpeakMultiplicity'],
							peakIntesityFraction=Attributes['LWpeakIntesityFraction'])
			metadata.loc[row[0], 'Line Width (Hz)'] = lwHz

			##
			# Interpolate onto common scale
			##
			intensityData[row[0], :] = interpolateSpectrum(spectrum, localPPM, ppm)

		except:
			metadata.loc[row[0], 'Warnings'] = 'Error loading file'
			warnings.warn("Error loading '%s'" % (row[1]['File Path']))

	return intensityData, ppm, metadata


def importBrukerSpectrum(path, offset, sw_p, nc_proc, sf, si, bytordp):
	"""
	Load processed 1D Bruker spectra (*1r* files) from *path*.

	:param str path: Path to *1r* file
	:param float offset: *offset* (ppm value of the first data point of the spectrum) parameter from *procs* file
	:param float sw_p: *SW_p* (spectral width) parameter from *procs* file
	:param int nc_proc: *NC_proc*  intensity scaling factor from *procs* file
	:param float sf: *SF* (spectral reference frequency) parameter from *procs* file
	:param int si: *SI* (number of points in the processed data) parameter from *procs* file
	:param int bytordp: *BYTORDP* parameter from *procs* file
	:param int xdim: *XDIM* (submatrix size) parameter from *procs* file (only relevant for 2D data)
	"""

	##
	# Determine type of spectrum from filename
	##
	fileName = os.path.basename(path)
	if fileName == '1r':
		dimensions = 1
	elif fileName == '2rr':
		raise NotImplementedError('Reading of 2D NMR data is not implemented')

	##
	# Check file exists
	##
	if not os.path.isfile(path):
		raise IOError('Unable to read %s' % (path))

	##
	# Parse endianness for numpy
	##
	if int(bytordp) == 0:
		machine_format = '<i4'
	else:
		machine_format = '>i4'

	##
	# Open and read spectrum
	##
	fid = open(path, 'rb')
	x1 = pow(2, int(nc_proc))
	dim1 = numpy.fromfile(fid, dtype=machine_format)
	spectra_real = (dim1 * x1)
	fid.close()

	##
	# Build ppm scale
	##
	swp = float(sw_p) / float(sf)
	dppm = swp / float(si)
	spectra_ppm = numpy.arange(float(offset), (float(offset) - swp), -dppm)

	return spectra_real, spectra_ppm

def importBruker2DSpectrum(path, offset, sw_p, nc_proc, sf, si, bytordp):
	"""
	Load processed 1D Bruker spectra (*1r* files) from *path*.

	:param str path: Path to *1r* file
	:param float offset: *offset* (ppm value of the first data point of the spectrum) parameter from *procs* file
	:param float sw_p: *SW_p* (spectral width) parameter from *procs* file
	:param int nc_proc: *NC_proc*  intensity scaling factor from *procs* file
	:param float sf: *SF* (spectral reference frequency) parameter from *procs* file
	:param int si: *SI* (number of points in the processed data) parameter from *procs* file
	:param int bytordp: *BYTORDP* parameter from *procs* file
	:param int xdim: *XDIM* (submatrix size) parameter from *procs* file (only relevant for 2D data)
	"""

	##
	# Determine type of spectrum from filename
	##
	fileName = os.path.basename(path)
	if fileName == '1r':
		dimensions = 1
	elif fileName == '2rr':
		raise NotImplementedError('Reading of 2D NMR data is not implemented')

	##
	# Check file exists
	##
	if not os.path.isfile(path):
		raise IOError('Unable to read %s' % (path))

	##
	# Parse endianness for numpy
	##
	if int(bytordp) == 0:
		machine_format = '<i4'
	else:
		machine_format = '>i4'

	##
	# Open and read spectrum
	##
	fid = open(path, 'rb')
	x1 = pow(2, int(nc_proc))
	dim1 = numpy.fromfile(fid, dtype=machine_format)
	spectra_real = (dim1 * x1)
	fid.close()

	##
	# Build ppm scale
	##
	swp = float(sw_p) / float(sf)
	dppm = swp / float(si)
	spectra_ppm = numpy.arange(float(offset), (float(offset) - swp), -dppm)

	return spectra_real, spectra_ppm


def parseQuantFactorSample(path):
	"""
	Parse Bruker QuantFactorSample.xml to get location of ERETIC signal

	:param str path: Path to xml
	:returns: Tuple of ERETIC signal postion in ppm, signal LW in Hz, effective concentration in mM
	:rtype: (float, float, float)
	"""
	tree = ElementTree.parse(path)
	position = float(tree.find('Eretic_Methods/Eretic/Artificial_Eretic_Position').text)
	lineWidth = float(tree.find('Eretic_Methods/Eretic/Artificial_Eretic_Line_Width').text)
	concentration = float(tree.find('Eretic_Methods/Eretic/Artificial_Eretic_Concentration').text)

	return position, lineWidth, concentration
