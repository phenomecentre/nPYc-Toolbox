import re
import numpy

##
# Harcoded list of adduct mass offsets, taken from ProgenesisQI, v2.3.6198.24128.
##
adducts = {'M+H':(1.007276, 1),
		   'M+K':(38.963158, 1),
		   'M+Na':(22.989218, 1),
		   'M+NH4':(18.0338, 1),
		   'M+CH3OH+H':(33.0335, 1),
		   'M+ACN+H':(42.033823, 1),
		   'M+2Na-H':(44.971160, 1),
		   '2M+H':(1.0073, 0.5),
		   '2M+Na':(22.989218, 0.5),
		   '2M+K':(38.963158, 0.5),
		   'M-H':(-1.007276, 1),
		   'M-H2O-H':(-19.01839, 1),
		   'M+Cl':(34.9694, 1),
		   'M+Na-2H':(20.974666, 1),
		   'M+K-2H':(36.9486, 1),
		   'M+FA-H':(44.9982, 1),
		   'M-2H':(-2.0146, 2),
		   'M-3H':(-3.0218, 3),
		   '2M-H':(-1.0073, 1),
		   '2M+FA-H':(44.9982, 0.5),
		   '2M+Hac-H':(59.0139, 0.5),
		   '3M-H':(-1.0073, 0.3333)
		  }

def buildMassSpectrumFromQIfeature(feature, adducts=adducts):
	"""
	buildMassSpectrumFromQIfeature(feature, adducts=adducts)

	Reconstructs a mass spectrum from a de-isotoped, de-adducted Progenesis QI feature.

	:param dict feature: Dictionary representation of a Progenesis QI feature
	:param dict adducts: Dictionary of *Adduct Name*:(*massoffset*, *massratio*) tuples, defaults to list taken from QI v2.3.6198.24128
	:return: Reconstructed mass spectrum as a list of (*mass*, *abundance*) tuples
	:rtype: list[(float, float),]
	"""

	if not isinstance(feature['Adducts'], str):
		# If no adducts, we only need to worry about isotopes
		spectrum = _buildSpectrumFromQIisotopes(feature['m/z'], feature['Isotope Distribution'])
	else:
		spectrum = list()

		# Parse neutral mass out of the compound name
		mzStr = re.match('\d+?\.\d+?_(\d+?\.\d+?)n', feature['Feature Name'])
		Ma = float(mzStr.group(1))

		for adduct in feature['Adducts'].split(', '):

			# Get mass offset
			if not adduct in adducts.keys():
				raise KeyError("Adduct \''%s\' not found in adducts dictionary." % (adduct))

			(offset, multiplier) = adducts[adduct]
			mz = (Ma / multiplier) + offset

			# Build spectrum
			spectrum.append(_buildSpectrumFromQIisotopes(mz, feature['Isotope Distribution']))
		
		# Flaten list of list
		spectrum = [item for sublist in spectrum for item in sublist]
		
	return spectrum


def _buildSpectrumFromQIisotopes(mz, isotopeDistribution, delta=1.0033550000000009):
	"""
	Build a mass spectrum from a QI mz value, and isotopic distribution pattern.

	:param float mz: m/z of the lightest isotope
	:param str isotopeDistribution: Hyphenated list of isotopic abundances ordered by 1Da intervals
	:param float delta: Isotopic mass difference, defaults to :sup:`13`\ C (1.0033550000000009)
	:returns: Reconstructed mass spectrum as a list of (mass, float) tuples
	:rtype: list[(float, float),]
	"""
	spectrum = list()
	count = 0
	if '-' in isotopeDistribution:
		for isotope in isotopeDistribution.split(' - '):
			spectrum.append((mz + (delta * count), float(isotope)))

			count += 1
	else:
		# No isotopes
		spectrum.append((mz, 100))

	return spectrum
