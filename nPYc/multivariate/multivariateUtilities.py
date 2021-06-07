import numpy
from scipy.stats import kruskal
from ..utilities._internal import _vcorrcoef
import datetime
import pandas

def pcaSignificance(values, classes, valueType):
	"""
	Local function to calculate whether there is a potential association between values (PCA scores) and classes (sample metadata fields). Either by correlation (continuous data) or Kruskal-Wallis test (categorical data).

	:params numpy.ndarray values: Array of values (e.g., PCA scores)
	:params pandas.series classes: Series of values (e.g., a sample metadata field)
	:params pandas.series valueType: Sample type of each class entry
	"""

	ns, nc = values.shape


	if valueType == 'continuous':
		out = _vcorrcoef(values[classes.notnull().values,:], classes.values[classes.notnull().values])

	elif valueType == 'categorical':

		out = numpy.full([nc], numpy.nan)
		uniq = classes.unique()

		# Count number in each unique group
		nmembers = numpy.zeros([len(uniq)])
		i = 0
		for c in uniq:
			if str(c) in {'nan', 'NaN', 'NaT', '', 'NA'}:
				nmembers[i] = 0
			else:
				nmembers[i] = sum(classes.values == c)
			i = i+1

		for c in numpy.arange(0, nc):
			i = 0
			inputArg = []
			for g in uniq:
				if nmembers[i] >= 5:
					inputArg.append(values[classes.values == g, c])
				i = i+1

			try:
				h_stat, out[c] = kruskal(*inputArg)
			except:
				out = None
			# CAZ need to check if this errors when insufficient numbers in each group

	return out


def metadataTypeGrouping(classes, sampleGroups=None, catVsContRatio=0.75):
	"""
	Local function to calculate whether there is a potential association between values (PCA scores) and classes (sample metadata fields). Either by correlation (continuous data) or Kruskal-Wallis test (categorical data).

	:params pandas.series classes: Series of values (e.g., a sample metadata field)
	:params pandas.series sampleGroups: Sample type of each class entry
	:params float catVsContRatio: Ratio for differentiating numerical categorical from numerical continuous data. If the ratio between number of unique entries/total number of samples exceeds this threshold data is treated as continuous, else data is treated as categorical.
	"""

	ns = classes.shape[0]
	nsnan = ns - sum(classes.isnull()) + 1

	if sampleGroups is None:
		sampleGroups = pandas.Series('Sample' for _ in range(ns))

	# Prep
	myset = set(list(type(classes[i]) for i in range(ns)))
	uniq = classes.unique()

	# If just one group
	if uniq.shape[0] == 1:
		valueType = 'uniform' # uniformClass

	# If numeric and ratio of number of unique groups to number of samples >= 0.75 - calculate correlation
	elif (numpy.issubdtype(classes, numpy.number) & (len(uniq)/nsnan >= catVsContRatio)):
		valueType = 'continuous' # correlation

	# If date
	elif any(my == pandas.Timestamp for my in myset) or any(my == datetime.datetime for my in myset):
		valueType = 'date'

	else:

		# Calculate the number of unique values in each sample type group
		uniqSampleType = sampleGroups.unique()
		nElements = numpy.zeros([len(uniqSampleType)])
		ix=0
		for u in uniqSampleType:
			nElements[ix] = len(classes[sampleGroups.values==u].unique())
			ix = ix+1

		# If classes the same as sampleType
		if all(nElements == 1):
			valueType = 'uniformBySampleType' # uniformClassBySampleType'

		# If all unique values (and not numeric)
		elif (sum(nElements) == ns):
			valueType = 'unique' # uniqueNonNumericClass

		# Else calculate Kruskal-Wallis p-value
		else:
			
			# Only include groups with 5 or more values
			nmembers = numpy.zeros([len(uniq)])
			i = 0
			nnans = 0
			for c in uniq:
				if str(c) in {'nan', 'NaN', 'NaT', '', 'NA'}:
					nmembers[i] = 0
#					nnans = sum(classes.isnull())
					nnans = sum(classes.values.astype(str) == str(c))
				else:
					nmembers[i] = sum(classes.values == c)
				i = i+1

			if sum(nmembers >= 5) >= 2:
				valueType = 'categorical' # KW

			elif ((len(nmembers)==2) & (nnans + sum(nmembers) == len(classes))):
				valueType =  'uniform' # uniformClass
				
			elif (len(nmembers)-1+nnans == ns):
				valueType = 'unique'

			else:
				valueType = 'categorical' # KWinsufficientClassNos

	return valueType
