"""
Generic Utility functions
"""
import json, numpy

def rsd(data):
	"""
	Calculate percentage :term:`relative standard deviation` for each column in *data*.

	:math:`\mathit{{rsd(x)}} = \\frac{\mathit{\sigma_{x}}}{\mathit{\mu_{x}}} \\times 100`

	Where RSDs cannot be calculated, (i.e. means of zero), ``numpy.finfo(numpy.float64).max`` is returned.

	:param numpy.ndarray data: *n* by *m* numpy array of data, with features in columns, and samples in rows
	:return: *m* vector of RSDs
	:rtype: numpy.ndarray
	"""

	std = numpy.std(data, axis=0)

	# If std is zero, note it
	stdMask = std == 0
	std[stdMask] = 1

	rsd = numpy.multiply(numpy.divide(std, numpy.mean(data, axis=0)), 100)

	rsd[numpy.isnan(rsd)] = numpy.finfo(numpy.float64).max
	rsd[stdMask] = 0

	return rsd

def removeDuplicateColumns(df):
	"""
	Removes duplicate columns from the passed dataframe
	Looks for columns that end with _x or _y
	"""
	cols = [c for c in df.columns if c[-2:] != '_y']
	df = df[cols]
	df = df.rename(columns=lambda x: x if x[-2:] != '_x' else x.replace('_x', ''))
	return df


def removeTrailingColumnNumbering(column_list):
	"""
	When pandas finds columns with same name, it numbers them
	This function receives a list of column names and removes the numbering if found
	Looks for columns that end with .1, .2, .3 and so on
	"""
	import re
	tmp = []
	for s in column_list:
		x = re.search('\.{1}\d+',s)
		if x != None:
			i = x.span()[0] #index of the .
			tmp.append(s[:i])
		else:
			tmp.append(s)

	return tmp
