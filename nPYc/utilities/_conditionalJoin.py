def conditionalJoin(first, second, separator='; '):
	"""
	Join two strings with seperator, if one string is empty or none, return the other unmodified.

	:param first: String to join infront of `separator`
	:type str
	:param second: String to join behind of `separator`
	:type str

	:param separator: String to join `first` and `second`. Defaults to '; '.
	:type str

	:returns
	:type MSDataset
	"""
	if not isinstance(separator, str):
		raise TypeError('`separator` must be a string.')

	# If either argument is None, skip
	if (first is None) or first == '':
		return second
	elif (second is None) or (second == ''):
		return first

	return separator.join((first, second))
