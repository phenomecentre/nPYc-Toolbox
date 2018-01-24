import numpy

def checkInRange(values, lowerBracket, upperBracket):
	"""
	Check if the distribution sampled in **values** fits within the (bound, value) tuple provided. If ``None`` is substituted for a tuple that test is omitted.

	:param numpy.ndarray values: Array of values to check, if the array is greater than 1D it is flattened
	:param lowerBracket: Tuple of (lower percentile, bound) values
	:type lowerBracket: (float, float) or None
	:param upperBracket: Tuple of (upper percentile, bound) values
	:type upperBracket: (float, float) or None
	:rtype: Bool
	"""

	sampleCount = values.size

	##
	# Start assuming we are ok
	##
	inRange = True

	if lowerBracket:
		outOfBoundsCount = values < lowerBracket[1]

		outOfBoundsCount = numpy.sum(outOfBoundsCount)

		outOfBoundsPercentage = (outOfBoundsCount / sampleCount) * 100

		inRange = inRange and (outOfBoundsPercentage < lowerBracket[0])

	if upperBracket:
		outOfBoundsCount = values > upperBracket[1]

		outOfBoundsCount = numpy.sum(outOfBoundsCount)

		outOfBoundsPercentage = (outOfBoundsCount / sampleCount) * 100

		inRange = inRange and (outOfBoundsPercentage < (100 - upperBracket[0]))

	return inRange
