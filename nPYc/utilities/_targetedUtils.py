import numpy


def calcAccuracy(measuredConc, expectedConc):
    """
    Calculate the accuracy of measurement for a column of data.
    accuracy = (mean(measuredConcentration)/expectedConcentration)*100

    :param numpy.ndarray measuredConc: *n* by 1 numpy array of data, with a single feature in column, and samples in rows
    :param float expectedConc: expected concentration
    :return: accuracy value
    :rtype: float
    """
    accuracy = (numpy.mean(measuredConc) / expectedConc) * 100
    return accuracy


def calcPrecision(measuredConc):
    """
    Calculate the precision of measurement (percent RSD) for a column of data.
    Allow for -inf, inf values in input.

    :param numpy.ndarray measuredConc: *n* by 1 numpy array of data, with a single feature in column, and samples in rows
    :return: precisin value
    :rtype: float
    """
    std = numpy.std(measuredConc)
    rsd = (std / numpy.mean(measuredConc)) * 100
    if numpy.isnan(rsd):
        rsd = numpy.inf
    return rsd
