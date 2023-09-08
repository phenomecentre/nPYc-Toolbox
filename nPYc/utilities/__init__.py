"""
The :py:mod:`~nPYc.utilities` module provides convenience functions for working with profiling datasets.
"""
import logging, sys
from .ms import *
from .generic import *
from .extractParams import extractParams, buildFileList
from .normalisation import *
from ._buildSpectrumFromQIfeature import buildMassSpectrumFromQIfeature
from ._massSpectrumBuilder import massSpectrumBuilder

def setupLogger(logger_name=None, logger_file=None):
    
    #print("---->>>logger name is " + str(logger_name))
    logger = logging.getLogger(logger_name)
    if logger_name is None:
        formatter = logging.Formatter('ROOT [%(asctime)s][%(levelname)s]: %(message)s')
        
    else:       
        formatter = logging.Formatter(logger_name + ' [%(asctime)s][%(levelname)s]: %(message)s')


    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.DEBUG)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    
    if logger_file:
        fh = logging.FileHandler()
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    return logger

__all__ = ['rsd', 'normalisation', 'buildFileList', 'buildMassSpectrumFromQIfeature',
           'massSpectrumBuilder', 'sequentialPrecision', 'rsdsBySampleType', 'setupLogger']
