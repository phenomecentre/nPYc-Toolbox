"""
The `nPYc-Toolbox <https://github.com/phenomecentre/nPYc-Toolbox>`_ defines objects for representing, and implements functions to manipulate and display, metabolic profiling datasets.
"""
from . import enumerations
from .objects import Dataset, MSDataset, NMRDataset, TargetedDataset
from . import utilities
from . import plotting
from . import reports
from . import batchAndROCorrection
from . import multivariate

import os

path = os.path.realpath(__file__)
path = os.path.dirname(path)
path = os.path.join(path, 'VERSION')

with open(path, 'r') as file:
	__version__ = file.readline().strip()

__all__ = ['Dataset', 'MSDataset', 'plotting', 'reports', 'extractParams', 'NMRDataset', 'multivariate', 'TargetedDataset']
