"""
The nPYc toolbox is built around a core :py:class:`Dataset` class, that represents a collection of measurements, with biological and analytical metadata associated with each sample, and analytical and chemical metadata associated with the observations.

Instances of the Dataset class and its subclasses are capable of instantiating themselves from common data types, including certain raw data formats, common interchange formats, and the outputs of popular data-processing tools.

The Dataset family of class include methods for mapping additional metadata into the object (see :py:meth:`~Dataset.addSampleInfo`), as well exporting a representation of themself in a variety of formats (see the :py:meth:`~Dataset.exportDataset` method)

All children of **Dataset** have three primary attributes:

* :py:attr:`~Dataset.sampleMetadata`: A :math:`n` × :math:`p` dataframe of sample identifiers and metadata
* :py:attr:`~Dataset.featureMetadata`: A :math:`m` × :math:`q`  pandas dataframe of feature identifiers and metadata
* :py:attr:`~Dataset.intensityData`: A :math:`n` × :math:`m` numpy matrix of measurements

.. figure:: _static/Dataset_structure.svg
	:alt: Structure of the key attributes of a dataset
	
	Structure of the key attributes of a :py:class:`Dataset` object. Of note, rows in the :py:attr:`~Dataset.featureMetadata` Dataframe correspond to columns in the :py:attr:`~Dataset.intensityData` matrix.

When initialised, :py:class:`Dataset` objects can be configured by loading :doc:`SOP parameters<configuration/configurationSOPs>` from JSON files specified in *sop*. The parameters are then stored in the :py:attr:`~Dataset.Attributes` dictionary.
"""
from ._dataset import Dataset
from ._msDataset import MSDataset
from ._nmrDataset import NMRDataset
from ._targetedDataset import TargetedDataset

__all__ = ['Dataset', 'MSDataset', 'NMRDataset','TargetedDataset']
