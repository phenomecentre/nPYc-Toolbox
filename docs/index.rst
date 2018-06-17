.. nPYc Toolbox documentation master file, created by
   sphinx-quickstart on Mon Jan 16 09:06:56 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

nPYc Toolbox
------------

.. automodule:: nPYc

Contents:

.. toctree::
   :maxdepth: 2
   :includehidden:
   :glob:

   tutorials/tutorial

   objects
   enumerations
   reports
   plots
   utilities
   configuration/configuration
   batchAndROCorrection
   multivariate
   nomenclature


Using the nPYc toolbox
======================
The :doc:`tutorial <tutorials/tutorial>` introduces the basic process of using the nPYc toolbox to load, perform quality-control on, and basic interpretation of a metabolic profiling dataset.


The :py:class:`~nPYc.objects.Dataset` classes
=============================================

The nPYc toolbox is built around a core :py:class:`~nPYc.objects.Dataset` class, that represents a collection of measurements, with biological and analytical metadata associated with each sample, and analytical and chemical metadata associated with the observations.

Instances of the Dataset class and its subclasses are capable of instantiating themselves from common data types, including certain raw data formats, common interchange formats, and the outputs of popular data-processing tools.

See :doc:`objects` for details.

Enumerations
============

The :py:mod:`~nPYc.enumerations` module defines a set of enums classes for common types and roles used to describe samples and assays in experiments.


Standard reports
================

.. automodule:: nPYc.reports

See :doc:`reports`

Plots
=====

The :py:mod:`~nPYc.plotting` module defines functions to generate common plots, both interactive and static version of many plots exist, suitable for use in an interactive setting such as a `Jupyter notebook <http://jupyter.org>`_, or saving figures for later use.

See the :doc:`gallery<plotsGallery>` for a visual overview.

Utilities
=========

Utility functions for working with profling datasets, see :doc:`utilities` for an overview.

The :py:mod:`~nPYc.utilities.normalisation` module defines objects that allows for the intensity normalisation of data matrices.

Configuration
=============

Behaviour of many aspects of the toobox can be modified in a repeatable manner by creating configuration files, see :doc:`configuration/configurationSOPs` and :doc:`configuration/targetedSOPs` for examples.

Batch *&* run-order correction
==============================

Functions for detecting and correcting longitudinal run-order trends and batch effects in datasets.

See :doc:`batchAndROCorrection`

Multivariate analysis
=====================

Functions for generating and working multivariate models of profling datasets.

See :doc:`multivariate`

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
