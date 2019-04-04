Introduction
------------

The nPYc-Toolbox is a general Python 3 implementation of the MRC-NIHR National Phenome Centre toolchain for the import, quality-control, and preprocessing of metabolic profiling datasets.


Tutorials
=========

This section provides detailed examples of using the nPYc-Toolbox to import, perform quality-control, and preprocess various types of metabolic profiling datasets.

See :doc:`tutorial` for details.


Datasets
========

The nPYc-Toolbox is built around a core :py:class:`~nPYc.objects.Dataset` class, that represents a collection of measurements, with biological and analytical metadata associated with each sample, and analytical and chemical metadata associated with each feature.

See :doc:`objects` for details.


Sample Metadata
===============

Additional study design parameters or sample metadata may be mapped into the Dataset using the :py:meth:`~nPYc.objects.Dataset.addSampleInfo` method.

See :doc:`samplemetadata` for details.


Reports
=======

.. automodule:: nPYc.reports

See :doc:`reports` for details.


Batch *&* Run-Order Correction
==============================

Functions for detecting and correcting longitudinal run-order trends and batch effects in datasets.

See :doc:`batchAndROCorrection` for details.


Multivariate analysis
=====================

Functions for generating and working multivariate models of profling datasets.

See :doc:`multivariate` for details.


Normalisation
=============

Funtion for normalising data to correct for dilution effects on global sample intensity.

See :doc:`normalisation` for details.


Exporting Data
==============

Function for exporting the data (measurements, and feature and sample related metadata).

See :doc:`exportingdata` for details.


Configuration Files
===================

Behaviour of many aspects of the toobox can be modified in a repeatable manner by creating configuration files.

See :doc:`configuration/configuration` for details.


Enumerations
============

The :py:mod:`~nPYc.enumerations` module defines a set of enums classes for common types and roles used to describe samples and assays in experiments.


Utility Funtions
================

Utility functions for working with profling datasets, see :doc:`utilities` for an overview.

The :py:mod:`~nPYc.utilities.normalisation` module defines objects that allows for the intensity normalisation of data matrices.


Plotting Functions
==================

The :py:mod:`~nPYc.plotting` module defines functions to generate common plots, both interactive and static version of many plots exist, suitable for use in an interactive setting such as a `Jupyter notebook <http://jupyter.org>`_, or saving figures for later use.

See the :doc:`Plot Gallery<plotsGallery>` for a visual overview.



