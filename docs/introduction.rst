Introduction
------------

The nPYc-Toolbox is a general Python 3 implementation of the MRC-NIHR National Phenome Centre toolchain for the import, quality-control, and preprocessing of metabolic profiling datasets.


Tutorials
=========

This section provides detailed examples of using the nPYc-Toolbox to import, perform quality-control, and preprocess various types of metabolic profiling datasets.

See :doc:`Using the nPYc-Toolbox<tutorial>` for details.


Datasets
========

The nPYc-Toolbox is built around a core :py:class:`~nPYc.objects.Dataset` class, that represents a collection of measurements, with biological and analytical metadata associated with each sample, and analytical and chemical metadata associated with the observations.

See :doc:`objects` for details.

Sample Metadata
===============

ADD SOME TEXT HERE

Reports
=======

.. automodule:: nPYc.reports

See :doc:`reports`




Enumerations
============

The :py:mod:`~nPYc.enumerations` module defines a set of enums classes for common types and roles used to describe samples and assays in experiments.




Plotting Functions
==================

The :py:mod:`~nPYc.plotting` module defines functions to generate common plots, both interactive and static version of many plots exist, suitable for use in an interactive setting such as a `Jupyter notebook <http://jupyter.org>`_, or saving figures for later use.

See the :doc:`gallery<plotsGallery>` for a visual overview.

Utilities
=========

Utility functions for working with profling datasets, see :doc:`utilities` for an overview.

The :py:mod:`~nPYc.utilities.normalisation` module defines objects that allows for the intensity normalisation of data matrices.

Configuration
=============

Behaviour of many aspects of the toobox can be modified in a repeatable manner by creating configuration files, see :doc:`configuration/builtinSOPs` and :doc:`configuration/targetedSOPs` for examples.

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