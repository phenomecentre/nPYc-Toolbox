Introduction
------------

The nPYc-Toolbox is a general Python 3 implementation of the MRC-NIHR National Phenome Centre toolchain for the import, quality-control, and preprocessing of metabolic profiling datasets.


Tutorials
=========

This section provides detailed examples of using the nPYc-Toolbox to import, perform quality-control, and preprocess various types of metabolic profiling datasets.

See :doc:`tutorial` for details.


Datasets
========

The nPYc-Toolbox is built around a core :py:class:`~nPYc.objects.Dataset` class, that represents a collection of measurements, with biological and analytical metadata associated with each sample, and analytical and chemical metadata associated with each feature. This section gives details of importing data into a Dataset, and describes key Dataset attributes.

See :doc:`objects` for details.


Sample Metadata
===============

Additional study design parameters or sample metadata may be mapped into the Dataset, this section describes the nomenclature and formats for adding data in order to maximise the utility of the toolbox for quality control.

See :doc:`samplemetadata` for details.


Reports
=======

The nPYc-Toolbox offers a series of `reports`, pre-set visualisations comprised of text, figures and tables to describe and summarise the characteristics of the dataset, and help the user assess the overall impact of quality control decisions, these are described in this section.

See :doc:`reports` for details.


Batch *&* Run-Order Correction
==============================

This section describes the tools available to detect, assess and correct longitudinal run-order trends and batch effects in LC-MS datasets.

See :doc:`batchAndROCorrection` for details.


Feature Filtering
=================

Feature filtering describes the removal of certain low-quality or uninformative features from the dataset. How and which features are removed is method specific, and described in this section.

See :doc:`featurefiltering` for details.


Multivariate Analysis
=====================

The nPYc-Toolbox provides the capacity to generate a PCA model of the data, and subesquently, to use this to assess data quality, identify potential sample and feature outliers, and determine any potential analytical associations with the main sources of variance in the data.

See :doc:`multivariate` for details.


Normalisation
=============

This section describes the process for normalising data to correct for dilution effects on global sample intensity.

See :doc:`normalisation` for details.


Exporting Data
==============

This section describes how to export your data (measurements, and feature and sample related metadata).

See :doc:`exportingdata` for details.


Configuration Files
===================

Behaviour of many aspects of the toobox can be modified in a repeatable manner by creating configuration files, this section describes the default configuration files and thier parameters across the different methods, and gives information on how to create your own configuration SOPs.

See :doc:`configuration/configuration` for details.


Enumerations
============

The nPYc-Toolbox uses a set of enumerations (complete listings of all possible items in a collection) for common types referenced in profiling experiments, these are described in this section.

See :doc:`enumerations` for details.


Utility Funtions
================

This section contains information on the nPYc-Toolbox utility functions, useful functions for working with profiling datasets.

See :doc:`utilities` for details.


Plotting Functions
==================

The :doc:`plots` sections describes the common plots available, both interactive and static version of many plots exist, suitable for use in an interactive setting such as a *Jupyter notebook*, or saving figures for later use.

See the :doc:`Plot Gallery<plotsGallery>` for a visual overview.



