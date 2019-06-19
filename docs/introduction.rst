Introduction
------------

The nPYc-Toolbox is a general Python 3 implementation of the MRC-NIHR National Phenome Centre toolchain for the import, quality-control, and preprocessing of metabolic profiling datasets.

The toolbox is built around creating an object for each imported dataset. This object contains the metabolic profiling data itself, alongside all associated sample and feature metadata; various methods for generating, reporting and plotting important quality control parameters; and methods for pre-processing such as filtering poor quality features or correcting trends in batch and run-order.

The following sections describe these, in approximate order of application, in more detail. However, we strongly recommend downloading and working through the :doc:`tutorials<tutorial>` and referring to the documentation when required.


Introduction to Metabolic Profiling
===================================

This section provides a brief introduction to metabolic profiling, the analytical background of the technologies used, and the motivation for the implementation of the nPYc-Toolbox.

See :doc:`metabolicprofiling` for details.


Tutorials
=========

This section provides detailed examples of using the nPYc-Toolbox to import, perform quality-control, and preprocess various types of metabolic profiling datasets.

See :doc:`tutorial` for details.


Recommended Study Design Elements
=================================

This section provides an introduction to recommended sample types and analytical study design elements to ensure standardised quality control (QC) procedures and generate high quality datasets.

See :doc:`studydesign` for details.


Datasets
========

The nPYc-Toolbox is built around a core :py:class:`~nPYc.objects.Dataset` object, which contains the metabolic profiling data itself, alongside all associated sample and feature metadata; various methods for generating, reporting and plotting important quality control parameters; and methods for pre-processing such as filtering poor quality features or correcting trends in batch and run-order. This section gives details of importing data into a Dataset, and gives details of supported data types.

See :doc:`objects` for details.


Sample Metadata
===============

Additional study design parameters or sample metadata may be mapped into the Dataset, this section describes the nomenclature and formats for adding data in order to maximise the utility of the toolbox for quality control.

See :doc:`samplemetadata` for details.


Sample and Feature Masks
========================

Each Dataset object contains a sample and feature masks that store whether a sample or feature, respectively, should be used when calculating QC metrics, in the visualisations in the report functions and when exporting the dataset. This section gives details of the masks, the key functions that modify them and how these are can be used.

See :doc:`masks` for details.


Reports
=======

The nPYc-Toolbox offers a series of `reports`, pre-set visualisations comprised of text, figures and tables to describe and summarise the characteristics of the dataset, and help the user assess the overall impact of quality control decisions.

See :doc:`reports` for details.


Batch *&* Run-Order Correction
==============================

This section describes the tools available to detect, assess and correct longitudinal run-order trends and batch effects in LC-MS datasets.

See :doc:`batchAndROCorrection` for details.


Multivariate Analysis
=====================

The nPYc-Toolbox provides the capacity to generate a PCA model of the data, and subsequently, to use this to assess data quality, identify potential sample and feature outliers, and determine any potential analytical associations with the main sources of variance in the data.

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

Behaviour of many aspects of the toolbox can be modified in a repeatable manner by creating configuration files, this section describes the default configuration files and their parameters across the different methods, and gives information on how to create your own configuration SOPs.

See :doc:`configuration/configuration` for details.


Enumerations
============

The nPYc-Toolbox uses a set of enumerations (complete listings of all possible items in a collection) for common types referenced in profiling experiments.

See :doc:`enumerations` for details.


Utility Functions
=================

This section contains information on the nPYc-Toolbox utility functions, useful functions for working with profiling datasets.

See :doc:`utilities` for details.


Plotting Functions
==================

The :doc:`plots` sections describes the common plots available, both interactive and static version of many plots exist, suitable for use in an interactive setting such as a *Jupyter notebook*, or saving figures for later use.

See the :doc:`Plot Gallery<plotsGallery>` for a visual overview.



