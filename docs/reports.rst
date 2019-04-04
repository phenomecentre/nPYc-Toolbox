Reports
-------

The nPYc-Toolbox offers a series of `reports`, pre-set visualizations comprised of text, figures and tables to describe and summarise the characteristics of the dataset, and help the user assess the overall impact of quality control decisions (e.g. whether to exclude samples or features or change filtering criteria).

There are four reports available for all data types:

•	Sample Summary: Presents a breakdown of available samples per sample type. If metadata from a csv file was added, extra information about mismatches between expected and acquired samples is given
•	Feature Summary: Summarises the main properties of the dataset and method specific quality control metrics
•	Multivariate Report: Plots the main outputs of a PCA model (R2/Q2, scores and loading plots) and also any potential associations between pertinent analytical metadata and the scores values
•	Final Report: Summarises report compiling information about the samples acquired, the overall quality of the dataset

These can be generated using (e.g. for the Feature Summary Report)::

	nPYc.reports.generateReport(dataset, 'feature summary')
	
In addition, there are a number of method-specific reports, see the method-specific sections below and :doc:`tutorial` for full details.


Quality Control
===============

The nPYc-Toolbox incorporates the concept of analytical quality directly into the subclasses of :py:class:`~nPYc.objects.Dataset`. Depending on the analytical platform and protocol, quality metrics may be judged on the basis of sample-by-sample or feature-by-feature comparisons, or both.

Parameters for the quality control procedure can be specified in the :doc:`Configuration File<../configuration/builtinSOPs>` as the Dataset object is created, and amended after creation by modifying the relevant entry of the :py:attr:`~nPYc.objects.Dataset.Attributes` dictionary.

The reporting functions have been specifically designed to summarise all important aspects of quality control throughout import and preprocessing, see the method-specific sections below and :doc:`tutorial` for full details.


Saving Reports
==============

By default, reports are generated inline (i.e. in a Jupyter notebook), however reports can also be saved as html documents with static images by supplying a destination path, for example::

	saveDir = '/path to save outputs'
	nPYc.reports.generateReport(dataset, 'feature summary', destinationPath=saveDir)


Templates
=========

Reporting used when saving reports as HTML are based on Jinja2, default reports are saved in the `Templates` directory, these may be customised if required.


Sample Report
=============

.. autoclass:: nPYc.reports._generateSampleReport
  :members:


LC-MS Reports
=============

.. autoclass:: nPYc.reports._generateReportMS
  :members:
  
  
NMR Reports
===========

.. autoclass:: nPYc.reports._generateReportNMR
  :members:

Targeted Reports
================

.. autoclass:: nPYc.reports._generateReportTargeted
  :members:
  
Multivariate Report
===================

.. autoclass:: nPYc.reports.multivariateReport
  :members: