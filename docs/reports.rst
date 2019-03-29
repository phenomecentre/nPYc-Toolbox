Reporting functions
-------------------

The nPYc-Toolbox offers a series of `reports`, pre-set visualizations comprised of text, figures and tables to describe and summarize the characteristics of the dataset, and help the user assess the overall impact of quality control decisions (ie, excluding samples or features and changing filtering criteria).


The main reports available are:

•	Sample Report: Presents a breakdown of available samples per sample type. If metadata from a csv file was added, extra information about expected samples which were not present in and extra samples not mentioned in the CSV file is given
•	Feature summary report: Summarizes the main properties of the dataset and technology specific quality control metrics
•	Multivariate Report: Plots the main outputs of a PCA model (R2/Q2, scores and loading plots) and also the potential association between pertinent analytical metadata and the scores values
•	Final report: Summary report compiling information about the samples acquired, the overall quality of the dataset

The nPYc toolbox incorporates the concept of analytical quality directly into the subclasses of :py:class:`~nPYc.objects.Dataset`. Depending on the analytical platform and protocol, quality metrics may be judged on the basis of sample-by-sample or feature-by-feature comparisons, or both.

Parameters for the quality control procedure can be specified in a :doc:`SOP JSON<../configuration/configurationSOPs>` file as the Dataset object is created, and amended after creation by modifying the relevant entry of the :py:attr:`~nPYc.objects.Dataset.Attributes` dictionary.

Each object type supports its own QC tests, see the tutorials for more details :doc:`tutorial`.

Saving reports
==============

Report generated interactively by the :py:mod:`~nPYc.reports` module can be saved as html documents with static images by supplying a path in which to save the report and figures to the *output=* parameter of the :py:mod:`~nPYc.reports.generateReport` function.


Templates
=========
Reporting used when saving reports as HTML are based on Jinja2, and may be customised by modifying the template documents in the `Templates` directory.


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