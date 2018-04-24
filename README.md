# nPYc Toolbox <img src="nPYc/Templates/toolbox_logo.png" width="200" height="200" align="right" />

[![Build Status](https://travis-ci.org/phenomecentre/nPYc-Toolbox.svg?branch=master)](https://travis-ci.org/phenomecentre/nPYc-Toolbox) [![Documentation Status](https://readthedocs.org/projects/npyc-toolbox/badge/?version=latest)](http://npyc-toolbox.readthedocs.io/en/latest/?badge=latest) [![codecov](https://codecov.io/gh/phenomecentre/nPYc-Toolbox/branch/master/graph/badge.svg)](https://codecov.io/gh/phenomecentre/nPYc-Toolbox) ![Python36](https://img.shields.io/badge/python-3.6-blue.svg) [![PyPI](https://img.shields.io/pypi/v/nPYc.svg)](https://pypi.org/project/nPYc/)

* Version 1.0.4

A Python implementation of the [NPC](http://phenomecentre.org) toolchain for the import, quality-control, and preprocessing of metabolic profiling datasets.

Imports:
 - Peak-picked LC-MS data (XCMS, Progenesis QI)
 - Raw NMR spectra (Bruker format)
 - Targeted datasets (TargetLynx, Bruker BI-LISA & BI-Quant-Ur)

Provides:
 - Batch *&* drift correction for LC-MS datasets
 - Feature filtering by RSD and linearity of response
 - Calculation of spectral line-width in NMR
 - PCA of datasets
 - Visualisation of datasets

Exports:
 - Basic tabular csv
 - [ISA-TAB](http://isa-tools.org)

## Installation

To install _via_ [pip](https://pypi.org/project/nPYc/), run:

    pip install nPYc 

To install from a local copy of the source, simply navigate to the main package folder and run:

    python setup.py install

Alternatively, using pip and a local copy of the source:

    pip install /nPYC-toolboxDirectory/

Installation with pip allows the usage of the uninstall command

    pip uninstall nPYc


## Documentation
Documentation is hosted on [Read the Docs](http://npyc-toolbox.readthedocs.io/en/latest/index.html).

Documentation is generated *via* [Sphinx Autodoc](http://www.sphinx-doc.org/), documentation markup is in [reStructuredText](http://docutils.sourceforge.net/rst.html).

To build the documentation locally, cd into the `docs` directory and run:

    make html

To clear the current documentation in order to rebuild after making changes, run:

    make clean

## Development

Source management is [git-flow](http://nvie.com/posts/a-successful-git-branching-model/)-like - no development in the master branch! When making a change, create a fork based on develop, and issue a pull request when ready.

When merging into the develop branch, all new code must include unit-tests, these tests should pass, and overall code-coverage for the toolbox should not drop.


### Releases
When merging from develop (or hotfix branches) into release, ensure:

* All references to the debugger are removed
* All paths are relative and platform agnostic
* All tests pass


### Testing

Unit testing is managed *via* the [`unittest` framework](https://docs.python.org/3.5/library/unittest.html). Test coverage can be found on [codecov.io](https://codecov.io/gh/phenomecentre/nPYc-Toolbox/).

To run all tests, cd into the `Tests` directory and run:

    python -m unittest discover -v

Individual test modules can be run with:

    python -m `test_filename` -v


## Standard measures and codings

When stored internally, and unless explicitly overriden, variables should conform to the units laid out in the [Nomenclature](http://npyc-toolbox.readthedocs.io/en/latest/nomenclature.html) of the documentation.
