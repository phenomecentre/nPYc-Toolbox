# nPYc Toolbox <img src="nPYc/Templates/toolbox_logo.png" width="200" style="max-width: 30%;" align="right" />

[![Build Status](https://travis-ci.com/phenomecentre/nPYc-Toolbox.svg?branch=master)](https://travis-ci.com/phenomecentre/nPYc-Toolbox) [![Documentation Status](https://readthedocs.org/projects/npyc-toolbox/badge/?version=latest)](http://npyc-toolbox.readthedocs.io/en/latest/?badge=latest) [![codecov](https://codecov.io/gh/phenomecentre/nPYc-Toolbox/branch/master/graph/badge.svg)](https://codecov.io/gh/phenomecentre/nPYc-Toolbox) ![Pythonv](https://img.shields.io/pypi/pyversions/nPYc) [![PyPI](https://img.shields.io/pypi/v/nPYc.svg)](https://pypi.org/project/nPYc/)

A Python implementation of the [NPC](http://phenomecentre.org) toolchain for the import, quality-control, and preprocessing of metabolic profiling datasets.

Imports:

 - Peak-picked LC-MS data (XCMS, Progenesis&nbsp;QI, *&* Metaboscape)
 - Raw NMR spectra (Bruker format)
 - Targeted datasets (TargetLynx, Bruker BI-LISA *&* BI-Quant-Ur)

Provides:

 - Batch *&* drift correction for LC-MS datasets
 - Feature filtering by RSD and linearity of response
 - Calculation of spectral line-width in NMR
 - PCA of datasets
 - Visualisation of datasets

Exports:

 - Basic tabular csv
 - [ISA-TAB](http://isa-tools.org)
 
Tutorials:
 
 - Available at [nPYc-toolbox-tutorials](https://github.com/phenomecentre/nPYc-toolbox-tutorials), see below.

## Installation

For full installation instructions see [Installing the nPYc-Toolbox](https://npyc-toolbox.readthedocs.io/en/latest/tutorial.html#installing-the-npyc-toolbox)

To install _via_ [pip](https://pypi.org/project/nPYc/), run:

    pip install nPYc

To install from a local copy of the source, simply navigate to the main package folder and run:

    python setup.py install

Alternatively, using pip and a local copy of the source:

    pip install /nPYC-toolboxDirectory/

To update the current installed version use:

    pip install --upgrade nPYc

Installation with pip allows the usage of the uninstall command

    pip uninstall nPYc


## Documentation
Documentation is hosted on [Read the Docs](http://npyc-toolbox.readthedocs.io/en/latest/index.html).

Documentation is generated *via* [Sphinx Autodoc](http://www.sphinx-doc.org/), documentation markup is in [reStructuredText](http://docutils.sourceforge.net/rst.html).

To build the documentation locally, cd into the `docs` directory and run:

    make html

To clear the current documentation in order to rebuild after making changes, run:

    make clean


## Tutorials

A repository containing exemplar datasets and Jupyter notebook tutorials to demonstrate the application of the nPYc-Toolbox for the preprocessing and quality control of LC-MS, NMR and targeted NMR (Bruker IVDr) metabolic profiling data is available for download from [nPYc-toolbox-tutorials](https://github.com/phenomecentre/nPYc-toolbox-tutorials). 

For new users, we strongly recommend downloading these tutorials, which provide detailed worked examples with links to relevant documentation.

## Development

Source management is [git-flow](http://nvie.com/posts/a-successful-git-branching-model/)-like - no development in the master branch! When making a change, create a fork based on develop, and issue a pull request when ready.

When merging into the develop branch, all new code must include unit-tests, all tests should pass, and overall code-coverage for the toolbox should not drop.


### Releases
When merging from develop (or hotfix branches) into release, ensure:

 - All references to the debugger are removed
 - All paths are relative and platform agnostic
 - All tests pass


### Testing

Unit testing is managed *via* the [`unittest` framework](https://docs.python.org/3.5/library/unittest.html). Test coverage can be found on [codecov.io](https://codecov.io/gh/phenomecentre/nPYc-Toolbox/).

To run all tests, cd into the `Tests` directory and run:

    python -m unittest discover -v

Individual test modules can be run with:

    python -m `test_filename` -v
