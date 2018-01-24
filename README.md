# nPYc Toolbox #

[![Build Status](https://travis-ci.org/phenomecentre/nPYc-Toolbox.svg?branch=master)](https://travis-ci.org/phenomecentre/nPYc-Toolbox) [![Codecov branch](https://img.shields.io/codecov/c/github/phenomecentre/nPYc-Toolbox/master.svg)](https://codecov.io/gh/phenomecentre/nPYc-Toolbox) ![Python36](https://img.shields.io/badge/python-3.6-blue.svg) [![Documentation Status](https://readthedocs.org/projects/npyc-toolbox/badge/?version=latest)](http://npyc-toolbox.readthedocs.io/en/latest/?badge=latest)

A Python implementation of the [NPC](http://phenomecentre.org) toolchain for import, quality-control, and preprocessing of metabolic profiling datasets.

* Version 1.0.1a


## Source control 
Source management is [git-flow](http://nvie.com/posts/a-successful-git-branching-model/)-like - no development in the master branch!


### Development
When merging into the develop branch, all new code must include unit-tests, and these tests should pass.


### Dependancies
The nPYc Toolbox is built upon Python 3.6, and most dependencies can be satisfied by installing the [Anaconda (v4.4.0 and above)](https://www.continuum.io/downloads) distribution.


### Releases
When merging from develop (or hotfix branches) into release, ensure:

* All references to the debugger are removed
* All paths are relative and platform agnostic
* All tests pass


### Documentation
Documentation is hosted on [Read the Docs](http://npyc-toolbox.readthedocs.io/en/latest/index.html).

Documentation is generated *via* [Sphinx Autodoc](http://www.sphinx-doc.org/), documentation markup is in [reStructuredText](http://docutils.sourceforge.net/rst.html).

To build the documentation locally, cd into the `docs` directory and run:

    make html

To clear the current documentation in order to rebuild after making changes, run:

    make clean

### Testing

Unit testing is managed *via* the [`unittest` framework](https://docs.python.org/3.5/library/unittest.html). Test coverage can be found on [codecov.io](https://codecov.io/gh/phenomecentre/nPYc-Toolbox/).

To run all tests, cd into the `Tests` directory and run:

    python -m unittest discover -v

Individual test modules can be run with:

    python -m `test_filename` -v


## Standard measures and codings

When stored internally, and unless explicitly overriden, variables should conform to the units laid out in the [Nomenclature](http://npyc-toolbox.readthedocs.io/en/latest/nomenclature.html) of the documentation.
