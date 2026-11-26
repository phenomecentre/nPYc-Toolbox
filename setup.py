from setuptools import setup, find_packages
from pathlib import Path

# Resolve the path to requirements.txt relative to this setup.py file,
# so it works no matter what directory `pip install` is run from.
HERE = Path(__file__).parent.resolve()

def read_requirements(filename="requirements.txt"):
    req_path = HERE / filename
    with open(req_path, encoding="utf-8") as f:
        lines = f.readlines()

    requirements = []
    for line in lines:
        line = line.strip()
        # Skip blank lines and comments
        if not line or line.startswith("#"):
            continue
        # Skip options like -r other_file.txt or --index-url ...
        if line.startswith("-"):
            continue
        requirements.append(line)
    return requirements

setup(name='nPYc',
	version='3.0',
	description='National Phenome Centre toolbox',
	url='https://github.com/phenomecentre/npyc-toolbox',
	author='National Phenome Centre',
	author_email='phenomecentre@imperial.ac.uk',
	license='MIT',
	packages=find_packages(),
	install_requires=read_requirements("requirements.txt"),
	classifiers = [
		"Programming Language :: Python",
		"Programming Language :: Python :: 3.13",
		"Intended Audience :: Science/Research",
		"License :: OSI Approved :: MIT License",
		"Operating System :: OS Independent",
		"Topic :: Scientific/Engineering :: Bio-Informatics",
	],
	long_description = """\
		Toolbox for preprocessing of metabolic profiling datasets
		---------------------------------------------------------

		.. image:: https://travis-ci.org/phenomecentre/nPYc-Toolbox.svg?branch=master
		   :target: https://travis-ci.org/phenomecentre/nPYc-Toolbox
		   :alt: Travis CI build status

		.. image:: https://readthedocs.org/projects/npyc-toolbox/badge/?version=latest
		   :target: http://npyc-toolbox.readthedocs.io/en/latest/?badge=latest
		   :alt: Documentation Status

		.. image:: https://codecov.io/gh/phenomecentre/nPYc-Toolbox/branch/master/graph/badge.svg
		   :target: https://codecov.io/gh/phenomecentre/nPYc-Toolbox
		   :alt: Test coverage

		|

		The nPYc toolbox offers functions for the import, preprocessing, and QC of metabolic profiling datasets.

		Documentation can be found on `Read the Docs <http://npyc-toolbox.readthedocs.io/en/latest/?badge=latest>`_.

		Imports
		 - Peak-picked LC-MS data (XCMS, Progenesis QI, *&* Metaboscape)
		 - Raw NMR spectra (Bruker format)
		 - Targeted datasets (TargetLynx, Bruker BI-LISA, *&* BI-Quant-Ur)

		Provides
		 - Batch *&* drift correction for LC-MS datasets
		 - Feature filtering by RSD *&* linearity of response
		 - Calculation of spectral line-width in NMR
		 - PCA of datasets
		 - Visualisation of datasets

		Exports
		 - Basic tabular csv
		 - `ISA-TAB <http://isa-tools.org>`_

		The nPYc toolbox is `developed <https://github.com/phenomecentre/npyc-toolbox>`_ by the informatics team at `The National Phenome Centre <http://phenomecentre.org/>`_ at `Imperial College London <http://imperial.ac.uk/>`_.
		""",
        long_description_content_type="text/markdown",
		documentation='http://npyc-toolbox.readthedocs.io/en/latest/?badge=stable',
		include_package_data=True,
		zip_safe=False
	)
