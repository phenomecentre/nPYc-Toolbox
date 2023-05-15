from setuptools import setup, find_packages

setup(name='nPYc',
	version='1.2.7',
	description='National Phenome Centre toolbox',
	url='https://github.com/phenomecentre/npyc-toolbox',
	author='National Phenome Centre',
	author_email='phenomecentre@imperial.ac.uk',
	license='MIT',
	packages=find_packages(),
	install_requires=[
		'cycler>=0.10.0',
		'iPython>=6.3.1',
		#'isaExplorer>=0.1',
		#'isatools>=0.9.3',
		'Jinja2>=3.0.1',
		'lmfit>=0.9.7',
		#'markupsafe==2.0.1',
		'matplotlib==3.5.2',
		'networkx>=2.5.1',
		#'numpy>=1.14.2',
		#'pandas>=0.23.0',
		'numpy~=1.23.3',
		'openpyxl',
        #'jsonschema~=3.2.0',
        'pandas~=1.5.0',
		'plotly>=3.1.0',
		'pyChemometrics>=0.1',
		'scikit-learn>=0.19.1',
		'scipy>=1.1.0',
		'seaborn>=0.8.1',
		'setuptools>=39.1.0',
		'statsmodels>=0.9.0'
	],
	classifiers = [
		"Programming Language :: Python",
		"Programming Language :: Python :: 3.9",
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
