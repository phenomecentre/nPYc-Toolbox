from setuptools import setup

setup(name='nPYc',
	version='1.0.1',
	description='National Phenome Centre toolbox',
	url='https://github.com/phenomecentre/npyc-toolbox',
	author='National Phenome Centre',
	author_email='phenomecentre@imperial.ac.uk',
	license='MIT',
	packages=['nPYc'],
	install_requires=[
		'numpy>=1.11.0',
		'scipy>=0.17.1 ',
		'pandas>=0.21.0',
		'matplotlib>=1.5.1',
		'seaborn>=0.8.0',
		'networkx>=2.0',
		'statsmodels>=0.6.1 ',
		'jinja2>=2.8',
		'plotly>=2.0.0',
		'scikit-learn>=0.18.1',
		'isatools>=0.9.2',
		'lmfit>=0.9.7',
		'cycler>=0.10.0',
		'pyChemometrics>0.11'
	],
	classifiers = [
		"Programming Language :: Python",
		"Programming Language :: Python :: 3.6",
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
		.. image:: https://readthedocs.org/projects/npyc-toolbox/badge/?version=latest
		:target: http://npyc-toolbox.readthedocs.io/en/latest/?badge=latest
		:alt: Documentation Status
		.. image:: https://codecov.io/gh/phenomecentre/nPYc-Toolbox/branch/master/graph/badge.svg
		  :target: https://codecov.io/gh/phenomecentre/nPYc-Toolbox

		The nPYc toolbox offers functions for the import preprocessing and QC of metabolic profiling datasets.

		Imports
		 - Peak-picked LC-MS data (XCMS, Progenesis QI)
		 - Raw NMR spectra (Bruker format)
		 - Targeted datasets (TargetLynx, Bruker BI-LISA & BI-Quant-Ur)
		""",
		documentation='http://npyc-toolbox.readthedocs.io/en/latest/?badge=stable',
	zip_safe=False)
