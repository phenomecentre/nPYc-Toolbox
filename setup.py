from setuptools import setup, find_packages
import os

basepath = os.path.realpath(__file__)
basepath = os.path.dirname(basepath)
path = os.path.join(basepath, 'nPYc', 'VERSION')

with open(path, 'r') as file:
	VERSION = file.readline().strip()

path = os.path.join(basepath, 'README.md')

with open(path, 'r') as file:
	README = file.read()

setup(name='nPYc',
	version=VERSION,
	description='National Phenome Centre toolbox',
	url='https://github.com/phenomecentre/npyc-toolbox',
	author='National Phenome Centre',
	author_email='phenomecentre@imperial.ac.uk',
	license='MIT',
	packages=find_packages(),
	install_requires=[
		'cycler>=0.10.0',
		'iPython>=6.3.1',
		'isaExplorer>=0.1',
		'isatools>=0.9.3',
		'Jinja2>=2.10',
		'lmfit>=0.9.7',
		'matplotlib>=2.2.2',
		'networkx>=2.1',
		'numpy>=1.14.2',
		'pandas>=0.23.0',
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
		"Programming Language :: Python :: 3.6",
		"Intended Audience :: Science/Research",
		"License :: OSI Approved :: MIT License",
		"Operating System :: OS Independent",
		"Topic :: Scientific/Engineering :: Bio-Informatics",
	],
	long_description_content_type='text/markdown',
	long_description = README,
	documentation='http://npyc-toolbox.readthedocs.io/en/latest/?badge=stable',
	include_package_data=True,
	zip_safe=False
	)
