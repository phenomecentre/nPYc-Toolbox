from setuptools import setup

setup(name='nPYc',
	version='1.0.0',
	description='National Phenome Centre toolset',
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
	zip_safe=False)
