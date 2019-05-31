Tutorials
---------

Installing the nPYc-Toolbox
===========================

Requirements

- Anaconda with Python 3.5 or above `Anaconda Download Link <https://www.anaconda.com/distribution/>`_
- The nPYc-Toolbox can be installed by using the Anaconda Prompt and typing ‘pip install nPYc’, this will install the toolbox alongside any required dependency and make it available as a general python package. Alternatively the toolbox source code can be downloaded directly from the `nPYc-Toolbox GitHub Repository <https://github.com/phenomecentre/nPYc-Toolbox>`_
- The nPYc-toolbox-tutorials, which can be downloaded from the `nPYc-toolbox-tutorial GitHub <https://github.com/phenomecentre/nPYc-toolbox-tutorials>`_

The tutorials below use Jupyter notebooks to demonstrate the application of the nPYc-Toolbox for the preprocessing and quality control of LC-MS and NMR metabolic profiling data, this is the format that we recommend for users. More details on using the Jupyter notebook can be found here `Jupyter ReadTheDocs <https://jupyter.readthedocs.io/en/latest/content-quickstart.html>`_

To open a Jupyter notebook:

Option 1: Graphical User Interface: UI

- Open the Anaconda explorer software and change the current directory to the directory containing the tutorials code and dataset. Launch a Jupyter Notebook session.
 
Option 2: Command line	

- Open the Anaconda prompt console, navigate to the folder where you want to initialize the notebook.
- Then type “jupyter notebook” on the console. This will open the Jupyter Notebook session in the browser.

Tutorial Datasets
=================

The dataset used in these tutorials (DEVSET) is comprised of six samples of pooled human urine, aliquoted, and independently prepared and measured by ultra-performance liquid chromatography coupled to reversed-phase positive ionisation mode spectrometry (LC-MS, RPOS) and 1H nuclear magnetic resonance (NMR) spectroscopy. Each source sample was separately prepared and assayed thirteen times. A pooled QC sample (study reference, SR) and independent external reference (long-term reference, LTR) of a comparable matrix was also acquired to assist in assessing analytical precision. See the `Metabolights Study MTBLS694 <https://www.ebi.ac.uk/metabolights/MTBLS694>`_

The nPYc-toolbox-tutorials contains all the data required to run the tutorial Juypyter notebooks.


Preprocessing and Quality Control of LC-MS Data with the nPYc-Toolbox
=====================================================================

This tutorial demonstrates how to use the LC-MS data processing modules of the nPYc-Toolbox, to import and perform some basic preprocessing and quality control of LC-MS data, and to output a final high quality dataset ready for data modeling.

Required files in nPYc-toolbox-tutorials:

- Preprocessing and Quality Control of LC-MS Data with the nPYc-Toolbox.ipynb: Jupyter notebook tutorial
- DEVSET U RPOS xcms.csv: feature extracted LC-MS data (using `XCMS <https://bioconductor.org/packages/release/bioc/html/xcms.html>`_ )
- DEVSET U RPOS Basic CSV.csv: CSV file containing basic metadata about each of the acquired samples

Additional files (for example, the raw LC-MS data files) can be found in `Metabolights <https://www.ebi.ac.uk/metabolights/MTBLS694>`_


Preprocessing and Quality Control of NMR Data with the nPYc-Toolbox
===================================================================

This tutorial demonstrates how to use the NMR data processing modules of the nPYc-Toolbox, to import and perform some basic preprocessing and quality control of NMR data, and to output a final high quality dataset ready for data modeling.

Required files in nPYc-toolbox-tutorials:

- Preprocessing and quality control of NMR data with the nPYc-Toolbox.ipynb: Jupyter notebook tutorial
- DEVSET U 1D NMR raw data files: folder containing the 1D NMR raw data files (Bruker format)
- DEVSET U 1D NMR Basic CSV.csv: CSV file containing basic metadata about each of the acquired samples


Preprocessing and Quality Control of NMR Targeted Data with the nPYc-Toolbox
============================================================================

*GONCALO!!!!!!!!*

Required files in nPYc-toolbox-tutorials:

- Preprocessing and quality control of targeted NMR data (Bruker IvDr) with the nPYc-toolbox.ipynb: Jupyter notebook tutorial
- DEVSET U 1D NMR raw data files: folder containing the 1D NMR raw data files and the Bruker ivDr xml quantification files
- DEVSET U 1D NMR Basic CSV.csv: CSV file containing basic metadata about each of the acquired samples