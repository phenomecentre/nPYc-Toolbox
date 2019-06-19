Recommended Study Design Elements
---------------------------------

For the purpose of standardising quality control (QC) procedures within the pipeline and generating high quality datasets, the nPYc-Toolbox defines a recommended set of reference sample types and design elements, based on quality control criteria previously described (Dona *et al* [#]_, Lewis *et al* [#]_).

One key element in this design is the use of a pooled QC sample, comprised of a mixture of aliquots taken from every sample in the study. The nature of the pooled sample, as a physical average of all samples in the study, guarantees that it will contain representative levels of the majority of compounds present in the samples, including previously unobserved molecules, which is particularly important in profiling studies where the constituents of the sample matrix are not known up-front.

This comprehensiveness allows the pooled QC to be useful in many ways, including, for example, in the generation of measures of analytical precision, such as calculating :term:`Relative Standard Deviation`, accuracy, for example, in calculating :term:`Correlation to Dilution` and to detect and potentially remove analytical batch and run-order effects (see :doc:`batchAndROCorrection`).

The following section describes recommended study design elements and key reference QC samples in more detail, alongside how these samples are defined when using the nPYc-Toolbox.


Sample and Study Design Nomenclature
====================================

The :mod:`nPYc` toolbox uses the following nomenclature when defining sample types and analytical study design elements. Certain terms are defined and controlled in the :mod:`~nPYc.enumerations` module.

.. figure:: _static/samplingNomenclature.svg
	:width: 70%
	:align: center
	:alt: Generation of samples
	
	Generation of samples
	 
The hierarchy of sample generation, :term:`Study samples<study sample>` are generated from :term:`participants<participant>` at one or more :term:`sampling events<sampling event>`. These sample are then :term:`assayed<assay>` by one or more methods, generating a unique dataset for each :term:`sample assay`.
	
In order to estimate analytical quality in a robust and extensible fashion, the nPYc-Toolbox characterises the samples constituting a study by two parameters; the sample type, i.e., the source and composition of the sample, and the assay role, the rational for a specific acquisition of data.
	
Sample Types are described in detail here :py:class:`~nPYc.enumerations.SampleType`, the most common are:

- 'Study Sample' comprise the study in question
- 'Study Pool' a mixture made from pooling aliquots from all/some study samples
- 'External Reference' a sample of a comparable matrix to the study samples, but not derived from samples acquired as part of the study
	
Assay Roles are described in detail here :py:class:`~nPYc.enumerations.AssayRole`, the most common are:

- 'Assay' form the core of an analysis
- 'Precision Reference' acquired to characterise analytical variability
- 'Linearity Reference' samples used assess the linearity of response (or :term:`Correlation to Dilution`) in the dataset

The main samples comprising the study are named :term:`Study Sample` (SS), and are a *Study Sample*, *Assay* combination.

In addition, common combinations of *Sample Type* and *Assay Role* are defined within the nPYc-Toolbox and used to characterise data quality, these include:

- :term:`Study Reference` (SR): A *Study Pool*, *Precision Reference* combination used to assess analytical stability across the acquisition run (such as :term:`Relative Standard Deviation`)
- :term:`Long-Term Reference` (LTR): An *External Reference*, *Precision Reference* combination used to assess analytical stability across the acquisition run, and furthermore between different studies
- :term:`Serial Dilution Sample` (SRD): A *Study Pool*, *Linearity Reference* combination used to assess linearity of response, often by repeated injection at varying concentrations or levels of dilution (see :term:`Correlation to Dilution`)

When using the nPYc-Toolbox, acquired samples can be matched to their experimental details (for example, reference sample type or associated biological metadata) as described in the :doc:`samplemetadata` section.


.. [#] Anthony C Dona, Beatriz Jiménez, Hartmut Schäfer, Eberhard Humpfer, Manfred Spraul, Matthew R Lewis, Jake TM Pearce, Elaine Holmes, John C Lindon and Jeremy K Nicholson. Precision High-Throughput Proton NMR Spectroscopy of Human Urine, Serum, and Plasma for Large-Scale Metabolic Phenotyping. Analytical Chemistry, 86(19):9887-9894, 2014. URL: http://dx.doi.org/10.1021/ac5025039
	
.. [#] Matthew R Lewis, Jake TM Pearce, Konstantina Spagou, Martin Green, Anthony C Dona, Ada HY Yuen, Mark David, David J Berry, Katie Chappell, Verena Horneffer-van der Sluis, Rachel Shaw, Simon Lovestone, Paul Elliott, John Shockcor, John C Lindon, Olivier Cloarec, Zoltan Takats, Elaine Holmes and Jeremy K Nicholson. Development and Application of Ultra-Performance Liquid Chromatography-TOF MS for Precision Large Scale Urinary Metabolic Phenotyping. Analytical Chemistry, 88(18):9004-9013, 2016. URL: http://dx.doi.org/10.1021/acs.analchem.6b01481

