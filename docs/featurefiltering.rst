Feature Filtering
-----------------

Feature filtering describes the removal of certain low-quality or uninformative features from the dataset. How and which features are removed is method specific, and described briefly here. More details with examples are given in the :doc:`tutorial`


Feature Filtering in NMR Datasets
=================================

For NMR datasets, feature filtering typically takes the form of removing one or more sections of the spectra known to contain unwanted or un-informative signals.

The regions typically removed are pre-defined in the :doc:`Configuration Files<configuration/builtinSOPs>`, and can be flagged for removal, and subsequently removed using::

	dataset.updateMasks(filterSamples=False, filterFeatures=True)
	dataset.applyMasks()


Feature Filtering in LC-MS Dataset
==================================

For LC-MS datasets, features should be filtered based on their individual precision and accuracy [ref] in the nPYc-Toolbox the default parameters for feature filtering are as follows:

.. table:: LC-MS Feature Filtering Criteria
   :widths: auto
   
   ========================================== ================================================ =================== =====================
   Criteria                                   In                                               Default Value       Assesses
   ========================================== ================================================ =================== =====================
   Correlation to dilution                    :term:`Serial Dilution Sample`                   > 0.7               Intensity responds to changes in abundance (accuracy)
   :term:`Relative Standard Deviation` (RSD)  :term:`Study Reference`                          < 30                Analytical stability (precision)
   RSD in SS * *default value* > RSD in SR    :term:`Study Sample` and :term:`Study Reference` 1.1                 Variation in SS should always be greater than variation in SR
   ========================================== ================================================ =================== =====================
   
The distribution of Correlation to dilution, and RSD can be visualised in the *Feature Summary Report*

A report summarising the number of features passing selection with different criteria can also be produced using::

	nPYc.reports.generateReport(dataset, 'feature selection')

Criteria can be modified if required, for example for the RSD threshold using::

	dataset.Attributes['rsdThreshold'] = 20
	
Features can be flagged for removal, and subsequently removed (as above) using::

	dataset.updateMasks(filterSamples=False, filterFeatures=True)
	dataset.applyMasks()