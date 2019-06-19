Normalisation
-------------

Dilution effects on global sample intensity can be normalised by attaching one of the classes in the :py:mod:`~nPYc.utilities.normalisation` sub-module to the :py:attr:`~nPYc.objects.Dataset.Normalisation` attribute of a :py:class:`~nPYc.objects.Dataset`. 

By default new :py:class:`~nPYc.objects.Dataset` objects have a :py:class:`~nPYc.utilities.normalisation.NullNormaliser` attached, which carries out no normalisation. By assigning an instance of a :py:class:`~nPYc.utilities.normalisation._normaliserABC.Normaliser` class all calls to :py:attr:`~nPYc.objects.Dataset.intensityData` to return values transformed by the normaliser. For example, to return total area normalised values::

	totalAreaNormaliser = nPYc.utilities.normalisation.TotalAreaNormaliser
	dataset.Normalisation = totalAreaNormaliser

There are three built-in normalisation objects:

- Null normaliser (:py:class:`~nPYc.utilities.normalisation.NullNormaliser`): no normalisation performed
- Probabilistic quotient normaliser (:py:class:`~nPYc.utilities.normalisation.ProbabilisticQuotientNormaliser`): performs probabilistic quotient normalisation (Dieterle *et al.* [#]_ )
- Total area normaliser (:py:class:`~nPYc.utilities.normalisation.TotalAreaNormaliser`): performs normalisation where each row (sample) is divided by the total sum of its variables (columns)


Normalisation Syntax and Parameters
===================================

The main function parameters (which may be of interest to advanced users) are as follows:

.. automodule:: nPYc.utilities.normalisation
   :members:
   
.. [#] Frank Dieterle, Alfred Ross, Götz Schlotterbeck and Hans Senn. Probabilistic quotient normalization as robust method to account for dilution of complex biological mixtures. application in :sup:`1`\ H NMR metabonomics. Analytical Chemistry, 78(13):4281 – 90, 2006. URL: https://pubs.acs.org/doi/10.1021/ac051632c