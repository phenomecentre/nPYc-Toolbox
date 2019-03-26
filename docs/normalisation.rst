Normalisation
-------------

Dilution effects on global sample intensity can be normalised by attaching one of the classes in the :py:mod:`~nPYc.utilities.normalisation` sub-module to the :py:attr:`~nPYc.objects.Dataset.Normalisation` attribute of a :py:class:`~nPYc.objects.Dataset`. 

By default new :py:class:`~nPYc.objects.Dataset` objects have a :py:class:`~nPYc.utilities.normalisation.NullNormaliser` attached, which carries out no normalisation. By assigning an instance of a :py:class:`~nPYc.utilities.normalisation._normaliserABC.Normaliser` class to this attribute::

	totalAreaNormaliser = nPYc.utilities.normalisation.TotalAreaNormaliser
	dataset.Normalisation = totalAreaNormaliser

will cause all calls to :py:attr:`~nPYc.objects.Dataset.intensityData` to return values transformed by the normaliser.


Normalisation
=============

.. automodule:: nPYc.utilities.normalisation
   :members:
