Configuration SOPs
==================

Default parameters for :py:class:`~nPYc.objects.Dataset` objects can be configured as they are initialised by suppling a SOP file containing the desired parameters.

SOP parameters include aesthetic factors such as figure sizes and formats as well as QC and analytical parameters.

By default SOPS are read from the :file:`nPYc/StudyDesigns/SOP/` directory, but this can be overridden by the directory specified in *sopPath=*, that will be searched before the builtin SOP directory.

SOP files are simple `JSON <http://www.json.org>`_ format files, that's contnets are used to populate the :py:attr:`~nPYc.objects.Dataset.Attributes` dictionary. See the default files in :file:`nPYc/StudyDesigns/SOP/` for examples.

.. literalinclude:: ../../nPYc/StudyDesigns/SOP/generic.json
   :caption: The default SOP loaded by all datasets includes settings for figure aesthetics

|

.. table::  Required SOP parameters for all :py:class:`~nPYc.objects.Dataset` objects
	:widths: auto

	================ ========================================= ============
	Key              dtype                                     Usage
	================ ========================================= ============
	'noFiles'        int                                       When showing a ranked list of files show only the top/bottom *noFiles*
	'dpi'            positive int                              Raster resolution when plotting figures
	'figureSize'     positive (float, float)                   Size to plot figures
	'figureFormat'   str                                       Format to save figures in
	'histBins'       positive int                              Number of bins to use when drawing histograms
	'Feature Names'  Column in :py:attr:`featureMetadata`      ID of the primary feature name, a column in :py:attr:`~nPYc.objects.Dataset.sampleMetadata`
	'quantiles'      tuple of (low, high) ints                 When calculating percentiles, use this default range
	================ ========================================= ============

|

.. table::  Additional SOP parameters for all :py:class:`~nPYc.objects.Dataset` objects
	:widths: auto

	======================== ============================== ==================
	Key                      dtype                          Usage
	======================== ============================== ==================
	'analyticalMeasurements' list of str
	'excludeFromPlotting'    list of str
	======================== ============================== ==================

|

.. table:: SOP parameters for all :py:class:`~nPYc.objects.MSDataset` objects
	:widths: auto

	============================= ============================== ==================
	Parameter                     type                           Role
	============================= ============================== ==================
	'rtWindow'                    float                          When grouping features by retention time, use this precision
	'corrThreshold'               float                          When filtering by :term:`linearity reference`, the correlation must be above this
	'corrMethod'                  str                            Type of correlation to linearity to calculate, must be 'pearson' or 'spearman'
	'rsdThreshold'                float                          Default percentage :term:`RSD` cutoff
	'deltaMzArtifactual'          float
	'overlapThresholdArtifactual' float
	'corrThresholdArtifactual'    float
	'msPrecision'                 float
	'varianceRatio'               float
	'blankThreshold'              float
	'filenameSpec'                str (regex)
	============================= ============================== ==================
