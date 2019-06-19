Exporting Data
--------------

Datasets can be exported in a variety of formats with the :py:meth:`~nPYc.objects.Dataset.exportDataset` method. 

The default export (*saveFormat=CSV*) results in production of three separate CSV files::

	saveDir = '/path to save outputs'
	dataset.exportDataset(destinationPath=saveDir)

- :py:attr:`~nPYc.objects.Dataset.sampleMetadata`: with a row for every sample and a column for every separate sample-related metadata field
- :py:attr:`~nPYc.objects.Dataset.featureMetadata`: with a row for every feature and a column for each separate feature-related metadata field
- :py:attr:`~nPYc.objects.Dataset.intensityData`: intensity value per variable (column) and sample (row)

An alternative option (*saveFormat=UnifiedCSV*) results in export of a single file, which contains the :py:attr:`~nPYc.objects.Dataset.sampleMetadata`, :py:attr:`~nPYc.objects.Dataset.featureMetadata`, and :py:attr:`~nPYc.objects.Dataset.intensityData` concatenated together, with samples in rows, and features in columns::

	dataset.exportDataset(saveFormat='UnifiedCSV', destinationPath=saveDir)

The nPYc-Toolbox also supports exporting metadata in ISATAB format.

Reports can also be saved to file, see :doc:`reports` for details.