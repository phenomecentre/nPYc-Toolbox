Exporting Data
--------------

Datasets can be exported in a variety of formats with the :py:meth:`~nPYc.objects.Dataset.exportDataset` method. 

The default export '*saveFormat=CSV*' results in production of three separate CSV files:

- :py:attr:`~nPYc.objects.Dataset.sampleMetadata`: with a row for every sample and a column for every separate sample-related metadata field
- :py:attr:`~nPYc.objects.Dataset.featureMetadata`: with a row for every feature and a column for each separate feature-related metadata field
- :py:attr:`~nPYc.objects.Dataset.intensityData`: intensity value per variable (column) and sample (row)

'*saveFormat=UnifiedCSV*' provides an alternative output, exporting the :py:attr:`~nPYc.objects.Dataset.sampleMetadata`, :py:attr:`~nPYc.objects.Dataset.featureMetadata`, and :py:attr:`~nPYc.objects.Dataset.intensityData` concatenated as a single CSV file, with samples in rows, and features in columns.

The nPYc toolbox also supports exporting metadata in ISATAB format.