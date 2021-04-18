import pandas
import numpy
import datetime


def _loadBiocratesDataset(self, path, noSampleParams=15, sheetName='Data Export'):
    # Read in data
    dataT = pandas.read_excel(path, sheet_name=sheetName, skiprows=[0])

    ##
    # Intensity matrix
    ##
    # Find start of data
    endIndex = len(dataT.index)

    # Now read  intensities
    self._intensityData = dataT.iloc[2:endIndex, noSampleParams + 1:].values

    ##
    # Feature info
    ##
    featureMetadata = dict()
    featureMetadata['Feature Name'] = list(dataT.columns.values)[noSampleParams + 1:]
    featureMetadata['Class'] = list(dataT.iloc[0, noSampleParams + 1:].values)
    featureMetadata['LOD (μM)'] = list(dataT.iloc[1, noSampleParams + 1:].values)

    self.featureMetadata = pandas.DataFrame(numpy.vstack([featureMetadata[c] for c in featureMetadata.keys()]).T,
                                            columns=featureMetadata.keys())
    self.featureMetadata['LOD (μM)'] = pandas.to_numeric(self.featureMetadata['LOD (μM)'])
    ##
    # Sample info
    ##
    self.sampleMetadata = pandas.read_excel(path, sheet_name=sheetName, skiprows=[0, 2, 3],
                                            usecols=range(noSampleParams + 1))

    # If there are multiple 'LOD (calc.) ' strings we have several sheets concatenated.
    sampleMask = self.sampleMetadata['Measurement Time'].str.match('LOD \(calc\.\).+').values

    # Take the highest overall LOD
    newLOD = numpy.amax(self._intensityData[sampleMask, :], axis=0)
    self.featureMetadata.loc[self.featureMetadata['LOD (μM)'].values < newLOD, 'LOD (μM)'] = newLOD[
        self.featureMetadata['LOD (μM)'].values < newLOD]
    self.featureMetadata['LOD (μM)'] = pandas.to_numeric(self.featureMetadata['LOD (μM)'])
    # Delete data
    self._intensityData = self._intensityData[sampleMask == False, :]

    # Delete sample data
    self.sampleMetadata = self.sampleMetadata[sampleMask == False]
    self.sampleMetadata.reset_index(drop=True, inplace=True)

    self.sampleMetadata['Collection Date'] = pandas.to_datetime(self.sampleMetadata['Collection Date'])
    self.sampleMetadata['Measurement Time'] = pandas.to_datetime(self.sampleMetadata['Measurement Time'])
    self.sampleMetadata['Sample Bar Code'] = self.sampleMetadata['Sample Bar Code'].astype(int)
    self.sampleMetadata['Well Position'] = self.sampleMetadata['Well Position'].astype(int)
    self.sampleMetadata['Run Number'] = self.sampleMetadata['Run Number'].astype(int)
    self.sampleMetadata['Acquired Time'] = self.sampleMetadata['Measurement Time']

    # Rename sample IDs
    ids = self.sampleMetadata['Sample Identification']
    self.sampleMetadata.drop(labels=['Sample Identification'], axis=1, inplace=True)
    self.sampleMetadata.insert(0, 'Sample ID', ids)

    # Put Feature Names first
    names = self.featureMetadata['Feature Name']
    self.featureMetadata.drop(labels=['Feature Name'], axis=1, inplace=True)
    self.featureMetadata.insert(0, 'Feature Name', names)

    self.sampleMetadata['Metadata Available'] = False
    self.sampleMetadata['Exclusion Details'] = None
    # self.initialiseMasks()

    self.Attributes['Log'].append([datetime.now(), 'Biocrates dataset loaded from %s' % (path)])
