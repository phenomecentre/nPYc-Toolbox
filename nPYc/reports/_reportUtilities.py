import pandas as pd


def _generateFeatureFilteringSummary(dataset):
    """
    Nicely format a summary table of all feature filtering steps applied, with parameters
    """

    # Set up dataframe

    filters = dict()

    # RSD in SR samples
    if dataset.Attributes['featureFilters']['rsdFilter']:
        filters['RSD in SR samples'] = True
        filters['RSD in SR Samples: Threshold'] = dataset.Attributes['filterParameters']['rsdThreshold']
    else:
        filters['Relative Standard Devation (RSD) in SR samples'] = False

    # Correlation to dilution factor
    if dataset.Attributes['featureFilters']['correlationToDilutionFilter']:
        filters['Correlation to Dilution Factor'] = True
        filters['Correlation to Dilution Factor: Method'] = dataset.Attributes['filterParameters']['corrMethod']
        filters['Correlation to Dilution Factor: Threshold'] = dataset.Attributes['filterParameters']['corrThreshold']
    else:
        filters['Correlation to Dilution Factor'] = False

    # RSD of SS Samples > RSD of SR Samples
    if dataset.Attributes['featureFilters']['varianceRatioFilter']:
        filters['RSD of SS Samples > RSD of SR Samples'] = True
        filters['RSD of SS Samples > RSD of SR Samples: Threshold'] = dataset.Attributes['filterParameters']['varianceRatio']
    else:
        filters['RSD of SS Samples > RSD of SR Samples'] = False

    # NOTE: there are also options for artifactualFilter and blankFilter, however, we no longer use artifactualFilter and blankFilter is not implemented

    # Convert to dataframe
    FeatureSelectionTable = pd.DataFrame(filters.items(),
                                         columns=['Feature Filter', 'Applied'])

    FeatureSelectionTable.set_index('Feature Filter', inplace=True)

    FeatureSelectionTable.index.name = None

    return FeatureSelectionTable


