from pyChemometrics.ChemometricsPCA import ChemometricsPCA
from pyChemometrics.ChemometricsScaler import ChemometricsScaler

from nPYc.objects._dataset import Dataset
import copy


def exploratoryAnalysisPCA(npycDataset, scaling=1, maxComponents=10, minQ2=0.05, withExclusions=False, **kwargs):
    """

    Performs and exploratory analysis using PCA on the data contained in an :py:class:`~nPYc:objects.Dataset`.

    :param Dataset npycDataset: Dataset to model
    :param scaling: Choice of scaling.
    :param int maxComponents: Maximum number of components to fit.
    :param minQ2: Minimum % of improvement in Q2Y over the previous component to add .
    :param Boolean withExclusions: If True, PCA will be fitted on the npyc_dataset after applying feature and sample Mask, if False the PCA is performed on whole dataset.
    :return: Fitted PCA model
    :rtype: ChemometricsPCA
    """

    try:

        if not isinstance(npycDataset, Dataset):
            raise TypeError('npycDataset argument must be one of the nPYc dataset objects')

        if not isinstance(scaling, (float, int)) or (scaling < 0 or scaling > 1):
            raise TypeError('scaling must be a number between 0 and 1. Recommended values are '
                            '0 (mean centring), 1 (Unit Variance)and 0.5 (Pareto)')
        if not isinstance(maxComponents, (float, int)) or maxComponents <= 0:
            raise TypeError('MinQ2 must be a positive number')

        if not isinstance(minQ2, (float, int)):
            raise TypeError('MinQ2 must be a number')


        scaler_obj = ChemometricsScaler(scaling)

        PCAmodel = ChemometricsPCA(ncomps=maxComponents, scaler=scaler_obj)


        # Parse the dara for the cases with exclusion = True and False
        if withExclusions:
            npycDatasetmaskApplied = copy.deepcopy(npycDataset)
            npycDatasetmaskApplied.applyMasks()
            data = npycDatasetmaskApplied.intensityData			

            # generate hash
            # samp_mask_hash = sha1(numpy.ascontiguousarray(npyc_dataset.sampleMask)).hexdigest()
            # feat_mask_hash = sha1(numpy.ascontiguousarray(npyc_dataset.featureMask)).hexdigest()
            # PCAmodel._npyc_hash = {'SampleMask': samp_mask_hash, 'FeatureMask': feat_mask_hash}
        else:
            data = npycDataset.intensityData

        PCAmodel._npyc_dataset_shape = {'NumberSamples': data.shape[0], 'NumberFeatures': data.shape[1]}

        # Do nothing else

        PCAmodel.fit(data)
        scree_cv = PCAmodel._screecv_optimize_ncomps(data, total_comps=maxComponents, stopping_condition=minQ2, **kwargs)

        # After choosing number of components, re-initialize the model

        PCAmodel.ncomps = scree_cv['Scree_n_components']
        
        # Set the miminum number of components to 2
        # TODO: fix plotScores to enable plotting of one component models
        if PCAmodel.ncomps == 1:
            scree_cv = PCAmodel._screecv_optimize_ncomps(data, total_comps=2, stopping_condition=-100000, **kwargs)
            PCAmodel.ncomps = scree_cv['Scree_n_components']
			
        PCAmodel.fit(data, **kwargs)
        # Append the old scree plot to the object
        PCAmodel.cvParameters = scree_cv
        PCAmodel.cvParameters['total_comps'] = maxComponents
        PCAmodel.cvParameters['stopping_condition'] = minQ2
		
        # Cross-validation
        PCAmodel.cross_validation(data, press_impute=False, **kwargs)

        return PCAmodel

    except TypeError as terr:
        raise terr
    except Exception as exp:
        raise exp
