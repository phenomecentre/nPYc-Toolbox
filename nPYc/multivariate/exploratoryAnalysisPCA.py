from .pcamodel import PCAmodel
from nPYc.objects._dataset import Dataset
import copy

def exploratoryAnalysisPCA(npycDataset,
                           withExclusions=False,
                           scaling='uv',
                           n_components=6,
                           svd_solver='auto',
                           f_crit_alpha=0.05,
                           minR2=None,
                           **kwargs):
    """

    Performs an exploratory analysis using PCA on the data contained in an :py:class:`~nPYc:objects.Dataset`.

    :param Dataset npycDataset: Dataset to model
    :param Boolean withExclusions: If True, PCA will be fitted on the npyc_dataset after applying feature and sample Mask, if False the PCA is performed on whole dataset.
    :param str scaling: Choice of scaling, either 'mc' for mean-centered only or 'uv' for univariate.
    :param n_components: Number of components to keep. See sklearn.decomposition.PCA n_components.
    :param svd_solver: SVD solver. See sklearn.decomposition.PCA scd_solver.
    :param float f_crit_alpha: Critical value for defining outliers in DModX
    :param minR2: Minimum % of improvement in R2 over the previous component to continue adding components
    :type minR2: None or float
    :return: Fitted PCA model
    :rtype: PCAmodel
    """

    # TODO - try out other options for setting n_components, also minR2 is not implemented yet!! Caro 2026-11-04

    # Some initial checks

    if not isinstance(npycDataset, Dataset):
        raise TypeError('npycDataset argument must be one of the nPYc dataset objects')

    if not isinstance(withExclusions, bool):
        raise TypeError('withExclusions must be a bool')

    if not isinstance(scaling, str) & (scaling in {'uv', 'mc'}):
        raise ValueError('reportType must be either `uv` (univariate) or `mc` (mean-centered only)')

    if minR2 is not None:
        if not isinstance(minR2, float):
            raise TypeError('MinR2 must be `None` or a number')

    # Apply sample/feature masks if exclusions to be applied
    dataset = copy.deepcopy(npycDataset)
    if withExclusions:
        dataset.applyMasks()

    # Fit a PCAmodel
    pcamodel = PCAmodel()

    fittedModel = pcamodel.doPCA(dataset,
                                 n_components=n_components,
                                 scaling=scaling,
                                 svd_solver=svd_solver,
                                 f_crit_alpha=f_crit_alpha,
                                 **kwargs)


    return fittedModel
