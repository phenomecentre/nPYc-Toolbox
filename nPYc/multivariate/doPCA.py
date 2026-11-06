import copy

from networkx.algorithms.components import number_attracting_components
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats
import pandas as pd
import numpy as np

from nPYc import MSDataset


# Can we just use sklearn-PCA?
class PCAmodel:
    """
    Base class for nPYc PCA objects, using scikit-learn's PCA and formatting output for ease of use
    """

    def __init__(self):

        # DataFrame to save sample metadata including scores
        self.sampleMetadata = pd.DataFrame(None)

        # DataFrame to save feature metadata including loadings
        self.featureMetadata = pd.DataFrame(None)

        self.scores = None
        self.loadings = None
        self.inputparams = dict()
        self.outputstats = dict()
        self.modelParameters = dict() # Tidy this up in due course
        self._isfitted = False

    def doPCA(self, dataset, withExclusions=False, scaling='uv', n_components=6, svd_solver='auto', f_crit_alpha = 0.05, **kwargs):

        """
        Perform PCA on the provided data matrix.

        Uses scikit-learn PCA: see documentation for available options
        """

        # Initial checks


        # Apply sample/feature masks if exclusions to be applied
        msData = copy.deepcopy(dataset)
        if withExclusions:
            msData.applyMasks()

        self._npyc_dataset_shape = {'NumberSamples': msData.intensityData.shape[0], 'NumberFeatures': msData.intensityData.shape[1]}

        # Scale features
        if scaling == 'uv': # Means-centre and uv scale
            scaler = StandardScaler(with_mean=True, with_std=True)
        else: # Means-centre only
            scaler = StandardScaler(with_mean=True, with_std=False)

        # Scale data
        x_scaled = scaler.fit_transform(msData.intensityData)

        # Fit PCA
        pca = PCA(n_components=n_components, svd_solver=svd_solver, **kwargs)
        self.scores = pca.fit_transform(x_scaled)

        # If auto-fit with one component, re-fit with two for plotting convenience
        if self.scores.shape[1] == 1:
            n_components = 2
            svd_solver = 'auto'
            pca = PCA(n_components=n_components, svd_solver=svd_solver)
            self.scores = pca.fit_transform(x_scaled)
            print("Note: n_components set to 2 as optimal fit with default/user defined settings was 1")

        # Loadings
        self.loadings = pca.components_

        self._isfitted = True

        # Useful bits for calcs
        n_components_fitted = self.scores.shape[1]
        [data_shape0, data_shape1] = msData.intensityData.shape

        # Can we determine R2 and Q2??
        self.outputstats['R2X_Scree'] = pca.explained_variance_ratio_
        self.modelParameters['VarExpRatio'] = pca.explained_variance_ratio_ # Caro tidy this up

        # Calculate DmodX
        #x_reconstructed = scaler.inverse_transform(pca.inverse_transform(self.scores))
        x_reconstructed = pca.inverse_transform(self.scores) # reconstruct data from scores
        resid_ssx = np.sum((x_scaled - x_reconstructed) ** 2, axis=1) # calculate residual sum of squares
        s = np.sqrt(resid_ssx / (data_shape1 - n_components_fitted))
        s0 = np.sqrt(resid_ssx.sum() / ((data_shape0 - n_components_fitted - 1) * (data_shape1 - n_components_fitted)))
        self.outputstats['DmodX'] = np.sqrt((s / s0) ** 2)


        # Calculate Fcrit
        # Degrees of freedom for the PCA model (denominator in F-stat) calculated as suggested in
        # Faber, Nicolaas (Klaas) M., Degrees of freedom for the residuals of a
        # principal component analysis - A clarification, Chemometrics and Intelligent Laboratory Systems 2008
        self.outputstats['DmodX_Fcrit'] = stats.f.ppf(1 - f_crit_alpha, data_shape1 - n_components_fitted - 1,
                                                      (data_shape0 - n_components_fitted - 1) * (data_shape1 - n_components_fitted))

        # Save inputs
        self.inputparams['n_components'] = n_components
        self.inputparams['svd_solver'] = svd_solver
        self.inputparams['scaling'] = scaling
        self.inputparams['withExclusions'] = withExclusions
        self.inputparams['f_crit_alpha'] = f_crit_alpha
        self.inputparams = {**self.inputparams, **kwargs}

        # Save outputs
        self.outputstats['n_components_fitted'] = n_components_fitted

        return self


    def hotelling_T2(self, comps=None, alpha=0.05):
        """

        Obtain the parameters for the Hotelling T2 ellipse at the desired significance level.

        :param list comps:
        :param float alpha: Significance level
        :return: The Hotelling T2 ellipsoid radii at vertex
        :rtype: numpy.ndarray
        :raise AtributeError: If the model is not fitted
        :raise ValueError: If the components requested are higher than the number of components in the model
        :raise TypeError: If comps is not None or list/numpy 1d array and alpha a float
        """

        # Initial checks
        if self._isfitted is False:
            raise AttributeError("Model is not fitted")

        # Add other checks

        #except AttributeError as atre:
        #    raise atre
        #except ValueError as valerr:
        #    raise valerr
        #except TypeError as typerr:
        #    raise typerr

        nsamples = self._npyc_dataset_shape['NumberSamples']

        ncomps = len(comps)
        ellips = self.scores[:, comps] ** 2
        ellips = 1 / nsamples * (ellips.sum(0))

        # F stat
        fs = (nsamples - 1) / nsamples * ncomps * (nsamples ** 2 - 1) / (nsamples * (nsamples - ncomps))
        fs = fs * stats.f.ppf(1-alpha, ncomps, nsamples - ncomps)

        hotelling_t2 = list()
        for comp in range(ncomps):
            hotelling_t2.append(np.sqrt((fs * ellips[comp])))

        return np.array(hotelling_t2)


    def main(self):
        print("Implementation of scikit-learn PCA")

    if __name__ == '__main__':
        main()
