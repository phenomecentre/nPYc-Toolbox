import copy
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Can we just use sklearn-PCA?

def doPCA(dataset, withExclusions=False, scaling='uv', n_components=6, svd_solver='auto', **kwargs):

    """
    Perform PCA on the provided data matrix.

    Uses scikit-learn PCA: see documentation for options
    """

    # Initial checks

    # Apply sample/feature masks if exclusions to be applied
    msData = copy.deepcopy(dataset)
    if withExclusions:
        msData.applyMasks()

    # Scale features
    if scaling == 'uv': # Means-centre and uv scale
        scaler = StandardScaler(with_mean=True, with_std=True)
    else: # Means-centre only
        scaler = StandardScaler(with_mean=True, with_std=False)

    # Scale data
    X_scaled = scaler.fit_transform(msData.intensityData)

    # Fit PCA
    pca = PCA(n_components=n_components, svd_solver=svd_solver, **kwargs)
    scores = pca.fit_transform(X_scaled)

    # Loadings
    loadings = pca.components_.T
