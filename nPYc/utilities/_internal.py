import os

def _copyBackingFiles(toolboxPath, output):
	"""
	Copy templates files to the 'graphics' sub-directory of the output directory when needed.
	"""
	import shutil

	# And finaly copy the css over, we're not checking for existance,as we always want the latest moved over.
	shutil.copy(os.path.join(toolboxPath, 'Templates', 'npc-main.css'), os.path.join(output, 'npc-main.css'))
	shutil.copy(os.path.join(toolboxPath, 'Templates', 'toolbox_logo.png'), os.path.join(output, 'toolbox_logo.png'))


def _vcorrcoef(X, Y, method='pearson', sampleMask=None, featureMask=None):
	"""
	Calculate correlation between each column in *X* and the vector *Y*. Correlations may be calculated either as Pearson's *r* [#]_ or Spearman's rho [#]_ .
	
	[#] Karl Pearson (20 June 1895) "Notes on regression and inheritance in the case of two parents," *Proceedings of the Royal Society of London*, 58 : 240–242.
	[#] Myers, Jerome L.; Well, Arnold D. (2003). *Research Design and Statistical Analysis (2nd ed.)*. Lawrence Erlbaum. p. 508. ISBN 0-8058-4037-0.
	
	:param numpy.ndarray X: 
	:param numpy.ndarray Y:
	:param str method: Correlation method to use, may be 'pearson', or 'spearman'
	:param sampleMask: If ``None`` calculate correlations based on all samples, otherwise use *sampleMask* as a boolean mask
	:type sampleMask: None or numpy.ndarray of bool
	:param featureMask: If ``None`` calculate correlations for all features
	:type featureMask: None or numpy.ndarray of bool
	"""
	import numpy
	import scipy

	if sampleMask is None:
		pass
	else:
		Y = Y[sampleMask]
		X = X[sampleMask,:]

	if featureMask is None:
		pass
	else:
		X = X[:,featureMask]

	if method == 'spearman':
		rankedMat = numpy.zeros_like(X)
		for col in range(X.shape[1]):
			rankedMat[:, col] = scipy.stats.rankdata(X[:,col])
		X = rankedMat
		Y = scipy.stats.rankdata(Y)

	X = X.T

	# From https://waterprogramming.wordpress.com/2014/06/13/numpy-vectorized-correlation-coefficient/
	Xm = numpy.reshape(numpy.mean(X,axis=1),(X.shape[0],1))
	ym = numpy.mean(Y)
	r_num = numpy.sum((X-Xm)*(Y-ym),axis=1)
	r_den = numpy.sqrt(numpy.sum(numpy.power((X-Xm), 2),axis=1)*numpy.sum(numpy.power((Y-ym), 2)))
	r = r_num/r_den

	# Set NaNs to zero correlation
	r[numpy.isnan(r)] = 0

	return r
