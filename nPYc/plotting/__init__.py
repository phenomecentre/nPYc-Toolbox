"""
The :py:mod:`~nPYc.plotting` module contains function to generate several common visualisations.

Plots are built upon `seaborn <http://seaborn.pydata.org>`_ for aesthetics, or when interactivity is required, `plotly <https://plot.ly>`_.

Most plots support a set of common configuration parameters to allow customisation of various display options. Common parameters that may be specified as keyword arguments are:

.. py:function:: plottingFunctions(*vars, **kwargs):

	:param str savePath: If ``None`` plot interactively, otherwise save the figure to the path specified
	:param str figureFormat: If saving the plot, use this format
	:param int dpi: Plot resolution
	:param figureSize: Dimensions of the figure
	:type figureSize: tuple(float, float)

Interactive plots utilise the plotly framework to provide controls, when using plotly you should ensure that the environment is configured according to the instructions at `Offline Plots in Plotly in Python <https://plot.ly/python/offline/>`_ 
"""

from ._plotting import histogram, plotTICinteractive, plotLRTIC, plotCorrelationToLRbyFeature
from ._nmrPlotting import plotPW, plotLineWidthInteractive, plotLineWidth
from ._plotNMRcalibration import plotCalibration, plotCalibrationInteractive
from ._plotNMRbaseline import plotBaseline, plotBaselineInteractive
from ._plotNMRwater import plotWaterResonance, plotWaterResonanceInteractive
from ._plotNMRspectra import plotSpectraInteractive
from ._jointplotRSDvCorrelation import jointplotRSDvCorrelation
from ._plotRSDs import plotRSDs, plotRSDsInteractive
from ._plotTIC import plotTIC
from ._plotIonMap import plotIonMap
from ._plotBatchAndROCorrection import plotBatchAndROCorrection
from ._multivariatePlotting import plotScree, plotScores, plotOutliers, plotLoadings, plotScoresInteractive, plotLoadingsInteractive, plotMetadataDistribution
from ._plotSpectralVariance import plotSpectralVariance, plotSpectralVarianceInteractive
from ._plotDiscreteLoadings import plotDiscreteLoadings
from ._plotFeatureRanges import plotFeatureRanges
from ._plotLOQRunOrder import plotLOQRunOrder
from ._plotFeatureLOQ import plotFeatureLOQ
from ._plotVariableScatter import plotVariableScatter
from ._plotFeatureAccuracyPrecision import plotAccuracyPrecision
from ._plotIonMap import plotIonMapInteractive
from ._plotBlandAltman import blandAltman

__all__ = ['histogram', 'plotBatchAndROCorrection', 'plotTIC', 'plotTICinteractive', 'plotLRTIC', 'jointplotRSDvCorrelation', 'plotCorrelationToLRbyFeature',
		   'plotIonMap', 'plotRSDs', 'plotRSDsInteractive', 'plotScree', 'plotOutliers', 'plotSpectralVariance', 'plotScores', 'plotScoresInteractive',
		   'plotLoadings', 'plotLoadingsInteractive', 'plotDiscreteLoadings', 'plotFeatureRanges', 'plotMetadataDistribution', 'plotLOQRunOrder', 
		   'plotFeatureLOQ', 'plotVariableScatter', 'plotAccuracyPrecision', 'plotCalibrationInteractive', 'plotLineWidth', 'plotLineWidthInteractive',
		   'plotBaseline', 'plotBaselineInteractive', 'plotWaterResonance', 'plotWaterResonanceInteractive', 'plotSpectraInteractive', 'plotIonMapInteractive',
		   'plotSpectralVarianceInteractive', 'blandAltman']
