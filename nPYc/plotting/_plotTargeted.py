import numpy
from ..enumerations import AssayRole, SampleType
import plotly.graph_objs as go


def plotFeatureInteractive(tData, featureNo=0, withExclusions=True):
    """
    Interactively visualise TIC (coloured by batch and sample type) with plotly, provides tooltips to allow identification of samples.

    Plots may be of two types:
    * **'Sample Type'** to plot by sample type and coloured by batch
    * **'Linearity Reference'** to plot LR samples coloured by dilution

    :param tDataset tData: Dataset object
    :param str plottype: Select plot type, may be either ``Sample Type`` or ``Linearity Reference``
    :return: Data object to use with plotly
    """
    import plotly.graph_objs as go
    from ..utilities import generateLRmask

    # Obtain the values and separate below LOQ and below LLOQ
    tempFeatureMask = numpy.sum(numpy.isfinite(tData.intensityData), axis=0)
    tempFeatureMask = tempFeatureMask < tData.intensityData.shape[0]
    tic = numpy.sum(tData.intensityData[:, tempFeatureMask == False], axis=1)

    if withExclusions:
        tempSampleMask = tData.sampleMask
    else:
        tempSampleMask = numpy.ones(shape=tData.sampleMask.shape, dtype=bool)

    SSmask = ((tData.sampleMetadata['SampleType'].values == SampleType.StudySample) & (
                tData.sampleMetadata['AssayRole'].values == AssayRole.Assay)) & tempSampleMask
    SPmask = ((tData.sampleMetadata['SampleType'].values == SampleType.StudyPool) & (
                tData.sampleMetadata['AssayRole'].values == AssayRole.PrecisionReference)) & tempSampleMask
    ERmask = ((tData.sampleMetadata['SampleType'].values == SampleType.ExternalReference) & (
                tData.sampleMetadata['AssayRole'].values == AssayRole.PrecisionReference)) & tempSampleMask

    SSplot = go.Scattergl(
        x=tData.sampleMetadata['Acquired Time'][SSmask],
        y=tic[SSmask],
        mode='markers',
        marker=dict(
            colorscale='Portland',
            color=tData.sampleMetadata['Correction Batch'][SSmask],
            symbol='circle'
        ),
        name='Study Sample',
        text=tData.sampleMetadata['Run Order'][SSmask]
    )

    SRplot = go.Scattergl(
        x=tData.sampleMetadata['Acquired Time'][SPmask],
        y=tic[SPmask],
        mode='markers',
        marker=dict(
            color='rgb(63, 158, 108)',
            symbol='cross'
        ),
        name='Study Reference',
        text=tData.sampleMetadata['Run Order'][SPmask]
    )

    LTRplot = go.Scatter(
        x=tData.sampleMetadata['Acquired Time'][ERmask],
        y=tic[ERmask],
        mode='markers',
        marker=dict(
            color='rgb(198, 83, 83)',
            symbol='cross'
        ),
        name='Long-Term Reference',
        text=tData.sampleMetadata['Run Order'][ERmask]
    )

    data = [SSplot, SRplot, LTRplot]
    Xlabel = 'Acquisition Time'
    title = 'TIC by Sample Type Coloured by Batch'


    # Add annotation
    layout = {
        'xaxis': dict(
            title=Xlabel,
        ),
        'yaxis': dict(
            title='TIC'
        ),
        'title': title,
        'hovermode': 'closest',
    }

    fig = {
        'data': data,
        'layout': layout,
    }

    return fig
