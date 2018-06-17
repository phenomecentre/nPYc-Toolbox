import numpy
import inspect
import os
import sys
import sqlite3
import types
import copy
import pandas
import logging
from .._toolboxPath import toolboxPath
from ..objects._msDataset import MSDataset
from ..utilities._internal import _copyBackingFiles as copyBackingFiles
from ..utilities._internal import _vcorrcoef
from ..plotting import plotCorrelationToLRbyFeature, plotBatchAndROCorrection, histogram
from ..enumerations import AssayRole, SampleType
from ..__init__ import __version__ as version

def generateMSIDrequests(msData, features, outputDir='', rawData=None, database=None, returnFiles=3, msDataPrecorrection=None):
	"""
	Produce feature ID reports for the features listed in *features*, from the dataset *msData*.

	Feature ID reports visualise the abundance of the feature in the dataset, identify the analytical data files with the greatest abundance of the feature, look for correlations with other features withing the dataset, and if specified, search against the database provided.

	:param MSDataset msData: Report on features in this dataset (dataset must be post-correction)
	:param list features: List of features IDs that will be plotted from *msData.featureMetadata*
	:param str outputDir: Save reports into this directory
	:param rawData: Location of raw data files
	:type rawData: None or str
	:param database: Attempt to lookup features in specified database
	:type database: None or str
	:param MSDataset msDataPrecorrection: None or MSDataset pre-correction, if present sample intensities will be plotted pre to post correction 
	"""

	# Validate inputs
	if not isinstance(msData, MSDataset):
		raise TypeError('msData must be an instance of nPYc.MSDataset')
	if not isinstance(outputDir, str):
		raise ValueError('outputDir must be a path')
	if msDataPrecorrection is not None:
		if not isinstance(msDataPrecorrection, MSDataset):
			raise TypeError('msData must be an instance of nPYc.MSDataset')
		if msData.intensityData.shape != msDataPrecorrection.intensityData.shape:
			raise ValueError('msData and msDataPrecorrection datasets must have the same samples and features')

	if database is not None:
		conn = sqlite3.connect('file:' + database + '?mode=ro', uri=True)
	else:
		conn = None

	# Build target file list for R scripts
	rOutput = pandas.DataFrame(None, columns=['feature', 'mz', 'rt', 'sample'])
	aOutput = pandas.DataFrame(None, columns=['feature', 'mz', 'rt', 'associated feature', 'mz', 'rt', 'correlation', 'mass difference'])

	# Prepare the data objects - exclude all samples that are not SS, SP or ER	
	sampleMask = numpy.zeros(msData.sampleMask.shape).astype(bool)
	SSmask = (msData.sampleMetadata['SampleType'].values == SampleType.StudySample) & (msData.sampleMetadata['AssayRole'].values == AssayRole.Assay)
	SPmask = (msData.sampleMetadata['SampleType'].values == SampleType.StudyPool) & (msData.sampleMetadata['AssayRole'].values == AssayRole.PrecisionReference)
	ERmask = (msData.sampleMetadata['SampleType'].values == SampleType.ExternalReference) & (msData.sampleMetadata['AssayRole'].values == AssayRole.PrecisionReference)
	sampleMask[SSmask|SPmask|ERmask] = True
	
	postData = copy.deepcopy(msData)
	postData.sampleMask = sampleMask
	postData.applyMasks()
	
	if msDataPrecorrection is not None:
		preData = copy.deepcopy(msDataPrecorrection)
		preData.sampleMask = sampleMask
		preData.applyMasks()
	else:
		preData = None

	for feature in features:
		logging.info('Working on %s.' % feature)
		compoundList = _msIDreport(postData, feature, outputDir=outputDir, rawData=rawData, dbConnection=conn, msDataPrecorrection=preData)
		if compoundList == -1:
			# Check to see if we found anything
			logging.warning('Feature \'%s\' not found.' % feature)
			continue

		for i in range(returnFiles):
			line = pandas.DataFrame([[feature, compoundList['mz'], compoundList['rt'], compoundList['files'][i]['id']]], columns=['feature', 'mz', 'rt', 'sample'])
			rOutput = rOutput.append(line, ignore_index=True)

		noAssociatedFeatures = len(compoundList['associatedFeatures'])
		for i in range(noAssociatedFeatures):
			line = pandas.DataFrame([[feature, compoundList['mz'], compoundList['rt'], compoundList['associatedFeatures'][i]['id'], compoundList['associatedFeatures'][i]['mz'], compoundList['associatedFeatures'][i]['rt'], compoundList['associatedFeatures'][i]['correlation'], abs(compoundList['mz']-compoundList['associatedFeatures'][i]['mz'])]], columns=['feature', 'mz', 'rt', 'associated feature', 'mz', 'rt', 'correlation', 'mass difference'])
			aOutput = aOutput.append(line, ignore_index=True)
			
	if database is not None:
		conn.close()

	# Get toolboxpath and copy template files
	copyBackingFiles(toolboxPath(), os.path.join(outputDir, 'graphics'))

	# Output for R: append to file if already exists
	rOutputPath = os.path.join(outputDir, 'R_file_list.csv')
	if os.path.exists(rOutputPath):
		rOutput.to_csv(rOutputPath, mode='a', header=False)
	else:
		rOutput.to_csv(rOutputPath)
		
	# Output associated features
	aOutputPath = os.path.join(outputDir, 'Associated_features_list.csv')
	if os.path.exists(aOutputPath):
		aOutput.to_csv(aOutputPath, mode='a', header=False)
	else:
		aOutput.to_csv(aOutputPath)	


def _msIDreport(msData, feature, outputDir='', rawData=None, dbConnection=None, msDataPrecorrection=None):

	from jinja2 import Template, Environment, FileSystemLoader

	import seaborn as sns
	import matplotlib as mpl
	import matplotlib.pyplot as plt
	import shutil

	env = Environment(loader=FileSystemLoader(os.path.join(toolboxPath(), 'Templates')))

	template = env.get_template('ID_request_MS.html')

	if not os.path.exists(os.path.join(outputDir, 'graphics')):
		os.makedirs(os.path.join(outputDir, 'graphics'))

	# Check if the desired feature is present in the dataset.
	if not feature in msData.featureMetadata['Feature Name'].values:
		import warnings

		errorMessage = 'Feature \'' + feature + '\' not found in dataset, skipping.'
		warnings.warn(errorMessage)

		return -1
		
	# Dig the feature metadata out of the object
	item = dict()
	item['FeatureID'] = feature

	item['Matrix'] = [x for x in msData.sampleMetadata['Matrix'].unique() if x]
	item['Chromatography'] = [x for x in msData.sampleMetadata['Chromatography'].unique() if x]

	localMetadata =  msData.featureMetadata[msData.featureMetadata['Feature Name'] == feature]
	featureNo = localMetadata.index
	featureNo = featureNo[0]
	item['PrimaryIon'] = localMetadata['m/z'].values[0]
	item['RetentionTime'] = localMetadata['Retention Time'].values[0]

	# Optional items
	if 'Isotope Distribution' in localMetadata.columns:
		item['IsotopeDistribution'] = localMetadata['Isotope Distribution'].values[0]

	if 'Adducts' in localMetadata.columns:
		item['Adducts'] = localMetadata['Adducts'].values
		
	# Find feature index
	featureIX = msData.featureMetadata[msData.featureMetadata['Feature Name']==feature].index
	featureIX = int(featureIX[0])

	##
	# Create a local copy of the DF to add in in fields for Seaborn plots
	##
	localDF = msData.sampleMetadata.copy()
	SSmask = (localDF['SampleType'].values == SampleType.StudySample) & (localDF['AssayRole'].values == AssayRole.Assay)
	SPmask = (localDF['SampleType'].values == SampleType.StudyPool) & (localDF['AssayRole'].values == AssayRole.PrecisionReference)
	ERmask = (localDF['SampleType'].values == SampleType.ExternalReference) & (localDF['AssayRole'].values == AssayRole.PrecisionReference)
	localDF.loc[SSmask, 'Sample Type'] = 'Study Sample'
	localDF.loc[SPmask, 'Sample Type'] = 'Study Pool'
	localDF.loc[ERmask, 'Sample Type'] = 'External Reference'
	localDF.loc[SPmask | ERmask, 'Reference Samples'] = 'Reference Sample'

	localDF[feature] = pandas.Series(numpy.squeeze(msData.intensityData[:, localMetadata.index]), index=localDF.index)

	##
	# Find and plot samples with max abundance vs project and reference varaince
	##
	maxIndex = numpy.squeeze(msData.intensityData[:, localMetadata.index]).argsort()[::-1][:msData.Attributes['noFiles']]
	item['AbundanceSamples'] = []
	ix=1
	for index in maxIndex:
		a_sample = dict(rank=ix, id=msData.sampleMetadata['Sample File Name'].iloc[index], value=msData.intensityData.item(index, featureNo))
		item['AbundanceSamples'].append(a_sample)
		ix=ix+1

	item['AbundanceFigure'] = os.path.join(outputDir, 'graphics', 'feature_' + feature.replace('/', '-') + '_abundance.' + msData.Attributes['figureFormat'])

	# Visualise abundance wrt to ST and LTR.
	sns.set_style("whitegrid")
	
	# Define colours - match reporting standard
	palette=sns.color_palette("deep")
	
	# Check if we have LTR and SR, or only SR.
	refCount = localDF.loc[localDF['Reference Samples'] == 'Reference Sample', 'Sample Type'].nunique()
	if refCount == 1:
		split = False
		hue_order = None
		if sum(SPmask) != 0:
			palette = [palette[1]]
		else:
			palette = [palette[2]]
	elif refCount == 2:
		split = True	
		hue_order = ['Study Pool', 'External Reference']
		palette = [palette[1], palette[2]]

	# Left hand plot
	fig, (ax1,ax2) = plt.subplots(2,1, sharex=True, figsize=(12,6))
		
	# Violin plot of study reference samples 
	sns.violinplot(x=item['FeatureID'], data=localDF[localDF['Reference Samples'] == 'Reference Sample'], 
			y='Reference Samples', 
			hue='Sample Type',
			hue_order=hue_order,
			split=split,
			palette=palette,
			scale='width',
			ax=ax1, bw=.2, cut=0)#top plot
			
	# Violin plot of study samples
	sns.violinplot(x=item['FeatureID'], data=localDF[localDF['Sample Type'] == 'Study Sample'],
			y='Sample Type',
			split=True,
			scale='width',
			ax=ax2, bw=.1, cut=0)
	
	# Add red dashes where highest abundance samples lie
	for sample in item['AbundanceSamples']:
		
		# Find sample index
		sampleIX = msData.sampleMetadata[msData.sampleMetadata['Sample File Name']==sample['id']].index
		sampleIX = int(sampleIX[0])
		
		if localDF.iloc[sampleIX]['Sample Type'] == 'Study Sample':
			ax2.plot([msData.intensityData[sampleIX, featureIX], msData.intensityData[sampleIX, featureIX]], [-0.05, 0.05], color='r', linestyle='-', linewidth=0.5)
		else:
			ax1.plot([msData.intensityData[sampleIX, featureIX], msData.intensityData[sampleIX, featureIX]], [-0.05, 0.05], color='r', linestyle='-', linewidth=0.5)
	
	# Clean up
	fig.suptitle('')
	ax1.set_xlabel('')
	ax1.set_ylabel('')
	ax2.set_ylabel('')
	ax2.set_title('')
	fig.set_facecolor('w')
	plt.tight_layout()
	plt.savefig(item['AbundanceFigure'], format=msData.Attributes['figureFormat'], dpi=msData.Attributes['dpi'])
	plt.close()


	##
	# Use biological correlation in the dataset to find other related features
	##

	# When working with samples, only use study samples
	sampleMask = numpy.logical_and(
								msData.sampleMask,
								SSmask
								  )

	item['RelatedFigure'] = os.path.join(outputDir, 'graphics', 'feature_' + feature.replace('/', '-') + '_related.' + msData.Attributes['figureFormat'])

	intCorrs = _vcorrcoef((msData.intensityData[sampleMask,:]), numpy.transpose(msData.intensityData[sampleMask, featureNo]))

	# Null out excluded features
	intCorrs[msData.featureMask == False] = 0
	intCorrs[numpy.isnan(intCorrs)] = 0

	color = sns.color_palette()[1]
	grid = sns.JointGrid(msData.featureMetadata[intCorrs > msData.Attributes['corrThreshold']]['m/z'].values, msData.featureMetadata[intCorrs > msData.Attributes['corrThreshold']]['Retention Time'].values,
					space=1, size=6, ratio=50,
					xlim=(msData.featureMetadata['m/z'].min(), msData.featureMetadata['m/z'].max()),
					ylim=(msData.featureMetadata['Retention Time'].min(), msData.featureMetadata['Retention Time'].max()))
	grid.plot_joint(plt.scatter, color=color, alpha=.8)
	grid.plot_marginals(sns.rugplot, color=color)

	grid.set_axis_labels('m/z', 'Retention Time')

	# sns.despine(trim=True)
	plt.tight_layout()
	plt.savefig(item['RelatedFigure'], format=msData.Attributes['figureFormat'], dpi=msData.Attributes['dpi'])
	plt.close()

	##
	# Find and plot correlating, co-elluting features.
	##
	# if no raw data
	if rawData == None:
		item['Coelutants'] = 'Features observed to co-elute'

		# find feature within rt bounds and peak width bounds
		try:
			associatedFeatures =  msData.featureMetadata[
					(msData.featureMetadata['Retention Time'] > (numpy.squeeze(localMetadata['Retention Time'].values) - (msData.Attributes['rtWindow'] / 60.0)))
					& (msData.featureMetadata['Retention Time'].values < (numpy.squeeze(localMetadata['Retention Time'].values) +( msData.Attributes['rtWindow'] / 60.0)))
					& (msData.featureMetadata['Peak Width'].values > (numpy.squeeze(localMetadata['Peak Width'].values) - msData.Attributes['peakWidthWindow']))
					& (msData.featureMetadata['Peak Width'].values < (numpy.squeeze(localMetadata['Peak Width'].values) + msData.Attributes['peakWidthWindow']))
					& (msData.featureMetadata['Feature Name'].values != feature)
					& msData.featureMask]
		except: # If no peak width information available
			associatedFeatures =  msData.featureMetadata[
					(msData.featureMetadata['Retention Time'] > (numpy.squeeze(localMetadata['Retention Time'].values) - (msData.Attributes['rtWindow'] / 60.0)))
					& (msData.featureMetadata['Retention Time'].values < (numpy.squeeze(localMetadata['Retention Time'].values) +( msData.Attributes['rtWindow'] / 60.0)))
					& (msData.featureMetadata['Feature Name'].values != feature)
					& msData.featureMask]			

		meanIntesities = numpy.mean(msData.intensityData, axis=0)

		# Then plot a psuedo spectrum
		fig = plt.figure()
		ax = fig.add_subplot(111)

		pallet = sns.color_palette("hls", 32)
		# create function for doing interpolation of the desired
		# ranges
		scaler = make_interpolater(-1, 1, 0, 31)
		
		item['AssociatedFeatures'] = []
		for associatedFeature in associatedFeatures.iterrows():

			# Horrible hack with NaNs to convince .stem to plot one stick.
			markerline, stemlines, baseline = ax.stem((numpy.nan, associatedFeature[1]['m/z']), (numpy.nan, meanIntesities[associatedFeature[0]]), basefmt='')

			plt.setp(stemlines, 'color', sns.color_palette()[1], 'linewidth', 2)
			plt.setp(markerline, 'color', pallet[scaler(intCorrs[associatedFeature[0]])])

			markerline.set_zorder(20)

			if intCorrs[associatedFeature[0]] > 0.7:
				ax.annotate(numpy.round(associatedFeature[1]['m/z'], decimals=4), xy=(associatedFeature[1]['m/z'], meanIntesities[associatedFeature[0]]),
							xycoords='data',
							xytext=(20, 100),
							textcoords='offset points',
							rotation=50,
							size=12,
							arrowprops=dict(arrowstyle="-"))
							
			
				a_sample = dict(id=associatedFeature[1]['Feature Name'], rt=associatedFeature[1]['Retention Time'], mz=associatedFeature[1]['m/z'] , correlation=intCorrs[associatedFeature[0]])
				item['AssociatedFeatures'].append(a_sample)


		markerline, stemlines, baseline = ax.stem(localMetadata['m/z'].values, msData.intensityData[maxIndex[0], localMetadata.index], basefmt='')

		plt.setp(stemlines, 'color', sns.color_palette()[0], 'linewidth', 2)
		plt.setp(markerline, 'color', sns.color_palette()[0])

		ax.set_xlabel('m/z')
		ax.set_ylabel('Abundance')

		item['CoelutantsFigure'] = os.path.join(outputDir, 'graphics', 'feature_' + feature.replace('/', '-') + '_coelutants.' + msData.Attributes['figureFormat'])
		# sns.despine()
		plt.tight_layout()
		fig.savefig(item['CoelutantsFigure'], bbox_inches="tight", format=msData.Attributes['figureFormat'], dpi=msData.Attributes['dpi'])
		plt.close()

		# Generate list of correlated features
		maxIndex = intCorrs.argsort()[::-1][:msData.Attributes['noFiles']+1]
		item['CorrelatedFeature'] = []
		for index in maxIndex:
			if msData.featureMetadata['Feature Name'].iloc[index] == feature:
				continue

			a_sample = dict(id=msData.featureMetadata['Feature Name'].iloc[index], 
						corr=intCorrs[index],
						mz=msData.featureMetadata['m/z'].iloc[index],
						rt=msData.featureMetadata['Retention Time'].iloc[index])
			item['CorrelatedFeature'].append(a_sample)

		# Potential suppression effects
		minIndex = intCorrs.argsort()[::-1][msData.Attributes['noFiles']+1:]
		item['AntiCorrelatedFeature'] = []
		for index in minIndex:
			if intCorrs[index] < -0.8:
				item['AntiCorrelatedFlag'] = 1

				a_sample = dict(id=msData.featureMetadata['Feature Name'].iloc[index], 
							corr=intCorrs[index],
							mz=msData.featureMetadata['m/z'].iloc[index],
							rt=msData.featureMetadata['Retention Time'].iloc[index])
				item['AntiCorrelatedFeature'].append(a_sample)
		

	else:
		pass
		# If raw data
		# Load datafile with max abundance

		# correlate peak shapes within bounds about the target feature.

		# Make a plot of Co-elutants
		# and peak shape?
	
	
	##
	# Plot intensities of pre and post corrected data and fit for feature, highlight samples with highest intensity
	## 	
	
	item['IntensityFigure'] = os.path.join(outputDir, 'graphics', 'feature_' + feature.replace('/', '-') + '_intensity.' + msData.Attributes['figureFormat'])
	
	# Plot (pre and post correction if available, else just msData)
	if (msDataPrecorrection is None):
		msDataPrecorrection = msData
		
	# Generate plot
	plotBatchAndROCorrection(msDataPrecorrection, 
		msData,
		featureIX,
		logy=True,
		title='Feature: ' + str(numpy.squeeze(msDataPrecorrection.featureMetadata.loc[featureIX, 'Feature Name'])),
		sampleAnnotation = item['AbundanceSamples'],	
		addViolin=False,
		savePath=item['IntensityFigure'],																																					
		figureFormat=msData.Attributes['figureFormat'],
		dpi=msData.Attributes['dpi'],
		figureSize=(12,6))
	
		
	##
	# DB search here
	##
	if dbConnection is not None:
		c = dbConnection.cursor()

		mzMin = item['PrimaryIon'] - msData.Attributes['msPrecision']
		mzMax = item['PrimaryIon'] + msData.Attributes['msPrecision']

		# Convert RT to seconds
		rtMin = (item['RetentionTime'] * 60.0) - msData.Attributes['rtWindow']
		rtMax = (item['RetentionTime'] * 60.0) + msData.Attributes['rtWindow']

		if msData.sampleMetadata.loc[1, 'Ionisation'] == 'POS':
			ionisation = 'ES+'
		elif msData.sampleMetadata.loc[1, 'Ionisation'] == 'NEG':
			ionisation = 'ES-'
		else:
			import warnings
			warnings.warn('Unknown ionisation')
			ionisation = ''

		acquisitionSOP = item['Chromatography'][0]

		query = 'SELECT compound.commonName, msPeakList.mz, msPeakList.rtSeconds\
				FROM msPeakList\
				INNER JOIN msDataset ON msPeakList.msDatasetID = msDataset.msDatasetID\
				INNER JOIN aliquot ON msDataset.aliquotID = aliquot.aliquotID\
				INNER JOIN compound ON aliquot.compoundID = compound.compoundID\
				WHERE (msDataset.ionisation = ?)\
				& (msDataset.acquisitionSOP = ?)\
				& (msPeakList.mz > ?)\
				& (msPeakList.mz < ?)\
				& (msPeakList.rtSeconds > ?)\
				& (msPeakList.rtSeconds < ?)'

		for row in c.execute(query, (ionisation, acquisitionSOP, mzMin, mzMax, rtMin, rtMax)):
			if not 'dbMatches' in item.keys():
				item['dbMatches'] = list()
			
			item['dbMatches'].append({'name':row[0], 'mz':row[1], 'rt':row[2]/60.0})

	##
	# Finally generate report here.
	##
	filename = os.path.join(outputDir, 'ID Request_' + feature.replace('/', '-') + '.html')


	# Make paths for graphics local not absolute for use in the HTML.
	item['AbundanceFigure'] = 'graphics/feature_' + feature.replace('/', '-') + '_abundance.' + msData.Attributes['figureFormat']
	item['IntensityFigure'] = 'graphics/feature_' + feature.replace('/', '-') + '_intensity.' + msData.Attributes['figureFormat']
	item['CoelutantsFigure'] = 'graphics/feature_' + feature.replace('/', '-') + '_coelutants.' + msData.Attributes['figureFormat']
	item['RelatedFigure'] = 'graphics/feature_' + feature.replace('/', '-') + '_related.' + msData.Attributes['figureFormat']

	# Format and save report.
	f = open(filename,'w')

	f.write(template.render(item=item, version=version))
	f.close()

	return {'mz':item['PrimaryIon'], 'rt':item['RetentionTime'], 'files':item['AbundanceSamples'], 'associatedFeatures':item['AssociatedFeatures']}


"""
From http://stackoverflow.com/questions/1969240/mapping-a-range-of-values-to-another, map corr floats to ints
"""
def make_interpolater(left_min, left_max, right_min, right_max):
	# Figure out how 'wide' each range is
	leftSpan = left_max - left_min
	rightSpan = right_max - right_min

	# Compute the scale factor between left and right values 
	scaleFactor = float(rightSpan) / float(leftSpan) 

	# create interpolation function using pre-calculated scaleFactor
	def interp_fn(value):
		scaledVal = (right_min + (value-left_min)*scaleFactor)
		if scaledVal < right_min:
			scaledVal = right_min
		elif scaledVal > right_max:
			scaledVal = right_max
		else:
			scaledVal = numpy.rint(scaledVal).astype(int)

		return scaledVal

	return interp_fn

