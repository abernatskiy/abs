#!/usr/bin/python2

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from abs import Abs

def loadData(datafile):
	features = np.loadtxt(datafile, delimiter=',', usecols=(0,1,2,3))
	strLabels = np.loadtxt(datafile, delimiter=',', usecols=(4,), dtype=np.str)
	labelEnc = LabelEncoder()
	labelEnc.fit(strLabels)
	labels = labelEnc.transform(strLabels)
	print('Read a dataset from ' + datafile + ': ' + str(features.shape[0]) + ' data points, labels: ' + str(labelEnc.classes_))
	return (features, labels, labelEnc)

def getDecisionSpaceCrossSectionMap(classifier, fixDims, fixDimVals, freeDimsLimits, resolution, rejectionCost=None, newLabelCost=None):
	totalDims = len(fixDims) + len(freeDimsLimits)

	freeDims = range(totalDims)
	for dim in fixDims:
		freeDims.remove(dim)

	gridPts = [ int(1+(end-start)/resolution) for start,end in freeDimsLimits ]
	numPoints = np.product(gridPts)
	def linearIdxToGrid(linIdx, dim):
		curPoints = numPoints
		curIdx = linIdx
		for d in freeDims[:-1]:
			curPoints = curPoints/gridPts[freeDims.index(d)]
			coord = curIdx / curPoints
			curIdx = curIdx % curPoints
			if d == dim:
				return coord
		return curIdx
	grid = []
	for i in range(numPoints):
		curPos = [ fixDimVals[fixDims.index(dim)] if dim in fixDims else freeDimsLimits[freeDims.index(dim)][0] + resolution*linearIdxToGrid(i, dim) for dim in range(totalDims) ]
		grid.append(curPos)

	gridVals = classifier.predict(grid, rejectionCost=rejectionCost, newLabelCost=newLabelCost)
	return gridVals.reshape(gridPts)

def discrete_cmap(N, base_cmap=None):
	"""Create an N-bin discrete colormap from the specified input map"""

	# Note that if base_cmap is a string or None, you can simply do
	#    return plt.cm.get_cmap(base_cmap, N)
	# The following works for string, None, or a colormap instance:

	base = plt.cm.get_cmap(base_cmap)
	color_list = base(np.linspace(0, 1, N))
	cmap_name = base.name + str(N)
	return base.from_list(cmap_name, color_list, N)

def plotDecisionSpaceCrossSectionMap(classifier, fixDims, fixDimVals, freeDimsLimits, resolution, rejectionCost=None, newLabelCost=None):
	crossection = getDecisionSpaceCrossSectionMap(classifier, fixDims, fixDimVals, freeDimsLimits, resolution, rejectionCost=rejectionCost, newLabelCost=newLabelCost)
	totalDims = len(fixDims) + len(freeDimsLimits)
	freeDims = range(totalDims)
	for dim in fixDims:
		freeDims.remove(dim)
	if len(freeDims) != 2:
		raise ValueError('Only two-dimensional decision maps are supported, ' + str(len(freeDim)) + '-dimensional map implied by fixed dims')
	if len(freeDimsLimits) != 2:
		raise ValueError('Only two-dimensional decision maps are supported, ' + str(len(freeDim)) + '-dimensional map implied by limits')

	ranges = [ [ start + resolution*float(i) for i in range(int((end-start)/resolution) + 1) ] for start,end in freeDimsLimits ]

	x,y = np.meshgrid(ranges[0], ranges[1])
#	cs = plt.contour(x.T,y.T,crossection, colors='k', nchunk=0)
#	csf = plt.contourf(x.T,y.T,crossection, len(np.unique(crossection)), cmap=plt.cm.Paired)
	plt.pcolor(x.T, y.T, crossection, cmap=discrete_cmap(len(np.unique(crossection)), plt.cm.jet))
#	cb = plt.colorbar(ticks=np.unique(crossection), label='')
	plt.xlabel('Dimension ' + str(freeDims[0]))
	plt.ylabel('Dimension ' + str(freeDims[1]))

if __name__ == '__main__':
	irisDataFile = 'data/iris/iris.data'
	features, labels, labelEnc = loadData(irisDataFile)
	testVectors = [	[ 6, 2.3, -0.5, 8 ],
									[ 6.4,  3.3,  5.55,  2.2]	] # Should be Iris-virginica

	abscla = Abs()
	abscla.fit(features, labels, max_iterations=100000)
	testLabelsAbs = abscla.predict(testVectors)
#	print('Got labels ' + str(labelEnc.inverse_transform(testLabelsAbs)) + ' from ABS classifier (numericals ' + str(testLabelsAbs) + ')')
	print('ABS: Got labels ' + str(testLabelsAbs))

	knncla = KNeighborsClassifier(n_neighbors=3)
	knncla.fit(features, labels)
	testLabelsKnn = knncla.predict(testVectors)
#	print('Got labels ' + str(labelEnc.inverse_transform(testLabelsKnn)) + ' from k-nn classifier (numericals ' + str(testLabelsKnn) + ')')
	print('KNN: Got labels ' + str(testLabelsKnn))

	plotDecisionSpaceCrossSectionMap(abscla, [0,1], [5.5, 3.0], [(0,6), (0,3)], 0.1, rejectionCost=0.5, newLabelCost=0.65)
	cb = plt.colorbar(ticks=[-2,-1,0,1,2])
	cb.set_ticklabels(['Novel', 'Reject', 'Iris Setosa', 'Iris Veriscolor', 'Iris Virginica'])
	plt.show()
