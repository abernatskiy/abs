#!/usr/bin/python2

import numpy as np
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

	gridVals = classifier.predict(grid)
	return gridVals.reshape(gridPts)

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

