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

if __name__ == '__main__':
	irisDataFile = 'data/iris/iris.data'
	features, labels, labelEnc = loadData(irisDataFile)
	testVectors = [	[ 6, 2.3, -0.5, 8 ],
									[ 6.4,  3.3,  5.55,  2.2]	] # Should be Iris-virginica

	abscla = Abs()
	abscla.fit(features, labels)
	testLabelsAbs = abscla.predict(testVectors)
#	print('Got labels ' + str(labelEnc.inverse_transform(testLabelsAbs)) + ' from ABS classifier (numericals ' + str(testLabelsAbs) + ')')
	print('ABS: Got labels ' + str(testLabelsAbs))

	knncla = KNeighborsClassifier(n_neighbors=3)
	knncla.fit(features, labels)
	testLabelsKnn = knncla.predict(testVectors)
#	print('Got labels ' + str(labelEnc.inverse_transform(testLabelsKnn)) + ' from k-nn classifier (numericals ' + str(testLabelsKnn) + ')')
	print('KNN: Got labels ' + str(testLabelsKnn))
