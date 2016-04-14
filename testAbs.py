#!/usr/bin/python2

import numpy as np
from abs import Abs

class TestAbs(Abs):
	def testLayer1(self):
		'''Checks if putting protypes through L1 yields alphas'''
		prototypeImagesCoincide = np.diagonal([ self._layer1(f) for f in self.prototypes ]) == self.alphas
		return prototypeImagesCoincide.all()
	def testLayer2(self):
		return False
	def testLayer3(self):
		return False

if __name__ == '__main__':
	from main import loadData
	features, labels, labelEnc = loadData('data/iris/iris.data')
	testAbsCla = TestAbs(n_prototypes=100)
	testAbsCla.fit(features, labels)
	if testAbsCla.testLayer1().all():
		print('Layer1 test success!')
	else:
		print('Layer1 test failed')
