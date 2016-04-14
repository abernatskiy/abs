#!/usr/bin/python2

import numpy as np
from abs import Abs

class TestAbs(Abs):
	def setRandomPrototypesMembersip(self):
		self.mem_degrees = np.zeros([self.n_prototypes, self.n_labels])
		for i in range(self.n_prototypes):
			self.mem_degrees[i][np.random.randint(self.n_labels)] = 1.

	def layer1OnPrototypes(self):
		return np.array([ self._layer1(f) for f in self.prototypes ])

	def layer2OnPrototypes(self):
		return [ self._layer2(pa) for pa in self.layer1OnPrototypes() ]

	def layer3OnPrototypes(self):
		return [ self._layer3(e) for e in self.layer2OnPrototypes() ]

	def testLayer1(self):
		'''Checks if putting protypes through L1 yields alphas'''
		prototypeImagesCoincide = np.diagonal(self.layer1OnPrototypes()) == self.alphas
		return prototypeImagesCoincide.all()

	def testLayer2(self):
		'''Checks if items of evidence provided by the prototypes regarding themselves are pointing to the label of the prototype with alpha weight'''
		self.setRandomPrototypesMembersip()
		l2p = self.layer2OnPrototypes()
		for i in range(self.n_prototypes):
			labelsMasses, unknownMass = l2p[i][i]
			if not (labelsMasses == self.mem_degrees[i]*self.alphas[i]).all() or unknownMass != 1-self.alphas[i]:
				return False
		return True

	def testLayer3(self):
		'''For each classifier, checks if the classifier assigns all the mass to a the prototypes label if the label is certain (alpha=1, mem_degree=delta(i,label))'''
		self.alphas = np.ones(self.n_prototypes)
		self.setRandomPrototypesMembersip()
		l3p = self.layer3OnPrototypes()
		for i in range(self.n_prototypes):
			labelMasses,uncMass = l3p[i]
			if uncMass != 0. or np.argmax(labelMasses) != np.argmax(self.mem_degrees[i]):
				return False
		return True

if __name__ == '__main__':
	from main import loadData
	features, labels, labelEnc = loadData('data/iris/iris.data')
	testAbsCla = TestAbs(n_prototypes=10)
	testAbsCla.fit(features, labels)
	if testAbsCla.testLayer1():
		print('Layer1 test success!')
	else:
		print('Layer1 test failed')
	if testAbsCla.testLayer2():
		print('Layer2 test success!')
	else:
		print('Layer2 test failed')
	if testAbsCla.testLayer3():
		print('Layer3 test success!')
	else:
		print('Layer3 test failed')
