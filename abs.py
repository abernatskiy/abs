import numpy as np
from copy import deepcopy

def _distance(x1, x2):
	return np.linalg.norm(x2-x1)

class Abs(object):
	def __init__(self, n_prototypes=10):
		print('Abs classifier initialized')
		self.n_prototypes = n_prototypes

	def fit(self, features, labels):
		print('Abs fit called')

		self.features = np.array(features)
		self.dim = self.features.shape[1]

		self.labels = np.array(labels)
		self.n_labels = len(np.unique(self.labels))

		self.prototypes = np.random.random([self.n_prototypes, self.dim]) # prototypes - the surrogate nearest neighbor vectors
		self.alphas = np.random.random(self.n_prototypes) # how certain we are in the membership degrees of the prototype
		self.gammas = np.random.random(self.n_prototypes) # how far away should the influence of the prototype extend
		self.mem_degrees = np.zeros([self.n_prototypes, self.n_labels]) # membership of the prototype in every label
		for i in range(self.n_prototypes):
			self.mem_degrees[i][np.random.randint(self.n_labels)] = 1.
		return self

	def _layer1(self, feat):
		return [ alpha*np.exp(-1.*gamma*_distance(feat, prot)) for alpha,gamma,prot in zip(self.alphas, self.gammas, self.prototypes) ]

	def _layer2(self, protoActivations):
		return [ (protoActivation*mem_degree, 1-protoActivation) for mem_degree,protoActivation in zip(self.mem_degrees, protoActivations) ]

	def _layer3(self, protoEvidence):
		curLabelEvid, curUncEvid = deepcopy(protoEvidence[0])
		for labelEvid,uncEvid in protoEvidence[1:]:
			curLabelEvid = curLabelEvid*(labelEvid + uncEvid*np.ones(self.n_labels)) + curUncEvid*labelEvid
			curUncEvid = curUncEvid*uncEvid
		return (curLabelEvid, curUncEvid)

	def predict(self, predFeatures):
		print('Abs predict called')
		layer1 = [ self._layer1(f) for f in predFeatures ]
		layer2 = [ self._layer2(pa) for pa in layer1 ]
		layer3 = [ self._layer3(ev) for ev in layer2 ]
		return [ None for f in predFeatures ]
