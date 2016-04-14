import numpy as np

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

		self.prototypes = np.random.random([self.n_prototypes, self.dim])
		self.alphas = np.random.random(self.n_prototypes)
		self.gammas = np.random.random(self.n_prototypes)
		self.mem_degrees = np.zeros([self.n_prototypes, self.n_labels])
		for i in range(self.n_prototypes):
			self.mem_degrees[i][np.random.randint(self.n_labels)] = 1.

		return self

	def _layer1(self, feat):
		return [ alpha*np.exp(-1.*gamma*_distance(feat, prot)) for alpha,gamma,prot in zip(self.alphas, self.gammas, self.prototypes) ]

	def _layer2(self, protoActivations):
		return None

	def predict(self, predFeatures):
		print('Abs predict called')
		layer1 = [ self._layer1(f) for f in predFeatures ]
		return [ None for f in predFeatures ]
