import numpy as np
from copy import deepcopy

def _distance(x1, x2):
	return np.linalg.norm(x2-x1)

class Abs(object):
	def __init__(self, n_prototypes=10):
		print('Abs classifier initialized')
		self.n_prototypes = n_prototypes

	def _initializeExplicitParametersRandomly(self):
		self.prototypes = np.random.random([self.n_prototypes, self.dim]) # prototypes - the surrogate nearest neighbor vectors
		self.alphas = np.random.random(self.n_prototypes) # how certain we are in the membership degrees of the prototype
		self.gammas = np.random.random(self.n_prototypes) # how far away should the influence of the prototype extend
		self.mem_degrees = np.zeros([self.n_prototypes, self.n_labels]) # membership of the prototype in every label
		for i in range(self.n_prototypes):
			self.mem_degrees[i][np.random.randint(self.n_labels)] = 1.

	def _initializeImplicitParametersRandomly(self):
		self.prototypes = np.random.random([self.n_prototypes, self.dim]) # prototypes - the surrogate nearest neighbor vectors
		self.ksi = 2.*np.random.random(self.n_prototypes) - 1.
		self.eta = 2.*np.random.random(self.n_prototypes) - 1.
		self.beta = np.zeros([self.n_prototypes, self.n_labels])
		for i in range(self.n_prototypes):
			self.beta[i][np.random.randint(self.n_labels)] = np.random.choice([-1., 1.])
		self.gradIdxs = [0,0,0]
		self.gradIdxs[0] = self.n_labels*self.n_prototypes
		self.gradIdxs[1] = self.gradIdxs[0]+self.n_prototypes
		self.gradIdxs[2] = self.gradIdxs[1]+self.n_prototypes

	def _updateExplicit(self):
		self.alphas = 1./(1. + np.exp(-1.*self.ksi))
		self.gammas = self.eta**2
		betaSquared = self.beta**2
		self.mem_degrees = np.array([ betaSquared[i]/sum(betaSquared[i]) for i in range(self.n_prototypes) ])

	def _getBatchGradient(self, features, labels, nu):
		gradient = np.zeros(self.n_prototypes*(2+self.dim+self.n_labels))
		for f,l in zip(features, labels):
			print 'Computing gradient of feature ' + str(f) + ' label ' + str(l)
			self._cacheAux(f, l, nu)
			gradient[0:self.gradIdxs[0]] += self._getBetaGradient(f, l)
			gradient[self.gradIdxs[0]:self.gradIdxs[1]] += self._getEtaGradient(f, l)
			gradient[self.gradIdxs[1]:self.gradIdxs[2]] += self._getKsiGradient(f, l)
			gradient[self.gradIdxs[2]:] += self._getPrototypesGradient(f, l)
		gradient /= len(self.labels)
		return gradient

	def _cacheAux(self, feature, label, nu):
		'''Caches the variables used in the gradient computation'''
		self._prototypeActivations = self._layer1(feature)
		self._evidenceItems = self._layer2(self._prototypeActivations)
		self._finalBBA = self._layer3(self._evidenceItems)
		self._evidenceItemsBar = [ self._evidenceBar(e) for e in self._evidenceItems ]
		pignisticBBA, uncertainty = self._finalBBA
		pignisticBBA += nu*uncertainty
		desiredBBA = np.zeros(self.n_labels)
		desiredBBA[label] = 1.
		self._pignisticError = pignisticBBA - desiredBBA
		self._sgrad = np.zeros(self.n_prototypes)
		for i in range(self.n_prototypes):
			lEvBar,uEvBar = self._evidenceItemsBar[i]
			self._sgrad[i] = sum( self._pignisticError*(self.mem_degrees[i]*(lEvBar+uEvBar) - lEvBar - nu*uEvBar) ) # eq (86)
			# print str(self._pignisticError) + ' * ' + str(self.mem_degrees[i]*(lEvBar+uEvBar) - lEvBar - nu*uEvBar)
		# print str(self._sgrad)
		self._dsquare = self._protSqDist(feature)

	def _evidenceBar(self, evidence):
		fLabEvid,fUncEvid = self._finalBBA
		labEvid,uncEvid = evidence
		uncEvidBar = fUncEvid/uncEvid # eq (77)
		labEvidBar = np.zeros(self.n_labels)
		for j in range(self.n_labels):
			labEvidBar[j] = ( fLabEvid[j] - labEvid[j]*uncEvidBar )/( labEvid[j] + uncEvid ) # eq (76)
		# print str(evidence) + ' + ' + str(self._finalBBA) + ' -> ' + str((labEvidBar, uncEvidBar))
		return (labEvidBar, uncEvidBar)

	def _getBetaGradient(self, feature, label):
		ugrad = np.zeros([self.n_prototypes, self.n_labels])
		for i in range(self.n_prototypes):
			lEvBar,uEvBar = self._evidenceItemsBar[i]
			ugrad[i] = self._pignisticError*(lEvBar + uEvBar)*self._prototypeActivations[i] # eq (79)
		bgrad = np.zeros([self.n_prototypes, self.n_labels])
		for i in range(self.n_prototypes):
			bsquared = self.beta[i]**2
			bnorm = sum(bsquared)
			wbnorm = sum(bsquared*ugrad[i])
			bgrad[i] = self.beta[i]*(ugrad[i]*bnorm - wbnorm)*2./(bnorm**2) # eq (68)
			# print str(self.beta[i]) + ' + ' + str(ugrad[i]) + ' + ' + str(bnorm) + ' + ' + str(wbnorm) + ' + ' + str(ugrad[i]*bnorm - wbnorm)
		# print repr(bgrad)
		if np.count_nonzero(bgrad) == 0:
			print 'WARNING: beta gradient vanished for feature ' + str(feature) + ' label ' + str(label)
		return bgrad.reshape(self.n_labels*self.n_prototypes)

	def _getEtaGradient(self, feature, label):
		etagrad = self._sgrad*self.eta*self._dsquare*self._prototypeActivations*(-2.)
		if np.count_nonzero(etagrad) == 0:
			print 'WARNING: eta gradient vanished for feature ' + str(feature) + ' label ' + str(label)
		# print repr(etagrad)
		return etagrad

	def _getKsiGradient(self, feature, label):
		ksigrad = self._sgrad*(1.-self.alphas)*self.alphas*np.exp(self._dsquare*self.gammas*(-1.))
		if np.count_nonzero(ksigrad) == 0:
			print 'WARNING: ksi gradient vanished for feature ' + str(feature) + ' label ' + str(label)
		# print repr(ksigrad)
		return ksigrad

	def _getPrototypesGradient(self, feature, label):
		return np.ones(self.n_prototypes*self.dim)

	def _getBatchError(self):
		return 0.

	def _stepOptimization(self, gradient, learning_rate):
		gradient *= learning_rate
		self.beta -= gradient[0:self.gradIdxs[0]].reshape(self.beta.shape)
		self.eta -= gradient[self.gradIdxs[0]:self.gradIdxs[1]]
		self.ksi -= gradient[self.gradIdxs[1]:self.gradIdxs[2]]
		self.prototypes -= gradient[self.gradIdxs[2]:].reshape(self.prototypes.shape)
		self._updateExplicit()

	def fit(self, features, labels, max_iterations=1000, epsilon=0.1, learning_rate=0.1, momentum=0.0):
		print('Abs fit called')
		self.features = np.array(features)
		self.dim = self.features.shape[1]
		self.labels = np.array(labels)
		self.n_labels = len(np.unique(self.labels))
		self._initializeImplicitParametersRandomly()
		self._updateExplicit()
		prevGrad = np.zeros(self.n_prototypes*(2+self.dim+self.n_labels))
		for i in range(max_iterations):
			curGrad = self._getBatchGradient(self.features, self.labels, 1./self.n_labels) # PIGNISTIC OUTPUT
			self._stepOptimization(curGrad + prevGrad*momentum, learning_rate)
			if self._getBatchError() < epsilon:
				break
			prevGrad = curGrad
		return self

	def _layer1(self, feat):
		return [ alpha*np.exp(-1.*gamma*sqDist) for alpha,gamma,sqDist in zip(self.alphas, self.gammas, self._protSqDist(feat)) ]

	def _protSqDist(self, feat):
		return [ sum((feat-prot)**2) for prot in self.prototypes ]

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
