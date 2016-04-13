import numpy as np

class Abs(object):
	def __init__(self, params):
		print('Abs classifier initialized')

	def fit(self, features, labels):
		print('Abs fit called')
		return self

	def _predict(self, predFeature):
		return None

	def predict(self, predFeatures):
		print('Abs predict called')
		return [ self._predict(f) for f in predFeatures ]
