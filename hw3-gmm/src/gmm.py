import numpy as np
from scipy.sparse import diags



class GMM:

	def __init__(self, 
				 ncluster,
				 randomSeed = 2333,
				 iterNum = 10):

		self.ncluster = ncluster
		self.iterNum = iterNum
		np.random.seed(randomSeed)

	def gammaprob(self, data):

		cents, covs, weights = self.cents, self.covs, self.weights
	
		dets = np.linalg.det(covs)
		try:
			invs = np.linalg.inv(covs)
		except:
			return []

		ncluster = len(dets)
		nsample = len(data)

		probs = np.zeros((ncluster, nsample))
		for i in xrange(ncluster):
			bias = data - cents[i]
			probs[i] = weights[i] / dets[i] * np.exp(-(np.sum(np.dot(bias, invs[i]) * bias, axis = 1)))

		probs = probs / np.sum(probs, axis = 0, keepdims = True)

		return probs

	def fit(self, data):
		'''
			input: [nsamples, nfeatures]
		'''
		##########################
		# internal functions
		def estep():
			
			return self.gammaprob(data)

		def mstep():
			
			ncluster, nsample = postProb.shape
			_, ndim = data.shape

			newWeights = np.sum(postProb, axis = 1) / nsample
			newCents = np.zeros((ncluster, ndim))
			newCovs = np.zeros((ncluster, ndim, ndim))

			for i in xrange(ncluster):
				newCents[i] = np.dot(postProb[i], data) / np.sum(postProb[i])
				bias = (data - newCents[i])
				cov  = np.dot(np.multiply(bias.T, postProb[i]), bias)
				newCovs[i] =  cov / np.sum(postProb[i])

			return newCents, newCovs, newWeights

		##########################


		data = np.array(data, dtype=np.float32)
		nsample, ndim = data.shape
		ncluster = self.ncluster
		iterNum = self.iterNum


		while True:
			
			# initialization
			cents = np.random.normal(size=(ncluster, ndim))
			covs = np.array([np.eye(ndim) * 1000 for i in xrange(ncluster)])
			weights = np.abs(np.random.normal(size=(ncluster)))
			weights /= np.sum(weights)
			self.cents = cents
			self.weights = weights
			self.covs = covs

			# iteration
			postProb = estep()
			for i in xrange(iterNum):
				if np.sum(postProb) == 0:
					break
				self.cents, self.covs, self.weights = mstep()
				postProb = estep()

			if len(postProb) != 0:
				break
			else:
				iterNum -= 1
				print 'bad initialization, start over.'

	def predict(self, data):
		'''
			input: [nsamples, nfeatures]
			output: [nsamples] indicating which cluster each sample belongs to
		'''
		postProb = self.gammaprob(data)
		return np.argmax(postProb, axis=0)

	def clusters(self):
		'''
			output: centers for each cluster
		'''
		return self.cents

	def covariances(self):
		'''
			output: variances for each cluster
		'''
		return self.covs

	def getWeights(self):

		return self.weights
