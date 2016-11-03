###################################
## logistic regression
###################################

import numpy as np

def sigmoid(x):

	return 1./(1+np.exp(-x))

class Logistic:

	def __init__(self, stepSize, iterNum):

		self.stepSize = stepSize
		self.iterNum = iterNum

	def train(self, train, label):

		if len(train) != len(label):
			print 'Sample mismatch!'
			return 0

		sample, dim = train.shape
		## x' = [x, 1]
		x = np.ones((sample, dim+1))
		x[:, 0:dim] = train
		t = label
		## initialization
		np.random.seed(0)
		coeff = np.random.rand(dim + 1)

		## gradient descent
		errs = list()
		for i in xrange(self.iterNum):
			y = sigmoid(np.dot(x, coeff))
			## gradient
			g = np.dot((y - t) * y * (1-y), x)
			## update
			coeff -= self.stepSize * g
			
			y = np.dot(x, coeff)
			algoLabel = (y > 0) * 1 + (y <= 0) * 0
			errs.append(sum(algoLabel!= label) / float(len(label)))

		self.dim = dim
		self.coeff = coeff

		return errs

	def test(self, test):

		sample, dim = test.shape
		if self.dim != dim:
			print 'Dimension mismatch!'
			return 0

		x = np.ones((sample, dim+1))
		x[:, 0:dim] = test
		y = np.dot(x, self.coeff)
		# thresholding
		return (y > 0)*1 + (y <= 0)*0
