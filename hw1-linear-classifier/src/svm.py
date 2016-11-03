###################################
## svm classifier
###################################

import numpy as np

class SVM:

	def __init__(self, stepSize, iterNum, c):

		self.stepSize = stepSize
		self.iterNum = iterNum
		self.c = c

	def train(self, train, label):

		label = 2*label - 1

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
			# gradient
			g = - self.c * np.dot(t*(1*(np.dot(x, coeff)*t < 1)), x)
			g[0:dim] += 2*coeff[0:dim]
			## update
			coeff -= self.stepSize * g
			
			# err
			y = np.dot(x, coeff)
			algoLabel = (y > 0) * 1 + (y <= 0) * (-1)
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
