import numpy as np

np.random.seed(2333)

def initWeightMatrix(shape):
	return np.random.normal(size = shape, scale = 0.1)

def initBias(shape, nonlinear):
	if nonlinear == 'relu':
		return np.ones(shape) * 0.1
	else:
		return np.random.normal(size = shape, scale = 0.1)


class FNN:

	def __init__(self, nfeatures, hiddenSize, nclasses, stepSize, batchSize, iterations, nonlinear):

		self.stepSize = stepSize
		self.batchSize = batchSize
		self.iterations = iterations
		self.nhidden = len(hiddenSize)
		self.nonlinear = nonlinear

		hiddenW = list()
		hiddenB = list()
		for i in xrange(self.nhidden):
			if i == 0:
				w = initWeightMatrix((nfeatures, hiddenSize[i]))
			else:
				w = initWeightMatrix((hiddenSize[i-1], hiddenSize[i]))
			b = initBias(hiddenSize[i], nonlinear)
			hiddenW.append(w)
			hiddenB.append(b)
		outputW = initWeightMatrix((hiddenSize[-1], nclasses))
		outputB = initBias(nclasses, nonlinear)

		self.hiddenW = hiddenW
		self.hiddenB = hiddenB
		self.outputW = outputW
		self.outputB = outputB

	def test(self, samples):

		# input -> hidden
		a = samples
		for i in xrange(self.nhidden):
			z = np.dot(a, self.hiddenW[i]) + self.hiddenB[i]
			a = self.activation(z)

		# hidden -> output
		z = np.dot(a, self.outputW) + self.outputB
		return np.argmax(z, 1)

	def train(self, trainData):

		hiddenA = [0] * self.nhidden
		hiddenZ = [0] * self.nhidden
		dhiddenW = [0] * self.nhidden
		dhiddenB = [0] * self.nhidden

		for j in xrange(self.iterations):
			x, y = trainData.nextBatch(self.batchSize)
			
			### feedforward
			# input -> hidden
			for i in xrange(self.nhidden):
				if i == 0:
					z = np.dot(x, self.hiddenW[i]) + self.hiddenB[i]
				else:
					z = np.dot(a, self.hiddenW[i]) + self.hiddenB[i]
				a = self.activation(z)
				hiddenA[i] = a
				hiddenZ[i] = z
			# hidden -> output
			zOut = np.dot(a, self.outputW) + self.outputB
			aOut = np.exp(zOut) / np.sum(np.exp(zOut), 1, keepdims = True)

			# if j % 1000 == 0 and j != 0:
			# 	loss = -np.sum(np.log(aOut[range(self.batchSize), y]))/self.batchSize
			# 	acc = sum(y == np.argmax(aOut, 1)) / float(self.batchSize)
			# 	print 'iter: %d acc: %f loss: %f' % (j, acc, loss)

			### back-propagation
			# output -> hidden
			dloss = aOut
			dloss[range(self.batchSize), y] -= 1
			dloss /= self.batchSize
			dOutputW = np.dot(hiddenZ[-1].T, dloss)
			dOutputB = np.sum(dloss, 0)

			# hidden -> input
			dloss = np.dot(dloss, self.outputW.T)
			for i in xrange(self.nhidden - 1, -1, -1):
				a = hiddenA[i]
				if i == 0:
					z = x
				else:
					z = hiddenZ[i-1]
				dloss *= self.gradient(a)
				dhiddenW[i] = np.dot(z.T, dloss)
				dhiddenB[i] = np.sum(dloss, 0)
				dloss = np.dot(dloss, self.hiddenW[i].T)

			### gradient update
			self.outputB -= self.stepSize * dOutputB
			self.outputW -= self.stepSize * dOutputW
			for i in xrange(self.nhidden):
				self.hiddenW[i] -= dhiddenW[i] * self.stepSize
				self.hiddenB[i] -= dhiddenB[i] * self.stepSize

	def gradient(self, y):
		if self.nonlinear == 'sigmoid':
			grad = y * (1-y)
		elif self.nonlinear == 'tanh':
			grad = 1 - y * y
		else: # relu
			grad = 1 * (y>0)
		return grad

	def activation(self, z):
		if self.nonlinear == 'sigmoid':
			a = 1.0 / (1 + np.exp(-z))
		elif self.nonlinear == 'tanh':
			a = 2.0 / (1 + np.exp(-z)) - 1
		else:
			a = z.clip(min=0)
		return a
