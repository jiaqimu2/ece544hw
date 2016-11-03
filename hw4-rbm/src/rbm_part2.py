import numpy as np

class RBM:

	def __init__(self, 
				 h_size,
				 step_size = 1,
				 iter_num = 5000,
				 batch_size = 500,
				 rand_seed = 2333):

		self.batch_size = batch_size
		self.h_size = h_size
		self.step_size = float(step_size)
		self.iter_num = iter_num
		np.random.seed(rand_seed)

	def nextBatch(self,):

		batch_size = self.batch_size
		n_sample = self.n_sample

		if self.start + batch_size < n_sample:
			self.start += batch_size
			return range(self.start - batch_size, self.start)
		else:
			ind = range(self.start, n_sample)
			self.start = (self.start + batch_size) % n_sample
			return ind + range(self.start)


	def fit(self, data):

		def probHidden(W, b, c, data):

			return 1 / (1 + np.exp(-np.dot(data, W) - c))

		def probVisible(W, b, c, h, data):

			return 1 / (1 + np.exp(-np.dot(h, W.T) - b))

		def randSampleHidden(prob_hidden):

			return 1 * (0.5 < prob_hidden) + 0 * (prob_hidden <= 0.5)

		np.random.shuffle(data)

		#########################
		# parameters
		h_size = self.h_size
		step_size = self.step_size
		iter_num = self.iter_num
		n_sample, v_size = data.shape
		batch_size = self.batch_size
		self.start = 0
		self.n_sample = n_sample

		#########################
		# initialization
		W = np.random.normal(size = (v_size, h_size)) * 0.1
		b = np.random.normal(size = (v_size)) * 0.1
		c = np.random.normal(size = (h_size)) * 0.1
		

		#########################
		# training
		for i in xrange(iter_num):


			ind = self.nextBatch()
			prob_hidden = probHidden(W, b, c, data[ind])
			h = randSampleHidden(prob_hidden)
			prob_visible = probVisible(W, b, c, h, data[ind])

			diff_W = np.dot(data[ind].T, prob_hidden) - np.dot(prob_visible.T, h)
			diff_b = np.sum((data[ind] - prob_visible), axis =0)
			diff_c = np.sum((prob_hidden - h), axis = 0)

			W += step_size / batch_size * diff_W
			b += step_size / batch_size * diff_b
			c += step_size / batch_size * diff_c

			if i % 500 == 0:
				print 'mse', np.linalg.norm(prob_visible - data[ind]) ** 2 / batch_size / v_size

		self.W = W
		self.b = b
		self.c = c


	def getWeight(self):

		return self.W
