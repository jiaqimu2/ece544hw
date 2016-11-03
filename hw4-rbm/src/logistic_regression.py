import tensorflow as tf
import numpy as np

## logistic regression 

class Logistic:

	def __init__(self, 
				 step_size = 0.2,
				 iter_num = 10000,
				 feature_num = 784,
				 class_num = 10,
				 batch_size = 50,
				 rand_seed = 2333):

		# initialization
		x = tf.placeholder(tf.float32, [None, feature_num])
		W = tf.Variable(tf.zeros([feature_num, class_num]))
		b = tf.Variable(tf.zeros([class_num]))

		# model
		y = tf.nn.softmax(tf.matmul(x, W) + b)
		y_ = tf.placeholder(tf.float32, [None, class_num])
		cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
		train_step = tf.train.GradientDescentOptimizer(step_size).minimize(cross_entropy)

		##########################

		self.batch_size = batch_size
		self.step_size = float(step_size)
		self.iter_num = iter_num
		self.feature_num = feature_num
		self.class_num = class_num
		tf.set_random_seed(rand_seed)

		self.x = x
		self.W = W
		self.b = b
		self.y = y
		self.y_ = y_
		self.cross_entropy = cross_entropy
		self.train_step = train_step

	def next_batch(self,):

		batch_size = self.batch_size
		n_sample = self.n_sample

		if self.start + batch_size < n_sample:
			self.start += batch_size
			return range(self.start - batch_size, self.start)
		else:
			ind = range(self.start, n_sample)
			self.start = (self.start + batch_size) % n_sample
			return ind + range(self.start)
		
	def fit(self, train_data, train_label):

		init = tf.initialize_all_variables()
		self.start = 0
		self.n_sample = len(train_data)
		self.sess = tf.Session()
		self.sess.run(init)

		for i in xrange(self.iter_num):
			ind = self.next_batch()
			self.sess.run(self.train_step, feed_dict = {self.x: train_data[ind], self.y_:train_label[ind]})
			# if i % 1000 == 0 and i != 0:
			# 	print 'iter-%d' % i, self.sess.run(self.cross_entropy, feed_dict = {self.x: train_data, self.y_:train_label})
	
	def transform(self, test_data):

		y = self.sess.run(self.y, feed_dict = {self.x: test_data})
		return np.argmax(y, axis = 1)