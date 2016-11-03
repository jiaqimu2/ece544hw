import numpy as np
import tensorflow as tf
import os


class RBM:

	def __init__(self, 
				 v_size,
				 h_size,
				 step_size = 1,
				 iter_num = 10000,
				 batch_size = 50,
				 rand_seed = 2333,):

		self.batch_size = batch_size
		self.v_size = v_size
		self.h_size = h_size
		self.step_size = float(step_size) / batch_size
		self.iter_num = iter_num
		tf.set_random_seed(rand_seed)

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

	def fit(self, data):

		self.build_model()
		self.tf_session = tf.Session()
		self.initialize_tf()
		self.train_model(data)


	def transform(self, data, hard):

		sess = self.tf_session
		para_dict = {self.input_data: data}
		h_prob, h_sample = sess.run([self.h_prob, self.h_sample], feed_dict = para_dict)

		if hard:
			return h_sample
		else:
			return h_prob
		
	def train_model(self, data):

		self.start = 0
		self.n_sample = len(data)

		updates = [self.diff_W, self.diff_b, self.diff_c]
		for i in xrange(self.iter_num):
			ind = self.next_batch()
			para_dict = {self.input_data: data[ind]}
			detail = self.tf_session.run(updates, feed_dict = para_dict)

	def initialize_tf(self):

		init_op = tf.initialize_all_variables()
		self.tf_session.run(init_op)


	def build_model(self):

		########################
		# variables
		self.input_data = tf.placeholder('float', [None, self.v_size], name = 'x-input')
		
		########################
		# parameters
		self.W = tf.Variable(tf.random_normal((self.v_size, self.h_size), mean = 0, stddev = 0.01), name = 'weights')
		self.b = tf.Variable(tf.zeros(self.v_size), name = 'visible-bias')
		self.c = tf.Variable(tf.zeros(self.h_size), name = 'hidden-bias')

		########################
		# from v to h
		h_prob, h_sample = self.prob_hidden(self.input_data)
		v_prob = self.prob_visible(h_sample)

		########################
		# update
		self.diff_W = tf.assign_add(self.W, self.step_size * (tf.matmul(tf.transpose(self.input_data), h_prob) - tf.matmul(tf.transpose(v_prob), h_sample)))
		self.diff_b = tf.assign_add(self.b, self.step_size * tf.reduce_mean(self.input_data - v_prob, 0))
		self.diff_c = tf.assign_add(self.c, self.step_size * tf.reduce_mean(h_prob - h_sample, 0))

		########################
		# loss function
		self.loss_function = tf.sqrt(tf.reduce_mean(tf.square(self.input_data - v_prob)))
		_ = tf.scalar_summary('cost', self.loss_function)
		
		########################
		# output
		self.h_prob = h_prob
		self.h_sample = h_sample

	def prob_hidden(self, visible):

		h_prob = tf.nn.sigmoid(tf.matmul(visible, self.W) + self.c)
		h_sample = (tf.sign(h_prob - 0.5) + 1) / 2

		return h_prob, h_sample

	def prob_visible(self, hidden):

		v_prob = tf.nn.sigmoid(tf.matmul(hidden, tf.transpose(self.W)) + self.b)

		return v_prob