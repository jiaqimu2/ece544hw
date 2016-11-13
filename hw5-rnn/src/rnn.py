import tensorflow as tf
import numpy as np

class RNN:

	def __init__(self, 
				 hidden_size = 100,
				 step_size = 0.01,
				 batch_size = 500,
				 epoch_size = 1000,
				 display_size = 100,
				 rand_seed = 2333):

		self.hidden_size = hidden_size
		self.step_size = step_size
		self.batch_size = 50
		self.start = 0
		self.epoch_size = epoch_size
		self.display_size = display_size
		tf.set_random_seed(rand_seed)

	def next_batch(self, ):

		batch_size = self.batch_size
		n_sample = self.n_sample

		if self.start + batch_size < n_sample:
			self.start += batch_size
			return range(self.start - batch_size, self.start)
		else:
			ind = range(self.start, n_sample)
			self.start = (self.start + batch_size) % n_sample
			return ind + range(self.start)

	def fit(self, data, label):
		
		self.n_sample, self.n_steps, self.n_dim = data.shape
		_, self.n_class = label.shape

		self.build_model()
		self.tf_session = tf.Session()
		self.initialize_tf()

		epochs = list()
		accuracies = list()

		epoch_size = self.epoch_size
		display_size = self.display_size

		for i in xrange(epoch_size):
			batch_ind = self.next_batch()
			self.tf_session.run(self.optimizer, feed_dict = {self.input_data: data[batch_ind], self.label: label[batch_ind], self.hidden_state: np.zeros((self.batch_size, self.hidden_size))})

			if i % display_size == 0:
				epochs.append(i)
				accuracies.append(self.tf_session.run(self.accuracy, feed_dict = {self.input_data: data, self.label: label, self.hidden_state: np.zeros((self.n_sample, self.hidden_size))}))

				print 'epoch: %d\taccuracy: %0.4f' % (epochs[-1], accuracies[-1])
			
		return epochs, accuracies

	def predict(self, data):

		return self.tf_session.run(self.pred, feed_dict = {self.input_data: data, self.hidden_state: np.zeros((len(data), self.hidden_size))})

	def build_model(self):

		########################
		# variables
		input_data = tf.placeholder('float', [None, self.n_steps, self.n_dim], name = 'x-input')
		hidden_state = tf.placeholder('float', [None, self.hidden_size], name = 'hidden')
		label = tf.placeholder('float', [None, self.n_class])

		########################
		# parameters
		self.hidden_weights = tf.Variable(tf.random_normal((self.n_dim, self.hidden_size), mean = 0, stddev = 0.01), name = 'hidden-weights')
		self.output_weights = tf.Variable(tf.random_normal((self.hidden_size, self.n_class), mean = 0, stddev = 0.01), name = 'output-weights')
		self.hidden_bias = tf.Variable(tf.zeros(self.hidden_size), name = 'hidden-bias')
		self.output_bias = tf.Variable(tf.zeros(self.n_class), name = 'output-bias')

		########################
		# model

		_X = tf.transpose(input_data, [1, 0, 2])
		_X = tf.reshape(_X, [-1, self.n_dim])
		_X = tf.matmul(_X, self.hidden_weights) + self.hidden_bias
		_X = tf.split(0, self.n_steps, _X)

		rnn_cell = tf.nn.rnn_cell.BasicRNNCell(self.hidden_size)
		outputs, states = tf.nn.rnn(rnn_cell, _X, initial_state = hidden_state)

		pred = tf.matmul(outputs[-1], self.output_weights) + self.output_bias

		cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, label))
		optimizer = tf.train.AdamOptimizer(learning_rate = self.step_size).minimize(cost)
		accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred, 1), tf.argmax(label, 1)), tf.float32))

		self.input_data = input_data
		self.hidden_state = hidden_state
		self.label = label
		self.optimizer = optimizer
		self.accuracy = accuracy
		self.pred = pred


	def initialize_tf(self):

		init_op = tf.initialize_all_variables()
		self.tf_session.run(init_op)

