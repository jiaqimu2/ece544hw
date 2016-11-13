#!/usr/bin/python

import numpy as np
import tensorflow as tf
import os
from tensorflow.examples.tutorials.mnist import input_data
from rnn import RNN
from lstm import LSTM

import matplotlib 
matplotlib.use('Agg')
from matplotlib import pyplot as plt

FIGURE_PATH = '/home/jiaqimu2/ece544hw/hw5-rnn/figures/'
DATA_PATH = '/home/jiaqimu2/ece544hw/hw5-rnn/data/'
os.system('mkdir -p %s' % FIGURE_PATH)
os.system('mkdir -p %s' % DATA_PATH)

def evaluate(true_Label, pred_Label):

	true_Label = np.argmax(true_Label, axis = 1)
	pred_Label = np.argmax(pred_Label, axis = 1)
	return np.sum(true_Label == pred_Label) / float(len(true_Label))

def preprocess(data, dimension):

	num_sample, num_dim = data.shape
	# downsampling
	if dimension <=2: 
		data = data.reshape(num_sample, 28, 28)
		data = data[:, ::2, ::2]
	return data.reshape((num_sample, -1, dimension))

def classification(train_data, train_label, test_data, test_label, label = 'RNN'):
	
	print '-'*50
	if label == 'RNN':
		classifier = RNN(epoch_size = 300, hidden_size = 100, batch_size = 128)
	else:
		classifier = LSTM(epoch_size = 300, hidden_size = 100, batch_size = 128)

	iter_nums, accuracies = classifier.fit(train_data, train_label)
	fig_file = FIGURE_PATH + '%s-%d-%d.pdf' % (label, train_data.shape[1], train_data.shape[2])
	plt.figure()
	plt.plot(iter_nums, accuracies, '-o', linewidth = 3)
	plt.xlabel('training iteration')
	plt.ylabel('training-corpus accuracy')
	plt.savefig(fig_file, format = 'pdf')

	pred_label = classifier.predict(test_data)
	classifier.tf_session.close()
	print '%s-%d-%d test accuracy: %0.4f' % (label, train_data.shape[1], train_data.shape[2], evaluate(pred_label, test_label))


if __name__ == '__main__':
	
	mnist = input_data.read_data_sets(DATA_PATH, one_hot=True)

	train_data, train_label = mnist.train.images, mnist.train.labels
	test_data, test_label = mnist.test.images, mnist.test.labels

	# setting 1 and setting 2 cannot run in one shot!!!!
	# setting 1
	# train_tmp_data = preprocess(train_data, 28)
	# test_tmp_data = preprocess(test_data, 28)

	# classification(train_tmp_data, train_label, test_tmp_data, test_label, 'RNN')
	# classification(train_tmp_data, train_label, test_tmp_data, test_label, 'LSTM')

	# setting 2 / downsampling 
	train_tmp_data = preprocess(train_data, 1)
	test_tmp_data = preprocess(test_data, 1)

	classification(train_tmp_data, train_label, test_tmp_data, test_label, 'RNN')
	classification(train_tmp_data, train_label, test_tmp_data, test_label, 'LSTM')
