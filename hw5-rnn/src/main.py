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

FIGURE_PATH = '/Users/mujq10/ece544hw/hw5-rnn/figures/'
DATA_PATH = '/Users/mujq10/ece544hw/hw5-rnn/data/'
os.system('mkdir -p %s' % FIGURE_PATH)
os.system('mkdir -p %s' % DATA_PATH)

def evaluate(true_Label, pred_Label):

	true_Label = np.argmax(true_Label, axis = 1)
	pred_Label = np.argmax(pred_Label, axis = 1)
	return np.sum(true_Label == pred_Label) / float(len(true_Label))

def preprocess(data, dimension):

	num_sample, _ = data.shape
	return data.reshape((num_sample, -1, dimension))

def classification(train_data, train_label, test_data, test_label, label = 'RNN'):
	
	if label == 'RNN':
		classifier = RNN()
	else:
		classifier = LSTM()

	iter_nums, accuracies = classifier.fit(train_data, train_label)
	fig_file = FIGURE_PATH + '%s-%d-%d.pdf' % (label, train_data.shape[1], train_data.shape[2])
	plt.figure()
	plt.plot(iter_nums, accuracies, '-o', linewidth = 3)
	plt.xlabel('training iteration')
	plt.ylabel('training-corpus accuracy')
	plt.savefig(fig_file, format = 'pdf')

	pred_label = classifier.predict(test_data)
	classifier.tf_session.close()
	print '-'*50
	print '%s-%d-%d test accuracy: %0.4f' % (label, train_data.shape[1], train_data.shape[2], evaluate(pred_label, test_label))

if __name__ == '__main__':
	
	mnist = input_data.read_data_sets(DATA_PATH, one_hot=True)

	train_data, train_label = mnist.train.images, mnist.train.labels
	test_data, test_label = mnist.test.images, mnist.test.labels

	train_tmp_data = preprocess(train_data, 28)
	test_tmp_data = preprocess(test_data, 28)

	classification(train_tmp_data, train_label, test_tmp_data, test_label, 'RNN')
	classification(train_tmp_data, train_label, test_tmp_data, test_label, 'LSTM')

	train_tmp_data = preprocess(train_data, 1)
	test_tmp_data = preprocess(test_data, 1)

	classification(train_tmp_data, train_label, test_tmp_data, test_label, 'RNN')
	classification(train_tmp_data, train_label, test_tmp_data, test_label, 'LSTM')