#!/usr/bin/python

import numpy as np
import tensorflow as tf
import os
from tensorflow.examples.tutorials.mnist import input_data
from logistic_regression import Logistic
from rbm_part3 import RBM
from sklearn.decomposition import PCA
from plot import plot_confusion_matrix

import matplotlib 
matplotlib.use('Agg')
from matplotlib import pyplot as plt

FIGURE_PATH = '/home/jiaqimu2/ece544hw/hw4-rbm/figures/'
DATA_PATH = '/home/jiaqimu2/ece544hw/hw4-rbm/data/'
os.system('mkdir -p %s' % FIGURE_PATH)
os.system('mkdir -p %s' % DATA_PATH)

def evaluate(true_Label, pred_Label, figFile):

	true_Label = np.argmax(true_Label, axis = 1)

	plt.figure()
	plot_confusion_matrix(true_Label, pred_Label)
	plt.savefig(figFile, format = 'pdf', transparent = True)
	
	return np.sum(true_Label == pred_Label) / float(len(true_Label))

def logistic(train_data, train_label, test_data, test_label, label):

	_, feature_num = train_data.shape
	_, class_num = train_label.shape

	logistic = Logistic(step_size = 0.2,
				 		iter_num = 10000,
						feature_num = feature_num,
						class_num = class_num,
						batch_size = 50,
						rand_seed = 2333)
	logistic.fit(train_data, train_label)

	pred_train_label = logistic.transform(train_data)
	pred_test_label = logistic.transform(test_data)

	train_fig = FIGURE_PATH + 'train-%s.pdf' % label
	test_fig = FIGURE_PATH + 'test-%s.pdf' % label
	print 'train:', evaluate(train_label, pred_train_label, train_fig)
	print 'test:', evaluate(test_label, pred_test_label, test_fig)


def raw(train_data, train_label, test_data, test_label):

	print '-' * 50
	label = 'raw'
	print 'predict with %s data using logistic regression:' % label
	logistic(train_data, train_label, test_data, test_label, label)


def pca(pca_size, train_data, train_label, test_data, test_label):

	print '-' * 50
	label = 'pca'
	print 'predict with %s using logistic regression' % label

	pca = PCA(n_components = pca_size)
	pca.fit(train_data)
	train_data_pca = pca.transform(train_data)
	test_data_pca = pca.transform(test_data)
	logistic(train_data_pca, train_label, test_data_pca, test_label, label)

def rbm(h_list, train_data, train_label, test_data, test_label):

	print '-' * 50
	label = 'rbm-%s' % '-'.join([str(v) for v in h_list])
	print 'predict with %s using logistic regression:' % label

	_, class_num = train_label.shape

	train_data_rbm = train_data
	test_data_rbm = test_data

	for i, h_size in enumerate(h_list):
		_, feature_num = train_data_rbm.shape

		rbm = RBM(feature_num, h_size)
		rbm.fit(train_data_rbm)

		if i < len(h_list) - 1:
			train_data_rbm = rbm.transform(train_data_rbm, True)
			test_data_rbm = rbm.transform(test_data_rbm, True)
		else:
			train_data_rbm = rbm.transform(train_data_rbm, False)
			test_data_rbm = rbm.transform(test_data_rbm, False)
	
	logistic(train_data_rbm, train_label, test_data_rbm, test_label, label)

if __name__ == '__main__':
	
	mnist = input_data.read_data_sets(DATA_PATH, one_hot=True)

	train_data, train_label = mnist.train.images, mnist.train.labels
	test_data, test_label = mnist.test.images, mnist.test.labels

	os.system('mkdir -p %s' % FIGURE_PATH)

	raw(train_data, train_label, test_data, test_label)
	pca(200, train_data, train_label, test_data, test_label)
	rbm([200], train_data, train_label, test_data, test_label)
	rbm([500, 200], train_data, train_label, test_data, test_label)

	
	