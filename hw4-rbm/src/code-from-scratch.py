#!/usr/bin/python

import numpy as np
import tensorflow as tf
from rbm_part2 import RBM
from tensorflow.examples.tutorials.mnist import input_data

import matplotlib 
import os
matplotlib.use('Agg')
from matplotlib import pyplot as plt

np.random.seed(2333)

FIGURE_PATH = '/home/jiaqimu2/ece544hw/hw4-rbm/figures/'
DATA_PATH = '/home/jiaqimu2/ece544hw/hw4-rbm/data/'
os.system('mkdir -p %s' % FIGURE_PATH)
os.system('mkdir -p %s' % DATA_PATH)

if __name__ == '__main__':
	
	h_size = 200
	data = input_data.read_data_sets(DATA_PATH, one_hot=True).train.images
	rbm = RBM(h_size)
	rbm.fit(data)

	filters = rbm.getWeight().T
	inds = np.random.choice(len(filters), size = 64)

	figFile = FIGURE_PATH + 'part2-filters.pdf' 
	fig = plt.figure(figsize=(6, 11))

	# axes = axes.ravel()

	for i, ind in enumerate(sorted(inds)):	
		filter = filters[ind].reshape(28, 28)
		ax = fig.add_subplot(11, 6, i+1)
		ax.axes.get_xaxis().set_visible(False)
		ax.axes.get_yaxis().set_visible(False)
		ax.imshow(filter, cmap=plt.get_cmap('gray'))
	plt.savefig(figFile, bbox_inches='tight', pad_inches=0)
	plt.close()

