import tensorflow as tf
import numpy as np
import matplotlib 
import sys
matplotlib.use('Agg')
from plot import plot_confusion_matrix
import matplotlib.pyplot as plt
from data import Data

directory = '/home/jiaqimu2/ece544hw/hw2-neural-network/data/'
train = 'train/lab/hw2train_labels.txt'
test = 'eval/lab/hw2eval_labels.txt'
dev = 'dev/lab/hw2dev_labels.txt'

batchSize = 50
stepSize = 1e-3

# weight initialization
def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1, seed = 2333)
	return tf.Variable(initial)

# bias initialization
def bias_variable(shape, nonlinear):
	if nonlinear == 'relu':
		initial = tf.constant(0.1, shape=shape)
	else:
		initial = tf.truncated_normal(shape, stddev=0.01, seed = 2333)
	return tf.Variable(initial)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='VALID')

if __name__ == '__main__':

	trainData = Data(directory, train)
	devData = Data(directory, dev)
	testData = Data(directory, test)

	for output in ['sum', 'fnn']:
		for nonlinear in ['sigmoid', 'relu']:

			if nonlinear == 'relu':
				iteration = 10000
			else:
				iteration = 100000

			sess = tf.InteractiveSession()
			x = tf.placeholder('float', shape = [None, 70, 16])
			y_ = tf.placeholder('float', shape = [None, 9])
			inX = tf.reshape(x, [-1, 70, 1, 16])
			# sanity check
			# print x.get_shape(), inX.get_shape()

			# first hidden layer
			Wconv1 = weight_variable([3, 1, 16, 8])
			bconv1 = bias_variable([8], nonlinear)
			if nonlinear == 'sigmoid':
				hconv1 = tf.sigmoid(conv2d(inX, Wconv1) + bconv1)
			elif nonlinear == 'relu':
				hconv1 = tf.nn.relu(conv2d(inX, Wconv1) + bconv1)
			# sanity check
			# print Wconv1.get_shape(), bconv1.get_shape(), hconv1.get_shape()

			# second hidden layer
			Wconv2 = weight_variable([5, 1, 8, 3])
			bconv2 = bias_variable([3], nonlinear)
			if nonlinear == 'sigmoid':
				hconv2 = tf.sigmoid(conv2d(hconv1, Wconv2) + bconv2)
			elif nonlinear == 'relu':
				hconv2 = tf.nn.relu(conv2d(hconv1, Wconv2) + bconv2)
			# sanity check
			# print Wconv2.get_shape(), bconv2.get_shape(), hconv2.get_shape()

			# output layer
			if output == 'sum':
				out = tf.reshape(hconv2, [-1, 64, 3])
				Wout = weight_variable([3, 9])
				bout = bias_variable([9], 'sigmoid')
				y_conv = tf.nn.softmax(tf.matmul(tf.reduce_sum(out, 1), Wout) + bout)
			if output == 'fnn':
				out = tf.reshape(hconv2, [-1, 64*3])
				Wout = weight_variable([64*3, 9])
				bout = bias_variable([9], 'sigmoid')
				y_conv = tf.nn.softmax(tf.matmul(out, Wout) + bout)
			# sanity check
			# print out.get_shape(), Wout.get_shape(), bout.get_shape(), y_conv.get_shape()

			# loss function
			cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))

			# model training
			train_step = tf.train.AdamOptimizer(stepSize).minimize(cross_entropy)
			correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
			accuracy =  tf.reduce_mean(tf.cast(correct_prediction, "float"))
			sess.run(tf.initialize_all_variables())

			print >>sys.stderr, nonlinear, output
			for i in xrange(iteration):
				batch = trainData.nextBatch(batchSize)
				if i % 10000 == 0:
					train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1]})
					loss = cross_entropy.eval(feed_dict={x:batch[0], y_: batch[1]})
					print >>sys.stderr, "step %d, training accuracy %g, loss function %g" % (i, train_accuracy, loss)

				sess.run(train_step, feed_dict={x: batch[0], y_: batch[1]})


			print >>sys.stderr, '\n'*5
			print >>sys.stdout, nonlinear, output

			train  = trainData.nextBatch()
			print >>sys.stdout, "train accuracy %g" % accuracy.eval(feed_dict={x: train[0], y_: train[1]})
			figureFile = '/home/jiaqimu2/ECE544NA/HW2/figures/%s-%s-train.pdf' % (nonlinear, output)
			plt.figure()
			trueLabel = sess.run(tf.argmax(y_,1), feed_dict={x: train[0], y_: train[1]})
			predLabel = sess.run(tf.argmax(y_conv,1), feed_dict={x: train[0], y_: train[1]})
			plot_confusion_matrix(trueLabel, predLabel, normalize=False)
			plt.savefig(figureFile, format = 'pdf', transparent=False)

 			dev  = devData.nextBatch()
 			figureFile = '/home/jiaqimu2/ECE544NA/HW2/figures/%s-%s-dev.pdf' % (nonlinear, output)
			plt.figure()
			trueLabel = sess.run(tf.argmax(y_,1), feed_dict={x: dev[0], y_: dev[1]})
			predLabel = sess.run(tf.argmax(y_conv,1), feed_dict={x: dev[0], y_: dev[1]})
			plot_confusion_matrix(trueLabel, predLabel, normalize=False)
			plt.savefig(figureFile, format = 'pdf', transparent=False)
			print >>sys.stdout, "dev accuracy %g" % accuracy.eval(feed_dict={x: dev[0], y_: dev[1]})


			test = testData.nextBatch()
			print >>sys.stdout, "test accuracy %g" % accuracy.eval(feed_dict={x: test[0], y_: test[1]})
			figureFile = '/home/jiaqimu2/ECE544NA/HW2/figures/%s-%s-test.pdf' % (nonlinear, output)
			plt.figure()
			trueLabel = sess.run(tf.argmax(y_,1), feed_dict={x: test[0], y_: test[1]})
			predLabel = sess.run(tf.argmax(y_conv,1), feed_dict={x: test[0], y_: test[1]})
			plot_confusion_matrix(trueLabel, predLabel, normalize=False)
			plt.savefig(figureFile, format = 'pdf', transparent=False)

			sess.close()

			


	
	








