import tensorflow as tf
import numpy as np
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt

directory = '/home/jiaqimu2/ece544hw/hw1-linear-classifier/data/'
figures = '/home/jiaqimu2/ece544hw/hw1-linear-classifier/figures/'
train = 'train/lab/hw1train_labels.txt'
test = 'eval/lab/hw1eval_labels.txt'
dev = 'dev/lab/hw1dev_labels.txt'

np.random.seed(23333)

def readFile(fileName):

	f = open(directory + fileName, 'r')
	labels = list()
	samples = list()
	for raw in f.readlines():
		label, sampleFile = raw.rstrip().split()
		fsample = open(directory + sampleFile, 'r')
		data = fsample.read().rstrip()

		if 'inf' in data.lower():
			continue
		if 'nan' in data.lower():
			continue

		samples.append([float(v) for v in data.split()])
		labels.append([1-float(label), float(label)])

	labels = np.array(labels)
	samples = np.array(samples)

	return labels, samples

if __name__ == '__main__':

	trainLabels, trainSamples = readFile(train)
	testLabels, testSamples = readFile(test)
	devLabels, devSamples = readFile(dev)
	_, dim = trainSamples.shape
	nclass = 2

	print len(trainLabels)
	print len(testLabels)
	print len(devLabels)
	
	# initialization
	x = tf.placeholder(tf.float32, [None, dim])
	W = tf.Variable(tf.zeros([dim, nclass]))
	b = tf.Variable(tf.zeros([nclass]))

	# model
	y = tf.nn.softmax(tf.matmul(x, W) + b)
	y_ = tf.placeholder(tf.float32, [None, nclass])
	cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
	train_step = tf.train.GradientDescentOptimizer(0.2).minimize(cross_entropy)

	correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	# session starts
	init = tf.initialize_all_variables()
	sess = tf.Session()
	sess.run(init)

	iterNum = np.arange(3000)
	errs = list()
	for i in iterNum:
		sess.run(train_step, feed_dict={x: trainSamples, y_: trainLabels})
		errs.append( 1- sess.run(accuracy, feed_dict={x: trainSamples, y_: trainLabels}))

	plt.figure()
	plt.plot(iterNum, errs, linewidth = 3, label='train')
	plt.xlabel('iteration')
	plt.ylabel('error rate')
	plt.savefig(figures + 'tensorflow.pdf', format = 'pdf')

	print 'train:', 1 - sess.run(accuracy, feed_dict={x: trainSamples, y_: trainLabels})
	print 'test:', 1 - sess.run(accuracy, feed_dict={x: testSamples, y_:testLabels})
