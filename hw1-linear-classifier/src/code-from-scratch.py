import numpy as np
from linear import Linear
from logistic import Logistic
from perceptron import Perceptron
from svm import SVM
from sklearn.decomposition import PCA

import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt

directory = '/home/jiaqimu2/ece544hw/hw1-linear-classifier/data/'
figures = '/home/jiaqimu2/ece544hw/hw1-linear-classifier/figures/'
train = 'train/lab/hw1train_labels.txt'
test = 'eval/lab/hw1eval_labels.txt'
dev = 'dev/lab/hw1dev_labels.txt'
stepSize = 2e-6
iterNum = 3000

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

		labels.append(float(label))
		samples.append([float(v) for v in data.split()])

	labels = np.array(labels)
	samples = np.array(samples)

	return labels, samples

def evaluate(testLabel, trueLabel):

	if len(testLabel) != len(trueLabel):
		print 'Test mismatch!'
		return 0

	return float(sum(testLabel == trueLabel)) / len(testLabel)


def part1():

	linear = Linear(stepSize, iterNum)
	plt.figure()
	plt.plot(linear.train(trainSamples, trainLabels), linewidth=3, label = 'linear')
	plt.legend()
	plt.xlabel('iterations')
	plt.ylabel('error rate')
	plt.savefig(figures + 'part1-linear.pdf', format='pdf')
	print 'Linear:', 1 - evaluate(testLabels, linear.test(testSamples))
	

	logistic = Logistic(stepSize, iterNum)
	plt.figure()
	plt.plot(logistic.train(trainSamples, trainLabels), linewidth=3, label = 'logistic')
	plt.legend()
	plt.xlabel('iterations')
	plt.ylabel('error rate')
	plt.savefig(figures + 'part1-logistic.pdf', format='pdf')
	print 'Logistic:', 1 - evaluate(testLabels, logistic.test(testSamples))
	

	perceptron = Perceptron(stepSize, iterNum)
	plt.figure()
	plt.plot(perceptron.train(trainSamples, trainLabels), linewidth=3, label = 'perceptron')
	plt.legend()
	plt.xlabel('iterations')
	plt.ylabel('error rate')
	plt.savefig(figures + 'part1-perceptron.pdf', format='pdf')
	print 'Perceptron:', 1 - evaluate(testLabels, perceptron.test(testSamples))
	
	c = 10
	svm = SVM(stepSize, iterNum, c)
	plt.figure()
	plt.plot(svm.train(trainSamples, trainLabels), linewidth=3, label = 'svm')
	plt.legend()
	plt.xlabel('iterations')
	plt.ylabel('error rate')
	plt.savefig(figures + 'part1-svm.pdf', format='pdf')
	print 'SVM-%3.3f:' % c, 1 - evaluate(testLabels, svm.test(testSamples))

def part2():
	
	trainErr = list()
	devErr = list()
	cs = list()
	for c in np.arange(5)-2:
		c = 10**c
		cs.append(c)
		svm = SVM(stepSize, iterNum, c)
		svm.train(trainSamples, trainLabels)
		trainErr.append(1 - evaluate(trainLabels, svm.test(trainSamples)))
		devErr.append(1 - evaluate(devLabels, svm.test(devSamples)))

	plt.figure()
	plt.semilogx(cs, trainErr, 'bo-', linewidth=3, label='train')
	plt.semilogy(cs, devErr, 'rs-', linewidth=3, label='dev')
	plt.legend()
	plt.xlabel('c')
	plt.ylabel('error rate')
	plt.savefig(figures + 'part2.pdf', format='pdf')


def part4():
	

	idx = np.random.choice(len(trainSamples), 300, replace=False)
	train = trainSamples[idx]
	label = trainLabels[idx]
	
	pca = PCA(n_components=2)
	train = pca.fit_transform(train)
	mean = pca.mean_
	basis = pca.components_

	plt.figure()
	ax = plt.subplot(111)
	idx = np.where(label == 0)[0]
	ax.scatter(train[idx, 0], train[idx, 1], color='k', marker='x', label = 'eh-set')
	idx = np.where(label == 1)[0]
	plt.scatter(train[idx, 0], train[idx, 1], color='k', marker='o', label = 'ee-set')

	linear = Linear(stepSize, 100)
	linear.train(trainSamples, trainLabels)
	a = linear.coeff[0:-1]
	c = linear.coeff[-1]
	c = np.dot(a, mean)
	a, b = np.dot(basis, a)
	x = np.arange(-1000, 1000)/float(100)
	y = (0.5 - (c + a*x))/b
	plt.plot(x, y, linewidth=3, label = 'linear')

	logistic = Logistic(stepSize, iterNum)
	logistic.train(trainSamples, trainLabels)
	a = logistic.coeff[0:-1]
	c = logistic.coeff[-1]
	c = np.dot(a, mean)
	a, b = np.dot(basis, a)
	x = np.arange(-1000, 1000)/float(100)
	y = - (c + a*x)/b
	plt.plot(x, y, linewidth=3, label = 'logistic')

	perceptron = Perceptron(stepSize, iterNum)
	perceptron.train(trainSamples, trainLabels)
	a = perceptron.coeff[0:-1]
	c = perceptron.coeff[-1]
	c = np.dot(a, mean)
	a, b = np.dot(basis, a)
	x = np.arange(-1000, 1000)/float(100)
	y = - (c + a*x)/b
	plt.plot(x, y, linewidth=3, label = 'perceptron')

	svm = SVM(stepSize, iterNum, 1)
	svm.train(trainSamples, trainLabels)
	a = svm.coeff[0:-1]
	c = svm.coeff[-1]
	c = np.dot(a, mean)
	a, b = np.dot(basis, a)
	x = np.arange(-1000, 1000)/float(100)
	y = - (c + a*x)/b
	plt.plot(x, y, linewidth=3, label = 'svm')


	plt.legend()
	ax.set_xlim(-10, 10)
	ax.set_ylim(-5, 5)
	plt.savefig(figures + 'part4.pdf', format='pdf')
	


if __name__ == '__main__':

	trainLabels, trainSamples = readFile(train)
	testLabels, testSamples = readFile(test)
	devLabels, devSamples = readFile(dev)
	
	print trainLabels.shape, trainSamples.shape
	print testLabels.shape, testSamples.shape
	print devLabels.shape, devSamples.shape

	# part1()
	# part2()
	part4()
