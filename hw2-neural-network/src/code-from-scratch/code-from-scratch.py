from fnn import FNN
import numpy as np
from data import Data
from time import time
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt

directory = '/home/jiaqimu2/ece544hw/hw2-neural-network/data/'
figure = '/home/jiaqimu2/ece544hw/hw2-neural-network/figures/'
train = 'train/lab/hw2train_labels.txt'
test = 'eval/lab/hw2eval_labels.txt'
dev = 'dev/lab/hw2dev_labels.txt'


if __name__ == '__main__':

	trainData = Data(directory, train)
	devData = Data(directory, dev)
	testData = Data(directory, test)

	stepSize = 1e-2
	iterations = 100000
	batchSize = 50
	nfeatures = 70*16
	nclasses = 9

	# compute accuracy
	for hidden in [10, 50]:
		for nonlinear in ['relu', 'sigmoid', 'tanh']:
			hiddenSize = [hidden] * 2
			fnn = FNN(nfeatures, hiddenSize, nclasses, stepSize, batchSize, iterations, nonlinear)
			fnn.train(trainData)

			print hidden, nonlinear
			print 'train:', trainData.evaluate(fnn.test(trainData.samples),'%s%s-%d-train-conf-matrix.pdf' % (figure, nonlinear, hidden))
			print 'dev:', devData.evaluate(fnn.test(devData.samples),'%s%s-%d-dev-conf-matrix.pdf' % (figure, nonlinear, hidden))
			print 'test:', testData.evaluate(fnn.test(testData.samples),'%s%s-%d-test-conf-matrix.pdf' % (figure, nonlinear, hidden))

	# compute running time
	iterations /= 100
	batchList = range(20, 220, 20)
	for nonlinear in ['relu', 'sigmoid', 'tanh']:
		figName = '%s%s-running-time.pdf' % (figure, nonlinear)
		plt.figure()
		for hidden in [10, 50]:
			hiddenSize = [hidden] * 2
			runningTime = list()
			for batchSize in batchList:
				fnn = FNN(nfeatures, hiddenSize, nclasses, stepSize, batchSize, iterations, nonlinear)
				start = time()
				fnn.train(trainData)
				end = time()
				runningTime.append((end - start) / iterations)
			plt.plot(batchList, runningTime, linewidth=3.0, label='hidden layer size = %d' % hidden, marker = 'o')
		plt.xlabel('batch size')
		plt.ylabel('running time')
		plt.legend()
		plt.savefig(figName, format = 'pdf', transparent = True)








