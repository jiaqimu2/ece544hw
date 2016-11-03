import numpy as np
import matplotlib 
matplotlib.use('Agg')
from plot import plot_confusion_matrix
import matplotlib.pyplot as plt

MIN_FRAME = 70

def readFile(directory, fileName):

	f = open(directory + fileName, 'r')
	labels = list()
	samples = list()
	for raw in f.readlines():
		label, sampleFile = raw.rstrip().split()

		# remove corrupted data
		data = open(directory + sampleFile).read()
		if 'inf' in data.lower() or 'nan' in data.lower():
			continue

		sample = np.loadtxt(directory + sampleFile)

		if len(sample) < MIN_FRAME:
			continue

		samples.append(sample[0:MIN_FRAME].flatten())
		labels.append(int(label))

	labels = np.array(labels)
	samples = np.array(samples)

	np.random.seed(2333)
	idx = np.random.permutation(len(labels))
	labels = labels[idx]
	samples = samples[idx]
	
	return labels, samples


class Data:

	def __init__(self, directory, fileName):

		self.labels, self.samples = readFile(directory, fileName)
		self.sampleNum = len(self.labels)
		self.batchInd = 0

	def evaluate(self, predLabels, figFile = None):
		
		if figFile:
			plt.figure()
			plot_confusion_matrix(self.labels, predLabels, normalize=False)
			plt.savefig(figFile, format = 'pdf', transparent = True)
			plt.close()

		return sum(self.labels == predLabels) / float(len(predLabels))

	def nextBatch(self, n=-1):

		if n == -1:
			return self.samples, self.labels

		if self.batchInd + n < self.sampleNum:
			oldInd = self.batchInd
			self.batchInd += n
			return self. samples[oldInd:self.batchInd], self.labels[oldInd:self.batchInd]
		else:
			oldInd = self.batchInd
			self.batchInd = (self.batchInd+n) % self.sampleNum
			return self.samples[range(oldInd, self.sampleNum) + range(self.batchInd)], self.labels[range(oldInd, self.sampleNum) + range(self.batchInd)]