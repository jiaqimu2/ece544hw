from gmm import GMM
import numpy as np
import os
from data import readData, saveData

IN_PATH = '/home/jiaqimu2/ece544hw/hw3-gmm/data/'
OUT_PATH = '/home/jiaqimu2/ece544hw/hw3-gmm/figures/'

def part2(nclusters, inName, stat = False):

	data, shape = readData(IN_PATH + inName)

	########################################
	# fit and predict data
	gmm = GMM(nclusters)
	gmm.fit(data)
	cind = gmm.predict(data)
	cents = gmm.clusters()
	covars = gmm.covariances()
	weights = gmm.getWeights()

	########################################
	# from centers to image
	if stat and nclusters == 3:
		print '\\begin{align*}'
	predData = np.ones(data.shape)
	for i in xrange(len(cents)):
		predData[np.where(cind == i)[0]] *= cents[i]
		# print statistics
		if stat and nclusters == 3:
			print 'w_{%d} &= %0.2f \\\\' % (i+1, weights[i])
			print '\\vec{\\mu}_{%d} &= \\begin{bmatrix} %s \\end{bmatrix} \\\\' % (i+1, ' '.join(['%0.2f & ' % val for val in cents[i]]))
			print '\\Sigma_{%d} &= \\begin{bmatrix} %s \\end{bmatrix} \\\\' % (i+1, ' \\\\'.join([' '.join(['%0.2f & ' % val for val in covars[i][j]]) for j in xrange(3)]))

	if stat and nclusters == 3:
		print '\\end{align*}'
	saveData(predData, shape, OUT_PATH + '.'.join(inName.split('.')[:-1]) + '-part2-%d.png' % nclusters)


if __name__ == '__main__':

	for fname in os.listdir(IN_PATH):
		if fname.startswith('.'):
			continue
		for n in [3, 5, 10]:
			print 'working on %s, m = %d' % (fname, n)
			part2(n, fname, True)

		