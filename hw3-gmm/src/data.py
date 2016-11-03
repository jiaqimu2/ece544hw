from PIL import Image
import numpy as np
import matplotlib 
import os
matplotlib.use('Agg')
from matplotlib import pyplot as plt


IN_PATH = '/home/jiaqimu2/ECE544NA/HW3/data/'
OUT_PATH = '/home/jiaqimu2/ECE544NA/HW3/figures/'

def readData(dataFile):

	im = Image.open(dataFile)
	height, width = im.size
	pix = im.load()
	data = np.array([pix[x, y][0:3] for x in xrange(height) for y in xrange(width)], dtype = np.float32)

	return data, (height, width)

def saveData(data, shape, dataFile):

	os.system('mkdir -p %s' % (OUT_PATH))
	
	height, width = shape
	imData = np.zeros((width, height, 3))
	for x in xrange(height):
		for y in xrange(width):
			imData[y, x] = data[x * width + y]
	imData = imData.astype(np.uint8)
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.axes.get_xaxis().set_visible(False)
	ax.axes.get_yaxis().set_visible(False)
	plt.imshow(imData)
	plt.savefig(dataFile, bbox_inches='tight', pad_inches=0)