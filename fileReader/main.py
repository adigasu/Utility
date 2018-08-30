from __future__ import print_function

" Read data and display files (.jpg, .mat, .csv)"

import os
import sys
import scipy as sp
import scipy.misc
import glob
import h5py as hp
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import datetime

fileType = '*.jpg'

def load_data(dataPath, is_zeroMean=False, is_randBatch=False, batch_size=1):
	dataFiles = glob.glob(os.path.join(dataPath, fileType))
	dataFiles.sort()

	if is_randBatch:
		batch_images = np.random.choice(dataFiles, size=batch_size)
	else:
		batch_images = dataFiles

	imgsX=[]
	for img_path in batch_images:
		img_x = imread(img_path)
		# print('Data size: ', imgsX.shape)
		# print('Data size: ', img_x.shape)
		imgsX.append(img_x)

	if is_zeroMean:
		imgsX = np.array(imgsX)/127.5 - 1.
	else:
		imgsX = np.array(imgsX)/255.0

	print('Data size: ', imgsX.shape)
	return imgsX

def imread(path, is_grayscale=False):

	## Using scipy    
    img = sp.misc.imread(path).astype(np.float)

    ## Using PIL Image
    # img = Image.open(path)
    # img = np.asarray(img)
    # img = img.astype('float32')

    if is_grayscale:
        img = np.expand_dims(img, -1)

    return img

def imshow(X):
	if len(X.shape)==3:
		plt.imshow((X-np.min(X))/(np.max(X)-np.min(X)));
		# plt.show(block=False)
	else:
		plt.imshow((X-np.min(X))/(np.max(X)-np.min(X)),cmap = plt.get_cmap('gray'))
		# plt.show(block=False)
	plt.show()


if __name__ == '__main__':
    dataPath = sys.argv[1]
    #dataPath = './test'

    ## for reading multiple files in folder
    trainX = load_data(dataPath)
    imshow(trainX[0])

    ## for single file
    # temp = imread(dataPath + '/5.jpg')
    # imshow(temp)    


    