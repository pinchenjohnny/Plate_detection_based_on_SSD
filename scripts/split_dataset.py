#########################################################################
# File Name: split_dataset.py
# Author: james
# mail: zxiaoci@mail.ustc.edu
#########################################################################
#!/usr/bin/python
import os
from os import system
import random
import sys

if len(sys.argv) != 1:
	print('python {}'.format(sys.argv[0]))
	exit(0)

images = os.listdir('Annotations/')
random.shuffle(images)

n_test = int(len(images) * 0.2)
n_trainval = len(images) - n_test
n_train = int(n_trainval * 0.8)
n_val = n_trainval - n_train

print('n_trainval:{}, n_test:{}, n_train:{}, n_val:{}'.format(n_trainval, n_test, n_train, n_val))

basename = 'ImageSets/Main/'
ftrainval = open(basename+'trainval.txt', 'w')
ftrain = open(basename+'train.txt', 'w')
fval = open(basename+'val.txt', 'w')
ftest = open(basename+'test.txt', 'w')

for i, im in enumerate(images):
	im = im.split('.')[0] + '\n'
	if i < n_trainval:
		ftrainval.write(im)
		if i < n_train:
			ftrain.write(im)
		else:
			fval.write(im)
	else:
		ftest.write(im)

print('Done!')

	

