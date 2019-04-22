# coding: utf-8

# # Detection with SSD
# 
# In this example, we will load a SSD model and use it to detect objects.

import cv2
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

import os
import sys
sys.path.insert(0, 'python')
import caffe
import argparse

from google.protobuf import text_format
from caffe.proto import caffe_pb2

def get_labelname(labelmap, labels):
    num_labels = len(labelmap.item)
    labelnames = []
    if type(labels) is not list:
        labels = [labels]
    for label in labels:
        found = False
        for i in xrange(0, num_labels):
            if label == labelmap.item[i].label:
                found = True
                labelnames.append(labelmap.item[i].display_name)
                break
        assert found == True

    return labelnames

def prediction(image_nm, min_conf, output):

    #为了写annos
    strs = ''
        
    image = caffe.io.load_image(image_nm)
    #plt.imshow(image)
    #plt.imshow(image)   #danpu

    # * Run the net and examine the top_k results

    transformed_image = transformer.preprocess('data', image)
    net.blobs['data'].data[...] = transformed_image

    # Forward pass.
    detections = net.forward()['detection_out']

    # Parse the outputs.
    det_label = detections[0,0,:,1]
    det_conf = detections[0,0,:,2]
    det_xmin = detections[0,0,:,3]
    det_ymin = detections[0,0,:,4]
    det_xmax = detections[0,0,:,5]
    det_ymax = detections[0,0,:,6]

    # Get detections with confidence higher than 0.6.
    top_indices = [i for i, conf in enumerate(det_conf) if conf >= min_conf]

    top_conf = det_conf[top_indices]
    top_label_indices = det_label[top_indices].tolist()
    top_labels = get_labelname(labelmap, top_label_indices)
    top_xmin = det_xmin[top_indices]
    top_ymin = det_ymin[top_indices]
    top_xmax = det_xmax[top_indices]
    top_ymax = det_ymax[top_indices]

    # 
    # * Plot the boxes

    colors = plt.cm.hsv(np.linspace(0, 1, 4)).tolist()
    #plt.imshow(image)  #danpu
    # currentAxis = plt.gca()
    fig, ax = plt.subplots(figsize=(12,12))
    ax.imshow(image, aspect='equal')

    sum_score = 0.0
    for i in xrange(top_conf.shape[0]):
        xmin = int(round(top_xmin[i] * image.shape[1]))
        ymin = int(round(top_ymin[i] * image.shape[0]))
        xmax = int(round(top_xmax[i] * image.shape[1]))
        ymax = int(round(top_ymax[i] * image.shape[0]))
        score = top_conf[i]
        sum_score += score
        label = int(top_label_indices[i])
        label_name = top_labels[i]
        #display_txt = '%s: %.2f'%(label_name, score)
        display_txt = '%.2f'%(score)
        coords = (xmin, ymin), xmax-xmin+1, ymax-ymin+1
        color = colors[label]
        
        ax.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
        #ax.text(xmin, ymin, display_txt,bbox={'facecolor':color, 'alpha':0.5})
        ax.text(xmin, ymin, display_txt,bbox={'facecolor':color, 'alpha':0.0})
        
        #annos
        strs += image_nm +'\t'+str(score)+'\t'+str(xmin)+'\t'+str(ymin)+'\t'+str(xmax)+'\t'+str(ymax)+'\n'


    img = os.path.split(image_nm)[-1]
    
    avg_conf = sum_score/top_conf.shape[0] if top_conf.shape[0]!=0 else 0.0
    print(('detected {} objects, avg_conf = {:.2f}').format(len(top_conf), avg_conf))
    plt.savefig(os.path.join(output,img))

    return avg_conf, len(top_conf), strs


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='SSD detect demo')
    #parser.add_argument('--caffe_root', dest='caffe_root', help='the root path of caffe', default='/home/lab508/caffe')
    #parser.add_argument('--class_name', dest='class_name', help='the name of the class to detect')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--min_conf', dest='min_conf', help='the min conf',
                        default=0.25, type=float)
    parser.add_argument('--image', dest='images', help='the file including all images abspath or the dir including all images', required=True)
    parser.add_argument('--caffe_model', dest='caffe_model', help='the caffe model file')
    parser.add_argument('--model_def', dest='model_def', help='the model prototxt file')
    parser.add_argument('--labelmap_file', dest='labelmap_file', help='the path of labelmap file')
    parser.add_argument('--output_dir', dest='output_dir', help='the folder for saving output-images, which will be created automatically if it not exists')

    return parser.parse_args()

if __name__ == '__main__':
    '''
    usage:
        python examples/ssd/ssd_detect_market.py --image ../tmp.txt --caffe_model models/VGGNet/cityscapes_vehilce/SSD_300x300/cityscapes_vehilce_SSD_300x300_person_iter_3000.caffemodel --model_def models/VGGNet/cityscapes_vehilce/SSD_300x300/deploy.prototxt --labelmap_file data/cityscapes_vehilce/labelmap_cityscapes_vehilce.prototxt --output_dir tmp
    '''
    args = parse_args()

    # * First, Load necessary libs and set up caffe and caffe_root
    gpu_id = args.gpu_id
    caffe.set_device(gpu_id)
    caffe.set_mode_gpu()

    # * Load LabelMap.
    # load PASCAL VOC labels
    file = open(args.labelmap_file, 'r')
    labelmap = caffe_pb2.LabelMap()
    text_format.Merge(str(file.read()), labelmap)

    # * Load the net in the test phase for inference, and configure input preprocessing.
    output = args.output_dir
    if not os.path.exists(output):
        os.makedirs(output)
    model_weights = args.caffe_model
    model_def = os.path.join(os.path.dirname(model_weights),args.model_def)

    net = caffe.Net(model_def,      # defines the structure of the model
                    model_weights,  # contains the trained weights
                    caffe.TEST)     # use test mode (e.g., don't perform dropout)

    # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_mean('data', np.array([104,117,123])) # mean pixel
    transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
    transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

    # ### 2. SSD detection
    image_resize = 512 # pinchen
    net.blobs['data'].reshape(1,3,image_resize,image_resize)

    imagePaths = []
    if os.path.isfile(args.images):
        with open(args.images) as f:
            lines = f.readlines()
            for line in lines:
                imagePaths.append(line.strip())
    elif os.path.isdir(args.images):
        files = os.listdir(args.images)
        for f in files:
            imagePaths.append(os.path.join(args.images, f))

    limit = 5
    sum_score = 0.0
    cnt_nonzero = 0
    min_conf = args.min_conf

    #annotations file
    annos_file = os.path.join(output,'annos_{}.txt'.format(min_conf))
    annfl = open(annos_file,'w')
    #test file
    test_file = os.path.join(output, 'test_{}.log'.format(min_conf))
    tfl = open(test_file,'w')


   # try:
    for i, im_path in enumerate(imagePaths):
		
		annStr = ''
		testStr = ''
		
		im_path = im_path.strip()
		print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
		print 'Predict for {}-{}'.format(i, im_path)
		
		score, detNums, tmps = prediction(im_path, args.min_conf, output)

        #将test写入文件
		testStr += 'Predict for {}-{}'.format(i, im_path)+'\n'
		testStr += 'detected {} objects, avg_conf = {:.2f}\n'.format(detNums, score)
        #将tmp写入annos文件
		annStr += tmps
		
		annfl.write(annStr) 
		tfl.write(testStr) 
		
		sum_score += score
		if score != 0.0: cnt_nonzero += 1
            #if i>limit:
            #    break
   # except: #KeyboardInterrupt:
   #     annfl.write(annStr) 
   #     tfl.write(testStr) 
   #     exit()
    
    print '\n\tmAp = {:.2f} without zero-AP-images'.format(sum_score / cnt_nonzero)
    print '\tmAp = {:.2f} with zero-AP-images'.format(sum_score / i)
    print '\tcurrent min_conf is {}'.format(args.min_conf)

    testStr += '\n\tmAp = {:.2f} without zero-AP-images'.format(sum_score / cnt_nonzero)+'\n\tmAp = {:.2f} with zero-AP-images'.format(sum_score / i)+'\n\tcurrent min_conf is {}'.format(args.min_conf)

    #annfl.write(annStr) 
    #tfl.write(testStr) 
