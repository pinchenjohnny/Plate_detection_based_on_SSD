#!/usr/bin/env python
# -*- coding: UTF-8 -*-

##########################################################################
#	> Author: Tingjian Lau
#	> Mail: tjliu@mail.ustc.edu.cn
#	> Created Time: 2017/10/31
#	> Detail: 
#        1. 根据参数自动生产[train/test/solver/deploy].prototxt
#        2. 是否从上次快照出继续训练
#        3. 生成log文件
#        4. 执行./build/tools/caffe train
#########################################################################

from __future__ import print_function
import math
import os
import shutil
import stat
import subprocess
import sys

import caffe
from caffe.model_libs import *
from google.protobuf import text_format

class emptyClass():
    def __init__(self):
        pass

cf = emptyClass()

# Add extra layers on top of a "base" network (e.g. VGGNet or Inception).
def AddExtraLayers(net, use_batchnorm=True, lr_mult=1):
    use_relu = True

    # Add additional convolutional layers.
    # 19 x 19
    from_layer = net.keys()[-1]

    # TODO(weiliu89): Construct the name using the last layer to avoid duplication.
    # 10 x 10
    out_layer = "conv6_1"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 256, 1, 0, 1,
        lr_mult=lr_mult)

    from_layer = out_layer
    out_layer = "conv6_2"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 512, 3, 1, 2,
        lr_mult=lr_mult)

    # 5 x 5
    from_layer = out_layer
    out_layer = "conv7_1"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 128, 1, 0, 1,
      lr_mult=lr_mult)

    from_layer = out_layer
    out_layer = "conv7_2"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 256, 3, 1, 2,
      lr_mult=lr_mult)

    # 3 x 3
    from_layer = out_layer
    out_layer = "conv8_1"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 128, 1, 0, 1,
      lr_mult=lr_mult)

    from_layer = out_layer
    out_layer = "conv8_2"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 256, 3, 0, 1,
      lr_mult=lr_mult)

    # 1 x 1
    from_layer = out_layer
    out_layer = "conv9_1"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 128, 1, 0, 1,
      lr_mult=lr_mult)

    from_layer = out_layer
    out_layer = "conv9_2"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 256, 3, 0, 1,
      lr_mult=lr_mult)

    return net

def createSolver(cf):
    # Create solver.
    solver = caffe_pb2.SolverParameter(
            train_net=cf.train_net_file,
            test_net=[cf.test_net_file],
            snapshot_prefix=cf.snapshot_prefix,
            **cf.solver_param)

    with open(cf.solver_file, 'w') as f:
        print(solver, file=f)
    shutil.copy(cf.solver_file, cf.job_dir)

def createTrainNet(cf):
    # Create train net.
    net = caffe.NetSpec()
    # 创建data层，主要目标是加载数据且实现数据扩充
    net.data, net.label = CreateAnnotatedDataLayer(cf.train_data, batch_size=cf.batch_size_per_device, train=True, output_label=True, label_map_file=cf.label_map_file, transform_param=cf.train_transform_param, batch_sampler=cf.batch_sampler)

    VGGNetBody(net, from_layer='data', fully_conv=True, reduced=True, dilated=True, dropout=False)

    AddExtraLayers(net, cf.use_batchnorm, lr_mult=cf.lr_mult)

    mbox_layers = CreateMultiBoxHead(net, data_layer='data', from_layers=cf.mbox_source_layers,
        use_batchnorm=cf.use_batchnorm, min_sizes=cf.min_sizes, max_sizes=cf.max_sizes,
        aspect_ratios=cf.aspect_ratios, steps=cf.steps, normalizations=cf.normalizations,
        num_classes=cf.num_classes, share_location=cf.share_location, flip=cf.flip, clip=cf.clip,
        prior_variance=cf.prior_variance, kernel_size=3, pad=1, lr_mult=cf.lr_mult)

    # Create the MultiBoxLossLayer.
    name = "mbox_loss"
    mbox_layers.append(net.label)
    net[name] = L.MultiBoxLoss(*mbox_layers, multibox_loss_param=cf.multibox_loss_param,
            loss_param=cf.loss_param, include=dict(phase=caffe_pb2.Phase.Value('TRAIN')),
            propagate_down=[True, True, False, False])

    with open(cf.train_net_file, 'w') as f:
        print('name: "{}_train"'.format(cf.model_name), file=f)
        print(net.to_proto(), file=f)
    shutil.copy(cf.train_net_file, cf.job_dir)

def createTestNet(cf):
    # Create test net.
    net = caffe.NetSpec()
    net.data, net.label = CreateAnnotatedDataLayer(cf.test_data, batch_size=cf.test_batch_size,
            train=False, output_label=True, label_map_file=cf.label_map_file,
            transform_param=cf.test_transform_param)

    VGGNetBody(net, from_layer='data', fully_conv=True, reduced=True, dilated=True, dropout=False)

    AddExtraLayers(net, cf.use_batchnorm, lr_mult=cf.lr_mult)

    mbox_layers = CreateMultiBoxHead(net, data_layer='data', from_layers=cf.mbox_source_layers,
            use_batchnorm=cf.use_batchnorm, min_sizes=cf.min_sizes, max_sizes=cf.max_sizes,
            aspect_ratios=cf.aspect_ratios, steps=cf.steps, normalizations=cf.normalizations,
            num_classes=cf.num_classes, share_location=cf.share_location, flip=cf.flip, clip=cf.clip,
            prior_variance=cf.prior_variance, kernel_size=3, pad=1, lr_mult=cf.lr_mult)

    conf_name = "mbox_conf"
    if cf.multibox_loss_param["conf_loss_type"] == P.MultiBoxLoss.SOFTMAX:
        reshape_name = "{}_reshape".format(conf_name)
        net[reshape_name] = L.Reshape(net[conf_name], shape=dict(dim=[0, -1, cf.num_classes]))
        softmax_name = "{}_softmax".format(conf_name)
        net[softmax_name] = L.Softmax(net[reshape_name], axis=2)
        flatten_name = "{}_flatten".format(conf_name)
        net[flatten_name] = L.Flatten(net[softmax_name], axis=1)
        mbox_layers[1] = net[flatten_name]
    elif cf.multibox_loss_param["conf_loss_type"] == P.MultiBoxLoss.LOGISTIC:
        sigmoid_name = "{}_sigmoid".format(conf_name)
        net[sigmoid_name] = L.Sigmoid(net[conf_name])
        mbox_layers[1] = net[sigmoid_name]

    net.detection_out = L.DetectionOutput(*mbox_layers,
        detection_output_param=cf.det_out_param,
        include=dict(phase=caffe_pb2.Phase.Value('TEST')))
    net.detection_eval = L.DetectionEvaluate(net.detection_out, net.label,
        detection_evaluate_param=cf.det_eval_param,
        include=dict(phase=caffe_pb2.Phase.Value('TEST')))

    with open(cf.test_net_file, 'w') as f:
        print('name: "{}_test"'.format(cf.model_name), file=f)
        print(net.to_proto(), file=f)
    shutil.copy(cf.test_net_file, cf.job_dir)

    return net 

def createDeployNet(cf,deploy_net):
    # Remove the first and last layer from test net.
    with open(cf.deploy_net_file, 'w') as f:
        net_param = deploy_net.to_proto()
        # Remove the first (AnnotatedData) and last (DetectionEvaluate) layer from test net.
        del net_param.layer[0]
        del net_param.layer[-1]
        net_param.name = '{}_deploy'.format(cf.model_name)
        net_param.input.extend(['data'])
        net_param.input_shape.extend([
            caffe_pb2.BlobShape(dim=[1, 3, cf.resize_height, cf.resize_width])])
        print(net_param, file=f)
    shutil.copy(cf.deploy_net_file, cf.job_dir)

def createScoreRunsh(cf):
    # Create job file.
    with open(cf.job_file, 'w') as f:
      f.write('cd {}\n'.format(cf.caffe_root))
      f.write('/home/pinchen/ssd/caffe/build/tools/caffe train \\\n') #pinchen
      f.write('--solver="{}" \\\n'.format(cf.solver_file))
      f.write('--weights="{}" \\\n'.format(cf.pretrain_model))
      if cf.solver_param['solver_mode'] == P.Solver.GPU:
        f.write('--gpu {} 2>&1 | tee {}/{}_test_{}.log\n'.format(cf.gpus, cf.job_dir, cf.model_name, cf.score_max_iter))
      else:
        f.write('2>&1 | tee {}/{}.log\n'.format(cf.job_dir, cf.model_name))

    # Copy the python script to job_dir.
    py_file = os.path.abspath(__file__)
    shutil.copy(py_file, cf.job_dir)


def createTrainRunSh(cf):
    max_iter = 0
    # Find most recent snapshot.
    for file in os.listdir(cf.snapshot_dir):
      if file.endswith(".solverstate"):
        basename = os.path.splitext(file)[0]
        iter = int(basename.split("{}_iter_".format(cf.model_name))[1])
        if iter > max_iter:
          max_iter = iter

    train_src_param = '--weights="{}" \\\n'.format(cf.pretrain_model)
    if cf.resume_training:
      if max_iter > 0:
        train_src_param = '--snapshot="{}_iter_{}.solverstate" \\\n'.format(cf.snapshot_prefix, max_iter)

    if cf.remove_old_models:
      # Remove any snapshots smaller than max_iter.
      for file in os.listdir(cf.snapshot_dir):
        if file.endswith(".solverstate"):
          basename = os.path.splitext(file)[0]
          iter = int(basename.split("{}_iter_".format(cf.model_name))[1])
          if max_iter > iter:
            os.remove("{}/{}".format(cf.snapshot_dir, file))
        if file.endswith(".caffemodel"):
          basename = os.path.splitext(file)[0]
          iter = int(basename.split("{}_iter_".format(cf.model_name))[1])
          if max_iter > iter:
            os.remove("{}/{}".format(cf.snapshot_dir, file))

    # Create job file.
    with open(cf.job_file, 'w') as f:
      f.write('cd {}\n'.format(cf.caffe_root))
      f.write('/home/pinchen/ssd/caffe/build/tools/caffe train \\\n') # pinchen
      f.write('--solver="{}" \\\n'.format(cf.solver_file))
      f.write(train_src_param)
      if cf.solver_param['solver_mode'] == P.Solver.GPU:
        f.write('--gpu {} 2>&1 | tee {}/{}.log\n'.format(cf.gpus, cf.job_dir, cf.model_name))
      else:
        f.write('2>&1 | tee {}/{}.log\n'.format(cf.job_dir, cf.model_name))

    # Copy the python script to job_dir.
    py_file = os.path.abspath(__file__)
    shutil.copy(py_file, cf.job_dir)

def getNumClasses(label_map_file):
    '''
    从label_map_file获取类别个数
    '''
    with open(label_map_file) as f:
        lines = f.readlines()

        lines = lines[::-1]

        for line in lines:
            if line.find('label') != -1:
                res = int(line.strip().split(':')[-1].strip())
                break

    return res+1

def findRecentSnapshot(cf, spec_max_iter=None):
    # Find most recent snapshot.
    max_iter = 0
    for file in os.listdir(cf.snapshot_dir):
        if file.endswith(".caffemodel"):
            basename = os.path.splitext(file)[0]
            iter = int(basename.split("{}_iter_".format(cf.model_name))[1])
            if iter > max_iter:
                max_iter = iter

    if spec_max_iter != None:
        max_iter = spec_max_iter
    if max_iter == 0:
        print("Cannot find snapshot in {}".format(cf.snapshot_dir))
        sys.exit()
    pretrain_model = "{}_iter_{}.caffemodel".format(cf.snapshot_prefix, max_iter)

    return max_iter, pretrain_model



if __name__ == '__main__':
    cf.caffe_root = os.getcwd()
    project_name = 'PlateDetection' # pinchen
    cf.gpus = "0"	# pinchen
    cf.batch_size = 8
    cf.accum_batch_size = 8
    cf.test_batch_size = 8 
    #cf.isScore = True  #test
    cf.isScore = False  #train
    cf.train_data = '/home/pinchen/ssd/caffe/data/PlateDetection/lmdb/PlateDetection_trainval_lmdb' # pinchen
    cf.test_data = '/home/pinchen/ssd/caffe/data/PlateDetection/lmdb/PlateDetection_test_lmdb' # pinchen
    cf.resize_width, cf.resize_height = 512, 512
    cf.max_iter = 3000
    cf.stepsize = 1000
    cf.test_interval = cf.max_iter+1
    cf.resume_training = True
    cf.run_soon = True
    cf.remove_old_models = False
    cf.pretrain_model = "/home/pinchen/ssd/caffe/models/VGGNet/VGG_ILSVRC_16_layers_fc_reduced.caffemodel" # pinchen

    gpulist = cf.gpus.split(",")
    num_gpus = len(gpulist)

    cf.job_name = "SSD_{}x{}".format(cf.resize_width, cf.resize_height)
    cf.snapshot_dir = "/home/pinchen/ssd/caffe/models/VGGNet/{}/{}".format(project_name, cf.job_name) # pinchen
    cf.model_name = "{}_{}".format(project_name, cf.job_name)
    cf.snapshot_prefix = "{}/{}".format(cf.snapshot_dir, cf.model_name)
    if cf.isScore:
        cf.save_dir = "/home/pinchen/ssd/caffe/models/VGGNet/{}/{}_score".format(project_name, cf.job_name) # pinchen
        cf.job_dir = "/home/pinchen/ssd/caffe/jobs/VGGNet/{}/{}_score".format(project_name, cf.job_name) # pinchen
        cf.output_result_dir = "/home/pinchen/ssd/caffe/output/VGGNet/{}/{}_score".format(project_name, cf.job_name) # pinchen
        if len(sys.argv) == 3:
            spec_max_iter, cf.gpus = int(sys.argv[1]), int(sys.argv[2])
        else:
            spec_max_iter = None
        cf.score_max_iter, cf.pretrain_model = findRecentSnapshot(cf, spec_max_iter)
        #cf.gpus = gpulist[-1]
        cf.batch_size, cf.accum_batch_size = 1, 1
        cf.snapshot_iter, cf.max_iter = 0, 0
        cf.snapshot_after_train, cf.test_initialization = False, True 
    else:
        cf.save_dir = "/home/pinchen/ssd/caffe/models/VGGNet/{}/{}".format(project_name, cf.job_name) # pinchen
        cf.job_dir = "/home/pinchen/ssd/caffe/jobs/VGGNet/{}/{}".format(project_name, cf.job_name) # pinchen
        cf.output_result_dir = "/home/pinchen/ssd/caffe/output/VGGNet/{}/{}".format(project_name, cf.job_name) # pinchen
        cf.snapshot_iter = 100 # 1000  # pinchen
        cf.snapshot_after_train, cf.test_initialization = True, False
    cf.train_net_file = "{}/train.prototxt".format(cf.save_dir)
    cf.test_net_file = "{}/test.prototxt".format(cf.save_dir)
    cf.solver_file = "{}/solver.prototxt".format(cf.save_dir)
    cf.deploy_net_file = "{}/deploy.prototxt".format(cf.save_dir)
    cf.label_map_file = "/home/pinchen/ssd/caffe/data/{}/labelmap.prototxt".format(project_name) # pinchen
    cf.test_image_file = '/home/pinchen/ssd/caffe/data/{}/test.txt'.format(project_name) # pinchen
    cf.job_file = "{}/{}.sh".format(cf.job_dir, cf.model_name)

    make_if_not_exist(cf.save_dir)
    make_if_not_exist(cf.job_dir)
    make_if_not_exist(cf.snapshot_dir)

    cf.use_batchnorm = False
    cf.normalization_mode = P.Loss.VALID
    cf.lr_mult = 1
    # Use different initial learning rate.
    if cf.use_batchnorm:
        cf.base_lr = 0.0004
    else:
        # A learning rate for batch_size = 1, num_gpus = 1.
        cf.base_lr = 0.000004

    cf.iter_size = cf.accum_batch_size / cf.batch_size
    cf.solver_mode = P.Solver.CPU
    cf.device_id = 0
    cf.batch_size_per_device = cf.batch_size

    if num_gpus > 0:
        cf.batch_size_per_device = int(math.ceil(float(cf.batch_size) / num_gpus))
        cf.iter_size = int(math.ceil(float(cf.accum_batch_size) / (cf.batch_size_per_device * num_gpus)))
        cf.solver_mode = P.Solver.GPU
        cf.device_id = int(gpulist[0])

    neg_pos_ratio = 3.
    loc_weight = (neg_pos_ratio + 1.) / 4.
    if cf.normalization_mode == P.Loss.NONE:
        cf.base_lr /= cf.batch_size_per_device
    elif cf.normalization_mode == P.Loss.VALID:
        cf.base_lr *= 25. / loc_weight
    elif normalization_mode == P.Loss.FULL:
        # Roughly there are 2000 prior bboxes per image.
        # TODO(weiliu89): Estimate the exact # of priors.
        cf.base_lr *= 2000.

    cf.num_test_image = len(open(cf.test_image_file).readlines())
    cf.test_iter = int(math.ceil(float(cf.num_test_image) / cf.test_batch_size))
    cf.solver_param = {
        # Train parameters
        'base_lr': cf.base_lr,
        'weight_decay': 0.0005,
        #'lr_policy': "multistep",
        #'stepvalue': [1000, 2000, 3000],
        'lr_policy': "step",
        'stepsize': cf.stepsize,
        'gamma': 0.1,
        'momentum': 0.9,
        'iter_size': cf.iter_size,
        'max_iter': cf.max_iter,
        'snapshot': cf.snapshot_iter,
        'display': 10,
        'average_loss': 10,
        'type': "Adam",
        'solver_mode': cf.solver_mode,
        'device_id': cf.device_id,
        'debug_info': False,
        'snapshot_after_train': cf.snapshot_after_train,
        # Test parameters
        'test_iter': [cf.test_iter],
        'test_interval': cf.test_interval,
        'eval_type': "detection",
        'ap_version': "11point",
        'test_initialization': cf.test_initialization,
        }

    cf.train_transform_param = {
        'mirror': True,
        'mean_value': [104, 117, 123],
        'resize_param': {
                'prob': 1,
                'resize_mode': P.Resize.WARP,
                'height': cf.resize_height,
                'width': cf.resize_width,
                'interp_mode': [
                        P.Resize.LINEAR,
                        P.Resize.AREA,
                        P.Resize.NEAREST,
                        P.Resize.CUBIC,
                        P.Resize.LANCZOS4,
                        ],
                },
        'distort_param': { # 关照/对比度...
                'brightness_prob': 0.5,
                'brightness_delta': 32,
                'contrast_prob': 0.5,
                'contrast_lower': 0.5,
                'contrast_upper': 1.5,
                'hue_prob': 0.5,
                'hue_delta': 18,
                'saturation_prob': 0.5,
                'saturation_lower': 0.5,
                'saturation_upper': 1.5,
                'random_order_prob': 0.0,
                },
        'expand_param': { # 将DistortImage的图片用像素0进行扩展
                'prob': 0.5,
                'max_expand_ratio': 4.0,
                },
        'emit_constraint': {
            'emit_type': caffe_pb2.EmitConstraint.CENTER,
            }
        }
    
    cf.batch_sampler = [
        {
                'sampler': {
                        },
                'max_trials': 1,
                'max_sample': 1,
        },
        {
                'sampler': {
                        'min_scale': 0.3,
                        'max_scale': 1.0,
                        'min_aspect_ratio': 0.5,
                        'max_aspect_ratio': 2.0,
                        },
                'sample_constraint': {
                        'min_jaccard_overlap': 0.1,
                        },
                'max_trials': 50,
                'max_sample': 1,
        },
        {
                'sampler': {
                        'min_scale': 0.3,
                        'max_scale': 1.0,
                        'min_aspect_ratio': 0.5,
                        'max_aspect_ratio': 2.0,
                        },
                'sample_constraint': {
                        'min_jaccard_overlap': 0.3,
                        },
                'max_trials': 50,
                'max_sample': 1,
        },
        {
                'sampler': {
                        'min_scale': 0.3,
                        'max_scale': 1.0,
                        'min_aspect_ratio': 0.5,
                        'max_aspect_ratio': 2.0,
                        },
                'sample_constraint': {
                        'min_jaccard_overlap': 0.5,
                        },
                'max_trials': 50,
                'max_sample': 1,
        },
        {
                'sampler': {
                        'min_scale': 0.3,
                        'max_scale': 1.0,
                        'min_aspect_ratio': 0.5,
                        'max_aspect_ratio': 2.0,
                        },
                'sample_constraint': {
                        'min_jaccard_overlap': 0.7,
                        },
                'max_trials': 50,
                'max_sample': 1,
        },
        {
                'sampler': {
                        'min_scale': 0.3,
                        'max_scale': 1.0,
                        'min_aspect_ratio': 0.5,
                        'max_aspect_ratio': 2.0,
                        },
                'sample_constraint': {
                        'min_jaccard_overlap': 0.9,
                        },
                'max_trials': 50,
                'max_sample': 1,
        },
        {
                'sampler': {
                        'min_scale': 0.3,
                        'max_scale': 1.0,
                        'min_aspect_ratio': 0.5,
                        'max_aspect_ratio': 2.0,
                        },
                'sample_constraint': {
                        'max_jaccard_overlap': 1.0,
                        },
                'max_trials': 50,
                'max_sample': 1,
        },
        ]

    createSolver(cf)

    # MultiBoxLoss parameters.
    cf.num_classes = getNumClasses(cf.label_map_file)
    cf.share_location = True
    background_label_id=0
    train_on_diff_gt = True
    normalization_mode = P.Loss.VALID
    code_type = P.PriorBox.CENTER_SIZE
    ignore_cross_boundary_bbox = False
    mining_type = P.MultiBoxLoss.MAX_NEGATIVE
    neg_pos_ratio = 3.
    loc_weight = (neg_pos_ratio + 1.) / 4.
    cf.multibox_loss_param = {
        'loc_loss_type': P.MultiBoxLoss.SMOOTH_L1,
        'conf_loss_type': P.MultiBoxLoss.SOFTMAX,
        'loc_weight': loc_weight,
        'num_classes': cf.num_classes,
        'share_location': cf.share_location,
        'match_type': P.MultiBoxLoss.PER_PREDICTION,
        'overlap_threshold': 0.5,
        'use_prior_for_matching': True,
        'background_label_id': background_label_id,
        'use_difficult_gt': train_on_diff_gt,
        'mining_type': mining_type,
        'neg_pos_ratio': neg_pos_ratio,
        'neg_overlap': 0.5,
        'code_type': code_type,
        'ignore_cross_boundary_bbox': ignore_cross_boundary_bbox,
        }
    cf.loss_param = {
        'normalization': normalization_mode,
        }


    # parameters for generating priors.
    # minimum dimension of input image
    min_dim = 300
    # conv4_3 ==> 38 x 38
    # fc7 ==> 19 x 19
    # conv6_2 ==> 10 x 10
    # conv7_2 ==> 5 x 5
    # conv8_2 ==> 3 x 3
    # conv9_2 ==> 1 x 1
    cf.mbox_source_layers = ['conv4_3', 'fc7', 'conv6_2', 'conv7_2', 'conv8_2', 'conv9_2']
    # in percent %
    min_ratio = 20
    max_ratio = 90
    step = int(math.floor((max_ratio - min_ratio) / (len(cf.mbox_source_layers) - 2)))
    cf.min_sizes = []
    cf.max_sizes = []
    for ratio in xrange(min_ratio, max_ratio + 1, step):
      cf.min_sizes.append(min_dim * ratio / 100.)
      cf.max_sizes.append(min_dim * (ratio + step) / 100.)
    # cf.min_sizes: [30.0, 60.0, 111.0, 162.0, 213.0, 264.0]
    # cf.max_sizes: [60.0, 111.0, 162.0, 213.0, 264.0, 315.0]
    cf.min_sizes = [min_dim * 10 / 100.] + cf.min_sizes
    cf.max_sizes = [min_dim * 20 / 100.] + cf.max_sizes
    cf.steps = [8, 16, 32, 64, 100, 300]
    cf.aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
    # L2 normalize conv4_3.
    cf.normalizations = [20, -1, -1, -1, -1, -1]
    # variance used to encode/decode prior bboxes.
    if code_type == P.PriorBox.CENTER_SIZE:
      cf.prior_variance = [0.1, 0.1, 0.2, 0.2]
    else:
      cf.prior_variance = [0.1]
    cf.flip = True
    cf.clip = False

    createTrainNet(cf)

    cf.test_transform_param = {
        'mean_value': [104, 117, 123],
        'resize_param': {
                'prob': 1,
                'resize_mode': P.Resize.WARP,
                'height': cf.resize_height,
                'width': cf.resize_width,
                'interp_mode': [P.Resize.LINEAR],
                },
        }
    # parameters for generating detection output.
    name_size_file = "/home/pinchen/ssd/caffe/data/{}/test_name_size.txt".format(project_name)
    cf.det_out_param = {
        'num_classes': cf.num_classes,
        'share_location': cf.share_location,
        'background_label_id': background_label_id,
        'nms_param': {'nms_threshold': 0.45, 'top_k': 400},
        'save_output_param': {
            'output_directory': cf.output_result_dir,
            'output_name_prefix': "comp4_det_test_",
            'output_format': "VOC",
            'label_map_file': cf.label_map_file,
            'name_size_file': name_size_file,
            'num_test_image': cf.num_test_image,
            },
        'keep_top_k': 200,
        'confidence_threshold': 0.01,
        'code_type': code_type,
        }
    # parameters for evaluating detection results.
    cf.det_eval_param = {
        'num_classes': cf.num_classes,
        'background_label_id': background_label_id,
        'overlap_threshold': 0.5,
        'evaluate_difficult_gt': False,
        'name_size_file': name_size_file,
        }

    net = createTestNet(cf)

    createDeployNet(cf, net)

    # Run the job.
    if not cf.isScore:
        createTrainRunSh(cf)
    else:
        createScoreRunsh(cf)
    os.chmod(cf.job_file, stat.S_IRWXU)
    if cf.run_soon:
        subprocess.call(cf.job_file, shell=True)
