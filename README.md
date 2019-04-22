本文采用Liu等人在2016年提出的[SSD](https://arxiv.org/pdf/1512.02325.pdf)目标检测方法, 解决了私有数据集上道路高处俯拍车辆图像的车牌定位问题.

请引用SSD:

```
@inproceedings{liu2016ssd,
  title = {{SSD}: Single Shot MultiBox Detector},
  author = {Liu, Wei and Anguelov, Dragomir and Erhan, Dumitru and Szegedy, Christian and Reed, Scott and Fu, Cheng-Yang and Berg, Alexander C.},
  booktitle = {ECCV},
  year = {2016}
}
```

---

本文实验环境为Linux CentOS 6.7, python 2.7, cuda 7.5.

**方法手册**如下:

- [Tips](#tips)
- [准备](#%E5%87%86%E5%A4%87)
- [数据预处理](#%E6%95%B0%E6%8D%AE%E9%A2%84%E5%A4%84%E7%90%86)
- [训练](#%E8%AE%AD%E7%BB%83)
- [测试](#%E6%B5%8B%E8%AF%95)

## Tips

1. 下面提到的脚本无法运行时, 请首先检查是否正确指定脚本内的路径参数. 本文实验中修改的参数都打了`#pinchen`标签, 在脚本中搜索, 辅助修改参数.

2. 运行脚本报no module caffe错时, 终端输入
   ```
   export PYTHONPATH=$PYTHONPATH:./python
   ```

   将./python临时添加到PYTHONPATH.
 
 3. 下面默认当前目录为`Plate_detection_based_on_SSD/`

## 准备

1. clone this repo

   ```
   git clone https://github.com/pinchenjohnny/Plate_detection_based_on_SSD.git
   ```

2. build Caffe
   - 先根据[Caffe instruction](http://caffe.berkeleyvision.org/installation.html), 确保已安装所有依赖包.
   - 修改`./`下`Makefile.config.example`, 本文修改如下:
     -  CUDA_DIR :=/usr/local/cuda_75
     -  注释掉CUDA_ARCH的-gencode arch=computer_61, code=sm_61
   -  ```
      make -j8
      # Make sure to include $CAFFE_ROOT/python to your PYTHONPATH.
      make py
      make test -j8
      # (Optional)
      make runtest -j8
      ```

3. 下载基础网络

   原始SSD基础网络为[VGG16](https://gist.github.com/weiliu89/2ed6e13bfd5b57cf81d6), 下载后存储在`./models/VGGNet/`下.

## 数据预处理

1. 仿照VOC数据集建立目录, 组织自己的数据集

   在`./data/your_dataset_name`下, 建立三个文件夹:
   - Annotations: 存放标注文件*.xml
   - ImageSets/Main: 存放下面产生的train.txt, val.txt, trainval.txt, test.txt
   - JPEGImage: 存放原图

2. 划分训练/验证/测试集

   运行`./scripts/`下的`split_dataset.py`, 先随机取20%的图片作测试集test, 剩余80%称为trainval, 然后从trainval中随机取80%作训练集train, 剩余20%为验证集val.

3. 由数据集生成Caffe SSD所需LMDB文件

    - 修改`./data/your_dataset_name`下的`labelmap_voc.prototxt`, 该文件为检测目标的标签文件, 记录需要训练识别的 $c$ 类目标的信息.
    - 运行`./scripts/`下的`create_list.sh`, 生成三类文件列表:
        - test_name_size.txt: 测试集图像大小, 就像:
  
          381_098 1536 2048

          381_098就是图片名称, 1536*2048就是图片尺寸(高,宽)

        - test.txt: 测试集图像-标签一一对应, 就像:
  
           our_dataset_name/JPEGImages/381_098.jpg our_dataset_name/Annotations/381_098.xml

        - trainval.txt: 训练集图像-标签一一对应, 就像:

          our_dataset_name/JPEGImages/346_015.jpg our_dataset_name/Annotations/346_015.xml
      - 运行`./scripts/`下的`create_data.sh`, 生成LMDB文件.

## 训练

在`./models/`下建立`your_dataset_name/`, 存放训练过程中产生的模型.

在`./jobs/`下建立`your_dataset_name/`, 存放训练过程中产生的日志文件.

修改`./scripts/`下`ssd_vgg_commom.py`参数后运行. 修改内容集中在__main__函数中, 如:
- project_name: 改为当前...
- cf.gpus 
- batch_size
- cf.train_data
- cf.test_data
- cf.resize_width,cf.resize_height
- cf.max_iter
- cf.pretrain_model: 指定预训练模型
- cf.isScore: False表示训练, True表示测试

## 测试

在`./output/`下建立`your_dataset_name/`, 存放测试过程中产生的结果和日志文件.

这里改用`./scripts/`下的`ssd_detect_common.py`, 测试并可视化结果.
