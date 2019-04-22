cd /home/pinchen/ssd/caffe/data/PlateDetection
/home/pinchen/ssd/caffe/build/tools/caffe train \
--solver="/home/pinchen/ssd/caffe/models/VGGNet/PlateDetection/SSD_512x512/solver.prototxt" \
--weights="/home/pinchen/ssd/caffe/models/VGGNet/VGG_ILSVRC_16_layers_fc_reduced.caffemodel" \
--gpu 0 2>&1 | tee /home/pinchen/ssd/caffe/jobs/VGGNet/PlateDetection/SSD_512x512/PlateDetection_SSD_512x512.log
