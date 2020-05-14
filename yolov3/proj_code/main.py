from __future__ import division
import time
from preprocessing import *
import numpy as np
import cv2 
from util import *
import argparse
import os 
import os.path as osp
from darknet_tf import Darknet
import pandas as pd
import random
from gen_prediction import *
import pickle as pkl
from detector_tf import *
from depth_calculation.common import *

#Set up the neural network
cfgfile_dir = "../cfg/yolov3-tiny.cfg"
weights_dir = "../weights/yolov3-tiny.weights"
class_name_dir = "../data/coco.names"
colors_dir = "../data/pallete.dms"
batch_size = 10
resolution = 416
num_classes = 80
confidence = 0.35
nms_thesh = 0.2
model = Darknet(num_classes, cfgfile_dir, class_name_dir, size=resolution, weight_file=weights_dir)
print("Network successfully loaded")



# Loading images
images_dir = "../images"
output_dir = "../results"
imlist = [ images_dir + "/" + img for img in os.listdir(images_dir) if (img[0]!=".")]
loaded_ims = [cv2.imread(x) for x in imlist]

#List containing dimensions of original images
im_dim_list = [(x.shape[1], x.shape[0]) for x in loaded_ims]
im_dim_list = np.array(im_dim_list, dtype=float)

# Create batches
im_batches = list(map(preprocess_image, loaded_ims, [resolution for x in range(len(imlist))]))
im_batches = create_batches(im_batches, batch_size)

# Create my detector
my_detector = Detector(model, batch_size=batch_size)
output = my_detector.predict(im_batches, confidence=confidence, nms_thesh=nms_thesh)

# Transform output based on input image size
# output shape = Nx6 (index 0 stores the picture id)
output = transform_output(output, im_dim_list, resolution)


# Print and save the draw box using output
my_detector.draw_box(output, loaded_ims, imlist, colors_dir, output_dir)

# find_depth(output, loaded_ims)
