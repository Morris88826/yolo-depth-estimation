from __future__ import division
import time
import numpy as np
import os 
import cv2 
import os.path as osp
import pandas as pd
import random
import pickle as pkl
from tensorflow import lite
from util import *
from models import *
from preprocessing import *
from convert_model import load_and_safe_model, convert_to_tflite

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


#Set up the neural network
cfgfile_dir = "../cfg/yolov3-tiny.cfg"
weights_dir = "../weights/yolov3-tiny.weights"
class_name_dir = "../data/coco.names"
colors_dir = "../data/pallete.dms"
batch_size = 10
resolution = 416
num_classes = 80
confidence = 0.5
nms_thesh = 0.5

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

net = Yolov3_Tiny()
print("Network successfully loaded")
print(net.summary())

### Save model here
saved_model_dir = "../models/detector-yolov3-tiny"
