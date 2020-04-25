from darkflow.net.build import TFNet
import cv2
import json
import os
import torch
from proj_code.utils import *
from proj_code.common import *
from proj_code.disparity import *
from proj_code.similarity_measures import *
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

options = {"model": "./darkflow/cfg/v1/yolo-tiny.cfg", "load": "./darkflow/bin/yolo-tiny.weights", "threshold": 0.1, "verbalise":False}

# tfnet = TFNet(options)
dim = (160,120)
img_left = resizeImage(cv2.imread("./test_img/left/bottle_left.jpg"), dim)
img_right = resizeImage(cv2.imread("./test_img/right/bottle_right.jpg"), dim)

# Predict 
# result_l = tfnet.return_predict(img_left)
# result_r = tfnet.return_predict(img_right)

# info = findCommonItems(result_l, result_r)

# Find depth
left_img_tensor = torch.Tensor(img_left)
right_img_tensor = torch.Tensor(img_right)
disparity_map = calculate_disparity_map(left_img_tensor, right_img_tensor, 9, sad_similarity_measure)
plot_disparity_map(left_img_tensor, right_img_tensor, disparity_map)
print(disparity_map)
# with open('result.txt', 'w') as outfile:
#     json.dump(info, outfile, indent=4)

