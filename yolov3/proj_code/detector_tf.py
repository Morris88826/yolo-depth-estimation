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

def transform_output(output, im_dim_list, resolution):
    inp_dim = resolution
    im_dim_list = np.take(im_dim_list, output[:,0].astype(np.long), axis=0)
    scaling_factor = np.reshape(np.min(inp_dim/im_dim_list,1), (-1,1))

    output[:,[1,3]] -= (inp_dim - scaling_factor*im_dim_list[:,0].reshape((-1,1)))/2
    output[:,[2,4]] -= (inp_dim - scaling_factor*im_dim_list[:,1].reshape((-1,1)))/2

    output[:,1:5] /= scaling_factor

    for i in range(output.shape[0]):
        output[i, [1,3]] = np.clip(output[i, [1,3]], 0.0, im_dim_list[i,0])
        output[i, [2,4]] = np.clip(output[i, [2,4]], 0.0, im_dim_list[i,1])
    
    return output

class Detector():
    def __init__(self, model, batch_size=10):
        self.model = model
        self.batch_size = batch_size
    
    def predict(self, im_batches, confidence=0.35, nms_thesh=0.2, print_info=True):

        init = False
        start_det_loop = time.time()
        for i, batch in enumerate(im_batches):

            start = time.time()
            output = self.model(tf.Variable(batch))
            prediction = get_prediction(output, confidence, num_classes, nms_threshold = nms_thesh)
            
            end = time.time()

            if type(prediction) == int:
                continue

            prediction[:,0] += i*self.batch_size    #transform the atribute from index in batch to index in imlist 

            if not init:                      #If we have't initialised output
                output = prediction  
                init = True
            else:
                output = np.concatenate((output,prediction))

            if print_info:
                print("Batch {}  Predicted in {:3f} seconds".format(i, end-start))
                for im_num in range(batch.shape[0]):
                    im_id = i*self.batch_size + im_num
                    objs = [self.model.classes[int(x[-1])] for x in output if int(x[0]) == im_id]
                    print("Image {} Objects Detected: {}".format(im_num, " ".join(objs)))
                print("----------------------------------------------------------")

        try:
            output
        except NameError:
            print ("No detections were made")
            return 
        return output
    
    def draw_box(self, output, loaded_ims, colors_dir, output_dir):
        # Drawing the bounding box
        colors = pkl.load(open(colors_dir, "rb"))
        list(map(lambda x: draw_box(x, loaded_ims, random.choice(colors), self.model.classes), output))

        # Same drawed images
        det_names = pd.Series(imlist).apply(lambda x: "{}/det_{}".format(output_dir,x.split("/")[-1]))
        list(map(cv2.imwrite, det_names, loaded_ims))


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
output = transform_output(output, im_dim_list, resolution)

# Print and save the draw box using output
my_detector.draw_box(output, loaded_ims, colors_dir, output_dir)

