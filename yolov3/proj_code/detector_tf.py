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

# For eager execution. See https://github.com/tensorflow/tensorflow/issues/34944
# This is to state whether it should be executed eagerly (default=True) explicitly
# but this does not solve the problem 
tf.config.experimental_run_functions_eagerly(True)

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
    
    new_output = np.zeros((output.shape[0], 6))
    new_output[:, :5] = output[:, :5]
    new_output[:, 5] = output[:, -1]

    return new_output.astype(int)

"""
    Only tf.Module can be converted
"""
class Detector(tf.Module):
    def __init__(self, model, batch_size=10):
        self.model = model
        self.batch_size = batch_size
    
    def __call__(self, im_batches, confidence=0.35, nms_thesh=0.2, print_info=True):
        """
            In runtime, call() will be called by tflite Interpreter for iOS
        """

        init = False
        start_det_loop = time.time()
        for i, batch in enumerate(im_batches):

            start = time.time()
            prediction = self.model(tf.Variable(batch))
            prediction = get_prediction(prediction, confidence, self.model.num_classes, nms_threshold = nms_thesh)
            
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
    
    def draw_box(self, output, loaded_ims, imlist, colors_dir, output_dir):
        # Drawing the bounding box
        colors = pkl.load(open(colors_dir, "rb"))
        list(map(lambda x: draw_box(x, loaded_ims, random.choice(colors), self.model.classes), output))

        # Same drawed images
        det_names = pd.Series(imlist).apply(lambda x: "{}/det_{}".format(output_dir,x.split("/")[-1]))
        list(map(cv2.imwrite, det_names, loaded_ims))


