import os
import time

import tensorflow as tf
from tensorflow.keras import Input, Model

from convert_model import *
from dataloader import create_batches, load_images, load_batch, get_data_size
from trainer import Trainer
from yolo import Yolov3_Tiny
from dataloader import *

def rmse(target_y, predicted_y):
    return tf.sqrt(tf.reduce_mean(tf.square(target_y - predicted_y)))

def mse(target_y, predicted_y):
    return tf.reduce_mean(tf.square(target_y - predicted_y))

cfg_file = "../cfg/yolov3-depth-tiny-v2.cfg"

if cfg_file == "../cfg/yolov3-depth-tiny.cfg":
  version = "v1"
elif cfg_file == "../cfg/yolov3-depth-tiny-v2.cfg":
  version = "v2"

inputs = Input(shape=(416, 416, 3))
outputs, depth = Yolov3_Tiny(inputs, cfg_file)
model = Model(inputs, (outputs, depth))

ckpt_dir = "../ckpt_5_1e-5_{}/cp_0".format(version)
model.load_weights(ckpt_dir)

testing_pkl_dir = "../data/nyu_pickle/testing"
# testing_data_dir = "../nyu_test.csv"
# pickle_dump(testing_data_dir, filename=testing_pkl_dir)

batch_size = 32
num_batches = get_data_size(testing_pkl_dir) // batch_size 

losses = []
rmse_losses = []
for idx in range(num_batches):
    images, gts = load_batch(testing_pkl_dir, idx*batch_size, batch_size)

    # normalize images and ground truth
    images = images/255.0
    gts = tf.convert_to_tensor(gts)
    gts = 1000 / tf.clip_by_value(gts * 1000, 10, 1000)

    start = time.time()
    _, output_img = model.predict(images)
    loss = mse(output_img, gts)
    rmseloss = rmse(output_img, gts)
    
    losses.append(loss)
    rmse_losses.append(rmseloss)
    
print("MSE: {}".format(np.mean(losses)))
print("RMSE: {}".format(np.mean(rmse_losses)))