import tensorflow as tf
from convert_model import *
from layers import *
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# yolo = YoloV3_Tiny(416)
saved_model_dir = "../models/yolov3-tiny.h5"
# print(yolo.summary())
input = tf.zeros((1,416,416,3))
t = YoloV3_Tiny(input.shape)
print(t.summary())

save_and_convert(t, saved_model_dir) 


model_path = "../models/yolov3-tiny.h5.tflite"
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()