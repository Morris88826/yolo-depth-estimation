import tensorflow as tf
from tensorflow.keras import Input, Model
from yolo import Yolov3_Tiny
from convert_model import *
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

        
inputs = Input(shape=(416, 416, 3))
outputs = Yolov3_Tiny(inputs)
model = Model(inputs, outputs)
print(model.summary())

output = model(tf.zeros((1,416,416,3)))

# Print output shape (Should be (N x ((13x13)+(15x15))*3 x (C+85)) )
print(output.shape)



# Save and convert model
save_and_convert(model)

# Load tflite
model_path = "../models/yolov3-tiny.tflite"
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
print("Load successfully")
