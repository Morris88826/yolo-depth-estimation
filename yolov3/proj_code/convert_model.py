import tensorflow as tf
from tensorflow import lite
from layers import *

def save_model(model, model_path):
    tf.keras.models.save_model(model, model_path)
    print("Model saved")

def convert_to_tflite(model_path, tflite_path, input_shape={'input_1': [1, 416, 416, 3]}):
    keras_model = tf.keras.models.load_model(model_path, custom_objects={"YoloLayer":YoloLayer})
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    # converter = tf.compat.v1.lite.TocoConverter.from_keras_model_file(model_path, input_shapes=input_shape)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.compat.v1.lite.constants.FLOAT16]
    tflite_model = converter.convert()
    open(tflite_path, "wb").write(tflite_model)
    print("TFLite Model converted")

def save_and_convert(model, model_path="../models/yolov3-tiny.h5", tflite_path="../models/yolov3-tiny.tflite", input_shape={'input_1': [1, 416, 416, 3]}):
    save_model(model, model_path)
    convert_to_tflite(model_path, tflite_path, input_shape=input_shape)