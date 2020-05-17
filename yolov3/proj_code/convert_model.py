import argparse
import tensorflow as tf
from tensorflow import lite

# Configurations

def save_and_convert(model, saved_model_dir):
    # Export the entire model
    # Ref: https://www.tensorflow.org/api_docs/python/tf/saved_model/save#example_usage
    tf.keras.models.save_model(model, saved_model_dir)
    converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file(saved_model_dir)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                              tf.lite.OpsSet.SELECT_TF_OPS]
    tflite_model = converter.convert()
    open(saved_model_dir+".tflite", "wb").write(tflite_model)
    print("Model converted")

def convert_model(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    return tflite_model


def convert_to_tflite(saved_model_dir):
    # Ref: https://www.tensorflow.org/api_docs/python/tf/lite/TFLiteConverter
    converter = lite.TFLiteConverter.from_saved_model(saved_model_dir)
    # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
    #                                     tf.lite.OpsSet.SELECT_TF_OPS]
    tflite_model = converter.convert()
    open(saved_model_dir+".tflite", "wb").write(tflite_model)
    print("Model converted")

def keras_converter(model, saved_model_dir):
    # Convert the model.
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                        # tf.lite.OpsSet.SELECT_TF_OPS]
    tflite_model = converter.convert()
    open(saved_model_dir+".tflite", "wb").write(tflite_model)

