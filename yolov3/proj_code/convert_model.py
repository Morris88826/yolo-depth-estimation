import argparse
import tensorflow as tf
from tensorflow import lite

# Configurations

def load_and_safe_model(model, saved_model_dir):
    # Export the entire model
    # Ref: https://www.tensorflow.org/api_docs/python/tf/saved_model/save#example_usage
    tf.saved_model.save(model, saved_model_dir)
    print("Model saved")

def convert_to_tflite(saved_model_dir):
    # Ref: https://www.tensorflow.org/api_docs/python/tf/lite/TFLiteConverter
    converter = lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                        tf.lite.OpsSet.SELECT_TF_OPS]
    tflite_model = converter.convert()
    open(saved_model_dir+".tflite", "wb").write(tflite_model)
    print("Model converted")

def keras_converter(model, saved_model_dir):
    # Convert the model.
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                        tf.lite.OpsSet.SELECT_TF_OPS]
    tflite_model = converter.convert()
    open(saved_model_dir+".tflite", "wb").write(tflite_model)

