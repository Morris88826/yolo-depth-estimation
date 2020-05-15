import argparse

import tensorflow as tf
from darknet_tf import Darknet
from detector_tf import Detector
from tensorflow import lite

# Configurations
saved_model_dir = "../models/detector-yolov3-tiny"

cfgfile_dir = "../cfg/yolov3-tiny.cfg"
weights_dir = "../weights/yolov3-tiny.weights"
class_name_dir = "../data/coco.names"
colors_dir = "../data/pallete.dms"
batch_size = 2 # Modified
resolution = 416
num_classes = 80
confidence = 0.35
nms_thesh = 0.2

def load_and_safe_model():
    # Load the model
    darknet = Darknet(num_classes, cfgfile_dir, class_name_dir, size=resolution, weight_file=weights_dir)
    detector = Detector(darknet, batch_size=batch_size)
    print("Network successfully loaded")

    # Export the entire model
    # Ref: https://www.tensorflow.org/api_docs/python/tf/saved_model/save#example_usage
    tf.saved_model.save(detector, saved_model_dir)
    print("Model saved")

def convert_to_tflite():
    # Ref: https://www.tensorflow.org/api_docs/python/tf/lite/TFLiteConverter
    converter = lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                        tf.lite.OpsSet.SELECT_TF_OPS]
    tflite_model = converter.convert()
    open(saved_model_dir+".lite", "wb").write(tflite_model)
    print("Model converted")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("--save", help="save the tf2/keras model", action="store_true")
    parser.add_argument("--convert", help="convert the model to .tflite", action="store_true")
    args = parser.parse_args()

    if args.save:
        load_and_safe_model()
    if args.convert:
        convert_to_tflite()
