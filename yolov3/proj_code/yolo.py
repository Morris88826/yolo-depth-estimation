import tensorflow as tf
import numpy as np
from layers import conv_layer, maxpool_layer, upsample_layer, route_layer, yolo_layer
from util import parse_cfg

cfg_file = "../cfg/yolov3-tiny.cfg"
blocks = parse_cfg(cfg_file)

def Yolov3_Tiny(inputs):
    x, layers, outputs = inputs, [], []
    weights_ptr = 0
    config = {}
    input_dims = 416
    
    for block in blocks:
        block_type = block["type"]
        if block_type == "convolutional":
            x, layers, weights_ptr = conv_layer(x, block, layers, weights_ptr)

        elif block_type == "maxpool":
            x, layers = maxpool_layer(x, block, layers)
        
        elif block_type == "upsample":
            x, layers = upsample_layer(x, block, layers)

        elif block_type == "route":
            x, layers = route_layer(x, block, layers)

        elif block_type == "yolo":
            x, layers, outputs = yolo_layer(x, block, layers, outputs, input_dims)

    # output_layers = [layers[i - 1] for i in range(len(layers)) if layers[i] is None]

    
    outputs = tf.keras.layers.Concatenate(axis=1)(outputs)

    # Run NMS
    # outputs = non_maximum_suppression(outputs, confidence=0.5, num_classes=80, nms_threshold=0.5)

    return outputs
