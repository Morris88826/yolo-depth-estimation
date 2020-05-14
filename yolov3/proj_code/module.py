import torch 
import torch.nn as nn
import numpy as np
import tensorflow as tf


class EmptyLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(EmptyLayer, self).__init__()

class BoundingBoxLayer(tf.keras.layers.Layer):
    def __init__(self, anchors):
        super(BoundingBoxLayer, self).__init__()
        self.anchors = anchors


def create_modules(blocks, data_format = "channels_last"):
    net_info = blocks[0]
    module_list = []
    in_channels = 3
    out_channels_all = [] # since route filter will route my previous filters

    for index, x in enumerate(blocks[1:]):
        type = x["type"]
        module = tf.keras.Sequential()

        if type == "convolutional":

            # Some convolutional layer won't have batch normalization
            try:
                require_normalize = int(x["batch_normalize"])
                bias = False
            except:
                require_normalize = False
                bias = True
            
            out_channels = int(x['filters'])
            kernel_size = int(x["size"])
            stride = int(x["stride"])
            require_padding = int(x["pad"])

            if require_padding:
                conv = tf.keras.layers.Conv2D(out_channels, kernel_size, stride, padding='SAME', use_bias=bias, data_format=data_format, name="Conv2d")
            else:
                conv = tf.keras.layers.Conv2D(out_channels, kernel_size, stride, padding='VALID', use_bias=bias, data_format=data_format, name="Conv2d")

            module.add(conv)

            if require_normalize:
                batch_norm = tf.keras.layers.BatchNormalization(axis=-1, name="BatchNorm2d")
                module.add(batch_norm)
            
            activation = x["activation"]

            if activation == "leaky":
                leaky = tf.keras.layers.LeakyReLU(0.1, name="LeakyReLU")
                module.add(leaky)
            
        elif type == "shortcut":
            shortcut = EmptyLayer()
            module.add(shortcut)
            out_channels = out_channels_all[index-1]

        elif type == "yolo":
            mask = x["mask"].split(",")
            mask = [int(x) for x in mask]

            anchors = x["anchors"].split(",")
            anchors = [int(i) for i in anchors]
            num_anchor_sets = len(anchors)//2
            anchors = [(anchors[i], anchors[i+1])for i in range(num_anchor_sets)]
            anchors = [anchors[idx] for idx in mask]
            boundingbox = BoundingBoxLayer(anchors)
            module.add(boundingbox)

        elif type == "route":
            layers = x["layers"].split(',')
            start = int(layers[0])
            if len(layers) == 1:
                end = 0
            else:
                end = int(layers[1])
            
            route = EmptyLayer()
            module.add(route)

            # calculating output channel
            # Two notations (negative, positive) or (negative, negative)
            if end > 0:
                end = end - index
            if end < 0:
                out_channels = out_channels_all[index+start] + out_channels_all[index+end]
            else:
                out_channels = out_channels_all[index+start]

        elif type == "upsample":
            size = int(x["stride"])
            upsample = tf.keras.layers.UpSampling2D(size, data_format=data_format, interpolation="bilinear", name="UpSampling2D")
            module.add(upsample)
        
        elif type == "maxpool":
            size = int(x["size"])
            stride = int(x["stride"])
            if size == 2 and stride == 1:
                maxpool = tf.keras.layers.MaxPool2D(size, stride, padding="SAME", data_format=data_format, name="MaxPool2D")
            else:
                maxpool = tf.keras.layers.MaxPool2D(size, stride, padding="VALID", data_format=data_format, name="MaxPool2D")
            module.add(maxpool)
        
        
        module_list.append(module)
        in_channels = out_channels
        out_channels_all.append(out_channels)

    return (net_info, module_list)


