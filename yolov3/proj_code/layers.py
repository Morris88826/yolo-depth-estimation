import torch 
import torch.nn as nn
import numpy as np
import tensorflow as tf
from util import predict_bounding_box


class EmptyLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(EmptyLayer, self).__init__()

class BoundingBoxLayer(tf.keras.layers.Layer):
    def __init__(self, anchors):
        super(BoundingBoxLayer, self).__init__()
        self.anchors = anchors

class ConvolutionBlock(tf.keras.layers.Layer):
    '''
    ConvolutionBlock.weights = if batch_norm, [kernel, gamma, beta, moving_mean, moving_variance]
                               else, [kernel, bias]
    '''
    def __init__(self, block_info):
        super(ConvolutionBlock, self).__init__()

        use_bias = True
        try:
            self.require_batchnorm = int(block_info["batch_normalize"])
            if self.require_batchnorm:
                use_bias = False
        except:
            self.require_batchnorm = False

        self.filters = int(block_info["filters"])
        self.kernel_size = int(block_info["size"])
        self.stride = int(block_info["stride"])
        self.padding = "SAME" if int(block_info["pad"]) else "VALID"

        self.conv = tf.keras.layers.Conv2D(self.filters, self.kernel_size, self.stride, padding=self.padding, use_bias=use_bias, name="Conv2d")
        self.batchnorm = None

        if self.require_batchnorm:
            self.batchnorm = tf.keras.layers.BatchNormalization(axis=-1, name="BatchNorm2d")

        self.activation = tf.keras.layers.LeakyReLU(0.1, name="LeakyReLU")
    
    def call(self, inputs):
        self.in_shape = inputs.shape[-1]
        x = self.conv(inputs)
        
        if self.batchnorm is not None:
            x = self.batchnorm(x)
        
        x = self.activation(x)
        return x

class UpSamplingBlock(tf.keras.layers.Layer):
    def __init__(self, block_info):
        super(UpSamplingBlock, self).__init__()
        self.stride = int(block_info["stride"])
        self.upsampling = tf.keras.layers.UpSampling2D(self.stride, interpolation="bilinear", name="UpSampling2D")
    
    def call(self, inputs):
        return self.upsampling(inputs)
    
class MaxPoolBlock(tf.keras.layers.Layer):
    def __init__(self, block_info):
        super(MaxPoolBlock, self).__init__()
        self.kernel_size = int(block_info["size"])
        self.stride = int(block_info["stride"])
        if self.kernel_size == 2 and self.stride == 1:
            self.maxpool = tf.keras.layers.MaxPool2D(self.kernel_size, self.stride, padding="SAME", name="MaxPool2D")
        else:
            self.maxpool = tf.keras.layers.MaxPool2D(self.kernel_size, self.stride, padding="VALID", name="MaxPool2D")
        
    def call(self, inputs):
        return self.maxpool(inputs)

class BoundingBoxBlock(tf.keras.layers.Layer):
    def __init__(self, net_info, block_info):
        super(BoundingBoxBlock, self).__init__()
        mask = block_info["mask"].split(",")
        mask = [int(x) for x in mask]

        anchors = block_info["anchors"].split(",")
        anchors = [int(i) for i in anchors]
        num_anchor_sets = len(anchors)//2
        anchors = [(anchors[i], anchors[i+1])for i in range(0, len(anchors), 2)]
        self.input_dim = int(net_info["height"])
        self.num_classes = int(block_info["classes"])
        self.anchors = [anchors[idx] for idx in mask]
        
    def call(self, inputs):
        return predict_bounding_box(inputs, self.input_dim, self.anchors, self.num_classes)

def create_layers(blocks):
    net_info = blocks[0]
    blocks_info = blocks[1:]
    layers = []

    for index, block_info in enumerate(blocks_info):
        type = block_info["type"]

        if type == "convolutional":
            layer = ConvolutionBlock(block_info)

        elif type == "shortcut":
            layer = EmptyLayer()

        elif type == "yolo":
            layer = BoundingBoxBlock(net_info, block_info)

        elif type == "route":
            route = EmptyLayer()

        elif type == "upsample":
            layer = UpSamplingBlock(block_info)

        elif type == "maxpool":
            layer = MaxPoolBlock(block_info)

        layers.append(layer)

    return (net_info, layers)
