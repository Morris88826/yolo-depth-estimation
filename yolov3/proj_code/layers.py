import functools

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (BatchNormalization, Concatenate, Conv2D,
                                     Conv2DTranspose, Lambda, LeakyReLU,
                                     MaxPool2D, ZeroPadding2D)

weights_file = "../weights/yolov3-tiny.weights"

fd = open(weights_file, 'rb')
header = np.fromfile(fd, dtype = np.int32, count = 5)
header = tf.constant(header)
seen = header[3]
m_weights = np.fromfile(fd, dtype = np.float32)
fd.close()


class YoloLayer(tf.keras.layers.Layer):
    def __init__(self, num_classes, anchors, input_dims, **kwargs):
        super(YoloLayer, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.anchors = anchors
        self.input_dims = input_dims

    def call(self, prediction):
        batch_size = tf.shape(prediction)[0]

        stride = self.input_dims // tf.shape(prediction)[1]
        grid_size = self.input_dims // stride
        num_anchors = len(self.anchors)

        prediction = tf.reshape(prediction,
                                shape=(batch_size, num_anchors * grid_size * grid_size, self.num_classes + 5))

        box_xy = tf.sigmoid(prediction[:, :, :2])  # t_x (box x and y coordinates)
        objectness = tf.sigmoid(prediction[:, :, 4])  # p_o (objectness score)
        objectness = tf.expand_dims(objectness, 2)  # To make the same number of values for axis 0 and 1

        grid = tf.range(grid_size)
        a, b = tf.meshgrid(grid, grid)

        x_offset = tf.reshape(a, (-1, 1))
        y_offset = tf.reshape(b, (-1, 1))


        x_y_offset = tf.concat([x_offset, y_offset], axis=-1)
        x_y_offset = tf.reshape(tf.tile(x_y_offset, [1, num_anchors]), [1, -1, 2])
        x_y_offset = tf.cast(x_y_offset, dtype='float32')

        box_xy += x_y_offset

        # Log space transform of the height and width
        anchors = tf.cast([(a[0] / stride, a[1] / stride) for a in self.anchors], dtype='float32')
        anchors = tf.tile(anchors, (grid_size * grid_size, 1))
        anchors = tf.expand_dims(anchors, 0)

        box_wh = tf.exp(prediction[:, :, 2:4]) * anchors

        # Sigmoid class scores
        class_scores = tf.sigmoid(prediction[:, :, 5:])

        # Resize detection map back to the input image size
        stride = tf.cast(stride, dtype='float32')
        box_xy *= stride
        box_wh *= stride

        # Convert centoids to top left coordinates
        box_xy -= box_wh / 2

        return Concatenate(axis=2)([box_xy, box_wh, objectness, class_scores])

    def get_config(self):
        base_config = super(YoloLayer, self).get_config()

        config = {
            'num_classes': self.num_classes,
            'anchors': self.anchors,
            'input_dims': self.input_dims
        }

        return dict(list(base_config.items()) + list(config.items()))


def yolo_layer(x, block, layers, outputs, input_dims):
    with tf.name_scope('yolo'):

        masks = [int(m) for m in block['mask'].split(',')]
        
        # Anchors used based on mask indices
        anchors = [a for a in block['anchors'].split(',  ')]
        anchors = [anchors[i] for i in range(len(anchors)) if i in masks]
        anchors = [[int(a) for a in anchor.split(',')] for anchor in anchors]
        classes = int(block['classes'])

        x = YoloLayer(num_classes=classes, anchors=anchors, input_dims=input_dims)(x)
        outputs.append(x)
        # NOTE: Here we append None to specify that the preceding layer is a output layer
        layers.append(None)

    return x, layers, outputs

def route_layer(x, block, layers):
    select_layer_idx = [int(l) for l in block['layers'].split(',')]

    if len(select_layer_idx) == 1:
        x = layers[select_layer_idx[0]]
        layers.append(x)

    else:
        x = Concatenate()([layers[select_layer_idx[0]], layers[select_layer_idx[1]]])
        layers.append(x)

    return x, layers


def upsample_layer(x, block, layers):
    size = int(block["stride"])
    x = Lambda(lambda _x: tf.compat.v1.image.resize_bilinear(_x, (size * tf.shape(_x)[1], size * tf.shape(_x)[2])))(x)
    layers.append(x)
    return x, layers

def maxpool_layer(x, block, layers):
    kernel_size = int(block["size"])
    stride = int(block["stride"])
    x = MaxPool2D(pool_size=kernel_size, strides=stride, padding='SAME')(x)
    layers.append(x)
    return x, layers


def conv_layer(x, block, layers, cur):
    with tf.name_scope('conv'):

        kernel_size = int(block["size"])
        strides = int(block["stride"])
        pad = int(block["pad"])
        filters = int(block["filters"])
        activation = block["activation"]
        batch_norm = 'batch_normalize' in block
        padding = "VALID"

        
        if strides == 1:
            padding = "SAME"


        prev_layer_shape = tf.keras.backend.int_shape(x)
        weights_shape = (kernel_size, kernel_size, prev_layer_shape[-1], filters) # The shape for weight height x width x in_channel x out_channel
        conv_weight_shape = (filters, prev_layer_shape[-1], kernel_size, kernel_size) # The sequence the stored in weight file
        num_conv_weights = np.product(weights_shape)

        if batch_norm:
            bn_weights = m_weights[cur:cur+4*filters] # [beta, gamma, moving_mean, moving_var]
            cur += 4*filters
            bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]] 
                    
        else:
            conv_bias = m_weights[cur:cur+filters]
            cur += filters

        conv_weights = m_weights[cur:cur+num_conv_weights]
        cur += num_conv_weights

        conv_weights = conv_weights.reshape(conv_weight_shape).transpose([2, 3, 1, 0])

        if batch_norm:
            conv_weights = [conv_weights]
        else:
            conv_weights = [conv_weights, conv_bias]


        if strides > 1:
            x = ZeroPadding2D(((1, 0), (1, 0)))(x)

        x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=not batch_norm, weights=conv_weights, trainable=False)(x)

        if batch_norm:
            x = BatchNormalization(weights=bn_weights, trainable=False)(x)
            x = LeakyReLU(alpha=0.1)(x)

        layers.append(x)
    
    return x, layers, cur



def conv_transpose_layer(x, block, layers):
    kernel_size = int(block["size"])
    strides = int(block["stride"])
    pad = int(block["pad"])
    filters = int(block["filters"])
    batch_norm = 'batch_normalize' in block
    padding = "SAME"

    if pad == True:
        padding = "VALID"

   
    x = Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)(x)

    if batch_norm:
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)

    layers.append(x)
    
    return x, layers
