import numpy as np
import tensorflow as tf
import time
from util import bbox_prediction_v2, bbox_prediction


yolo_tiny_anchors = np.array([(10, 14), (23, 27), (37, 58),
                              (81, 82), (135, 169),  (344, 319)],
                             np.float32) / 416

yolo_tiny_anchor_masks = np.array([[3,4,5],[0,1,2]])

def Conv2D_Block(x, filters, kernel_size, strides=1, batch_norm=True):
    padding = "VALID"
    if strides == 1:
        padding = "SAME"
    
    x = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
               use_bias=not batch_norm)(x)

    if batch_norm:
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    return x

def Convolution(batch_size=None, name=None):
    x = inputs = tf.keras.layers.Input([416, 416, 3], batch_size=batch_size)
    x = Conv2D_Block(x, 16, 3)
    x = tf.keras.layers.MaxPool2D(2, 2, 'same')(x)
    x = Conv2D_Block(x, 32, 3)
    x = tf.keras.layers.MaxPool2D(2, 2, 'same')(x)
    x = Conv2D_Block(x, 64, 3)
    x = tf.keras.layers.MaxPool2D(2, 2, 'same')(x)
    x = Conv2D_Block(x, 128, 3)
    x = tf.keras.layers.MaxPool2D(2, 2, 'same')(x)
    x = x_8 = Conv2D_Block(x, 256, 3)  # skip connection
    x = tf.keras.layers.MaxPool2D(2, 2, 'same')(x)
    x = Conv2D_Block(x, 512, 3)
    x = tf.keras.layers.MaxPool2D(2, 1, 'same')(x)
    x = Conv2D_Block(x, 1024, 3)
    return tf.keras.Model(inputs, (x_8, x), name=name)


def YoloConv(filters, batch_size = None, name=None):
    def yolo_conv(x_in):
        if isinstance(x_in, tuple):
            print(batch_size)
            inputs = tf.keras.layers.Input(x_in[0].shape[1:], batch_size=batch_size), tf.keras.layers.Input(x_in[1].shape[1:], batch_size=batch_size)
            x, x_skip = inputs
            # concat with skip connection
            x = Conv2D_Block(x, filters, 1)
            x = tf.keras.layers.UpSampling2D(2)(x)
            x = tf.keras.layers.Concatenate()([x, x_skip])
            print(x.shape)
        else:
            x = inputs = tf.keras.layers.Input(x_in.shape[1:], batch_size=batch_size)
            x = Conv2D_Block(x, filters, 1)

        return tf.keras.Model(inputs, x, name=name)(x_in)
    return yolo_conv

def YoloOutput(filters, anchors, classes, batch_size=None, name=None):
    def yolo_output(x_in):
        x = inputs = tf.keras.layers.Input(x_in.shape[1:], batch_size=batch_size)
        x = Conv2D_Block(x, filters * 2, 3)
        x = Conv2D_Block(x, anchors * (classes + 5), 1, batch_norm=False)
        # x = tf.keras.layers.Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[1], tf.shape(x)[2],
        #                                     anchors, classes + 5)))(x)
        return tf.keras.Model(inputs, x, name=name)(x_in)
    return yolo_output

def YoloV3_Tiny(input_shape=None, channels=3, anchors=yolo_tiny_anchors,
               masks=yolo_tiny_anchor_masks, classes=80, score_threshold = 0.5, iou_threshold = 0.5, resolution=416):
    
    batch_size, size, _, channels = input_shape
    # batch_size = None
    x = inputs = tf.keras.layers.Input([size, size, channels], name='input', batch_size=batch_size)

    x_8, x = Convolution(batch_size=batch_size, name='convolution')(x)

    x = YoloConv(256, batch_size=batch_size, name='yolo_conv_0')(x)
    output_0 = YoloOutput(256, len(masks[0]), classes, batch_size=batch_size, name='yolo_output_0')(x)

    x = YoloConv(128, batch_size=batch_size, name='yolo_conv_1')((x, x_8))
    output_1 = YoloOutput(128, len(masks[1]), classes, batch_size=batch_size, name='yolo_output_1')(x)


    boxes_0 = tf.keras.layers.Lambda(lambda x: bbox_prediction(x, np.array([(81, 82), (135, 169),  (344, 319)]), classes, resolution),
                     name='yolo_boxes_0')(output_0)
    boxes_1 = tf.keras.layers.Lambda(lambda x: bbox_prediction(x, np.array([(10, 14), (23, 27), (37, 58)]), classes, resolution),
                     name='yolo_boxes_1')(output_1)
    
    outputs = tf.keras.layers.Concatenate(axis=1)([boxes_0, boxes_1])
    # print(outputs.shape)
    # outputs = tf.keras.layers.Lambda(lambda x: non_maximum_suppression(x, score_threshold, iou_threshold),
    #                  name='yolo_nms')((boxes_0[:3], boxes_1[:3]))
    return tf.keras.Model(inputs, outputs, name='yolov3_tiny')


def testing(input_shape=None, channels=3, anchors=yolo_tiny_anchors,
               masks=yolo_tiny_anchor_masks, classes=80, score_threshold = 0.5, iou_threshold = 0.5):
    
    batch_size, height, width, channels = input_shape

    batch_size = None

    x = inputs = tf.keras.layers.Input([height, width, channels], name='input', batch_size=batch_size)

    x_8, x = Convolution(batch_size=batch_size, name='convolution')(x)
    x = YoloConv(256, batch_size=batch_size, name='yolo_conv_0')(x)
    output_0 = YoloOutput(256, len(masks[0]), classes, batch_size=batch_size, name='yolo_output_0')(x)

    x = YoloConv(128, batch_size=batch_size, name='yolo_conv_1')((x, x_8))
    output_1 = YoloOutput(128, len(masks[1]), classes, batch_size=batch_size, name='yolo_output_1')(x)


    boxes_0 = tf.keras.layers.Lambda(lambda x: bbox_prediction(x, np.array([(81, 82), (135, 169),  (344, 319)]), classes, height),
                     name='yolo_boxes_0')(output_0)

    boxes_1 = tf.keras.layers.Lambda(lambda x: bbox_prediction(x, np.array([(10, 14), (23, 27), (37, 58)]), classes, height),
                     name='yolo_boxes_1')(output_1)

    
    # outputs = tf.keras.layers.Concatenate(axis=1)([boxes_0, boxes_1])

    return tf.keras.Model(inputs, output_1, name="yolo")