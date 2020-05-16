import numpy as np
import tensorflow as tf
import time
from tensorflow.keras.layers import (
    Concatenate,
    Conv2D,
    Input,
    Lambda,
    LeakyReLU,
    MaxPool2D,
    UpSampling2D,
    ZeroPadding2D,
    BatchNormalization,
)
from util import bbox_prediction, non_maximum_suppression

class ConvolutionLayer(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides=1, batch_norm=True):
        super(ConvolutionLayer, self).__init__(name="Conv2D")
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.batch_norm = batch_norm
        self.padding = "VALID"
        if strides == 1:
            self.padding = "SAME"


        self.conv = tf.keras.layers.Conv2D(filters=self.filters, kernel_size=self.kernel_size, strides=self.strides, padding=self.padding, use_bias=not self.batch_norm, name="conv")
        if self.batch_norm:
            self.batch_norm = tf.keras.layers.BatchNormalization(name="batchnorm")
        
        self.in_shape = None

    def call(self, x):
        self.in_shape = x.shape[-1]

        x = self.conv(x)
        
        if self.batch_norm:
            x = self.batch_norm(x)
            x = tf.nn.leaky_relu(x, 0.1)

        return x



class Yolov3_Tiny(tf.keras.Model):
    def __init__(self, resolution=416):
        super(Yolov3_Tiny, self).__init__()
        self.resolution = resolution
        # input = tf.keras.layers.Input([416, 416, 3])
        input = tf.zeros((1,resolution,resolution,3))
        self.conv_0 = ConvolutionLayer(filters=16, kernel_size=3, strides=1, batch_norm=1)
        self.maxpool_0 = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding="SAME", name="MaxPool2D")
        self.conv_1 = ConvolutionLayer(filters=32, kernel_size=3, strides=1, batch_norm=1)
        self.maxpool_1 = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding="SAME", name="MaxPool2D")
        self.conv_2 = ConvolutionLayer(filters=64, kernel_size=3, strides=1, batch_norm=1)
        self.maxpool_2 = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding="SAME", name="MaxPool2D")
        self.conv_3 = ConvolutionLayer(filters=128, kernel_size=3, strides=1, batch_norm=1)
        self.maxpool_3 = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding="SAME", name="MaxPool2D")
        self.conv_4 = ConvolutionLayer(filters=256, kernel_size=3, strides=1, batch_norm=1)
        self.maxpool_4 = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding="SAME", name="MaxPool2D")
        self.conv_5 = ConvolutionLayer(filters=512, kernel_size=3, strides=1, batch_norm=1)
        self.maxpool_5 = tf.keras.layers.MaxPool2D(pool_size=2, strides=1, padding="SAME", name="MaxPool2D")
        self.conv_6 = ConvolutionLayer(filters=1024, kernel_size=3, strides=1, batch_norm=1)

        self.conv_7 = ConvolutionLayer(filters=256, kernel_size=1, strides=1, batch_norm=1)
        self.conv_8 = ConvolutionLayer(filters=512, kernel_size=3, strides=1, batch_norm=1)
        self.conv_9 = ConvolutionLayer(filters=255, kernel_size=1, strides=1, batch_norm=0)

        self.conv_10 = ConvolutionLayer(filters=128, kernel_size=1, strides=1, batch_norm=1)
        self.upsample = tf.keras.layers.UpSampling2D(size=2, name="UpSampling2D")
        self.concat = tf.keras.layers.Concatenate()
        self.conv_11 = ConvolutionLayer(filters=256, kernel_size=3, strides=1, batch_norm=1)
        self.conv_12 = ConvolutionLayer(filters=255, kernel_size=1, strides=1, batch_norm=0)

        self.classes = 80

        # first build
        start = time.time()
        self(input)
        print("Finish build in {}".format(time.time()-start))
    
    def load_weights(self, weight_file):
        fd = open(weight_file, 'rb')
        header = np.fromfile(fd, dtype = np.int32, count = 5)
        self.header = tf.constant(header)
        self.seen = self.header[3]

        weights = np.fromfile(fd, dtype = np.float32)

        cur = 0
        for layer in self.layers:
            if layer.name == "Conv2D":
                filters = layer.filters
                kernel_size = layer.kernel_size
                batch_norm = layer.batch_norm
                if batch_norm:
                    bn_weights = weights[cur:cur+4*filters] # [beta, gamma, moving_mean, moving_var]
                    cur += 4*filters
                    bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]] 
                
                else:
                    conv_bias = weights[cur:cur+filters]
                    cur += filters

                in_dim = layer.in_shape

                conv_shape = (filters, in_dim, kernel_size, kernel_size)
                conv_weights = weights[cur:cur+np.product(conv_shape)]
                cur += np.product(conv_shape)
                conv_weights = conv_weights.reshape(conv_shape).transpose([2, 3, 1, 0])

                if batch_norm:
                    weight_list = [conv_weights]
                    for i in bn_weights:
                        weight_list.append(i)
                    layer.set_weights(weight_list)
                else:
                    layer.set_weights([conv_weights, conv_bias])
                
        return cur


    def call(self, x):
        x = input = self.conv_0(x)
        x = self.maxpool_0(x)
        x = self.conv_1(x)
        x = self.maxpool_1(x)
        x = self.conv_2(x)
        x = self.maxpool_2(x)
        x = self.conv_3(x)
        x = self.maxpool_3(x)
        x = x_8 = self.conv_4(x)
        x = self.maxpool_4(x)
        x = self.conv_5(x)
        x = self.maxpool_5(x)
        x = self.conv_6(x)
        x1 = x2 = self.conv_7(x)

        x1 = self.conv_8(x1)
        x1 = self.conv_9(x1)
        anchors1 = np.array([[81, 82], [135, 169],  [344, 319]])/416
        output1 = tf.keras.layers.Lambda(lambda x: tf.reshape(x, (-1, x.shape[1], x.shape[2],
                                    len(anchors1), self.classes+5)))(x1)

        # bbox1 contains (bbox, objectness, class_probs, pred_box)
        box1 = tf.keras.layers.Lambda(lambda x: bbox_prediction(x, anchors1, self.classes),
                     name='yolo_boxes_0')(output1)

        x2 = self.conv_10(x2)
        x2 = self.upsample(x2)
        x2 = self.concat([x2, x_8])
        x2 = self.conv_11(x2)
        x2 = self.conv_12(x2)

        anchors2 = np.array([[10, 14], [23, 27], [37, 58]])/416
        output2 = tf.keras.layers.Lambda(lambda x: tf.reshape(x, (-1, x.shape[1], x.shape[2],
                                    len(anchors2), self.classes+5)))(x2)
        box2 = tf.keras.layers.Lambda(lambda x: bbox_prediction(x, anchors2, self.classes),
                     name='yolo_boxes_1')(output2)
        
        score_threshold = 0.5
        iou_threshold = 0.5

        if x.shape[0] is None:
            return (box1, box2)
        
        outputs = tf.keras.layers.Lambda(lambda x:non_maximum_suppression(x, score_threshold, iou_threshold), name="nms")([box1, box2])

        return outputs
