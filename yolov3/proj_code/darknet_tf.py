import tensorflow as tf
from util import *
from layers import *
from gen_prediction import get_prediction
import os 
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# For eager execution. See https://github.com/tensorflow/tensorflow/issues/34944
# This is to state whether it should be executed eagerly (default=True) explicitly
# but this does not solve the problem 
# tf.config.experimental_run_functions_eagerly(True)

class Darknet(tf.keras.Model):
    def __init__(self, num_classes, cfgfile, classes_file, size=None, channels=3, classes=80, weight_file=""):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.classes = class_names(classes_file)
        self.weight_file = weight_file
        
        self.net_info, self.m_layers = create_layers(self.blocks)
        
        self.net_info = self.blocks[0]
        self.net_info["height"] = size
        self.num_classes = num_classes

        inp_dim = int(self.net_info["height"])
        assert inp_dim % 32 == 0 
        assert inp_dim > 32
        
        self.build_network(tf.zeros((1,size, size, channels)))

        if self.weight_file != "":
            self.load_weights(weight_file)
        
    
    def build_network(self, x):
        self.call(x)

    def call(self, x):
        blocks_info = self.blocks[1:]
        self.hidden_outputs = {}
        detections = None
        init = False

        for idx, block_info in enumerate(blocks_info):

            type = block_info["type"]

            if type == "convolutional" or type == "upsample" or type == "maxpool":
                x = self.m_layers[idx](x)

            elif type == "route": 
                layers_idx = block_info["layers"].split(',')
                layers_idx = [int(l) for l in layers_idx]
                if len(layers_idx) == 1:
                    layers_idx.append(0)
                x = self.route_layers(idx, layers_idx[0], layers_idx[1])
                
            elif type == "shortcut":
                start = int(block_info["from"])
                hidden_output_1 = self.hidden_outputs[idx + start]
                hidden_output_2 = self.hidden_outputs[idx - 1]
                x = hidden_output_1 + hidden_output_2

            elif type == "yolo":
                x = self.m_layers[idx](x)

                if x is None: # if in build weight stage
                    continue

                if not init:
                    detections = tf.Variable(x)
                    init = True
                else:  
                    detections = tf.concat((detections, x), 1)

            self.hidden_outputs[idx] = x

        return detections
    
    def route_layers(self, current_idx, layer1, layer2=0):
        '''
        layer1, layer2 represent the layer number

        Two kinds of representation: (negative, positive) or (negative, negative)

        '''
        layer1 = current_idx + layer1
        if layer2 == 0:
            output = self.hidden_outputs[layer1]
        else:
            if layer2 < 0:
                layer2 = current_idx + layer2
            hidden_output_1 = self.hidden_outputs[layer1]
            hidden_output_2 = self.hidden_outputs[layer2]
            output = tf.concat((hidden_output_1,hidden_output_2), axis=3)
        
        return output
    
    def load_weights(self, weight_file):
        fd = open(weight_file, 'rb')
        header = np.fromfile(fd, dtype = np.int32, count = 5)
        self.header = tf.constant(header)
        self.seen = self.header[3]

        weights = np.fromfile(fd, dtype = np.float32)
        cur = 0
        for idx in range(len(self.blocks)-1):
            module_info = self.blocks[1+idx]
            type = module_info["type"]

            # load weights for convolutional
            if type == "convolutional":
                layer = self.m_layers[idx]
                
                num_filters = layer.filters
                kernel_size = layer.kernel_size
                
                if layer.require_batchnorm: # with batch norm
                    bn_weights = weights[cur:cur+4*num_filters]
                    cur += 4*num_filters
                    bn_weights = bn_weights.reshape((4, num_filters))[[1, 0, 2, 3]] 

                else:
                    conv_bias = weights[cur:cur+num_filters]
                    cur += num_filters

                in_dim = layer.in_shape

                conv_shape = (num_filters, in_dim, kernel_size, kernel_size)
                conv_weights = weights[cur:cur+np.product(conv_shape)]
                cur += np.product(conv_shape)
                conv_weights = conv_weights.reshape(conv_shape).transpose([2, 3, 1, 0])

                if layer.require_batchnorm:
                    weight_list = [conv_weights]
                    for i in bn_weights:
                        weight_list.append(i)
                    layer.set_weights(weight_list)
                else:
                    layer.set_weights([conv_weights, conv_bias])



                