import tensorflow as tf
from util import *
from module import *
from gen_prediction import get_prediction
import os 
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# For eager execution. See https://github.com/tensorflow/tensorflow/issues/34944
# This is to state whether it should be executed eagerly (default=True) explicitly
# but this does not solve the problem 
tf.config.experimental_run_functions_eagerly(True)

class Darknet(tf.Module):
    def __init__(self, num_classes, cfgfile, classes_file, size=None, channels=3, classes=80, weight_file=""):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.classes = class_names(classes_file)
        self.weight_file = weight_file
        self.net_info, self.module_list = create_modules(self.blocks)
        self.net_info["height"] = size
        self.num_classes = num_classes

        inp_dim = int(self.net_info["height"])
        assert inp_dim % 32 == 0 
        assert inp_dim > 32
        
        x = inputs = tf.keras.Input([size, size, channels], name='input')
        self.__call__(x)

        if self.weight_file != "":
            self.load_weights(weight_file)
        

    def __call__(self, x, CUDA=False):
        modules_info = self.blocks[1:]
        hidden_outputs = {}
        detections = None
        init = False
        for idx, module_info in enumerate(modules_info):
            type = module_info["type"]
            if type == "convolutional" or type == "upsample" or type == "maxpool":
                output = self.module_list[idx](x)

            elif type == "route":
                layers = module_info["layers"].split(',')
                layers = [int(l) for l in layers]
                start = layers[0]

                if len(layers) == 1:
                    output = hidden_outputs[idx + start]
                else:
                    end = layers[1]
                    if end > 0:
                        end = end - idx
                    hidden_output_1 = hidden_outputs[idx + start]
                    hidden_output_2 = hidden_outputs[idx + end]
                    output = tf.concat((hidden_output_1,hidden_output_2), axis=3)
                
            elif type == "shortcut":
                start = int(module_info["from"])
                hidden_output_1 = hidden_outputs[idx + start]
                hidden_output_2 = hidden_outputs[idx - 1]
                output = hidden_output_1 + hidden_output_2

            elif type == "yolo":
                anchors = self.module_list[idx].get_layer(index=0).anchors
                num_classes = int(module_info["classes"])
                
                x = predict_bounding_box(x, int(self.net_info["height"]), anchors, num_classes, CUDA)

                if x is None:
                    continue

                if not init:

                    detections = tf.Variable(x)
                    init = True
                else:  
                    detections = tf.concat((detections, x), 1)

            x = output
            hidden_outputs[idx] = x

        return detections
    
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
                _model = self.module_list[idx]
                for i , layer in enumerate(_model.layers):
                    if layer.name != "Conv2d":
                        continue

                    batch_norm_layer = None
                    if (i+1) < len(_model.layers) and _model.layers[i+1].name == "BatchNorm2d":
                        batch_norm_layer = _model.layers[i+1]

                    filters = layer.filters
                    size = layer.kernel_size[0]
                    in_dim = layer.input_shape[-1]

                    if batch_norm_layer == None:
                        conv_bias = weights[cur:cur+filters]
                        cur += filters
                    else:
                        bn_weights = weights[cur:cur+4*filters]
                        cur += 4*filters
                        bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]] 

                    conv_shape = (filters, in_dim, size, size)
                    conv_weights = weights[cur:cur+np.product(conv_shape)]
                    cur += np.product(conv_shape)
                    conv_weights = conv_weights.reshape(conv_shape).transpose([2, 3, 1, 0])

                    if batch_norm_layer == None:
                        layer.set_weights([conv_weights, conv_bias])
                    else:
                        layer.set_weights([conv_weights])
                        batch_norm_layer.set_weights(bn_weights)
                