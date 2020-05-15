from __future__ import division
from tensorflow import keras
from preprocessing import *
import tensorflow as tf
import cv2
import numpy as np


def parse_cfg(cfgfile):
    file = open(cfgfile, 'r')
    lines = file.read().split('\n')  
    lines = [ x for x in lines if (len(x)>0 and x[0]!='#')]
    lines = [x.rstrip().lstrip() for x in lines] # strip white spaces

    block = {}
    blocks = []

    for line in lines:
        if line[0] == '[':
            if len(block) > 0: # Not empty
                blocks.append(block)
                block = {}
            block["type"] = line[1:-1].rstrip() 
        else:
            key, value = line.split('=')
            key = key.rstrip()
            value = value.lstrip()
            block[key] = value
    blocks.append(block)

    return blocks

def predict_bounding_box(prediction, input_dim, anchors, num_classes):
    batch_size, H, W, _ = prediction.shape # channel last
    if batch_size == None:
        return None

    stride = input_dim // H
    # print(H)
    
    grid_size = input_dim //stride
    num_bounding_box = len(anchors)
    C = num_classes
    B = num_bounding_box
    entries = (5+C)*B
    
    prediction = tf.reshape(prediction, [batch_size, grid_size*grid_size*B, 5+C]) # batch_size x (5+C)*B x vectorize_grid
    prediction = tf.Variable(prediction)
    
    prediction = prediction[:,:,0].assign(tf.sigmoid(prediction[:,:,0]))
    prediction = prediction[:,:,1].assign(tf.sigmoid(prediction[:,:,1]))
    prediction = prediction[:,:,4].assign(tf.sigmoid(prediction[:,:,4]))

    anchors = [(a[0]/stride, a[1]/stride) for a in anchors]
    xv, yv = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
    
    cx = tf.reshape(tf.constant(xv, dtype=tf.float32), [-1,1])
    cy = tf.reshape(tf.constant(yv, dtype=tf.float32), [-1,1])

    offset = tf.concat((cx,cy), axis=1)
    offset = tf.repeat(offset, repeats=num_bounding_box, axis=0) # shape = (grid_size*grid_size*B)x(5+C)  
    offset = tf.expand_dims(offset, 0) # shape = 1x(grid_size*grid_size*B)x(5+C)


    prediction = prediction[:,:,:2].assign(prediction[:,:,:2]+offset)

    
    anchors = tf.Variable(anchors)
    anchors = tf.expand_dims(tf.repeat(anchors, grid_size*grid_size, axis=0), 0)
    # print(anchors)

    prediction = prediction[:,:,2:4].assign(tf.exp(prediction[:,:,2:4])*anchors)
    prediction = prediction[:,:, 5:].assign(tf.sigmoid(prediction[:,:,5:]))
    prediction = prediction[:,:,:4].assign(prediction[:,:,:4]*stride) # put the coordinate back to the size relatively to the image
    
    return prediction

def bbox_prediction(prediction, anchors, classes):
    grid_size = prediction.shape[1]
    box_xy, box_wh, objectness, class_probs = tf.split(prediction, (2, 2, 1, classes), axis=-1)

    box_xy = tf.sigmoid(box_xy)
    objectness = tf.sigmoid(objectness)
    class_probs = tf.sigmoid(class_probs)
    pred_box = tf.concat((box_xy, box_wh), axis=-1)  #[N,H,W,4]

    grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
    grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)  # [gx, gy, 1, 2]

    box_xy = (box_xy + tf.cast(grid, tf.float32)) / tf.cast(grid_size, tf.float32)
    box_wh = tf.exp(box_wh) * anchors

    box_x1y1 = box_xy - box_wh / 2 # top left
    box_x2y2 = box_xy + box_wh / 2 # bottom right
    bbox = tf.concat([box_x1y1, box_x2y2], axis=-1)

    return bbox, objectness, class_probs, pred_box

def non_maximum_suppression(outputs, nms_threshold):

    b = []
    c = []
    t = []

    for o in outputs:
        b.append(tf.reshape(o[0], (tf.shape(o[0])[0], -1, tf.shape(o[0])[-1])))
        c.append(tf.reshape(o[1], (tf.shape(o[1])[0], -1, tf.shape(o[1])[-1])))
        t.append(tf.reshape(o[2], (tf.shape(o[2])[0], -1, tf.shape(o[2])[-1])))
    
    bbox = tf.concat(b, axis=1)
    confidence = tf.concat(c, axis=1)
    class_probs = tf.concat(t, axis=1)
    
    return bbox

def get_test_input(size):
    img = cv2.imread("../images/test1.png")
    img = preprocess_image(img, size)         #Resize to the input dimension
    return img

def class_names(name_file):
    file = open(name_file, 'r')
    lines = file.read().split('\n')[:-1]
    return lines

def draw_box(x, results, color, classes):
    c1 = tuple(x[1:3].astype(int))
    c2 = tuple(x[3:5].astype(int))

    img = results[int(x[0])]
    cls = int(x[-1])
    label = "{0}".format(classes[cls])
    cv2.rectangle(img, c1, c2,color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2,color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1)
    return img