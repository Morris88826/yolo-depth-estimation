from __future__ import division
import tensorflow as tf
import cv2
import numpy as np
import random
import pandas as pd

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

def non_maximum_suppression(outputs, score_threshold, iou_threshold):
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

    scores = confidence * class_probs
    nmsed_boxes, nmsed_scores, nmsed_classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(bbox, (tf.shape(bbox)[0], -1, 1, 4)),
        scores=tf.reshape(scores, (tf.shape(scores)[0], -1, tf.shape(scores)[-1])),
        max_output_size_per_class=100,
        max_total_size=100,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
        clip_boxes=False
    )


    nmsed_boxes = nmsed_boxes*416
    nmsed_scores = tf.expand_dims(nmsed_scores, axis=2)
    nmsed_classes = tf.expand_dims(nmsed_classes, axis=2)


    valid_outputs = None
    for i in range(tf.shape(bbox)[0]):
        valid = valid_detections[i]
        if valid == 0:
            continue

        idx = tf.ones_like(nmsed_scores[i, :valid, :]) * i
        valid_output = tf.concat((idx, nmsed_boxes[i, :valid, :], nmsed_scores[i, :valid, :], nmsed_classes[i, :valid, :]), axis=1)
        
        if valid_outputs is None:
            valid_outputs = valid_output
        else: 
            valid_outputs = tf.concat((valid_outputs, valid_output), axis=0)
    

    return valid_outputs

def transform_output(output, im_dim_list, resolution):
    output = output.numpy()
    inp_dim = resolution
    im_dim_list = np.take(im_dim_list, output[:,0].astype(np.long), axis=0)
    scaling_factor = np.reshape(np.min(inp_dim/im_dim_list,1), (-1,1))

    output[:,[1,3]] -= (inp_dim - scaling_factor*im_dim_list[:,0].reshape((-1,1)))/2
    output[:,[2,4]] -= (inp_dim - scaling_factor*im_dim_list[:,1].reshape((-1,1)))/2

    output[:,1:5] /= scaling_factor

    for i in range(output.shape[0]):
        output[i, [1,3]] = np.clip(output[i, [1,3]], 0.0, im_dim_list[i,0])
        output[i, [2,4]] = np.clip(output[i, [2,4]], 0.0, im_dim_list[i,1])
    
    new_output = np.zeros((output.shape[0], 6))
    new_output[:, :5] = output[:, :5]
    new_output[:, 5] = output[:, -1]

    return new_output.astype(int)

def class_names(name_file):
    file = open(name_file, 'r')
    lines = file.read().split('\n')[:-1]
    return lines

def draw_boxes(batches_predictions, loaded_ims, colors, classes, batch_size):
    def draw_box(x, loaded_ims, color, classes, batch_id):
        c1 = tuple(x[1:3].astype(int))
        c2 = tuple(x[3:5].astype(int))

        img = loaded_ims[batch_id+int(x[0])]
        cls = int(x[-1])
        label = "{0}".format(classes[cls])
        cv2.rectangle(img, c1, c2,color, 1)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
        c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
        cv2.rectangle(img, c1, c2,color, -1)
        cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1)
        return img

    cur = 0
    for idx, batch_images in enumerate(batches_predictions):
        list(map(lambda x: draw_box(x, loaded_ims, random.choice(colors), classes, batch_id=cur), batch_images))
        cur += batch_size

def save_drawed_images(imlist, output_dir, loaded_ims):
    det_names = pd.Series(imlist).apply(lambda x: "{}/det_{}".format(output_dir, x.split("/")[-1]))
    list(map(cv2.imwrite, det_names, loaded_ims))
