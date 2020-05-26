from matplotlib import image
from tqdm import tqdm
import tensorflow as tf
import cv2
import pandas as pd
import numpy as np

def get_img_dir(csvfile):
    frame = pd.read_csv(csvfile, header=None)
    return (frame[0], frame[1])

def load_images(csvfile, size = 416):

    images_dir, gts_dir = get_img_dir(csvfile)
    num_data = len(images_dir)
    images = list()
    gts = list()

    print("Loading traing images")
    for i in tqdm(range(num_data)): 
        # load image
        
        _image = image.imread("../"+images_dir[i])
        resized_image = cv2.resize(_image, (size,size), interpolation = cv2.INTER_CUBIC)
        images.append(resized_image)
        
        _gt = image.imread("../"+gts_dir[i])
        resized_gt = cv2.resize(_gt, (size,size), interpolation = cv2.INTER_CUBIC)
        gts.append(np.expand_dims(resized_gt, 2))
    
    return np.array(images), np.array(gts)

def create_batches(images, gts, batch_size):
    num_batches = (images.shape[0]-1+batch_size) // batch_size
    batches = []

    for i in range(num_batches):
        if i == (num_batches-1):
            _imgs = images[batch_size*i:]
            _gts = gts[batch_size*i:]
        else:
            _imgs = images[batch_size*i:batch_size*i+batch_size]
            _gts =  gts[batch_size*i:batch_size*i+batch_size]
        
        
        batches.append((_imgs, _gts))
    
    return batches
