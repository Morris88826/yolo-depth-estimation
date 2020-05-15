import numpy as np
import cv2
import tensorflow as tf

def letterbox_image(img, inp_dim):
    '''resize image with unchanged aspect ratio using padding'''
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w/img_w, h/img_h))
    new_h = int(img_h * min(w/img_w, h/img_h))
    resized_image = cv2.resize(img, (new_w,new_h), interpolation = cv2.INTER_CUBIC)
    
    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)

    canvas[(h-new_h)//2:(h-new_h)//2 + new_h,(w-new_w)//2:(w-new_w)//2 + new_w,  :] = resized_image
    
    return canvas
    
def preprocess_image(img, reshape_size):
    model_image_size = (reshape_size, reshape_size)
    boxed_image = letterbox_image(img, tuple(reversed(model_image_size)))
    image_data = np.array(boxed_image, dtype='float32')
    image_data /= 255
    image_data = np.expand_dims(image_data, 0)
    image_data = tf.Variable(image_data)
    return image_data

def create_batches(im_batches, batch_size):
    remaining = 0
    if (len(im_batches) % batch_size):
        remaining = 1

    if batch_size != 1:
        num_batches = len(im_batches) // batch_size + remaining    

        im_batches = [tf.concat((im_batches[i*batch_size : min((i+1)*batch_size, len(im_batches))]), axis=0)  
                        for i in range(num_batches)] 

    return im_batches 

def get_test_input(size):
    img = cv2.imread("../images/test1.png")
    img = preprocess_image(img, size)         #Resize to the input dimension
    return img