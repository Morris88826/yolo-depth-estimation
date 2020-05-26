from matplotlib import image
from tqdm import tqdm
import tensorflow as tf
import cv2
import pandas as pd
import numpy as np
import pickle

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

def pickle_dump(csvfile, filename="../data/nyu_pickle/training", size=416):
    img_dict = {}
    gt_dict = {}
    images_dir, gts_dir = get_img_dir(csvfile)
    print(gts_dir)
    num_data = len(images_dir)
    images = list()
    gts = list()

    print("Loading traing images")
    for i in tqdm(range(num_data)): 
        # load image
        
        _image = image.imread("../"+images_dir[i])
        resized_image = cv2.resize(_image, (size,size), interpolation = cv2.INTER_CUBIC)
        images.append(resized_image)

        img_dict[i] = resized_image
        
        _gt = image.imread("../"+gts_dir[i])
        resized_gt = cv2.resize(_gt, (size,size), interpolation = cv2.INTER_CUBIC)
        gts.append(np.expand_dims(resized_gt, 2))
        gt_dict[i] = resized_gt
    

    outfile = open(filename,'wb')
    pickle.dump([img_dict, gt_dict], outfile)
    outfile.close()
    return 

def load_pickle(filename):
    infile = open(filename,'rb')
    img_dict, gt_dict = pickle.load(infile)
    infile.close()

    return img_dict, gt_dict

def load_batch(filename, start_idx=None, size=None):
    infile = open(filename,'rb')
    img_dict, gt_dict = pickle.load(infile)
    infile.close()


    images = []
    gts = []



    for i in range(size):
        if i == len(img_dict.keys()):
            print("break;")
            break
        images.append(img_dict[start_idx+i])
        gts.append(gt_dict[start_idx+i])
        
    return np.array(images), np.expand_dims(np.array(gts), axis=3)
    
    
def get_data_size(filename):
    infile = open(filename,'rb')
    img_dict, gt_dict = pickle.load(infile)
    infile.close()

    return len(img_dict.keys())

# class DataLoader():
#     def __init__(self, pklfile='../data/nyu_pickle/training', TEST=True):
#         self.shape_rgb = (480, 640, 3)
#         self.shape_depth = (240, 320, 1)
#         self.read_nyu_data(pklfile, TEST=TEST)

#     def nyu_resize(self, img, resolution=416, padding=6):
#         from skimage.transform import resize
#         return resize(img, (resolution, int(resolution*4/3)), preserve_range=True, mode='reflect', anti_aliasing=True )

#     def read_nyu_data(self, pklfile, TEST=True):
#         pkl = open(pklfile, 'r').read()
#         load_pickle(pklfile)
#         # Dataset shuffling happens here
#         nyu2_train = shuffle(nyu2_train, random_state=0)

#         # Test on a smaller dataset
#         if TEST: nyu2_train = nyu2_train[:10]
        
#         # A vector of RGB filenames.
#         self.images_dir = [i[0] for i in nyu2_train]

#         # A vector of depth filenames.
#         self.gts_dir = [i[1] for i in nyu2_train]

#         # Length of dataset
#         self.length = get_data_size(pklfile)

#     def _parse_function(self, filename, label): 
#         # Read images from disk
#         image_decoded = tf.image.decode_jpeg(tf.io.read_file(filename))
#         depth_resized = tf.image.resize(tf.image.decode_jpeg(tf.io.read_file(label)), [self.shape_depth[0], self.shape_depth[1]])

#         # Format
#         rgb = tf.image.convert_image_dtype(image_decoded, dtype=tf.float32)
#         depth = tf.image.convert_image_dtype(depth_resized / 255.0, dtype=tf.float32)
        
#         # Normalize the depth values (in cm)
#         depth = 1000 / tf.clip_by_value(depth * 1000, 10, 1000)

#         return rgb, depth

#     def get_batched_dataset(self, batch_size):
#         self.dataset = tf.data.Dataset.from_tensor_slices((self.filenames, self.labels))
#         self.dataset = self.dataset.shuffle(buffer_size=len(self.filenames), reshuffle_each_iteration=True)
#         self.dataset = self.dataset.repeat()
#         self.dataset = self.dataset.map(map_func=self._parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
#         self.dataset = self.dataset.batch(batch_size=batch_size)

#         return self.dataset

