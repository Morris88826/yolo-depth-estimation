from gluoncv import data, utils
from matplotlib import pyplot as plt
import time


start = time.time()
train_dataset = data.COCODetection(splits=['instances_train2017'])
val_dataset = data.COCODetection(splits=['instances_val2017'])
print("Loading time: {:3f}s".format(time.time()-start))
print('Num of training images:', len(train_dataset))
print('Num of validation images:', len(val_dataset))
