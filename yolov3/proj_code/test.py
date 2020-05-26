from gluoncv import data, utils
from matplotlib import pyplot as plt
import time
from dataloader import *
import pickle
import matplotlib.pyplot as plt

training_pkl_dir = "../data/nyu_pickle/training"
train_data_dir = "../nyu_train.csv"
# pickle_dump(train_data_dir, filename=training_pkl_dir)

batch_size = 32 
imgs, gts = load_batch(training_pkl_dir, 0*batch_size, batch_size)

fig=plt.figure(figsize=(15, 10))
fig.add_subplot(1, 4, 1)
plt.imshow(gts[0].squeeze())

fig.add_subplot(1, 4, 2)
plt.imshow(gts[1].squeeze())

plt.show()

# start = time.time()
# train_dataset = data.COCODetection(splits=['instances_train2017'])
# val_dataset = data.COCODetection(splits=['instances_val2017'])
# print("Loading time: {:3f}s".format(time.time()-start))
# print('Num of training images:', len(train_dataset))
# print('Num of validation images:', len(val_dataset))
