import os
import time

import tensorflow as tf
from tensorflow.keras import Input, Model

from convert_model import *
from dataloader import create_batches, load_images, load_batch, get_data_size
from trainer import Trainer
from yolo import Yolov3_Tiny

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

     
inputs = Input(shape=(416, 416, 3))
outputs, depth = Yolov3_Tiny(inputs)
model = Model(inputs, (outputs, depth))



# model.compile(optimizer='rmsprop', loss='mean_squared_error')
# print(model.summary())



# Prepare traininng images for depth
# train_data_dir = "../nyu_train.csv"
# images, gts = load_images(train_data_dir)
# batches = create_batches(images, gts, batch_size=100)

train = False

if train:
    trainer = Trainer(model)
    epochs = 1
    batches = []
    training_pkl_dir = "../data/nyu_pickle/training"
    batch_size = 100
    num_batches = get_data_size(training_pkl_dir) // batch_size
    print("Start training")
    for e in range(epochs):
        ckpt_dir = "../ckpt/cp_{}".format(e)
        for idx in range(num_batches):
            images, gts = load_batch(training_pkl_dir, idx*batch_size, batch_size)
            start = time.time()
            loss = trainer.train(images, gts)
            print('Epoch {} Batch {} loss={} ---- {}s'.format(e, idx, loss, time.time()-start)) 
        
        model.save_weights(ckpt_dir)
        print("Epoch {}, saving weights".format(e))
    print("Finish Train")




ckpt_dir = "../ckpt_5_1e-4_{}/cp_1".format("v1")
model_path = "../models/yolov3-depth-tiny.h5"
tflite_path = "../models/yolov3-depth-tiny.tflite"
model.load_weights(ckpt_dir)
# Save and convert model
save_model(model, model_path=model_path)
save_and_convert(model, model_path=model_path, tflite_path=tflite_path)

# # Load tflite
interpreter = tf.lite.Interpreter(model_path=tflite_path)
interpreter.allocate_tensors()
print("Load successfully")
