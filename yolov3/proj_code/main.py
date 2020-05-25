import os
import time

import tensorflow as tf
from tensorflow.keras import Input, Model

from convert_model import *
from dataloader import create_batches, load_images
from trainer import Trainer
from yolo import Yolov3_Tiny

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

        
inputs = Input(shape=(416, 416, 3))
outputs, depth = Yolov3_Tiny(inputs)
model = Model(inputs, (outputs, depth))

model.compile(optimizer='rmsprop', loss='mean_squared_error')

trainer = Trainer(model)
epochs = 1
batches = []

# Prepare traininng images for depth
train_data_dir = "../nyu_test.csv"
images, gts = load_images(train_data_dir)
batches = create_batches(images, gts, batch_size=100)

# x_eval = images[-2:]
# y_eval = gts[-2:]
# x_train = images[:-2]
# y_train = gts[:-2]


# Train the model.
# print('# Fit model on training data')
# history = model.fit(x_train, y_train, )
#                     # batch_size=100, epochs=epochs, 
#                     # validation_data=(x_val, y_val),)

# print('\nhistory dict:', history.history)

print("Start training")
for e in range(epochs):
    ckpt_dir = "../ckpt/cp_{}".format(e)
    for idx, batch in enumerate(batches):
        images, gts = batch
        start = time.time()
        loss = trainer.train(images, gts)
        print('Epoch {} Batch {} loss={} ---- {}s'.format(e, idx, loss, time.time()-start)) 
    
    model.save_weights(ckpt_dir)
    print("Epoch {}, saving weights".format(e))
print("Finish Train")


# for layer in model.layers[]:
#     print(layer.name)
#     print(layer.trainable)
# print(model.summary())




# Save and convert model
# save_model(model, model_path="../models/yolov3-depth-tiny.h5")
# save_and_convert(model)

# # Load tflite
# model_path = "../models/yolov3-tiny.tflite"
# interpreter = tf.lite.Interpreter(model_path=model_path)
# interpreter.allocate_tensors()
# print("Load successfully")
