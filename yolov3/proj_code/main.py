import tensorflow as tf
from tensorflow.keras import Input, Model
from yolo import Yolov3_Tiny
from convert_model import *
from trainer import Trainer
from dataloader import load_images, create_batches
import os
import time 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

        
inputs = Input(shape=(416, 416, 3))
outputs, depth = Yolov3_Tiny(inputs)
model = Model(inputs, (outputs, depth))

trainer = Trainer(model)
epochs = 3
batches = []

# Prepare traininng images for depth
train_data_dir = "../nyu_train.csv"
images, gts = load_images(train_data_dir)
batches = create_batches(images, gts, batch_size=100)

print("Start training")
for e in range(epochs):
    for idx, batch in enumerate(batches):
        images, gts = batch
        start = time.time()
        loss = trainer.train(images, gts)
        print('Epoch {} Batch {} loss={} ---- {}s'.format(e, idx, loss, time.time()-start)) 
print("Finish Train")


# for layer in model.layers[]:
#     print(layer.name)
#     print(layer.trainable)
# print(model.summary())




# Save and convert model
# save_and_convert(model)

# # Load tflite
# model_path = "../models/yolov3-tiny.tflite"
# interpreter = tf.lite.Interpreter(model_path=model_path)
# interpreter.allocate_tensors()
# print("Load successfully")
