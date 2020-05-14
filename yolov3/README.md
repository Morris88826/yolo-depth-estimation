# Yolov3 Tensorflow 2.0

## Build the environment
Download weights file using wget, save under the directory "yolov3/weights"
```clike
mkdir weights
cd weights
## weights file for yolov3
wget https://pjreddie.com/media/files/yolov3.weights
## weights file for yolov3-tiny
wget https://pjreddie.com/media/files/yolov3-tiny.weights
``` 

## Brief explanation
All the related code is stores in proj_code folder. 

"detector_tf.py" contains the Detector class which takes in a model and batch_size

We first create our detector given our model then use **predict** function to give out the prediction output. 

```clike
def predict(self, im_batches, confidence=0.35, nms_thesh=0.2, print_info=True):
    ### Input
    ### im_batches:  a list of batches, each batch is in shape (N, H, W, C)
    
    ### Output
    ### prediction: (number_of_detection x 8)
    ###             for dim=1, the first index stores the batch_id, 
    ###             index[1:5] stores the bounding box coordinates(topleft, bottomright) 
    ###             index[-1] stores the predicted class index


```
## Test case
Just run "detector.py"







