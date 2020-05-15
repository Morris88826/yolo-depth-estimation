import tensorflow as tf
import numpy as np
import cv2


def get_prediction(net_output, confidence, num_classes, nms_threshold = 0.4):
    net_output = net_output.numpy()
    confidence_mask = np.expand_dims((net_output[:,:,4] > confidence).astype(float), 2)
    net_output = net_output*confidence_mask
    
    
    # net_output = net_output.assign(net_output * confidence_mask) # setting value less then confidence to 0 

    bounding_box_corner = np.copy(net_output) # create a new tensor using the same data and device
    # top left
    bounding_box_corner[:,:,0] = (net_output[:,:,0] - net_output[:,:,2]/2)
    bounding_box_corner[:,:,1] = (net_output[:,:,1] - net_output[:,:,3]/2)
    # bottom right
    bounding_box_corner[:,:,2] = (net_output[:,:,0] + net_output[:,:,2]/2) 
    bounding_box_corner[:,:,3] = (net_output[:,:,1] + net_output[:,:,3]/2)
    
    net_output[:,:,:4] = bounding_box_corner[:,:,:4]

    # since the number of true detections in every image may be different.
    batch_size = net_output.shape[0]
    init = False
    for i in range(batch_size):
        object_prediction = net_output[i] # shape = ((13x13+26x26)x3) x (5+C)
        max_indices = np.argmax(object_prediction[:,5:], axis=1) # max_indices -> (2535,)
        max_values = np.max(object_prediction[:,5:], axis=1)

        max_values = np.expand_dims(max_values.astype(float), 1) # (2535,1)
        max_indices = np.expand_dims(max_indices.astype(float), 1) # (2535, 1)
        

        # object_prediction.shape = 2535 x (5+2) = 2535 x 7
        object_prediction = np.concatenate((object_prediction[:,:5], max_values, max_indices), 1)

        non_zero_idx = np.nonzero(object_prediction[:,4])[0]

        # non_zero_idx = tf.where(object_prediction[:,4]!=0)
        if non_zero_idx.shape[0] == 0:
            continue

        object_prediction = object_prediction[non_zero_idx, :]
        
        image_classes = find_unique(object_prediction[:, -1])
        
        for c in image_classes:
            object_class = object_prediction * np.expand_dims((object_prediction[:, -1] == c).astype(float), 1)
            class_mask_ind = np.nonzero(object_class[:,-2])[0] # Since previously set those below confidence to 0. squeeze to 1D.
            
            
            prediction_output = object_prediction[class_mask_ind].reshape(-1,7)

            sorted_id  = np.argsort(prediction_output[:,4])[::-1]

            prediction_output = prediction_output[sorted_id] # sorted prediction_output

            num_output = prediction_output.shape[0]

            for idx in range(num_output):
                try:
                    if idx >= prediction_output.shape[0]-1:
                        break
                    ious = bounding_box_IoU(np.expand_dims(prediction_output[idx], 0), prediction_output[idx+1:])
                    # print(ious)
                except ValueError:
                    break

                except IndexError:
                    break


                #Zero out all the detections that have IoU > threshold
                iou_mask = np.expand_dims((ious < nms_threshold).astype(float), 1)
                prediction_output[idx+1:] *= iou_mask # leaving only value < threshold
                #Remove the non-zero entries
                non_zero_ind = np.nonzero(prediction_output[:,4])[0]
                prediction_output = prediction_output[non_zero_ind]

            batch_ind = np.ones((prediction_output.shape[0], 1))*i      
            #Repeat the batch_id for as many detections of the class cls in the image
            seq = (batch_ind, prediction_output)
            # print("End", prediction_output)

            if not init:
                output = np.concatenate(seq,1)
                init = True
            else:
                out = np.concatenate(seq,1)
                output = np.concatenate((output,out))  
    try:
        return output
    except:
        return 0

def find_unique(input):
    input = np.unique(input)
    return input




def bounding_box_IoU(box1, box2):
    # box1.shape[N,7] box2.shape[M,7]
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]

    #get the corrdinates of the intersection rectangle
    inter_rect_x1 =  np.maximum(b1_x1, b2_x1)
    inter_rect_y1 =  np.maximum(b1_y1, b2_y1)
    inter_rect_x2 =  np.minimum(b1_x2, b2_x2)
    inter_rect_y2 =  np.minimum(b1_y2, b2_y2)
    
    #Intersection area
    inter_area = np.clip(inter_rect_x2 - inter_rect_x1 + 1,a_max=None, a_min=0) * np.clip(inter_rect_y2 - inter_rect_y1 + 1,a_max=None, a_min=0)
    #Union Area
    b1_area = (b1_x2 - b1_x1 + 1)*(b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1)*(b2_y2 - b2_y1 + 1)
    
    # iou = inter_area / (b1_area + b2_area - inter_area)
    iou = inter_area / np.minimum(b1_area, b2_area)
    
    return iou
    
