import numpy as np
from depth_calculation.disparity import *
from depth_calculation.similarity_measures import *
import torch

def extractPrediction(prediction):
    my_dict = {}
    my_dict['label'] = prediction['label']
    width = prediction['bottomright']['x'] - prediction['topleft']['x']
    height = prediction['bottomright']['y'] - prediction['topleft']['y']
    my_dict['ROI'] = {'origin': prediction['topleft'], "width":width, "height":height}
    return my_dict

# def findCommonItems(left_img, right_img):
#     info_l = []
#     info_r = []
#     info = []

#     for l in left_img:
#         info_l.append(extractPrediction(l))

#     for r in right_img:
#         info_r.append(extractPrediction(r))


#     for l in info_l:
#         for r in info_r:
#             if l['label'] == r['label']:
#                 m_dict = {}
#                 m_dict['label'] = l['label']
#                 m_dict['left_ROI'] = l['ROI']
#                 m_dict['right_ROI'] = r['ROI']
#                 info.append(m_dict)
    
#     return info

def find_depth(output, loaded_ims):
    left_idx = 0
    right_idx = 1

    left_img = loaded_ims[left_idx]
    right_img = loaded_ims[right_idx]

    for o in output:
        if o[0] == 0:
            roi = o[1:5]
            sub_img1 = left_img[roi[1]:roi[3], roi[0]:roi[2], :] # left
            sub_img2 = right_img[roi[1]:roi[3], roi[0]:roi[2], :] # right

            disparity_map = calculate_disparity_map(torch.tensor(sub_img2), torch.tensor(sub_img1), 9, ssd_similarity_measure)
            plot_disparity_map(disparity_map)

            

    
    raise NotImplementedError
