"""
This file holds the main code for disparity map calculations
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple


def calculate_disparity_map(right_img: torch.Tensor,
                            left_img: torch.Tensor,
                            block_size: Tuple[int, int],
                            sim_measure_function: Callable,
                            max_search_bound: int = 50) -> torch.Tensor:


  assert left_img.shape == right_img.shape

  upperbound = max_search_bound + 1
  border = block_size//2
  h_range = np.arange(border, right_img.shape[0]-border, dtype=int)
  w_range = np.arange(border, right_img.shape[1]-border, dtype=int)
  disparity_map = (np.ones((h_range.shape[0], w_range.shape[0]))) 

  for h in h_range:
    for w in w_range:
        l_start_x = w - border
        l_start_y = h - border
        error = np.ones(upperbound)*1000000 #initialize to be very big
        for d in range(upperbound):
            r_start_x = w - border - d
            if r_start_x <=0:
                break
            error[d] = sim_measure_function(right_img[l_start_y:l_start_y+block_size, l_start_x:l_start_x+block_size, :], left_img[l_start_y:l_start_y+block_size, r_start_x:r_start_x+block_size, :])
        index = np.argmin(error)
        
        disparity_map[h-h_range[0],w-w_range[0]] = index

  return torch.from_numpy(disparity_map)

def calculate_cost_volume(left_img: torch.Tensor,
                          right_img: torch.Tensor,
                          max_disparity: int,
                          sim_measure_function: Callable,
                          block_size: int = 9):

  H = left_img.shape[0]
  W = right_img.shape[1]
  cost_volume = torch.zeros(H, W, max_disparity)

  border = block_size//2
  h_range = np.arange(border, left_img.shape[0]-border, dtype=int)
  w_range = np.arange(border, left_img.shape[1]-border, dtype=int)
  cost_volume = (cost_volume+1)*255 

  for h in h_range:
    for w in w_range:
        l_start_x = w - border
        l_start_y = h - border
        error = np.ones(max_disparity)*255 #initialize to be very big
        for d in range(max_disparity):
            r_start_x = w - border - d
            if r_start_x <=0:
                break
            error[d] = sim_measure_function(left_img[l_start_y:l_start_y+block_size, l_start_x:l_start_x+block_size, :], right_img[l_start_y:l_start_y+block_size, r_start_x:r_start_x+block_size, :])
        
        cost_volume[h,w] = torch.from_numpy(error)

  return cost_volume


def plot_disparity_map(disparity_map):

  fig, ax1 = plt.subplots()

  im = ax1.imshow(disparity_map, cmap='jet')
  ax1.set_title('Disparity Map - SSD ({}x{} patch)'.format(9, 9))
  ax1.autoscale(True)
  ax1.set_axis_off()
  cbar = fig.colorbar(im, ax=ax1, cmap='jet', shrink=0.3)

  plt.show()