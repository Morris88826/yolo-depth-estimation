"""
Convert NYU Depth v2 DS from the official .mat file on
https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html
"""
import skimage.io as io
from PIL import Image
import numpy as np
import h5py
import os

def convert_images_from_mat(matfile):

    f = h5py.File(matfile)

    for i in range(1499):
        # [3 x 640 x 480], uint8
        img = f['images'][i]
        img = np.transpose(img, (2, 1, 0))

        im = Image.fromarray(img)
        im.save(f"../data/nyu_datasets/{i:05}.jpg")

        # [640 x 480], float64
        depth = f['depths'][i]
        depth = np.transpose(depth, (1, 0))

        im = Image.fromarray(np.uint8(depth/10.0 * 255), 'L')
        im.save(f"../data/nyu_datasets/{i:05}.png", "PNG")


if __name__ == "__main__":
    matfile = "../data/nyu_depth_v2_labeled.mat"
    convert_images_from_mat(matfile)
