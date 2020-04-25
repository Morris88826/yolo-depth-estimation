from PIL import Image
import numpy as np
import cv2
from torchvision import transforms

def resizeImage(img, dim):
    width, height = dim
    output = cv2.resize(img, (width, height))
    return output

