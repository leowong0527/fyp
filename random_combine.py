import random
import numpy as np
from PIL import Image as im
from os.path import join

def random_combine(img_input = "sample_data/1 (43)_B.jpeg", img_gt = "sample_data/1 (43).jpeg", minSize = 500, maxSize =1000): # just random initialization

    width, height = img_input.size
    left = random.randint(0, width - minSize)
    right = left + random.randint(minSize, maxSize)
    top = random.randint(0, height - minSize)
    bottom = top + random.randint(minSize, maxSize)

    img_input = np.array(img_input)
    img_gt = np.array(img_gt)
    
    img_gt[top:bottom, left:right, :] = 0
    img_input[:, 0:left, :] = 0
    img_input[:, right:, :] = 0
    img_input[0:top, :, :] = 0
    img_input[bottom:, :, :] = 0
    
    result = img_gt + img_input
    data = im.fromarray(result)
    return data
