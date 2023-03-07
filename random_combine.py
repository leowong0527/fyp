import random
import numpy as np
from PIL import Image as im
from os.path import join

def random_combine(imgIn, imgGT, minSize, maxSize):
    # imgIn = im.open(join("pix2pix-pytorch-master", imgIn)) 
    # imgGT = im.open(join("pix2pix-pytorch-master", imgGT)) 

    imgIn = im.open(join(imgIn)) 
    imgGT = im.open(join(imgGT))

    width, height = imgIn.size
    left = random.randint(0, width - minSize)
    right = left + random.randint(minSize, maxSize)
    top = random.randint(0, height - minSize)
    bottom = top + random.randint(minSize, maxSize)
    
    imgIn = np.array(imgIn)
    imgGT = np.array(imgGT)
    
    imgIn[left:right, top:bottom, :] = 0
    imgGT[0:left, :, :] = 0
    imgGT[right:, :, :] = 0
    imgGT[:, 0:top, :] = 0
    imgGT[:, bottom:, :] = 0
    
    result = imgGT + imgIn
    data = im.fromarray(result)
    
    return data.save("dummy.png")