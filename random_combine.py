import random
import numpy as np
from PIL import Image as im
from os.path import join

def random_combine(imgIn = "sample_data/1 (43)_B.jpeg", imgGT = "sample_data/1 (43).jpeg", minSize = 500, maxSize =1000, count=0):
    # imgIn = im.open(join("pix2pix-pytorch-master", imgIn)) 
    # imgGT = im.open(join("pix2pix-pytorch-master", imgGT)) 

    imgIn = im.open(join(imgIn)) 
    imgGT = im.open(join(imgGT))

    width, height = imgIn.size
    print(width, height)
    left = random.randint(0, width - minSize)
    right = left + random.randint(minSize, maxSize)
    top = random.randint(0, height - minSize)
    bottom = top + random.randint(minSize, maxSize)
    print(left, right, top, bottom)

    imgIn = np.array(imgIn)
    imgGT = np.array(imgGT)
    
    imgIn[top:bottom, left:right, :] = 0
    imgGT[:, 0:left, :] = 0
    imgGT[:, right:, :] = 0
    imgGT[0:top, :, :] = 0
    imgGT[bottom:, :, :] = 0
    
    result = imgGT + imgIn
    data = im.fromarray(result)
    filename = str(i) + ".png"
    return data.save(filename), im.open(filename).show()
