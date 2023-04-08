import random
import numpy as np
from PIL import Image

##input_image = im.open("/content/pexels-yunus-tuğ-14306688.jpg")

def random_crop(img_input, img_gt,square_size=400):
    ##img_input.thumbnail((1500,1500)) ##?
    ##img_gt.thumbnail((1500,1500))
    
    width,height = img_gt.size
    ##print(width,height)
    if width < square_size:
        img_gt = img_gt.resize((square_size, round(square_size/width*height)), Image.BICUBIC)
        img_input = img_input.resize((square_size, round(square_size/width*height)), Image.BICUBIC)
        width,height = img_gt.size  
    
    if height < square_size:
        img_gt = img_gt.resize((round(square_size*width/height), square_size), Image.BICUBIC)
        img_input = img_input.resize((round(square_size*width/height), square_size), Image.BICUBIC)
        width,height = img_gt.size
    
    left = random.randint(0, width - square_size)
    top = random.randint(0, height - square_size)
    right = left + square_size
    bottom = top + square_size
    img_input = img_input.crop((left,top,right,bottom))
    img_gt = img_gt.crop((left,top,right,bottom))
    ##print(cropped_image.size)
    ##filename = str(i)+ ".png"
    return img_input, img_gt

'''  
for i in range(2):
    print(input_image.size)
    random_crop(input_image)
'''