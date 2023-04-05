import random
import numpy as np
from PIL import Image as im

input_image = im.open("/content/pexels-yunus-tuğ-14306688.jpg")

def random_crop(input_image,square_size=400,count=0):
  input_image.thumbnail((1500,1500))
  width,height = input_image.size 
  left = random.randint(0, width - square_size)
  top = random.randint(0, height - square_size)
  right = left + square_size
  bottom = top + square_size
  cropped_image = input_image.crop((left,top,right,bottom))
  print(cropped_image.size)
  filename = str(i)+ ".png"
  return cropped_image.save(filename), im.open(filename).show()
  
  for i in range(2):
  print(input_image.size)
  random_crop(input_image)
