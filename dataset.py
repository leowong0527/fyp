from os import listdir
from os.path import join
import random

##import imageio.core.functions
from imageio import *
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
##from torchvision.transforms import RandomCrop

from random_crop import random_crop
from random_combine import random_combine

#from utils import is_image_file, load_img
class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, direction):
        super(DatasetFromFolder, self).__init__()
        self.direction = direction
        self.gt_path = join(image_dir, "gt")
        self.input_path = join(image_dir, "input")
        self.image_filenames = [x for x in listdir(self.gt_path)]

        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        img_gt = Image.open(join(self.gt_path, self.image_filenames[index])).convert('RGB')
        img_input = Image.open(join(self.input_path, self.image_filenames[index])).convert('RGB')

        ##img_gt = img_gt.resize((286, 286), Image.BICUBIC) #resize change to crop (pillow crop)
        ##img_input = img_input.resize((286, 286), Image.BICUBIC) #resize change to crop (pillow crop)

        img_input, img_gt = random_crop(img_input, img_gt, 286)
        img_input = random_combine(img_input, img_gt, minSize=100, maxSize=285)

        img_gt = transforms.ToTensor()(img_gt)
        img_input = transforms.ToTensor()(img_input)
        w_offset = random.randint(0, max(0, 286 - 256 - 1))
        h_offset = random.randint(0, max(0, 286 - 256 - 1))
    
        img_gt = img_gt[:, h_offset:h_offset + 256, w_offset:w_offset + 256]
        img_input = img_input[:, h_offset:h_offset + 256, w_offset:w_offset + 256]
    
        img_gt = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(img_gt)
        img_input = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(img_input)

        if random.random() < 0.5:
            idx = [i for i in range(img_gt.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            img_gt = img_gt.index_select(2, idx)
            img_input = img_input.index_select(2, idx)

        if self.direction == "gt2input":
            return img_gt, img_input
        else:
            return img_input, img_gt
            
    def __len__(self):
        return len(self.image_filenames)
        
'''

      #open the gt and input image
        gt = Image.open(join(self.gt_path, self.image_filenames[index])).convert('RGB')
        input = Image.open(join(self.input_path, self.image_filenames[index])).convert('RGB')
        for i in range(1,31):
      #resize the image
          gt = gt.resize((286, 286), Image.BICUBIC) #resize change to crop (pillow crop)
          input = input.resize((286, 286), Image.BICUBIC) #resize change to crop (pillow crop)
      
      #convert from numpy array to a tensor then convenient to processing e.g normalizaion and cropping or anything else
          trans = transforms.Compose([transforms.ToTensor(),
          transforms.RandomCrop(100)])
          seed = torch.random.seed()
          torch.random.manual_seed(seed)
          cropped_input = trans(input)
          torch.random.manual_seed(seed)
          cropped_gt = trans(gt)

        #random crop >> replace this part
          #w_offset = random.randint(0, max(0, 286 - 256 - 1))
          #h_offset = random.randint(0, max(0, 286 - 256 - 1))
          #gt = gt[:, h_offset:h_offset + 256, w_offset:w_offset + 256]
          #input = input[:, h_offset:h_offset + 256, w_offset:w_offset + 256]
          

        
        #Normalize: convert the pixels values from 0-255 to 0-1
          cropped_gt = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(cropped_gt)
          cropped_input = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(cropped_input)
        
        #use the 50% sample to train
          if random.random() < 0.5:
              idx = [i for i in range(cropped_gt.size(2) - 1, -1, -1)]
              idx = torch.LongTensor(idx)
              cropped_gt = cropped_gt.index_select(2, idx)
              cropped_input = cropped_input.index_select(2, idx)
        
        #portrait to sketch or sketch to portrait 
          if self.direction == "gt2input":
              return cropped_gt, cropped_input
          else:
              return cropped_input, cropped_gt
'''

    

