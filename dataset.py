from os import listdir
from os.path import join
import random

from imageio import *
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from random_crop import random_crop
from random_combine import random_combine

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
        img_gt = Image.open(join(self.gt_path, self.image_filenames[index]))
        img_gt = img_gt.convert('RGB')
        img_input = Image.open(join(self.input_path, self.image_filenames[index]))
        img_input = img_input.convert('RGB')

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