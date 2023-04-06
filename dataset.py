from os import listdir
from os.path import join
import random

import imageio.core.functions
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

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
        gt = Image.open(join(self.gt_path, self.image_filenames[index])).convert('RGB')
        input = Image.open(join(self.input_path, self.image_filenames[index])).convert('RGB')

        gt = gt.resize((286, 286), Image.BICUBIC) #resize change to crop (pillow crop)
        input = input.resize((286, 286), Image.BICUBIC) #resize change to crop (pillow crop)
        #import random
        #random.randint
        #left = 5
        #top = height / 4
        #right = 164
        #bottom = 3 * height / 4

        #def random_crop(im, square_size=268):
        #    width, height = im.size
        #    left = random.randint(0,width - square_size)
        #    top = random.randint(0,height - square_size)
        #    right = left +square_size
        #    bottom = top +square_size

        gt = transforms.ToTensor()(gt)
        input = transforms.ToTensor()(input)
        w_offset = random.randint(0, max(0, 286 - 256 - 1))
        h_offset = random.randint(0, max(0, 286 - 256 - 1))
    
        gt = gt[:, h_offset:h_offset + 256, w_offset:w_offset + 256]
        input = input[:, h_offset:h_offset + 256, w_offset:w_offset + 256]
    
        gt = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(gt)
        input = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(input)

        if random.random() < 0.5:
            idx = [i for i in range(gt.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            gt = gt.index_select(2, idx)
            input = input.index_select(2, idx)

        if self.direction == "gt2input":
            return gt, input
        else:
            return input, gt

    def __len__(self):
        return len(self.image_filenames)
