from __future__ import print_function
import argparse
import os

import torch
import torchvision.transforms as transforms

import numpy as np

#from utils import is_image_file, load_img, save_img

from PIL import Image

from random_crop import random_crop

# Testing settings
#parser = argparse.ArgumentParser(description='pix2pix-pytorch-implementation')
#parser.add_argument('--dataset', required=True, help='facades')
#parser.add_argument('--direction', type=str, default='b2a', help='a2b or b2a')
#parser.add_argument('--nepochs', type=int, default=4, help='saved model of which epochs')
#parser.add_argument('--cuda', action='store_true', help='use cuda')
#opt = parser.parse_args()
#print(opt)

dataset = 'portrait'
direction = 'input2gt'
nepochs = 450
cuda = 'store_true'

device = torch.device("cuda:0" if cuda else "cpu")

model_path = "checkpoint/{}/netG_model_epoch_{}.pth".format(dataset, nepochs)

net_g = torch.load(model_path).to(device)

if direction == "gt2input":
    image_dir = "dataset/{}/test/gt/".format(dataset)
else:
    image_dir = "dataset/{}/test/input/".format(dataset)

image_filenames = [x for x in os.listdir(image_dir)]

transform_list = [transforms.ToTensor(),
                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

transform = transforms.Compose(transform_list)

for image_name in image_filenames:
    img = Image.open(image_dir + image_name).convert('RGB')
    ##img = img.resize((256, 256), Image.BICUBIC)
    img, img = random_crop(img, img, 256)
    img = transform(img)
    input = img.unsqueeze(0).to(device)
    out = net_g(input)
    out_img = out.detach().squeeze(0).cpu()

    if not os.path.exists(os.path.join("result", dataset)):
        os.makedirs(os.path.join("result", dataset))
    filename = "result/{}/{}".format(dataset, image_name)
    image_numpy = out_img.float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    image_numpy = image_numpy.clip(0, 255)
    image_numpy = image_numpy.astype(np.uint8)
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(filename)
    print("Image saved as {}".format(filename))