from __future__ import print_function
import argparse
import os

import torch
import torchvision.transforms as transforms
import pandas as pd
import numpy as np

from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from PIL import Image
from random_crop import random_crop

torch.manual_seed(123)
dataset = 'portrait_shuffle'
direction = 'input2gt'
nepochs = 340
cuda = 'store_true'

device = torch.device("cuda:0" if cuda else "cpu")

model_path = "checkpoint/{}/netG_model_epoch_{}.pth".format(dataset, nepochs)

net_g = torch.load(model_path).to(device)

image_dir_gt = "dataset/portrait_shuffle/new_test/gt/"
image_dir_input = "dataset/portrait_shuffle/new_test/input/"

image_filenames_gt = [x for x in os.listdir(image_dir_gt)]
image_filenames = [x for x in os.listdir(image_dir_input)]

transform_list = [transforms.ToTensor(),
                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

transform = transforms.Compose(transform_list)

for image_name in image_filenames:
    img_input = Image.open(image_dir_input + image_name).convert('RGB')
    img_gt = Image.open(image_dir_gt + image_name).convert('RGB')
    img_input = transform(img_input)
    input = img_input.unsqueeze(0).to(device)
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




























