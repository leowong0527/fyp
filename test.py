from __future__ import print_function
import argparse
import os
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image


#from utils import is_image_file, load_img, save_img

# Testing settings
#parser = argparse.ArgumentParser(description='pix2pix-pytorch-implementation')
#parser.add_argument('--dataset', required=True, help='facades')
#parser.add_argument('--direction', type=str, default='b2a', help='a2b or b2a')
#parser.add_argument('--nepochs', type=int, default=200, help='saved model of which epochs')
#parser.add_argument('--cuda', action='store_true', help='use cuda')
#opt = parser.parse_args()
#print(opt)


dataset = 'facades'
direction = 'b2a'
save_model_which_epochs = 200
cuda = 'store_true'

device = torch.device("cuda:0" if cuda else "cpu")

model_path = "checkpoint/{}/netG_model_epoch_{}.pth".format(dataset, save_model_which_epochs)

net_g = torch.load(model_path).to(device)

if direction == "a2b":
    image_dir = "dataset/{}/test/a/".format(dataset)
else:
    image_dir = "dataset/{}/test/b/".format(dataset)

image_filenames = [x for x in os.listdir(image_dir)]

transform_list = [transforms.ToTensor(),
                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))] #transform the image from 255 to -1.0~1.0 range

transform = transforms.Compose(transform_list)

for image_name in image_filenames:
    img = Image.open(image_dir + image_name).convert('RGB') #load_image function
    img = img.resize((256, 256), Image.BICUBIC)
    #img = load_img(image_dir + image_name)

    img = transform(img)
    input = img.unsqueeze(0).to(device)
    out = net_g(input)
    out_img = out.detach().squeeze(0).cpu()

    if not os.path.exists(os.path.join("result", dataset)):
        os.makedirs(os.path.join("result", dataset))

    image_numpy = out_img.float().numpy() #save_image function
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    image_numpy = image_numpy.clip(0, 255)
    image_numpy = image_numpy.astype(np.uint8)  # change to integer
    image_pil = Image.fromarray(image_numpy)
    image_pil.save("result/{}/{}".format(dataset, image_name))
    print("Image saved as {}".format("result/{}/{}".format(dataset, image_name)))
    #save_img(out_img, "result/{}/{}".format(dataset, image_name))
