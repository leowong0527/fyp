from __future__ import print_function
import argparse
import os
from math import log10

from os.path import join
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

from networks import define_G, define_D, GANLoss, get_scheduler, update_learning_rate

from dataset import DatasetFromFolder

dataset = 'portrait_shuffle'
batch_size = 1
test_batch_size = 1
direction = 'input2gt'
input_nc = 3

output_nc = 3
ngf = 64
ndf = 64
epoch_count = 1
niter = 250
niter_decay = 250

learning_rate = 0.0002
lr_policy = 'lambda'
lr_decay_iters = 50
beta1 = 0.5
cuda = 'store_true'

threads = 4
seed = 123
lamb = 10

cudnn.benchmark = True

torch.manual_seed(seed)

print('===> Loading datasets')
root_path = "dataset/"

train_dir = join(root_path + dataset, "train")
train_set = DatasetFromFolder(train_dir, direction)
test_dir = join(root_path + dataset, "test")
test_set = DatasetFromFolder(train_dir, direction)
training_data_loader = DataLoader(dataset=train_set, num_workers=threads, batch_size=batch_size, shuffle=True)
testing_data_loader = DataLoader(dataset=test_set, num_workers=threads, batch_size=test_batch_size, shuffle=False)

device = torch.device("cuda:0" if cuda else "cpu")

print('===> Building models')
net_g = define_G(input_nc, output_nc, ngf, 'batch', False, 'normal', 0.02, gpu_id=device)
net_d = define_D(input_nc + output_nc, ndf, 'basic', gpu_id=device)

criterionGAN = GANLoss().to(device)
criterionL1 = nn.L1Loss().to(device)
criterionMSE = nn.MSELoss().to(device)

# setup optimizer
optimizer_g = optim.Adam(net_g.parameters(), lr=learning_rate, betas=(beta1, 0.999))
optimizer_d = optim.Adam(net_d.parameters(), lr=learning_rate, betas=(beta1, 0.999))
net_g_scheduler = get_scheduler(optimizer_g, lr_policy, epoch_count, niter, niter_decay, lr_decay_iters)
net_d_scheduler = get_scheduler(optimizer_d, lr_policy, epoch_count, niter, niter_decay, lr_decay_iters)

if __name__ == '__main__':
    for epoch in range(epoch_count, niter + niter_decay + 1):
        # train
        for iteration, batch in enumerate(training_data_loader, 1):
            # forward
            real_a, real_b = batch[0].to(device), batch[1].to(device)
            fake_b = net_g(real_a)

            ######################
            # (1) Update D network
            ######################

            optimizer_d.zero_grad()

            # train with fake
            fake_ab = torch.cat((real_a, fake_b), 1)
            pred_fake = net_d.forward(fake_ab.detach())
            loss_d_fake = criterionGAN(pred_fake, False)

            # train with real
            real_ab = torch.cat((real_a, real_b), 1)
            pred_real = net_d.forward(real_ab)
            loss_d_real = criterionGAN(pred_real, True)

            # Combined D loss
            loss_d = (loss_d_fake + loss_d_real) * 0.5

            loss_d.backward()

            optimizer_d.step()

            ######################
            # (2) Update G network
            ######################

            optimizer_g.zero_grad()

            # First, G(A) should fake the discriminator
            fake_ab = torch.cat((real_a, fake_b), 1)
            pred_fake = net_d.forward(fake_ab)
            loss_g_gan = criterionGAN(pred_fake, True)

            # Second, G(A) = B
            loss_g_l1 = criterionL1(fake_b, real_b) * lamb

            loss_g = loss_g_gan + loss_g_l1

            loss_g.backward()

            optimizer_g.step()

            print("===> Epoch[{}]({}/{}): Loss_D: {:.4f} Loss_G: {:.4f}".format(
                epoch, iteration, len(training_data_loader), loss_d.item(), loss_g.item()))

        update_learning_rate(net_g_scheduler, optimizer_g)
        update_learning_rate(net_d_scheduler, optimizer_d)

        #checkpoint
        if epoch % 5 == 0:
            if not os.path.exists("checkpoint"):
                os.mkdir("checkpoint")
            if not os.path.exists(os.path.join("checkpoint", dataset)):
                os.mkdir(os.path.join("checkpoint", dataset))
            net_g_model_out_path = "checkpoint/{}/netG_model_epoch_{}.pth".format(dataset, epoch)
            net_d_model_out_path = "checkpoint/{}/netD_model_epoch_{}.pth".format(dataset, epoch)
            torch.save(net_g, net_g_model_out_path)
            torch.save(net_d, net_d_model_out_path)
            print("Checkpoint saved to {}".format("checkpoint" + dataset))
