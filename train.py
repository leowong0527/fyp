from __future__ import print_function
import argparse
import os
from math import log10

import torch
import torch.nn as nn #relate to convolution
import torch.optim as optim #create optimizer
from torch.utils.data import DataLoader #load the data iterative
import torch.backends.cudnn as cudnn 

from networks import define_G, define_D, GANLoss, get_scheduler, update_learning_rate
#from data import get_training_set, get_test_set
from dataset import DatasetFromFolder

#change the variable

# Training settings
#parser = argparse.ArgumentParser(description='pix2pix-pytorch-implementation') #help people easy to read
#parser.add_argument('--dataset', required=True, help='facades')
#parser.add_argument('--batch_size', type=int, default=1, help='training batch size')
#parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')
#parser.add_argument('--direction', type=str, default='b2a', help='a2b or b2a')
#parser.add_argument('--input_nc', type=int, default=3, help='input image channels')
#parser.add_argument('--output_nc', type=int, default=3, help='output image channels')
#parser.add_argument('--ngf', type=int, default=64, help='generator filters in first conv layer')
#parser.add_argument('--ndf', type=int, default=64, help='discriminator filters in first conv layer')
#parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count')
#parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
#parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
#parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
#parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau|cosine')
#parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
#parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
#parser.add_argument('--cuda', action='store_true', help='use cuda?')
#parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
#parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
#parser.add_argument('--lamb', type=int, default=10, help='weight on L1 term in objective')
#opt = parser.parse_args()

dataset = 'facades'
cuda = 'store_true'
batch_size = 1
test_batch_size = 1
input_image_channel = 3
output_image_channel = 3
generator_filters_first_conv_layer = 64
discriminator_filters_conv_layer = 64
starting_epoch_count = 1
number_of_iteration = 100
number_of_iteration_decay = 100
initial_learning_rate_adam = 0.0002
learning_rate_policy = 'lambda'
learning_rate_decay_iteration = 50
adam_beta1 = 0.5
seed = 1
threads = 4
l1_weight = 10
direction = 'b2a'



#print(opt) #predefine variable name

#opt.dataset = "./dataset"

#if cuda and not torch.cuda.is_available():
#    raise Exception("No GPU found, please run without --cuda") #can delete

cudnn.benchmark = True

torch.manual_seed(seed)
#if cuda:
#    torch.cuda.manual_seed(seed) #can delete

print('===> Loading datasets')
#root_path = "dataset/"
#train_set = get_training_set(root_path + dataset, direction) #default b2a that means sratch to real
#test_set = get_test_set(root_path + dataset, direction) #same as train set

root_path_train_set = "dataset/facades/train"
root_path_test_set = "dataset/facades/test"
train_set = DatasetFromFolder(root_path_train_set,direction)
test_set = DatasetFromFolder(root_path_test_set,direction)

#print('train set:', train_set)
#print('test set:', test_set)
training_data_loader = DataLoader(dataset=train_set, num_workers=threads, batch_size=batch_size, shuffle=True)
testing_data_loader = DataLoader(dataset=test_set, num_workers=threads, batch_size=test_batch_size, shuffle=False)

#num_workers = greater than 0 turn on multiprocessing? the more the better? depends on gpu?
#batch size use stochastic mode because size = 1. smaller batch size prevent over-fitting and better learning result
#easy to fit in memory
#shuffle = random the dataset

device = torch.device("cuda:0" if cuda else "cpu") #choose cpu/gpu

print('===> Building models')
net_g = define_G(input_image_channel, output_image_channel, generator_filters_first_conv_layer, 'batch', False, 'normal', 0.02, gpu_id=device)
net_d = define_D(input_image_channel + output_image_channel, discriminator_filters_conv_layer, 'basic', gpu_id=device)
#input/output_nc image channel = RGB = 3
#ngf/ndf = 64 filters in convolution layers = filter size? usually odd but this one is even?
#batch refers to networks.norm = number matrix?
#False refers to networks.dropout but dropout is good why set false?
#dropot reduce overfitting by reducing weights?
#initial type normal & gain? why does it means?



criterionGAN = GANLoss().to(device)
criterionL1 = nn.L1Loss().to(device)
criterionMSE = nn.MSELoss().to(device)

#total_loss = criterionGAN(x,y)+0.5

# setup optimizer backpropagation
optimizer_g = optim.Adam(net_g.parameters(), initial_learning_rate_adam, betas=(adam_beta1, 0.999))
optimizer_d = optim.Adam(net_d.parameters(), initial_learning_rate_adam, betas=(adam_beta1, 0.999))
net_g_scheduler = get_scheduler(optimizer_g, learning_rate_policy, starting_epoch_count, number_of_iteration, learning_rate_decay_iteration) #change opt to all variable
net_d_scheduler = get_scheduler(optimizer_d, learning_rate_policy, starting_epoch_count, number_of_iteration, learning_rate_decay_iteration)

if __name__ == '__main__':
    for epoch in range(starting_epoch_count, number_of_iteration + number_of_iteration_decay + 1):
        # train    
        for iteration, batch in enumerate(training_data_loader, 1):
            # forward
            real_a, real_b = batch[0].to(device), batch[1].to(device) #put in the gpu memory
            fake_b = net_g(real_a) #net_g neural = network, real_a = input file

            #######################
            # (1) Update D network#
            #######################

            optimizer_d.zero_grad() #initialize

            #b is sctrach while a is real image
            #fake_b = use a dataset to generate generated real image???

            # train with fake
            fake_ab = torch.cat((real_a, fake_b), 1) #concat the real image and generated real image together
            pred_fake = net_d.forward(fake_ab.detach()) #pred_fake = net_d(fake_ab)
            loss_d_fake = criterionGAN(pred_fake, False) #true,false means turn on/off gpu

            # train with real
            real_ab = torch.cat((real_a, real_b), 1) #concat both real image and real scratch together
            pred_real = net_d.forward(real_ab) #pass the dataset to the discriminator?
            loss_d_real = criterionGAN(pred_real, True) #calculate the mse loss? other one is bce loss bce = binary cross entropy
            
            # Combined D loss
            loss_d = (loss_d_fake + loss_d_real) * 0.5 #why add up both mse * 0.5?? add weight to tune

            loss_d.backward() #try to tune parameters from the back?
           
            optimizer_d.step() #perform update single parameter of discriminator

            ######################
            # (2) Update G network
            ######################
            #both can swap

            optimizer_g.zero_grad() #set the gradient to zero otherwise it will update the gradient with the old value not point to maximum or minimum #initialize

            # First, G(A) should fake the discriminator
            fake_ab = torch.cat((real_a, fake_b), 1) # a is real real image while b is generated scratch image and concat them
            pred_fake = net_d.forward(fake_ab) #forward concat ab to the discriminator and train
            loss_g_gan = criterionGAN(pred_fake, True) #calculate mse loss

            # Second, G(A) = B
            loss_g_l1 = criterionL1(fake_b, real_b) * l1_weight #lamb = weight L1 = least absolute deviations
            
            loss_g = loss_g_gan + loss_g_l1 #mse loss+least absolute deviations
            
            loss_g.backward() #indciate to tune parameters from the back

            optimizer_g.step() #perform update single parameter of generator

            print("===> Epoch[{}]({}/{}): Loss_D: {:.4f} Loss_G: {:.4f}".format(
                epoch, iteration, len(training_data_loader), loss_d.item(), loss_g.item()))

        update_learning_rate(net_g_scheduler, optimizer_g) #scheduler for speed up the process
        update_learning_rate(net_d_scheduler, optimizer_d) #adjust the learning rate

        # test
        avg_psnr = 0 #peak signal to noise ratio = show the difference of true and noise(false) image part with mathematics
        for batch in testing_data_loader:
            input, target = batch[0].to(device), batch[1].to(device)

            prediction = net_g(input)
            mse = criterionMSE(prediction, target)
            psnr = 10 * log10(1 / mse.item())
            avg_psnr += psnr
        print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))

        #checkpoint
        if epoch % 50 == 0: #every 50 dataset training save as a checkpoint #5 will be better
            if not os.path.exists("checkpoint"):
                os.mkdir("checkpoint")
            if not os.path.exists(os.path.join("checkpoint", dataset)):
                os.mkdir(os.path.join("checkpoint", dataset))
            net_g_model_out_path = "checkpoint/{}/netG_model_epoch_{}.pth".format(dataset, epoch)
            net_d_model_out_path = "checkpoint/{}/netD_model_epoch_{}.pth".format(dataset, epoch)
            torch.save(net_g, net_g_model_out_path)
            torch.save(net_d, net_d_model_out_path)
            print("Checkpoint saved to {}".format("checkpoint" + dataset)) #for checking whether have valid improvement
