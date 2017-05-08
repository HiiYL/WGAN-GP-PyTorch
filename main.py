from __future__ import print_function
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import numpy as np
import os

MODE = 'wgan-gp' # Valid options are dcgan, wgan, or wgan-gp
DIM = 128 # This overfits substantially; you're probably better off with 64
LAMBDA = 10 # Gradient penalty lambda hyperparameter
CRITIC_ITERS = 5 # How many critic iterations per generator iteration
BATCH_SIZE = 64 # Batch size
ITERS = 200000 # How many generator iterations to train for
OUTPUT_DIM = 3072 # Number of pixels in CIFAR10 (3*32*32)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar10 | lsun | imagenet | folder | lfw ')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to network')
parser.add_argument('--nc', type=int, default=3, help='input image channels')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lrD', type=float, default=1e-4, help='learning rate for Critic, default=0.00005')
parser.add_argument('--lrG', type=float, default=1e-4, help='learning rate for Generator, default=0.00005')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--beta2', type=float, default=0.9, help='beta2 for adam. default=0.9')
parser.add_argument('--cuda'  , action='store_true', help='enables cuda')
parser.add_argument('--ngpu'  , type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--clamp_lower', type=float, default=-0.01)
parser.add_argument('--clamp_upper', type=float, default=0.01)
parser.add_argument('--Diters', type=int, default=5, help='number of D iters per each G iter')
parser.add_argument('--noBN', action='store_true', help='use batchnorm or not (only for DCGAN)')
parser.add_argument('--mlp_G', action='store_true', help='use MLP for G')
parser.add_argument('--mlp_D', action='store_true', help='use MLP for D')
parser.add_argument('--n_extra_layers', type=int, default=0, help='Number of extra layers on gen and disc')
parser.add_argument('--experiment', default=None, help='Where to store samples and models')
parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
opt = parser.parse_args()
print(opt)

if opt.experiment is None:
    opt.experiment = 'logs'
os.system('mkdir {0}'.format(opt.experiment))

opt.manualSeed = random.randint(1, 10000) # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

if opt.dataset in ['imagenet', 'folder', 'lfw']:
    # folder dataset
    dataset = dset.ImageFolder(root=opt.dataroot,
                               transform=transforms.Compose([
                                   transforms.Scale(opt.imageSize),
                                   transforms.CenterCrop(opt.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
elif opt.dataset == 'lsun':
    dataset = dset.LSUN(db_path=opt.dataroot, classes=['bedroom_train'],
                        transform=transforms.Compose([
                            transforms.Scale(opt.imageSize),
                            transforms.CenterCrop(opt.imageSize),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))
elif opt.dataset == 'cifar10':
    dataset = dset.CIFAR10(root=opt.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Scale(opt.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])
    )
assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))



class G(nn.Module):
    def __init__(self, h, n, output_dim=(3,64,64)):
        super(G, self).__init__()
        self.n = n
        self.h = h

        channel, width, height = output_dim
        self.blocks = int(np.log2(width) - 2)

        print("[!] {} blocks in G ".format(self.blocks))

        self.fc = nn.Linear(h, 8 * 8 * n)

        conv_layers = []
        for i in range(self.blocks):
            conv_layers.append(nn.Conv2d(n, n, kernel_size=3, stride=1, padding=1))
            conv_layers.append(nn.ELU())
            conv_layers.append(nn.Conv2d(n, n, kernel_size=3, stride=1, padding=1))
            conv_layers.append(nn.ELU())

            if i < self.blocks - 1:
                conv_layers.append(nn.UpsamplingNearest2d(scale_factor=2))

        conv_layers.append( nn.Conv2d(n,channel, kernel_size=3, stride=1, padding=1) )
        self.conv = nn.Sequential(*conv_layers)

        #self.tanh = nn.Tanh()


    def forward(self, x):
        fc_out = self.fc(x).view(-1,self.n,8,8)
        return self.conv(fc_out)


class D(nn.Module):
    def __init__(self, h, n, input_dim=(3, 64,64)):
        super(D, self).__init__()

        self.n = n
        self.h = h

        channel, width, height = input_dim
        self.blocks = int(np.log2(width) - 2)

        print("[!] {} blocks in D ".format(self.blocks))

        encoder_layers = []
        encoder_layers.append(nn.Conv2d(3, n, kernel_size=3, stride=1, padding=1))

        prev_channel_size = n
        for i in range(self.blocks):
            channel_size = ( i + 1 ) * n
            encoder_layers.append(nn.Conv2d(prev_channel_size, channel_size, kernel_size=3, stride=1, padding=1))
            encoder_layers.append(nn.ELU())
            encoder_layers.append(nn.Conv2d(channel_size, channel_size, kernel_size=3, stride=1, padding=1))
            encoder_layers.append(nn.ELU())

            if i < self.blocks - 1:
                # Downsampling
                encoder_layers.append(nn.Conv2d(channel_size, channel_size, kernel_size=3, stride=2, padding=1))
                encoder_layers.append(nn.ELU())

            prev_channel_size = channel_size

        self.encoder = nn.Sequential(*encoder_layers)

        self.fc = nn.Linear(8 * 8 * self.blocks * n, 1)
        #self.tanh = nn.Tanh()


    def forward(self,x):
        #   encoder        [ flatten ] 
        x = self.encoder(x).view(x.size(0), -1)
        x = self.fc(x)


        return x.mean(0).view(1)

nz = 128
dim = 128
netG = G(nz, dim, (3,opt.imageSize,opt.imageSize))
netD = D(nz, dim, (3,opt.imageSize,opt.imageSize))
noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)

alpha = torch.FloatTensor(opt.batchSize,1,1,1).uniform_(0,1)

input = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)

if opt.cuda:
    input = input.cuda()
    netD.cuda()
    netG.cuda()
    noise = noise.cuda()
    alpha = alpha.cuda()


optimizerD = optim.Adam(netD.parameters(), lr=opt.lrD, betas=(opt.beta1, opt.beta2))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lrG, betas=(opt.beta1, opt.beta2))

for i, data in enumerate(dataloader):

    netD.zero_grad()

    noise.resize_(opt.batchSize, nz).normal_(0, 1)
    noisev = Variable(noise)

    images, _ = data
    images = images.cuda()
    input.resize_as_(images).copy_(images)

    real_data = Variable(input)
    fake_data = netG(noisev)

    D_real    = netD(real_data)
    D_fake    = netD(fake_data.detach())

    alpha.uniform_(0, 1)
    #differences = fake_data - real_data
    alpha_ex = alpha.expand(opt.batchSize, real_data.size(1), real_data.size(2), real_data.size(3))
    interpolates = (alpha_ex * real_data.data) + (( 1 - alpha_ex ) * fake_data.data)

    sample = Variable(interpolates, requires_grad=True)
    output = netD(sample)
    torch.autograd.backward(output, create_graph=True)

    gradients = sample.grad.view(opt.batchSize,-1)
    slopes = torch.sum(gradients ** 2, 1).sqrt()
    gradient_penalty = (torch.mean(slopes - 1.) ** 2)


    D_loss = D_fake - D_real + 10 * gradient_penalty

    D_loss.backward()
    optimizerD.step()

    netG.zero_grad()
    D_fake    = netD(fake_data)
    G_loss    = -D_fake
    G_loss.backward()
    optimizerG.step()

    print('[%d/%d]Loss_D: %f Loss_G: %f' % (i, len(dataloader), D_loss.data[0], G_loss.data[0]))

