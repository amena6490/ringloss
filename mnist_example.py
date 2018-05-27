from __future__ import print_function

import argparse
import os
import numpy as np
import torch
from torchvision.datasets import MNIST
from torchvision import transforms
from net import LeNet

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size',         type=int, default=300, help='input batch size for training (default: 300)')
parser.add_argument('--epochs',             type=int, default=80, help='number of epochs to train (default: 80)')
parser.add_argument('--lr',                 type=float, default=0.01, help='learning rate (default: 0.01)')
parser.add_argument('--momentum',           type=float, default=0.9, help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay',       type=float, default=0.0005, help='SGD weight decay (default: 0.0005)')
parser.add_argument('--gamma',              type=float, default=0.8, help='decreasing learning rate by gamma (default: 0.8)')
parser.add_argument('--stepvalue',          type=int, default=[40, 70, 80], nargs='+', help='epochs specified to decrease learning rate')
parser.add_argument('--no-cuda',            action='store_true', default=False, help='enables CUDA training')
parser.add_argument('--seed',               type=int, default=1, help='random seed (default: 1)')
parser.add_argument('--feat-size',          type=int, default=5, help='how long is the extracted feature vector, set 2 if you want to visualize')
parser.add_argument('--modelname',          type=str, default='ringloss', help='the additional name to specify a model')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

savedir = 'model/{}'.format(args.modelname)
if not os.path.isdir(savedir):
    os.makedirs(savedir)

solver_info = {'lr': args.lr, 'momentum': args.momentum, 'weight_decay': args.weight_decay, 'gamma': args.gamma,
               'stepvalue': args.stepvalue}

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    MNIST('data', train=True, download=True, transform=transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
val_loader = torch.utils.data.DataLoader(
    MNIST('data', train=False, transform=transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])),
    batch_size=args.batch_size, shuffle=True, **kwargs)

# Initialize Model
train_lenet = LeNet(args.feat_size, train_loader, val_loader, solver_info, cuda=args.cuda)

for epoch in range(1, args.epochs + 1):
    train_lenet.train_step(epoch, log_interval=10)
    if epoch % 10 == 0:
        model_dict = train_lenet.model.state_dict()
        model_dict.update(train_lenet.loss.state_dict())
        torch.save(model_dict, '{}/featlength-{}_rand-{}_epoch-{}'.format(savedir, args.feat_size, args.seed, epoch))
