import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
from loss_func import *
from time import gmtime, strftime

class LeNet(nn.Module):
    def __init__(self, feat_size, train_loader, val_loader, solver_info, cuda=True):
        super(LeNet, self).__init__()
        self.feat_size = feat_size
        self.solver_info = solver_info
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.on_cuda = cuda

        self.conv1_1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.prelu1_1 = nn.PReLU()
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
        self.prelu1_2 = nn.PReLU()
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.prelu2_1 = nn.PReLU()
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.prelu2_2 = nn.PReLU()
        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=5, padding=2)
        self.prelu3_1 = nn.PReLU()
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=5, padding=2)
        self.prelu3_2 = nn.PReLU()
        self.fc1 = nn.Linear(1152, self.feat_size, bias=False)
        self.softmax = AngleSoftmax(self.feat_size, output_size=10, normalize=True)
        self.ringloss = RingLoss(type='auto', loss_weight=1.0)

        if cuda:
            self.cuda()
        self.optimizer = self.get_optimizer(solver_info['lr'], solver_info['momentum'], solver_info['weight_decay'])

    def forward(self, x, y):
        x = self.prelu1_1(self.conv1_1(x))
        x = F.max_pool2d(self.prelu1_2(self.conv1_2(x)), 2)

        x = self.prelu2_1(self.conv2_1(x))
        x = F.max_pool2d(self.prelu2_2(self.conv2_2(x)), 2)

        x = self.prelu3_1(self.conv3_1(x))
        x = F.max_pool2d(self.prelu3_2(self.conv3_2(x)), 2)

        x = x.view(-1, 1152)
        x = self.fc1(x)
        return self.softmax(x, y), self.ringloss(x)

    def get_optimizer(self, lr, momentum, weight_decay):
        params_dict = dict(self.named_parameters())
        params = []
        for key, value in params_dict.items():
            if key[:4] == 'conv':
                if key[-4:] == 'bias': # bias learning rate is 2
                    params += [{'params': [value], 'lr': 2.0 * lr, 'weight_decay': 0.0}]
                else:
                    params += [{'params': [value]}]
            elif key[:2] == 'fc':
                if key[-4:] == 'bias':
                    params += [{'params': [value], 'lr': 0.0, 'weight_decay': 0.0}]
                else:
                    params += [{'params': [value]}]
            elif key[:10] == 'softmax.fc':
                if key[-4:] == 'bias':
                    params += [{'params': [value], 'lr': 2.0, 'weight_decay': 0.0}]
                else:
                    params += [{'params': [value]}]
            elif key[-6:] == 'radius': # ring radius learning rate is 2
                params += [{'params': [value], 'lr': 2.0 * lr, 'weight_decay': 0.0}]
            else:
                params += [{'params': [value]}]
        return optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)

    def train_step(self, epoch, log_interval=100):
        self.train()
        print(strftime("%Y-%m-%d %H:%M:%S", gmtime()) + '\tlearning rate: {}'.format(self.solver_info['lr']))
        for batch_idx, (data, target) in enumerate(self.train_loader):
            if self.on_cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            self.optimizer.zero_grad()
            softmaxloss, ringloss = self(data, target)
            loss = softmaxloss + ringloss
            loss.backward()
            self.optimizer.step()

            if batch_idx % log_interval == 0:
                print(
                    strftime("%Y-%m-%d %H:%M:%S", gmtime())
                    + '\tTrain Epoch: {} [{}/{} ({:.0f}%)]\t'.format(epoch, batch_idx * len(data),
                                                                     len(self.train_loader.dataset),
                                                                     100. * batch_idx / len(self.train_loader))
                    + 'softmax: {:.6f}\t ringloss: {:.6f}'.format(softmaxloss, ringloss))
        if epoch in self.solver_info['stepvalue']:
            self.solver_info['lr'] *= self.solver_info['gamma']
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= self.solver_info['gamma']
        self.val(epoch)

    def val(self, epoch):
        self.eval()
        correct = 0
        n_samples = 0
        softmaxloss = 0
        ringloss = 0
        for batch_idx, (data, target) in enumerate(self.val_loader):
            if self.on_cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)
            sl, rl = self(data, target)
            softmaxloss += sl
            ringloss += rl
            pred = self.softmax.prob.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data).cpu().sum()
            n_samples += len(target)
        print(
            strftime("%Y-%m-%d %H:%M:%S", gmtime())
            + '\tVal Epoch: {}'.format(epoch)
            + '\tAcc: {}\t'.format(float(correct)/float(n_samples))
            + 'softmax: {:.6f}\t ringloss: {:.6f}'.format(softmaxloss/(batch_idx+1), ringloss/(batch_idx+1)))