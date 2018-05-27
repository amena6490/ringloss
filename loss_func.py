import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Function
import numpy as np

class SoftmaxLoss(nn.Module):
    def __init__(self, input_size, output_size, normalize=True, loss_weight=1.0):
        super(SoftmaxLoss, self).__init__()
        self.fc = nn.Linear(input_size, int(output_size), bias=False)
        nn.init.kaiming_uniform_(self.fc.weight, 0.25)
        self.weight = self.fc.weight
        self.loss_weight = loss_weight
        self.normalize = normalize

    def forward(self, x, y):
        if self.normalize:
            self.fc.weight.renorm(2, 1, 1e-5).mul(1e5)
        prob = F.log_softmax(self.fc(x), dim=1)
        self.prob = prob
        loss = F.nll_loss(prob, y)
        return loss.mul_(self.loss_weight)


class RingLoss(nn.Module):
    def __init__(self, type='auto', loss_weight=1.0):
        """
        :param type: type of loss ('l1', 'l2', 'auto')
        :param loss_weight: weight of loss, for 'l1' and 'l2', try with 0.01. For 'auto', try with 1.0.
        :return:
        """
        super(RingLoss, self).__init__()
        self.radius = Parameter(torch.Tensor(1))
        self.radius.data.fill_(-1)
        self.loss_weight = loss_weight
        self.type = type

    def forward(self, x):
        x = x.pow(2).sum(dim=1).pow(0.5)
        if self.radius.data[0] < 0: # Initialize the radius with the mean feature norm of first iteration
            self.radius.data.fill_(x.mean().data)
        if self.type == 'l1': # Smooth L1 Loss
            loss1 = F.smooth_l1_loss(x, self.radius.expand_as(x)).mul_(self.loss_weight)
            loss2 = F.smooth_l1_loss(self.radius.expand_as(x), x).mul_(self.loss_weight)
            ringloss = loss1 + loss2
        elif self.type == 'auto': # Divide the L2 Loss by the feature's own norm
            diff = x.sub(self.radius.expand_as(x)) / (x.mean().detach().clamp(min=0.5))
            diff_sq = torch.pow(torch.abs(diff), 2).mean()
            ringloss = diff_sq.mul_(self.loss_weight)
        else: # L2 Loss, if not specified
            diff = x.sub(self.radius.expand_as(x))
            diff_sq = torch.pow(torch.abs(diff), 2).mean()
            ringloss = diff_sq.mul_(self.loss_weight)
        return ringloss


class AngleSoftmax(nn.Module):
    def __init__(self, input_size, output_size, normalize=True, m=4, lambda_max=1000.0, lambda_min=5.0,
                 power=1.0, gamma=0.1, loss_weight=1.0):
        """
        :param input_size: Input channel size.
        :param output_size: Number of Class.
        :param normalize: Whether do weight normalization.
        :param m: An integer, specifying the margin type, take value of [0,1,2,3,4,5].
        :param lambda_max: Starting value for lambda.
        :param lambda_min: Minimum value for lambda.
        :param power: Decreasing strategy for lambda.
        :param gamma: Decreasing strategy for lambda.
        :param loss_weight: Loss weight for this loss.
        """
        super(AngleSoftmax, self).__init__()
        self.loss_weight = loss_weight
        self.normalize = normalize
        self.weight = Parameter(torch.Tensor(int(output_size), input_size))
        nn.init.kaiming_uniform_(self.weight, 1.0)
        self.m = m

        self.it = 0
        self.LambdaMin = lambda_min
        self.LambdaMax = lambda_max
        self.gamma = gamma
        self.power = power

    def forward(self, x, y):

        if self.normalize:
            wl = self.weight.pow(2).sum(1).pow(0.5)
            wn = self.weight / wl.view(-1, 1)
            self.weight.data.copy_(wn.data)
        if self.training:
            lamb = max(self.LambdaMin, self.LambdaMax / (1 + self.gamma * self.it)**self.power)
            self.it += 1
            phi_kernel = PhiKernel(self.m, lamb)
            feat = phi_kernel(x, self.weight, y)
            loss = F.nll_loss(F.log_softmax(feat, dim=1), y)
        else:
            feat = x.mm(self.weight.t())
            self.prob = F.log_softmax(feat, dim=1)
            loss = F.nll_loss(self.prob, y)

        return loss.mul_(self.loss_weight)

class PhiKernel(Function):
    def __init__(self, m, lamb):
        self.mlambda = [
            lambda x: x ** 0,
            lambda x: x,
            lambda x: 2 * x ** 2 - 1,
            lambda x: 4 * x ** 3 - 3 * x,
            lambda x: 8 * x ** 4 - 8 * x ** 2 + 1,
            lambda x: 16 * x ** 5 - 20 * x ** 3 + 5 * x
        ]
        self.mcoeff_w = [
            lambda x: 0,
            lambda x: 1,
            lambda x: 4 * x,
            lambda x: 12 * x ** 2 - 3,
            lambda x: 32 * x ** 3 - 16 * x,
            lambda x: 80 * x ** 4 - 60 * x ** 2 + 5
        ]
        self.mcoeff_x = [
            lambda x: -1,
            lambda x: 0,
            lambda x: 2 * x ** 2 + 1,
            lambda x: 8 * x ** 3,
            lambda x: 24 * x ** 4 - 8 * x ** 2 - 1,
            lambda x: 64 * x ** 5 - 40 * x ** 3
        ]
        self.m = m
        self.lamb = lamb

    def forward(self, input, weight, label):
        xlen = input.pow(2).sum(1).pow(0.5).view(-1, 1)  # size=B
        wlen = weight.pow(2).sum(1).pow(0.5).view(1, -1)
        cos_theta = (input.mm(weight.t()) / xlen / wlen).clamp(-1, 1)
        cos_m_theta = self.mlambda[self.m](cos_theta)

        k = (self.m * cos_theta.acos() / np.pi).floor()
        phi_theta = (-1) ** k * cos_m_theta - 2 * k

        feat = cos_theta * xlen
        phi_theta = phi_theta * xlen

        index = (feat * 0.0).scatter_(1, label.view(-1, 1), 1).byte()
        feat[index] -= feat[index] / (1.0 + self.lamb)
        feat[index] += phi_theta[index] / (1.0 + self.lamb)
        self.save_for_backward(input, weight, label)
        self.cos_theta = (cos_theta * index.float()).sum(dim=1).view(-1, 1)
        self.k = (k * index.float()).sum(dim=1).view(-1, 1)
        self.xlen, self.index = xlen, index
        return feat

    def backward(self, grad_outputs):
        input, weight, label = self.saved_variables
        input, weight, label = input.data, weight.data, label.data
        grad_input = grad_weight = None
        grad_input = grad_outputs.mm(weight) * self.lamb / (1.0 + self.lamb)
        grad_outputs_label = (grad_outputs*self.index.float()).sum(dim=1).view(-1, 1)
        coeff_w = ((-1) ** self.k) * self.mcoeff_w[self.m](self.cos_theta)
        coeff_x = (((-1) ** self.k) * self.mcoeff_x[self.m](self.cos_theta) + 2 * self.k) / self.xlen
        coeff_norm = (coeff_w.pow(2) + coeff_x.pow(2)).pow(0.5)
        grad_input += grad_outputs_label * (coeff_w / coeff_norm) * torch.index_select(weight, 0, label) / (1.0 + self.lamb)
        grad_input -= grad_outputs_label * (coeff_x / coeff_norm) * input / (1.0 + self.lamb)
        grad_input += (grad_outputs * (1 - self.index).float()).mm(weight) / (1.0 + self.lamb)
        grad_weight = grad_outputs.t().mm(input)
        return grad_input, grad_weight, None