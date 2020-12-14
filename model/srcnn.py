import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from model.base_networks import *
import skimage as sk
import math

# import pytorch_ssim as ps
from torch.autograd import Variable
from skimage.measure import compare_psnr, compare_ssim


class ConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=4, stride=2, padding=1, bias=True, activation='relu', norm='batch'):
        super(ConvBlock, self).__init__()
        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm =='batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.conv(x))
        else:
            out = self.conv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out

class Net(nn.Module):

    def __init__(self, params):
        super(Net, self).__init__()
        self.num_channels = 64
        # self.dropout_rate = params.dropout_rate

        self.layers = torch.nn.Sequential(
            ConvBlock(3, self.num_channels, 9, 1, 4, norm=None),  # 144*144*64 # conv->batchnorm->activation
            ConvBlock(self.num_channels, self.num_channels // 2, 5, 1, 2, norm=None),  # 144*144*32
            ConvBlock(self.num_channels // 2, 3, 5, 1, 2, activation=None, norm=None)  # 144*144*1
        )

    def forward(self, s):
        out = F.interpolate(s, scale_factor=4, mode='bicubic')
        out = self.layers(out)
        return out


def loss_fn(outputs, labels):
    N, C, H, W = outputs.shape

    mse_loss = torch.sum((outputs - labels) ** 2) / N / C  # each photo, each channel
    mse_loss *= 255 * 255
    mse_loss /= H * W
    # average loss on each pixel(0-255)
    return mse_loss


def accuracy(outputs, labels):
    N, _, _, _ = outputs.shape
    psnr = 0
    for i in range(N):
        psnr += compare_psnr(labels[i], outputs[i])
    return psnr / N



def ssim(outputs, labels):
    N, _, _, _ = outputs.shape
    ssim = 0
    for i in range(N):
        ssim += compare_ssim(labels[i], outputs[i], win_size=3, multichannel=True)
    return ssim / N



# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    'PSNR': accuracy,
    'SSIM': ssim,
    # could add more metrics such as accuracy for each token type
}
