import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from model.base_networks import *
import math
# import pytorch_ssim as ps
from torch.autograd import Variable

import torch
import torch.nn as nn
from math import sqrt


# class Net(torch.nn.Module):
#     def __init__(self, args, n_channels=3, d=56, s=12, m=4):
#         # too big network may leads to over-fitting
#         super(Net, self).__init__()
#
#         # Feature extraction
#         self.first_part = nn.Sequential(
#             nn.Conv2d(in_channels=n_channels, out_channels=d, kernel_size=5, stride=1, padding=0),
#             nn.PReLU())
#         # H_out = floor((H_in+2*padding-(kernal_size-1)-1)/stride+1)
#         #       = floor(H_in-4)
#         # for x2  floor(H_in-2)
#         self.layers = []
#         # Shrinking
#         self.layers.append(nn.Sequential(nn.Conv2d(in_channels=d, out_channels=s, kernel_size=1, stride=1, padding=0),
#                                          nn.PReLU()))
#
#         # Non-linear Mapping
#         for _ in range(m):
#             self.layers.append(
#                 nn.Sequential(nn.Conv2d(in_channels=s, out_channels=s, kernel_size=3, stride=1, padding=1),
#                               nn.PReLU()))
#
#         # # Expanding
#         self.layers.append(nn.Sequential(nn.Conv2d(in_channels=s, out_channels=d, kernel_size=1, stride=1, padding=0),
#                                          nn.PReLU()))
#
#         self.mid_part = torch.nn.Sequential(*self.layers)
#
#         # Deconvolution
#         self.last_part = nn.ConvTranspose2d(in_channels=d, out_channels=n_channels, kernel_size=9, stride=3, padding=4,
#                                             output_padding=0)
#         # self.last_part = nn.Sequential(
#         #     nn.Conv2d(in_channels=d, out_channels=n_channels * 2 * 2, kernel_size=3, stride=1, padding=1),
#         #     nn.PixelShuffle(2))
#         # H_out = (H_in-1)*stride-2*padding+kernal_size+out_padding
#         #       = (H_in-1)*3+1
#         # test input should be (y-5)*3+1
#         # for x2 2x-3
#         # for x4 4x-25
#
#     def forward(self, x):
#         out = self.first_part(x)
#         out = self.mid_part(out)
#         out = self.last_part(out)
#         return out
#
#     def weight_init(self):
#         """
#         Initial the weights.
#         """
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
#                 # m.weight.data.normal_(0.0, 0.2)
#                 m.weight.data.normal_(0.0, sqrt(2 / m.out_channels / m.kernel_size[0] / m.kernel_size[0]))  # MSRA
#                 # nn.init.xavier_normal(m.weight) # Xavier
#                 if m.bias is not None:
#                     m.bias.data.zero_()

class Net(nn.Module):

    def __init__(self, params):
        super(Net, self).__init__()
        # self.num_channels = params.num_channels
        # self.dropout_rate = params.dropout_rate

        self.layers = torch.nn.Sequential(
            nn.Conv2d(3, 56, 5, padding=2),  # 72
            # nn.InstanceNorm2d(56),
            nn.PReLU(),
            nn.Conv2d(56, 12, 1),
            # nn.InstanceNorm2d(12),
            nn.PReLU(),
            nn.Conv2d(12, 12, 3, padding=1),
            # nn.InstanceNorm2d(12),
            nn.PReLU(),
            nn.Conv2d(12, 12, 3, padding=1),
            # nn.InstanceNorm2d(12),
            nn.PReLU(),
            nn.Conv2d(12, 12, 3, padding=1),
            # nn.InstanceNorm2d(12),
            nn.PReLU(),
            nn.Conv2d(12, 12, 3, padding=1),
            # nn.InstanceNorm2d(12),
            nn.PReLU(),
            nn.Conv2d(12, 56, 1),  # 72
            # nn.InstanceNorm2d(12),
            nn.PReLU(),
            nn.ConvTranspose2d(56, 3, 9, stride=4, output_padding=1, padding=3)  # (72-1) * 2 + 9 - 8 + 1 = 144
            # stride = 2   padding=4  changed!
        )

    def forward(self, s):
        out = self.layers(s)
        return out


def loss_fn(outputs, labels):
    N, C, H, W = outputs.shape

    mse_loss = torch.sum((outputs - labels) ** 2) / N / C  # each photo, each channel
    mse_loss *= 255 * 255
    mse_loss /= H * W
    # average loss on each pixel(0-255)
    return mse_loss


def accuracy(outputs, labels):
    N, C, H, W = outputs.shape

    nume = np.max(outputs, axis=(1, 2, 3), keepdims=True)  # (N,)
    deno = np.sum((outputs.reshape(-1, 3, 144, 144) - labels.reshape(-1, 3, 144, 144)) ** 2, axis=(1, 2, 3),
                  keepdims=True) / C
    deno *= 255 * 255 / H / W  # (N,)  range from 0-255, pixel avg

    psnr = (nume * 255) ** 2 / deno  # (N,)
    psnr = np.log(psnr)
    psnr = 10 * np.sum(psnr)
    psnr /= math.log(10) * N

    return psnr


def ssim(outputs, labels):
    if torch.cuda.is_available():
        outputs = Variable(torch.from_numpy(outputs)).cuda()
        labels = Variable(torch.from_numpy(labels)).cuda()

    ssim = ps.ssim(outputs, labels)

    return ssim


# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    'PSNR': accuracy,
    'SSIM': ssim,
    # could add more metrics such as accuracy for each token type
}