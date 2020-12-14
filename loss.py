from torchvision.models.vgg import vgg16
from model import common
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class PerceptionLoss(nn.Module):
    def __init__(self):
        super(PerceptionLoss, self).__init__()

        # vgg = vgg16(pretrained=True)
        # loss_network = nn.Sequential(*list(vgg.features)[:29]).eval()
        # for param in loss_network.parameters():
        #     param.requires_grad = False
        # self.loss_network = loss_network
        # self.l1_loss = nn.L1Loss()
        # self.upsampler = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=True)
        vgg19 = models.vgg19(pretrained=False)
        vgg19.load_state_dict(torch.load('vgg19.pth'))
        vgg_features = vgg19.features
        modules = [m for m in vgg_features]
        # if conv_index.find('22') >= 0:
        self.vgg1 = nn.Sequential(*modules[:8])
        # elif conv_index.find('54') >= 0:
        self.vgg2 = nn.Sequential(*modules[:35])
        rgb_range = 255
        vgg_mean = (0.485, 0.456, 0.406)
        vgg_std = (0.229 * rgb_range, 0.224 * rgb_range, 0.225 * rgb_range)
        self.sub_mean = common.MeanShift(rgb_range, vgg_mean, vgg_std)
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, sr, hr):
        # high_resolution = self.upsampler(high_resolution)
        # fake_high_resolution = self.upsampler(fake_high_resolution)
        # perception_loss = self.l1_loss(self.loss_network(fake_high_resolution), self.loss_network(high_resolution))
        # return perception_loss
        def _forward(x):
            x = self.sub_mean(x)
            x1 = self.vgg1(x)
            x2 = self.vgg2(x)
            return x1, x2

        vgg_sr1,vgg_sr2 = _forward(sr)
        with torch.no_grad():
            vgg_hr1, vgg_hr2 = _forward(hr.detach())

        loss = F.mse_loss(vgg_sr1, vgg_hr1) * 0.2 + F.mse_loss(vgg_sr2, vgg_hr2)

        return loss


class VGG(nn.Module):
    def __init__(self, conv_index='54', rgb_range=1):
        super(VGG, self).__init__()
        # vgg_features = models.vgg19(pretrained=True).features
        vgg19 = models.vgg19(pretrained=False)
        vgg19.load_state_dict(torch.load('vgg19.pth'))
        vgg_features = vgg19.features

        modules = [m for m in vgg_features]
        if conv_index.find('22') >= 0:
            self.vgg = nn.Sequential(*modules[:8])
        elif conv_index.find('54') >= 0:
            self.vgg = nn.Sequential(*modules[:35])

        vgg_mean = (0.485, 0.456, 0.406)
        vgg_std = (0.229 * rgb_range, 0.224 * rgb_range, 0.225 * rgb_range)
        self.sub_mean = common.MeanShift(rgb_range, vgg_mean, vgg_std)
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, sr, hr):
        def _forward(x):
            x = self.sub_mean(x)
            x = self.vgg(x)
            return x

        vgg_sr = _forward(sr)
        with torch.no_grad():
            vgg_hr = _forward(hr.detach())

        loss = F.mse_loss(vgg_sr, vgg_hr)

        return loss