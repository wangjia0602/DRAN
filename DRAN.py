from model import common
import torch.nn as nn

import torch.nn
from model.common import *

class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y + x

class SALayer(nn.Module):
    def __init__(self, kernel_size=7):
        super(SALayer, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        # self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.conv_du = nn.Sequential(
                nn.Conv2d(1, 1, kernel_size, padding=padding, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(1, 1, kernel_size, padding=padding, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        # max_out, _ = torch.max(x, dim=1, keepdim=True)
        # x = torch.cat([avg_out, max_out], dim=1)
        y = self.conv_du(avg_out)
        return x * y + x



class RG(nn.Module):
    def __init__(self, conv, n_feat, kernel_size=3, n_resblocks=10):
        super(RG, self).__init__()
        module_body = [ResBlock(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1)for _ in range(n_resblocks)]

        module_body.append(conv(n_feat, n_feat, kernel_size))
        self.modules_body = nn.Sequential(*module_body)
        # self.ca = ChannelAttention(in_planes=n_feat)
        # self.sa = SpatialAttention()
        # self.ca = CAM_Module(n_feat)
        # self.sa = PAM_Module(n_feat)

    def forward(self, x):
        residual = x
        res = self.modules_body(x)

        res = 0.2 * res + residual

        return res



class DRAN(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(DRAN, self).__init__()

        n_feat = args.n_feats
        self.n_blocks = args.n_resblocks
        self.n_resgroups = args.n_resgroups

        kernel_size = 3
        scale = args.scale
        act = nn.ReLU(True)

        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = MeanShift(255.0, rgb_mean, rgb_std)

        # define head module
        modules_head = [conv(args.n_colors, n_feat, kernel_size)]
        # modules_head = [nn.Conv2d(args.n_colors, n_feat, 5, 1, 2)]

        # modules_head_2 = [conv(n_feat, n_feat, kernel_size)]
        self.conv1 = nn.Conv2d(n_feat, n_feat, kernel_size=1, stride=1, padding=0, bias=True),

        # define body module
        self.modules_body = nn.ModuleList(
            [RG(conv, n_feat, kernel_size, self.n_blocks) for _ in range(self.n_resgroups)])

        modules_tail = [
            conv(n_feat, n_feat, kernel_size),
            common.Upsampler(conv, scale, n_feat, act=False)
        ]
        self.conv = conv(n_feat, args.n_colors, kernel_size)
        self.add_mean = MeanShift(255, rgb_mean, rgb_std, 1)

        self.head_1 = nn.Sequential(*modules_head)
        # self.head_2 = nn.Sequential(*modules_head_2)
        self.fusion = nn.Sequential(*[nn.Conv2d(n_feat * 2, n_feat, 1, padding=0, stride=1)])
        self.fusion_end = nn.Sequential(*[nn.Conv2d(n_feat * self.n_resgroups, n_feat, 1, padding=0, stride=1)])

        # self.body = nn.Sequential(*self.modules_body)
        self.tail = nn.Sequential(*modules_tail)
        self.hfeb = HFIRM(nf=args.n_feats)

        self.ca = CALayer(channel=n_feat)
        self.sa = SALayer(kernel_size=7)

    def forward(self, x):

        # x = self.sub_mean(x)
        first = x
        x = self.head_1(x)
        # print(x[0][0])
        res = x
        # x = self.head_2(x)

        res_x = x

        fusions = []
        for i, l in enumerate(self.modules_body):
            figure = x
            x = l(x)
            fusions.append(self.fusion(torch.cat((self.sa(x), self.ca(x)), 1)))
            # fusions.append(x)
        # y = self.fusion_end(torch.cat(fusions, 1))
        y = self.ca(self.fusion_end(torch.cat(fusions, 1)))
        hfeb = self.hfeb(first)
        res = res + self.sa(x) + hfeb + y
        # res = self.sa(x) + res + hfeb
        # res = x + y + res
        x = self.tail(res)
        x = self.conv(x)
        # x = self.add_mean(x)
        return x

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))
