# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import skimage.transform
import scipy.io
from torch.autograd import Variable
dtype = torch.cuda.FloatTensor


class _DS_Block(nn.Module):
    def __init__(self):
        super(_DS_Block, self).__init__()

        self.ds_block = nn.Sequential(
            nn.Conv2d(32, 32, 5, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 5, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 5, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

    def forward(self, x):
        output = self.ds_block(x)
        return output

class DenoiseModel(nn.Module):
    def __init__(self):
        super(DenoiseModel, self).__init__()

        self.ds1 = nn.Sequential(
            nn.Conv3d(1, 1, 7, stride=2, padding=3, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(1),
            nn.ReLU(),
        )
        self.ds2 = nn.Sequential(
            nn.Conv3d(1, 1, 5, stride=2, padding=2, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(1),
            nn.ReLU(),
        )
        self.ds3 = nn.Sequential(
            nn.Conv3d(1, 1, 3, stride=2, padding=1, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(1),
            nn.ReLU(),
        )

        self.up1 = nn.Sequential(
            nn.ConvTranspose3d(36, 36, 6, stride=2, padding=2, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(36),
            nn.ReLU(),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose3d(28, 28, 6, stride=2, padding=2, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(28),
            nn.ReLU(),
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose3d(16, 16, 6, stride=2, padding=2, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(16),
            nn.ReLU(),
        )

        self.refine = nn.Sequential(
            nn.Conv3d(40, 16, 7, stride=1, padding=3, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16, 16, 7, stride=1, padding=3, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16, 16, 7, stride=1, padding=3, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(16),
            nn.ReLU(),
        )

        self.regress = nn.Sequential(
            nn.Conv3d(16, 1, 7, stride=1, padding=3, dilation=1, groups=1, bias=True),
        )

        self.conv0 = nn.Sequential(
            nn.Conv3d(1, 4, 9, stride=1, padding=4, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(4),
            nn.ReLU(),
            nn.Conv3d(4, 4, 9, stride=1, padding=4, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(4),
            nn.ReLU(),
            nn.Conv3d(4, 4, 9, stride=1, padding=4, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(4),
            nn.ReLU(),
        )
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 8, 7, stride=1, padding=3, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.Conv3d(8, 8, 7, stride=1, padding=3, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.Conv3d(8, 8, 7, stride=1, padding=3, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(8),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(1, 12, 5, stride=1, padding=2, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(12),
            nn.ReLU(),
            nn.Conv3d(12, 12, 5, stride=1, padding=2, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(12),
            nn.ReLU(),
            nn.Conv3d(12, 12, 5, stride=1, padding=2, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(12),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(1, 16, 3, stride=1, padding=1, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16, 16, 3, stride=1, padding=1, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16, 16, 3, stride=1, padding=1, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(16),
            nn.ReLU(),
        )

        self.intensity_ds = nn.Sequential(
            torch.nn.Conv2d(1, 8, (7, 7), (2, 2), 3),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 8, (7, 7), (2, 2), 3),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 8, (5, 5), (2, 2), 2),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 1, (5, 5), (1, 1), 2),
        )

        self.refine_depth1 = nn.Sequential(
            torch.nn.Conv2d(1, 32, (5, 5), (1, 1), 2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, (5, 5), (1, 1), 2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, (5, 5), (1, 1), 2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
        )

        self.refine_depth2 = nn.Sequential(
            torch.nn.Conv2d(33, 32, (5, 5), (1, 1), 2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, (5, 5), (1, 1), 2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, (5, 5), (1, 1), 2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
        )

        self.ids_in = nn.Sequential(
            torch.nn.Conv2d(1, 32, (5, 5), (1, 1), 2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, (5, 5), (1, 1), 2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, (5, 5), (1, 1), 2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
        )

        self.ids1 = _DS_Block()
        self.ids2 = _DS_Block()

        self.iskip = nn.Sequential(
            torch.nn.ConvTranspose2d(65, 32, (6, 6), (2, 2), 2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(32, 32, (6, 6), (2, 2), 2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(32, 1, (6, 6), (2, 2), 2)
        )

        self.iup1 = nn.Sequential(
            torch.nn.ConvTranspose2d(65, 32, (6, 6), (2, 2), 2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
        )
        self.iup1_refine = nn.Sequential(
            torch.nn.Conv2d(64, 32, (5, 5), (1, 1), 2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, (5, 5), (1, 1), 2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, (5, 5), (1, 1), 2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
        )
        self.iup2 = nn.Sequential(
            torch.nn.ConvTranspose2d(32, 16, (6, 6), (2, 2), 2),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
        )
        self.iup2_refine = nn.Sequential(
            torch.nn.Conv2d(48, 16, (5, 5), (1, 1), 2),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 16, (5, 5), (1, 1), 2),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 16, (5, 5), (1, 1), 2),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
        )
        self.iup3 = nn.Sequential(
            torch.nn.ConvTranspose2d(16, 8, (6, 6), (2, 2), 2),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),
        )
        self.iup3_refine = nn.Sequential(
            torch.nn.Conv2d(40, 8, (5, 5), (1, 1), 2),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 8, (5, 5), (1, 1), 2),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 1, (5, 5), (1, 1), 2),
        )

        # https://github.com/twtygqyy/pytorch-LapSRN/blob/master/lapsrn.py#L6
        def get_upsample_filter(size):
            """Make a 2D bilinear kernel suitable for upsampling"""
            factor = (size + 1) // 2
            if size % 2 == 1:
                center = factor - 1.
            else:
                center = factor - 0.5
            og = np.ogrid[:size, :size, :size]
            filter = (1 - abs(og[0] - center) / factor) * \
                     (1 - abs(og[1] - center) / factor) * \
                     (1 - abs(og[2] - center) / factor)
            return torch.from_numpy(filter).float()

        for n in [self.up1, self.up2, self.up3, self.ds1, self.ds2, self.ds3]:
            for m in n:
                if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
                    c1, c2, d, h, w = m.weight.data.size()
                    weight = get_upsample_filter(h)
                    m.weight.data = weight.view(1, 1, d, h, w).repeat(c1, c2, 1, 1, 1)
                    if m.bias is not None:
                        m.bias.data.zero_()

    def forward(self, spad):

        # pass spad through autoencoder
        smax = torch.nn.Softmax2d()

        ds1_out = self.ds1(spad)
        ds2_out = self.ds2(ds1_out)
        ds3_out = self.ds3(ds2_out)

        conv0_out = self.conv0(spad)
        conv1_out = self.conv1(ds1_out)
        conv2_out = self.conv2(ds2_out)
        conv3_out = self.conv3(ds3_out)

        up3_out = self.up3(conv3_out)
        up2_out = self.up2(torch.cat((conv2_out, up3_out), 1))
        up1_out = self.up1(torch.cat((conv1_out, up2_out), 1))
        up0_out = torch.cat((conv0_out, up1_out), 1)

        refine_out = self.refine(up0_out)
        regress_out = self.regress(refine_out)




        # squeeze and softmax for pixelwise classification loss
        denoise_out = torch.squeeze(regress_out, 1)
        # Get the index of larggest photon amount
        smax_denoise_out = smax(denoise_out)

        # soft argmax
        weights = Variable(torch.linspace(0, 1, steps=spad.size()[2]).unsqueeze(1).unsqueeze(1).type(torch.cuda.FloatTensor))
        weighted_smax = weights * smax_denoise_out
        soft_argmax = weighted_smax.sum(1).unsqueeze(1)

        return denoise_out, soft_argmax

class FusionDenoiseModel(nn.Module):
    def __init__(self):
        super(FusionDenoiseModel, self).__init__()
        self.ds1 = nn.Sequential(
            nn.Conv3d(1, 1, 7, stride=2, padding=3, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(1),
            nn.ReLU(),
        )
        self.ds2 = nn.Sequential(
            nn.Conv3d(1, 1, 5, stride=2, padding=2, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(1),
            nn.ReLU(),
        )
        self.ds3 = nn.Sequential(
            nn.Conv3d(1, 1, 3, stride=2, padding=1, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(1),
            nn.ReLU(),
        )

        self.up1 = nn.Sequential(
            nn.ConvTranspose3d(36, 36, 6, stride=2, padding=2, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(36),
            nn.ReLU(),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose3d(28, 28, 6, stride=2, padding=2, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(28),
            nn.ReLU(),
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose3d(16, 16, 6, stride=2, padding=2, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(16),
            nn.ReLU(),
        )

        self.refine = nn.Sequential(
            nn.Conv3d(41, 16, 7, stride=1, padding=3, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16, 16, 7, stride=1, padding=3, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16, 16, 7, stride=1, padding=3, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(16),
            nn.ReLU(),
        )

        self.regress = nn.Sequential(
            nn.Conv3d(16, 1, 7, stride=1, padding=3, dilation=1, groups=1, bias=True),
        )

        self.conv0 = nn.Sequential(
            nn.Conv3d(1, 4, 9, stride=1, padding=4, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(4),
            nn.ReLU(),
            nn.Conv3d(4, 4, 9, stride=1, padding=4, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(4),
            nn.ReLU(),
            nn.Conv3d(4, 4, 9, stride=1, padding=4, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(4),
            nn.ReLU(),
        )
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 8, 7, stride=1, padding=3, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.Conv3d(8, 8, 7, stride=1, padding=3, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.Conv3d(8, 8, 7, stride=1, padding=3, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(8),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(1, 12, 5, stride=1, padding=2, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(12),
            nn.ReLU(),
            nn.Conv3d(12, 12, 5, stride=1, padding=2, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(12),
            nn.ReLU(),
            nn.Conv3d(12, 12, 5, stride=1, padding=2, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(12),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(1, 16, 3, stride=1, padding=1, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16, 16, 3, stride=1, padding=1, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16, 16, 3, stride=1, padding=1, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(16),
            nn.ReLU(),
        )

        self.intensity_ds = nn.Sequential(
            torch.nn.Conv2d(1, 8, (7, 7), (2, 2), 3),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 8, (7, 7), (2, 2), 3),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 8, (5, 5), (2, 2), 2),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 1, (5, 5), (1, 1), 2),
        )

        self.refine_depth1 = nn.Sequential(
            torch.nn.Conv2d(1, 32, (5, 5), (1, 1), 2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, (5, 5), (1, 1), 2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, (5, 5), (1, 1), 2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
        )

        self.refine_depth2 = nn.Sequential(
            torch.nn.Conv2d(33, 32, (5, 5), (1, 1), 2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, (5, 5), (1, 1), 2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, (5, 5), (1, 1), 2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
        )

        self.ids_in = nn.Sequential(
            torch.nn.Conv2d(1, 32, (5, 5), (1, 1), 2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, (5, 5), (1, 1), 2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, (5, 5), (1, 1), 2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
        )

        self.ids1 = _DS_Block()
        self.ids2 = _DS_Block()

        self.iskip = nn.Sequential(
            torch.nn.ConvTranspose2d(65, 32, (6, 6), (2, 2), 2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(32, 32, (6, 6), (2, 2), 2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(32, 1, (6, 6), (2, 2), 2)
        )

        self.iup1 = nn.Sequential(
            torch.nn.ConvTranspose2d(65, 32, (6, 6), (2, 2), 2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
        )
        self.iup1_refine = nn.Sequential(
            torch.nn.Conv2d(64, 32, (5, 5), (1, 1), 2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, (5, 5), (1, 1), 2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, (5, 5), (1, 1), 2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
        )
        self.iup2 = nn.Sequential(
            torch.nn.ConvTranspose2d(32, 16, (6, 6), (2, 2), 2),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
        )
        self.iup2_refine = nn.Sequential(
            torch.nn.Conv2d(48, 16, (5, 5), (1, 1), 2),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 16, (5, 5), (1, 1), 2),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 16, (5, 5), (1, 1), 2),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
        )
        self.iup3 = nn.Sequential(
            torch.nn.ConvTranspose2d(16, 8, (6, 6), (2, 2), 2),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),
        )
        self.iup3_refine = nn.Sequential(
            torch.nn.Conv2d(40, 8, (5, 5), (1, 1), 2),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 8, (5, 5), (1, 1), 2),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 1, (5, 5), (1, 1), 2),
        )

        # https://github.com/twtygqyy/pytorch-LapSRN/blob/master/lapsrn.py#L6
        def get_upsample_filter(size):
            """Make a 2D bilinear kernel suitable for upsampling"""
            factor = (size + 1) // 2
            if size % 2 == 1:
                center = factor - 1.
            else:
                center = factor - 0.5
            og = np.ogrid[:size, :size, :size]
            filter = (1 - abs(og[0] - center) / factor) * \
                     (1 - abs(og[1] - center) / factor) * \
                     (1 - abs(og[2] - center) / factor)
            return torch.from_numpy(filter).float()

        for n in [self.up1, self.up2, self.up3, self.ds1, self.ds2, self.ds3]:
            for m in n:
                if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
                    c1, c2, d, h, w = m.weight.data.size()
                    weight = get_upsample_filter(h)
                    m.weight.data = weight.view(1, 1, d, h, w).repeat(c1, c2, 1, 1, 1)
                    if m.bias is not None:
                        m.bias.data.zero_()

    def forward(self, spad, intensity):

        # downsample intensity image
        intensity_ds_out = intensity
        tiled_intensity_ds_out = intensity_ds_out.repeat(1, spad.size()[2], 1, 1).unsqueeze(1)

        # pass spad through autoencoder
        smax = torch.nn.Softmax2d()

        ds1_out = self.ds1(spad)
        ds2_out = self.ds2(ds1_out)
        ds3_out = self.ds3(ds2_out)

        conv0_out = self.conv0(spad)
        conv1_out = self.conv1(ds1_out)
        conv2_out = self.conv2(ds2_out)
        conv3_out = self.conv3(ds3_out)

        up3_out = self.up3(conv3_out)
        up2_out = self.up2(torch.cat((conv2_out, up3_out), 1))
        up1_out = self.up1(torch.cat((conv1_out, up2_out), 1))
        up0_out = torch.cat((conv0_out, up1_out), 1)

        refine_out = self.refine(torch.cat((tiled_intensity_ds_out, up0_out), 1))
        regress_out = self.regress(refine_out)

        # squeeze and softmax for pixelwise classification loss
        denoise_out = torch.squeeze(regress_out, 1)

        smax_denoise_out = smax(denoise_out)

        # soft argmax
        weights = Variable(torch.linspace(0, 1, steps=spad.size()[2]).unsqueeze(1).unsqueeze(1).type(torch.cuda.FloatTensor))
        weighted_smax = weights * smax_denoise_out
        soft_argmax = weighted_smax.sum(1).unsqueeze(1)

        return denoise_out, soft_argmax

Linear_NUMBIN = 1024
NUMBIN = 128
Q = 1.02638

class SPADnet(nn.Module):
    def __init__(self):
        super(SPADnet, self).__init__()
        self.ds1 = nn.Sequential(
            nn.Conv3d(1, 1, 7, stride=2, padding=3, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(1),
            nn.ReLU(),
        )
        self.ds2 = nn.Sequential(
            nn.Conv3d(1, 1, 5, stride=2, padding=2, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(1),
            nn.ReLU(),
        )
        self.ds3 = nn.Sequential(
            nn.Conv3d(1, 1, 3, stride=2, padding=1, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(1),
            nn.ReLU(),
        )

        self.up1 = nn.Sequential(
            nn.ConvTranspose3d(36, 36, 6, stride=2, padding=2, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(36),
            nn.ReLU(),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose3d(28, 28, 6, stride=2, padding=2, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(28),
            nn.ReLU(),
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose3d(16, 16, 6, stride=2, padding=2, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(16),
            nn.ReLU(),
        )

        self.refine = nn.Sequential(
            nn.Conv3d(41, 16, 7, stride=1, padding=3, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16, 16, 7, stride=1, padding=3, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16, 16, 7, stride=1, padding=3, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(16),
            nn.ReLU(),
        )

        self.regress = nn.Sequential(
            nn.Conv3d(16, 1, 7, stride=1, padding=3, dilation=1, groups=1, bias=True),
        )

        self.conv0 = nn.Sequential(
            nn.Conv3d(1, 4, 9, stride=1, padding=4, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(4),
            nn.ReLU(),
            nn.Conv3d(4, 4, 9, stride=1, padding=4, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(4),
            nn.ReLU(),
            nn.Conv3d(4, 4, 9, stride=1, padding=4, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(4),
            nn.ReLU(),
        )
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 8, 7, stride=1, padding=3, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.Conv3d(8, 8, 7, stride=1, padding=3, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.Conv3d(8, 8, 7, stride=1, padding=3, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(8),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(1, 12, 5, stride=1, padding=2, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(12),
            nn.ReLU(),
            nn.Conv3d(12, 12, 5, stride=1, padding=2, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(12),
            nn.ReLU(),
            nn.Conv3d(12, 12, 5, stride=1, padding=2, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(12),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(1, 16, 3, stride=1, padding=1, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16, 16, 3, stride=1, padding=1, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16, 16, 3, stride=1, padding=1, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(16),
            nn.ReLU(),
        )

    # log-scale rebinning parameters
        self.linear_numbin = Linear_NUMBIN
        self.numbin = NUMBIN
        self.q = Q

    def inference(self, smax_denoise_out):
        
        ## 3D-2D projection with log scale
        bin_idx = np.arange(1, self.numbin + 1)
        dup = np.floor((np.power(self.q, bin_idx) - 1) / (self.q - 1)) / self.linear_numbin
        dlow = np.floor((np.power(self.q, bin_idx - 1) - 1) / (self.q - 1)) / self.linear_numbin
        dmid = torch.from_numpy((dup + dlow) / 2)

        dmid = dmid.squeeze().unsqueeze(1).unsqueeze(1).type(torch.cuda.FloatTensor)
        dmid.requires_grad_(requires_grad = True)

        weighted_smax = dmid * smax_denoise_out
        soft_argmax = weighted_smax.sum(1).unsqueeze(1) 
            
        return soft_argmax

    def forward(self, spad, mono_pc):

        # pass spad through U-net
        smax = torch.nn.Softmax2d()

        ds1_out = self.ds1(spad)
        ds2_out = self.ds2(ds1_out)
        ds3_out = self.ds3(ds2_out)

        conv0_out = self.conv0(spad)
        conv1_out = self.conv1(ds1_out)
        conv2_out = self.conv2(ds2_out)
        conv3_out = self.conv3(ds3_out)

        up3_out = self.up3(conv3_out)
        up2_out = self.up2(torch.cat((conv2_out, up3_out), 1))
        up1_out = self.up1(torch.cat((conv1_out, up2_out), 1))
        up0_out = torch.cat((conv0_out, up1_out), 1)

        refine_out = self.refine(torch.cat((mono_pc, up0_out), 1))
        regress_out = self.regress(refine_out)

        # squeeze and softmax for each-bin classification loss
        denoise_out = torch.squeeze(regress_out, 1)
        smax_denoise_out = smax(denoise_out)

        soft_argmax = self.inference(smax_denoise_out)

        return denoise_out, soft_argmax


import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import torch.nn.functional as F

# feature extraction part
class MsFeat(nn.Module):
    def __init__(self, in_channels):
        outchannel_MS = 2
        super(MsFeat, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv3d(in_channels, outchannel_MS, 3, stride=(1,1,1), padding=1, dilation=1, bias=True), nn.ReLU(inplace=True))
        init.kaiming_normal_(self.conv1[0].weight, 0, 'fan_in', 'relu'); init.constant_(self.conv1[0].bias, 0.0)

        self.conv2 = nn.Sequential(nn.Conv3d(in_channels, outchannel_MS, 3, stride=(1,1,1), padding=2, dilation=2, bias=True), nn.ReLU(inplace=True))
        init.kaiming_normal_(self.conv2[0].weight, 0, 'fan_in', 'relu'); init.constant_(self.conv2[0].bias, 0.0)

        self.conv3 = nn.Sequential(nn.Conv3d(outchannel_MS, outchannel_MS, 3, padding=1, dilation=1, bias=True), nn.ReLU(inplace=True))
        init.kaiming_normal_(self.conv3[0].weight, 0, 'fan_in', 'relu'); init.constant_(self.conv3[0].bias, 0.0)

        self.conv4 = nn.Sequential(nn.Conv3d(outchannel_MS, outchannel_MS, 3, padding=2, dilation=2, bias=True), nn.ReLU(inplace=True))
        init.kaiming_normal_(self.conv4[0].weight, 0, 'fan_in', 'relu'); init.constant_(self.conv4[0].bias, 0.0)

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        conv2 = self.conv2(inputs)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv1)
        return torch.cat((conv1, conv2, conv3, conv4), 1)


class NonLocal(nn.Module):
    def __init__(self, inplanes, use_scale=False, groups=None):
        self.use_scale = use_scale
        self.groups = groups

        super(NonLocal, self).__init__()
        # conv theta
        self.t = nn.Conv3d(inplanes, inplanes//1, kernel_size=1, stride=1, bias=False)
        init.kaiming_normal_(self.t.weight, 0, 'fan_in', 'relu')
        # conv phi
        self.p = nn.Conv3d(inplanes, inplanes//1, kernel_size=1, stride=1, bias=False)
        init.kaiming_normal_(self.p.weight, 0, 'fan_in', 'relu')
        # conv g
        self.g = nn.Conv3d(inplanes, inplanes//1, kernel_size=1, stride=1, bias=False)
        init.kaiming_normal_(self.g.weight, 0, 'fan_in', 'relu')
        # conv z
        self.z = nn.Conv3d(inplanes//1, inplanes, kernel_size=1, stride=1,
                           groups=self.groups, bias=False)
        init.kaiming_normal_(self.z.weight, 0, 'fan_in', 'relu')
        # concat groups
        self.gn = nn.GroupNorm(num_groups=self.groups, num_channels=inplanes)
        init.constant_(self.gn.weight, 0); nn.init.constant_(self.gn.bias, 0)

        if self.use_scale:
            print("=> WARN: Non-local block uses 'SCALE'")
        if self.groups:
            print("=> WARN: Non-local block uses '{}' groups".format(self.groups))

    def kernel(self, t, p, g, b, c, d, h, w):
        """The linear kernel (dot production).

        Args:
            t: output of conv theata
            p: output of conv phi
            g: output of conv g
            b: batch size
            c: channels number
            d: depth of featuremaps
            h: height of featuremaps
            w: width of featuremaps
        """
        t = t.view(b, 1, c * d * h * w)
        p = p.view(b, 1, c * d * h * w)
        g = g.view(b, c * d * h * w, 1)

        att = torch.bmm(p, g)

        if self.use_scale:
            att = att.div((c * d * h * w) ** 0.5)

        x = torch.bmm(att, t)
        x = x.view(b, c, d, h, w)

        return x

    def forward(self, x):
        residual = x

        t = self.t(x) #b,ch,d,h,w
        p = self.p(x) #b,ch,d,h,w
        g = self.g(x) #b,ch,d,h,w

        b, c, d, h, w = t.size()

        if self.groups and self.groups > 1:
            _c = int(c / self.groups)

            ts = torch.split(t, split_size_or_sections=_c, dim=1)
            ps = torch.split(p, split_size_or_sections=_c, dim=1)
            gs = torch.split(g, split_size_or_sections=_c, dim=1)

            _t_sequences = []
            for i in range(self.groups):
                _x = self.kernel(ts[i], ps[i], gs[i],
                                 b, _c, d, h, w)
                _t_sequences.append(_x)

            x = torch.cat(_t_sequences, dim=1)
        else:
            x = self.kernel(t, p, g,
                            b, c, d, h, w)

        x = self.z(x)
        x = self.gn(x) + residual

        return x


# feature integration
class Block(nn.Module):
    def __init__(self, in_channels):
        outchannel_block = 16
        super(Block, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv3d(in_channels, outchannel_block, 1, padding=0, dilation=1, bias=True), nn.ReLU(inplace=True))
        init.kaiming_normal_(self.conv1[0].weight, 0, 'fan_in', 'relu'); init.constant_(self.conv1[0].bias, 0.0)

        self.feat1 = nn.Sequential(nn.Conv3d(outchannel_block, 8, 3, padding=1, dilation=1, bias=True), nn.ReLU(inplace=True))
        init.kaiming_normal_(self.feat1[0].weight, 0, 'fan_in', 'relu'); init.constant_(self.feat1[0].bias, 0.0)

        self.feat15 = nn.Sequential(nn.Conv3d(8, 4, 3, padding=2, dilation=2, bias=True), nn.ReLU(inplace=True))
        init.kaiming_normal_(self.feat15[0].weight, 0, 'fan_in', 'relu'); init.constant_(self.feat15[0].bias, 0.0)

        self.feat2 = nn.Sequential(nn.Conv3d(outchannel_block, 8, 3, padding=2, dilation=2, bias=True), nn.ReLU(inplace=True))
        init.kaiming_normal_(self.feat2[0].weight, 0, 'fan_in', 'relu'); init.constant_(self.feat2[0].bias, 0.0)

        self.feat25 = nn.Sequential(nn.Conv3d(8, 4, 3, padding=1, dilation=1, bias=True), nn.ReLU(inplace=True))
        init.kaiming_normal_(self.feat25[0].weight, 0, 'fan_in', 'relu'); init.constant_(self.feat25[0].bias, 0.0)

        self.feat = nn.Sequential(nn.Conv3d(24, 8, 1, padding=0, dilation=1, bias=True), nn.ReLU(inplace=True))
        init.kaiming_normal_(self.feat[0].weight, 0, 'fan_in', 'relu'); init.constant_(self.feat[0].bias, 0.0)
    # note the channel for each layer
    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        feat1 = self.feat1(conv1)
        feat15 = self.feat15(feat1)
        feat2 = self.feat2(conv1)
        feat25 = self.feat25(feat2)
        feat = self.feat(torch.cat((feat1, feat15, feat2, feat25), 1))
        return torch.cat((inputs, feat), 1)


# build the model
class DeepBoosting(nn.Module):
    def __init__(self, in_channels=1):
        super(DeepBoosting, self).__init__()
        self.msfeat = MsFeat(in_channels)
        self.C1 = nn.Sequential(nn.Conv3d(8,2,kernel_size=1, stride=(1,1,1),bias=True),nn.ReLU(inplace=True))
        init.kaiming_normal_(self.C1[0].weight, 0, 'fan_in', 'relu'); init.constant_(self.C1[0].bias, 0.0)
        self.nl = NonLocal(2, use_scale=False, groups=1)

        self.ds1 = nn.Sequential(nn.Conv3d(2,4,kernel_size=3,stride=(2,1,1),padding=(1,1,1),bias=True),nn.ReLU(inplace=True))
        init.kaiming_normal_(self.ds1[0].weight, 0, 'fan_in', 'relu'); init.constant_(self.ds1[0].bias, 0.0)
        self.ds2 = nn.Sequential(nn.Conv3d(4, 8, kernel_size=3, stride=(2, 1, 1), padding=(1, 1, 1), bias=True),nn.ReLU(inplace=True))
        init.kaiming_normal_(self.ds2[0].weight, 0, 'fan_in', 'relu'); init.constant_(self.ds2[0].bias, 0.0)
        self.ds3 = nn.Sequential(nn.Conv3d(8,16,kernel_size=3,stride=(2,1,1),padding=(1,1,1),bias=True),nn.ReLU(inplace=True))
        init.kaiming_normal_(self.ds3[0].weight, 0, 'fan_in', 'relu'); init.constant_(self.ds3[0].bias, 0.0)
        self.ds4 = nn.Sequential(nn.Conv3d(16, 32, kernel_size=3, stride=(2, 1, 1), padding=(1, 1, 1), bias=True),nn.ReLU(inplace=True))
        init.kaiming_normal_(self.ds4[0].weight, 0, 'fan_in', 'relu'); init.constant_(self.ds4[0].bias, 0.0)

        self.dfus_block0 = Block(32)
        self.dfus_block1 = Block(40)
        self.dfus_block2 = Block(48)
        self.dfus_block3 = Block(56)
        self.dfus_block4 = Block(64)
        self.dfus_block5 = Block(72)
        self.dfus_block6 = Block(80)
        self.dfus_block7 = Block(88)
        self.dfus_block8 = Block(96)
        self.dfus_block9 = Block(104)
        self.convr = nn.Sequential(
            nn.ConvTranspose3d(112, 56, kernel_size=(6, 3, 3), stride=(2, 1, 1), padding=(2, 1, 1), bias=False),nn.ReLU(inplace=True),
            nn.ConvTranspose3d(56, 28, kernel_size=(6, 3, 3), stride=(2, 1, 1), padding=(2, 1, 1), bias=False),nn.ReLU(inplace=True),
            nn.ConvTranspose3d(28, 14, kernel_size=(6, 3, 3), stride=(2, 1, 1), padding=(2, 1, 1), bias=False),nn.ReLU(inplace=True),
            nn.ConvTranspose3d(14, 7, kernel_size=(6, 3, 3), stride=(2, 1, 1), padding=(2, 1, 1), bias=False))
        init.kaiming_normal_(self.convr[0].weight, 0, 'fan_in', 'relu')
        init.kaiming_normal_(self.convr[2].weight, 0, 'fan_in', 'relu')
        init.kaiming_normal_(self.convr[4].weight, 0, 'fan_in', 'relu')
        init.normal_(self.convr[6].weight, mean=0.0, std=0.001)

        self.C2 = nn.Sequential(nn.Conv3d(7,1,kernel_size=1, stride=(1,1,1),bias=True),nn.ReLU(inplace=True))
        init.kaiming_normal_(self.C2[0].weight, 0, 'fan_in', 'relu'); init.constant_(self.C2[0].bias, 0.0)
        
    def forward(self, inputs):
        smax = torch.nn.Softmax2d()

        msfeat = self.msfeat(inputs) 
        c1 = self.C1(msfeat)
        nlout = self.nl(c1)
        dsfeat1 = self.ds1(nlout)
        dsfeat2 = self.ds2(dsfeat1) 
        dsfeat3 = self.ds3(dsfeat2) 
        dsfeat4 = self.ds4(dsfeat3) 
        b0 = self.dfus_block0(dsfeat4)
        b1 = self.dfus_block1(b0)
        b2 = self.dfus_block2(b1)
        b3 = self.dfus_block3(b2)
        b4 = self.dfus_block4(b3)
        b5 = self.dfus_block5(b4)
        b6 = self.dfus_block6(b5)
        b7 = self.dfus_block7(b6)
        b8 = self.dfus_block8(b7)
        b9 = self.dfus_block9(b8)
        convr = self.convr(b9)
        convr = self.C2(convr)

        denoise_out = torch.squeeze(convr, 1)

        weights = Variable(torch.linspace(0, 1, steps=denoise_out.size()[1]).unsqueeze(1).unsqueeze(1).type(torch.cuda.FloatTensor))
        weighted_smax = weights * smax(denoise_out)
        soft_argmax = weighted_smax.sum(1).unsqueeze(1)

        return denoise_out, soft_argmax




