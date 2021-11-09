# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import skimage.transform
from torch.autograd import Variable, Function
dtype = torch.cuda.FloatTensor


class UnetPP3D(nn.Module):
    def __init__(self):
        super(UnetPP3D, self).__init__()
        #Level-1
        self.conv00 = nn.Sequential(
            nn.Conv3d(1, 4, kernel_size=5, stride=1, padding=2, groups=1, bias=True),
            nn.BatchNorm3d(4),
            nn.ReLU(),
            nn.Conv3d(4, 4, kernel_size=5, stride=1, padding=2, groups=1, bias=True),
            nn.BatchNorm3d(4),
            nn.ReLU(),
        )
        
        self.pool00=nn.MaxPool3d(kernel_size=5,stride=2,padding=2)
        
        self.conv10 = nn.Sequential(
            nn.Conv3d(4, 8, kernel_size=5, stride=1, padding=2, groups=1, bias=True),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.Conv3d(8, 8, kernel_size=5, stride=1, padding=2, groups=1, bias=True),
            nn.BatchNorm3d(8),
            nn.ReLU(),
        )
        
        self.up10=nn.Sequential(
            nn.ConvTranspose3d(8, 8,kernel_size=6,stride=2,padding=2),
            nn.BatchNorm3d(8),
            nn.ReLU()
        )
        
        self.conv01 = nn.Sequential(
            nn.Conv3d(12, 6, kernel_size=5, stride=1, padding=2, groups=1, bias=True),
            nn.BatchNorm3d(6),
            nn.ReLU(),
            nn.Conv3d(6, 6, kernel_size=5, stride=1, padding=2, groups=1, bias=True),
            nn.BatchNorm3d(6),
            nn.ReLU(),
        )
        
        #Level-2
        self.pool10=nn.MaxPool3d(kernel_size=5,stride=2,padding=2)
        
        self.conv20 = nn.Sequential(
            nn.Conv3d(8, 12, kernel_size=5, stride=1, padding=2, groups=1, bias=True),
            nn.BatchNorm3d(12),
            nn.ReLU(),
            nn.Conv3d(12, 12, kernel_size=5, stride=1, padding=2, groups=1, bias=True),
            nn.BatchNorm3d(12),
            nn.ReLU(),
        )
        
        self.up20=nn.Sequential(
            nn.ConvTranspose3d(12, 12,kernel_size=6,stride=2,padding=2),
            nn.BatchNorm3d(12),
            nn.ReLU(),
        )
        
        self.conv11 = nn.Sequential(
            nn.Conv3d(20, 12, kernel_size=5, stride=1, padding=2, groups=1, bias=True),
            nn.BatchNorm3d(12),
            nn.ReLU(),
            nn.Conv3d(12, 12, kernel_size=5, stride=1, padding=2, groups=1, bias=True),
            nn.BatchNorm3d(12),
            nn.ReLU(),
        )
        
        self.up11=nn.Sequential(
            nn.ConvTranspose3d(12, 12,kernel_size=6,stride=2,padding=2),
            nn.BatchNorm3d(12),
            nn.ReLU(),
        )
          
        #conv00+conv01+conv11 -> conv02
        self.conv02 = nn.Sequential(
            nn.Conv3d(22, 11, kernel_size=5, stride=1, padding=2, groups=1, bias=True),
            nn.BatchNorm3d(11),
            nn.ReLU(),
            nn.Conv3d(11, 11, kernel_size=5, stride=1, padding=2, groups=1, bias=True),
            nn.BatchNorm3d(11),
            nn.ReLU(),
        )
        
        #Level-3
        self.pool20=nn.MaxPool3d(kernel_size=5,stride=2,padding=2)
        
        self.conv30=nn.Sequential(
            nn.Conv3d(12, 15, kernel_size=5, stride=1, padding=2, groups=1, bias=True),
            nn.BatchNorm3d(15),
            nn.ReLU(),
            nn.Conv3d(15, 15, kernel_size=5, stride=1, padding=2, groups=1, bias=True),
            nn.BatchNorm3d(15),
            nn.ReLU(),   
        )
        self.up30=nn.Sequential(
            nn.ConvTranspose3d(15, 15,kernel_size=6,stride=2,padding=2),
            nn.BatchNorm3d(15),
            nn.ReLU(),
        )
        
        self.conv21=nn.Sequential(
            nn.Conv3d(27, 10, kernel_size=5, stride=1, padding=2, groups=1, bias=True),
            nn.BatchNorm3d(10),
            nn.ReLU(),
            nn.Conv3d(10, 10, kernel_size=5, stride=1, padding=2, groups=1, bias=True),
            nn.BatchNorm3d(10),
            nn.ReLU(),   
        )
        
        self.up21=nn.Sequential(
            nn.ConvTranspose3d(10, 10,kernel_size=6,stride=2,padding=2),
            nn.BatchNorm3d(10),
            nn.ReLU(),
        )
        #conv10+conv11+conv21 -> conv12 8 12 10
        self.conv12=nn.Sequential(
            nn.Conv3d(30, 15, kernel_size=5, stride=1, padding=2, groups=1, bias=True),
            nn.BatchNorm3d(15),
            nn.ReLU(),
            nn.Conv3d(15, 15, kernel_size=5, stride=1, padding=2, groups=1, bias=True),
            nn.BatchNorm3d(15),
            nn.ReLU(),   
        )
        
        self.up12=nn.Sequential(
            nn.ConvTranspose3d(15, 15,kernel_size=6,stride=2,padding=2),
            nn.BatchNorm3d(15),
            nn.ReLU(),
        )
        #conv00+conv01+conv02+conv12 -> conv03 4 6 11 15
        self.conv03=nn.Sequential(
            nn.Conv3d(36, 10,kernel_size=5,stride=1,padding=2,groups=1,bias=True),
            nn.BatchNorm3d(10),
            nn.ReLU(),
        )
        
        self.final=nn.Conv3d(10, 1, kernel_size=1)
        
    def forward(self,spad):
        smax = torch.nn.Softmax2d()
        #level-1
        x00=self.conv00(spad)
        x10=self.conv10(self.pool00(x00))
        x01=self.conv01(torch.cat([x00,self.up10(x10)],1))
        #level-2
        x20=self.conv20(self.pool10(x10))
        x11=self.conv11(torch.cat([x10,self.up20(x20)],1))
        x02=self.conv02(torch.cat([x00,x01,self.up11(x11)],1))
        #level-3
        x30=self.conv30(self.pool20(x20))
        x21=self.conv21(torch.cat([x20,self.up30(x30)],1))
        x12=self.conv12(torch.cat([x10,x11,self.up21(x21)],1))
        x03=self.conv03(torch.cat([x00,x01,x02,self.up12(x12)],1))
        
        final=self.final(x03)
        
        # squeeze and softmax for pixelwise classification loss
        denoise_out = torch.squeeze(final, 1)
        smax_denoise_out = smax(denoise_out)

        # soft argmax
        weights = Variable(torch.linspace(0, 1, steps=spad.size()[2]).unsqueeze(1).unsqueeze(1).type(torch.cuda.FloatTensor))
        weighted_smax = weights * smax_denoise_out
        soft_argmax = weighted_smax.sum(1).unsqueeze(1)

        return denoise_out, soft_argmax
        





