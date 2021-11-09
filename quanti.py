import copy

import torch
import torch.nn as nn #Class interface
import torch.nn.functional as F #Function interface
from torch.autograd import Function

class Round(Function):
    @staticmethod
    def forward(self, input):
        output = torch.round(input)
        return output

    @staticmethod
    def backward(self, grad_output):
        grad_input = grad_output.clone()
        return grad_input

class ActivationQuantizer(nn.Module):
    def __init__(self, a_bits):
        super(ActivationQuantizer, self).__init__()
        self.a_bits = a_bits  
    
    def round(self, input):
        output = Round.apply(input)
        return output 

    def forward(self, input):
        if self.a_bits == 32:
            output = input
        elif self.a_bits == 1:
            print('！Binary quantization is not supported ！')
            assert self.a_bits != 1
        else:
            output = torch.clamp(input * 0.1, 0, 1)
            scale = 1 / float(2 ** self.a_bits - 1)
            output = self.round(output / scale) * scale  
        return output

class WeightQuantizer(nn.Module):
    def __init__(self, w_bits):
        super(WeightQuantizer, self).__init__()
        self.w_bits = w_bits  

    def round(self, input):
        output = Round.apply(input)
        return output 

    def forward(self, input):
        if self.w_bits == 32:
            output = input
        elif self.w_bits == 1:
            print('！Binary quantization is not supported ！')
            assert self.w_bits != 1                      
        else:
            output = torch.tanh(input)
            output = output / 2 / torch.max(torch.abs(output)) + 0.5  #  Normalization
            scale = 1 / float(2 ** self.w_bits - 1)                   
            output = self.round(output / scale) * scale
            output = 2 * output - 1
        return output

class QuantConv2d(nn.Conv2d):
    def __init__(self,in_channels,out_channels, kernel_size, stride, padding, dilation,
                 groups,bias,a_bits,w_bits,first_layer
                 ):
        
        super(QuantConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, 
                                          padding, dilation, groups,bias)
        
        self.activation_quantizer = ActivationQuantizer(a_bits=a_bits)
        self.weight_quantizer = WeightQuantizer(w_bits=w_bits)    
        self.first_layer = first_layer   
        
    def forward(self, input):
        if not self.first_layer:
          input = self.activation_quantizer(input)
        quant_input = input
        quant_weight = self.weight_quantizer(self.weight) 
        output = F.conv2d(quant_input, quant_weight, self.bias, self.stride, self.padding, self.dilation,
                          self.groups)
        return output

class QuantConvTranspose2d(nn.ConvTranspose2d):
    def __init__(self,in_channels,out_channels, kernel_size, stride, padding, dilation,
                 groups, bias, a_bits, w_bits
                 ):
        
        super(QuantConvTranspose2d, self).__init__(in_channels, out_channels, kernel_size, stride, 
                                                   padding, dilation, groups, bias)
        
        self.activation_quantizer = ActivationQuantizer(a_bits=a_bits)
        self.weight_quantizer = WeightQuantizer(w_bits=w_bits) 

    def forward(self, input):
        quant_input = self.activation_quantizer(input)
        quant_weight = self.weight_quantizer(self.weight)
        output = F.conv_transpose2d(quant_input, quant_weight, self.bias, self.stride, self.padding,
                                    self.groups, self.dilation)
        return output
    
    
#3D quanti_conv
class QuantConv3d(nn.Conv3d):
  def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation=1,
        groups=1,
        bias=True,
        a_bits=4,
        w_bits=2,
        first_layer=0
      ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )
        # 实例化调用A和W量化器
        self.activation_quantizer = ActivationQuantizer(a_bits=a_bits)
        self.weight_quantizer = WeightQuantizer(w_bits=w_bits)    
        self.first_layer = first_layer

  def forward(self, input):
    # 量化A和W
    if not self.first_layer:
      input = self.activation_quantizer(input)
    q_input = input
    q_weight = self.weight_quantizer(self.weight) 
    # 量化卷积
    output = F.conv3d(
            input=q_input,
            weight=q_weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups
        )
    return output

class QuantConvTranspose3d(nn.ConvTranspose3d):
  def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation=1,
        groups=1,
        bias=True,
        a_bits=4,
        w_bits=2,
        first_layer=0
      ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )
        # 实例化调用A和W量化器
        self.activation_quantizer = ActivationQuantizer(a_bits=a_bits)
        self.weight_quantizer = WeightQuantizer(w_bits=w_bits)    
        self.first_layer = first_layer

  def forward(self, input):
    # 量化A和W
    if not self.first_layer:
      input = self.activation_quantizer(input)
    q_input = input
    q_weight = self.weight_quantizer(self.weight) 
    # 量化卷积
    output = F.conv_transpose3d(
            input=q_input,
            weight=q_weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups
        )
    return output
