# -*- coding: utf-8 -*-

import numpy as np
import torch
import math
import scipy.io 
from torch.autograd import Variable
import matplotlib.pyplot as plt

dtype = torch.cuda.FloatTensor

linear_numbin=1024
numbin_128 = 128;
Q=1.02638; #floor((q^128 - 1) / (q-1)  = 1024

numbin_100 = 100;
Q1=1.0374;# for 100 compressed bins

spad = scipy.io.loadmat('../spad_0351_p10.mat')['spad']
spad = scipy.sparse.csc_matrix.todense(spad)
spad = np.asarray(spad).reshape([64, 64, -1])
spad = torch.from_numpy(np.transpose(spad, (2, 1, 0)))
spad = spad.unsqueeze(0).unsqueeze(0)#dim->(1,1,1024,72,88)
spad = Variable(spad.type(dtype))

depth_data=scipy.io.loadmat('../spad_0351_p10.mat')['bin']
depth_data = torch.from_numpy(np.double(depth_data))
depth_data = Variable(depth_data.type(dtype))

bin_128=0
if bin_128:
    numbin=numbin_128
    q=Q
else:
    numbin=numbin_100
    q=Q1

def NonLDiscretization (spad_data, numbin,q):
    
    batchsize, _, _, H, W = spad_data.size()
    
    bin_idx = np.arange(1,numbin+1)
    
    up = np.floor((np.power(q, bin_idx) - 1) / (q-1))
    low = np.floor((np.power(q, bin_idx - 1) - 1) / (q-1))
    
    m_up=np.floor((1-np.exp(-bin_idx*0.012))*1024)
    m_low=np.floor((1-np.exp(-(bin_idx-1)*0.012))*1024)
    
    interval=[]
    ori_interval=[]
    log_spads = torch.zeros(batchsize, 1, numbin, 64, 64)
    for ii in range(2,numbin): #range should be ajusted
        if 2<ii<16:
            log_spads[:,:,ii:,:] = torch.sum(spad[:,:,int(m_low[ii]):int(m_up[ii]),:,:], dim=2)
            interval.append(m_up[ii])
        elif 58<=ii<=numbin:
            log_spads[:,:,ii,:,:]= torch.sum(spad[:,:,int(low[ii]):int(up[ii]),:,:], dim=2)
            ori_interval.append(up[ii])
    return log_spads.cuda()

log_spads=NonLDiscretization(spad,numbin,q);
weights = Variable(torch.linspace(0, 1, steps=spad.size()[2]).unsqueeze(1).unsqueeze(1).type(torch.cuda.FloatTensor))
weighted_smax = weights * log_spads
soft_argmax = weighted_smax.sum(1).unsqueeze(1)


bin_idx = np.arange(1, numbin + 1)
dup = np.floor((np.power(q, bin_idx) - 1) / (q - 1)) / linear_numbin
dlow = np.floor((np.power(q, bin_idx - 1) - 1) / (q - 1)) / linear_numbin

H, W = depth_data.size()

rates = torch.zeros(numbin, H, W).cuda()
for ii in np.arange(numbin):
    rates[ii,:,:] = (depth_data <= dup[ii]) & (depth_data >= dlow[ii])
rates = Variable(rates.type(dtype))
rates.requires_grad_(requires_grad = True)
rates=rates.data.cpu().numpy()
out = {'rates':rates}
scipy.io.savemat('./', out)

# numbin_128 = 128;
# Q=1.02638; #floor((q^128 - 1) / (q-1)  = 1024

# numbin_100 = 100;
# Q1=1.0374;# for 100 compressed bins

# spad = scipy.io.loadmat('../spad_0351_p10.mat')['spad']
# spad = scipy.sparse.csc_matrix.todense(spad)

# bin_128=0
# if bin_128:
#     numbin=numbin_128
#     q=Q
# else:
#     numbin=numbin_100
#     q=Q1

# bin_idx = np.arange(1,numbin+1)

# up = np.floor((np.power(q, bin_idx) - 1) / (q-1))
# low = np.floor((np.power(q, bin_idx - 1) - 1) / (q-1))

# m_up=np.floor((1-np.exp(-bin_idx*0.012))*1024)
# m_low=np.floor((1-np.exp(-(bin_idx-1)*0.012))*1024)

# interval=[]
# ori_interval=[]
# log_rates = np.zeros((4096,numbin))
# for ii in range(2,numbin): #range should be ajusted
#     if 2<ii<16:
#         log_rates[:,ii] = (np.sum(spad[:,int(m_low[ii]):int(m_up[ii])], axis=1)).squeeze()
#         interval.append(m_up[ii])
#     elif 58<=ii<=numbin:
#         log_rates[:,ii]= (np.sum(spad[:,int(low[ii]):int(up[ii])], axis=1)).squeeze()
#         ori_interval.append(up[ii])
        
        
        
        
        
        
        
        