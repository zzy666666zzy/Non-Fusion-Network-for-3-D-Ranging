# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
from glob import glob
from torch.autograd import Variable
import pathlib
import scipy
import os
import scipy.io as scio
import time
import matplotlib.pyplot as plt
import imageio
from tqdm import tqdm

dtype = torch.cuda.FloatTensor

def test_nyu(model, name_test,outdir_m):
    # for lr
    # depth = scipy.io.loadmat(name_test)['bin']/1024
    # depth = np.asarray(depth).astype(np.float32)
    # s1, s2 = depth.shape 

    # spad = scipy.io.loadmat(name_test)['spad']
    # spad = scipy.sparse.csc_matrix.todense(spad)
    # spad = np.asarray(spad).reshape([s2, s1, -1])

    # spad = torch.from_numpy(np.transpose(spad, (2, 1, 0)))
    # spad = spad.unsqueeze(0).unsqueeze(0)#dim->(1,1,1024,72,88)
    # spad_var = Variable(spad.type(dtype))
    
    # denoise_out, sargmax = model(spad_var)
    # denoise = np.argmax(denoise_out.data.cpu().numpy(), axis=1)
    # denoise = denoise.squeeze()
    # smax = sargmax.data.cpu().numpy().squeeze()
    
    # rmse = np.sqrt(np.mean((smax - depth)**2)) #(for NYUv2, should times 12.276 to meter)
    # print("The RMSE: {}".format(rmse))
    # # plt.imshow(depth)
    # # plt.show()
    # plt.imshow(smax)
    # plt.show()
    
    #%% for hr
    depth = scipy.io.loadmat(name_test)['depth']/1024
    depth = np.asarray(depth).astype(np.float32)
    s1, s2 = depth.shape 

    spad = scipy.io.loadmat(name_test)['spad']
    spad = scipy.sparse.csc_matrix.todense(spad)
    spad = np.asarray(spad).reshape([s2, s1, -1])

    spad = torch.from_numpy(np.transpose(spad, (2, 1, 0)))
    spad = spad.unsqueeze(0).unsqueeze(0)#dim->(1,1,1024,72,88)
    spad_var = Variable(spad.type(dtype))
    
    dim1 = 64#patch
    dim2 = 64
    step = 32
    num_rows = int(np.floor(s1/step))
    num_cols = int(np.floor(s2/step))
    
    smax = np.zeros((1,s1, s2))
    
    for i in tqdm(range(num_rows)):
        for j in range(num_cols):

            spad_patch = spad_var[:, :, :, i*step:(i*step + dim1), 
                                           j*step:(j*step + dim2)]
            
            denoise_out, sargmax_patch = model(spad_patch)
            
            beginx = 0 if (i == 0) else step//2
            endx = dim1 if (i == (num_rows - 1)) else (dim1 - step//2)
            beginy = 0 if (j == 0) else step//2
            endy = dim2 if (j == (num_cols - 1)) else (dim2 - step//2)
            
            sargmax_patch=sargmax_patch.data.cpu().numpy().squeeze()
            
            smax[:, (i*step + beginx):(i*step + endx), (j*step + beginy):(j*step + endy)] =\
                sargmax_patch[beginx:endx, beginy:endy]
    smax=smax.squeeze()
    #smax=smax*(9.76e-11)*(3e8)/2*1024
    # smax_final=smax[25: s1-17, 20: s2-20]
    # depth_final=depth[25: s1-17, 20: s2-20]
    # plt.imshow(depth)   
    # plt.show()   
    plt.imshow(smax)
    plt.show()    
    rmse = np.sqrt(np.mean((smax - depth)**2))
    np.save('PredictedResult/smax_retrained.npy',smax)
    
    print("The RMSE: {}".format(rmse))
    
    return rmse
