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
import cv2 
from scipy.stats import ttest_ind

dtype = torch.cuda.FloatTensor
Linear_NUMBIN = 1024
NUMBIN = 128
Q = 1.02638 ## Solution for (q^128 - 1) / (q - 1) = 1024
#%% to log
def tologscale(rates, numbin, q):

    ## convert pc to log scale (log rebinning)
    batchsize, _, _, H, W = rates.size()

    bin_idx = np.arange(1, numbin + 1)
    up = np.floor((np.power(q, bin_idx) - 1) / (q - 1))
    low = np.floor((np.power(q, bin_idx - 1) - 1) / (q - 1))

    log_rates = torch.zeros(batchsize, 1, numbin, H, W)
    for ii in range(numbin):
        log_rates[:,:,ii,:,:] = torch.sum(rates[:, :, int(low[ii]):int(up[ii]), :, :], dim = 2)

    return log_rates.cuda()

#%%2d to 3d
def dmap2pc(dmap, numbin, q, linear_numbin):

    ## 2D-3D up-projection
    bin_idx = np.arange(1, numbin + 1)
    dup = np.floor((np.power(q, bin_idx) - 1) / (q - 1)) / linear_numbin
    dlow = np.floor((np.power(q, bin_idx - 1) - 1) / (q - 1)) / linear_numbin
    dmid = (dup + dlow) / 2

    batchsize, _, H, W = dmap.size()

    rates = torch.zeros(batchsize, 1, numbin, H, W).cuda()
    for ii in np.arange(NUMBIN):
        rates[:,:,ii,:,:] = (dmap <= dup[ii]) & (dmap >= dlow[ii])
    rates = Variable(rates.type(dtype))
    rates.requires_grad_(requires_grad = True)
    
    return rates

#%%
# test function for Middlebury dataset 
def test_wi(model, path,name_test,outdir_m):
    #spadnet (spad+intensity) method for hr
    # intensity = scipy.io.loadmat(name_test)['intensity']
    # intensity = np.asarray(intensity).astype(np.float32)
    # s1, s2 = intensity.shape
    # intensity = torch.from_numpy(intensity).type(dtype)
    # intensity = intensity.unsqueeze(0).unsqueeze(0)
    # intensity_var = Variable(intensity)
    
    # depth = scipy.io.loadmat(name_test)['depth']
    # depth = np.asarray(depth).astype(np.float32)
    # s1, s2 = depth.shape 

    # spad = scipy.io.loadmat(name_test)['spad']
    # spad = scipy.sparse.csc_matrix.todense(spad)
    # spad = np.asarray(spad).reshape([s2, s1, -1])
    # spad = torch.from_numpy(np.transpose(spad, (2, 1, 0)))
    # spad = spad.unsqueeze(0).unsqueeze(0)#dim->(1,1,1024,72,88)
    # spad_var = Variable(spad.type(dtype))
    
    # dim1 = 64#patch
    # dim2 = 64
    # step = 32
    # num_rows = int(np.floor(s1/step))
    # num_cols = int(np.floor(s2/step))
    
    # smax = np.zeros((1,s1, s2))
    
    # for i in tqdm(range(num_rows)):
    #     for j in range(num_cols):

    #         spad_patch = spad_var[:, :, :, i*step:(i*step + dim1), 
    #                                         j*step:(j*step + dim2)]
            
    #         intensity_patch = intensity_var[:, :, i*step:(i*step + dim1), 
    #                                         j*step:(j*step + dim2)]
            
            
    #         denoise_out, sargmax_patch = model(spad_patch,intensity_patch)
            
    #         beginx = 0 if (i == 0) else step//2
    #         endx = dim1 if (i == (num_rows - 1)) else (dim1 - step//2)
    #         beginy = 0 if (j == 0) else step//2
    #         endy = dim2 if (j == (num_cols - 1)) else (dim2 - step//2)
            
    #         sargmax_patch=sargmax_patch.data.cpu().numpy().squeeze()
            
    #         smax[:, (i*step + beginx):(i*step + endx), (j*step + beginy):(j*step + endy)] =\
    #             sargmax_patch[beginx:endx, beginy:endy]
                
    # smax=smax.squeeze()*1024 #anti-normalize(0~1)->(0,1024)
    # smax=smax*(8e-11)*(3e8)/2 # covert to meter
    # # #Remember to change this
    # if name_test==path+'Laundry_2_2.mat':
    #     smax_final=smax[45: s1-35, 38: s2-40]
    #     depth_final=depth[45: s1-35, 38: s2-40]
    # else:
    #     # smax_final=smax[38: s1-29, 30: s2-28]
    #     # depth_final=depth[38: s1-29, 30: s2-28]
    #     smax_final=smax[27: s1-29, 28: s2-28]
    #     depth_final=depth[27: s1-29, 28: s2-28]
    # # plt.imshow(depth_final)   
    # # plt.show()   
    # plt.imshow(smax_final)
    # plt.show()    
    # rmse = np.sqrt(np.mean((smax_final - depth_final)**2))
    # #Accurate-threshold
    # thresh = np.maximum((depth_final / smax_final), (smax_final / depth_final))
    # a1 = (thresh < 1.25   ).mean()
    # a2 = (thresh < 1.25 ** 2).mean()
    # a3 = (thresh < 1.25 ** 3).mean()
    # #rmse_log
    # rmse_log = (np.log(depth_final) - np.log(smax_final)) ** 2
    # rmse_log = np.sqrt(rmse_log.mean())
    # #abs_rel
    # abs_rel = np.mean(np.abs(depth_final - smax_final) / depth_final)
    # #sq_rel
    # sq_rel = np.mean(((depth_final - smax_final)**2) / depth_final)
    # # out = {'smax_final': smax_final,'rmse': rmse}
    # # scipy.io.savemat('./results_middlebury/FuDenoise_Art2_100.mat', out)
    # # print("The RMSE: {}".format(rmse))  
    # # np.save('PredictedResult/smax_wi.npy',smax)
    # return rmse,rmse_log,a1,a2,a3,abs_rel,sq_rel
    
    #%% spadnet method (spad and monocular) works for hr
    depth = scipy.io.loadmat(name_test)['depth'] #(for NYUv2, should devided by 1024)
    depth = np.asarray(depth).astype(np.float32)
    s1, s2 = depth.shape
    
    mono_pred = scipy.io.loadmat(name_test)['mono']
    mono_pred = torch.from_numpy(mono_pred).type(dtype)
    mono_pred = mono_pred.unsqueeze(0).unsqueeze(0)
    mono_pred_var = Variable(mono_pred.type(dtype))

    spad = scipy.io.loadmat(name_test)['spad']
    spad = scipy.sparse.csc_matrix.todense(spad)
    spad = np.asarray(spad).reshape([s2, s1, -1])
    #middlebury
    #spad = torch.from_numpy(np.transpose(spad, (2, 1, 0)))
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
            
            mono_patch = mono_pred_var[:, :, i*step:(i*step + dim1), 
                                            j*step:(j*step + dim2)]
            
            spad_patch = tologscale(spad_patch, NUMBIN, Q)
            mono_rates_patch = dmap2pc(mono_patch, NUMBIN, Q, Linear_NUMBIN)
            mono_rates_patch = Variable(mono_rates_patch.type(dtype))
            
            denoise_out, sargmax_patch = model(spad_patch,mono_rates_patch)
            
            beginx = 0 if (i == 0) else step//2
            endx = dim1 if (i == (num_rows - 1)) else (dim1 - step//2)
            beginy = 0 if (j == 0) else step//2
            endy = dim2 if (j == (num_cols - 1)) else (dim2 - step//2)
            
            sargmax_patch=sargmax_patch.data.cpu().numpy().squeeze()
            
            smax[:, (i*step + beginx):(i*step + endx), (j*step + beginy):(j*step + endy)] =\
                sargmax_patch[beginx:endx, beginy:endy]
                
    smax=smax.squeeze()*1024
    smax=smax*(8e-11)*(3e8)/2 #convert to meters
    #for classroom
    # smax_final=smax[29: s1, 0: s2-24]
    # depth_final=depth[29: s1, 0: s2-24]
    #for md
    #Remember to change this
    if name_test==path+'Laundry_2_100.mat':
        smax_final=smax[45: s1-35, 38: s2-40]
        depth_final=depth[45: s1-35, 38: s2-40]
    else:
        smax_final=smax[27: s1-29, 28: s2-28]
        depth_final=depth[27: s1-29, 28: s2-28]
        # smax_final=smax[38: s1-29, 30: s2-28]
        # depth_final=depth[38: s1-29, 30: s2-28]
    # plt.imshow(depth_final)   
    # plt.show()   
    plt.imshow(smax_final)
    plt.show()    
    #rmse-log
    rmse = np.sqrt(np.mean((smax_final - depth_final)**2))
    #accurate
    thresh = np.maximum((depth_final / smax_final), (smax_final / depth_final))
    a1 = (thresh < 1.25   ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()
    #rmse_log
    rmse_log = (np.log(depth_final) - np.log(smax_final)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())
    #abs_rel
    abs_rel = np.mean(np.abs(depth_final - smax_final) / depth_final)
    #sq_rel
    sq_rel = np.mean(((depth_final - smax_final)**2) / depth_final)
    p_val = ttest_ind(depth_final, smax_final).pvalue
    # out = {'smax_final': smax_final,'rmse': rmse}
    # scipy.io.savemat('./results_middlebury/Fumono_Art2_50.mat', out)
    # print("The RMSE: {}".format(rmse))  
    # np.save('PredictedResult/smax_wmono.npy',smax)
    return rmse,rmse_log,a1,a2,a3,abs_rel,sq_rel,p_val




