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
from scipy.stats import ttest_ind


dtype = torch.cuda.FloatTensor
#%%
# test function for Middlebury dataset, all data are linear presented
def test_sm(model,path, name_test,outdir_m):
#%%(works for lr)
    # depth = scipy.io.loadmat(name_test)['depth'] #(for NYUv2, should devided by 1024)
    # depth = np.asarray(depth).astype(np.float32)
    # s1, s2 = depth.shape #72 88

    # spad = scipy.io.loadmat(name_test)['spad']
    # spad = scipy.sparse.csc_matrix.todense(spad)
    # spad = np.asarray(spad).reshape([s2, s1, -1])

    # spad = torch.from_numpy(np.transpose(spad, (2, 1, 0)))
    # spad = spad.unsqueeze(0).unsqueeze(0)#dim->(1,1,1024,72,88)
    # spad_var = Variable(spad.type(dtype))
    
    # denoise_out, sargmax = model(spad_var)
    # denoise = np.argmax(denoise_out.data.cpu().numpy(), axis=1)
    # denoise = denoise.squeeze()
    # smax = sargmax.data.cpu().numpy().squeeze()*1024 #Normalized to real(for NYUv2, dun time 1024)
    
    # smax=smax*(9.76e-11)*(3e8)/2 #(for MB testset, for NYUv2 times 1024 further,RMSR(m))
    # rmse = np.sqrt(np.mean((smax - depth)**2)) #(for NYUv2, should times 12.276 to meter)
    # print("The RMSE: {}".format(rmse))
    # plt.imshow(depth)
    # plt.show()
    # plt.imshow(smax)
    # plt.show()
    #%% spadnet strategy, works for hr
    depth = scipy.io.loadmat(name_test)['depth']
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
    smax=smax.squeeze()*1024 #denormalise
    smax=smax*(80e-12)*(3e8)/2
    
    #for classroom
    # smax_final=smax[29: s1, 0: s2-24]
    # depth_final=depth[29: s1, 0: s2-24]
    #for md
    #Remember to change this
    if name_test==path+'Laundry_2_10.mat':
        smax_final=smax[45: s1-35, 38: s2-40]
        depth_final=depth[45: s1-35, 38: s2-40]
    else:
        smax_final=smax[29: s1-27, 28: s2-28]
        depth_final=depth[29: s1-27, 28: s2-28]
        # smax_final=smax[38: s1-29, 30: s2-28]
        # depth_final=depth[38: s1-29, 30: s2-28]
        
    # plt.imshow(depth_final)   
    # plt.show()   
    plt.imshow(smax_final)
    plt.show()    
    
    rmse = np.sqrt(np.mean((smax_final - depth_final)**2))
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
    #p-value
    p_val = ttest_ind(depth_final, smax_final).pvalue
    # out = {'smax_final': smax_final,'rmse': rmse}
    # scipy.io.savemat('./results_middlebury/FuDenoise_Art2_100.mat', out)
    # print("The RMSE: {}".format(rmse))  
    # np.save('PredictedResult/smax_wi.npy',smax)
    return rmse,rmse_log,a1,a2,a3,abs_rel,sq_rel,p_val
    
    #%%sensor fusion strategy, works for hr
    # intensity = scipy.io.loadmat(name_test)['intensity']
    # intensity = np.asarray(intensity).astype(np.float32)
    # s1, s2 = intensity.shape
    
    # depth = scipy.io.loadmat(name_test)['depth']
    # depth = np.asarray(depth).astype(np.float32)

    # spad = scipy.io.loadmat(name_test)['spad']
    # spad = scipy.sparse.csc_matrix.todense(spad)
    # spad = np.asarray(spad).reshape([s2, s1, -1])

    # spad = torch.from_numpy(np.transpose(spad, (2, 1, 0)))
    # spad = spad.unsqueeze(0).unsqueeze(0)
    # spad_var = Variable(spad.type(dtype))

    # batchsize = 1
    # dim1 = 64
    # dim2 = 64
    # step = 32
    # num_rows = int(np.floor((s1 - dim1)/step + 1))
    # num_cols = int(np.floor((s2 - dim2)/step + 1))
    # im = np.zeros((s1, s2))
    # smax_im = np.zeros((s1, s2))
    # for i in tqdm(range(num_rows)):
    #     for j in range(0, num_cols, batchsize):
    #         # set dimensions
    #         begin_idx = step//2
    #         end_idx = dim1 - step//2
    #         b_idx = 0
    #         for k in range(batchsize):
    #             test = s2 - ((j+k)*step + dim2)
    #             if test >= 0:
    #                 b_idx += 1
    #         iter_batchsize = b_idx

    #         sp1 = Variable(torch.zeros(iter_batchsize,
    #                                     1, 1024, dim1, dim2))
    #         for k in range(iter_batchsize):
    #             sp1[k, :, :, :, :] = spad_var[:, :, :, i*step:(i)*step + dim1,
    #                                           (j+k)*step:(j+k)*step + dim2]

    #         denoise_out, sargmax = model(sp1.type(dtype))
            
    #         denoise = np.argmax(denoise_out.data.cpu().numpy(), axis=1)
    #         denoise = denoise.squeeze()
    #         smax = sargmax.data.cpu().numpy().squeeze()
            
    #         im[i*step:(i+1)*step, (j+k)*step:(j+k+1)*step] = \
    #             denoise[begin_idx:end_idx, begin_idx:end_idx].squeeze()
    #         smax_im[i*step:(i+1)*step, (j+k)*step:(j+k+1)*step] = \
    #             smax[begin_idx:end_idx, begin_idx:end_idx].squeeze()
                
    # smax_im=smax_im.squeeze()*1024
    # smax_im=smax_im*(9.76e-11)*(3e8)/2
    # smax_final=smax_im[6: s1-31, 1: s2-31]
    # depth_final=depth[6: s1-31, 1: s2-31]
    # plt.imshow(smax_final)
    # plt.show()
    # rmse = np.sqrt(np.mean((smax_final - depth_final)**2))
    # print("The RMSE: {}".format(rmse))

    #return out
# -*- coding: utf-8 -*-

