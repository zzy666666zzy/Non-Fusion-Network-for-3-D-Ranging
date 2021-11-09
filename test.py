import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.utils.prune as prune
import configparser
from configparser import ConfigParser
from models import DenoiseModel,FusionDenoiseModel, SPADnet
#from new_model_level2 import UnetPP3D
from new_model_level3 import UnetPP3D
#from new_model_level4 import UnetPP3D
from q_new_model_level3 import Q_UnetPP3D
from models import DeepBoosting
from util.test_matrice import test_sm
from util.test_wi import test_wi
from util.test_nyu import test_nyu
from quanti_util import quanti
import scipy
import scipy.io
import os
from glob import glob
import pathlib
from torchsummary import summary

cudnn.benchmark = True
dtype = torch.cuda.FloatTensor

outdir = './results_middlebury/'
pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)

#Quantized model w2 a2
#pre_model='C:/Users/Zhenya/Desktop/NN1_20/logging/Q_level_3W2A2/epoch_3_11999.pth'
#Quantized model w2 a4
#pre_model='C:/Users/Zhenya/Desktop/NN1_20/logging/Q_level_3W2A4/epoch_3_11999.pth'
#Quantized model w4 a8
#pre_model='C:/Users/Zhenya/Desktop/NN1_20/logging/Q_level_3W4A8/epoch_3_9999.pth'

#pre_model='logging/01_31_level4/epoch_3_11199.pth'
pre_model='logging/01_28_level3/epoch_3_11099.pth'#11099->0.228
#pre_model='logging/01_27_level2/epoch_3_12764.pth'
#pre_model=r'E:\Public\lindell_2018_code\code\pth\denoise.pth'
#pre_model=r'E:\Public\lindell_2018_code\code\pth\fusion_denoise.pth'
#pre_model='E:\Public\lindell_2018_code\code1\pth\spadnet.pth' #monocular fusion
#pre_model='C:/Users/Zhenya/Desktop/PENonLocal-master/training/logging/non_local_date_04_27-01_11/epoch_8_27999.pth'

#name_test='E:/Public/lindell_2018_code/code1/middlebury/processed/Bowling1_2_2.mat'
#name_test = 'E:/Public/lindell_2018_code/code1/middlebury/processed/basement_0001a191_2_50.mat'
#name_test=r'C:\Users\Zhenya\Downloads\lindell_2018_data\data\captured\lamp.mat'
#name_test = 'E:\Public\scan_00183\classroom2_50.mat'
#name_test=r'C:\Users\Zhenya\Downloads\Shin\lamp_cell.mat'

#%%
def main():
    # set gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #model = DenoiseModel()
    #model = FusionDenoiseModel()
    #model = SPADnet()
    model = UnetPP3D()
    #model = Q_UnetPP3D()
    #model = DeepBoosting()
        
    model = model.to(device=device)
    model.type(dtype)

    print('=> Loading checkpoint {}'.format(pre_model))
    ckpt = torch.load(pre_model)
    model_dict = model.state_dict()
    try:
        ckpt_dict = ckpt['state_dict']
    except KeyError:
        print('Key error loading state_dict from checkpoint; assuming \
              checkpoint contains only the state_dict')
        ckpt_dict = ckpt

    for k in ckpt_dict.keys():
        model_dict.update({k: ckpt_dict[k]})
    model.load_state_dict(model_dict)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
#%% model-pruning
#     for name, module in model.named_modules():
#         if isinstance(module, torch.nn.Conv3d):
#             prune.l1_unstructured(module, name='weight', amount=0.3)
#             prune.remove(module, 'weight')
# #%%chech para-size
#     torch.save(model, 'pruned_model.pth')
#     model=torch.load('pruned_model.pth')
    #summary(model,(1,1024,64,64))
#%% Test function
    
    path = r'E:\Public\lindell_2018_code\code1\middlebury\2_10' '/' 
    files= os.listdir(path) 
    rmse_list=[]
    log_rmse_list=[]
    a1_list=[]
    a2_list=[]
    a3_list=[]
    abs_rel_list=[]
    sq_rel_list=[]
    p_value_list=[]
    p_mean_list=[]

    for i in range (len(files)):
        rmse,rmse_log,a1,a2,a3,abs_rel,sq_rel,p_val = test_sm(model,path,path+files[i], outdir)
        rmse_list.append(round(rmse,4))
        log_rmse_list.append(round(rmse_log,4))
        a1_list.append(round(a1,4))
        a2_list.append(round(a2,4))
        a3_list.append(round(a3,4))
        abs_rel_list.append(round(abs_rel,4))
        sq_rel_list.append(round(sq_rel,4))
        p_value_list.append(p_val)
        
    rmse_mean=sum(rmse_list)/len(rmse_list)
    log_rmse_mean=sum(log_rmse_list)/len(log_rmse_list)
    a1_mean=sum(a1_list)/len(a1_list)
    a2_mean=sum(a2_list)/len(a2_list)
    a3_mean=sum(a3_list)/len(a3_list)
    abs_rel_mean=sum(abs_rel_list)/len(abs_rel_list)
    sq_rel_mean=sum(sq_rel_list)/len(sq_rel_list)
    
    for i in range(7):
        pal_mean=np.mean(p_value_list[i])
        p_mean_list.append(round(pal_mean,3))
        
    # #the test function for middlebury dataset
    # rmse = test_sm(model,path,name_test, outdir)
    # rmse = test_wi(model, path,name_test, outdir) # for fushion
    # rmse = test_nyu(model, name_test, outdir) # for fushion
    # #the test function for real-world indoor and outdoor dataset
    # rmse, runtime = test_inrw(model, opt, outdir_m)
    # rmse, runtime = test_outrw(model, opt, outdir_m)

    print("processing end")

if __name__ == '__main__':
    main()

