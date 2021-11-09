# -*- coding: utf-8 -*-
import torch
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from torch.autograd import Variable

dtype = torch.cuda.FloatTensor

rates = scipy.io.loadmat(r'C:/Users/Zhenya/Desktop/NN/spad_0351_p10.mat')['spad']
rates = scipy.sparse.csc_matrix.todense(rates)

Linear_NUMBIN = 1024
NUMBIN = 128
Q = 1.02638 ## Solution for (q^128 - 1) / (q - 1) = 1024

## convert pc to log scale (log rebinning)

bin_idx = np.arange(1, NUMBIN + 1)
up = np.floor((np.power(Q, bin_idx) - 1) / (Q - 1))
low = np.floor((np.power(Q, bin_idx - 1) - 1) / (Q - 1))

log_rates = np.zeros((4096,NUMBIN))
for ii in range(NUMBIN):
    log_rates[:,ii] = np.sum(rates[:,int(low[ii]):int(up[ii])], axis = 1).squeeze()


# plt.imshow(rates)
# plt.show()

