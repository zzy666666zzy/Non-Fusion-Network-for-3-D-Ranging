import numpy as np
from glob import glob
import re
import os.path

# specify dataset folder here
# that contains the output of
# SimulateSpadMeasurements.m
dataset_folder = os.path.abspath(r'E:\Public\lindell_2018_code\code1\simulated_data\processed') + '/'

# this value should be set to whatever param_idx
# values were simulated with SimulateSpadMeasurements.m
# If multiple values were simulated, enter them in a list
# e.g. [value1, value2, value3]
# The below code takes the intersection of the output scenes
# to ensure that the same number of scene files are used
# for training at each param_idx.
simulation_param_idx = 2

def intersect_files(train_files):
    intensity_train_files = []
    for t in train_files:
        #Return a list of paths matching a pathname pattern
        intensity_train_files.append(glob(dataset_folder + t + 'intensity*.mat'))
    intensity_train_files = [file for sublist in intensity_train_files for file in sublist]

    spad_train_files = []
    if simulation_param_idx is not None:
        noise_param = [simulation_param_idx]

    for p in noise_param:
        spad_train_files.append([])
        for t in train_files:
            spad_train_files[-1].append(glob(dataset_folder + t + 'spad*p{}.mat'.format(p)))
        spad_train_files[-1] = [file for sublist in spad_train_files[-1] for file in sublist]
        spad_train_files[-1] = [re.sub(r'spad_(.*)_p.*.mat', r'intensity_\1.mat',
                                 file) for file in spad_train_files[-1]]

    intensity_train_files = set(intensity_train_files)
    for idx, p in enumerate(noise_param):
        spad_train_files[idx] = set(spad_train_files[idx])
    intensity_train_files = intensity_train_files.intersection(*tuple(spad_train_files))
    return intensity_train_files
#%%
def main():

    with open('train.txt') as f:
        train_files = f.read().split()
    with open('val.txt') as f:
        val_files = f.read().split()

    print('Sorting training files')
    intensity_train_files = intersect_files(train_files)
    print('Sorting validation files')
    intensity_val_files = intersect_files(val_files)

    print('Writing training files')
    with open('train_intensity.txt', 'w') as f:
        for file in intensity_train_files:
            f.write(file + '\n')
    print('Writing validation files')
    with open('val_intensity.txt', 'w') as f:
        for file in intensity_val_files:
            f.write(file + '\n')

    print('Wrote {} train, {} validation files'.format(len(intensity_train_files),
                                                       len(intensity_val_files)))
    return

if __name__ == '__main__':
    main()






