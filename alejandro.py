import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import analysis
import tools.utils as utils

class OFC_SINGLE_PHASE:
    # naive odor presentation: day 0
    # training start: day 1
    # fully learned: day 4
    path = r'C:\Users\P\Desktop\PYTHON\PHD_experiment\_DATA\timepoint\OFC' #specify path on your local machine
    odors = ['pin', 'msy', 'euy', 'lim']
    csp = ['pin', 'msy']

class BLA_SINGLE_PHASE:
    # naive odor presentation: day 0
    # training start: day 1
    # fully learned: day 4
    path = r'C:\Users\P\Desktop\PYTHON\PHD_experiment\_DATA\timepoint\BLA' #specify path on your local machine
    odors = ['pin', 'msy', 'euy', 'lim']
    csp = ['pin', 'msy']

class OFC_TWO_PHASE:
    # naive odor presentation: day 0
    # pretraining start: day 1. the naive odor as referenced are trials of octanol delivered without reward
    # pretraining fully learned: day 4
    # discrimination start day: day 4
    # discrimination fully learned: day 5
    path = r'C:\Users\P\Desktop\PYTHON\PHD_experiment\_DATA\timepoint\OFC_COMPOSITE' #specify path on your local machine
    pretraining_odors = ['naive', 'oct']
    discrimination_odors = ['pin', 'msy', 'euy', 'lim']
    csp = ['oct','pin', 'msy']

class MPFC_TWO_PHASE:
    # naive odor presentation: day 0
    # pretraining start: day 1
    # pretraining fully learned: day 3
    # discrimination start day: day 4
    # discrimination fully learned: day 5
    path = r'C:\Users\P\Desktop\PYTHON\PHD_experiment\_DATA\timepoint\MPFC_COMPOSITE' #specify path on your local machine
    pretraining_odors = ['oct']
    discrimination_odors = ['pin', 'msy', 'euy', 'lim']
    csp = ['oct','pin', 'msy']



condition = BLA_SINGLE_PHASE()


res = analysis.load_data(condition.path)
analysis.add_indices(res)
analysis.add_time(res)

for i, data in enumerate(res['data']):
    frames_per_trial = res['TRIAL_FRAMES'][i]
    data_r = utils.reshape_data(data, nFrames=frames_per_trial, cell_axis=0, trial_axis=1, time_axis=2)
    res['data'][i] = data_r

print(res.keys())
print(res['day'])

day = 1
print('Imaging day = {}'.format(res['day'][day]))
print('corresponding folder name = {}'.format(res['DIR'][day]))

print('frame number for when odor is on. not applicable for water trials = {}'.format(res['DAQ_O_ON_F'][day]))
print('frame number for when odor is off. not applicable for water trials = {}'.format(res['DAQ_O_OFF_F'][day]))
print('frame number for when water is on. only applicable for CS+ odors and water = {}'.format(res['DAQ_W_ON_F'][day]))

print('names of the odors = {}'.format(res['ODOR_TRIALS'][day]))
print('raw data. dimensions are cells X trials X frames = {}'.format(res['data'][day].shape))


