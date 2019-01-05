#!/usr/bin/env python
# coding: utf-8

import os
import pickle
import itertools
import numpy as np

def make_odor_ix_dictionary(list_of_odors):
    dict = {}
    for i, odor in enumerate(list_of_odors):
        dict[odor] = i
    return dict

def reshape_data(data, trial_axis, cell_axis, time_axis, nFrames):
    '''
    reshape data with axis specified by trial, cell, and time axis arguments
    :param data: calcium data. Dim 0 = cells. Dim 1 = trials X conditions
    :return:
    '''
    # TODO: unit tests
    data_cell_trial_time = data.reshape(data.shape[0], data.shape[1] // nFrames, nFrames)
    permuted_ixs = np.argsort((cell_axis, trial_axis, time_axis))
    permuted = np.transpose(data_cell_trial_time, permuted_ixs)
    return permuted


def load_variable(notebook, name, folder='../autorestore'):
    with open(os.path.join(folder, notebook, name)) as f:
        toret = pickle.load(f)
    return toret


def dist_uv(u, v): 
    return np.sqrt(np.sum((u-v)**2, 1)) 


def smooth_variable(x, window=7):
    xs = np.convolve(x, 1.*np.ones(window)/window, mode='same')
    #xs[:window/2] = x[:window/2]
    #xs[-window/2:] = x[-window/2:]
    return xs


def compute_ranking(cell_values, randomize=False):
    if randomize:
        # adds a small value to randomize discrete valued rankings
        ranking = np.argsort(cell_values + np.random.rand()*1e-3)[::-1]
    else:
        ranking = np.argsort(cell_values)[::-1]
    order = np.r_[[np.where(cell==ranking)[0]
                   for cell in range(len(cell_values))]].flatten()
    return ranking, order


def compute_weights(model, normed=True, norm='l2'):
    """
    Compute weights as the mean of the absolute values of the weights for each cell.
    If normed is True, each classifier's weights are normalized to 1.
    """
    if normed:
        if norm == 'l2':
            normz = np.r_[[np.sqrt(np.dot(c, c)) for c in model.coef_]]
        if norm == 'l1':
            normz = np.r_[[np.sum(abs(c)) for c in model.coef_]]
    else:
        normz = 1
    return np.r_[[(abs(c)/normz).mean() for c in model.coef_.T]]


def convert_matrix_to_events(matrix, time_ax):
    events = []
    for cell in range(matrix.shape[1]):
        which = matrix[:, cell]>0
        events.append(np.c_[time_ax[which],  # event times
                            [cell]*(which.sum()),  # cell id
                            matrix[:, cell][which]])  # magnitude
    return np.row_stack(events)


def convert_events_to_matrix(events, time_ax, n_cells=None, usemag=False):
    if n_cells is None:
        n_cells = len(np.unique(events[:, 1]))
    matrix = np.zeros((len(time_ax), n_cells))
    time_ids = [np.argmin(abs(e-time_ax)) for e in events[:, 0]]
    for e, t in zip(events, time_ids):
        matrix[t][int(e[1])] += e[2] if usemag else 1
    return matrix


def append_defaultdicts(dictA, dictB):
    #TODO: should this be in np array format
    for k in dictB.keys():
        dictA[k].append(dictB[k])

def chain_defaultdicts(dictA, dictB):
    for k in dictB.keys():
        dictA[k] = list(itertools.chain(dictA[k], dictB[k]))
    for key, val in dictA.items():
        dictA[key] = np.array(val)