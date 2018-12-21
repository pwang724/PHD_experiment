#!/usr/bin/env python
# coding: utf-8

import numpy as np
from tools import utils

from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

def decode(cons, data, chosen_odors, csp_odors, decode_config):
    def _masks_from_ixs(labels, list_of_ixs):
        list_of_masks = []
        for ixs in list_of_ixs:
            list_of_masks.append(np.isin(labels, ixs))
        return list_of_masks

    def _assign_labels(list_of_masks):
        decode_labels = np.zeros_like(list_of_masks[0]).astype(int)
        for i, mask in enumerate(list_of_masks):
            decode_labels[mask] = i+1
        return decode_labels

    data_trial_cell_time = utils.reshape_data(data, trial_axis=0, cell_axis=1, time_axis=2,
                                              nFrames= cons.TRIAL_FRAMES)
    odor_ix_dict = utils.make_odor_ix_dictionary(cons.ODOR_UNIQUE)
    us = 'water'
    csm_odors = list(set(chosen_odors) - set(csp_odors) - set(us))

    csp1_ix, csp2_ix, csm1_ix, csm2_ix = odor_ix_dict[csp_odors[0]], odor_ix_dict[csp_odors[1]], \
                                         odor_ix_dict[csm_odors[0]], odor_ix_dict[csm_odors[1]]
    if us in odor_ix_dict.keys():
        water_ix = odor_ix_dict[us]
    else:
        water_ix = None

    decode_style = decode_config.decode_style
    decode_neurons = decode_config.neurons
    decode_shuffle = decode_config.shuffle

    labels = cons.ODOR_TRIALIDX
    if decode_style == 'valence':
        list_of_masks = _masks_from_ixs(labels, [[csp1_ix, csp2_ix],[csm1_ix, csm2_ix]])
    elif decode_style == 'identity':
        list_of_masks = _masks_from_ixs(labels, [[csp1_ix], [csp2_ix],[csm1_ix], [csm2_ix]])
    elif decode_style == 'csp_identity':
        list_of_masks = _masks_from_ixs(labels, [[csp1_ix], [csp2_ix]])
    elif decode_style == 'csm_identity':
        list_of_masks = _masks_from_ixs(labels, [[csm1_ix], [csm2_ix]])
    else:
        raise ValueError('Unknown decoding type {:s}'.format(decode_style))

    good_trials = np.any(list_of_masks, axis=0)
    decode_labels = _assign_labels(list_of_masks)
    scores = decode_odors_time_bin(data_trial_cell_time[good_trials], decode_labels[good_trials],
                                   number_of_cells=decode_neurons,
                                   shuffle=decode_shuffle,
                                   cv=5)
    return scores


def decode_odors_time_bin(data, labels, model=None, number_of_cells=None, shuffle=False, **cv_args):
    '''
    Decoding as a function of time.

    :param data: calcium activity, in format of trials X cells X time
    :param labels: the label of the odor trials, with integers defining the label.
                Must be equal to number of trials, or data[0].shape
    :param number_of_cells: int, randomly select a subset of size number_of_cells to keep
    :param shuffle: boolean, whether to shuffle the labels randomly
    :param model: sklearn model. default is SVM with linear kernel
    :param cv_args:
    :return:
    '''

    if model is None:
        model = SVC(kernel='linear')

    if number_of_cells is not None:
        nCells = data.shape[1]
        if number_of_cells > nCells:
            raise ValueError('number of cells for decoding is larger than existing cells in dataset')
        cell_ixs = np.random.choice(nCells, size=number_of_cells, replace=False)
        data = data[:,cell_ixs,:]

    if shuffle == True:
        labels = np.random.permutation(labels)

    scores = []
    for t in range(data.shape[-1]):
        scores.append(cross_val_score(model,
                                      data[:, :, t],
                                      labels,
                                      **cv_args))
    return np.r_[scores]
