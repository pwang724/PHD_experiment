#!/usr/bin/env python
# coding: utf-8

import numpy as np

from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score


def decode_odors_time_bin(data, labels, good_trials=None, model=None, **cv_args):
    if model is None:
        model = SVC(kernel='linear')
    if good_trials is None:
        good_trials = [True]*data.shape[0]
    scores = []
    for t in range(data.shape[-1]):
        scores.append(cross_val_score(model,
                                      data[:, :, t][good_trials],
                                      labels[good_trials],
                                      **cv_args))
    return np.r_[scores]
