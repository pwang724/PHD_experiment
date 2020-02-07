import numpy as np
from tools import utils

from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from collections import defaultdict
import filter

def _get_odor_masks(cons, chosen_odors, csp_odors, decode_style):
    def _masks_from_ixs(labels, list_of_ixs):
        list_of_masks = []
        for ixs in list_of_ixs:
            list_of_masks.append(np.isin(labels, ixs))
        return list_of_masks

    odor_ix_dict = utils.make_odor_ix_dictionary(cons.ODOR_UNIQUE)
    if decode_style == 'identity':
        odor_ixs = []
        for odor in chosen_odors:
            odor_ixs.append([odor_ix_dict[odor]])
    else:
        us = 'water'
        csm_odors = list(set(chosen_odors) - set(csp_odors) - set(us))

        csp1_ix, csp2_ix, csm1_ix, csm2_ix = odor_ix_dict[csp_odors[0]], odor_ix_dict[csp_odors[1]], \
                                             odor_ix_dict[csm_odors[0]], odor_ix_dict[csm_odors[1]]
        if us in odor_ix_dict.keys():
            water_ix = odor_ix_dict[us]
        else:
            water_ix = None

        if decode_style == 'valence':
            odor_ixs = [[csp1_ix, csp2_ix],[csm1_ix, csm2_ix]]
        elif decode_style == 'csp_identity':
            odor_ixs = [[csp1_ix], [csp2_ix]]
        elif decode_style == 'csm_identity':
            odor_ixs = [[csm1_ix], [csm2_ix]]
        else:
            raise ValueError('Unknown decoding type {:s}'.format(decode_style))

    labels = cons.ODOR_TRIALIDX
    list_of_masks = _masks_from_ixs(labels, odor_ixs)
    return list_of_masks

def _assign_odor_labels(list_of_masks):
    decode_labels = np.zeros_like(list_of_masks[0]).astype(int)
    for i, mask in enumerate(list_of_masks):
        decode_labels[mask] = i+1
    return decode_labels

def decode_odor_labels(cons, data, chosen_odors, csp_odors, decode_config):
    '''

    :param cons:
    :param data:
    :param chosen_odors:
    :param csp_odors:
    :param decode_config:
    :return: decoding: returns CV scores in dimensions of Time X Cross-Folds X Repeats
    '''

    decode_style = decode_config.decode_style
    decode_neurons = decode_config.neurons
    decode_shuffle = decode_config.shuffle
    decode_repeat = decode_config.repeat
    decode_average_time = decode_config.average_time
    data_trial_cell_time = utils.reshape_data(data, trial_axis=0, cell_axis=1, time_axis=2,
                                              nFrames=cons.TRIAL_FRAMES)

    list_of_masks= _get_odor_masks(cons, chosen_odors, csp_odors, decode_style)
    good_trials = np.any(list_of_masks, axis=0)
    decode_labels = _assign_odor_labels(list_of_masks)[good_trials]
    input = data_trial_cell_time[good_trials]
    if decode_average_time:
        on = cons.DAQ_O_ON_F
        if decode_config.no_end_time:
            input = np.max(input[:, :, on:], axis=2, keepdims=True)
        else:
            off = cons.DAQ_W_ON_F
            input = np.max(input[:, :, on:off], axis=2, keepdims=True)

    list_of_scores = []
    for _ in range(decode_repeat):
        scores = decode_odors_time_bin(input, decode_labels,
                                       number_of_cells=decode_neurons,
                                       shuffle=decode_shuffle,
                                       cv=5)
        list_of_scores.append(scores)
    out = np.stack(list_of_scores, axis=2)
    return out

def test_split(list_of_cons, list_of_data, chosen_odors, csp_odors, decode_config):
    def _helper(cons, traces):
        list_of_masks = _get_odor_masks(cons, chosen_odors, csp_odors, decode_style)
        data_trial_cell_time = utils.reshape_data(traces, trial_axis=0, cell_axis=1, time_axis=2,
                                                  nFrames=cons.TRIAL_FRAMES)
        if average_time:
            on = cons.DAQ_O_ON_F
            off = cons.DAQ_W_ON_F
            if decode_config.no_end_time:
                data_trial_cell_time = np.max(data_trial_cell_time[:, :, on:], axis=2, keepdims=True)
            else:
                data_trial_cell_time = np.max(data_trial_cell_time[:,:,on:off], axis=2, keepdims=True)

        start = np.round(cons.DAQ_O_ON * cons.DAQ_SAMP).astype(int)
        end = np.round(cons.DAQ_W_ON * cons.DAQ_SAMP).astype(int)
        licks = cons.DAQ_DATA[start:end,cons.DAQ_L,:] > 0.5
        licks = np.transpose(licks)

        if decode_config.split_style == 'onset':
            factor = []
            for l in licks:
                if np.any(l):
                    loc = np.where(l)[0][0]
                else:
                    loc = 1000
                factor.append(loc)
            factor = np.array(factor)
        elif decode_config.split_style == 'time':
            factor = np.arange(len(licks))
        elif decode_config.split_style == 'magnitude':
            on_off = np.diff(licks, n=1, axis=1)
            factor = np.sum(on_off > 0, axis=1)
        else:
            raise ValueError('split style {} not recognized'.format(decode_config.split_style))

        data = []
        labels = []
        for mask in list_of_masks:
            factor_current_odor = factor[mask]
            ixs = np.argsort(factor_current_odor)
            cutoff = len(ixs)//2

            current_data = data_trial_cell_time[mask][ixs]
            current_labels = np.ones_like(ixs)
            current_labels[cutoff:] = 2
            data.append(current_data)
            labels.append(current_labels)

        return data, labels

    decode_style = decode_config.decode_style
    decode_neurons = decode_config.neurons
    decode_shuffle = decode_config.shuffle
    decode_repeat = decode_config.repeat
    average_time = decode_config.average_time

    res = defaultdict(list)
    for day, _ in enumerate(list_of_cons):
        data, labels = _helper(list_of_cons[day], list_of_data[day])
        for d, l in zip(data, labels):
            for r in range(decode_repeat):
                scores = decode_odors_time_bin(d, l, number_of_cells=decode_neurons, cv=5)
                scores = np.mean(scores)
                res['test_day'].append(day)
                res['scores'].append(scores)

    for key, val in res.items():
        res[key] = np.array(val)
    return res

def test_fp_fn(list_of_cons, list_of_data, chosen_odors, csp_odors, decode_config, list_of_ixs = None):
    def _helper(cons, traces):
        list_of_masks = _get_odor_masks(cons, chosen_odors, csp_odors, decode_style)
        data_trial_cell_time = utils.reshape_data(traces, trial_axis=0, cell_axis=1, time_axis=2,
                                                  nFrames=cons.TRIAL_FRAMES)
        if average_time:
            on = cons.DAQ_O_ON_F
            off = cons.DAQ_W_ON_F
            if decode_config.no_end_time:
                data_trial_cell_time = np.max(data_trial_cell_time[:, :, on:], axis=2, keepdims=True)
            else:
                data_trial_cell_time = np.max(data_trial_cell_time[:,:,on:off], axis=2, keepdims=True)

        licks = cons.DAQ_DATA[:,cons.DAQ_L,:] > 0.5
        start = np.round(cons.DAQ_O_ON * cons.DAQ_SAMP).astype(int)
        end = np.round(cons.DAQ_W_ON * cons.DAQ_SAMP).astype(int)
        lick_boolean = np.sum(licks[start:end], axis=0) > 0

        csp_lick = np.bitwise_and(list_of_masks[0], lick_boolean)
        csp_nolick = np.bitwise_and(list_of_masks[0], np.invert(lick_boolean))
        csm_lick = np.bitwise_and(list_of_masks[1], lick_boolean)
        csm_nolick = np.bitwise_and(list_of_masks[1], np.invert(lick_boolean))

        train_trials = np.bitwise_or(csp_lick, csm_nolick)
        train_labels = _assign_odor_labels([csp_lick, csm_nolick])[train_trials]
        train_data = data_trial_cell_time[train_trials]

        test_trials = np.bitwise_or(csp_nolick, csm_lick)
        test_labels = _assign_odor_labels([csp_nolick, csm_lick])[test_trials]
        test_data = data_trial_cell_time[test_trials]
        return train_data, train_labels, test_data, test_labels

    decode_style = decode_config.decode_style
    decode_shuffle = decode_config.shuffle
    decode_repeat = decode_config.repeat
    average_time = decode_config.average_time
    minimum_cell_number = 7
    if list_of_ixs is None:
        decode_neurons = decode_config.neurons
    else:
        decode_neurons = minimum_cell_number

    res = defaultdict(list)
    for day, _ in enumerate(list_of_cons):
        train_data, train_labels, test_data, test_labels = _helper(list_of_cons[day], list_of_data[day])
        if list_of_ixs is not None:
            ixs = list_of_ixs[day].astype(np.bool)
            train_data = train_data[:,ixs,:]
            test_data = test_data[:,ixs,:]
            print('number of US responsive neurons: {}'.format(np.sum(ixs)))
            # assert decode_neurons <= np.sum(ixs), print('number of US responsive neurons: {}'.format(np.sum(ixs)))
            if decode_neurons > np.sum(ixs):
                continue

        arg = ['CS+','CS-']
        for i, current_label in enumerate([1,2]):
            ix = test_labels == current_label
            data = test_data[ix]
            labels = test_labels[ix]

            if len(labels):
                for r in range(decode_repeat):
                    if decode_shuffle:
                        scores = decode_odors_time_bin(train_data, train_labels, number_of_cells=decode_neurons, cv=5)
                        scores = np.mean(scores)
                    else:
                        nCells = train_data.shape[1]
                        cell_ixs = np.random.choice(nCells, size=decode_neurons, replace=False)
                        subset_of_train_data = train_data[:, cell_ixs, :]
                        models = model_odors_time_bin(subset_of_train_data, train_labels, shuffle=False)
                        scores = test_odors_time_bin(data[:, cell_ixs, :], labels, models)
                    res['test_day'].append(day)
                    res['scores'].append(scores)
                    res['n'].append(len(labels))
                    res['odor_valence'].append(arg[i])
            else:
                print('no data for mouse: {}, day: {}, valence: {}'.format(list_of_cons[day].NAME_MOUSE, day, arg[i]))

    for key, val in res.items():
        res[key] = np.array(val)
    return res

def test_odors_across_days(list_of_cons, list_of_data, chosen_odors, csp_odors, decode_config):
    def _helper(day):
        list_of_masks = _get_odor_masks(list_of_cons[day], chosen_odors, csp_odors, decode_style)
        data_trial_cell_time = utils.reshape_data(list_of_data[day], trial_axis=0, cell_axis=1, time_axis=2,
                                                  nFrames=list_of_cons[day].TRIAL_FRAMES)
        good_trials = np.any(list_of_masks, axis=0)
        labels = _assign_odor_labels(list_of_masks)[good_trials]
        data = data_trial_cell_time[good_trials]
        if average_time:
            on = list_of_cons[day].DAQ_O_ON_F
            off = list_of_cons[day].DAQ_W_ON_F
            if decode_config.no_end_time:
                data = np.max(data[:, :, on:], axis=2, keepdims=True)
            else:
                data = np.max(data[:,:,on:off], axis=2, keepdims=True)
        return data, labels

    decode_style = decode_config.decode_style
    decode_neurons = decode_config.neurons
    decode_shuffle = decode_config.shuffle
    decode_repeat = decode_config.repeat
    average_time = decode_config.average_time

    res = defaultdict(list)
    for train_day, _ in enumerate(list_of_cons):
        data, labels = _helper(train_day)
        for r in range(decode_repeat):
            nCells = data.shape[1]
            cell_ixs = np.random.choice(nCells, size=decode_neurons, replace=False)
            models = model_odors_time_bin(data[:, cell_ixs, :], labels, shuffle=decode_shuffle)

            for test_day, _ in enumerate(list_of_cons):
                test_data, test_labels = _helper(test_day)
                if test_day != train_day:
                    scores = test_odors_time_bin(test_data[:, cell_ixs, :], test_labels, models)
                else:
                    if decode_shuffle:
                        test_labels = np.random.permutation(test_labels)
                    scores = decode_odors_time_bin(test_data, test_labels, number_of_cells=decode_neurons, cv=5)
                    scores = np.mean(scores)
                res['Training Day'].append(train_day)
                res['Test Day'].append(test_day)
                res['scores'].append(scores)
    for key, val in res.items():
        res[key] = np.array(val)
    return res

def decode_day_labels(list_of_cons, list_of_data, chosen_odors, csp_odors, decode_config):
    #TODO: obsolete, need to fix before using
    '''

    :param cons:
    :param data:
    :param chosen_odors:
    :param csp_odors:
    :param decode_config:
    :return: decoding: returns CV scores in dimensions of Time X Cross-Folds X Repeats
    '''
    decode_style = decode_config.decode_style
    decode_neurons = decode_config.neurons
    decode_shuffle = decode_config.shuffle
    decode_repeat = decode_config.repeat

    day_odor_masks = []
    for i, cons in enumerate(list_of_cons):
        list_of_masks = _get_odor_masks(cons, chosen_odors, csp_odors, decode_style)
        day_odor_masks.append(list_of_masks)
    odor_day_masks = np.swapaxes(np.array(day_odor_masks), 0, 1)

    list_of_reshaped_data = []
    for data, cons in zip(list_of_data, list_of_cons):
        data_trial_cell_time = utils.reshape_data(data, trial_axis=0, cell_axis=1, time_axis=2,
                                              nFrames=cons.TRIAL_FRAMES)
        list_of_reshaped_data.append(data_trial_cell_time)

    list_of_scores = []
    for i, day_masks in enumerate(odor_day_masks):
        cur_data = []
        cur_label = []
        for j, day_mask in enumerate(day_masks):
            cur_data.append(list_of_reshaped_data[j][day_mask])
            cur_label.append(np.ones(np.sum(day_mask)) * j)
        cur_data = np.concatenate(cur_data, axis=0)
        cur_label = np.concatenate(cur_label, axis=0)

        for _ in range(decode_repeat):
            scores = decode_odors_time_bin(cur_data, cur_label,
                                           number_of_cells=decode_neurons,
                                           shuffle=decode_shuffle,
                                           cv=5)
            list_of_scores.append(scores)
    out = np.stack(list_of_scores, axis=2)
    return out

def model_odors_time_bin(data, labels, model=None, shuffle=False):
    '''

    :param data: calcium activity, in format of trials X cells X time
    :param labels: the label of the odor trials, with integers defining the label.
                Must be equal to number of trials, or data[0].shape
    :param number_of_cells: int, randomly select a subset of size number_of_cells to keep
    :param shuffle: boolean, whether to shuffle the labels randomly
    :param model: sklearn model. default is SVM with linear kernel
    :return:
    '''
    if model is None:
        model = SVC(kernel='linear')

    if shuffle == True:
        labels = np.random.permutation(labels)

    models = []
    for t in range(data.shape[-1]):
        models.append(
            model.fit(X=data[:,:,t], y=labels)
        )
    return models

def test_odors_time_bin(data, labels, models):
    scores = []
    for t, model in enumerate(models):
        scores.append(model.score(data[:,:,t], labels))
    return np.array(scores)


def decode_odors_time_bin(data, labels, model=None, number_of_cells=None, shuffle=False, **cv_args):
    '''
    Decoding odor identity as a function of time.

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
            raise ValueError('number of cells for decoding: {} is larger than '
                             'existing cells in dataset: {}'.format(
                number_of_cells, nCells))
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





