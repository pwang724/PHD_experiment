import numpy as np

def sort_by_selectivity(list_of_psths, odor_on, water_on, condition_config, delete_nonselective = False):
    list_of_ixs = []
    for psth in list_of_psths:
        max_dff = np.max(psth[:, odor_on:water_on], axis=1)
        ixs = np.argsort(max_dff)[::-1]
        cutoff = np.argmin(max_dff[ixs] > condition_config.threshold)
        list_of_ixs.append(ixs[:cutoff])

    #respond to all
    if delete_nonselective:
        respond_to_all = list(set.intersection(*[set(x) for x in list_of_ixs]))
    else:
        respond_to_all = []

    for psth in list_of_psths:
        max_dff = np.min(psth[:, odor_on:water_on], axis=1)
        ixs = np.argsort(max_dff)[::-1]
        cutoff = np.argmin(max_dff[ixs] > condition_config.negative_threshold)
        list_of_ixs.append(ixs[:cutoff])

    list_of_ixs.append(np.arange(list_of_psths[0].shape[0]))
    final_ixs = _sort_by_ixs(list_of_ixs)
    final_ixs = np.array([x for x in final_ixs if x not in respond_to_all])
    return final_ixs

def sort_by_onset(list_of_psths, odor_on, water_on, condition_config):
    list_of_argmax = []
    cutoff = 3
    for psth in list_of_psths:
        binary_psth = psth[:, odor_on:water_on+ cutoff] > condition_config.threshold
        responsive = np.any(binary_psth, axis=1)
        argmax = np.argmax(binary_psth == 1, axis=1)
        argmax[np.invert(responsive)] = 100
        list_of_argmax.append(argmax)

    list_of_argmin = []
    for psth in list_of_psths:
        binary_psth = psth[:, odor_on:water_on+ cutoff] < condition_config.negative_threshold
        responsive = np.any(binary_psth, axis=1)
        argmax = np.argmax(binary_psth == 1, axis=1)
        argmax[np.invert(responsive)] = 100
        list_of_argmin.append(argmax)

    list_of_ixs = []
    cutoff_ix = water_on - odor_on - 1 + cutoff
    if condition_config.sort_onset_style == 'individual':
        for argmax in list_of_argmax:
            ixs = np.argsort(argmax)
            cutoff = np.argmax(argmax[ixs] > cutoff_ix)
            list_of_ixs.append((ixs[:cutoff]))

    if condition_config.sort_onset_style == 'CS+':
        if condition_config.period == 'pt':
            number_of_odors = 1
        else:
            number_of_odors = 2
        argmax = np.mean(list_of_argmax[0:number_of_odors], axis=0)
        ixs = np.argsort(argmax)
        cutoff = np.argmax(argmax[ixs] > cutoff_ix)
        list_of_ixs.append((ixs[:cutoff]))

        argmin = np.mean(list_of_argmin[0:number_of_odors], axis=0)
        ixs = np.argsort(argmin)[::-1]
        cutoff = np.argmin(argmin[ixs] > cutoff_ix)
        list_of_ixs.append((ixs[:cutoff]))

    list_of_ixs.append(np.arange(list_of_psths[0].shape[0]))
    final_ixs = _sort_by_ixs(list_of_ixs)
    return final_ixs

def sort_by_plus_minus(list_of_psths, odor_on, water_on, condition_config):
    list_of_argmax = []
    for psth in list_of_psths:
        binary_psth = psth[:, odor_on:water_on] > condition_config.threshold
        argmax = _sort_onset(binary_psth, odor_on, water_on, condition_config)
        list_of_argmax.append(argmax)

    list_of_argmin = []
    for psth in list_of_psths:
        binary_psth = psth[:, odor_on:water_on] < condition_config.negative_threshold
        argmax = _sort_onset(binary_psth, odor_on, water_on, condition_config)
        list_of_argmin.append(argmax)

    list_of_ixs = []
    cutoff_ix = water_on - odor_on - 4
    argmax = np.mean(list_of_argmax[0:2], axis=0)
    ixs = np.argsort(argmax)
    cutoff = np.argmax(argmax[ixs] > cutoff_ix)
    list_of_ixs.append((ixs[:cutoff]))

    argmax = np.mean(list_of_argmax[2:4], axis=0)
    ixs = np.argsort(argmax)
    cutoff = np.argmax(argmax[ixs] > cutoff_ix)
    list_of_ixs.append((ixs[:cutoff]))

    argmin = np.mean(list_of_argmin[0:4], axis=0)
    ixs = np.argsort(argmin)[::-1]
    cutoff = np.argmin(argmin[ixs] > cutoff_ix)
    list_of_ixs.append((ixs[:cutoff]))

    list_of_ixs.append(np.arange(list_of_psths[0].shape[0]))
    final_ixs = _sort_by_ixs(list_of_ixs)
    return final_ixs

def _sort_onset(binary_psth, odor_on, water_on, condition_config):
    responsive = np.any(binary_psth, axis=1)
    argmax = np.argmax(binary_psth == 1, axis=1)
    argmax[np.invert(responsive)] = 100
    return argmax


def sort_max(psth, odor_on, water_on, condition_config):
    max = np.max(psth[:, odor_on:water_on])
    argmax = np.argmax(max)
    return argmax



def _sort_by_ixs(list_of_ixs):
    final_ixs = []
    for ixs in list_of_ixs:
        for ix in ixs:
            if ix not in final_ixs:
                final_ixs.append(ix)
    return final_ixs