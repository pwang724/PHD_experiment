import numpy as np

def sort_by_selectivity(list_of_psths, odor_on, water_on, condition_config):
    list_of_ixs = []
    for psth in list_of_psths:
        max_dff = np.max(psth[:, odor_on:water_on], axis=1)
        ixs = np.argsort(max_dff)[::-1]
        cutoff = np.argmin(max_dff[ixs] > condition_config.threshold)
        list_of_ixs.append(ixs[:cutoff])

    for psth in list_of_psths:
        max_dff = np.min(psth[:, odor_on:water_on], axis=1)
        ixs = np.argsort(max_dff)[::-1]
        cutoff = np.argmin(max_dff[ixs] > condition_config.negative_threshold)
        list_of_ixs.append(ixs[:cutoff])

    list_of_ixs.append(np.arange(list_of_psths[0].shape[0]))
    final_ixs = _sort_by_ixs(list_of_ixs)
    return final_ixs

def sort_by_onset(list_of_psths, odor_on, water_on, condition_config):
    list_of_argmax = []
    for psth in list_of_psths:
        binary_psth = psth[:, odor_on:water_on] > condition_config.threshold
        responsive = np.any(binary_psth, axis=1)
        argmax = np.argmax(binary_psth == 1, axis=1)
        argmax[np.invert(responsive)] = 100
        list_of_argmax.append(argmax)

    list_of_argmin = []
    for psth in list_of_psths:
        binary_psth = psth[:, odor_on:water_on] < condition_config.negative_threshold
        responsive = np.any(binary_psth, axis=1)
        argmax = np.argmax(binary_psth == 1, axis=1)
        argmax[np.invert(responsive)] = 100
        list_of_argmin.append(argmax)

    list_of_ixs = []
    cutoff_ix = water_on - odor_on - 1
    if condition_config.sort_style == 'individual':
        for argmax in list_of_argmax:
            ixs = np.argsort(argmax)
            cutoff = np.argmax(argmax[ixs] > cutoff_ix)
            list_of_ixs.append((ixs[:cutoff]))

    if condition_config.sort_style == 'CS+':
        argmax = np.array([(x + y)/2 for x, y in zip(list_of_argmax[0], list_of_argmax[1])])
        ixs = np.argsort(argmax)
        cutoff = np.argmax(argmax[ixs] > cutoff_ix)
        list_of_ixs.append((ixs[:cutoff]))

        argmax = np.array([(x + y)/2 for x, y in zip(list_of_argmin[0], list_of_argmin[1])])
        ixs = np.argsort(argmax)[::-1]
        cutoff = np.argmin(argmax[ixs] > cutoff_ix)
        list_of_ixs.append((ixs[:cutoff]))

    list_of_ixs.append(np.arange(list_of_psths[0].shape[0]))
    final_ixs = _sort_by_ixs(list_of_ixs)
    return final_ixs


def _sort_by_ixs(list_of_ixs):
    final_ixs = []
    for ixs in list_of_ixs:
        for ix in ixs:
            if ix not in final_ixs:
                final_ixs.append(ix)
    return final_ixs