
from collections import defaultdict
import numpy as np
import filter
import itertools
import matplotlib.pyplot as plt

def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def get_responsive_cells(res):
    list_of_data = res['sig']
    for data in list_of_data:
        res['Fraction Responsive'].append(np.mean(data))
    res['Fraction Responsive'] = np.array(res['Fraction Responsive'])


def _overlap(ix1, ix2, arg = 'max'):
    if arg == 'max':
        size = np.max((ix1.size, ix2.size))
    elif arg == 'over':
        size = ix2.size
    else:
        raise ValueError('overlap arg not recognized')
    intersect = float(len(np.intersect1d(ix1, ix2)))
    return intersect / size

def _respond_to_all(list_of_masks):
    arr = np.stack(list_of_masks, axis=1)
    non_selective_mask = np.all(arr, axis=1)
    return non_selective_mask

def get_overlap(res, delete_non_selective):
    def _subsets(S, m):
        return set(itertools.combinations(S, m))
    new = defaultdict(list)
    mice =np.unique(res['mouse'])
    for mouse in mice:
        mouse_res = filter.filter(res, filter_dict={'mouse':mouse})
        days = np.unique(mouse_res['day'])
        for day in days:
            mouse_day_res = filter.filter(mouse_res,
                                          filter_dict={'day':day, 'odor_valence':['CS+','CS-']})
            non_selective_mask = _respond_to_all(mouse_day_res['sig'])
            print(np.sum(non_selective_mask))

            odors, odor_ix = np.unique(mouse_day_res['odor_standard'], return_index= True)
            assert len(odor_ix) == 4, 'Number of odors does not equal 4'
            all_comparisons = _subsets(odor_ix, 2)
            for comparison in all_comparisons:
                mask1 = mouse_day_res['sig'][comparison[0]]
                mask2 = mouse_day_res['sig'][comparison[1]]

                if delete_non_selective:
                    mask1 = np.all([mask1, np.invert(non_selective_mask)], axis=0).astype(int)
                    mask2 = np.all([mask2, np.invert(non_selective_mask)], axis=0).astype(int)

                overlap = _overlap(np.where(mask1)[0], np.where(mask2)[0])
                new['Overlap'].append(overlap)
                new['mouse'].append(mouse)
                new['day'].append(day)
                if comparison == (0,1):
                    new['condition'].append('+:+')
                elif comparison == (2,3):
                    new['condition'].append('-:-')
                else:
                    new['condition'].append('+:-')
    for key, val in new.items():
        new[key] = np.array(val)
    return new

def normalize_across_days(res):
    combinations, list_of_ixs = filter.retrieve_unique_entries(res, loop_keys=['mouse', 'odor'])
    res['ndff'] = np.copy(res['dff'])
    for ixs in list_of_ixs:
        assert res['day'][ixs[0]] == 0, 'not the first day as reference'
        first_dff = res['dff'][ixs[0]]
        nF = first_dff[0].size
        max = np.max(first_dff, axis=1)
        min = np.min(first_dff, axis=1)
        max = np.repeat(max[:, np.newaxis], nF, axis=1)
        min = np.repeat(min[:, np.newaxis], nF, axis=1)
        for ix in ixs:
            dff = res['dff'][ix]
            ndff = (dff - min) / (max - min)
            res['ndff'][ix] = ndff

def get_power(res, normalize_across_days=True):
    if normalize_across_days:
        key = 'ndff'
    else:
        key = 'dff'
    res['Power'] = np.copy(res['sig'])
    res['Time'] = np.copy(res['sig'])
    combinations, list_of_ixs = filter.retrieve_unique_entries(res, loop_keys=['mouse', 'odor'])
    for ixs in list_of_ixs:
        assert res['day'][ixs[0]] == 0, 'not the first day as reference'
        mask = res['sig'][ixs[0]]
        for ix in ixs:
            dff = res[key][ix][mask.astype(bool)]
            power = np.mean(dff, axis=0)
            res['Power'][ix] = power
            res['Time'][ix] = np.arange(0, len(power))
    res['Power'] = np.array(res['Power'])
    res['Time'] = np.array(res['Time'])

def get_overlap_water(res, arg):
    def _helper(list_of_name_ix_tuple, desired_tuple):
        for tuple in list_of_name_ix_tuple:
            if tuple[0] == desired_tuple:
                ix = tuple[1]
                assert len(ix) == 1, 'more than 1 unique entry'
                return ix[0]

    res['Overlap'] = np.zeros(res['day'].shape)
    names, ixs = filter.retrieve_unique_entries(res, ['mouse', 'day', 'odor_standard'])
    list_of_name_ix_tuples = list(zip(names, ixs))

    mice =np.unique(res['mouse'])
    for mouse in mice:
        mouse_res = filter.filter(res, filter_dict={'mouse':mouse})
        days = np.unique(mouse_res['day'])
        for day in days:
            mouse_day_res = filter.filter(mouse_res, filter_dict={'day':day})
            odors = np.unique(mouse_day_res['odor_standard'])
            if 'US' in odors:
                us_ix = _helper(list_of_name_ix_tuples, (mouse, day, 'US'))
                us_cells = np.where(res['sig'][us_ix])[0]
                for odor in odors:
                    odor_ix = _helper(list_of_name_ix_tuples, (mouse, day, odor))
                    odor_cells = np.where(res['sig'][odor_ix])[0]
                    if arg == 'US/CS+':
                        overlap = _overlap(us_cells, odor_cells, arg='over')
                    elif arg == 'CS+/US':
                        overlap = _overlap(odor_cells, us_cells, arg='over')
                    else:
                        raise ValueError('overlap arg not recognized')
                    res['Overlap'][odor_ix] = overlap


def add_naive_learned(res, start_day_per_mouse, learned_day_per_mouse):
    for i in range(len(res['day'])):
        day = res['day'][i]
        mouse = res['mouse'][i]
        if start_day_per_mouse[mouse] == day:
            res['training_day'].append('Naive')
        elif learned_day_per_mouse[mouse] == day:
            res['training_day'].append('Learned')
        else:
            raise ValueError('day is not either start day or learned day')
    res['training_day'] = np.array(res['training_day'])