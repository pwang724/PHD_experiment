
from collections import defaultdict
import numpy as np
import filter
import itertools
import reduce


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

def get_valence_sig(res):
    #TODO: not tested yet
    def _helper(res):
        assert res['odor_valence'][0] == 'CS+', 'wrong odor'
        assert res['odor_valence'][1] == 'CS-', 'wrong odor'
        on = res['DAQ_O_ON_F'][0]
        off = res['DAQ_W_ON_F'][0]
        sig_p = res['sig'][0]
        sig_m = res['sig'][1]
        dff_p = res['dff'][0]
        dff_m = res['dff'][1]
        sig_p_mask = sig_p == 1
        sig_m_mask = sig_m == 1
        dff_mask = dff_p - dff_m
        dff_mask = np.mean(dff_mask[:, on:off], axis=1)
        p = [a and b for a, b in zip(sig_p_mask, dff_mask>0)]
        m = [a and b for a, b in zip(sig_m_mask, dff_mask<0)]
        return np.array(p), np.array(m)

    mice = np.unique(res['mouse'])
    res = filter.filter(res, filter_dict={'odor_valence':['CS+','CS-']})
    sig_res = reduce.new_filter_reduce(res, reduce_key='sig', filter_keys=['mouse','day','odor_valence'])
    dff_res = reduce.new_filter_reduce(res, reduce_key='dff', filter_keys=['mouse','day','odor_valence'])
    sig_res['dff'] = dff_res['dff']

    reversal_res = defaultdict(list)
    day_strs = ['Lrn', 'Rev']
    for mouse in mice:
        mouse_res = filter.filter(sig_res, filter_dict={'mouse': mouse})
        days = np.unique(mouse_res['day'])
        p_list = []
        m_list = []
        for i, day in enumerate(days):
            mouse_day_res = filter.filter(mouse_res, filter_dict={'day': day})
            p, m = _helper(mouse_day_res)
            reversal_res['mouse'].append(mouse)
            reversal_res['mouse'].append(mouse)
            reversal_res['day'].append(day_strs[i])
            reversal_res['day'].append(day_strs[i])
            reversal_res['odor_valence'].append('CS+')
            reversal_res['odor_valence'].append('CS-')
            reversal_res['sig'].append(p)
            reversal_res['sig'].append(m)
            reversal_res['Fraction'].append(np.mean(p))
            reversal_res['Fraction'].append(np.mean(m))
            p_list.append(p)
            m_list.append(m)
    for key, val in reversal_res.items():
        reversal_res[key] = np.array(val)

def get_reversal_sig(res):
    def _helper(res):
        assert res['odor_valence'][0] == 'CS+', 'wrong odor'
        assert res['odor_valence'][1] == 'CS-', 'wrong odor'
        on = res['DAQ_O_ON_F'][0]
        off = res['DAQ_W_ON_F'][0]
        sig_p = res['sig'][0]
        sig_m = res['sig'][1]
        dff_p = res['dff'][0]
        dff_m = res['dff'][1]
        sig_p_mask = sig_p == 1
        sig_m_mask = sig_m == 1
        dff_mask = dff_p - dff_m
        dff_mask = np.mean(dff_mask[:, on:off], axis=1)
        p = [a and b for a, b in zip(sig_p_mask, dff_mask>0)]
        m = [a and b for a, b in zip(sig_m_mask, dff_mask<0)]
        return np.array(p), np.array(m)

    mice = np.unique(res['mouse'])
    res = filter.filter(res, filter_dict={'odor_valence':['CS+','CS-']})
    sig_res = reduce.new_filter_reduce(res, reduce_key='sig', filter_keys=['mouse','day','odor_valence'])
    dff_res = reduce.new_filter_reduce(res, reduce_key='dff', filter_keys=['mouse','day','odor_valence'])
    sig_res['dff'] = dff_res['dff']

    reversal_res = defaultdict(list)
    day_strs = ['Lrn','Rev']
    for mouse in mice:
        mouse_res = filter.filter(sig_res, filter_dict={'mouse':mouse})
        days = np.unique(mouse_res['day'])
        p_list = []
        m_list = []
        for i, day in enumerate(days):
            mouse_day_res = filter.filter(mouse_res, filter_dict={'day':day})
            p, m = _helper(mouse_day_res)
            reversal_res['mouse'].append(mouse)
            reversal_res['mouse'].append(mouse)
            reversal_res['day'].append(day_strs[i])
            reversal_res['day'].append(day_strs[i])
            reversal_res['odor_valence'].append('CS+')
            reversal_res['odor_valence'].append('CS-')
            reversal_res['sig'].append(p)
            reversal_res['sig'].append(m)
            reversal_res['Fraction'].append(np.mean(p))
            reversal_res['Fraction'].append(np.mean(m))
            p_list.append(p)
            m_list.append(m)
    for key, val in reversal_res.items():
        reversal_res[key] = np.array(val)

    stats_res = defaultdict(list)
    for mouse in mice:
        mouse_res = filter.filter(reversal_res, filter_dict={'mouse':mouse})
        combinations, list_of_ixs = filter.retrieve_unique_entries(mouse_res, ['day','odor_valence'])

        assert len(combinations) == 4, 'not equal to 4'
        assert combinations[0][-1] == 'CS+'
        assert combinations[1][-1] == 'CS-'
        assert combinations[2][-1] == 'CS+'
        assert combinations[3][-1] == 'CS-'
        assert combinations[0][0] == day_strs[0]
        assert combinations[1][0] == day_strs[0]
        assert combinations[2][0] == day_strs[1]
        assert combinations[3][0] == day_strs[1]

        p_before = mouse_res['sig'][0]
        m_before = mouse_res['sig'][1]
        n_before = np.invert([a or b for a, b in zip(p_before, m_before)])
        p_after = mouse_res['sig'][2]
        m_after = mouse_res['sig'][3]
        n_after = np.invert([a or b for a, b in zip(p_after, m_after)])

        list_before = [p_before, m_before, n_before]
        list_after = [p_after, m_after, n_after]
        str = ['p', 'm', 'none']
        for i, before in enumerate(list_before):
            for j, after in enumerate(list_after):
                ix_intersect =np.intersect1d(np.where(before)[0], np.where(after)[0])
                fraction = len(ix_intersect) / np.sum(before)
                stats_res['mouse'].append(mouse)
                stats_res['condition'].append(str[i]  + '-' + str[j])
                stats_res['Fraction'].append(fraction)
    for key, val in stats_res.items():
        stats_res[key] = np.array(val)
    return reversal_res, stats_res


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

            odors, odor_ix = np.unique(mouse_day_res['odor_standard'], return_index= True)
            assert len(odor_ix) == 4, 'Number of odors does not equal 4'
            all_comparisons = _subsets(odor_ix, 2)
            for comparison in all_comparisons:
                mask1 = mouse_day_res['sig'][comparison[0]]
                mask2 = mouse_day_res['sig'][comparison[1]]

                if delete_non_selective:
                    non_selective_mask = _respond_to_all(mouse_day_res['sig'])
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