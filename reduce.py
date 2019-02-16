import itertools
from collections import defaultdict

import numpy as np

import filter
from scipy import stats as sstats
import copy

def _regularize_length(res, key):
    data = res[key]
    if type(data[0]) == np.ndarray or type(data[0]) == list:
        min_length = np.min([x.shape for x in data])
        for k, v in res.items():
            if type(v[0]) == np.ndarray or type(v[0]) == list:
                new_array = []
                for i, x in enumerate(v):
                    if len(x) > min_length:
                        new_array.append(x[:min_length])
                    else:
                        new_array.append(x)
                res[k] = new_array
        for key, val in res.items():
            res[key] = np.array(val)

def filter_reduce(res, filter_keys, reduce_key):
    '''
    #CAN ONLY HANDLE ONE FILTER KEY
    :param res:
    :param filter_keys:
    :param reduce_key:
    :return:
    '''
    out = defaultdict(list)
    unique_ixs = sorted(np.unique(res[filter_keys], return_index=True)[-1])
    unique_vals = res[filter_keys][unique_ixs]
    for v in unique_vals:
        filter_dict = {filter_keys: v}
        cur_res = filter.filter(res, filter_dict)
        try:
            _regularize_length(cur_res, reduce_key)
        except:
            print('cannot regularize the length of {}'.format(reduce_key))
        temp_res = reduce_by_mean(cur_res, reduce_key)
        append_defaultdicts(out, temp_res)
    for key, val in out.items():
        out[key] = np.array(val)
    return out

def new_filter_reduce(res, filter_keys, reduce_key):
    out = defaultdict(list)
    if isinstance(filter_keys, str):
        filter_keys = [filter_keys]
    unique_combinations, ixs = filter.retrieve_unique_entries(res, filter_keys)
    for v in unique_combinations:
        filter_dict = {filter_key: val for filter_key, val in zip(filter_keys, v)}
        cur_res = filter.filter(res, filter_dict)

        if len(cur_res[reduce_key]):
            try:
                _regularize_length(cur_res, reduce_key)
            except:
                print('cannot regularize the length of {}'.format(reduce_key))
            temp_res = reduce_by_mean(cur_res, reduce_key)
            append_defaultdicts(out, temp_res)
    for key, val in out.items():
        out[key] = np.array(val)
    return out



def append_defaultdicts(dictA, dictB):
    for k in dictB.keys():
        dictA[k].append(dictB[k])

def chain_defaultdicts(dictA, dictB, copy_dict = False):
    '''
    works as intended
    :param dictA:
    :param dictB:
    :return:
    '''

    if copy_dict:
        dictA = copy.copy(dictA)

    for k in dictB.keys():
        dictA[k] = list(itertools.chain(dictA[k], dictB[k]))
    for key, val in dictA.items():
        dictA[key] = np.array(val)

    if copy_dict:
        return dictA


def reduce_by_concat(res, key, rank_keys=None, verbose=False):
    def _sort(res, rank_keys):
        rank = []
        for rank_key in rank_keys:
            rank.append(res[rank_key])
        rank = np.array(rank).transpose()
        sorted_ranks = sorted(enumerate(rank), key=lambda x: (x[1][0], x[1][1]))
        sorted_ranks = [i[0] for i in sorted_ranks]
        return sorted_ranks
    data = res[key]
    if rank_keys:
        ixs = _sort(res, rank_keys)
        data = data[ixs]
    data = np.array(data)
    concatenated_res = defaultdict(list)
    for k, v in res.items():
        if k == key:
            concatenated_res[k] = data
        elif rank_keys and k in rank_keys:
            pass
        else:
            try:
                if len(set(v)) == 1:
                    concatenated_res[k] = v[0]
            except:
                str = 'Did not parse {}'.format(k)
                if verbose:
                    print(str)
    return concatenated_res

def reduce_by_mean(res, key, verbose = False):
    data = res[key].astype(float)

    try:
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        sem = sstats.sem(data, axis=0)
    except:
        if np.all(np.isnan(data)):
            mean = None
            std = None
            sem = None
        else:
            ix = np.invert(np.isnan(data))
            mean = np.mean(data[ix], axis=0)
            std = np.std(data[ix], axis=0)
            if len(data[data!=None]) == 1:
                sem = 0
            else:
                sem = sstats.sem(data[ix], axis=0)
            print('mean of entries for {} could not be computed. took the mean of non-None entries: {}'
                  ' for mouse {}'.
                format(key, data, res['mouse']))

    out_res = defaultdict(list)
    for k, v in res.items():
        if k == key:
            out_res[k] = mean
            out_res[k + '_std'] = std
            out_res[k + '_sem'] = sem
        else:
            out_res[k] = v[0]
                # try:
                #     if len(set(v)) == 1:
                #         out_res[k] = v[0]
                #     else:
                #         out_res[k] = np.mean(v)
                # except:
                #     str = 'Did not parse {}'.format(k)
                #     if verbose:
                #         print(str)
    return out_res