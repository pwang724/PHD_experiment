from collections import defaultdict

import numpy as np

import filter
from tools.utils import append_defaultdicts


def _sort(res, rank_keys):
    rank = []
    for rank_key in rank_keys:
        rank.append(res[rank_key])
    rank = np.array(rank).transpose()
    sorted_ranks = sorted(enumerate(rank), key=lambda x: (x[1][0], x[1][1]))
    sorted_ranks = [i[0] for i in sorted_ranks]
    return sorted_ranks


def reduce_by_concat(res, key, rank_keys = None, verbose = False):
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
    data = res[key]

    try:
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
    except:
        mean = np.mean(data[data!=None], axis=0)
        std = np.std(data[data!=None], axis=0)
        print('mean of entries for {} could not be computed. took the mean of non-None entries: {}'
              ' for mouse {}'.
            format(key, data, res['mouse']))

    out_res = defaultdict(list)
    for k, v in res.items():
        if k == key:
            out_res[k] = mean
            out_res[k + '_std'] = std
        else:
            if type(v[0]) == np.ndarray or type(v[0]) == list:
                out_res[k] = v[0]
            else:
                try:
                    if len(set(v)) == 1:
                        out_res[k] = v[0]
                except:
                    str = 'Did not parse {}'.format(k)
                    if verbose:
                        print(str)
    return out_res

def regularize_length(res, key):
    data = res[key]

    if type(data[0]) == np.ndarray or type(data[0] == list):
        min_length = np.min([x.shape for x in data])
        for k, v in res.items():
            if type(v[0]) == np.ndarray or type(v[0]) == list:
                for i, x in enumerate(v):
                    if len(x) > min_length:
                        v[i] = x[:min_length]
                res[k] = v
        for key, val in res.items():
            res[key] = np.array(val)

def filter_reduce(res, filter_key, reduce_key):
    out = defaultdict(list)
    unique_ixs = sorted(np.unique(res[filter_key], return_index=True)[-1])
    unique_vals = res[filter_key][unique_ixs]
    for v in unique_vals:
        filter_dict = {filter_key: v}
        cur_res = filter.filter(res, filter_dict)
        try:
            regularize_length(cur_res, reduce_key)
        except:
            print('cannot regularize the length of {}'.format(reduce_key))
        temp_res = reduce_by_mean(cur_res, reduce_key)
        append_defaultdicts(out, temp_res)
    for key, val in out.items():
        out[key] = np.array(val)
    return out