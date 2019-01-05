from collections import defaultdict

import numpy as np


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
        out = np.mean(data)
    except:
        out = np.mean(data[data!=None])
        print('mean of entries for {} could not be computed. took the mean of non-None entries: {}'.format(
            key, data
        ))

    out_res = defaultdict(list)
    for k, v in res.items():
        if k == key:
            out_res[k] = out
        else:
            try:
                if len(set(v)) == 1:
                    out_res[k] = v[0]
            except:
                str = 'Did not parse {}'.format(k)
                if verbose:
                    print(str)
    return out_res