import copy
import itertools
import numpy as np
import plot

def filter_days_per_mouse(res, days_per_mouse, key_day ='day'):
    '''
    Filter results to only include the days specified by days_per_mouse.

    :param res: flattened dict of results
    :param days_per_mouse: list (iter = mouse) of list (iter = days). I.E.
    [[0,5],[0,1]] == include days [0, 5] from mouse 0, and days [0, 1] from mouse 1
    len(days_per_mouse) must equal the number of mice within res
    :return: a copy of res with filter applied
    '''
    out = copy.copy(res)
    list_of_ixs = []
    list_of_dates = res[key_day]
    list_of_mice = res['mouse']
    mouse_names, mouse_ixs = np.unique(list_of_mice, return_inverse=True)
    if mouse_names.size != len(days_per_mouse):
        raise ValueError("res has {0:d} mice, but filter has {1:d} mice".
                         format(mouse_names.size, len(days_per_mouse)))

    for i, mouse_ix in enumerate(np.unique(mouse_ixs)):
        ix = mouse_ixs == mouse_ix
        mouse_dates = list_of_dates[ix]
        membership = np.isin(mouse_dates, days_per_mouse[i])
        ix[ix == True]= membership
        list_of_ixs.append(ix)

    select_ixs = np.any(list_of_ixs, axis=0)
    for key, value in res.items():
        try:
            out[key] = value[select_ixs]
        except:
            print(key)
            raise ValueError('{} had issues'.format(key))
    return out


def filter(res, filter_dict):
    '''
    Filter results to only include entries containing the key, value pairs in select_dict
    For example, select_dict = {'day', 0} will filter all data from res whose 'day' == 0

    :param res: flattened dict of results
    :param filter_dict:
    :return: a copy of res with filter applied
    '''
    out = copy.copy(res)
    list_of_ixs = []
    for key, vals in filter_dict.items():
        membership = np.isin(res[key], vals)
        list_of_ixs.append(membership)
    select_ixs = np.all(list_of_ixs, axis=0)
    for key, value in res.items():
        try:
            out[key] = value[select_ixs]
        except:
            print(key)
            raise ValueError('{} had issues'.format(key))
    return out

def exclude(res, exclude_dict):
    out = copy.copy(res)
    list_of_ixs = []
    for key, vals in exclude_dict.items():
        membership = np.isin(res[key], vals)
        list_of_ixs.append(membership)
    exclude_ixs = np.all(list_of_ixs, axis=0)
    select_ixs = np.invert(exclude_ixs)

    for key, value in res.items():
        out[key] = value[select_ixs]
    return out

def retrieve_unique_entries(res, loop_keys):
    unique_entries_per_loopkey = []
    for x in loop_keys:
        a = res[x]
        # indexes = np.unique(a, return_index=True)[1]
        # unique_entries_per_loopkey.append([a[index] for index in sorted(indexes)])
        unique_entries_per_loopkey.append(np.unique(a))
    unique_entry_combinations = list(itertools.product(*unique_entries_per_loopkey))
    nlines = len(unique_entry_combinations)

    list_of_ind = []
    for x in range(nlines):
        list_of_ixs = []
        cur_combination = unique_entry_combinations[x]
        for i, val in enumerate(cur_combination):
            list_of_ixs.append(val == res[loop_keys[i]])
        ind = np.all(list_of_ixs, axis=0)
        ind_ = np.where(ind)[0]
        list_of_ind.append(ind_)

    a, b = [], []
    for i, ix in enumerate(list_of_ind):
        if len(ix):
            a.append(unique_entry_combinations[i])
            b.append(list_of_ind[i])
    return a, b

def assign_composite(res, loop_keys):
    '''
    loop keys change last first
    :param res:
    :param loop_keys:
    :return:
    '''
    unique_entry_combinations, list_of_ind = retrieve_unique_entries(res, loop_keys)
    out = np.zeros_like(res[loop_keys[0]]).astype(object)
    for i, name in enumerate(unique_entry_combinations):
        name = '_'.join([plot.nice_names(str(n)) for n in name])
        out[list_of_ind[i]] = name
    key = '_'.join(key for key in loop_keys)
    res[key] = out