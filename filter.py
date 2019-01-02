import copy
import numpy as np


def get_last_day_per_mouse(res):
    '''
    returns the value of the last day of imaging per mouse
    :param res: flattened dict of results
    :return: list of last day per each mouse
    '''
    out = []
    list_of_dates = res['NAME_DATE']
    list_of_mice = res['NAME_MOUSE']
    _, mouse_ixs = np.unique(list_of_mice, return_inverse=True)
    for mouse_ix in np.unique(mouse_ixs):
        mouse_dates = list_of_dates[mouse_ixs == mouse_ix]
        counts = np.unique(mouse_dates).size - 1
        out.append(counts)
    return out

def filter_days_per_mouse(res, days_per_mouse):
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
    list_of_dates = res['day']
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
        out[key] = value[select_ixs]
    return out

def filter_odors_per_mouse(res, odors):
    out = copy.copy(res)
    list_of_ixs = []
    list_of_odors = res['odor']
    list_of_mice = res['mouse']
    mouse_names, mouse_ixs = np.unique(list_of_mice, return_inverse=True)
    for i, mouse_name in enumerate(mouse_names):
        mouse_mask = list_of_mice == mouse_name
        odor_mask = np.isin(list_of_odors, odors[i])
        list_of_ixs.append(np.all((mouse_mask, odor_mask), axis=0))
    select_ixs = np.any(list_of_ixs, axis=0)
    for key, value in res.items():
        out[key] = value[select_ixs]
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
        out[key] = value[select_ixs]
    return out