import copy
import numpy as np


def _get_last_day_per_mouse(res):
    out = []
    list_of_dates = res['NAME_DATE']
    list_of_mice = res['NAME_MOUSE']
    _, mouse_ixs = np.unique(list_of_mice, return_inverse=True)
    for mouse_ix in np.unique(mouse_ixs):
        mouse_dates = list_of_dates[mouse_ixs == mouse_ix]
        counts = np.unique(mouse_dates).size - 1
        out.append(counts)
    return out

def _filter_days_per_mouse(res, days_per_mouse):
    out = copy.copy(res)
    list_of_ixs = []
    list_of_dates = res['day']
    list_of_mice = res['mouse']
    _, mouse_ixs = np.unique(list_of_mice, return_inverse=True)
    for i, mouse_ix in enumerate(np.unique(mouse_ixs)):
        ix = mouse_ixs == mouse_ix
        mouse_dates = list_of_dates[ix]
        membership = np.isin(mouse_dates, days_per_mouse[i])
        ix[ix]= membership
        list_of_ixs.append(ix)

    select_ixs = np.any(list_of_ixs, axis=0)
    for key, value in res.items():
        out[key] = value[select_ixs]
    return out


def _filter_results(res, select_dict):
    out = copy.copy(res)
    list_of_ixs = []
    for key, vals in select_dict.items():
        membership = np.isin(res[key], vals)
        list_of_ixs.append(membership)
    select_ixs = np.all(list_of_ixs, axis=0)
    for key, value in res.items():
        out[key] = value[select_ixs]
    return out