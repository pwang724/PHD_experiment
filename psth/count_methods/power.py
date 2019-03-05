import copy

import numpy as np
from matplotlib import pyplot as plt

import filter
import plot
import reduce
from analysis import add_naive_learned
from psth.format import *


def plot_power(res, start_days, end_days, figure_path):
    res = copy.copy(res)
    _normalize_across_days(res)
    _get_power(res, normalize_across_days=False)

    list_of_days = list(zip(start_days, end_days))
    start_end_day_res = filter.filter_days_per_mouse(res, days_per_mouse=list_of_days)
    start_end_day_res = reduce.new_filter_reduce(start_end_day_res, filter_keys=['day', 'odor_valence'],
                                                 reduce_key='Power')
    add_naive_learned(start_end_day_res, start_days, end_days)

    ax_args_copy = trace_ax_args.copy()
    ax_args_copy.update({'xticks':[res['DAQ_O_ON_F'][0], res['DAQ_W_ON_F'][0]], 'xticklabels':['ON', 'US'],
                         'ylim':[0, .2]})
    colors = ['Green','Gray']
    plot.plot_results(start_end_day_res, select_dict={'odor_valence':'CS+'},
                      x_key='Time', y_key='Power', loop_keys= 'day', error_key='Power_sem',
                      path=figure_path,
                      plot_function=plt.fill_between, plot_args=fill_args, ax_args=ax_args_copy,
                      colors = colors,
                      fig_size=(2, 1.5), save=False)

    plot.plot_results(start_end_day_res, select_dict={'odor_valence':'CS+'},
                      x_key='Time', y_key='Power', loop_keys= 'day',
                      path=figure_path, plot_args=trace_args, ax_args=ax_args_copy,
                      colors = colors,
                      fig_size=(2, 1.5), reuse=True)


def _get_power(res, normalize_across_days=True):
    if normalize_across_days:
        key = 'ndff'
    else:
        key = 'dff'
    res['Power'] = np.copy(res['sig'])
    res['Time'] = np.copy(res['sig'])
    combinations, list_of_ixs = filter.retrieve_unique_entries(res, loop_keys=['mouse', 'odor'])
    for ixs in list_of_ixs:
        assert res['day'][ixs[0]] == 0, 'not the first day as reference'
        mask_a = res['sig'][ixs[0]]
        mask_b = res['sig'][ixs[1]]
        mask = np.array([a or b for a, b in zip(mask_a, mask_b)]).astype(bool)
        # mask = np.array(mask_a).astype(bool)
        # mask = np.array(mask_b).astype(bool)

        for ix in ixs:
            dff = res[key][ix][mask.astype(bool)]
            power = np.mean(dff, axis=0)
            res['Power'][ix] = power
            res['Time'][ix] = np.arange(0, len(power))
    res['Power'] = np.array(res['Power'])
    res['Time'] = np.array(res['Time'])


def _normalize_across_days(res):
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