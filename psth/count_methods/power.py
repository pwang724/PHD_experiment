import copy

import numpy as np
from matplotlib import pyplot as plt

import filter
import plot
import reduce
from analysis import add_naive_learned
from format import *
from collections import defaultdict


def plot_power(res, start_days, end_days, figure_path):
    res = copy.copy(res)
    _get_power_1(res)

    list_of_days = list(zip(start_days, end_days))
    start_end_day_res = filter.filter_days_per_mouse(res, days_per_mouse=list_of_days)
    start_end_day_res = reduce.new_filter_reduce(start_end_day_res, filter_keys=['day', 'odor_valence'],
                                                 reduce_key='Power')
    add_naive_learned(start_end_day_res, start_days, end_days)

    ax_args_copy = trace_ax_args.copy()
    ax_args_copy.update({'xticks':[res['DAQ_O_ON_F'][0], res['DAQ_W_ON_F'][0]], 'xticklabels':['ON', 'US'],
                         'ylim':[0, .1]})
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


def _get_power(res):
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

def _get_power_1(res):
    list_of_dff = res['dff']
    list_odor_on = res['DAQ_O_ON_F']
    list_water_on = res['DAQ_W_ON_F']
    for i, dff in enumerate(list_of_dff):
        s, e = list_odor_on[i], list_water_on[i]
        y_ = np.mean(dff[:, s:e], axis=1)
        ix = y_>0.0
        y = np.mean(np.abs(dff[ix,:]), axis=0)
        x = np.arange(0, len(y))
        res['Power'].append(y)
        res['Time'].append(x)
    res['Power'] = np.array(res['Power'])
    res['Time'] = np.array(res['Time'])

def _max_dff(res):
    list_of_dff = res['dff']
    list_odor_on = res['DAQ_O_ON_F']
    list_water_on = res['DAQ_W_ON_F']

    for i, dff in enumerate(list_of_dff):
        s, e = list_odor_on[i], list_water_on[i]
        y = np.max(dff[:,s:e], axis=1)
        y_base = np.max(dff[:,:s], axis=1)
        # y = y-y_base
        y_ = np.mean(dff[:,s:e], axis=1)
        y = y[y_>0.0] - y_base[y_>0.0]
        res['max_dff'].append(np.mean(y))
    res['max_dff'] = np.array(res['max_dff'])

def plot_mean_dff(res, start_days, end_days, figure_path):
    res = copy.copy(res)
    # list_of_days = list(zip(start_days, end_days))
    list_of_days = end_days
    start_end_day_res = filter.filter_days_per_mouse(res, days_per_mouse=list_of_days)
    start_end_day_res = filter.filter(start_end_day_res, {'odor_valence': ['CS+','CS-']})
    _get_power_1(start_end_day_res)
    start_end_day_res = reduce.new_filter_reduce(start_end_day_res, filter_keys=['odor_valence','mouse'],
                                                 reduce_key='mean_dff')
    add_naive_learned(start_end_day_res, start_days, end_days)
    ax_args_copy = trace_ax_args.copy()
    ax_args_copy.update({'xticks':[res['DAQ_O_ON_F'][-1], res['DAQ_W_ON_F'][-1]], 'xticklabels':['ON', 'US'],
                         'ylim':[0, .2]})
    nMice = len(np.unique(res['mouse']))
    colors = ['Green'] * nMice + ['Red'] * nMice

    trace_args_copy = trace_args.copy()
    trace_args_copy.update({'linestyle':'--','alpha':.5, 'linewidth':.75})

    plot.plot_results(start_end_day_res, loop_keys=['odor_valence','mouse'],
                      x_key='Time', y_key='mean_dff',
                      path=figure_path, plot_args=trace_args_copy, ax_args=ax_args_copy,
                      colors = colors, legend=False,
                      fig_size=(2, 1.5))

def plot_max_dff_valence(res, start_days, end_days, figure_path):
    res = copy.copy(res)
    # list_of_days = list(zip(start_days, end_days))
    list_of_days = end_days
    start_end_day_res = filter.filter_days_per_mouse(res, days_per_mouse=list_of_days)
    start_end_day_res = filter.filter(start_end_day_res, {'odor_valence': ['CS+','CS-']})
    _max_dff(start_end_day_res)
    start_end_day_res = reduce.new_filter_reduce(start_end_day_res, filter_keys=['odor_valence','mouse'],
                                                 reduce_key='max_dff')
    add_naive_learned(start_end_day_res, start_days, end_days)
    ax_args_copy = ax_args.copy()
    # ax_args_copy.update({'xticks':[res['DAQ_O_ON_F'][-1], res['DAQ_W_ON_F'][-1]], 'xticklabels':['ON', 'US'],
    #                      'ylim':[0, .2]})
    nMice = len(np.unique(res['mouse']))
    # colors = ['Green'] * nMice + ['Red'] * nMice

    # trace_args_copy = trace_args.copy()
    # trace_args_copy.update({'linestyle':'--','alpha':.5, 'linewidth':.75})

    plot.plot_results(start_end_day_res, loop_keys='mouse',
                      x_key='odor_valence', y_key='max_dff',
                      path=figure_path,
                      colors = ['gray']*10, legend=False,
                      fig_size=(2, 1.5))

def _normalize_across_days(res):
    combinations, list_of_ixs = filter.retrieve_unique_entries(res, loop_keys=['mouse', 'odor'])
    for ixs in list_of_ixs:
        assert res['day'][ixs[0]] < res['day'][ixs[1]], 'not the first day as reference'
        first_dff = res['max_dff'][ixs[0]]
        second_dff = res['max_dff'][ixs[1]]
        second_dff = second_dff/ first_dff
        res['max_dff'][ixs[0]] = 1.0
        res['max_dff'][ixs[1]] = second_dff

def plot_max_dff_days(res, days_per_mouse, odor_valence, save, reuse, day_pad, ylim = .115, colors=None,
                      normalize=False, figure_path=None):

    res = copy.copy(res)
    res['day_'] = np.zeros_like(res['day'])

    res_ = defaultdict(list)
    for i, days in enumerate(days_per_mouse):
        temp = filter.filter_days_per_mouse(res, days_per_mouse=days)
        temp = filter.filter(temp, {'odor_valence': odor_valence[i]})
        temp['day_'] = np.array([i + day_pad] * len(temp['day_']))
        reduce.chain_defaultdicts(res_, temp)

    _max_dff(res_)
    res_ = reduce.new_filter_reduce(res_, filter_keys=['odor_valence','mouse','day_'], reduce_key='max_dff')
    if normalize:
        _normalize_across_days(res_)
        yticks = [0, 1, 2, 3, 4, 5, 6]
    else:
        yticks = np.arange(0, ylim, .05)

    dict = {'CS+': 'Green', 'CS-': 'Red', 'US': 'Turquoise', 'PT CS+': 'Orange'}
    n_mice = len(np.unique(res['mouse']))
    ax_args_copy = ax_args.copy()
    ax_args_copy.update({'ylim':[0, ylim], 'yticks': yticks, 'xticks':list(range(20))})
    line_args_copy = line_args.copy()
    line_args_copy.update({'marker':'.', 'linestyle':'--', 'linewidth':.5, 'alpha': .5, 'markersize':2})
    if colors is None:
        colors = [dict[x] for x in odor_valence]
    else:
        line_args_copy.update({'marker':None})
    plot.plot_results(res_, loop_keys='mouse', x_key='day_', y_key='max_dff',
                      path=figure_path,
                      colors= colors * n_mice, legend=False, plot_args=line_args_copy, ax_args=ax_args_copy,
                      fig_size=(2, 2), save=save, reuse=reuse)



def plot_bar(res, days_per_mouse, odor_valence, day_pad, save, reuse, figure_path, color ='black', normalize=False):
    res = copy.copy(res)
    res['day_'] = np.zeros_like(res['day'])

    res_ = defaultdict(list)
    for i, days in enumerate(days_per_mouse):
        temp = filter.filter_days_per_mouse(res, days_per_mouse=days)
        temp = filter.filter(temp, {'odor_valence': odor_valence[i]})
        temp['day_'] = np.array([i + day_pad] * len(temp['day_']))
        reduce.chain_defaultdicts(res_, temp)

    _max_dff(res_)
    res_ = reduce.new_filter_reduce(res_, filter_keys=['odor_valence', 'mouse', 'day_'], reduce_key='max_dff')
    res_.pop('max_dff_sem')
    summary = reduce.new_filter_reduce(res_, filter_keys=['day_', 'odor_valence'], reduce_key='max_dff')

    if normalize:
        _normalize_across_days(summary)

    # plot.plot_results(summary, x_key='day_', y_key='max_dff', error_key='max_dff_sem',
    #                   path=figure_path,
    #                   colors='black', legend=False, plot_args=error_args, plot_function= plt.errorbar,
    #                   fig_size=(2, 1.5), save=False, reuse=reuse)

    line_args_copy = line_args.copy()
    line_args_copy.update({'alpha':.75, 'linewidth':1})
    plot.plot_results(summary, x_key='day_', y_key='max_dff',
                      path=figure_path,
                      colors=color, legend=False, plot_args=line_args_copy,
                      fig_size=(2, 1.5), save=save, reuse=reuse, name_str=odor_valence[-1])




# def plot_max_dff_days_bar(res, ref, test, odor_valence, save, reuse, figure_path):
#     res = copy.copy(res)
#     res['day_'] = np.zeros_like(res['day'])
#
#     res_ = defaultdict(list)
#     for i, days in enumerate(days_per_mouse):
#         temp = filter.filter_days_per_mouse(res, days_per_mouse=days)
#         temp['day_'] = np.array([i + day_pad] * len(temp['day_']))
#         reduce.chain_defaultdicts(res_, temp)
#
#     res_ = filter.filter(res_, {'odor_valence': odor_valence})
#     _max_dff(res_)
#     res_ = reduce.new_filter_reduce(res_, filter_keys=['odor_valence','mouse','day_'], reduce_key='max_dff')
#
#     dict = {'CS+': 'Green', 'CS-': 'Red', 'US': 'Turquoise', 'PT CS+': 'Orange'}
#     colors = [dict[x] for x in odor_valence]
#     n_mice = len(np.unique(res['mouse']))
#
#
#     # ax_args_copy = ax_args.copy()
#     # ax_args_copy.update({'ylim':[0, .1], 'yticks':[0, .2, .4, .6], 'xticks':list(range(20))})
#     line_args_copy = line_args.copy()
#     line_args_copy.update({'marker':'.', 'linestyle':'--', 'linewidth':.5, 'alpha': .75, 'markersize':2})
#     plot.plot_results(res_, loop_keys='mouse', x_key='day_', y_key='max_dff',
#                       path=figure_path,
#                       colors= colors * n_mice, legend=False, plot_args=line_args_copy,
#                       fig_size=(2, 1.5), save=save, reuse=reuse)






