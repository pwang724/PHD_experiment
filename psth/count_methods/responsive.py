import copy
import numpy as np
import filter
import plot
import reduce
from analysis import add_naive_learned
from psth.plot_formatting import *

def plot_summary_odor(res, start_days, end_days, figure_path):
    ax_args_copy = ax_args.copy()
    res = copy.copy(res)
    get_responsive_cells(res)
    list_of_days = list(zip(start_days, end_days))
    mice = np.unique(res['mouse'])
    start_end_day_res = filter.filter_days_per_mouse(res, days_per_mouse= list_of_days)
    add_naive_learned(start_end_day_res, start_days, end_days)
    filter.assign_composite(start_end_day_res, loop_keys=['odor_standard', 'training_day'])

    odor_list = ['CS+1', 'CS+2','CS-1', 'CS-2']
    colors = ['Green','Green','Red','Red']
    ax_args_copy = ax_args_copy.copy()
    ax_args_copy.update({'xlim':[-1, 8]})
    for i, odor in enumerate(odor_list):
        save_arg = False
        reuse_arg = True
        if i == 0:
            reuse_arg = False
        if i == len(odor_list) -1:
            save_arg = True

        plot.plot_results(start_end_day_res,
                          select_dict= {'odor_standard':odor},
                          x_key='odor_standard_training_day', y_key='Fraction Responsive', loop_keys='mouse',
                          colors= [colors[i]]*len(mice),
                          path =figure_path, plot_args=line_args, ax_args= ax_args_copy,
                          save=save_arg, reuse=reuse_arg,
                          fig_size=(2.5, 1.5),legend=False, name_str = ','.join([str(x) for x in start_days]))


def plot_summary_odor_pretraining(res, start_days, end_days, arg_naive, figure_path):
    ax_args_copy = ax_args.copy()
    res = copy.copy(res)
    get_responsive_cells(res)
    list_of_days = list(zip(start_days, end_days))
    mice = np.unique(res['mouse'])
    start_end_day_res = filter.filter_days_per_mouse(res, days_per_mouse= list_of_days)

    if arg_naive:
        day_start = filter.filter(start_end_day_res, {'odor_standard': 'PT Naive'})
        day_start['odor_standard'] = np.array(['PT CS+'] * len(day_start['odor_standard']))
        day_end = filter.filter_days_per_mouse(res, days_per_mouse= end_days)
        reduce.chain_defaultdicts(day_start, day_end)
        start_end_day_res = day_start

    add_naive_learned(start_end_day_res, start_days, end_days)
    filter.assign_composite(start_end_day_res, loop_keys=['odor_standard', 'training_day'])

    odor_list = ['PT CS+']
    colors = ['Orange']
    ax_args_copy = ax_args_copy.copy()
    ax_args_copy.update({'xlim':[-1, 2]})
    for i, odor in enumerate(odor_list):
        save_arg = False
        reuse_arg = True
        if i == 0:
            reuse_arg = False
        if i == len(odor_list) -1:
            save_arg = True

        plot.plot_results(start_end_day_res,
                          select_dict= {'odor_standard':odor},
                          x_key='odor_standard_training_day', y_key='Fraction Responsive', loop_keys='mouse',
                          colors= [colors[i]]*len(mice),
                          path =figure_path, plot_args=line_args, ax_args= ax_args_copy,
                          save=save_arg, reuse=reuse_arg,
                          fig_size=(1.5, 1.5),legend=False)


def plot_summary_water(res, start_days, end_days, figure_path):
    ax_args_copy = ax_args.copy()
    res = copy.copy(res)
    get_responsive_cells(res)
    list_of_days = list(zip(start_days, end_days))
    mice = np.unique(res['mouse'])
    start_end_day_res = filter.filter_days_per_mouse(res, days_per_mouse= list_of_days)
    add_naive_learned(start_end_day_res, start_days, end_days)
    odor_list = ['US']
    colors = ['Turquoise']
    ax_args_copy.update({'xlim':[-1, 2]})
    for i, odor in enumerate(odor_list):
        plot.plot_results(start_end_day_res, select_dict={'odor_standard':odor},
                          x_key='training_day', y_key='Fraction Responsive', loop_keys='mouse',
                          colors= [colors[i]]*len(mice),
                          path =figure_path, plot_args=line_args, ax_args= ax_args_copy,
                          fig_size=(1.6, 1.5), legend=False)


def get_responsive_cells(res):
    list_of_data = res['sig']
    for data in list_of_data:
        res['Fraction Responsive'].append(np.mean(data))
    res['Fraction Responsive'] = np.array(res['Fraction Responsive'])


def plot_individual(res, lick_res, figure_path):
    ax_args_copy = ax_args.copy()
    ax_args_copy.update({'ylim':[0,.65], 'yticks':[0, .3, .6]})
    overlap_ax_args_copy = overlap_ax_args.copy()
    res = copy.copy(res)
    get_responsive_cells(res)
    summary_res = reduce.new_filter_reduce(res, reduce_key= 'Fraction Responsive',
                                           filter_keys=['odor_valence', 'mouse','day'])
    for mouse in np.unique(summary_res['mouse']):
        select_dict = {'mouse':mouse}

        plot.plot_results(summary_res, x_key='day', y_key='Fraction Responsive', loop_keys='odor_valence',
                          colors=['green','red','turquoise'],
                          select_dict=select_dict, path=figure_path, ax_args=ax_args_copy, plot_args = line_args,
                          save=False, sort=True)

        plot.plot_results(lick_res, x_key='day', y_key='lick_boolean', loop_keys='odor_valence',
                          select_dict={'mouse': mouse},
                          colors=['green','red'],
                          ax_args=overlap_ax_args_copy, plot_args=behavior_line_args,
                          path=figure_path,
                          reuse=True, save=True, twinax=True)