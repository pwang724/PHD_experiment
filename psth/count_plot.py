import os
from _CONSTANTS.config import Config
import filter
import reduce
import numpy as np
import matplotlib.pyplot as plt
import tools.file_io as fio
import plot
from behavior.behavior_analysis import get_days_per_mouse
import copy
from collections import defaultdict
import psth.count_analyze
import psth.psth_helper
from psth.count_helper import get_responsive_cells, get_overlap_water
from analysis import add_naive_learned
import psth.count_helper

ax_args = {'yticks': [0, .2, .4, .6, .8], 'ylim': [0, .85]}
overlap_ax_args = {'yticks': [0, .5, 1], 'ylim': [0, 1.05]}
trace_ax_args = {'yticks': [0, .1, .2, .3], 'ylim': [0, .3]}

trace_args = {'alpha':1, 'linewidth':1}
line_args = {'alpha': .5, 'linewidth': 1, 'marker': 'o', 'markersize': 2}
scatter_args = {'marker':'o', 's':5, 'facecolors': 'none', 'alpha': .5}
error_args = {'fmt': '.', 'capsize': 2, 'elinewidth': 1, 'markersize': 2, 'alpha': .8}
fill_args = {'zorder': 0, 'lw': 0, 'alpha': 0.3}
behavior_line_args = {'alpha': .5, 'linewidth': 1, 'marker': '.', 'markersize': 0, 'linestyle': '--'}

def plot_individual(res, lick_res):
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

def plot_summary_odor(res, start_days, end_days):
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

        temp = filter.filter(start_end_day_res, {'odor_standard':odor})
        plot.plot_results(temp,
                          x_key='odor_standard_training_day', y_key='Fraction Responsive', loop_keys='mouse',
                          colors= [colors[i]]*len(mice),
                          path =figure_path, plot_args=line_args, ax_args= ax_args_copy,
                          save=save_arg, reuse=reuse_arg,
                          fig_size=(2.5, 1.5),legend=False)

def plot_summary_water(res, start_days, end_days):
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
                          fig_size=(1.5, 1.5), legend=False)

def plot_overlap(res, start_days, end_days, delete_non_selective = False):
    ax_args_copy = overlap_ax_args.copy()
    res = copy.copy(res)
    res = psth.count_helper.get_overlap(res, delete_non_selective)
    list_of_days = list(zip(start_days, end_days))
    mice = np.unique(res['mouse'])
    start_end_day_res = filter.filter_days_per_mouse(res, days_per_mouse= list_of_days)
    start_end_day_res = reduce.new_filter_reduce(start_end_day_res, filter_keys=['mouse', 'day', 'condition'],
                                                 reduce_key='Overlap')
    add_naive_learned(start_end_day_res, start_days, end_days)
    odor_list = ['+:+', '-:-','+:-']
    colors = ['Green','Red','Gray']
    ax_args_copy.update({'xlim': [-1, 6]})
    filter.assign_composite(start_end_day_res, loop_keys=['condition', 'training_day'])

    for i, odor in enumerate(odor_list):
        save_arg = False
        reuse_arg = True
        if i == 0:
            reuse_arg = False
        if i == len(odor_list) -1:
            save_arg = True

        temp = filter.filter(start_end_day_res, {'condition':odor})
        plot.plot_results(temp,
                          x_key='condition_training_day', y_key='Overlap', loop_keys='mouse',
                          colors= [colors[i]]*len(mice),
                          path =figure_path, plot_args=line_args, ax_args= ax_args_copy,
                          save=save_arg, reuse=reuse_arg,
                          fig_size=(2, 1.5), legend=False)

def plot_overlap_water(res, start_days, end_days):
    ax_args_copy = overlap_ax_args.copy()
    res = copy.copy(res)
    mice = np.unique(res['mouse'])
    res = filter.filter_days_per_mouse(res, days_per_mouse= end_days)
    add_naive_learned(res, start_days, end_days)
    ax_args_copy.update({'xlim': [-1, 2]})
    y_keys = ['US/CS+', 'CS+/US']
    summary_res = defaultdict(list)
    for arg in y_keys:
        get_overlap_water(res, arg = arg)
        new_res = reduce.new_filter_reduce(res, filter_keys=['mouse','day','odor_valence'],
                                             reduce_key='Overlap')
        new_res['Type'] = np.array([arg] * len(new_res['training_day']))
        reduce.chain_defaultdicts(summary_res, new_res)

    summary_res.pop('Overlap_sem')
    summary_res.pop('Overlap_std')
    summary_res = filter.filter(summary_res, {'odor_valence':'CS+'})
    mean_std_res = reduce.new_filter_reduce(summary_res, filter_keys='Type', reduce_key='Overlap')
    types = np.unique(summary_res['Type'])
    for i, type in enumerate(types):
        reuse_arg = True
        if i == 0:
            reuse_arg = False
        temp = filter.filter(summary_res, {'Type':type})
        plot.plot_results(temp,
                          x_key='Type', y_key='Overlap', loop_keys='mouse',
                          colors=['Black'] * len(mice),
                          plot_function= plt.scatter,
                          path=figure_path, plot_args=scatter_args, ax_args=ax_args_copy,
                          save=False, reuse=reuse_arg,
                          fig_size=(1.5, 1.5), legend = False)

    plot.plot_results(mean_std_res,
                      x_key='Type', y_key='Overlap', error_key='Overlap_sem',
                      path=figure_path, plot_function=plt.errorbar, plot_args=error_args, ax_args=ax_args,
                      save=True, reuse=True,
                      fig_size=(1.5, 1.5), legend=False)

def plot_power(res, start_days, end_days):
    ax_args_copy = trace_ax_args.copy()
    ax_args_copy.update({'xticks':[res['DAQ_O_ON_F'][0], res['DAQ_W_ON_F'][0]], 'xticklabels':['ON', 'US']})
    res = copy.copy(res)
    psth.count_helper.normalize_across_days(res)
    psth.count_helper.get_power(res, normalize_across_days=False)
    list_of_days = list(zip(start_days, end_days))
    start_end_day_res = filter.filter_days_per_mouse(res, days_per_mouse=list_of_days)
    start_end_day_res = reduce.new_filter_reduce(start_end_day_res, filter_keys=['day', 'odor_valence'],
                                                 reduce_key='Power')


    add_naive_learned(start_end_day_res, start_days, end_days)
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

def plot_reversal(res, start_days, end_days):
    ax_args_copy = ax_args.copy()
    res = copy.copy(res)
    list_of_days = list(zip(start_days, end_days))
    start_end_day_res = filter.filter_days_per_mouse(res, days_per_mouse=list_of_days)
    reversal_res, stats_res = psth.count_helper.get_reversal_sig(start_end_day_res)
    filter.assign_composite(reversal_res, loop_keys=['day','odor_valence'])

    mean_res = reduce.new_filter_reduce(reversal_res, filter_keys=['day','odor_valence'], reduce_key='Fraction')
    plot.plot_results(mean_res,
                      x_key='day_odor_valence', y_key='Fraction', error_key='Fraction_sem',
                      path=figure_path,
                      plot_function=plt.errorbar, plot_args=error_args, ax_args=ax_args_copy,
                      fig_size=(2, 1.5), save=False)

    plot.plot_results(reversal_res,
                      x_key='day_odor_valence', y_key='Fraction', loop_keys= 'day_odor_valence',
                      path=figure_path,
                      colors = ['Green','Red','Red','Green'],
                      plot_function=plt.scatter, plot_args=scatter_args, ax_args=ax_args_copy,
                      fig_size=(2, 1.5), reuse=True, save=True,
                      legend=False)


    titles = ['','CS+', 'CS-', 'None']
    conditions = [['none-p','p-m','p-none', 'p-p'], ['p-m','p-none', 'p-p'],['m-m','m-none', 'm-p'],['none-m','none-none', 'none-p']]
    labels = [['Added','Reversed','Lost','Retained'], ['Reversed','Lost','Retained'], ['Retained','Lost','Reversed'], ['to CS-','Retained','to CS+']]
    for i, title in enumerate(titles):
        mean_stats = reduce.new_filter_reduce(stats_res, filter_keys='condition', reduce_key= 'Fraction')
        ax_args_copy.update({'ylim':[-.1, 1], 'yticks':[0, .5, 1], 'xticks':[0,1,2,3],
                             'xticklabels':labels[i]})
        plot.plot_results(mean_stats,
                          select_dict={'condition': conditions[i]},
                          x_key='condition', y_key='Fraction', loop_keys='mouse', error_key='Fraction_sem',
                          sort=True,
                          path=figure_path,
                          colors=['Black'] * 10,
                          plot_function=plt.errorbar, plot_args=error_args, ax_args=ax_args_copy,
                          fig_size=(2, 1.5), save=False)
        plt.title(title)

        plot.plot_results(stats_res, select_dict={'condition': conditions[i]},
                          x_key='condition', y_key='Fraction', loop_keys='mouse', sort=True,
                          path = figure_path,
                          colors=['Black']*10,
                          plot_function=plt.scatter, plot_args=scatter_args, ax_args=ax_args_copy,
                          fig_size=(2, 1.5),
                          legend=False, save=True, reuse=True)

condition_config = psth.count_analyze.OFC_LONGTERM_Config()

config = psth.psth_helper.PSTHConfig()
condition = condition_config.condition
data_path = os.path.join(Config.LOCAL_DATA_PATH, Config.LOCAL_DATA_TIMEPOINT_FOLDER, condition.name)
save_path = os.path.join(Config.LOCAL_EXPERIMENT_PATH, 'COUNTING', condition.name)
figure_path = os.path.join(Config.LOCAL_FIGURE_PATH, 'OTHER', 'COUNTING',  condition.name)

#retrieving relevant days
learned_day_per_mouse, last_day_per_mouse = get_days_per_mouse(data_path, condition)
if condition_config.start_at_training and hasattr(condition, 'training_start_day'):
    start_days_per_mouse = condition.training_start_day
else:
    start_days_per_mouse = [0] * len(learned_day_per_mouse)
training_start_day_per_mouse = condition.training_start_day
print(learned_day_per_mouse)

#analysis
res = fio.load_pickle(os.path.join(save_path, 'dict.pkl'))
psth.count_analyze.analyze_data(res, condition_config)

import behavior
import analysis
lick_res = behavior.behavior_analysis.get_licks_per_day(data_path, condition)
analysis.add_odor_value(lick_res, condition)
lick_res = filter.filter(lick_res, {'odor_valence': ['CS+', 'CS-', 'PT CS+']})
lick_res = reduce.new_filter_reduce(lick_res, ['odor_valence', 'day', 'mouse'], reduce_key='lick_boolean')

if condition.name == 'OFC' or condition.name == 'BLA':
    plot_individual(res, lick_res)
    plot_summary_odor(res, start_days_per_mouse, learned_day_per_mouse)
    plot_summary_water(res, training_start_day_per_mouse, learned_day_per_mouse)
    plot_overlap_water(res, training_start_day_per_mouse, learned_day_per_mouse)
    plot_overlap(res, start_days_per_mouse, learned_day_per_mouse)

if condition.name == 'BLA_LONGTERM':
    plot_individual(res, lick_res)
    plot_summary_odor(res, start_days_per_mouse, learned_day_per_mouse)
    plot_overlap(res, start_days_per_mouse, learned_day_per_mouse)

if condition.name == 'OFC_LONGTERM':
    plot_individual(res, lick_res)
    plot_summary_odor(res, learned_day_per_mouse, last_day_per_mouse)

if condition.name == 'PIR':
    # plot_individual(res, lick_res)
    # plot_summary_odor(res, start_days_per_mouse, learned_day_per_mouse)
    # plot_summary_water(res, training_start_day_per_mouse, learned_day_per_mouse)
    plot_overlap(res, start_days_per_mouse, learned_day_per_mouse, delete_non_selective=True)

if condition.name == 'OFC_STATE':
    # plot_summary_odor(res, start_days_per_mouse, last_day_per_mouse)
    plot_power(res, start_days_per_mouse, last_day_per_mouse)

if condition.name == 'OFC_REVERSAL':
    # plot_summary_odor(res, start_days_per_mouse, last_day_per_mouse)
    plot_reversal(res, start_days_per_mouse, last_day_per_mouse)


if condition.name == 'OFC_CONTEXT':
    # plot_summary_odor(res, start_days_per_mouse, last_day_per_mouse)
    plot_power(res, start_days_per_mouse, last_day_per_mouse)