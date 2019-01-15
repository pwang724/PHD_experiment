import os
from collections import defaultdict

import filter
from CONSTANTS import conditions as experimental_conditions
from CONSTANTS.config import Config
from behavior.behavior_analysis import convert, agglomerate_days, analyze_behavior
from reduce import filter_reduce
from tools.utils import chain_defaultdicts
import plot
import matplotlib.pyplot as plt
import numpy as np
import analysis

core_experiments = ['individual', 'individual_half_max', 'summary','basic_3']
# experiments = ['individual', 'individual_half_max', 'basic_3']
conditions = [experimental_conditions.PIR, experimental_conditions.OFC, experimental_conditions.BLA,
              experimental_conditions.OFC_LONGTERM, experimental_conditions.BLA_LONGTERM,
              experimental_conditions.OFC_JAWS, experimental_conditions.BLA_JAWS]

experiments = core_experiments
conditions = [experimental_conditions.PIR, experimental_conditions.OFC, experimental_conditions.BLA]

list_of_res = []
for i, condition in enumerate(conditions):
    data_path = os.path.join(Config.LOCAL_DATA_PATH, Config.LOCAL_DATA_TIMEPOINT_FOLDER, condition.name)
    res = analysis.load_all_cons(data_path)
    analysis.add_indices(res)
    analysis.add_time(res)
    lick_res = convert(res, condition)
    plot_res = agglomerate_days(lick_res, condition, condition.training_start_day,
                                filter.get_last_day_per_mouse(res))
    analyze_behavior(plot_res, condition)
    list_of_res.append(plot_res)

if 'individual' in experiments:
    for plot_res, condition in zip(list_of_res, conditions):
        save_path = os.path.join(Config.LOCAL_FIGURE_PATH, 'BEHAVIOR', condition.name)
        colors = ['green', 'lime', 'red', 'maroon']
        plot_args = {'marker': 'o', 'markersize': 1, 'alpha': .6, 'linewidth': 1}
        ax_args = {'yticks': [0, 10, 20, 30, 40], 'ylim': [-1, 41], 'xticks': [0, 20, 40, 60, 80, 100],
                   'xlim': [0, 100]}
        bool_ax_args = {'yticks': [0, 25, 50, 75, 100], 'ylim': [-5, 105], 'xticks': [0, 20, 40, 60, 80, 100],
                   'xlim': [0, 100]}

        mice = np.unique(plot_res['mouse'])
        for i, mouse in enumerate(mice):
            select_dict = {'mouse': mouse}
            plot.plot_results(plot_res, x_key='trial', y_key='lick_smoothed', loop_keys='odor_standard',
                              select_dict=select_dict, colors=colors, ax_args=ax_args, plot_args=plot_args,
                              path=save_path)
            plot.plot_results(plot_res, x_key='trial', y_key='lick', loop_keys='odor_standard',
                              select_dict=select_dict, colors=colors, ax_args=ax_args, plot_args=plot_args,
                              path=save_path)
            plot.plot_results(plot_res, x_key='trial', y_key='boolean_smoothed', loop_keys='odor_standard',
                              select_dict=select_dict, colors=colors, ax_args=bool_ax_args, plot_args=plot_args,
                              path=save_path)


if 'individual_half_max' in experiments:
    for plot_res, condition in zip(list_of_res, conditions):
        save_path = os.path.join(Config.LOCAL_FIGURE_PATH, 'BEHAVIOR', condition.name)

        # bar plot
        colors = ['black', 'black']
        select_dict = {'odor_valence': 'CS+'}
        nMouse = np.unique(plot_res['mouse']).size
        ax_args = {'yticks': [0, 20, 40, 60, 80], 'ylim': [0, 80], 'xticks': np.arange(nMouse)}
        plot_args = {'marker': 'o', 's': 10, 'facecolors': 'none', 'alpha': .6}
        plot.plot_results(plot_res, x_key='mouse', y_key='half_max', loop_keys='odor_standard', colors=colors,
                          select_dict=select_dict, path=save_path, plot_function=plt.scatter, plot_args=plot_args,
                          ax_args=ax_args, save=False)

        try:
            csp_res = filter.filter(plot_res, {'odor_valence':'CS+'})
            summary_res = filter_reduce(csp_res, filter_key='mouse', reduce_key='half_max')
            ax_args = {'yticks': [0, 20, 40, 60, 80], 'ylim': [0, 80], }
            plot_args = {'alpha': .6, 'fill': False}
            plot.plot_results(summary_res, x_key='mouse', y_key='half_max', loop_keys=None,
                              select_dict=select_dict, path=save_path, plot_function=plt.bar, plot_args=plot_args,
                              ax_args=ax_args, save=True, reuse=True)
        except:
            print('Cannot get half_max data for: {}'.format(condition.name))
            # raise ValueError('Cannot summarize: {}'.format(condition.name))

if 'summary' in experiments:
    plot_args = {'marker': '.', 'markersize': 1, 'alpha': .6, 'linewidth': 1}
    ax_args = {'yticks': [0, 10, 20, 30, 40], 'ylim': [-1, 41], 'xticks': [0, 20, 40, 60, 80, 100],
               'xlim': [0, 100]}
    for plot_res, condition in zip(list_of_res, conditions):
        save_path = os.path.join(Config.LOCAL_FIGURE_PATH, 'BEHAVIOR', condition.name)

        csp_res = filter.filter(plot_res, {'odor_valence':'CS+'})
        csp_summary = filter_reduce(csp_res, filter_key='mouse', reduce_key='lick_smoothed')
        csm_res = filter.filter(plot_res, {'odor_valence':'CS-'})
        csm_summary = filter_reduce(csm_res, filter_key='mouse', reduce_key='lick_smoothed')
        chain_defaultdicts(csp_summary, csm_summary)
        summary_res = csp_summary

        plot.plot_results(summary_res, x_key='trial', y_key='lick_smoothed', loop_keys= 'odor_valence',
                            colors= ['lime','salmon'], ax_args=ax_args, plot_args=plot_args,
                            path=save_path)


if 'basic_3' in experiments:
    summary_all = defaultdict(list)
    for plot_res, condition in zip(list_of_res, conditions):
        try:
            csp_res = filter.filter(plot_res, {'odor_valence':'CS+'})
            summary_res = filter_reduce(csp_res, filter_key='mouse', reduce_key='half_max')
            summary_res['condition_name'] = [condition.name] * len(summary_res['half_max'])
            chain_defaultdicts(summary_all, summary_res)
        except:
            print('Cannot get half_max data for: {}'.format(condition.name))

    mean_std_res = filter_reduce(summary_all, filter_key='condition_name', reduce_key='half_max')
    select_dict = {'condition_name': ['PIR','OFC','BLA']}
    save_path = os.path.join(Config.LOCAL_FIGURE_PATH, 'BEHAVIOR', 'TEST')
    ax_args = {'yticks':[0, 20, 40, 60], 'ylim':[-5, 65]}

    plot_args = {'marker':'o', 's':10, 'facecolors': 'none', 'alpha':1}
    plot.plot_results(summary_all, x_key='condition_name', y_key='half_max', loop_keys=None, select_dict= select_dict,
                      path=save_path,
                      plot_function= plt.scatter, plot_args= plot_args, ax_args= ax_args, save=False)

    plot_args = {'fmt':'.', 'capsize':2, 'elinewidth':1, 'markersize':2, 'alpha': .5}
    plot.plot_results(mean_std_res, x_key='condition_name', y_key='half_max', loop_keys=None, path=save_path,
                      select_dict= select_dict,
                      plot_function= plt.errorbar, plot_args= plot_args, ax_args = ax_args, save=True, reuse=True)

    plot_args = {'marker':'o', 's':10, 'facecolors': 'none', 'alpha':1}
    plot.plot_results(summary_all, x_key='condition_name', y_key='half_max', loop_keys=None,
                      path=save_path,
                      plot_function= plt.scatter, plot_args= plot_args, ax_args= ax_args, save=False)

    plot_args = {'fmt':'.', 'capsize':2, 'elinewidth':1, 'markersize':2, 'alpha': .5}
    plot.plot_results(mean_std_res, x_key='condition_name', y_key='half_max', loop_keys=None, path=save_path,
                      plot_function= plt.errorbar, plot_args= plot_args, ax_args = ax_args, save=True, reuse=True)