import os
from collections import defaultdict

import filter
from _CONSTANTS import conditions as experimental_conditions
from _CONSTANTS.config import Config
from behavior.behavior_analysis import analyze_behavior
from reduce import filter_reduce, chain_defaultdicts
import plot
import matplotlib.pyplot as plt
import numpy as np
import reduce
import analysis

experiments = [
    'individual',
    # 'individual_half_max',
    # 'basic_3'
]
conditions = [
    # experimental_conditions.PIR,
    # experimental_conditions.OFC,
    # experimental_conditions.BLA,
    # experimental_conditions.OFC_LONGTERM,
    # experimental_conditions.BLA_LONGTERM,
    experimental_conditions.BEHAVIOR_OFC_JAWS_MUSH,
    # experimental_conditions.BLA_JAWS,
    # experimental_conditions.OFC_REVERSAL,
    # experimental_conditions.OFC_STATE
]

colors = ['green', 'lime', 'red', 'maroon']
line_args = {'marker': 'o', 'markersize': 0, 'alpha': .7, 'linewidth': 1}
ax_args = {'yticks': [0, 10, 20], 'ylim': [-1, 21], 'xticks': [0, 25, 50, 75],
           'xlim': [0, 75]}
bool_ax_args = {'yticks': [0, 50, 100], 'ylim': [-5, 105], 'xticks': [0, 25, 50, 75],
                'xlim': [0, 75]}


list_of_res = []
for i, condition in enumerate(conditions):
    data_path = os.path.join(Config.LOCAL_DATA_PATH, Config.LOCAL_DATA_TIMEPOINT_FOLDER, condition.name)
    res = analyze_behavior(data_path, condition)
    list_of_res.append(res)

if 'individual' in experiments:
    for res, condition in zip(list_of_res, conditions):
        if condition.name == 'BEHAVIOR_OFC_JAWS_MUSH':
            ax_args.update({'xticks':[0,50,100,150], 'xlim':[0, 150]})
            bool_ax_args.update({'xticks':[0,50,100,150], 'xlim':[0, 150]})
        elif condition.name == 'OFC_STATE':
            ax_args.update({'xticks': [0, 25], 'xlim': [0, 25]})
            bool_ax_args.update({'xticks': [0, 25], 'xlim': [0, 25]})
        else:
            ax_args.update({'xticks': [0, 25, 50, 75], 'xlim': [0, 100]})
            bool_ax_args.update({'xticks': [0, 25, 50, 75], 'xlim': [0, 100]})

        save_path = os.path.join(Config.LOCAL_FIGURE_PATH, 'BEHAVIOR', condition.name)

        mice = np.unique(res['mouse'])
        for i, mouse in enumerate(mice):
            select_dict = {'mouse': mouse}
            plot.plot_results(res, x_key='trial', y_key='lick_smoothed', loop_keys='odor_standard',
                              select_dict=select_dict, colors=colors, ax_args=ax_args, plot_args=line_args,
                              path=save_path)
            plot.plot_results(res, x_key='trial', y_key='lick', loop_keys='odor_standard',
                              select_dict=select_dict, colors=colors, ax_args=ax_args, plot_args=line_args,
                              path=save_path)
            plot.plot_results(res, x_key='trial', y_key='boolean_smoothed', loop_keys='odor_standard',
                              select_dict=select_dict, colors=colors, ax_args=bool_ax_args, plot_args=line_args,
                              path=save_path)

        summary_res = reduce.new_filter_reduce(res, filter_keys=['mouse', 'odor_valence'], reduce_key= 'lick_smoothed')
        plot.plot_results(summary_res, x_key='trial', y_key='lick_smoothed', loop_keys= 'odor_valence',
                          colors= ['green','red'], ax_args=ax_args, plot_args=line_args,
                          path=save_path)

        summary_res = reduce.new_filter_reduce(res, filter_keys=['mouse', 'odor_valence'], reduce_key= 'boolean_smoothed')
        plot.plot_results(summary_res, x_key='trial', y_key='boolean_smoothed', loop_keys= 'odor_valence',
                          colors= ['green','red'], ax_args=bool_ax_args, plot_args=line_args,
                          path=save_path)


if 'individual_half_max' in experiments:
    for res, condition in zip(list_of_res, conditions):
        save_path = os.path.join(Config.LOCAL_FIGURE_PATH, 'BEHAVIOR', condition.name)

        # bar plot
        colors = ['black', 'black']
        select_dict = {'odor_valence': 'CS+'}
        nMouse = np.unique(res['mouse']).size
        ax_args = {'yticks': [0, 20, 40, 60, 80], 'ylim': [0, 80], 'xticks': np.arange(nMouse)}
        line_args = {'marker': 'o', 's': 10, 'facecolors': 'none', 'alpha': .6}
        plot.plot_results(res, x_key='mouse', y_key='half_max', loop_keys='odor_standard', colors=colors,
                          select_dict=select_dict, path=save_path, plot_function=plt.scatter, plot_args=line_args,
                          ax_args=ax_args, save=False)

        try:
            csp_res = filter.filter(res, {'odor_valence': 'CS+'})
            summary_res = filter_reduce(csp_res, filter_keys='mouse', reduce_key='half_max')
            ax_args = {'yticks': [0, 20, 40, 60, 80], 'ylim': [0, 80], }
            line_args = {'alpha': .6, 'fill': False}
            plot.plot_results(summary_res, x_key='mouse', y_key='half_max', loop_keys=None,
                              select_dict=select_dict, path=save_path, plot_function=plt.bar, plot_args=line_args,
                              ax_args=ax_args, save=True, reuse=True)
        except:
            print('Cannot get half_max data for: {}'.format(condition.name))

if 'basic_3' in experiments:
    summary_all = defaultdict(list)
    for res, condition in zip(list_of_res, conditions):
        try:
            csp_res = filter.filter(res, {'odor_valence': 'CS+'})
            summary_res = reduce.new_filter_reduce(csp_res, filter_keys='mouse', reduce_key='half_max')
            summary_res['condition_name'] = [condition.name] * len(summary_res['half_max'])
            chain_defaultdicts(summary_all, summary_res)
        except:
            print('Cannot get half_max data for: {}'.format(condition.name))
    #TODO: bad work-around
    summary_all.pop('half_max_sem')
    summary_all.pop('half_max_std')
    mean_std_res = filter_reduce(summary_all, filter_keys='condition_name', reduce_key='half_max')
    save_path = os.path.join(Config.LOCAL_FIGURE_PATH, 'BEHAVIOR', 'COMPOSITE',
                             ','.join([c.name for c in conditions]))
    ax_args = {'yticks':[0, 50, 100], 'ylim':[-5, 105]}
    line_args = {'marker': 'o', 's':10, 'facecolors': 'none', 'alpha':1}
    plot.plot_results(summary_all, x_key='condition_name', y_key='half_max', loop_keys=None,
                      path=save_path,
                      plot_function= plt.scatter, plot_args= line_args, ax_args= ax_args, save=False)

    line_args = {'fmt': '.', 'capsize':2, 'elinewidth':1, 'markersize':2, 'alpha': .5}
    plot.plot_results(mean_std_res, x_key='condition_name', y_key='half_max', error_key='half_max_sem',
                      loop_keys=None, path=save_path,
                      plot_function= plt.errorbar, plot_args= line_args, ax_args = ax_args, save=True, reuse=True)