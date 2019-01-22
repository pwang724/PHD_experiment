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

core_experiments = ['individual', 'individual_half_max', 'summary','basic_3']
conditions = [
    experimental_conditions.BEHAVIOR_OFC_YFP,
    experimental_conditions.BEHAVIOR_OFC_JAWS_PRETRAINING,
    experimental_conditions.BEHAVIOR_OFC_JAWS_DISCRIMINATION,
]

experiments = ['individual','summary', 'basic_3']
experiments = ['individual']
conditions = [
    experimental_conditions.BEHAVIOR_OFC_YFP,
    experimental_conditions.BEHAVIOR_OFC_JAWS_PRETRAINING,
    # experimental_conditions.BEHAVIOR_OFC_JAWS_DISCRIMINATION
              ]
# conditions = [experimental_conditions.BEHAVIOR_OFC_JAWS_DISCRIMINATION]

list_of_res = []
for i, condition in enumerate(conditions):
    data_path = os.path.join(Config.LOCAL_DATA_PATH, Config.LOCAL_DATA_BEHAVIOR_FOLDER, condition.name)
    res = analyze_behavior(data_path, condition)
    res['condition_name'] = np.array([condition.name] * len(res['half_max']))
    list_of_res.append(res)

# ax_args = {'yticks': [0, .2, .4, .6, .8, 1.0], 'ylim': [-.05, 1.05]}
line_args = {'alpha': .7, 'linewidth': 1, 'marker': '.', 'markersize': 0}
bool_ax_args = {'yticks': [0, 25, 50, 75, 100], 'ylim': [-5, 105], 'xticks': [0, 50, 100, 150, 200],
                'xlim': [0, 200]}
ax_args_pt = {'yticks': [0, 10, 20], 'ylim': [-1, 21], 'xticks': [0, 50, 100, 150, 200],
              'xlim': [0, 200]}
ax_args_dt = {'yticks': [0, 10, 20], 'ylim': [-1, 21], 'xticks': [0, 50],
              'xlim': [0, 50]}
bool_ax_args_dt = {'yticks': [0, 25, 50, 75, 100], 'ylim': [-5, 105], 'xticks': [0, 100],
                   'xlim': [0, 100]}
bool_ax_args_pt = {'yticks': [0, 25, 50, 75, 100], 'ylim': [-5, 105], 'xticks': [0, 50, 100, 150, 200],
                   'xlim': [0, 200]}

if 'individual' in experiments:
    for res, condition in zip(list_of_res, conditions):
        save_path = os.path.join(Config.LOCAL_FIGURE_PATH, 'BEHAVIOR', condition.name)
        colors = ['green', 'lime', 'red', 'maroon']

        mice = np.unique(res['mouse'])
        for i, mouse in enumerate(mice):
            select_dict = {'mouse': mouse, 'odor': condition.dt_odors[i]}
            plot.plot_results(res, x_key='trial', y_key='lick_smoothed', loop_keys='odor_standard',
                              select_dict=select_dict, colors=colors, ax_args=ax_args_dt, plot_args=line_args,
                              path=save_path)
            plot.plot_results(res, x_key='trial', y_key='lick', loop_keys='odor_standard',
                              select_dict=select_dict, colors=colors, ax_args=ax_args_dt, plot_args=line_args,
                              path=save_path)
            plot.plot_results(res, x_key='trial', y_key='boolean_smoothed', loop_keys='odor_standard',
                              select_dict=select_dict, colors=colors, ax_args=bool_ax_args_dt, plot_args=line_args,
                              path=save_path)

            select_dict = {'mouse': mouse, 'odor': condition.pt_odors[i]}
            plot.plot_results(res, x_key='trial', y_key='lick_smoothed', loop_keys='odor_standard',
                              select_dict=select_dict, colors=colors, ax_args=ax_args_pt, plot_args=line_args,
                              path=save_path)
            plot.plot_results(res, x_key='trial', y_key='lick', loop_keys='odor_standard',
                              select_dict=select_dict, colors=colors, ax_args=ax_args_pt, plot_args=line_args,
                              path=save_path)
            plot.plot_results(res, x_key='trial', y_key='boolean_smoothed', loop_keys='odor_standard',
                              select_dict=select_dict, colors=colors, ax_args=bool_ax_args_pt, plot_args=line_args,
                              path=save_path)


if 'individual_half_max' in experiments:
    for res, condition in zip(list_of_res, conditions):
        save_path = os.path.join(Config.LOCAL_FIGURE_PATH, 'BEHAVIOR', condition.name)

        # bar plot
        colors = ['black', 'black']
        select_dicts = [
            {'odor_valence': 'PT CS+'},
            {'odor_valence': 'CS+'},
            {'odor_valence': 'CS-'}
        ]
        y_keys = ['half_max', 'criterion']

        for select_dict in select_dicts:
            for y_key in y_keys:
                nMouse = np.unique(res['mouse']).size
                ax_args = {'yticks': [0, 50, 100, 150], 'ylim': [0, 150], 'xticks': np.arange(nMouse)}
                line_args = {'marker': 'o', 's': 10, 'facecolors': 'none', 'alpha': .6}
                plot.plot_results(res, x_key='mouse', y_key=y_key, loop_keys='odor_standard', colors=colors,
                                  select_dict=select_dict, path=save_path, plot_function=plt.scatter, plot_args=line_args,
                                  ax_args=ax_args, save=False)

                csp_res = filter.filter(res, select_dict)
                summary_res = filter_reduce(csp_res, filter_key='mouse', reduce_key=y_key)
                ax_args = {'yticks': [0, 50, 100, 150], 'ylim': [0, 150], }
                line_args = {'alpha': .6, 'fill': False}
                plot.plot_results(summary_res, x_key='mouse', y_key=y_key,
                                  select_dict=select_dict, path=save_path, plot_function=plt.bar, plot_args=line_args,
                                  ax_args=ax_args, save=True, reuse=True)


if 'summary' in experiments:
    for res, condition in zip(list_of_res, conditions):
        save_path = os.path.join(Config.LOCAL_FIGURE_PATH, 'BEHAVIOR', condition.name)

        nMouse = np.unique(res['mouse']).size
        select_dict = {'odor': condition.pt_csp}
        plot.plot_results(res, x_key='trial', y_key='false_negative', loop_keys='mouse',
                          select_dict=select_dict,
                          ax_args=bool_ax_args, plot_args=line_args,
                          path=save_path)

        summary_res = defaultdict(list)
        mice = np.unique(res['mouse'])
        for valence in np.unique(res['odor_valence']):
            for mouse in mice:
                temp_res = filter.filter(res, filter_dict={'mouse': mouse, 'odor_valence': valence})
                out_res = filter_reduce(temp_res, filter_key='odor_valence', reduce_key='lick_smoothed')
                reduce.chain_defaultdicts(summary_res, out_res)

        for valence in np.unique(res['odor_valence']):
            plot.plot_results(summary_res, x_key='trial', y_key='lick_smoothed',loop_keys='mouse',
                              select_dict={'odor_valence': valence},
                              ax_args=ax_args_pt, plot_args=line_args,
                              path=save_path)


if 'basic_3' in experiments:
    reduce_key = 'half_max'
    for valence in ['CS+','CS-', 'PT CS+']:
        summary_all = defaultdict(list)
        for res, condition in zip(list_of_res, conditions):
            try:
                csp_res = filter.filter(res, {'odor_valence': valence})
                summary_res = filter_reduce(csp_res, filter_key='mouse', reduce_key=reduce_key)
                chain_defaultdicts(summary_all, summary_res)
            except:
                print('Cannot get half_max data for: {}'.format(condition.name))

        #TODO: bad work-around
        summary_all.pop(reduce_key + '_std')
        summary_all.pop(reduce_key + '_sem')
        mean_std_res = filter_reduce(summary_all, filter_key='condition_name', reduce_key=reduce_key)
        save_path = os.path.join(Config.LOCAL_FIGURE_PATH, 'BEHAVIOR', 'COMPOSITE')
        ax_args = {'yticks':[0, 50, 100, 150], 'ylim':[-5, 155]}

        # select_dict = {'condition_name': ['PIR','OFC','BLA']}

        line_args = {'marker': 'o', 's':10, 'facecolors': 'none', 'alpha':1}
        plot.plot_results(summary_all, x_key='condition_name', y_key=reduce_key, loop_keys=None,
                          path=save_path,
                          plot_function= plt.scatter, plot_args= line_args, ax_args= ax_args, save=False)

        line_args = {'fmt': '.', 'capsize':2, 'elinewidth':1, 'markersize':2, 'alpha': .5}
        plot.plot_results(mean_std_res, x_key='condition_name', y_key=reduce_key, error_key=reduce_key + '_sem',
                          select_dict= {'odor_valence':valence},
                          loop_keys=None, path=save_path,
                          plot_function= plt.errorbar, plot_args= line_args, ax_args = ax_args, save=True, reuse=True)