import os
from collections import defaultdict

import filter
from _CONSTANTS import conditions as experimental_conditions
from _CONSTANTS.config import Config
from behavior.behavior_analysis import analyze_behavior
from reduce import chain_defaultdicts
import plot
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import reduce
from format import *
from scipy.stats import ranksums

plt.style.use('default')
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.size'] = 5
mpl.rcParams['font.family'] = 'arial'

experiments = [
    # 'individual',
    # 'individual_half_max',
    'summary',
    # 'trials_to_criterion',
    # 'basic_3'
]
conditions = [
    experimental_conditions.BEHAVIOR_OFC_YFP,
    experimental_conditions.BEHAVIOR_OFC_JAWS_PRETRAINING,
    experimental_conditions.BEHAVIOR_OFC_JAWS_DISCRIMINATION,
    experimental_conditions.OFC_COMPOSITE,
    experimental_conditions.MPFC_COMPOSITE
]
collapse_arg = 'condition_pretraining'
def _collapse_conditions(res, experimental_condition, str):
    conditions = res['condition'].copy()
    control_ix = conditions != experimental_condition
    conditions[control_ix] = 'YFP_ALL'
    conditions[np.invert(control_ix)] = 'JAWS'
    res[str] = conditions

list_of_res = []
for i, condition in enumerate(conditions):
    if condition == experimental_conditions.OFC_COMPOSITE or condition == experimental_conditions.MPFC_COMPOSITE:
        data_path = os.path.join(Config.LOCAL_DATA_PATH, Config.LOCAL_DATA_TIMEPOINT_FOLDER, condition.name)
    else:
        data_path = os.path.join(Config.LOCAL_DATA_PATH, Config.LOCAL_DATA_BEHAVIOR_FOLDER, condition.name)
    res = analyze_behavior(data_path, condition)

    if 'COMPOSITE' in condition.name:
        res['condition'] = np.array([condition.name] * len(res['mouse']))
    else:
        res['condition'] = np.array([condition.name.rsplit('_', 1)[-1]] * len(res['mouse']))
    list_of_res.append(res)

bool_ax_args = {'yticks': [0, 25, 50, 75, 100], 'ylim': [-5, 105], 'xticks': [0, 50, 100, 150, 200],
                'xlim': [0, 200]}
ax_args_pt = {'yticks': [0, 5, 10, 15], 'ylim': [-1, 16],
              # 'xticks': [0, 50, 100, 150, 200], 'xlim': [0, 200]
              }
ax_args_dt = {'yticks': [0, 5, 10, 15], 'ylim': [-1, 16],
              # 'xticks': [0, 50, 100],'xlim': [0, 100]
              }
bool_ax_args_dt = {'yticks': [0, 25, 50, 75, 100], 'ylim': [-5, 105], 'xticks': [0, 100],
                   'xlim': [0, 100]}
bool_ax_args_pt = {'yticks': [0, 25, 50, 75, 100], 'ylim': [-5, 105], 'xticks': [0, 50, 100, 150, 200],
                   'xlim': [0, 200]}
bar_args = {'alpha': .6, 'fill': False}
scatter_args = {'marker': 'o', 's': 10, 'alpha': .6}

if 'individual' in experiments:
    line_args_copy = line_args.copy()

    for res, condition in zip(list_of_res, conditions):
        save_path = os.path.join(Config.LOCAL_FIGURE_PATH, 'BEHAVIOR', condition.name)
        colors = ['green', 'lime', 'red', 'maroon']

        mice = np.unique(res['mouse'])
        for i, mouse in enumerate(mice):
            select_dict = {'mouse': mouse, 'odor': condition.dt_odors[i]}
            plot.plot_results(res, x_key='trial', y_key='lick_smoothed', loop_keys='odor_standard',
                              select_dict=select_dict, colors=colors, ax_args=ax_args_dt, plot_args=line_args_copy,
                              path=save_path)
            plot.plot_results(res, x_key='trial', y_key='lick', loop_keys='odor_standard',
                              select_dict=select_dict, colors=colors, ax_args=ax_args_dt, plot_args=line_args_copy,
                              path=save_path)
            plot.plot_results(res, x_key='trial', y_key='boolean_smoothed', loop_keys='odor_standard',
                              select_dict=select_dict, colors=colors, ax_args=bool_ax_args_dt, plot_args=line_args_copy,
                              path=save_path)

            select_dict = {'mouse': mouse, 'odor': condition.pt_csp[i]}
            plot.plot_results(res, x_key='trial', y_key='lick_smoothed', loop_keys='odor_standard',
                              select_dict=select_dict, colors=colors, ax_args=ax_args_pt, plot_args=line_args_copy,
                              path=save_path)
            plot.plot_results(res, x_key='trial', y_key='lick', loop_keys='odor_standard',
                              select_dict=select_dict, colors=colors, ax_args=ax_args_pt, plot_args=line_args_copy,
                              path=save_path)
            plot.plot_results(res, x_key='trial', y_key='boolean_smoothed', loop_keys='odor_standard',
                              select_dict=select_dict, colors=colors, ax_args=bool_ax_args_pt, plot_args=line_args_copy,
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
                plot.plot_results(res, x_key='mouse', y_key=y_key, loop_keys='odor_standard', colors=colors,
                                  select_dict=select_dict, path=save_path, plot_function=plt.scatter, plot_args=scatter_args,
                                  ax_args=ax_args, save=False)

                csp_res = filter.filter(res, select_dict)
                summary_res = reduce.new_filter_reduce(csp_res, filter_keys='mouse', reduce_key=y_key)
                ax_args = {'yticks': [0, 50, 100, 150], 'ylim': [0, 150], }
                plot.plot_results(summary_res, x_key='mouse', y_key=y_key,
                                  select_dict=select_dict, path=save_path, plot_function=plt.bar, plot_args=bar_args,
                                  ax_args=ax_args, save=True, reuse=True)

if 'summary' in experiments:
    all_res = defaultdict(list)

    for res, condition in zip(list_of_res, conditions):
        reduce.chain_defaultdicts(all_res, res)
    save_path = os.path.join(Config.LOCAL_FIGURE_PATH, 'BEHAVIOR', ','.join([x for x in np.unique(all_res['condition'])]))

    all_res = filter.filter(all_res, {'odor_valence':['CS+','CS-', 'PT CS+']})
    all_res_lick = reduce.new_filter_reduce(all_res, filter_keys=['condition', 'odor_valence','mouse'], reduce_key='lick_smoothed')
    all_res_bool = reduce.new_filter_reduce(all_res, filter_keys=['condition', 'odor_valence','mouse'], reduce_key='boolean_smoothed')
    if collapse_arg == 'condition_pretraining':
        _collapse_conditions(all_res_lick, experimental_condition='PRETRAINING', str = collapse_arg)
        _collapse_conditions(all_res_bool, experimental_condition='PRETRAINING', str = collapse_arg)
    elif collapse_arg == 'condition_discrimination':
        _collapse_conditions(all_res_lick, experimental_condition='DISCRIMINATION', str = collapse_arg)
        _collapse_conditions(all_res_bool, experimental_condition='DISCRIMINATION', str = collapse_arg)
    else:
        collapse_arg = 'condition'
    all_res_bool.pop('boolean_smoothed_sem')
    all_res_lick.pop('lick_smoothed_sem')

    line_args_copy = line_args.copy()
    line_args_copy.update({'marker': None, 'linewidth':.75})
    for valence in np.unique(all_res['odor_valence']):
        for condition in np.unique(all_res_lick[collapse_arg]):
            if condition == 'YFP' or condition == 'YFP_ALL':
                color = 'black'
            else:
                color = 'red'

            if valence == 'PT CS+':
                ax_args = ax_args_pt.copy()
                ax_args.update({'xlim':[-10, 170],'xticks':[0, 50, 100, 150]})
                bool_ax_args = bool_ax_args_pt
                bool_ax_args.update({'xlim':[-10, 170],'xticks':[0, 50, 100, 150]})
            else:
                ax_args = ax_args_dt.copy()
                ax_args.update({'xlim':[-5, 55],'xticks':[0, 25, 50]})
                bool_ax_args = bool_ax_args_dt
                bool_ax_args.update({'xlim': [-5, 55], 'xticks': [0, 25, 50]})
            #
            # plot.plot_results(all_res_lick, x_key='trial', y_key='lick_smoothed',
            #                   select_dict={'odor_valence':valence, collapse_arg:condition},
            #                   colors = color,
            #                   ax_args = ax_args, plot_args= line_args_copy,
            #                   path = save_path)
            #
            # plot.plot_results(all_res_bool, x_key='trial', y_key='boolean_smoothed',
            #                   select_dict={'odor_valence':valence, collapse_arg:condition},
            #                   colors = color,
            #                   ax_args = bool_ax_args, plot_args= line_args_copy,
            #                   path = save_path)

        plot.plot_results(all_res_bool, x_key='trial', y_key='boolean_smoothed',
                          select_dict={'odor_valence': valence, collapse_arg: 'JAWS'},
                          colors='red',
                          ax_args=bool_ax_args, plot_args=line_args_copy,
                          reuse = False, save=False,
                          path=save_path)
        summary = reduce.new_filter_reduce(all_res_bool, filter_keys=['odor_valence', collapse_arg], reduce_key='boolean_smoothed',
                                           regularize='max')
        plot.plot_results(summary, x_key='trial', y_key='boolean_smoothed',
                          select_dict={'odor_valence': valence, collapse_arg: 'YFP_ALL'},
                          ax_args=bool_ax_args,
                          plot_args=line_args_copy,
                          colors='black', reuse=True, save=False,
                          path=save_path)
        plot.plot_results(summary, x_key='trial', y_key= 'boolean_smoothed', error_key='boolean_smoothed_sem',
                          select_dict={'odor_valence': valence, collapse_arg: 'YFP_ALL'},
                          ax_args=bool_ax_args,
                          plot_function=plt.fill_between, plot_args=fill_args,
                          colors='black', reuse=True, save=True,
                          path=save_path)


        plot.plot_results(all_res_lick, x_key='trial', y_key='lick_smoothed',
                          select_dict={'odor_valence': valence, collapse_arg: 'JAWS'},
                          colors='red',
                          ax_args=ax_args, plot_args=line_args_copy,
                          reuse = False, save=False,
                          path=save_path)
        summary = reduce.new_filter_reduce(all_res_lick, filter_keys=['odor_valence', collapse_arg], reduce_key='lick_smoothed',
                                           regularize='max')
        plot.plot_results(summary, x_key='trial', y_key='lick_smoothed',
                          select_dict={'odor_valence': valence, collapse_arg: 'YFP_ALL'},
                          ax_args=ax_args,
                          plot_args=line_args_copy,
                          colors='black', reuse=True, save=False,
                          path=save_path)
        plot.plot_results(summary, x_key='trial', y_key= 'lick_smoothed', error_key='lick_smoothed_sem',
                          select_dict={'odor_valence': valence, collapse_arg: 'YFP_ALL'},
                          ax_args=ax_args,
                          plot_function=plt.fill_between, plot_args=fill_args,
                          colors='black', reuse=True, save=True,
                          path=save_path)


        # if valence == 'PT CS+':
        #     ax_args = ax_args_pt.copy()
        #     ax_args.update({'xlim': [-10, 170], 'xticks': [0, 50, 100, 150]})
        #     bool_ax_args = bool_ax_args_pt
        #     bool_ax_args.update({'xlim': [-10, 170], 'xticks': [0, 50, 100, 150]})
        # else:
        #     ax_args = ax_args_dt.copy()
        #     ax_args.update({'xlim': [-5, 55], 'xticks': [0, 25, 50]})
        #     bool_ax_args = bool_ax_args_dt
        #     bool_ax_args.update({'xlim': [-5, 55], 'xticks': [0, 25, 50]})
        # plot.plot_results(all_res_lick, x_key='trial', y_key='lick_smoothed', loop_keys=collapse_arg,
        #                   select_dict={'odor_valence': valence},
        #                   colors=['red','black'],
        #                   ax_args=ax_args, plot_args=line_args_copy,
        #                   path=save_path)
        #
        # plot.plot_results(all_res_bool, x_key='trial', y_key='boolean_smoothed', loop_keys=collapse_arg,
        #                   select_dict={'odor_valence': valence},
        #                   colors=['red','black'],
        #                   ax_args=bool_ax_args, plot_args=line_args_copy,
        #                   path=save_path)

if 'trials_to_criterion' in experiments:

    reduce_key = 'criterion'
    all_res = defaultdict(list)
    for res, condition in zip(list_of_res, conditions):
        reduce.chain_defaultdicts(all_res, res)
    save_path = os.path.join(Config.LOCAL_FIGURE_PATH, 'BEHAVIOR',
                                 ','.join([x for x in np.unique(all_res['condition'])]))
    all_res = reduce.new_filter_reduce(all_res, filter_keys=['mouse', 'odor_valence', 'condition'], reduce_key=reduce_key)
    all_res.pop(reduce_key + '_std')
    all_res.pop(reduce_key + '_sem')
    if collapse_arg == 'condition_pretraining':
        _collapse_conditions(all_res, experimental_condition='PRETRAINING', str = collapse_arg)
        _collapse_conditions(all_res, experimental_condition='PRETRAINING', str = collapse_arg)
    elif collapse_arg == 'condition_discrimination':
        _collapse_conditions(all_res, experimental_condition='DISCRIMINATION', str = collapse_arg)
        _collapse_conditions(all_res, experimental_condition='DISCRIMINATION', str = collapse_arg)
    else:
        collapse_arg = 'condition'

    filter.assign_composite(all_res, [collapse_arg, 'odor_valence'])
    mean_std_res = reduce.new_filter_reduce(all_res, filter_keys=[collapse_arg, 'odor_valence'],reduce_key=reduce_key)
    x = all_res[reduce_key]

    scatter_args_copy = scatter_args.copy()
    scatter_args_copy.update({'marker': '.', 'alpha': .5, 's': 10})
    error_args_copy = error_args.copy()
    error_args_copy.update({'elinewidth': .5, 'markeredgewidth': .5, 'markersize': 0})
    ylim_1 = np.unique(all_res[collapse_arg]).size
    ax_args_pt_ = {'yticks': [0, 50, 100, 150], 'ylim': [-10, 160], 'xlim':[-1, ylim_1]}
    ax_args_dt_ = {'yticks': [0, 25, 50], 'ylim': [-5, 55], 'xlim':[-1, ylim_1]}

    x_key = collapse_arg + '_odor_valence'
    for valence in ['PT CS+', 'CS+','CS-']:
        if valence == 'PT CS+':
            ax_args = ax_args_pt_
        else:
            ax_args = ax_args_dt_

        swarm_args_copy = swarm_args.copy()

        if collapse_arg == 'condition_pretraining':
            swarm_args_copy.update({'palette':['black','red'], 'size':5})
        else:
            swarm_args_copy.update({'palette':['red','black'], 'size':5})
        path, name = plot.plot_results(all_res, x_key='odor_valence', y_key= reduce_key, loop_keys= x_key,
                          select_dict={'odor_valence': valence},
                          ax_args=ax_args,
                          plot_function= sns.swarmplot,
                        plot_args= swarm_args_copy,
                          fig_size=[2, 1.5],
                           path=save_path, reuse=False, save=False)
        plt.xlim(-1, 1)

        test = filter.filter(all_res, {'odor_valence': valence})
        ixs = test[collapse_arg] == 'YFP_ALL'
        x = test[reduce_key][ixs]
        y = test[reduce_key][np.invert(ixs)]
        rs = ranksums(x, y)[-1]
        ylim = plt.gca().get_ylim()
        sig_str = plot.significance_str(x=.4, y=.7 * (ylim[-1] - ylim[0]), val=rs)
        plot._easy_save(path, name, pdf=True)


if 'basic_3' in experiments:
    reduce_key = 'half_max'


    for valence in ['CS+','CS-', 'PT CS+']:
        summary_all = defaultdict(list)
        for res, condition in zip(list_of_res, conditions):
            try:
                csp_res = filter.filter(res, {'odor_valence': valence})
                summary_res = reduce.new_filter_reduce(csp_res, filter_keys='mouse', reduce_key=reduce_key)
                chain_defaultdicts(summary_all, summary_res)
            except:
                print('Cannot get half_max data for: {}'.format(condition.name))

        #TODO: bad work-around
        summary_all.pop(reduce_key + '_std')
        summary_all.pop(reduce_key + '_sem')
        mean_std_res = reduce.new_filter_reduce(summary_all, filter_keys='condition_name', reduce_key=reduce_key)
        save_path = os.path.join(Config.LOCAL_FIGURE_PATH, 'BEHAVIOR', 'COMPOSITE',
                                 ','.join([c.name for c in conditions]))
        ax_args = {'yticks':[0, 50, 100, 150], 'ylim':[-5, 155]}

        # select_dict = {'condition_name': ['PIR','OFC','BLA']}

        line_args_copy = {'marker': 'o', 's':10, 'facecolors': 'none', 'alpha':1}
        plot.plot_results(summary_all, x_key='condition_name', y_key=reduce_key, loop_keys=None,
                          path=save_path,
                          plot_function= plt.scatter, plot_args= line_args_copy, ax_args= ax_args, save=False)

        line_args_copy = {'fmt': '.', 'capsize':2, 'elinewidth':1, 'markersize':2, 'alpha': .5}
        plot.plot_results(mean_std_res, x_key='condition_name', y_key=reduce_key, error_key=reduce_key + '_sem',
                          select_dict= {'odor_valence':valence},
                          loop_keys=None, path=save_path,
                          plot_function= plt.errorbar, plot_args= line_args_copy, ax_args = ax_args, save=True, reuse=True)