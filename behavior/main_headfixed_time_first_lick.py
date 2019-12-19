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
from scipy.stats import wilcoxon
import behavior.behavior_config

plt.style.use('default')
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.size'] = 5
mpl.rcParams['font.family'] = 'arial'

experiments = [
    'summary_raw',
    # 'summary_line'
]

conditions = [
    # experimental_conditions.OFC_COMPOSITE,
    # experimental_conditions.MPFC_COMPOSITE,
    # experimental_conditions.BEHAVIOR_OFC_YFP_PRETRAINING,
    # experimental_conditions.BEHAVIOR_OFC_JAWS_PRETRAINING,
    # experimental_conditions.BEHAVIOR_OFC_HALO_PRETRAINING,
    experimental_conditions.BEHAVIOR_OFC_YFP_DISCRIMINATION,
    experimental_conditions.BEHAVIOR_OFC_JAWS_DISCRIMINATION,
    # experimental_conditions.BEHAVIOR_OFC_MUSH_HALO,
    # experimental_conditions.BEHAVIOR_OFC_MUSH_JAWS,
    # experimental_conditions.BEHAVIOR_OFC_MUSH_YFP,
    # experimental_conditions.OFC,
    # experimental_conditions.PIR,
    # experimental_conditions.OFC_LONGTERM,
    # experimental_conditions.BLA_LONGTERM,
    # experimental_conditions.BEHAVIOR_OFC_JAWS_MUSH,
    # experimental_conditions.BEHAVIOR_OFC_HALO_MUSH,
    # experimental_conditions.BEHAVIOR_OFC_JAWS_MUSH_UNUSED,
    # experimental_conditions.BEHAVIOR_OFC_MUSH_YFP,
    # experimental_conditions.BLA,
    # experimental_conditions.BLA_JAWS,
    # experimental_conditions.OFC_REVERSAL,
    # experimental_conditions.OFC_STATE
]

collapse_arg = 'condition'
def _collapse_conditions(res, control_condition, str):
    conditions = res['condition'].copy().astype('<U20')
    control_ix = conditions == control_condition
    conditions[control_ix] = 'YFP'
    conditions[np.invert(control_ix)] = 'INH'
    res[str] = conditions

list_of_res = []
names = []
behavior_strings = ['YFP', 'HALO', 'JAWS']
for i, condition in enumerate(conditions):
    if any(s in condition.name for s in behavior_strings):
        data_path = os.path.join(Config.LOCAL_DATA_PATH, Config.LOCAL_DATA_BEHAVIOR_FOLDER, condition.name)
    else:
        data_path = os.path.join(Config.LOCAL_DATA_PATH, Config.LOCAL_DATA_TIMEPOINT_FOLDER, condition.name)
    res = analyze_behavior(data_path, condition)

    if condition.name == 'OFC_LONGTERM':
        res = filter.exclude(res, {'mouse':3})

    if 'YFP' in condition.name:
        res['condition'] = np.array(['YFP'] * len(res['mouse']))
    elif 'JAWS' in condition.name:
        res['condition'] = np.array(['JAWS'] * len(res['mouse']))
    elif 'HALO' in condition.name:
        res['condition'] = np.array(['HALO'] * len(res['mouse']))
    else:
        res['condition'] = np.array([condition.name] * len(res['mouse']))

    list_of_res.append(res)
    names.append(condition.name)
directory_name = ','.join(names)
all_res = defaultdict(list)
for res, condition in zip(list_of_res, conditions):
    reduce.chain_defaultdicts(all_res, res)
save_path = os.path.join(Config.LOCAL_FIGURE_PATH, 'BEHAVIOR', directory_name)

color_dict_valence = {'PT CS+': 'C1', 'CS+': 'green', 'CS-': 'red'}
color_dict_condition = {'HALO': 'C1', 'JAWS':'red','YFP':'black'}
bool_ax_args = {'yticks': [0, 50, 100], 'ylim': [-5, 105], 'xticks': [0, 50, 100, 150, 200],
                'xlim': [0, 200]}
ax_args_mush = {'yticks': [0, 5, 10], 'ylim': [-1, 12],'xticks': [0, 50, 100],'xlim': [0, 100]}
bool_ax_args_mush = {'yticks': [0, 50, 100], 'ylim': [-5, 105], 'xticks': [0, 100], 'xlim': [0, 100]}
ax_args_dt = {'yticks': [0, 5, 10], 'ylim': [-1, 12],'xticks': [0, 50],'xlim': [0, 50]}
bool_ax_args_dt = {'yticks': [0, 50, 100], 'ylim': [-5, 105], 'xticks': [0, 50], 'xlim': [0, 50]}
ax_args_pt = {'yticks': [0, 5, 10], 'ylim': [-1, 12], 'xticks': [0, 50, 100, 150], 'xlim': [0, 150]}
bool_ax_args_pt = {'yticks': [0, 50, 100], 'ylim': [-5, 105], 'xticks': [0, 50, 100, 150], 'xlim': [0, 150]}
bar_args = {'alpha': .6, 'fill': False}
scatter_args = {'marker': 'o', 's': 10, 'alpha': .6}

collection = False
if collection:
    reduce_key = 'time_first_lick_collection_smoothed'
    xkey = 'time_first_lick_collection_trial'
    ax_args_local = {'yticks': [0, 1], 'ylim': [-.1, 1.5], 'xlabel': 'Time',
                     'xticks': [0, 50, 100], 'xlim': [0, 130]}
else:
    reduce_key = 'time_first_lick_smoothed'
    xkey = 'time_first_lick_trial'
    ax_args_local = {'yticks': [0, 2, 5], 'ylim': [-.1, 5], 'yticklabels': ['odor on', 'odor off', 'US'], 'xlabel': 'Time',
                     'xticks': [0, 50, 100], 'xlim': [0, 130]}

if 'summary_raw' in experiments:
    line_args_local = line_args.copy()
    line_args_local.update({'marker': '.', 'markersize':.5, 'linewidth': .75, 'alpha':.5})

    all_res_ = filter.filter(all_res, {'odor_valence': ['CS+', 'CS-', 'PT CS+']})
    _collapse_conditions(all_res_, control_condition='YFP', str=collapse_arg)
    filter.assign_composite(all_res_, [collapse_arg, 'odor_valence'])
    composite_arg = collapse_arg + '_' + 'odor_valence'

    valences = np.unique(all_res_['odor_valence'])
    for valence in valences:
        color = [color_dict_valence[valence]]
        for i in range(len(color)):
            color.append('black')

        if 'PT CS+' in valence or 'PT Naive' in valence:
            ax_args = ax_args_pt
        elif 'PT CS+' in all_res_['odor_valence']:
            ax_args = ax_args_dt
        else:
            ax_args = ax_args_mush

        plot.plot_results(all_res_, x_key=xkey, y_key= reduce_key, loop_keys= composite_arg,
                          rect = (.3, .2, .6, .6),
                          colors=color, select_dict={'odor_valence': valence},
                          ax_args=ax_args_local, plot_args=line_args_local,
                          path=save_path)

        if 'CS+' in valence:
            line_args_mean_sem = {'marker': '.', 'markersize': 0, 'linewidth': .75, 'alpha': .5}
            temp = filter.filter(all_res_,{'odor_valence':valence})
            mean_sem = reduce.new_filter_reduce(temp, filter_keys=['condition'], reduce_key=reduce_key, regularize='max')

            path, name = plot.plot_results(mean_sem, x_key=xkey, y_key= reduce_key,
                              loop_keys= composite_arg,
                              colors=color, select_dict={'odor_valence':valence},
                              ax_args=ax_args_local, plot_args=line_args_mean_sem,
                              save=False,
                              path=save_path)

            plot.plot_results(mean_sem, x_key=xkey, y_key=reduce_key, error_key=reduce_key+'_sem',
                              loop_keys= composite_arg,
                              rect = (.3, .2, .6, .6),
                              colors=color, select_dict={'odor_valence': valence},
                              ax_args=ax_args_local,
                              plot_function=plt.fill_between,
                              plot_args=fill_args,
                              reuse=True,
                              path=save_path, name_str='_mean_sem')

if 'summary_line' in experiments:
    r = defaultdict(list)
    duration = 20
    for i, x in enumerate(all_res['time_first_lick']):
        all_res['first_lick_A'].append(np.mean(x[:duration]))
        all_res['first_lick_B'].append(np.mean(x[-duration:]))
    for k, v in all_res.items():
        all_res[k] = np.array(v)
    all_res_lick = reduce.new_filter_reduce(all_res, filter_keys=['condition', 'odor_valence', 'mouse'],
                                            reduce_key='first_lick_A')
    _ = reduce.new_filter_reduce(all_res, filter_keys=['condition', 'odor_valence', 'mouse'],
                                            reduce_key='first_lick_B')
    all_res_lick['first_lick_B'] = _['first_lick_B']

    res_use = all_res
    valences = np.unique(all_res['odor_valence'])
    scatter_args_copy = scatter_args.copy()
    scatter_args_copy.update({'marker': '.', 'alpha': 1, 's': 10})
    ax_args_local = {'yticks': [0, 2, 5], 'ylim': [0, 5], 'xticks': [0, 2, 5], 'xlim': [0, 5],
                  'xticklabels':['ON', 'OFF', 'US'], 'yticklabels':['ON', 'OFF', 'US']}
    for valence in valences:
        color = color_dict_valence[valence]
        path, name = plot.plot_results(res_use, x_key='first_lick_A', y_key='first_lick_B',
                          select_dict={'odor_valence':valence},
                          rect=(.3, .2, .6, .6),
                          plot_function=plt.scatter,
                          plot_args=scatter_args_copy,
                                       ax_args = ax_args_local,
                          colors=color,
                          save=False,
                          path=save_path)
        plt.plot([0, 5], [0, 5], '--', color='gray')

        try:
            stat_res = filter.filter(res_use, {'odor_valence':valence})
            a, b = stat_res['first_lick_A'], stat_res['first_lick_B']
            stat = wilcoxon(a, b)[-1]
            ylim = plt.gca().get_ylim()
            sig_str = plot.significance_str(x=.4, y=.7 * (ylim[-1] - ylim[0]), val=stat)
            plot._easy_save(path, name)
            print(np.mean(a), np.mean(b))
            print(stat)
        except:
            print('no stats')


    reduce_key = 'time_first_lick_smoothed'


