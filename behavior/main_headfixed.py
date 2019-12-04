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
import behavior.behavior_config

plt.style.use('default')
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.size'] = 5
mpl.rcParams['font.family'] = 'arial'

experiments = [
    # 'licks_per_day'
    # 'individual',
    # 'summary',
    'mean_sem',
    # 'trials_to_criterion',
    # 'roc',
]

conditions = [
    experimental_conditions.BEHAVIOR_OFC_YFP_PRETRAINING,
    experimental_conditions.BEHAVIOR_OFC_JAWS_PRETRAINING,
    experimental_conditions.BEHAVIOR_OFC_HALO_PRETRAINING,
    # experimental_conditions.BEHAVIOR_OFC_YFP_DISCRIMINATION,
    # experimental_conditions.BEHAVIOR_OFC_JAWS_DISCRIMINATION,
    # experimental_conditions.BEHAVIOR_OFC_MUSH_JAWS_HALO,
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
    # experimental_conditions.BEHAVIOR_OFC_JAWS_DISCRIMINATION
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

color_dict = {'PT CS+': 'C1', 'CS+':'green', 'CS-':'red'}
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
    lick = 'lick_collection'
    lick_smoothed = 'lick_collection_smoothed'
    boolean_smoothed = 'boolean_collection_smoothed'
    boolean_sem = 'boolean_collection_smoothed_sem'
    lick_sem = 'lick_collection_smoothed_sem'
else:
    lick = 'lick'
    lick_smoothed = 'lick_smoothed'
    boolean_smoothed = 'boolean_smoothed'
    boolean_sem = 'boolean_smoothed_sem'
    lick_sem = 'lick_smoothed_sem'

if 'licks_per_day' in experiments:
    line_args_copy = line_args.copy()
    line_args_copy.update({'linestyle':'--', 'linewidth':.5,'markersize':1.5, 'alpha':.3})
    ax_args_cur = ax_args.copy()
    ax_args_cur.update({'xticks':[0, 2, 4, 6, 8], 'xlim':[0, 8], 'ylim':[0, 11], 'yticks':[0, 5, 10]})
    for condition in conditions:
        if any(s in condition.name for s in behavior_strings):
            data_path = os.path.join(Config.LOCAL_DATA_PATH, Config.LOCAL_DATA_BEHAVIOR_FOLDER, condition.name)
        else:
            data_path = os.path.join(Config.LOCAL_DATA_PATH, Config.LOCAL_DATA_TIMEPOINT_FOLDER, condition.name)
        res = behavior.behavior_analysis.get_licks_per_day(data_path, condition, return_raw=True)
        save_path = os.path.join(Config.LOCAL_FIGURE_PATH, 'BEHAVIOR', condition.name)
        res_ = reduce.new_filter_reduce(res, ['odor_valence', 'day', 'mouse'], 'lick')

        colors = {'CS+':'green', 'CS-':'red','PT CS+':'orange'}
        for valence in np.unique(res_['odor_valence']):
            plot.plot_results(res_, x_key='day', y_key='lick', loop_keys='mouse',
                              select_dict={'odor_valence': valence},
                              colors= [colors[valence]] * 10, plot_args=line_args_copy, ax_args=ax_args_cur,
                              fig_size=[2, 1.5],
                              path=save_path, reuse=False, save=True)

if 'individual' in experiments:
    line_args_copy = line_args.copy()
    line_args_copy.update({'markersize':0})

    for res, condition in zip(list_of_res, conditions):
        save_path = os.path.join(Config.LOCAL_FIGURE_PATH, 'BEHAVIOR', condition.name)
        colors = ['green', 'lime', 'red', 'maroon']

        mice = np.unique(res['mouse'])
        for i, mouse in enumerate(mice):
            try:
                select_dict = {'mouse': mouse, 'odor': condition.pt_odors[i]}
                plot.plot_results(res, x_key='trial', y_key=lick_smoothed, loop_keys='odor_standard',
                                  select_dict=select_dict, colors=colors, ax_args=ax_args_pt, plot_args=line_args_copy,
                                  path=save_path)
                plot.plot_results(res, x_key='trial', y_key=boolean_smoothed, loop_keys='odor_standard',
                                  select_dict=select_dict, colors=colors, ax_args=bool_ax_args_pt, plot_args=line_args_copy,
                                  path=save_path)

                select_dict = {'mouse': mouse, 'odor': condition.dt_odors[i]}
                plot.plot_results(res, x_key='trial', y_key=lick_smoothed, loop_keys='odor_standard',
                                  select_dict=select_dict, colors=colors, ax_args=ax_args_dt, plot_args=line_args_copy,
                                  path=save_path)
                plot.plot_results(res, x_key='trial', y_key=boolean_smoothed, loop_keys='odor_standard',
                                  select_dict=select_dict, colors=colors, ax_args=bool_ax_args_dt, plot_args=line_args_copy,
                                  path=save_path)
            except:
                print('not two-phase')

            try:
                select_dict = {'mouse': mouse, 'odor': condition.odors[i]}
                plot.plot_results(res, x_key='trial', y_key=lick_smoothed, loop_keys='odor_standard',
                                  select_dict=select_dict, colors=colors, ax_args=ax_args_mush, plot_args=line_args_copy,
                                  path=save_path)
                plot.plot_results(res, x_key='trial', y_key=boolean_smoothed, loop_keys='odor_standard',
                                  select_dict=select_dict, colors=colors, ax_args=bool_ax_args_mush, plot_args=line_args_copy,
                                  path=save_path)
            except:
                print('not one-phase')

if 'summary' in experiments:
    all_res = defaultdict(list)

    for res, condition in zip(list_of_res, conditions):
        reduce.chain_defaultdicts(all_res, res)
    save_path = os.path.join(Config.LOCAL_FIGURE_PATH, 'BEHAVIOR', directory_name)

    all_res = filter.filter(all_res, {'odor_valence':['CS+','CS-', 'PT CS+']})
    _collapse_conditions(all_res, control_condition='YFP', str=collapse_arg)
    filter.assign_composite(all_res, [collapse_arg, 'odor_valence'])
    composite_arg = collapse_arg + '_' + 'odor_valence'
    all_res_lick = reduce.new_filter_reduce(all_res, filter_keys=['condition', 'odor_valence','mouse'], reduce_key=lick_smoothed)
    all_res_bool = reduce.new_filter_reduce(all_res, filter_keys=['condition', 'odor_valence','mouse'], reduce_key=boolean_smoothed)

    line_args_copy = line_args.copy()
    line_args_copy.update({'marker': None, 'linewidth':.75})

    valences = np.unique(all_res['odor_valence'])
    valences = [[x] for x in valences]
    valences.append(['CS+','CS-'])
    for valence in valences:
        color = [color_dict[x] for x in valence]
        for i in range(len(color)):
            color.append('black')

        all_res_bool_ = all_res_bool.copy()
        all_res_bool_ = filter.filter(all_res_bool_, {'odor_valence':valence})
        all_res_lick_ = all_res_lick.copy()
        all_res_lick_ = filter.filter(all_res_lick_, {'odor_valence':valence})

        if 'PT CS+' in valence or 'PT Naive' in valence:
            ax_args = ax_args_pt
            bool_ax_args = bool_ax_args_pt
        elif 'PT CS+' in all_res['odor_valence']:
            ax_args = ax_args_dt
            bool_ax_args = bool_ax_args_dt
        else:
            ax_args = ax_args_mush
            bool_ax_args = bool_ax_args_mush

        path, name = plot.plot_results(all_res_bool_, x_key='trial', y_key=boolean_smoothed, loop_keys= composite_arg,
                          colors=color, select_dict={'odor_valence':valence},
                          ax_args=bool_ax_args, plot_args=line_args_copy,
                          reuse = False, save=False,
                          path=save_path)
        c = behavior.behavior_config.behaviorConfig()

        if 'CS+' in valence or 'PT CS+' in valence:
            y = c.fully_learned_threshold_up
            plt.plot(plt.xlim(), [y, y], '--', color = 'gray', linewidth =.5)

        if 'CS-' in valence:
            y = c.fully_learned_threshold_down
            plt.plot(plt.xlim(), [y, y], '--', color='gray', linewidth=.5)
        plot._easy_save(path=path, name=name)

        plot.plot_results(all_res_lick_, x_key='trial', y_key=lick_smoothed, loop_keys= composite_arg,
                          colors=color, select_dict={'odor_valence':valence},
                          ax_args=ax_args, plot_args=line_args_copy,
                          reuse = False, save=True,
                          path=save_path)

if 'mean_sem' in experiments:
    all_res = defaultdict(list)

    for res, condition in zip(list_of_res, conditions):
        reduce.chain_defaultdicts(all_res, res)
    save_path = os.path.join(Config.LOCAL_FIGURE_PATH, 'BEHAVIOR', directory_name)

    all_res = filter.filter(all_res, {'odor_valence':['CS+','CS-', 'PT CS+']})
    _collapse_conditions(all_res, control_condition='YFP', str = collapse_arg)
    composite_arg = collapse_arg + '_' + 'odor_valence'
    filter.assign_composite(all_res, [collapse_arg, 'odor_valence'])
    all_res_lick = reduce.new_filter_reduce(all_res, filter_keys=['condition', 'odor_valence'], reduce_key=lick_smoothed,
                                            regularize='max')
    all_res_bool = reduce.new_filter_reduce(all_res, filter_keys=['condition', 'odor_valence'], reduce_key=boolean_smoothed,
                                            regularize='max')

    line_args_copy = line_args.copy()
    line_args_copy.update({'marker': None, 'linewidth':.75})

    valences = np.unique(all_res['odor_valence'])
    valences = [[x] for x in valences]
    valences.append(['CS+','CS-'])
    for valence in valences:
        color = [color_dict[x] for x in valence]
        for i in range(len(color)):
            color.append('black')

        if 'PT CS+' in valence or 'PT Naive' in valence:
            ax_args = ax_args_pt
            bool_ax_args = bool_ax_args_pt
        elif 'PT CS+' in all_res['odor_valence']:
            ax_args = ax_args_dt
            bool_ax_args = bool_ax_args_dt
        else:
            ax_args = ax_args_mush
            bool_ax_args = bool_ax_args_mush

        path, name = plot.plot_results(all_res_bool, x_key='trial', y_key=boolean_smoothed,
                          loop_keys= composite_arg,
                          colors=color, select_dict={'odor_valence':valence},
                          ax_args=bool_ax_args, plot_args=line_args_copy,
                          save=False,
                          path=save_path)

        plot.plot_results(all_res_bool, x_key='trial', y_key=boolean_smoothed, error_key='boolean_smoothed_sem',
                          loop_keys= composite_arg,
                          colors= color, select_dict={'odor_valence':valence},
                          ax_args=bool_ax_args, plot_args= fill_args,
                          save = False, reuse=True,
                          plot_function= plt.fill_between,
                          path=save_path)

        c = behavior.behavior_config.behaviorConfig()

        if 'CS+' in valence or 'PT CS+' in valence:
            y = c.fully_learned_threshold_up
            plt.plot(plt.xlim(), [y, y], '--', color = 'gray', linewidth =.5)

        if 'CS-' in valence:
            y = c.fully_learned_threshold_down
            plt.plot(plt.xlim(), [y, y], '--', color='gray', linewidth=.5)
        plot._easy_save(path=path, name=name + '_mean_sem')

        path, name = plot.plot_results(all_res_lick, x_key='trial', y_key=lick_smoothed,
                          loop_keys= composite_arg,
                          colors=color, select_dict={'odor_valence':valence},
                          ax_args=ax_args, plot_args=line_args_copy,
                          save=False,
                          path=save_path)

        plot.plot_results(all_res_lick, x_key='trial', y_key=lick_smoothed, error_key='lick_smoothed_sem',
                          loop_keys= composite_arg,
                          colors= color, select_dict={'odor_valence':valence},
                          ax_args=ax_args, plot_args= fill_args,
                          save = True, reuse=True,
                          plot_function= plt.fill_between,
                          path=save_path, name_str='_mean_sem')

if 'roc' in experiments:
    save_path = os.path.join(Config.LOCAL_FIGURE_PATH, 'BEHAVIOR', directory_name)
    x = 'roc_trial'
    y = 'roc'

    all_res = defaultdict(list)
    from behavior import behavior_analysis
    for res, condition in zip(list_of_res, conditions):
        if 'PT CS+' in res['odor_valence']:
            temp = filter.filter(res, {'odor_valence':['CS+','CS-']})
        else:
            temp = res.copy()
        res_ = behavior_analysis.get_roc(temp)
        reduce.chain_defaultdicts(all_res, res_)

    _collapse_conditions(all_res, control_condition='YFP', str = collapse_arg)
    filter.assign_composite(all_res, [collapse_arg, 'odor_valence'])

    ax_args_local = {'yticks': [0, .5, 1], 'ylim': [-.05, 1.05], 'xticks': [0, 50, 100],
                     'xlim': [0, 125]}
    line_args_local = {'alpha': .5, 'linewidth': 1, 'marker': '.', 'markersize': 0}

    ctrl_res = filter.filter(all_res, {'odor_valence':'CS+', collapse_arg:'YFP'})
    summary = reduce.new_filter_reduce(ctrl_res, filter_keys=['odor_valence', collapse_arg],
                                       reduce_key=y, regularize='max')
    summary[x] = np.arange(summary[x].size).reshape(1,-1)

    plot.plot_results(all_res, x_key=x, y_key=y,
                      select_dict={'odor_valence': 'CS+', collapse_arg: 'YFP'},
                      loop_keys=['mouse'],
                      ax_args=ax_args_local,
                      plot_args=line_args_local,
                      colors = ['black']*20, reuse=False, save=False,
                      path=save_path)

    plot.plot_results(all_res, x_key=x, y_key=y,
                      select_dict={'odor_valence':'CS+', collapse_arg:'INH'},
                      loop_keys=['mouse'],
                      colors=['green'] * 20,
                      ax_args=ax_args_local,
                      plot_args=line_args_local,
                      reuse =True, save = True,
                      legend=False,
                      path=save_path)


if 'trials_to_criterion' in experiments:

    reduce_key = 'criterion'
    all_res = defaultdict(list)
    for res, condition in zip(list_of_res, conditions):
        reduce.chain_defaultdicts(all_res, res)
    save_path = os.path.join(Config.LOCAL_FIGURE_PATH, 'BEHAVIOR', directory_name)
    all_res = reduce.new_filter_reduce(all_res, filter_keys=['mouse', 'odor_valence', 'condition'], reduce_key=reduce_key)
    _collapse_conditions(all_res, control_condition='YFP', str = collapse_arg)
    all_res.pop(reduce_key + '_std')
    all_res.pop(reduce_key + '_sem')

    filter.assign_composite(all_res, [collapse_arg, 'odor_valence'])
    mean_std_res = reduce.new_filter_reduce(all_res, filter_keys=[collapse_arg, 'odor_valence'],reduce_key=reduce_key)
    x = all_res[reduce_key]

    scatter_args_copy = scatter_args.copy()
    scatter_args_copy.update({'marker': '.', 'alpha': .5, 's': 10})
    error_args_copy = error_args.copy()
    error_args_copy.update({'elinewidth': .5, 'markeredgewidth': .5, 'markersize': 0})
    xlim_1 = np.unique(all_res[collapse_arg]).size
    ax_args_pt_ = {'yticks': [0, 50, 100, 150], 'ylim': [-10, 160], 'xlim':[-1, xlim_1]}
    ax_args_dt_ = {'yticks': [0, 25, 50], 'ylim': [-5, 55], 'xlim':[-1, xlim_1]}
    ax_args_mush_ = {'yticks': [0, 50, 100], 'ylim': [-5, 125], 'xlim':[-1, xlim_1]}

    x_key = collapse_arg + '_odor_valence'
    for valence in np.unique(all_res['odor_valence']):
        if valence == 'PT CS+' or valence == 'PT Naive':
            ax_args = ax_args_pt_
        elif 'PT CS+' in all_res['odor_valence']:
            ax_args = ax_args_dt_
        else:
            ax_args = ax_args_mush_

        swarm_args_copy = swarm_args.copy()
        swarm_args_copy.update({'palette':[color_dict[valence],'black'], 'size':5})

        path, name = plot.plot_results(all_res, x_key=collapse_arg, y_key= reduce_key,
                                       select_dict={'odor_valence': valence},
                                       ax_args=ax_args,
                                       plot_function= sns.stripplot,
                                       plot_args= swarm_args_copy,
                                       sort=True,
                                       fig_size=[2, 1.5],
                                       path=save_path, reuse=False, save=False)

        plot.plot_results(mean_std_res, x_key=collapse_arg, y_key= reduce_key, error_key= reduce_key + '_sem',
                          select_dict={'odor_valence': valence},
                          ax_args=ax_args,
                          plot_function= plt.errorbar,
                          plot_args= error_args,
                          fig_size=[2, 1.5],
                          path=save_path, reuse=True, save=False)
        plt.xlim(-1, 2)

        test = filter.filter(all_res, {'odor_valence': valence})
        ixs = test[collapse_arg] == 'YFP'
        x = test[reduce_key][ixs]
        y = test[reduce_key][np.invert(ixs)]
        rs = ranksums(x, y)[-1]
        ylim = plt.gca().get_ylim()
        sig_str = plot.significance_str(x=.4, y=.7 * (ylim[-1] - ylim[0]), val=rs)
        plot._easy_save(path, name, pdf=True)

    #stats
    print(mean_std_res[x_key])
    print(mean_std_res[reduce_key])
    ix_a = all_res[x_key] == 'YFP_CS+'
    ix_b = all_res[x_key] == 'INH_CS+'
    ix_c = all_res[x_key] == 'YFP_CS-'
    ix_d = all_res[x_key] == 'INH_CS-'
    from scipy.stats import ranksums
    rsplus = ranksums(all_res[reduce_key][ix_a], all_res[reduce_key][ix_b])
    rsminus = ranksums(all_res[reduce_key][ix_c], all_res[reduce_key][ix_d])
    print(rsplus)
    print(rsminus)

    try:
        ix_e = all_res[x_key] == 'YFP_PT CS+'
        ix_f = all_res[x_key] == 'INH_PT CS+'
        rspt = ranksums(all_res[reduce_key][ix_e], all_res[reduce_key][ix_f])
        print(all_res[reduce_key][ix_e])
        print(all_res[reduce_key][ix_f])
        print(rspt)
    except:
        print('no pt')