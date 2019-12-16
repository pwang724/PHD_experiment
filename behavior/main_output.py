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
    # 'licks_per_day',
    # 'summary',
    # 'mean_sem',
    # 'trials_to_criterion',
    'cdf'
]

conditions = [
    experimental_conditions.BEHAVIOR_OFC_OUTPUT_CHANNEL,
    experimental_conditions.BEHAVIOR_OFC_OUTPUT_YFP,
]

list_of_res = []
names = []
behavior_strings = ['YFP', 'HALO', 'JAWS']
for i, condition in enumerate(conditions):
    data_path = os.path.join(Config.LOCAL_DATA_PATH, Config.LOCAL_DATA_BEHAVIOR_FOLDER, condition.name)
    res = analyze_behavior(data_path, condition)

    if 'YFP' in condition.name:
        res['condition'] = np.array(['YFP'] * len(res['mouse']))
    elif 'CHANNEL' in condition.name:
        res['condition'] = np.array(['CHANNEL'] * len(res['mouse']))
    else:
        res['condition'] = np.array([condition.name] * len(res['mouse']))

    list_of_res.append(res)
    names.append(condition.name)
directory_name = ','.join(names)
save_path_all = os.path.join(Config.LOCAL_FIGURE_PATH, 'BEHAVIOR', directory_name)
all_res = defaultdict(list)
for res, condition in zip(list_of_res, conditions):
    reduce.chain_defaultdicts(all_res, res)

color_dict = {'PT CS+': 'C1', 'CS+':'green', 'CS-':'red'}
ax_args_pt = {'yticks': [0, 5, 10], 'ylim': [-1, 12], 'xticks': [0, 100, 200, 300], 'xlim': [0, 300]}
bool_ax_args_pt = {'yticks': [0, 50, 100], 'ylim': [-5, 105], 'xticks': [0, 100, 200, 300], 'xlim': [0, 300]}
bar_args = {'alpha': .6, 'fill': False}
scatter_args = {'marker': 'o', 's': 10, 'alpha': .6}


lick = 'lick'
lick_smoothed = 'lick_smoothed'
boolean_smoothed = 'boolean_smoothed'
boolean_sem = 'boolean_smoothed_sem'
lick_sem = 'lick_smoothed_sem'

if 'individual' in experiments:
    line_args_copy = line_args.copy()
    line_args_copy.update({'markersize':0})

    for res, condition in zip(list_of_res, conditions):
        save_path = os.path.join(Config.LOCAL_FIGURE_PATH, 'BEHAVIOR', condition.name)
        colors = ['green', 'lime', 'red', 'maroon']

        mice = np.unique(res['mouse'])
        for i, mouse in enumerate(mice):
            select_dict = {'mouse': mouse, 'odor': condition.pt_odors[i]}
            plot.plot_results(res, x_key='trial', y_key=lick_smoothed, loop_keys='odor_standard',
                              select_dict=select_dict, colors=colors, ax_args=ax_args_pt, plot_args=line_args_copy,
                              path=save_path)
            plot.plot_results(res, x_key='trial', y_key=boolean_smoothed, loop_keys='odor_standard',
                              select_dict=select_dict, colors=colors, ax_args=bool_ax_args_pt, plot_args=line_args_copy,
                              path=save_path)

if 'cdf' in experiments:
    ctrl = filter.filter(all_res, {'condition':'YFP'})
    channel = filter.filter(all_res, {'condition':'CHANNEL'})
    ctrl_licks = ctrl['lick']
    channel_licks = channel['lick']

    # #shorten
    # ctrl_min = np.min([len(x) for x in ctrl_licks])
    # channel_min = np.min([len(x) for x in channel_licks])
    # both_min = np.min([ctrl_min, channel_min])
    # _shorten = lambda array, length: [x[:length] for x in array]
    # ctrl_licks = _shorten(ctrl_licks, both_min)
    # channel_licks = _shorten(channel_licks, both_min)

    #concatenate
    ctrl_licks = np.concatenate(ctrl_licks)
    channel_licks = np.concatenate(channel_licks)

    #get cdf
    range = [-.1, 13]
    def _cdf(data, range):
        num_bins = np.arange(range[0], range[1], .1)
        counts, bin_edges = np.histogram(data, bins=num_bins, range=range)
        counts = counts / np.sum(counts)
        cdf = np.cumsum(counts)
        return cdf, bin_edges

    ctrl_cdf, edges = _cdf(ctrl_licks, range)
    channel_cdf, _ = _cdf(channel_licks, range)

    #plot cdf
    res = defaultdict(list)
    res['condition'] = ['YFP', 'CHANNEL']
    res['lick'] = [edges[1:], edges[1:]]
    res['cdf'] = [ctrl_cdf, channel_cdf]
    for k, v in res.items():
        res[k] = np.array(v)

    line_args_copy = line_args.copy()
    line_args_copy.update({'marker': None, 'linewidth':1})
    plot.plot_results(res, x_key='lick', y_key='cdf', loop_keys='condition', colors=['blue','black'],
                      plot_args=line_args_copy,
                      path=save_path_all)

if 'summary' in experiments:
    all_res_lick = reduce.new_filter_reduce(all_res, filter_keys=['condition', 'odor_valence','mouse'], reduce_key=lick_smoothed)
    all_res_bool = reduce.new_filter_reduce(all_res, filter_keys=['condition', 'odor_valence','mouse'], reduce_key=boolean_smoothed)

    line_args_copy = line_args.copy()
    line_args_copy.update({'marker': None, 'linewidth':.75})

    valences = np.unique(all_res['odor_valence'])
    for valence in valences:
        color = [color_dict[valence]]
        color.append('black')

        path, name = plot.plot_results(all_res_bool, x_key='trial', y_key=boolean_smoothed, loop_keys= 'condition',
                                       colors=color, select_dict={'odor_valence':valence},
                                       ax_args=bool_ax_args_pt, plot_args=line_args_copy,
                                       reuse = False, save=False,
                                       path=save_path_all)
        c = behavior.behavior_config.behaviorConfig()
        y = c.fully_learned_threshold_up
        plt.plot(plt.xlim(), [y, y], '--', color = 'gray', linewidth =.5)
        plot._easy_save(path=path, name=name)

        plot.plot_results(all_res_lick, x_key='trial', y_key=lick_smoothed, loop_keys= 'condition',
                          colors=color, select_dict={'odor_valence':valence},
                          ax_args=ax_args_pt, plot_args=line_args_copy,
                          reuse = False, save=True,
                          path=save_path_all)

if 'mean_sem' in experiments:
    all_res_lick = reduce.new_filter_reduce(all_res, filter_keys=['condition', 'odor_valence'], reduce_key=lick_smoothed,
                                            regularize='max')
    all_res_bool = reduce.new_filter_reduce(all_res, filter_keys=['condition', 'odor_valence'], reduce_key=boolean_smoothed,
                                            regularize='max')

    line_args_copy = line_args.copy()
    line_args_copy.update({'marker': None, 'linewidth':.75})

    valences = np.unique(all_res['odor_valence'])
    for valence in valences:
        color = [color_dict[valence]]
        color.append('black')

        path, name = plot.plot_results(all_res_bool, x_key='trial', y_key=boolean_smoothed,
                                       loop_keys= 'condition',
                                       colors=color, select_dict={'odor_valence':valence},
                                       ax_args=bool_ax_args_pt, plot_args=line_args_copy,
                                       save=False,
                                       path=save_path_all)

        plot.plot_results(all_res_bool, x_key='trial', y_key=boolean_smoothed, error_key='boolean_smoothed_sem',
                          loop_keys= 'condition',
                          colors= color, select_dict={'odor_valence':valence},
                          ax_args=bool_ax_args_pt, plot_args= fill_args,
                          save = False, reuse=True,
                          plot_function= plt.fill_between,
                          path=save_path_all)

        c = behavior.behavior_config.behaviorConfig()
        y = c.fully_learned_threshold_up
        plt.plot(plt.xlim(), [y, y], '--', color = 'gray', linewidth =.5)
        plot._easy_save(path=path, name=name + '_mean_sem')

        plot.plot_results(all_res_lick, x_key='trial', y_key=lick_smoothed,
                                       loop_keys= 'condition',
                                       colors=color, select_dict={'odor_valence':valence},
                                       ax_args=ax_args_pt, plot_args=line_args_copy,
                                       save=False,
                                       path=save_path_all)

        plot.plot_results(all_res_lick, x_key='trial', y_key=lick_smoothed, error_key='lick_smoothed_sem',
                          loop_keys= 'condition',
                          colors= color, select_dict={'odor_valence':valence},
                          ax_args=ax_args_pt, plot_args= fill_args,
                          save = True, reuse=True,
                          plot_function= plt.fill_between,
                          path=save_path_all, name_str='_mean_sem')

if 'trials_to_criterion' in experiments:
    reduce_key = 'criterion'
    collapse_arg = 'condition'
    mean_std_res = reduce.new_filter_reduce(all_res, filter_keys=[collapse_arg, 'odor_valence'],reduce_key=reduce_key)
    x = all_res[reduce_key]

    scatter_args_copy = scatter_args.copy()
    scatter_args_copy.update({'marker': '.', 'alpha': .5, 's': 10})
    error_args_copy = error_args.copy()
    error_args_copy.update({'elinewidth': .5, 'markeredgewidth': .5, 'markersize': 0})
    xlim_1 = np.unique(all_res[collapse_arg]).size
    ax_args_pt_ = {'yticks': [0, 50, 100, 150, 200], 'ylim': [-10, 210], 'xlim':[-1, xlim_1]}
    ax_args_dt_ = {'yticks': [0, 25, 50], 'ylim': [-5, 55], 'xlim':[-1, xlim_1]}
    ax_args_mush_ = {'yticks': [0, 50, 100], 'ylim': [-5, 125], 'xlim':[-1, xlim_1]}

    x_key = collapse_arg
    for valence in np.unique(all_res['odor_valence']):
        swarm_args_copy = swarm_args.copy()
        swarm_args_copy.update({'palette':[color_dict[valence],'black'], 'size':5})

        path, name = plot.plot_results(all_res, x_key=collapse_arg, y_key= reduce_key,
                                       select_dict={'condition': 'CHANNEL'},
                                       ax_args=ax_args_pt_,
                                       plot_function= sns.stripplot,
                                       plot_args= swarm_args_copy,
                                       sort=True,
                                       fig_size=[1.5, 1.5], rect=(.3, .2, .6, .6),
                                       path=save_path_all, reuse=False, save=False)

        plot.plot_results(mean_std_res, x_key=collapse_arg, y_key= reduce_key, error_key= reduce_key + '_sem',
                          select_dict={'condition': 'CHANNEL'},
                          ax_args=ax_args_pt_,
                          plot_function= plt.errorbar,
                          plot_args= error_args,
                          fig_size=[2, 1.5],
                          path=save_path_all, reuse=True, save=False)
        plt.xlim(-.1, .1)
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