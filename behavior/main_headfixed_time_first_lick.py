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
mpl.rcParams['font.size'] = 7
mpl.rcParams['font.family'] = 'arial'

experiments = [
    # 'summary_raw',
    # 'summary_line',
    # 'summary_hist',
    'summary_mouse_line'
]

conditions = [
    # experimental_conditions.OFC_COMPOSITE,
    # experimental_conditions.MPFC_COMPOSITE,
    # experimental_conditions.BEHAVIOR_OFC_YFP_PRETRAINING,
    # experimental_conditions.BEHAVIOR_OFC_JAWS_PRETRAINING,
    # experimental_conditions.BEHAVIOR_OFC_HALO_PRETRAINING,
    experimental_conditions.BEHAVIOR_OFC_YFP_DISCRIMINATION,
    experimental_conditions.BEHAVIOR_OFC_HALO_DISCRIMINATION,
    # experimental_conditions.BEHAVIOR_OFC_JAWS_DISCRIMINATION,
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

arg = 'com' #first, com, lick
collection = False

if arg == 'first':
    if collection:
        reduce_key_raw = 'time_first_lick_collection'
        reduce_key = 'time_first_lick_collection_smoothed'
        xkey = 'time_first_lick_collection_trial'
        ax_args_local = {'yticks': [0, 1], 'ylim': [-.1, 1.5], 'xlabel': 'Time',
                         'xticks': [0, 50, 100], 'xlim': [0, 130]}
    else:
        reduce_key_raw = 'time_first_lick'
        reduce_key = 'time_first_lick_smoothed'
        xkey = 'time_first_lick_trial'
        ax_args_local = {'yticks': [0, 2, 5], 'ylim': [-.1, 5], 'yticklabels': ['ON', 'OFF', 'US'],
                         'xlabel': 'Time', 'xticks': [0, 50, 100], 'xlim': [0, 130]}
elif arg == 'com':
    reduce_key_raw = 'lick_com'
    reduce_key = 'lick_com_smoothed'
    xkey = 'lick_com_trial'
    ax_args_local = {'yticks': [0, 2, 5], 'ylim': [-.1, 5], 'yticklabels': ['ON', 'OFF', 'US'],
                     'xlabel': 'Time','xticks': [0, 50, 100], 'xlim': [0, 130]}\

elif arg == 'lick':
    reduce_key_raw = 'lick'
    reduce_key = 'lick'
    xkey = 'trial'
    ax_args_local = {'yticks': [0, 15, 35], 'ylim': [0, 35],
                     'xlabel': 'Time','xticks': [0, 50, 100], 'xlim': [0, 130]}
elif arg == 'lick_5s':
    reduce_key_raw = 'lick_5s'
    reduce_key = 'lick_5s'
    xkey = 'trial'
    ax_args_local = {'yticks': [0, 15, 35], 'ylim': [0, 35],
                     'xlabel': 'Time','xticks': [0, 50, 100], 'xlim': [0, 130]}
else:
    raise ValueError('wtf')



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
                          rect = (.3, .3, .6, .6),
                          colors=color, select_dict={'odor_valence': valence},
                          ax_args=ax_args_local, plot_args=line_args_local,
                          path=save_path)

        if 'CS+' in valence:
            line_args_mean_sem = {'marker': '.', 'markersize': 0, 'linewidth': .75, 'alpha': .5}
            temp = filter.filter(all_res_,{'odor_valence':valence})

            from scipy import interpolate
            for i, y in enumerate(temp[reduce_key]):
                x = temp[xkey][i]
                f = interpolate.interp1d(x, y, fill_value='extrapolate')
                newx = np.arange(0, x[-1])
                newy = f(newx)
                temp[xkey][i] = newx
                temp[reduce_key][i] = newy

            mean_sem = reduce.new_filter_reduce(temp, filter_keys=['condition'], reduce_key=reduce_key,
                                                regularize='max')

            path, name = plot.plot_results(mean_sem, x_key=xkey, y_key= reduce_key,
                              loop_keys= composite_arg,
                               rect = (.3, .3, .6, .6),
                              colors=color, select_dict={'odor_valence':valence},
                              ax_args=ax_args_local, plot_args=line_args_mean_sem,
                              save=False,
                              path=save_path)

            plot.plot_results(mean_sem, x_key=xkey, y_key=reduce_key, error_key=reduce_key+'_sem',
                              loop_keys= composite_arg,
                              rect = (.3, .3, .6, .6),
                              colors=color, select_dict={'odor_valence': valence},
                              ax_args=ax_args_local,
                              plot_function=plt.fill_between,
                              plot_args=fill_args,
                              reuse=True,
                              path=save_path, name_str='_mean_sem')

if 'summary_line' in experiments:
    r = defaultdict(list)
    duration = 10
    before_key = reduce_key_raw + '_A'
    after_key = reduce_key_raw + '_B'
    for i, x in enumerate(all_res[reduce_key_raw]):
        all_res[before_key].append(np.mean(x[:duration]))
        all_res[after_key].append(np.mean(x[-duration:]))
    for k, v in all_res.items():
        all_res[k] = np.array(v)
    all_res_lick = reduce.new_filter_reduce(all_res, filter_keys=['condition', 'odor_valence', 'mouse'],
                                            reduce_key=before_key)
    _ = reduce.new_filter_reduce(all_res, filter_keys=['condition', 'odor_valence', 'mouse'],
                                            reduce_key=after_key)
    all_res_lick[after_key] = _[after_key]

    res_use = all_res
    valences = np.unique(all_res['odor_valence'])
    scatter_args_copy = scatter_args.copy()
    scatter_args_copy.update({'marker': '.', 'alpha': 1, 's': 10})
    ax_args_local = {'yticks': [0, 2, 5], 'ylim': [0, 5], 'xticks': [0, 2, 5], 'xlim': [0, 5],
                  'xticklabels':['ON', 'OFF', 'US'], 'yticklabels':['ON', 'OFF', 'US']}
    for valence in valences:
        color = color_dict_valence[valence]
        path, name = plot.plot_results(res_use, x_key=before_key, y_key=after_key,
                          select_dict={'odor_valence':valence},
                          rect=(.3, .3, .6, .6),
                          plot_function=plt.scatter,
                          plot_args=scatter_args_copy,
                                       ax_args = ax_args_local,
                          colors=color,
                          save=False,
                          path=save_path)
        plt.plot([0, 5], [0, 5], '--', color='gray')

        try:
            stat_res = filter.filter(res_use, {'odor_valence':valence})
            a, b = stat_res[before_key], stat_res[after_key]
            stat = wilcoxon(a, b)
            ylim = plt.gca().get_ylim()
            sig_str = plot.significance_str(x=.4, y=.7 * (ylim[-1] - ylim[0]), val= stat[-1])
            plot._easy_save(path, name)
            print('Before: {}, After: {}'.format(np.mean(a), np.mean(b)))
            print(stat)
        except:
            print('no stats')

if 'summary_mouse_line' in experiments:
    _collapse_conditions(all_res, control_condition='YFP', str=collapse_arg)
    ykey = reduce_key_raw + '_mean'
    for i, v in enumerate(all_res[reduce_key_raw]):
        all_res[ykey].append(np.mean(v[v>0]))
    all_res[ykey] = np.array(all_res[ykey])

    res_modified = reduce.new_filter_reduce(all_res, filter_keys=['condition', 'mouse', 'odor_valence'], reduce_key=ykey)
    res_modified.pop(ykey + '_std')
    res_modified.pop(ykey + '_sem')
    mean_std = reduce.new_filter_reduce(res_modified, filter_keys=['condition','odor_valence'], reduce_key = ykey)

    line_args = {'alpha': .5, 'linewidth': 0, 'marker': '.', 'markersize': 2}
    error_args = {'fmt': '.', 'capsize': 2, 'elinewidth': 1, 'markersize': 0, 'alpha': .6}
    swarm_args = {'marker': '.', 'size': 5, 'facecolors': 'none', 'alpha': .5, 'palette': ['red','black'], 'jitter': .1}

    if arg in ['lick','lick_5s']:
        ax_args = {'yticks': [0, 10, 20, 30], 'ylim': [0, 30], 'xlim': [-1, 2]}
    elif arg in ['com', 'first']:
        ax_args = {'yticks': [0, 2, 5], 'ylim': [0, 5], 'yticklabels': ['ON', 'OFF', 'US'],
                   'xlim': [-1, 2]}

    for valence in np.unique(res_modified['odor_valence']):
        path, name = plot.plot_results(res_modified, x_key='condition', y_key= ykey,
                                       select_dict={'odor_valence': valence},
                                       ax_args = ax_args,
                                       plot_function=sns.stripplot,
                                       plot_args=swarm_args,
                                       save=False,
                                       rect=(.3, .25, .6, .6),
                                       path=save_path)

        plot.plot_results(mean_std, x_key='condition', y_key= ykey, error_key=ykey + '_sem',
                          select_dict={'odor_valence':valence},
                          ax_args= ax_args,
                          plot_function= plt.errorbar,
                          plot_args= error_args,
                          path=save_path, reuse=True, save=False)

        test = filter.filter(res_modified, {'odor_valence': valence})
        ixs = test['condition'] == 'YFP'
        y_yfp = test[ykey][ixs]
        y_combined = test[ykey][np.invert(ixs)]
        rs = ranksums(y_yfp, y_combined)
        ylim = plt.gca().get_ylim()
        sig_str = plot.significance_str(x=.4, y=.7 * (ylim[-1] - ylim[0]), val=rs[-1])
        plot._easy_save(path, name)

        print('YFP: {}'.format(np.mean(y_yfp)))
        print('YFP: {}'.format(y_yfp))
        print('HALO: {}'.format(np.mean(y_combined)))
        print('HALO: {}'.format(y_combined))
        print(rs)




        # path, name = plot.plot_results(all_res, x_key=collapse_arg, y_key= reduce_key,
        #                                select_dict={'odor_valence': valence},
        #                                ax_args=ax_args,
        #                                colors = colors,
        #                                plot_function= sns.stripplot,
        #                                plot_args= swarm_args_copy,
        #                                sort=True,
        #                                fig_size=[2, 1.5],
        #                                path=save_path, reuse=False, save=False)




if 'summary_hist' in experiments:
    def _helper(real, label, bin, range, ax):
        density, bins = np.histogram(real, bins=bin, density=True, range= range)
        unity_density = density / density.sum()
        widths = bins[:-1] - bins[1:]
        ax.bar(bins[:-1], unity_density, width=widths, alpha=.5, label=label)

    duration = 500
    before_key = reduce_key_raw + '_hist_A'
    after_key = reduce_key_raw + '_hist_B'

    valences = np.unique(all_res['odor_valence'])
    for valence in valences:
        temp = filter.filter(all_res, {'odor_valence': valence})


        for i, x in enumerate(temp[reduce_key_raw]):
            temp[before_key].append(x[:duration])
            temp[after_key].append(x[-duration:])
        for k,v in temp.items():
            temp[k] = np.array(v)

        ctrl_ixs = temp['condition'] == 'YFP'
        exp_ixs = np.invert(ctrl_ixs)
        keys = [before_key, after_key]
        for k in keys:
            ctrl_data = np.concatenate(temp[k][ctrl_ixs])
            exp_data = np.concatenate(temp[k][exp_ixs])

            if arg == 'lick':
                ctrl_data = ctrl_data[ctrl_data>0]
                exp_data = exp_data[exp_data>0]

            bins = 20

            fig = plt.figure(figsize=(2, 1.5))
            ax = fig.add_axes([.2, .2, .7, .7])

            if collection and arg == 'first':
                plt.xticks([0, 2, 4], ['US', '2s', '4s'])
                range = [0, 5]
            elif arg == 'lick':
                plt.xticks([0, 5, 10])
                range = [0, 15]
                bins=15
            else:
                plt.xticks([0, 2, 5], ['ON', 'OFF', 'US'])
                range = [0, 5]

            _helper(ctrl_data, 'YFP', bins, range, ax)
            _helper(exp_data, 'INH', bins, range, ax)
            plt.legend(['YFP','INH'], fontsize=5, frameon=False)
            plt.ylabel('Count')

            if arg == 'lick':
                xlabel = 'Number of licks'
            elif arg == 'com':
                xlabel = 'First moment of licking'
            elif arg == 'first':
                xlabel = 'Time of first lick'
            else:
                raise ValueError('what')
            plt.xlabel(xlabel)

            plt.xlim([range[0]-.5, range[1]])
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')

            stat = ranksums(ctrl_data, exp_data)[-1]
            xlim = plt.xlim()
            ylim = plt.ylim()
            plot.significance_str(xlim[1] * .5, ylim[1] * .9, stat)
            ax.set_title(reduce_key_raw)

            plot._easy_save(os.path.join(save_path, reduce_key_raw + '_hist' + '_' + valence), k)


