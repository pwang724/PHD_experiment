import numpy as np
import os
import glob
from _CONSTANTS.config import Config
import  behavior.cristian_behavior_analysis as analysis
import filter
import reduce
import plot
import matplotlib.pyplot as plt
import matplotlib as mpl
from format import *
from collections import defaultdict
from scipy.stats import ranksums
import seaborn as sns
import behavior.behavior_config

plt.style.use('default')
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.size'] = 5
mpl.rcParams['font.family'] = 'arial'

ax_args_copy = ax_args.copy()
ax_args_copy.update({'ylim':[-5, 65], 'yticks':[0, 30, 60]})
bool_ax_args_copy = ax_args.copy()
bool_ax_args_copy.update({'ylim':[-0.05, 1.05], 'yticks':[0, .5, 1]})

class OFC_PT_Config():
    path = r'C:\Users\P\Desktop\MANUSCRIPT_DATA\Nikki Data\OFC Pretraining'
    name = 'OFC_PT'

class OFC_PT_ZERO_TRIALS_Config():
    path = r'C:\Users\P\Desktop\MANUSCRIPT_DATA\Nikki Data\OFC Pretraining\Pretraining_zero_trials'
    name = 'OFC_PT_ZERO_TRIALS'

class OFC_PT_ZERO_TRIALS_RELEASED_Config():
    path = r'C:\Users\P\Desktop\MANUSCRIPT_DATA\Nikki Data\OFC Pretraining\Pretraining_zero_trials_Laser OFF'
    name = 'OFC_PT_ZERO_TRIALS_RELEASED'

class OFC_DT_Config():
    path = r'C:\Users\P\Desktop\MANUSCRIPT_DATA\Nikki Data\OFC Discrim'
    name = 'OFC_DT'

class MPFC_PT_Config():
    path = r'C:\Users\P\Desktop\MANUSCRIPT_DATA\Nikki Data\mPFC Pretraining'
    name = 'MPFC_PT'

class MPFC_DT_Config():
    path = r'C:\Users\P\Desktop\MANUSCRIPT_DATA\Nikki Data\mPFC Discrim'
    name = 'MPFC_DT'

indices = analysis.Indices()
constants = analysis.Constants()
config = Config()

experiments = [OFC_PT_Config, OFC_DT_Config, MPFC_PT_Config, MPFC_DT_Config]
# experiments = [OFC_PT_ZERO_TRIALS_RELEASED_Config]
collapse_arg = 'OFC_DT'
plotting = [
    # 'individual_separate',
    # 'individual_together',
    # 'trials_to_criterion',
    # 'trials_per_day',
    'summary',
    # 'control',
    # 'fraction_licks_per_day',
    # 'release_of_inhibition'
]

names = ','.join([x.name for x in experiments]) + '__' + collapse_arg
save_path = os.path.join(Config.LOCAL_FIGURE_PATH, 'BEHAVIOR_CRISTIAN', names)
directories = [constants.pretraining_directory, constants.discrimination_directory]

color_dict = {'Pretraining_CS+': 'C1', 'Discrimination_CS+':'green', 'Discrimination_CS-':'red'}
res = defaultdict(list)
for experiment in experiments:
    for directory in directories:
        halo_files = sorted(glob.glob(os.path.join(experiment.path, directory, constants.halo + '*')))
        yfp_files = sorted(glob.glob(os.path.join(experiment.path, directory, constants.yfp + '*')))
        res1 = analysis.parse(halo_files, experiment=experiment, condition=constants.halo, phase = directory)
        res1['experiment'] = np.array([experiment.name] * len(res1['odor_valence']))
        res2 = analysis.parse(yfp_files, experiment=experiment, condition=constants.yfp, phase = directory)
        res2['experiment'] = np.array([experiment.name] * len(res2['odor_valence']))

        if experiment.name == 'MPFC_DT':
            res1 = filter.exclude(res1, {'mouse':['H01']})
            res2 = filter.exclude(res2, {'mouse':['H01']})
        if experiment.name == 'MPFC_PT':
            res1 = filter.exclude(res1, {'mouse': ['Y01']})
            res2 = filter.exclude(res2, {'mouse': ['Y01']})
        reduce.chain_defaultdicts(res, res1)
        reduce.chain_defaultdicts(res, res2)

if collapse_arg:
    res1 = filter.filter(res, filter_dict={'experiment': collapse_arg, 'condition':'H'})
    res2 = filter.filter(res, filter_dict={'condition': 'Y'})
    res = reduce.chain_defaultdicts(res1, res2, copy_dict=True)

analysis.analyze(res)
analysis.shift_discrimination_index(res)
filter.assign_composite(res, ['phase', 'odor_valence'])
res = filter.filter(res, {'phase_odor_valence': ['Pretraining_CS+', 'Discrimination_CS+', 'Discrimination_CS-']})

if 'release_of_inhibition' in plotting:
    trace_args_copy = trace_args.copy()
    trace_args_copy.update({'alpha': .5, 'linewidth': .75})
    line_args_copy = line_args.copy()
    line_args.update({'marker':',','markersize':0, 'linewidth':.5})

    y_key = 'bin_ant_23_smooth'
    y_key_bool = 'bin_ant_23_boolean'

    phase_odor_valence = np.unique(res['phase_odor_valence'])
    conditions = np.unique(res['condition'])

    ax_args_cur = ax_args_copy.copy()
    ax_args_bool_cur = bool_ax_args_copy.copy()
    for phase in phase_odor_valence:
        if 'Pretraining' in phase:
            ax_args_cur.update({'xlim': [-50, 1050], 'xticks': [0,500,1000]})
            ax_args_bool_cur.update({'xlim': [-50, 1050], 'xticks': [0,500,1000]})
        else:
            ax_args_cur.update({'xlim': [-10, 410], 'xticks': [0, 200, 400]})
            ax_args_bool_cur.update({'xlim': [-10, 410], 'xticks': [0, 200, 400]})

        for condition in conditions:
            if condition == 'H':
                color = color_dict[phase]

            fp = plot.plot_results(res, x_key='trials', y_key=y_key_bool,
                              select_dict={'phase_odor_valence': phase, 'condition':'H'},
                              ax_args=ax_args_bool_cur, plot_args=trace_args_copy,
                              colors=color,
                              save=False,
                              path=save_path)


            c = behavior.behavior_config.behaviorConfig()
            y = c.fully_learned_threshold_up / 100.
            plt.plot(plt.xlim(), [y, y], '--', color='gray', linewidth=.5)

            plot._easy_save(fp[0],fp[1],pdf=True)


            fp = plot.plot_results(res, x_key='trials', y_key=y_key,
                              select_dict={'phase_odor_valence': phase, 'condition':'H'},
                              ax_args=ax_args_cur, plot_args=trace_args_copy,
                              colors=color,
                              save=False,
                              path=save_path)

            c = behavior.behavior_config.behaviorConfig()
            y = c.fully_learned_threshold_up / 100.
            plt.plot(plt.xlim(), [y, y], '--', color='gray', linewidth=.5)

            plot._easy_save(fp[0], fp[1], pdf=True)


    line_args_copy = line_args.copy()
    line_args_copy.update({'linestyle':'--', 'linewidth':.5,'markersize':1.5})
    ax_args_cur = ax_args.copy()
    ax_args_cur.update({'ylim':[-25, 300], 'yticks':[0, 100, 200, 300], 'xticks':[1, 3, 5, 7, 9]})

    phase_odor_valence = np.unique(res['phase_odor_valence'])
    y_key = 'trials_per_day'
    for phase in phase_odor_valence:
        plot.plot_results(res, x_key='days', y_key=y_key,
                          select_dict={'phase_odor_valence':phase, 'condition':'H'},
                          colors= color_dict[phase], plot_args=line_args_copy, ax_args=ax_args_cur,
                          fig_size=[2, 1.5],
                          path=save_path, save=True)


if 'trials_to_criterion' in plotting:
    scatter_args_copy = scatter_args.copy()
    scatter_args_copy.update({'marker': '.', 'alpha': .5, 's': 10})
    error_args_copy = error_args.copy()
    error_args_copy.update({'elinewidth': .5, 'markeredgewidth': .5, 'markersize': 0})
    ax_args_cur = ax_args.copy()

    keyword = 'bin_ant_23_trials_to_criterion'
    # keyword = 'bin_ant_23_trials_to_half_max'
    res_ = res.copy()
    if collapse_arg == 'OFC_PT':
        res_ = filter.exclude(res_, {'mouse':['H01','H02','H04'],'experiment':'OFC_PT'})
    filter.assign_composite(res_, ['odor_valence','condition'])

    phase_odor_valence = np.unique(res_['phase_odor_valence'])
    summary_res = reduce.new_filter_reduce(res_, filter_keys=['condition','phase_odor_valence'], reduce_key=keyword)
    for phase in np.unique(res_['phase_odor_valence']):
    # for phase in phase_odor_valence:
        if 'Pretraining' in phase:
            ax_args_cur.update({'xlim':[-1, 1],'ylim':[-20, 600], 'yticks':[0, 200, 400, 600]})
        else:
            ax_args_cur.update({'xlim': [-1, 1], 'ylim': [-10, 225], 'yticks': [0, 100, 200]})

        swarm_args_copy = swarm_args.copy()
        swarm_args_copy.update({'palette':[color_dict[phase],'black'], 'size':5})

        path, name = plot.plot_results(res_, x_key='odor_valence_condition', y_key= keyword,
                          select_dict={'phase_odor_valence':phase},
                          ax_args=ax_args_cur,
                          plot_function= sns.stripplot,
                            plot_args= swarm_args_copy,
                            colors = [color_dict[phase],'black'],
                          fig_size=[2, 1.5],
                           path=save_path, reuse=False, save=False)

        plot.plot_results(summary_res, x_key='odor_valence_condition', y_key=keyword, error_key=keyword + '_sem',
                      select_dict={'phase_odor_valence': phase},
                      ax_args=ax_args,
                      plot_function=plt.errorbar,
                      plot_args=error_args,
                      fig_size=[2, 1.5],
                      path=save_path, reuse=True, save=False)

        plt.xlim(-1, 2)

        test = filter.filter(res_, {'phase_odor_valence': phase})
        ixs = test['condition'] == 'Y'
        x = test[keyword][ixs]
        y = test[keyword][np.invert(ixs)]
        rs = ranksums(x, y)[-1]
        ylim = plt.gca().get_ylim()
        sig_str = plot.significance_str(x=.4, y= .7 * (ylim[-1] - ylim[0]), val= rs)
        plot._easy_save(path, name, pdf=True)

    print(summary_res[keyword])
    print(summary_res['odor_valence_condition'])
    print(summary_res['phase_odor_valence'])

if 'trials_per_day' in plotting:
    line_args_copy = line_args.copy()
    line_args_copy.update({'linestyle':'--', 'linewidth':.5,'markersize':1.5})
    ax_args_cur = ax_args.copy()
    ax_args_cur.update({'ylim':[-25, 300], 'yticks':[0, 100, 200, 300], 'xticks':[1, 3, 5, 7, 9]})

    phase_odor_valence = np.unique(res['phase_odor_valence'])
    y_key = 'trials_per_day'
    for phase in phase_odor_valence:
        plot.plot_results(res, x_key='days', y_key=y_key,
                          select_dict={'phase_odor_valence':phase, 'condition':'H'},
                          colors= color_dict[phase], plot_args=line_args_copy, ax_args=ax_args_cur,
                          fig_size=[2, 1.5],
                          path=save_path, reuse=False, save=False)

        summary = reduce.new_filter_reduce(res, filter_keys=['phase_odor_valence', 'condition'], reduce_key=y_key,
                                           regularize='max')
        plot.plot_results(summary, x_key='days', y_key=y_key,
                          select_dict={'phase_odor_valence': phase, 'condition': 'Y'},
                          ax_args=ax_args_copy,
                          plot_args=line_args,
                          colors='black', reuse=True, save=False,
                          path=save_path)
        plot.plot_results(summary, x_key='days', y_key=y_key, error_key=y_key + '_sem',
                          select_dict={'phase_odor_valence': phase, 'condition': 'Y'},
                          ax_args=ax_args_copy,
                          plot_function=plt.fill_between, plot_args=fill_args,
                          colors='black', reuse=True, save=True,
                          path=save_path)

if 'fraction_licks_per_day' in plotting:
    line_args_copy = line_args.copy()
    line_args_copy.update({'linestyle': '--', 'linewidth': .5, 'markersize': 1.5})
    ax_args_cur = ax_args.copy()
    ax_args_cur.update({'ylim': [-.05, 1.05], 'yticks': [0, .5, 1]})

    phase_odor_valence = np.unique(res['phase_odor_valence'])
    for phase in phase_odor_valence:
        plot.plot_results(res, x_key='days', y_key='performance', loop_keys='condition',
                          select_dict={'phase_odor_valence':phase},
                          colors= ['red','black'], plot_args=line_args_copy, ax_args=ax_args_cur,
                          fig_size=[2, 1.5],
                          path=save_path, reuse=False, save=True)

#summary
if 'summary' in plotting:
    trace_args_copy = trace_args.copy()
    trace_args_copy.update({'alpha': .5, 'linewidth': .75})
    line_args_copy = line_args.copy()
    line_args.update({'marker':',','markersize':0, 'linewidth':.5})

    y_key = 'bin_ant_23_smooth'
    y_key_bool = 'bin_ant_23_boolean'

    phase_odor_valence = np.unique(res['phase_odor_valence'])
    conditions = np.unique(res['condition'])

    ax_args_cur = ax_args_copy.copy()
    ax_args_bool_cur = bool_ax_args_copy.copy()
    for phase in phase_odor_valence:
        print(phase)
        if 'Pretraining' in phase:
            ax_args_cur.update({'xlim': [-50, 1050], 'xticks': [0,500,1000]})
            ax_args_bool_cur.update({'xlim': [-50, 1050], 'xticks': [0,500,1000]})
        else:
            ax_args_cur.update({'xlim': [-10, 410], 'xticks': [0, 200, 400]})
            ax_args_bool_cur.update({'xlim': [-10, 410], 'xticks': [0, 200, 400]})

        for condition in conditions:
            if condition == 'H':
                color = color_dict[phase]
            # else:
            #     color = 'black'
            # plot.plot_results(res, x_key='trials', y_key=y_key,
            #                   select_dict={'phase_odor_valence':phase, 'condition': condition},
            #                    ax_args=ax_args_cur, plot_args=trace_args_copy,
            #                    colors= color,
            #                    path=save_path, name_str= 'indv')
            # plot.plot_results(res, x_key='trials', y_key=y_key_bool,
            #                   select_dict={'phase_odor_valence': phase, 'condition': condition},
            #                    ax_args=ax_args_bool_cur, plot_args=trace_args_copy,
            #                    colors= color,
            #                    path=save_path, name_str= 'indv')
            # #
            # plot.plot_results(res, x_key='trials', y_key=y_key_bool, loop_keys='condition',
            #                   select_dict={'phase_odor_valence': phase},
            #                   ax_args=ax_args_bool_cur, plot_args=trace_args_copy,
            #                   colors=[color, 'black'],
            #                   path=save_path)
            #
            # plot.plot_results(res, x_key='trials', y_key=y_key, loop_keys='condition',
            #                   select_dict={'phase_odor_valence': phase},
            #                   ax_args=ax_args_cur, plot_args=trace_args_copy,
            #                   colors=[color, 'black'],
            #                   path=save_path)

            #fill bool plot
            plot.plot_results(res, x_key='trials', y_key=y_key_bool,
                              select_dict={'phase_odor_valence': phase, 'condition':'H'},
                              ax_args=ax_args_bool_cur, plot_args=trace_args_copy,
                              colors=color,
                              save=False,
                              path=save_path)

            if phase == 'Discrimination_CS+' or phase == 'Pretraining_CS+':
                c = behavior.behavior_config.behaviorConfig()
                y = c.fully_learned_threshold_up / 100.
                plt.plot(plt.xlim(), [y, y], '--', color='gray', linewidth=.5)

            if phase == 'Discrimination_CS-':
                y = c.fully_learned_threshold_down
                plt.plot(plt.xlim(), [y, y], '--', color='gray', linewidth=.5)

            summary = reduce.new_filter_reduce(res, filter_keys=['phase_odor_valence', 'condition'], reduce_key=y_key_bool,
                                               regularize='max')
            plot.plot_results(summary, x_key='trials', y_key=y_key_bool,
                              select_dict={'phase_odor_valence': phase, 'condition':'Y'},
                              ax_args=ax_args_bool_cur,
                              plot_args=line_args,
                              colors= 'black', reuse=True, save=False,
                              path=save_path)
            plot.plot_results(summary, x_key='trials', y_key=y_key_bool, error_key= y_key_bool + '_sem',
                              select_dict={'phase_odor_valence': phase, 'condition':'Y'},
                              ax_args=ax_args_bool_cur,
                              plot_function=plt.fill_between, plot_args=fill_args,
                              colors= 'black', reuse=True,
                              path=save_path)

            plot.plot_results(res, x_key='trials', y_key=y_key,
                              select_dict={'phase_odor_valence': phase, 'condition':'H'},
                              ax_args=ax_args_cur, plot_args=trace_args_copy,
                              colors=color,
                              save=False,
                              path=save_path)

            summary = reduce.new_filter_reduce(res, filter_keys=['phase_odor_valence', 'condition'], reduce_key=y_key,
                                               regularize='max')
            plot.plot_results(summary, x_key='trials', y_key=y_key,
                              select_dict={'phase_odor_valence': phase, 'condition':'Y'},
                              ax_args=ax_args_cur,
                              plot_args=line_args,
                              colors= 'black', reuse=True, save=False,
                              path=save_path)
            plot.plot_results(summary, x_key='trials', y_key=y_key, error_key= y_key + '_sem',
                              select_dict={'phase_odor_valence': phase, 'condition':'Y'},
                              ax_args=ax_args_cur,
                              plot_function=plt.fill_between, plot_args=fill_args,
                              colors= 'black', reuse=True,
                              path=save_path)

#summary
if 'control' in plotting:
    trace_args_copy = trace_args.copy()
    trace_args_copy.update({'alpha': .5, 'linewidth': .75})
    line_args_copy = line_args.copy()
    line_args.update({'marker':',','markersize':0, 'linewidth':.5})

    y_key = 'bin_ant_23_smooth'
    y_key_bool = 'bin_ant_23_boolean'

    phase_odor_valence = np.unique(res['phase_odor_valence'])
    conditions = np.unique(res['condition'])

    ax_args_cur = ax_args_copy.copy()
    ax_args_bool_cur = bool_ax_args_copy.copy()
    for phase in phase_odor_valence:
        if 'Pretraining' in phase:
            ax_args_cur.update({'xlim': [-50, 1050], 'xticks': [0,500,1000]})
            ax_args_bool_cur.update({'xlim': [-50, 1050], 'xticks': [0,500,1000]})
        else:
            ax_args_cur.update({'xlim': [-10, 410], 'xticks': [0, 200, 400]})
            ax_args_bool_cur.update({'xlim': [-10, 410], 'xticks': [0, 200, 400]})

        for condition in conditions:
            if condition == 'H':
                color = color_dict[phase]

            #fill bool plot
            plot.plot_results(res, x_key='trials', y_key=y_key_bool, loop_keys='condition',
                              select_dict={'phase_odor_valence': phase, 'condition':'Y'},
                              ax_args=ax_args_bool_cur, plot_args=trace_args_copy,
                              colors=['black']*50,
                              path=save_path)

            plot.plot_results(res, x_key='trials', y_key=y_key, loop_keys='condition',
                              select_dict={'phase_odor_valence': phase, 'condition':'Y'},
                              ax_args=ax_args_cur, plot_args=trace_args_copy,
                              colors=['black']*50,
                              path=save_path)

#individual
def _add_session_lines(res_mouse):
    session = res_mouse['session']
    ax = plt.gca()
    ylims = ax.get_ylim()
    if len(session) == 1:
        unique_sessions, session_ix = np.unique(session[0], return_index=True)
    else:
        unique_sessions, session_ix = np.unique(session[0], return_index=True)
        unique_sessions2, session_ix2 = np.unique(session[1], return_index=True)
        total = len(session[0])
        last_session = session[0][-1]
        unique_sessions2 += last_session
        session_ix2 += total
        unique_sessions = np.concatenate((unique_sessions, unique_sessions2))
        session_ix = np.concatenate((session_ix, session_ix2))
    for session, x in zip(unique_sessions, session_ix):
        plt.plot([x, x], ylims, linestyle='--', linewidth=.5, color='gray')
        plt.text(x, ylims[1] + .05, session.astype(int))

if 'individual_separate' in plotting:
    trace_args_copy = trace_args.copy()
    trace_args_copy.update({'alpha': 1, 'linewidth': 1})
    y_key = 'bin_ant_23_smooth'
    y_key_bool = 'bin_ant_23_boolean'

    phases = ['Pretraining', 'Discrimination']
    colors = {'Pretraining': ['orange'], 'Discrimination':['green','red']}
    for phase in phases:
        color = colors[phase]
        res_ = filter.filter(res, {'phase': phase})

        mice = np.unique(res_['mouse'])
        for mouse in mice:
            if phase == 'Pretraining':
                res_ = filter.filter(res_, {'odor_valence':'CS+'})
            res_mouse = filter.filter(res_, {'mouse':mouse})
            res_mouse_valence = filter.filter(res_mouse, {'odor_valence': 'CS+'})
            path, name = plot.plot_results(res_, x_key='trials', y_key=y_key,
                                           select_dict={'mouse':mouse, 'phase':phase}, loop_keys='odor_valence',
                                          ax_args=ax_args_copy, plot_args = trace_args_copy,
                                          colors = color,
                                          path=save_path, save=False)
            _add_session_lines(res_mouse_valence)
            plot._easy_save(path, name, pdf=True)

            path, name = plot.plot_results(res_, x_key='trials', y_key=y_key_bool,
                                           select_dict={'mouse':mouse, 'phase':phase}, loop_keys='odor_valence',
                              ax_args=bool_ax_args_copy, plot_args = trace_args_copy,
                              colors = color,
                              path=save_path, save=False)
            _add_session_lines(res_mouse_valence)
            plot._easy_save(path, name, pdf=True)

if 'individual_together' in plotting:
    y_key = 'bin_ant_23_smooth'
    y_key_bool = 'bin_ant_23_boolean'

    mice = np.unique(res['mouse'])
    trace_args_copy = trace_args.copy()
    trace_args_copy.update({'alpha': 1, 'linewidth': 1})
    for mouse in mice:
        filtered_res = filter.filter(res, {'mouse':mouse})
        if len(np.unique(filtered_res['phase_odor_valence'])) == 1:
            colors = ['orange']
        else:
            colors = ['green','red','orange']

        path, name = plot.plot_results(filtered_res, x_key='together_trials', y_key=y_key, loop_keys='phase_odor_valence',
                          select_dict={'mouse': mouse},
                          ax_args=ax_args_copy, plot_args=trace_args_copy,
                          colors=colors,
                          path=save_path, sort=False, save=False)
        res_mouse_valence = filter.filter(filtered_res, {'odor_valence': 'CS+'})
        _add_session_lines(res_mouse_valence)
        plot._easy_save(path, name, pdf=True)














