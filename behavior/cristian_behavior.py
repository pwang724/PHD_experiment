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
from psth.format import *
from collections import defaultdict

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'arial'
ax_args_copy = ax_args.copy()
ax_args_copy.update({'ylim':[-5, 85], 'yticks':[0, 40, 80]})
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

experiment = OFC_PT_ZERO_TRIALS_RELEASED_Config
save_path = os.path.join(Config.LOCAL_FIGURE_PATH, 'BEHAVIOR_CRISTIAN', experiment.name)
directories = [constants.pretraining_directory, constants.discrimination_directory]
res = defaultdict(list)
for directory in directories:
    halo_files = sorted(glob.glob(os.path.join(experiment.path, directory, constants.halo + '*')))
    yfp_files = sorted(glob.glob(os.path.join(experiment.path, directory, constants.yfp + '*')))
    res1 = analysis.parse(halo_files, experiment=experiment, condition=constants.halo, phase = directory)
    res2 = analysis.parse(yfp_files, experiment=experiment, condition=constants.yfp, phase = directory)
    reduce.chain_defaultdicts(res, res1)
    reduce.chain_defaultdicts(res, res2)
analysis.analyze(res)
analysis.shift_discrimination_index(res)
filter.assign_composite(res, ['phase', 'odor_valence'])
res = filter.filter(res, {'phase_odor_valence': ['Pretraining_CS+', 'Discrimination_CS+', 'Discrimination_CS-']})
plotting = [
    # 'individual_separate',
    # 'individual_together',
    # 'trials_to_criterion',
    # 'trials_per_day'
    'summary',
]

if 'trials_to_criterion' in plotting:
    # keyword = 'bin_ant_23_trials_to_criterion'
    keyword = 'bin_ant_23_trials_to_half_max'
    res_ = res.copy()
    if experiment.name == 'OFC_PT':
        res_ = filter.filter(res_, {'mouse':['H03','H05','Y01','Y02','Y03','Y04','Y05','Y06']})
    elif experiment.name == 'MPFC_DT':
        res_ = filter.filter(res_, {'mouse': ['H02','H05', 'H07','H08', 'Y10', 'Y07', 'Y08', 'Y09']})
    else:
        res_ = res_

    phase_odor_valence = np.unique(res_['phase_odor_valence'])
    summary_res = reduce.new_filter_reduce(res_, filter_keys=['condition','phase_odor_valence'], reduce_key=keyword)
    for phase in phase_odor_valence:
        scatter_args_copy = scatter_args.copy()
        scatter_args_copy.update({'marker':'.','facecolors': 'k', 'alpha': .5, 's': 10})
        error_args_copy = error_args.copy()
        error_args_copy.update({'elinewidth':.5, 'markeredgewidth':.5,'markersize':0})
        ax_args_cur = ax_args.copy()
        if 'Pretraining' in phase:
            ax_args_cur.update({'xlim':[-1, 2],'ylim':[-20, 600], 'yticks':[0, 200, 400, 600]})
        else:
            ax_args_cur.update({'xlim': [-1, 2], 'ylim': [-10, 225], 'yticks': [0, 100, 200]})

        if experiment.name == 'FC_PT':
            reuse_arg = False
        else:
            plot.plot_results(summary_res,
                              x_key='condition', y_key=keyword, select_dict={'phase_odor_valence':phase},
                              error_key=keyword + '_sem',
                              plot_function=plt.errorbar, plot_args=error_args_copy, ax_args=ax_args_cur,
                              path=save_path,
                              save=False, reuse=False, legend=False)
            reuse_arg = True

        plot.plot_results(res_, x_key='condition', y_key= keyword, select_dict={'phase_odor_valence':phase},
                          ax_args=ax_args_cur,
                          plot_function= plt.scatter,
                          plot_args=scatter_args_copy,
                            colors= 'black', legend=False, fig_size=[2, 1.5],
                           path=save_path, reuse=reuse_arg, save=True)

if 'trials_per_day' in plotting:
    line_args_copy = line_args.copy()
    line_args_copy.update({'linestyle':'--', 'linewidth':.5,'markersize':1.5})
    ax_args_cur = ax_args.copy()
    ax_args_cur.update({'ylim':[-25, 300], 'yticks':[0, 100, 200, 300], 'xticks':[1, 3, 5, 7, 9]})

    phase_odor_valence = np.unique(res['phase_odor_valence'])
    for phase in phase_odor_valence:
        plot.plot_results(res, x_key='days', y_key='trials_per_day', loop_keys='condition',
                          select_dict={'phase_odor_valence':phase},
                          colors= ['red','black'], plot_args=line_args_copy, ax_args=ax_args_cur,
                          fig_size=[2, 1.5],
                          path=save_path, reuse=False, save=True)

#summary
if 'summary' in plotting:
    trace_args_copy = trace_args.copy()
    trace_args_copy.update({'alpha': .5, 'linewidth': .75})
    y_key = 'bin_ant_23_smooth'
    y_key_bool = 'bin_ant_23_boolean'

    phase_odor_valence = np.unique(res['phase_odor_valence'])
    for phase in phase_odor_valence:
        plot.plot_results(res, x_key='trials', y_key=y_key, loop_keys='condition',
                          select_dict={'phase_odor_valence':phase},
                           ax_args=ax_args_copy, plot_args=trace_args_copy,
                           colors= ['red','black'],
                           path=save_path)
        plot.plot_results(res, x_key='trials', y_key=y_key_bool, loop_keys='condition',
                          select_dict={'phase_odor_valence': phase},
                                       ax_args=bool_ax_args_copy, plot_args=trace_args_copy,
                                       colors= ['red','black'],
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














