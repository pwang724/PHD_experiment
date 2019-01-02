import CONSTANTS.conditions as experimental_conditions
import os
import numpy as np
import glob
from CONSTANTS.config import Config
from collections import defaultdict
import analysis
from init.cons import Cons
import matplotlib.pyplot as plt
import filter
import plot
from scipy.signal import savgol_filter

#TODO parcellate things into functions

class behaviorConfig(object):
    def __init__(self):
        #extra time (seconds) given to CS- odors after water onset for quantification.
        self.extra_csm_time = 3
        self.smoothing_window = 5
        self.smoothing_window_boolean = 9
        self.polynomial_degree = 1

        self.halfmax_up_threshold = 50

def _getCurrentIndex(res, i):
    out = Cons()
    for k, v in res.items():
        setattr(out, k, v[i])
    return out

def half_max_up(vec):
    config = behaviorConfig()
    vec_binary = vec > config.halfmax_up_threshold
    last_ix_below_threshold = np.where(vec_binary == 0)[0][-1]
    vec_binary[:last_ix_below_threshold] = 0
    if np.any(vec_binary):
        half_max = np.where(vec_binary==1)[0][0]
    else:
        half_max = None
    return half_max

#TODO: implement find half-max last down
def half_max_down(vec):
    pass

def analyze_lick(odors, csp_odors, cons):
    '''
    data is in format TIME X DAQ_CHANNEL X TRIAL

    :param condition:
    :param cons:
    :return: licks_per_odor is sorted by the order defined in condition.odors
    '''

    def _parseLick(mat, start, end):
        mask = mat > 1
        on_off = np.diff(mask, n=1)
        n_licks = np.sum(on_off[start:end] > 0)
        return n_licks

    config = behaviorConfig()
    odor_trials = cons.ODOR_TRIALS
    lick_data = cons.DAQ_DATA[:, cons.DAQ_L, :]
    csm_odors = list(set(odors) - set(csp_odors))
    csm_ix = np.isin(odor_trials, csm_odors)

    n_licks = np.zeros(len(odor_trials))
    for i in range(len(odor_trials)):
        end = int(cons.DAQ_W_ON * cons.DAQ_SAMP)
        if csm_ix[i]:
            end += config.extra_csm_time
        n_licks[i] = _parseLick(lick_data[:, i],
                             start = int(cons.DAQ_O_ON * cons.DAQ_SAMP),
                             end = end)
    n_licks_per_odor = []
    for odor in odors:
        n_licks_per_odor.append(n_licks[odor == odor_trials])
    return n_licks, n_licks_per_odor

def stitch_by_day(res):
    '''

    :param res:
    :param odors:
    :return: list (odor) of list (days) of number of licks
    '''
    days_ix = np.argsort(res['day'])

    day_odor_list = []
    for ix in days_ix:
        day_odor_list.append(res['licks_per_odor'][ix])
    odor_day_lick_list =np.transpose(day_odor_list)
    odor_lick_list = []
    for day_list in odor_day_lick_list:
        odor_lick_list.append(np.hstack(day_list.flat))
    return odor_lick_list, odor_day_lick_list


#inputs
condition = experimental_conditions.OFC
config = behaviorConfig()

#analyze per day
data_path = os.path.join(Config.LOCAL_DATA_PATH, Config.LOCAL_DATA_TIMEPOINT_FOLDER, condition.name)
res = analysis.load_all_cons(data_path)
analysis._add_days(res)
analysis._add_time(res)

nCons = res['DAQ_L'].size
list_of_licks = []
list_of_licks_per_odor = []
for i in range(0, nCons):
    cur_cons = _getCurrentIndex(res, i)
    mouse = cur_cons.mouse
    odors = condition.odors[mouse]
    csps = condition.csp[mouse]

    n_licks, n_licks_per_odor = analyze_lick(odors, csps, cur_cons)
    list_of_licks.append(n_licks)
    list_of_licks_per_odor.append(n_licks_per_odor)
res['licks'] = np.array(list_of_licks)
res['licks_per_odor'] = np.array(list_of_licks_per_odor)

#stitch data into format: each odor condition for all days per mouse
plot_res = defaultdict(list)
mice = np.unique(res['mouse'])
first_day = condition.training_start_day
last_day = filter.get_last_day_per_mouse(res)

for i, mouse in enumerate(mice):
    filter_dict = {'mouse': mouse, 'day': np.arange(first_day[i], last_day[i])}
    filtered_res = filter.filter(res, filter_dict)
    odor_lick_list, odor_day_lick_list = stitch_by_day(filtered_res)

    csp_ix, csm_ix = 1, 1
    odors = condition.odors[i]
    csp = condition.csp[i]
    csm = list(set(odors)-set(csp))
    for j, lick_list in enumerate(odor_lick_list):
        plot_res['mouse'].append(mouse)
        plot_res['lick'].append(lick_list)
        plot_res['odor'].append(odors[j])
        plot_res['trial'].append(np.arange(len(lick_list)))
        if np.isin(odors[j], csp):
            plot_res['odor_standard'].append('CS+' + str(csp_ix))
            plot_res['odor_valence'].append('CS+')
            csp_ix+=1
        elif np.isin(odors[j], csm):
            plot_res['odor_standard'].append('CS-' + str(csp_ix))
            plot_res['odor_valence'].append('CS-')
            csm_ix+=1
        else:
            raise ValueError('odor {} is not in either CS+ or CS- odor'.format(odors[j]))
for key, val in plot_res.items():
    plot_res[key] = np.array(val)

#analysis: each odor condition for all days per mouse
odor_lick_list = plot_res['lick']
plot_res['lick_smoothed'] = [savgol_filter(y, config.smoothing_window, config.polynomial_degree)
                         for y in odor_lick_list]
plot_res['boolean_smoothed'] = [100 * savgol_filter(y > 0, config.smoothing_window_boolean, config.polynomial_degree)
                                   for y in odor_lick_list]
plot_res['half_max'] = [half_max_up(x) for x in plot_res['boolean_smoothed']]
for key, val in plot_res.items():
    plot_res[key] = np.array(val)

#summarize data
summary_res = defaultdict(list)
for i, mouse in enumerate(mice):
    filter_dict = {'mouse': mouse, 'odor': condition.csp[i]}
    cur_res = filter.filter(plot_res, filter_dict)
    summary_res['half_max'].append(np.mean(cur_res['half_max']))
    summary_res['mouse'].append(mouse)


#plot
save_path = os.path.join(Config.LOCAL_FIGURE_PATH, 'BEHAVIOR', condition.name)
colors = ['green','lime','red','maroon']
plot_args = {'marker':'o', 'markersize':1, 'alpha':.6, 'linewidth':1}
ax_args = {'yticks':[0, 10, 20, 30, 40], 'ylim':[-1, 41], 'xticks':[0, 10, 20, 30, 40, 50], 'xlim':[0, 60]}

mice = np.unique(plot_res['mouse'])
for i, mouse in enumerate(mice):
    select_dict = {'mouse':mouse}
    plot.plot_results(plot_res, x_key='trial', y_key = 'lick_smoothed', loop_keys = 'odor',
                      select_dict= select_dict, colors=colors, ax_args=ax_args, plot_args=plot_args, path = save_path)
    plot.plot_results(plot_res, x_key='trial', y_key = 'lick', loop_keys = 'odor',
                      select_dict= select_dict, colors=colors, ax_args=ax_args, plot_args=plot_args, path = save_path)

ax_args = {'yticks':[0, 50, 100], 'ylim':[-5, 105], 'xticks':[0, 10, 20, 30, 40, 50], 'xlim':[0, 60]}
mice = np.unique(plot_res['mouse'])
for i, mouse in enumerate(mice):
    select_dict = {'mouse':mouse}
    plot.plot_results(plot_res, x_key='trial', y_key = 'boolean_smoothed', loop_keys = 'odor',
                      select_dict= select_dict, colors=colors, ax_args=ax_args, plot_args=plot_args, path = save_path)

#bar plot
csp_plot_res = filter.filter_odors_per_mouse(plot_res, condition.csp)
colors = ['black','black']
select_dict = {'odor_valence':'CS+'}
ax_args = {'yticks':[0, 10, 20, 30, 40], 'ylim':[-1, 41]}
plot_args = {'marker':'o', 's':10, 'facecolors': 'none', 'alpha':.6}
plot.plot_results(plot_res, x_key='mouse', y_key = 'half_max', loop_keys='odor_standard', colors = colors,
                  select_dict= select_dict, path=save_path, plot_function= plt.scatter, plot_args=plot_args,
                  ax_args=ax_args, save = False)

plot_args = {'alpha':.6, 'fill': False}
plot.plot_results(summary_res, x_key='mouse', y_key = 'half_max', loop_keys=None,
                  path=save_path, plot_function= plt.bar, plot_args=plot_args,
                  ax_args=ax_args, save = True, reuse= True)








