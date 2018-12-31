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


class behaviorConfig(object):
    def __init__(self):
        #extra time (seconds) given to CS- odors after water onset for quantification.
        self.extra_csm_time = 3
        self.smoothing_window = 5
        self.smoothing_window_boolean = 9
        self.polynomial_degree = 1

def _getCurrentIndex(res, i):
    out = Cons()
    for k, v in res.items():
        setattr(out, k, v[i])
    return out

#TODO: implement find half-max last up, find half-max last down
def find_last_up(vec):
    pass

def find_last_down(vec):
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

#stitch
plot_res = defaultdict(list)
mice = np.unique(res['mouse'])
first_day = condition.training_start_day
last_day = filter.get_last_day_per_mouse(res)

for i, mouse in enumerate(mice):
    filter_dict = {'mouse': mouse, 'day': np.arange(first_day[i], last_day[i])}
    filtered_res = filter.filter(res, filter_dict)
    odor_lick_list, odor_day_lick_list = stitch_by_day(filtered_res)
    for j, lick_list in enumerate(odor_lick_list):
        plot_res['mouse'].append(mouse)
        plot_res['lick'].append(lick_list)
        plot_res['odor'].append(condition.odors[i][j])
        plot_res['trial'].append(np.arange(len(lick_list)))
for key, val in plot_res.items():
    plot_res[key] = np.array(val)

#analyze per mouse
odor_lick_list = plot_res['lick']
plot_res['lick_smoothed'] = [savgol_filter(y, config.smoothing_window, config.polynomial_degree)
                         for y in odor_lick_list]
plot_res['boolean_smoothed'] = [savgol_filter(y > 0, config.smoothing_window_boolean, config.polynomial_degree)
                                   for y in odor_lick_list]
for key, val in plot_res.items():
    plot_res[key] = np.array(val)

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

#plot
ax_args = {'yticks':[0, .5, 1], 'ylim':[-.05, 1.05], 'xticks':[0, 10, 20, 30, 40, 50], 'xlim':[0, 60]}
mice = np.unique(plot_res['mouse'])
for i, mouse in enumerate(mice):
    select_dict = {'mouse':mouse}
    plot.plot_results(plot_res, x_key='trial', y_key = 'boolean_smoothed', loop_keys = 'odor',
                      select_dict= select_dict, colors=colors, ax_args=ax_args, plot_args=plot_args, path = save_path)








