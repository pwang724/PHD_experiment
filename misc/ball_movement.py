import filter
import reduce
import behavior.behavior_analysis
from _CONSTANTS import conditions as experimental_conditions
from _CONSTANTS.config import Config
import os
import analysis
from collections import defaultdict
import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from scipy.stats import ranksums
import plot
from format import *


def _convert(res, condition):
    ball_ix = 2
    lick_ix = res['DAQ_L'][0]
    new_res = defaultdict(list)
    toConvert = ['day', 'mouse']
    res_odorTrials = res['ODOR_TRIALS']
    res_data = res['DAQ_DATA']
    for i, odorTrials in enumerate(res_odorTrials):
        mouse = res['mouse'][i]

        if hasattr(condition, 'csp'):
            #standard
            relevant_odors = condition.odors[mouse]
        elif hasattr(condition, 'dt_csp'):
            #composite
            relevant_odors = condition.dt_odors[mouse] + condition.pt_odors[mouse]
        else:
            raise ValueError('cannot find odors')

        for j, odor in enumerate(odorTrials):
            if odor in relevant_odors:
                on = int((res['DAQ_O_ON'][i]) * res['DAQ_SAMP'][i])
                off = int((res['DAQ_O_OFF'][i]) * res['DAQ_SAMP'][i])
                end = int(res['DAQ_W_ON'][i] * res['DAQ_SAMP'][i])
                lick_data = res_data[i][:, lick_ix,j]
                ball_data = res_data[i][:, ball_ix,j]


                new_res['lick_data'].append(lick_data)
                new_res['ball_data'].append(ball_data)
                new_res['on'].append(on)
                new_res['off'].append(off)
                new_res['end'].append(end)
                new_res['odor'].append(odor)
                new_res['sampling_rate'].append(res['DAQ_SAMP'][i])
                new_res['ix'].append(j)
                for names in toConvert:
                    new_res[names].append(res[names][i])
    for key, val in new_res.items():
        new_res[key] = np.array(val)
    return new_res

def _filter(res):
    out = defaultdict(list)
    for mouse in np.unique(res['mouse']):
        temp = filter.filter(res, {'mouse': mouse})
        data = temp['ball_data'].flatten()
        max, min = np.max(data), np.min(data)
        if (max - min) > 4:
            reduce.chain_defaultdicts(out, temp)
    return out

def _angle(res):
    window = 11 #really 11/50 which is the same samp period, so filtering is in 200 ms windows
    data = res['ball_data'].flatten()
    max, min = np.max(data), np.min(data)
    conversion = lambda x: 360 * (x - min) / (max - min)

    for i, data in enumerate(res['ball_data']):
        angle = conversion(data)
        diff = np.diff(angle)
        positive_transition = np.where(diff > 300)[0]
        negative_transition = np.where(diff < -300)[0]
        for transition in positive_transition:
            angle[transition+1:] = angle[transition+1:] - 360
        for transition in negative_transition:
            angle[transition+1:] = angle[transition+1:] + 360

        filtered_angle = savgol_filter(angle, window_length= window, polyorder=0)
        filtered_angle -= np.min(filtered_angle)
        velocity = np.diff(filtered_angle) * res['sampling_rate'][i]
        velocity = np.append(velocity, velocity[-1])

        res['angle'].append(filtered_angle)
        res['velocity'].append(velocity)
        res['trial'].append(np.arange(velocity.size))
    for key, val in res.items():
        res[key] = np.array(val)

def _hist(res, save_path):
    ## distribution
    def histogram(real, label, bin, range, ax):
        density, bins = np.histogram(real, bins=bin, density=True, range= range)
        unity_density = density / density.sum()
        widths = bins[:-1] - bins[1:]
        ax.bar(bins[1:], unity_density, width=widths, alpha=.5, label=label)

    for mouse in np.unique(res['mouse']):
        pt_csp = filter.filter(res, {'mouse':mouse, 'odor_valence': 'PT CS+'})
        csp = filter.filter(res, {'mouse':mouse})
        csm = filter.filter(res, {'mouse':mouse})

        data = pt_csp['velocity']
        start = pt_csp['on'][0]
        end = pt_csp['end'][0]
        data_before = data[:,:start].flatten()
        data_during = data[:,start:end].flatten()
        data_after = data[:,end:].flatten()

        bins = 50
        range = [-70, 70]
        fig = plt.figure(figsize=(2, 1.5))
        ax = fig.add_axes([0.2, 0.2, 0.7, 0.7])
        histogram(data_before, 'before', bin = bins, range= range, ax=ax)
        histogram(data_during, 'during', bin = bins, range= range, ax=ax)
        plt.xlim([range[0] - 0.5, range[1] + .5])
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        rs = ranksums(data_before, data_during)[-1]
        xlim = plt.xlim()
        ylim = plt.ylim()
        x = xlim[0] + .7 * (xlim[1] - xlim[0])
        y = ylim[0] + .7 * (ylim[1] - ylim[0])
        plot.significance_str(x, y, rs)
        name = 'before_during_mouse_{}'.format(mouse)
        plot._easy_save(save_path, name=name)

        fig = plt.figure(figsize=(2, 1.5))
        ax = fig.add_axes([0.2, 0.2, 0.7, 0.7])
        histogram(data_during, 'before', bin = bins, range= range, ax=ax)
        histogram(data_after, 'during', bin = bins, range= range, ax=ax)
        plt.xlim([range[0] - 0.5, range[1] + .5])
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        rs = ranksums(data_during, data_after)[-1]
        xlim = plt.xlim()
        ylim = plt.ylim()
        x = xlim[0] + .7 * (xlim[1] - xlim[0])
        y = ylim[0] + .7 * (ylim[1] - ylim[0])
        plot.significance_str(x, y, rs)
        name = 'during_after_mouse_{}'.format(mouse)
        plot._easy_save(save_path, name=name)

def _average_velocity(res, save_path):
    xkey = 'trial'
    ykey = 'velocity'
    error_key = ykey + '_sem'
    color_dict = {'CS+':'green','CS-':'red','PT CS+':'C1'}

    valences = [['CS+','CS-', 'PT CS+'],['CS+','CS-'], ['PT CS+']]
    for valence in valences:
        colors = [color_dict[x] for x in valence]
        temp = filter.filter(res, {'odor_valence': valence})
        start = temp['on'][0]
        off = temp['off'][0]
        end = temp['end'][0]
        ax_args = {'xticks': [start, off, end], 'xticklabels':['ON','OFF','US'], 'ylim':[-5, 50]}

        mean_res = reduce.new_filter_reduce(temp, filter_keys=['odor_valence'], reduce_key=ykey)
        for i, v in enumerate(mean_res[ykey]):
            v_ =savgol_filter(v, window_length=21, polyorder=0)
            mean_res[ykey][i] = v_
        for i, v in enumerate(mean_res[error_key]):
            v_ =savgol_filter(v, window_length=21, polyorder=0)
            mean_res[error_key][i] = v_
        plot.plot_results(mean_res, x_key=xkey, y_key=ykey, loop_keys= 'odor_valence',
                          select_dict={'odor_valence': valence},
                          error_key=error_key,
                          plot_function=plt.fill_between,
                          colors=colors,
                          plot_args=fill_args,
                          ax_args=ax_args,
                          path=save_path)

def _example_velocity(res, save_path):
    xkey = 'trial'
    ykey = 'velocity'

    line_args = {'alpha': .5, 'linewidth': .25, 'marker': 'o', 'markersize': 0}
    mouse = 0
    odor = 'PT CS+'
    temp = filter.filter(res, {'odor_valence': odor, 'mouse': mouse})
    start = temp['on'][0]
    off = temp['off'][0]
    end = temp['end'][0]
    ax_args = {'xticks': [start, off, end], 'xticklabels': ['ON', 'OFF', 'US'], 'ylim': [-5, 100]}

    for i, v in enumerate(temp[ykey]):
        v_ = savgol_filter(v, window_length=41, polyorder=0)
        temp[ykey][i] = v_

    plot.plot_results(temp, x_key=xkey, y_key=ykey, loop_keys= ['day','ix'],
                      select_dict={'odor_valence': odor, 'mouse': mouse},
                      colors = ['black'] * 200,
                      plot_args = line_args,
                      ax_args=ax_args,
                      legend=False,
                      path=save_path)

        # mean_res = reduce.new_filter_reduce(temp, filter_keys='mouse', reduce_key= ykey)
        # for i, v in enumerate(mean_res[ykey]):
        #     v_ =savgol_filter(v, window_length=21, polyorder=0)
        #     mean_res[ykey][i] = v_
        # for i, v in enumerate(mean_res[error_key]):
        #     v_ =savgol_filter(v, window_length=21, polyorder=0)
        #     mean_res[error_key][i] = v_
        #
        # for mouse in np.unique(mean_res['mouse']):
        #     plot.plot_results(mean_res, x_key='trial',y_key=ykey, loop_keys='mouse',
        #                       select_dict={'mouse':mouse, 'odor_valence':valence},
        #                       error_key= error_key,
        #                       plot_function=plt.fill_between,
        #                       colors = 'black',
        #                       plot_args=fill_args,
        #                       ax_args = ax_args,
        #                       path=save_path)







condition = experimental_conditions.BEHAVIOR_MPFC_YFP_DISCRIMINATION
data_path = os.path.join(Config.LOCAL_DATA_PATH, Config.LOCAL_DATA_BEHAVIOR_FOLDER, condition.name)
save_path = os.path.join(Config.LOCAL_DATA_PATH, Config.LOCAL_FIGURE_PATH, 'MISC', 'ball_movement', condition.name)

res = analysis.load_all_cons(data_path)
analysis.add_indices(res)
res = _convert(res, condition)
res = _filter(res)
behavior.behavior_analysis.add_odor_value(res, condition)
print(np.unique(res['mouse']))

_angle(res)
# _average_velocity(res, save_path)
# _hist(res, save_path)
_example_velocity(res, save_path)



# print(res.keys())
# print(res['mouse'])
# print(res['day'])

