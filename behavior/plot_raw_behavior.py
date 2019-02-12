import os
import filter
from _CONSTANTS import conditions as experimental_conditions
from _CONSTANTS.config import Config
import behavior.behavior_analysis
import plot
import matplotlib.pyplot as plt
import numpy as np
import analysis
from matplotlib.colors import LinearSegmentedColormap

condition = experimental_conditions.OFC_JAWS
mouse = 0

data_path = os.path.join(Config.LOCAL_DATA_PATH, Config.LOCAL_DATA_TIMEPOINT_FOLDER, condition.name)
save_path = os.path.join(Config.LOCAL_FIGURE_PATH, 'OTHER', 'BEHAVIOR',  condition.name, 'RAW_LICKS')

res = analysis.load_all_cons(data_path)
analysis.add_indices(res)
analysis.add_time(res)
lick_res = behavior.behavior_analysis.convert(res, condition, includeRaw=True)
analysis.add_odor_value(lick_res, condition)
mouse_res = filter.filter(lick_res, filter_dict={'mouse': mouse})

mouse_cons = filter.filter(res, filter_dict={'mouse': mouse})
sample_rate = mouse_cons['DAQ_SAMP'][0]
odor_on = mouse_cons['DAQ_O_ON'][0] * sample_rate
odor_off = mouse_cons['DAQ_O_OFF'][1] * sample_rate
water_on = mouse_cons['DAQ_W_ON'][2] * sample_rate


odors_standard, odor_ix = np.unique(mouse_res['odor_standard'], return_index=True)
odor_names = mouse_res['odor'][odor_ix]

for i, odor_standard in enumerate(odors_standard):
    odor_res = filter.filter(mouse_res, filter_dict={'odor_standard': odor_standard})


    #inputs
    odor_name = odor_names[i]
    odor_colors = ['green', 'red']
    data = odor_res['lick_raw_data'] > .1
    trials_per_day = np.unique(odor_res['day'], return_counts=True)[1]
    cumulative_trials = np.cumsum(trials_per_day)
    cm = LinearSegmentedColormap.from_list('black_and_white', [(1, 1, 1), (0, 0, 0)], N=2)

    #plotting
    fig = plt.figure(figsize=(1.5, 2))
    rect = [.2, .25, .7, .65]
    plt.imshow(data, cmap = cm)
    plt.axis('tight')

    ylim = plt.ylim()
    xlim = plt.xlim()
    [plt.plot(xlim, [x+.5, x+.5], '--', linewidth=.5, color='gray') for x in cumulative_trials]

    if odor_res['odor_valence'][0] =='CS+':
        plt.plot([water_on, water_on], ylim, '-', linewidth=1, color='turquoise')
        color = odor_colors[0]
        xticks = [odor_on, odor_off, water_on]
        xticklabels = ['ON', 'OFF', 'US']
    elif odor_res['odor_valence'][0] == 'CS-':
        color = odor_colors[1]
        xticks = [odor_on, odor_off]
        xticklabels = ['ON', 'OFF']
    else:
        raise ValueError('unrecognized odor valence for plotting')
    plt.fill_between([odor_on, odor_off], ylim[0],ylim[1], alpha = .7, facecolor=color)

    # plt.xticks(xticks, xticklabels)
    plt.xticks([])
    plt.yticks([])
    plt.title(odor_name.upper())

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    plot._easy_save(save_path, odor_standard)
