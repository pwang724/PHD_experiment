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

condition = experimental_conditions.PIR
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

days = np.unique(mouse_res['day'])
odor_dict = {'PT CS+':'orange', 'PT Naive':'orange','CS+1':'lime','CS+2':'darkgreen','CS-1':'red','CS-2':'magenta'}


for j, day in enumerate(days):
    # plotting
    fig = plt.figure(figsize=(3, 2.5))
    rect = [.2, .1, .7, .65]
    fig.add_axes(rect)

    plot_data = []
    trials = []
    valences = []
    odor_standard_names = []

    temp = filter.filter(mouse_res, {'day': day})
    odors = np.unique(temp['odor_standard'])
    odor_catalogue = ['PT CS+', 'CS+1', 'CS+2', 'CS-1', 'CS-2']
    odors_standard = []
    for odor in odor_catalogue:
        if odor in odors:
            odors_standard.append(odor)

    for i, odor_standard in enumerate(odors_standard):
        temp = filter.filter(mouse_res, filter_dict={'odor_standard': odor_standard, 'day':day})

        x = temp['lick_raw_data']
        x = np.stack(x)
        #inputs
        data = x > .1
        plot_data.append(data)
        trials.append(data.shape[0])
        valences.append(temp['odor_valence'][0])
        odor_standard_names.append(odor_standard)

    plot_data = np.concatenate(plot_data, axis=0)
    trials.insert(0, 0)
    cumsum = np.cumsum(trials)
    cm = LinearSegmentedColormap.from_list('black_and_white', [(1, 1, 1), (0, 0, 0)], N=2)

    plt.imshow(plot_data, cmap = cm)
    plt.axis('tight')
    ylim = plt.ylim()
    xlim = plt.xlim()
    [plt.plot(xlim, [x-.5, x-.5], '--', linewidth=.5, color='gray') for x in cumsum]

    for i, valence in enumerate(valences):
        y = np.array(cumsum[i:i+2]) - .5
        if valence == 'CS+' or valence == 'PT CS+':
            plt.plot([water_on, water_on], y, '-', linewidth=1, color='turquoise')
            xticks = [odor_on, odor_off, water_on]
            xticklabels = ['ON', 'OFF', 'US']
        elif valence == 'CS-' or valence == 'PT Naive':
            xticks = [odor_on, odor_off]
            xticklabels = ['ON', 'OFF']
        elif valence == 'US':
            plt.plot([water_on, water_on], y, '-', linewidth=1, color='turquoise')
            xticks = [water_on]
            xticklabels = ['US']
        else:
            raise ValueError('unrecognized odor valence for plotting')

        if valence != 'US':
            color = odor_dict[odor_standard_names[i]]
            plt.fill_between([odor_on, odor_off], cumsum[i]-.5, cumsum[i+1]-.5, alpha = .7, facecolor=color)
            plt.text(0, (cumsum[i+1] - cumsum[i]) / 2 + cumsum[i], odor_standard_names[i])
    # plt.xticks(xticks, xticklabels)
    plt.title('Day {}'.format(day))
    plt.xticks([])
    plt.yticks([])
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    plot._easy_save(save_path, 'across_day_mouse_{}_day_{}'.format(mouse, day), dpi=1000)
