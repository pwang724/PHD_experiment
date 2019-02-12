import os
import filter
from _CONSTANTS import conditions as experimental_conditions
from _CONSTANTS.config import Config
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import analysis
import tools.utils as utils
from psth.psth_helper import PSTHConfig, subtract_baseline
import psth.sorting as sort
import plot
import copy

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'arial'
mpl.rcParams['axes.linewidth'] = 0.5

class PIR_Config(object):
    def __init__(self):
        self.condition = experimental_conditions.PIR
        self.mouse = 1
        self.days = [0]
        self.sort_day_ix = 0
        self.vlim = .25
        self.threshold = .1
        self.negative_threshold = -0.05
        self.title = 'odor'
        self.sort_method = 'selectivity'
        self.include_water = True

class OFC_Config(object):
    def __init__(self):
        self.condition = experimental_conditions.OFC
        self.mouse = 0
        self.days = [0, 4]
        self.sort_day_ix = 1
        self.vlim = .25
        self.threshold = .05
        self.negative_threshold = -0.05
        self.title = 'odor'
        self.sort_method = 'onset'
        self.sort_style = 'CS+'
        self.include_water = True

class BLA_Config(object):
    def __init__(self):
        self.condition = experimental_conditions.BLA
        self.mouse = 3
        self.days = [1, 5]
        self.sort_day_ix = 1
        self.vlim = .2
        self.threshold = .02
        self.negative_threshold = -0.05
        self.title = 'odor'
        self.sort_method = 'onset'
        self.sort_style = 'CS+'
        self.include_water = True

class OFC_LONGTERM_Config(object):
    def __init__(self):
        self.condition = experimental_conditions.OFC_LONGTERM
        self.mouse = 0
        self.days = [3,6]
        self.sort_day_ix = 0
        self.vlim = .2
        self.threshold = .02
        self.negative_threshold = -0.05
        self.title = 'odor'
        self.sort_method = 'onset'
        self.sort_style = 'CS+'
        self.independent_sort = True
        self.include_water = False

class OFC_COMPOSITE_Config(object):
    def __init__(self):
        self.condition = experimental_conditions.OFC_COMPOSITE
        self.mouse = 0
        self.days = [0,1,2]
        self.sort_day_ix = 0
        self.vlim = .2
        self.threshold = .02
        self.negative_threshold = -0.05
        self.title = 'odor'
        self.sort_method = 'onset'
        self.sort_style = 'CS+'
        self.independent_sort = True
        self.include_water = False

config = PSTHConfig()
condition_config = OFC_COMPOSITE_Config()
condition = condition_config.condition
mouse = condition_config.mouse
days = condition_config.days

data_path = os.path.join(Config.LOCAL_DATA_PATH, Config.LOCAL_DATA_TIMEPOINT_FOLDER, condition.name)
save_path = os.path.join(Config.LOCAL_FIGURE_PATH, 'OTHER', 'PSTH',  condition.name, 'POPULATION')

res = analysis.load_data(data_path)
analysis.add_indices(res)
analysis.add_time(res)
res_mouse = filter.filter(res, filter_dict={'mouse': mouse, 'day':days})

#sorting step
if not condition_config.independent_sort:
    i=condition_config.sort_day_ix
    odors = condition.odors[mouse]
    if days[i] >= condition.training_start_day[mouse] and condition_config.include_water:
        odors.append('water')
    odor_on = res_mouse['DAQ_O_ON_F'][i]
    water_on = res_mouse['DAQ_W_ON_F'][i]
    odor_trials = res_mouse['ODOR_TRIALS'][i]
    frames_per_trial = res_mouse['TRIAL_FRAMES'][i]

    data = utils.reshape_data(res_mouse['data'][i], nFrames= frames_per_trial,
                                                    cell_axis=0, trial_axis=1, time_axis=2)
    list_of_psths = []
    for j, odor in enumerate(odors):
        ix = odor == odor_trials
        cur_data = data[:, ix, :]
        for k, cell in enumerate(cur_data):
            cur_data[k,:,:] = subtract_baseline(cell, config.baseline_start, odor_on - config.baseline_end)
        mean = np.mean(cur_data, axis=1)
        list_of_psths.append(mean)

    if condition_config.sort_method == 'selectivity':
        ixs = sort.sort_by_selectivity(list_of_psths, odor_on, water_on, condition_config)
    elif condition_config.sort_method == 'onset':
        ixs = sort.sort_by_onset(list_of_psths, odor_on, water_on, condition_config)
    else:
        print('sorting method is not recognized')
    condition.odors[mouse].remove('water')

#plotting step
for i,_ in enumerate(days):
    odors = copy.copy(condition.odors[mouse])
    if days[i] >= condition.training_start_day[mouse] and condition_config.include_water:
        odors.append('water')

    odor_on = res_mouse['DAQ_O_ON_F'][i]
    water_on = res_mouse['DAQ_W_ON_F'][i]
    odor_trials = res_mouse['ODOR_TRIALS'][i]
    frames_per_trial = res_mouse['TRIAL_FRAMES'][i]

    data = utils.reshape_data(res_mouse['data'][i], nFrames=frames_per_trial,
                              cell_axis=0, trial_axis=1, time_axis=2)
    list_of_psths = []
    for j, odor in enumerate(odors):
        ix = odor == odor_trials
        cur_data = data[:, ix, :]
        for k, cell in enumerate(cur_data):
            cur_data[k, :, :] = subtract_baseline(cell, config.baseline_start, odor_on - config.baseline_end)
        mean = np.mean(cur_data, axis=1)
        list_of_psths.append(mean)

    if condition_config.independent_sort:
        if condition_config.sort_method == 'selectivity':
            ixs = sort.sort_by_selectivity(list_of_psths, odor_on, water_on, condition_config)
        elif condition_config.sort_method == 'onset':
            ixs = sort.sort_by_onset(list_of_psths, odor_on, water_on, condition_config)
        else:
            print('sorting method is not recognized')

    psth = np.concatenate(list_of_psths, axis=1)
    psth = psth[ixs,:]

    fig = plt.figure(figsize=(3.5, 3))
    fig_width = .14 * len(odors)
    rect = [.1, .1, fig_width, .7]
    rect_cb = [fig_width + .1 + .02, 0.1, 0.02, .7]
    ax = fig.add_axes(rect)
    plt.imshow(psth, vmin=-condition_config.vlim, vmax=condition_config.vlim, cmap='bwr')
    plt.axis('tight')

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    condition_lines = np.cumsum([frames_per_trial] * len(odors))[:-1]
    odor_on_lines_raw = np.arange(odor_on, frames_per_trial * len(odors), frames_per_trial)
    water_on_lines = np.arange(water_on, frames_per_trial * len(odors), frames_per_trial)
    if 'water' in odors:
        water_on_lines = water_on_lines[[0,1,4]]
        odor_on_lines = odor_on_lines_raw[:-1]
    else:
        water_on_lines = water_on_lines[[0,1]]
        odor_on_lines = odor_on_lines_raw

    if days[i] >= condition.training_start_day[mouse]:
        xticks = np.append(odor_on_lines, water_on_lines)
    else:
        xticks = odor_on_lines

    plt.xticks(xticks, '')
    plt.yticks([])
    plt.tick_params(direction='out', length=2, width=.5, grid_alpha=0.5)

    for line in condition_lines:
        plt.plot([line, line], plt.ylim(), '--', color='grey', linewidth=.5)

    for j, x in enumerate(odor_on_lines_raw):
        plt.text(x, -1, odors[j].upper())

    ax = fig.add_axes(rect_cb)
    cb = plt.colorbar(cax=ax, ticks=[-condition_config.vlim, condition_config.vlim])
    cb.outline.set_linewidth(0.5)
    cb.set_label(r'$\Delta$ F/F', fontsize=7, labelpad=-10)
    plt.tick_params(axis='both', which='major', labelsize=7)

    plot._easy_save(save_path,
                    'mouse_' + str(mouse) +
                    '_day_' + str(res_mouse['day'][i]))



