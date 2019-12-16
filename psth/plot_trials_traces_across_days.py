import os
import filter
from _CONSTANTS import conditions as experimental_conditions
from _CONSTANTS.config import Config
import plot
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import analysis
import tools.utils as utils
from psth.psth_helper import PSTHConfig, subtract_baseline
import copy

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'arial'

class OFC_Config(object):
    def __init__(self):
        self.condition = experimental_conditions.OFC
        self.mouse = 0
        self.days = [[1,2,3,4,5]]
        self.cells = [31]
        self.ylim = .7
        self.title = ['Naive','Learning','Learning','Learned','Learned']

class BLA_Config(object):
    def __init__(self):
        self.condition = experimental_conditions.BLA
        self.mouse = 3
        self.days = [[1,2,4]]
        self.cells = [1]
        self.ylim = .7
        self.title = ['Naive','Learning','Learned']

class PIR_Config(object):
    def __init__(self):
        self.condition = experimental_conditions.PIR
        self.mouse = 1
        self.days = [[0, 1, 2]]
        self.cells = [18]
        self.ylim = .6
        self.title = ['Naive', 'Learning', 'Learned']

class MPFC_Config(object):
    def __init__(self):
        #TODO
        self.condition = experimental_conditions.MPFC_COMPOSITE
        # self.mouse = 0
        # self.days = [[0, 1, 2]]
        # self.cells = [18]
        # self.ylim = .4
        # self.title = ['Naive', 'Learning', 'Learned']

condition_config = OFC_Config()

condition = condition_config.condition
mouse = condition_config.mouse
days = condition_config.days
cells = condition_config.cells

data_path = os.path.join(Config.LOCAL_DATA_PATH, Config.LOCAL_DATA_TIMEPOINT_FOLDER, condition.name)
save_path = os.path.join(Config.LOCAL_FIGURE_PATH, 'OTHER', 'PSTH',  condition.name, 'TRIAL_TRACE')

res = analysis.load_data(data_path)
analysis.add_indices(res)
analysis.add_time(res)
res_mouse = filter.filter(res, filter_dict={'mouse': mouse, 'day':days})

i = 0
cell_days = days[i]
cell = cells[i]

gap = 0
colors = ['lime', 'darkgreen', 'red', 'magenta']

for i,_ in enumerate(cell_days):
    odors = copy.copy(condition.odors[mouse])
    if cell_days[i] >= condition.training_start_day[mouse]:
        odors.append('water')

    odor_on = res_mouse['DAQ_O_ON_F'][i]
    odor_off = res_mouse['DAQ_O_OFF_F'][i]
    water_on = res_mouse['DAQ_W_ON_F'][i]
    odor_trials = res_mouse['ODOR_TRIALS'][i]
    frames_per_trial = res_mouse['TRIAL_FRAMES'][i]
    trial_period = res_mouse['TRIAL_PERIOD'][i]

    data = utils.reshape_data(res_mouse['data'][i], nFrames=frames_per_trial,
                              cell_axis=0, trial_axis=1, time_axis=2)
    list_of_psths = []
    for j, odor in enumerate(odors):
        ix = odor == odor_trials
        cur_data = data[cell, ix, :]
        cur_data -= 1
        list_of_psths.append(cur_data)
    min_trial = np.min([x.shape[0] for x in list_of_psths])
    list_of_psths = [x[:min_trial,:] for x in list_of_psths]

    fig, axs = plt.subplots(1, 5, figsize=(3, 3))

    space_x = frames_per_trial
    space_y = gap + condition_config.ylim
    total_y = space_y * min_trial
    for o, psth in enumerate(list_of_psths):
        cur_x = np.arange(0, space_x)
        cur_y = total_y - space_y
        plt.sca(axs[o])
        for j, y in enumerate(psth):
            y = y + cur_y
            plt.plot(cur_x,y, 'k', linewidth=.5)
            cur_y -= space_y
        cur_x += space_x

        plt.axis('tight')
        # plt.set_ticks_position('bottom')
        # plt.yaxis.set_ticks_position('left')
        plt.ylim([-space_y, total_y])
        plt.xlim([0, frames_per_trial])

        xticks = []
        xticklabels = []
        if o != 4:
            a, b = -space_y/5, total_y
            plt.fill_between([odor_on, odor_off], [a, a], [b, b], color=colors[o], alpha=.5, linewidth=0)
            xticks.append((odor_on + odor_off)/2)
            xticklabels.append('Odor')

        if o in [0, 1, 4] and cell_days[i] >= condition.training_start_day[mouse]:
            plt.plot([water_on, water_on], [-space_y/5, total_y], '--', color='grey', linewidth=.5)
            xticks.append(water_on)
            xticklabels.append('US')

        plt.title(odors[o].upper())
        plt.xticks(xticks, xticklabels, fontsize=5)
        plt.yticks([])
        plt.tick_params(direction='out', length=0)
        ax = axs[o]
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

    axs[0].set_ylabel('Trials')
    time_bar = 5
    time_frames = time_bar / trial_period
    time_coordinates = [frames_per_trial-time_frames, frames_per_trial]
    axs[4].plot(time_coordinates, [-space_y/2, -space_y/2], linewidth=1.5, color='black')
    axs[4].text(time_coordinates[0], -space_y, '{} s'.format(time_bar), fontsize=7)

    axs[4].plot([])
    plot._easy_save(save_path,
                    'mouse_' + str(mouse) +
                    '_cell_' + str(cell) +
                    '_day_' + str(res_mouse['day'][i]))