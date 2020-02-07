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
import scipy.signal as signal
import copy

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'arial'

class OFC_Config(object):
    def __init__(self):
        self.condition = experimental_conditions.OFC
        self.mouse = 0
        self.cells = [1]
        self.days = [[1, 2, 3, 4, 5]]
        # self.mouse = 2
        # self.cells = [40]
        self.ylim = 1.25
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
        self.cells = [22]
        self.ylim = .6
        self.title = ['Naive', 'Learning', 'Learned']

class MPFC_Config(object):
    def __init__(self):
        self.condition = experimental_conditions.MPFC_COMPOSITE
        self.mouse = 2
        self.days = [[3,4,5]]
        self.cells = [65]
        self.ylim = .3
        self.title = ['Naive', 'Learning', 'Learned']

csp_only = False
plot_licks = True
condition_config = MPFC_Config()

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
if csp_only:
    us_ix = 2
else:
    us_ix = 4

for i,_ in enumerate(cell_days):
    try:
        odors = copy.copy(condition.odors[mouse])
    except:
        odors = copy.copy(condition.dt_odors[mouse])

    if csp_only:
        odors = odors[:2]
    if cell_days[i] >= condition.training_start_day[mouse]:
        odors.append('water')

    odor_on = res_mouse['DAQ_O_ON_F'][i]
    odor_off = res_mouse['DAQ_O_OFF_F'][i]
    water_on = res_mouse['DAQ_W_ON_F'][i] + 1
    print(odor_on,water_on)
    odor_trials = res_mouse['ODOR_TRIALS'][i]
    frames_per_trial = res_mouse['TRIAL_FRAMES'][i]
    trial_period = res_mouse['TRIAL_PERIOD'][i]
    samp = res_mouse['DAQ_SAMP'][i]

    data = utils.reshape_data(res_mouse['data'][i], nFrames=frames_per_trial,
                              cell_axis=0, trial_axis=1, time_axis=2)
    lick_data = res_mouse['DAQ_DATA'][i][:,3,:] > 0.5
    lick_limit = np.round(frames_per_trial * trial_period * samp).astype(int)
    lick_data = lick_data[:lick_limit, :]
    list_of_psths = []
    list_of_licks = []
    for j, odor in enumerate(odors):
        ix = odor == odor_trials
        cur_data = data[cell, ix, :]
        cur_data -= 1
        cur_licks = np.transpose(lick_data[:, ix])
        list_of_psths.append(cur_data)
        list_of_licks.append(cur_licks)

    min_trial = np.min([x.shape[0] for x in list_of_psths])
    list_of_psths = [x[:min_trial,:] for x in list_of_psths]
    list_of_licks = [x[:min_trial, :] for x in list_of_licks]

    fig, axs = plt.subplots(1, us_ix + 1, figsize=(3.75, 3))

    space_x = frames_per_trial
    space_y = gap + condition_config.ylim
    total_y = space_y * min_trial
    resample = 50
    for o, psth in enumerate(list_of_psths):
        licks = list_of_licks[o]
        cur_x = np.arange(0, space_x)
        cur_x_lick = np.linspace(0, space_x, lick_limit)
        cur_y = total_y - space_y
        plt.sca(axs[o])
        for j, y in enumerate(psth):
            y_lick = licks[j] * condition_config.ylim/ 6 + cur_y
            if plot_licks:
                plt.step(cur_x_lick,y_lick, 'r', linewidth=.25, alpha=1)

            y = y + cur_y
            plt.plot(cur_x,y, 'k', linewidth=.5)
            cur_y -= space_y
        cur_x += space_x
        cur_x_lick += space_x

        plt.axis('tight')
        # plt.set_ticks_position('bottom')
        # plt.yaxis.set_ticks_position('left')
        plt.ylim([-space_y, total_y])
        plt.xlim([0, frames_per_trial])

        xticks = []
        xticklabels = []


        if o != us_ix:
            a, b = -space_y/5, total_y
            plt.fill_between([odor_on, odor_off], [a, a], [b, b], color=colors[o], alpha=.5, linewidth=0)
            xticks.append((odor_on + odor_off)/2)
            xticklabels.append('Odor')

        if o in [0, 1, us_ix] and cell_days[i] >= condition.training_start_day[mouse]:
            plt.plot([water_on, water_on], [-space_y/5, total_y], '--', color='grey', linewidth=.5)
            xticks.append(water_on)
            xticklabels.append('US')

        plt.title(odors[o].upper())
        plt.xticks([])
        plt.tick_params(direction='out', length=0)
        # plt.xticks(xticks, xticklabels, fontsize=5)
        plt.yticks([])
        ax = axs[o]
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

    axs[0].set_ylabel('Trials')
    time_bar = 5
    time_frames = time_bar / trial_period
    time_coordinates = [frames_per_trial-time_frames, frames_per_trial]
    axs[us_ix].plot(time_coordinates, [-space_y/2, -space_y/2], linewidth=1, color='black')
    axs[us_ix].text(time_coordinates[0], -space_y, '{} s'.format(time_bar), fontsize=7)

    df_bar = 0.5
    df_coordinates = [0, df_bar]
    axs[us_ix].plot([frames_per_trial-5, frames_per_trial-5], df_coordinates, linewidth=1, color='black')
    axs[us_ix].text(frames_per_trial+5, df_bar, '{} DF/F'.format(df_bar), fontsize=7)

    axs[us_ix].plot([])
    plot._easy_save(save_path,
                    'mouse_' + str(mouse) +
                    '_cell_' + str(cell) +
                    '_day_' + str(res_mouse['day'][i]))