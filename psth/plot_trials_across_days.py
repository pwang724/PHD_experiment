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
from psth.psth_helper import PSTHConfig, subtract_baseline, draw_scale_line_xy
from scipy.stats import sem
import copy

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'arial'

class OFC_Config(object):
    def __init__(self):
        self.condition = experimental_conditions.OFC
        self.mouse = 0
        self.days = [[1,4],[1,5],[1,5],[1,5]]
        self.cells = [31, 33, 48, 7]
        self.vlim = .7
        self.title = ['Naive','Learned']

config = PSTHConfig()
condition_config = OFC_Config()
condition = condition_config.condition
mouse = condition_config.mouse
days = condition_config.days
cells = condition_config.cells

data_path = os.path.join(Config.LOCAL_DATA_PATH, Config.LOCAL_DATA_TIMEPOINT_FOLDER, condition.name)
save_path = os.path.join(Config.LOCAL_FIGURE_PATH, 'OTHER', 'PSTH',  condition.name, 'TRIAL')

res = analysis.load_data(data_path)
analysis.add_indices(res)
analysis.add_time(res)
res_mouse = filter.filter(res, filter_dict={'mouse': mouse, 'day':days})

i = 0
cell_days = days[i]
cell = cells[i]

for i,_ in enumerate(cell_days):
    odors = copy.copy(condition.odors[mouse])
    if cell_days[i] >= condition.training_start_day[mouse]:
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
        cur_data = data[cell, ix, :]
        cur_data = subtract_baseline(cur_data, config.baseline_start, odor_on - config.baseline_end)
        list_of_psths.append(cur_data)
    min_trial = np.min([x.shape[0] for x in list_of_psths])
    list_of_psths = [x[:min_trial,:] for x in list_of_psths]
    psth = np.concatenate(list_of_psths, axis=1)

    fig = plt.figure(figsize=(3, 1.5))
    rect = [.1, .1, .7, .7]
    rect_cb = [.82, 0.1, 0.02, .7]
    ax = fig.add_axes(rect)
    plt.imshow(psth, vmin=-condition_config.vlim, vmax=condition_config.vlim, cmap='bwr')
    plt.axis('tight')

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    condition_lines = np.cumsum([frames_per_trial] * len(odors))[:-1]
    odor_on_lines = np.arange(odor_on, frames_per_trial * len(odors), frames_per_trial)
    water_on_lines = np.arange(water_on, frames_per_trial * len(odors), frames_per_trial)
    if 'water' in odors:
        water_on_lines = water_on_lines[[0,1,4]]
    else:
        water_on_lines = water_on_lines[[0,1]]

    if cell_days[i] >= condition.training_start_day[mouse]:
        xticks = np.append(odor_on_lines, water_on_lines)
    else:
        xticks = odor_on_lines

    plt.xticks(xticks, '')
    plt.yticks([])
    plt.tick_params(direction='out', length=2, width=.5, grid_alpha=0.5)
    plt.ylabel(condition_config.title[i])
    for line in condition_lines:
        plt.plot([line, line], plt.ylim(), '--', color='grey', linewidth=.5)

    for j, x in enumerate(odor_on_lines):
        plt.text(x, -1, odors[j].upper())

    ax = fig.add_axes(rect_cb)
    cb = plt.colorbar(cax=ax, ticks=[-condition_config.vlim, condition_config.vlim])
    cb.outline.set_linewidth(0.5)
    cb.set_label(r'$\Delta$ F/F', fontsize=7, labelpad=-10)
    plt.tick_params(axis='both', which='major', labelsize=7)

    plot._easy_save(save_path,
                    'mouse_' + str(mouse) +
                    '_cell_' + str(cell) +
                    '_day_' + str(res_mouse['day'][i]))