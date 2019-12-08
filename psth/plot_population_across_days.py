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
from psth.sorting import sort_by_selectivity, sort_by_onset
import plot

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'arial'
mpl.rcParams['axes.linewidth'] = 0.5

class PIRConfig(object):
    def __init__(self):
        self.condition = experimental_conditions.PIR
        self.mouse = [1]
        self.days = [0, 2]
        self.vlim = .25
        self.threshold = .1
        self.negative_threshold = -0.05
        self.sort_method = 'onset'
        self.title = 'odor'

class OFC_COMPOSITE_Config(object):
    def __init__(self):
        self.condition = experimental_conditions.OFC_COMPOSITE
        self.mouse = [0]
        self.days = [1,3]
        self.sort_day_ix = 0
        self.vlim = .2
        self.threshold = .02
        self.negative_threshold = -0.05
        self.title = 'odor'
        self.sort_method = 'onset'
        self.sort_style = 'CS+'
        self.independent_sort = True
        self.include_water = False
        self.period = 'pt'

class PIR_CONTEXT_Config(object):
    def __init__(self):
        self.condition = experimental_conditions.PIR_CONTEXT
        self.mouse = [0, 1]
        self.days = [0, 1]
        self.vlim = .25
        self.threshold = .1
        self.negative_threshold = -0.05
        self.sort_method = 'onset'
        self.title = 'odor'
        self.period = 'mush'

config = PSTHConfig()
condition_config = PIR_CONTEXT_Config()
condition = condition_config.condition
mouse = condition_config.mouse
days = condition_config.days

data_path = os.path.join(Config.LOCAL_DATA_PATH, Config.LOCAL_DATA_TIMEPOINT_FOLDER, condition.name)
save_path = os.path.join(Config.LOCAL_FIGURE_PATH, 'OTHER', 'PSTH',  condition.name, 'POPULATION_ACROSS')

res = analysis.load_data(data_path)
analysis.add_indices(res)
analysis.add_time(res)

#ODOR IDENTITIES, ONSET TIMES, NUMBER OF FRAMES, ETC HAVE TO MATCH ACROSS MICE
for o in range(len(condition.odors[0])):
    list_of_psths = []
    for i, day in enumerate(days):
        means = []
        for m in mouse:
            odor = condition.odors[m][o]
            res_mouse = filter.filter(res, filter_dict={'mouse': m, 'day': days})

            odor_on = res_mouse['DAQ_O_ON_F'][i]
            odor_off = res_mouse['DAQ_O_OFF_F'][i]
            water_on = res_mouse['DAQ_W_ON_F'][i]
            odor_trials = res_mouse['ODOR_TRIALS'][i]
            time_odor_on = 0
            time_odor_off = res_mouse['DAQ_O_OFF'][i] - res['DAQ_O_ON'][i]
            time_water_on = res_mouse['DAQ_W_ON'][i] - res['DAQ_O_ON'][i]
            time = res_mouse['time'][i]
            frames_per_trial = res_mouse['TRIAL_FRAMES'][i]

            data = utils.reshape_data(res_mouse['data'][i], nFrames= frames_per_trial,
                                                            cell_axis=0, trial_axis=1, time_axis=2)
            ix = odor == odor_trials
            cur_data = data[:, ix, :]
            for k, cell in enumerate(cur_data):
                cur_data[k,:,:] = subtract_baseline(cell, config.baseline_start, odor_on - config.baseline_end)
            mean = np.mean(cur_data, axis=1)
            means.append(mean)
        list_of_psths.append(np.concatenate(means,axis=0))

    ixs = sort_by_selectivity([list_of_psths[-1]], odor_on, water_on, condition_config)
    psth = np.concatenate(list_of_psths, axis=1)
    psth = psth[ixs,:]

    fig = plt.figure(figsize=(1.5, 3))
    rect = [.1, .1, .65, .7]
    rect_cb = [0.76, 0.1, 0.02, 0.7]
    ax = fig.add_axes(rect)
    plt.imshow(psth, vmin=-condition_config.vlim, vmax=condition_config.vlim, cmap='bwr')
    plt.axis('tight')

    # ax.spines['top'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    condition_lines = np.cumsum([frames_per_trial] * len(days))[:-1]
    odor_on_lines = np.arange(odor_on, frames_per_trial * len(days), frames_per_trial)
    odor_off_lines = np.arange(odor_off, frames_per_trial * len(days), frames_per_trial)
    if condition_config.period == 'pt' or odor in condition.csp[mouse[0]]:
        water_on_lines = np.arange(water_on, frames_per_trial * len(days), frames_per_trial)
        xticks = np.concatenate((odor_on_lines, odor_off_lines, water_on_lines))
    else:
        xticks = np.concatenate((odor_on_lines, odor_off_lines))

    plt.xticks(xticks, '')
    plt.yticks([])
    plt.tick_params(direction='out', length=2, width=.5, grid_alpha=0.5)

    for line in condition_lines:
        plt.plot([line, line], plt.ylim(), '--', color='grey', linewidth=.5)

    for line in xticks:
        plt.plot([line, line], plt.ylim(), '--', color='grey', linewidth=.25, alpha=0.5)


    titles = ['Naive','Learned']
    for j, x in enumerate(odor_on_lines):
        plt.text(x, -1, titles[j])
    plt.title(odor.upper())

    ax = fig.add_axes(rect_cb)
    cb = plt.colorbar(cax=ax, ticks=[-condition_config.vlim, condition_config.vlim])
    cb.outline.set_linewidth(0.5)
    cb.set_label(r'$\Delta$ F/F', fontsize=7, labelpad=-10)
    plt.tick_params(axis='both', which='major', labelsize=7)

    plot._easy_save(save_path,
                    'mouse_' + str(mouse) +
                    '_day_' + str(days) +
                    '_odor_' + str(odor))



