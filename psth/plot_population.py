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

class Base_Config(object):
    def __init__(self):
        self.condition = None
        self.mouse = None
        self.days = None
        self.sort_day_ix = None
        self.sort_method = 'onset'
        self.sort_onset_style = 'CS+'
        self.vlim = .25
        self.threshold = 0.05
        self.negative_threshold = -0.05
        self.title = 'odor'
        self.include_water = True
        self.independent_sort = False
        self.period = None
        self.across_days = False
        self.across_day_titles = None

class PIR_Config(Base_Config):
    def __init__(self):
        super(PIR_Config, self).__init__()
        self.condition = experimental_conditions.PIR
        self.mouse = 1
        self.days = [0]
        self.sort_day_ix = 0
        self.threshold = .1
        self.sort_method = 'selectivity'

class OFC_Config(Base_Config):
    def __init__(self):
        super(OFC_Config, self).__init__()
        self.condition = experimental_conditions.OFC
        self.mouse = 1
        self.days = [0, 4]
        self.sort_day_ix = 1

class BLA_Config(Base_Config):
    def __init__(self):
        super(BLA_Config, self).__init__()
        self.condition = experimental_conditions.BLA
        self.mouse = 3
        self.days = [1, 5]
        self.sort_day_ix = 1
        self.vlim = .2
        self.threshold = .02

class OFC_LONGTERM_Config(Base_Config):
    def __init__(self):
        super(OFC_LONGTERM_Config, self).__init__()
        self.condition = experimental_conditions.OFC_LONGTERM
        self.mouse = 0
        self.days = [3,6]
        self.sort_day_ix = 0
        self.vlim = .2
        self.threshold = .02
        self.independent_sort = True
        self.include_water = False

class OFC_COMPOSITE_PT_Config(Base_Config):
    def __init__(self):
        super(OFC_COMPOSITE_PT_Config, self).__init__()
        self.condition = experimental_conditions.OFC_COMPOSITE
        self.mouse = 0
        self.days = [1,3]
        self.sort_day_ix = 0
        self.vlim = .2
        self.threshold = .02
        self.independent_sort = True
        self.include_water = False
        self.period = 'pt'

        self.across_days = True
        self.across_days_titles = ['Naive', 'Learned']

class MPFC_COMPOSITE_PT_Config(Base_Config):
    def __init__(self):
        super(MPFC_COMPOSITE_PT_Config, self).__init__()
        self.condition = experimental_conditions.MPFC_COMPOSITE
        self.mouse = 0
        self.days = [1,3]
        self.sort_day_ix = 0
        self.vlim = .25
        self.threshold = .02
        self.independent_sort = True
        self.include_water = False
        self.period = 'pt'

        self.across_days = True
        self.across_days_titles = ['Naive', 'Learned']

class MPFC_COMPOSITE_DT_Config(Base_Config):
    def __init__(self):
        super(MPFC_COMPOSITE_DT_Config, self).__init__()
        self.condition = experimental_conditions.MPFC_COMPOSITE
        self.mouse = 0
        self.days = [0, 4, 7]
        self.sort_day_ix = 0
        self.vlim = .25
        self.threshold = .02
        self.independent_sort = True
        self.include_water = False
        self.period = 'dt'
        self.sort_method = 'plus_minus'

class OFC_COMPOSITE_DT_Config(Base_Config):
    def __init__(self):
        super(OFC_COMPOSITE_DT_Config, self).__init__()
        self.condition = experimental_conditions.OFC_COMPOSITE
        self.mouse = 0
        self.days = [9]
        self.sort_day_ix = 0
        self.vlim = .2
        self.threshold = .02
        self.independent_sort = True
        self.include_water = False
        self.period = 'dt'


config = PSTHConfig()
condition_config = MPFC_COMPOSITE_PT_Config()
condition = condition_config.condition
mouse = condition_config.mouse
days = condition_config.days

if hasattr(condition, 'odors'):
    odors = condition.odors[mouse]
elif condition_config.period == 'pt':
    odors = condition.pt_csp[mouse]
elif condition_config.period == 'dt':
    odors = condition.dt_odors[mouse]
else:
    raise ValueError('odor condition not recognized')

data_path = os.path.join(Config.LOCAL_DATA_PATH, Config.LOCAL_DATA_TIMEPOINT_FOLDER, condition.name)
save_path = os.path.join(Config.LOCAL_FIGURE_PATH, 'OTHER', 'PSTH',  condition.name, 'POPULATION')

res = analysis.load_data(data_path)
analysis.add_indices(res)
analysis.add_time(res)
res_mouse = filter.filter(res, filter_dict={'mouse': mouse, 'day':days})

#sorting step
if not condition_config.independent_sort:
    odors_copy = copy.copy(odors)
    i=condition_config.sort_day_ix
    if days[i] >= condition.training_start_day[mouse] and condition_config.include_water:
        odors_copy.append('water')
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

#plotting step
images = []
odor_on_times = []
water_on_times = []
list_of_odor_names = []
for i,_ in enumerate(days):
    odors_copy = copy.copy(odors)
    if days[i] >= condition.training_start_day[mouse] and condition_config.include_water:
        odors_copy.append('water')

    odor_on = res_mouse['DAQ_O_ON_F'][i]
    water_on = res_mouse['DAQ_W_ON_F'][i]
    odor_trials = res_mouse['ODOR_TRIALS'][i]
    frames_per_trial = res_mouse['TRIAL_FRAMES'][i]

    data = utils.reshape_data(res_mouse['data'][i], nFrames=frames_per_trial,
                              cell_axis=0, trial_axis=1, time_axis=2)
    list_of_psths = []
    for j, odor in enumerate(odors_copy):
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
        elif condition_config.sort_method == 'plus_minus':
            ixs = sort.sort_by_plus_minus(list_of_psths, odor_on, water_on, condition_config)
        else:
            raise ValueError('sorting method is not recognized')

    psth = np.concatenate(list_of_psths, axis=1)
    psth = psth[ixs,:]
    images.append(psth)
    odor_on_times.append(odor_on)
    water_on_times.append(water_on)
    list_of_odor_names.append(odors_copy)

if condition_config.across_days:
    images = [np.concatenate(images, axis=1)]
    odor_on_times = [odor_on_times[0]]
    water_on_times = [water_on_times[0]]
    if condition_config.across_days_titles:
        list_of_odor_names = [condition_config.across_days_titles]
    else:
        list_of_odor_names = [np.array(list_of_odor_names).flatten()]

for i, image in enumerate(images):
    odor_on = odor_on_times[i]
    water_on = water_on_times[i]
    titles = list_of_odor_names[i]
    n_plots = int(image.shape[1] / frames_per_trial)

    fig = plt.figure(figsize=(3.5, 3))
    fig_width = .14 * n_plots
    rect = [.1, .1, fig_width, .7]
    rect_cb = [fig_width + .1 + .02, 0.1, 0.02, .7]
    ax = fig.add_axes(rect)
    plt.imshow(image, vmin=-condition_config.vlim, vmax=condition_config.vlim, cmap='bwr')
    plt.axis('tight')

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    condition_lines = np.cumsum([frames_per_trial] * n_plots)[:-1]
    odor_on_lines_raw = np.arange(odor_on, frames_per_trial * n_plots, frames_per_trial)
    water_on_lines = np.arange(water_on, frames_per_trial * n_plots, frames_per_trial)
    if 'water' in titles:
        water_on_lines = water_on_lines[[0,1,4]]
        odor_on_lines = odor_on_lines_raw[:-1]
    else:
        if condition_config.period == 'pt':
            odor_on_lines = odor_on_lines_raw
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

    for j, x in enumerate(odor_on_lines):
        plt.text(x, -1, titles[j].upper())

    ax = fig.add_axes(rect_cb)
    cb = plt.colorbar(cax=ax, ticks=[-condition_config.vlim, condition_config.vlim])
    cb.outline.set_linewidth(0.5)
    cb.set_label(r'$\Delta$ F/F', fontsize=7, labelpad=-10)
    plt.tick_params(axis='both', which='major', labelsize=7)

    plot._easy_save(save_path,
                    'mouse_' + str(mouse) +
                    '_day_' + str(res_mouse['day'][i]))



