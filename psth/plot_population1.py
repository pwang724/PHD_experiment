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
        self.sort_day_ix = None
        self.sort_method = 'onset'
        self.sort_onset_style = 'CS+'
        self.vlim = .25
        self.threshold = 0.05
        self.negative_threshold = -0.05
        self.title = 'odor'
        self.include_water = True

        self.mouse = None #if not plotting big, the mouse id
        self.days = None #if not plotting big, the list of days to plot

        self.independent_sort = True #whether to sort independently on a given day
        self.sort_days = None #if not independent sort, the day to align all other days to. if plot big, list of days for each mouse to sort to

        self.period = None
        self.across_days = False
        self.across_day_titles = None

        self.plot_big = False #whether to collapse all mice together
        self.plot_big_days = None #if plotting big, the days to plot
        self.plot_big_naive = None #if plotting big, whether to use water ticks on odors
        self.delete_nonselective = False

        self.filter_ix = None #filter odor ix

class PIR_Config(Base_Config):
    def __init__(self):
        super(PIR_Config, self).__init__()
        self.condition = experimental_conditions.PIR
        self.mouse = 1
        self.days = [0, 1, 2]
        self.sort_day_ix = 0
        self.threshold = .1
        self.sort_method = 'selectivity'
        self.delete_nonselective = True
        self.independent_sort = False
        self.sort_days = self.days[-1]
        self.vlim = .35
        self.include_water = True

class PIR_CSP_Config(Base_Config):
    def __init__(self):
        super(PIR_CSP_Config, self).__init__()
        self.condition = experimental_conditions.PIR
        self.mouse = 1
        self.days = [0, 1, 2]
        self.sort_day_ix = 0
        self.threshold = .1
        self.sort_method = 'selectivity'
        self.delete_nonselective = True
        self.independent_sort = False
        self.sort_days = self.days[-1]
        self.include_water = True
        self.vlim = .35
        self.filter_ix = 0

class PIR_CSM_Config(Base_Config):
    def __init__(self):
        super(PIR_CSM_Config, self).__init__()
        self.condition = experimental_conditions.PIR
        self.mouse = 1
        self.days = [0, 1, 2]
        self.sort_day_ix = 0
        self.threshold = .1
        self.sort_method = 'selectivity'
        self.delete_nonselective = True
        self.independent_sort = False
        self.sort_days = self.days[-1]
        self.include_water = True
        self.vlim = .35
        self.filter_ix = 2

class PIR_CONTEXT_Config(Base_Config):
    def __init__(self):
        super(PIR_CONTEXT_Config, self).__init__()
        self.condition = experimental_conditions.PIR_CONTEXT
        self.mouse = 0
        self.days = [1]
        self.sort_day_ix = 0
        self.threshold = .1
        self.sort_method = 'selectivity'
        self.delete_nonselective = True
        self.include_water = False

class OFC_Config(Base_Config):
    def __init__(self):
        super(OFC_Config, self).__init__()
        self.condition = experimental_conditions.OFC
        self.mouse = 0
        self.days = [0, 1, 2, 3, 4, 5]
        self.vlim = .25
        self.sort_day_ix = 1
        self.include_water = False
        self.independent_sort = False
        self.sort_days = 5

class OFC_REVERSAL_Config(Base_Config):
    def __init__(self):
        super(OFC_REVERSAL_Config, self).__init__()
        self.condition = experimental_conditions.OFC_REVERSAL
        self.mouse = 0
        self.days = [0, 1, 2, 3, 4, 5]
        self.vlim = .25
        self.include_water = False
        self.independent_sort = False
        self.sort_days = 1

class BLA_Config(Base_Config):
    def __init__(self):
        super(BLA_Config, self).__init__()
        self.condition = experimental_conditions.BLA
        self.mouse = 3
        self.days = [0, 1, 2, 3, 4, 5, 6]
        self.sort_days = None
        self.vlim = .2
        self.threshold = .02
        self.independent_sort = True

class OFC_LONGTERM_Config(Base_Config):
    def __init__(self):
        super(OFC_LONGTERM_Config, self).__init__()
        self.condition = experimental_conditions.OFC_LONGTERM
        self.mouse = 0
        self.days = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        self.independent_sort = False
        self.sort_days = 3
        self.vlim = .25
        self.threshold = .03
        self.include_water = False

class OFC_COMPOSITE_PT_Config(Base_Config):
    def __init__(self):
        super(OFC_COMPOSITE_PT_Config, self).__init__()
        self.condition = experimental_conditions.OFC_COMPOSITE
        self.mouse = 1
        self.days = [0,1,2,3,4,5,6,7,8,9]
        # self.days = [1,2,3,4]
        # self.days = [0,1,2,3,4,5,6,7,8,9]
        self.sort_day_ix = 0
        self.vlim = .25
        self.threshold = .03
        self.independent_sort = False
        self.include_water = False
        self.period = 'ptdt'
        self.sort_days = 4

        self.across_days = True
        self.across_days_titles = ['Naive', 'Learned','c','d']

class OFC_COMPOSITE_DT_Config(Base_Config):
    def __init__(self):
        super(OFC_COMPOSITE_DT_Config, self).__init__()
        self.condition = experimental_conditions.OFC_COMPOSITE
        self.mouse = 1
        self.days = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.sort_day_ix = 0
        self.vlim = .25
        self.threshold = .03
        self.independent_sort = False
        self.sort_days = 4
        self.include_water = False
        self.period = 'ptdt'

class MPFC_COMPOSITE_PT_Config(Base_Config):
    def __init__(self):
        super(MPFC_COMPOSITE_PT_Config, self).__init__()
        self.condition = experimental_conditions.MPFC_COMPOSITE
        self.mouse = 0
        self.days = [0,3,4,5,6,7,8]
        self.sort_day_ix = 0
        self.vlim = .25
        self.threshold = .02
        self.independent_sort = False
        self.include_water = False
        self.period = 'dt'
        self.sort_days = 8

        self.across_days = True
        self.across_days_titles = ['Naive', 'Learned']

class MPFC_COMPOSITE_DT_Config(Base_Config):
    def __init__(self):
        super(MPFC_COMPOSITE_DT_Config, self).__init__()
        self.condition = experimental_conditions.MPFC_COMPOSITE
        self.mouse = 1
        self.days = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        self.sort_day_ix = 0
        self.vlim = .25
        self.threshold = .02
        self.independent_sort = False
        self.sort_days = 8
        self.include_water = False
        self.period = 'ptdt'
        self.sort_method = 'plus_minus'


class PIR_BIG_Config(Base_Config):
    def __init__(self):
        super(PIR_BIG_Config, self).__init__()
        self.condition = experimental_conditions.PIR
        self.threshold = .1
        self.sort_method = 'selectivity'
        self.plot_big = True
        self.plot_big_days = [3,2,3,3,3,3]
        self.plot_big_naive = False
        self.sort_day_ix = 0

class PIR_CONTEXT_BIG_Config(Base_Config):
    def __init__(self):
        super(PIR_CONTEXT_BIG_Config, self).__init__()
        self.condition = experimental_conditions.PIR_CONTEXT
        self.threshold = .1
        self.sort_method = 'selectivity'
        self.plot_big = True
        self.plot_big_days = [1,1]
        self.plot_big_naive = False
        self.sort_day_ix = 0


class OFC_BIG_Config(Base_Config):
    def __init__(self):
        super(OFC_BIG_Config, self).__init__()
        self.condition = experimental_conditions.OFC
        self.plot_big = True
        self.threshold = 0.04
        self.vlim = .25
        self.plot_big_days = [4,4,3,3,3]
        # self.sort_days = [3, 3, 2, 2, 2]
        self.sort_day_ix = 0
        self.plot_big_naive = False
        self.include_water = True
        self.sort_method = 'plus_minus'

        # self.plot_big_days = [0,0,0,0,0]
        # self.plot_big_naive = True
        # self.include_water = False

class OFC_LT_BIG_Config(Base_Config):
    def __init__(self):
        super(OFC_LT_BIG_Config, self).__init__()
        self.condition = experimental_conditions.OFC_LONGTERM
        self.plot_big = True
        self.threshold = 0.03
        self.vlim = .25
        self.plot_big_days = [8, 7, 7, -1]
        # self.plot_big_days = [3, 2, 2, -1]
        self.sort_day_ix = 0
        self.plot_big_naive = False
        self.include_water = False

        # self.plot_big_days = [0,0,0,-1]
        # self.plot_big_naive = True
        # self.include_water = False

class OFC_REVERSAL_BIG_Config(Base_Config):
    def __init__(self):
        super(OFC_REVERSAL_BIG_Config, self).__init__()
        self.condition = experimental_conditions.OFC_REVERSAL
        self.plot_big = True
        self.threshold = 0.03
        self.vlim = .25
        self.plot_big_days = [3] * 5
        self.sort_days = [3] * 5
        self.plot_big_naive = False
        self.include_water = False
        self.sort_onset_style = 'CS-'

class OFC_STATE_BIG_Config(Base_Config):
    def __init__(self):
        super(OFC_STATE_BIG_Config, self).__init__()
        self.condition = experimental_conditions.OFC_STATE
        self.plot_big = True
        self.threshold = 0.03
        self.vlim = .25
        self.plot_big_days = [0] * 5
        self.sort_days = [0] * 5
        self.sort_day_ix = 0
        self.plot_big_naive = False
        self.include_water = False
        self.sort_onset_style = 'CS+'

class OFC_CONTEXT_BIG_Config(Base_Config):
    def __init__(self):
        super(OFC_CONTEXT_BIG_Config, self).__init__()
        self.condition = experimental_conditions.OFC_CONTEXT
        self.plot_big = True
        self.threshold = 0.03
        self.vlim = .25
        self.plot_big_days = [0] * 4
        self.sort_days = [0] * 4
        self.sort_day_ix = 0
        self.plot_big_naive = False
        self.include_water = False
        self.sort_onset_style = 'CS-'

class OFC_COMPOSITE_BIG_Config(Base_Config):
    def __init__(self):
        super(OFC_COMPOSITE_BIG_Config, self).__init__()
        self.condition = experimental_conditions.OFC_COMPOSITE
        self.plot_big = True
        self.threshold = 0.04
        self.vlim = .3
        self.sort_day_ix = 0

        # self.plot_big_days = [1,1,1,1]
        self.plot_big_days = [3,4,4,4]
        self.period = 'pt'

        # self.plot_big_days = [0, 0, 0, 0]
        # self.plot_big_days = [4,4,6,4]
        # self.plot_big_days = [5,5,9,5]
        # self.plot_big_days = [8,9,10,8]
        # self.period = 'dt'
        self.plot_big_naive = False
        self.include_water = False

class MPFC_COMPOSITE_BIG_Config(Base_Config):
    def __init__(self):
        super(MPFC_COMPOSITE_BIG_Config, self).__init__()
        self.condition = experimental_conditions.MPFC_COMPOSITE
        self.sort_day_ix = 0
        self.vlim = .25
        self.threshold = .03
        self.sort_method = 'plus_minus'
        # self.sort_method = 'selectivity'
        self.plot_big = True

        # self.period = 'pt'
        # self.plot_big_days = [1, 1, 1, 1]
        # self.plot_big_days = [3, 3, 3, 3]

        self.period = 'dt'
        self.plot_big_days = [0, 0, 0, 0]
        self.plot_big_days = [3, 3, 4, 4]
        self.plot_big_days = [4, 4, 5, 5]
        self.plot_big_days = [8, 8, 5, 8]

        self.plot_big_naive = False
        self.include_water = True

class BLA_BIG_Config(Base_Config):
    def __init__(self):
        super(BLA_BIG_Config, self).__init__()
        self.condition = experimental_conditions.BLA
        self.sort_method = 'selectivity'
        self.sort_day_ix = 0
        self.vlim = .2
        self.threshold = .02
        self.plot_big = True
        self.plot_big_days = [5, 4, 6, 6]
        self.sort_days = [4, 3, 5, 5]
        self.plot_big_naive = False

def helper(res, mouse, day, condition_config):
    def _pad(data, diff):
        newp = np.zeros_like(data)
        if diff > 0:
            newp[:, :diff] = np.repeat(data[:,-1].reshape(-1,1), diff, axis= 1)
            newp[:, diff:] = data[:, :-diff]
            print('early odor time. mouse: {}, day: {}'.format(res_mouse['mouse'][0], res_mouse['day'][0]))
        else:
            newp[:,:diff] = data[:,-diff:]
            newp[:,diff:] = np.repeat(data[:,-1].reshape(-1,1), -diff, axis= 1)
            print('late odor time. mouse: {}, day: {}'.format(res_mouse['mouse'][0], res_mouse['day'][0]))
        return newp

    def _align(data, diff):
        newp = np.zeros([data.shape[0], data.shape[1] + diff])
        newp[:, :data.shape[1]] = data
        newp[:,data.shape[1]:] = np.repeat(data[:,-1].reshape(-1,1), diff, axis= 1)
        print('pad frames. mouse: {}, day: {}'.format(res_mouse['mouse'][0], res_mouse['day'][0]))
        return newp

    right_frame = np.max(res['TRIAL_FRAMES'])
    right_on = np.median(res['DAQ_O_ON_F'])
    res_mouse = filter.filter(res, filter_dict={'mouse': mouse, 'day': day})

    if hasattr(condition, 'odors'):
        odors = condition.odors[mouse]
    elif condition_config.period == 'pt':
        odors = ['naive'] + condition.pt_csp[mouse]
    elif condition_config.period == 'dt':
        odors = condition.dt_odors[mouse]
    elif condition_config.period == 'ptdt':
        odors = ['naive'] + condition.pt_csp[mouse] + condition.dt_odors[mouse]
    else:
        raise ValueError('odor condition not recognized')

    odors_copy = copy.copy(odors)
    if condition_config.include_water and day >= condition.training_start_day[mouse]:
        odors_copy.append('water')

    odor_on = res_mouse['DAQ_O_ON_F'][0]
    water_on = res_mouse['DAQ_W_ON_F'][0]
    odor_trials = res_mouse['ODOR_TRIALS'][0]
    frames_per_trial = res_mouse['TRIAL_FRAMES'][0]

    data = utils.reshape_data(res_mouse['data'][0], nFrames=frames_per_trial,
                              cell_axis=0, trial_axis=1, time_axis=2)
    list_of_psth = []
    odors_out = []
    for odor in odors_copy:
        ix = odor == odor_trials
        if np.any(ix):
            cur_data = data[:, ix, :]
            for k, cell in enumerate(cur_data):
                cur_data[k, :, :] = subtract_baseline(cell, config.baseline_start, odor_on - config.baseline_end)
            mean = np.mean(cur_data, axis=1)

            if np.abs(odor_on - right_on) > 2:
                diff = (right_on - odor_on).astype(int)
                mean = _pad(mean, diff)
                odor_on = odor_on + diff
                water_on = water_on + diff

            if frames_per_trial < right_frame:
                diff = right_frame - frames_per_trial
                mean = _align(mean, diff)

            list_of_psth.append(mean)
            odors_out.append(odor)

    # if 'naive' in odors_out:
    #     ix = odors_out.index('oct')
    #     odors_out.pop(ix)
    #     list_of_psth.pop(ix)

    return list_of_psth, odor_on, water_on, odors_out

def plotter(image, odor_on, water_on, odor_names, condition_config, save_path, name_str = ''):
    if black:
        plt.style.use('dark_background')

    frames_per_trial = 75

    titles = odor_names
    n_plots = int(image.shape[1] / frames_per_trial)

    fig = plt.figure(figsize=(3.5, 3))
    fig_width = .14 * n_plots
    rect = [.1, .1, fig_width, .7]
    rect_cb = [fig_width + .1 + .02, 0.1, 0.02, .7]
    ax = fig.add_axes(rect)

    if black:
        from matplotlib.colors import LinearSegmentedColormap
        cdict1 = {'red': ((0.0, 0.0, 0.0),
                          (0.5, 0.0, 0.1),
                          (1.0, 1.0, 1.0)),

                  'green': ((0.0, 0.0, 0.0),
                            (1.0, 0.0, 0.0)),

                  'blue': ((0.0, 0.0, 0.55),
                           (0.5, 0.1, 0.0),
                           (1.0, 0.0, 0.0))
                  }
        cmap = LinearSegmentedColormap('BlueRed1', cdict1)
        cmap = LinearSegmentedColormap.from_list("", ["turquoise", "black", "red"])

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.tick_params(axis=u'both', which=u'both', length=0)
    else:
        cmap = 'bwr'
        plt.tick_params(direction='out', length=2, width=.5, grid_alpha=0.5)

    plt.imshow(image, vmin=-condition_config.vlim, vmax=condition_config.vlim, cmap=cmap)
    plt.axis('tight')

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    condition_lines = np.cumsum([frames_per_trial] * n_plots)[:-1]
    odor_on_lines_raw = np.arange(odor_on, frames_per_trial * n_plots, frames_per_trial)
    water_on_lines = np.arange(water_on, frames_per_trial * n_plots, frames_per_trial)
    if 'water' in titles:
        water_on_lines = water_on_lines[[0, 1, 4]]
        # water_on_lines = water_on_lines[[2, 3, 4]]
        odor_on_lines = odor_on_lines_raw[:-1]
    else:
        if condition_config.period == 'pt':
            odor_on_lines = odor_on_lines_raw
        else:
            if condition_config.filter_ix is None:
                water_on_lines = water_on_lines[:2]
                # water_on_lines = water_on_lines[2:]
            else:
                water_on_lines = water_on_lines[[0]]
            odor_on_lines = odor_on_lines_raw

    if not naive:
        xticks = np.concatenate((odor_on_lines, odor_on_lines + 8, water_on_lines))
        xticklabels = ['ON'] * len(odor_on_lines) + ['OFF'] * len(odor_on_lines) + ['US'] * len(water_on_lines)
    else:
        xticks = np.concatenate((odor_on_lines, odor_on_lines + 8))
        xticklabels = ['ON'] * len(odor_on_lines) + ['OFF'] * len(odor_on_lines)

    plt.xticks(xticks, xticklabels, fontsize = 5)
    range = image.shape[0]
    if range > 100:
        interval = 50
    else:
        interval = 25
    plt.yticks(np.arange(0, range, interval))

    for line in xticks:
        plt.plot([line, line], plt.ylim(), '--', color='grey', linewidth=.5, alpha=0.5)

    for line in condition_lines:
        plt.plot([line, line], plt.ylim(), '--', color='darkgrey', linewidth=.75)

    for j, x in enumerate(odor_on_lines_raw):
        plt.text(x, -1, titles[j].upper())

    axcb = fig.add_axes(rect_cb)
    cb = plt.colorbar(cax=axcb, ticks=[-condition_config.vlim, condition_config.vlim])

    if black:
        cb.outline.set_visible(False)
        cb.set_ticks([])
    else:
        cb.outline.set_linewidth(0.5)
        cb.set_label(r'$\Delta$ F/F', fontsize=7, labelpad=-10)

    plt.tick_params(axis='both', which='major', labelsize=7)

    name_black = '_black' if black else ''
    if condition_config.plot_big:
        name = 'big_mouse_' + ','.join([str(x) for x in condition_config.plot_big_days])
        name += '_sorted_to_' + ','.join([str(x) for x in condition_config.sort_days])
        name += name_black
    else:
        name = 'mouse_' + str(mouse) + name_str
        if not condition_config.independent_sort:
            name += '_sorted_to_' + str(condition_config.sort_days)

    if condition_config.filter_ix is not None:
        name += '_odor_' + str(condition_config.filter_ix)

    plt.sca(ax)
    # plt.title(name)
    plot._easy_save(save_path, name)

def sort_helper(list_of_psth, odor_on, water_on, condition_config):
    if condition_config.sort_method == 'selectivity':
        ixs = sort.sort_by_selectivity(list_of_psth[:4], odor_on, water_on, condition_config,
                                       delete_nonselective=condition_config.delete_nonselective)
    elif condition_config.sort_method == 'onset':
        ixs = sort.sort_by_onset(list_of_psth, odor_on, water_on, condition_config)
    elif condition_config.sort_method == 'plus_minus':
        if len(list_of_psth) == 5:
            ixs = sort.sort_by_plus_minus(list_of_psth[1:], odor_on, water_on, condition_config)
            # ixs = sort.sort_by_plus_minus(list_of_psth[:5], odor_on, water_on, condition_config)
        elif len(list_of_psth) < 4:
            ixs = sort.sort_by_onset(list_of_psth, odor_on, water_on, condition_config)
        else:
            ixs = sort.sort_by_plus_minus(list_of_psth, odor_on, water_on, condition_config)
    else:
        raise ValueError('sorting method is not recognized')

    list_of_psth = [x[ixs, :] for x in list_of_psth]
    return list_of_psth, ixs


black = False
config = PSTHConfig()
condition_config = OFC_CONTEXT_BIG_Config()
condition = condition_config.condition

data_path = os.path.join(Config.LOCAL_DATA_PATH, Config.LOCAL_DATA_TIMEPOINT_FOLDER, condition.name)
save_path = os.path.join(Config.LOCAL_FIGURE_PATH, 'OTHER', 'PSTH',  condition.name, 'POPULATION')

res = analysis.load_data(data_path)
analysis.add_indices(res)
analysis.add_time(res)
if condition_config.plot_big:
    mice = np.unique(res['mouse'])
    days_per_mouse = condition_config.plot_big_days
    list_of_image = []
    list_of_list_of_psth = []
    for i, day in enumerate(days_per_mouse):
        mouse = mice[i]
        if day != -1:
            list_of_psth, odor_on, water_on, odor_names = helper(res, mouse, day, condition_config)
            image = np.concatenate(list_of_psth, axis=1)
            list_of_list_of_psth.append(list_of_psth)
            list_of_image.append(image)

    list_of_psth = np.concatenate(list_of_list_of_psth, axis=1)
    image = np.concatenate(list_of_image, axis=0)

    #sort
    sort_list = []
    if condition_config.sort_days is None:
        condition_config.sort_days = condition_config.plot_big_days
    sort_days_per_mouse = condition_config.sort_days
    for i, day in enumerate(sort_days_per_mouse):
        mouse = mice[i]
        if day != -1:
            temp, _, _, _ = helper(res, mouse, day, condition_config)
            sort_list.append(temp)
    # sort_list = np.concatenate(sort_list, axis=1)
    sort_list = np.concatenate(sort_list, axis=1)
    _, ixs = sort_helper(sort_list,odor_on,water_on, condition_config)
    image = image[ixs,:]
    image = image
    naive = condition_config.plot_big_naive
    # list_of_odor_names = [['CS+1','CS+2','CS-1','CS-2']]
    # if condition_config.include_water:
    #     list_of_odor_names[0].append('water')

    plotter(image, odor_on-1, water_on, odor_names, condition_config, save_path)
else:
    mouse = condition_config.mouse
    days = condition_config.days

    if not condition_config.independent_sort:
        sort_day = condition_config.sort_days
        assert isinstance(sort_day, int)
        list_of_psth, odor_on, water_on, odor_names = helper(res, mouse, sort_day, condition_config)
        list_of_psth, ixs = sort_helper(list_of_psth, odor_on, water_on, condition_config)

    for day in days:
        list_of_psth, odor_on, water_on, odor_names = helper(res, mouse, day, condition_config)

        if condition_config.independent_sort:
            list_of_psth, _ = sort_helper(list_of_psth, odor_on, water_on, condition_config)
        else:
            list_of_psth = [x[ixs, :] for x in list_of_psth]

        #filter
        filter_ix = condition_config.filter_ix
        if filter_ix is not None:
            list_of_psth = [list_of_psth[filter_ix]]
            odor_names = [odor_names[filter_ix]]


        image = np.concatenate(list_of_psth, axis=1)

        naive = day < condition_config.condition.training_start_day[mouse]
        name_str = '_day_' + str(day)
        if condition_config.period == 'pt' or condition_config.period == 'ptdt':
            name_str += '_' + condition_config.period
        plotter(image, odor_on-1, water_on, odor_names, condition_config, save_path, name_str=name_str)


    # if condition_config.across_days:
    #     image = [np.concatenate(list_of_psth, axis=1)]
    #     odor_on_times = [odor_on]
    #     water_on_times = [water_on]
    #     if condition_config.across_days_titles:
    #         list_of_odor_names = [condition_config.across_days_titles]
    #     else:
    #         list_of_odor_names = [np.array(list_of_odor_names).flatten()]





