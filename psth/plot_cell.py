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

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'arial'

class PIR_Config(object):
    def __init__(self):
        self.condition = experimental_conditions.PIR
        self.mouse = 1
        self.days = [[0, 2], [0,2],[0,2],[0,2]]
        self.cells = [0, 22, 45, 18]
        self.ylim = [-.1, 1]
        self.colors = ['blue', 'cyan', 'lime', 'magenta']

class OFC_Config(object):
    def __init__(self):
        self.condition = experimental_conditions.OFC
        self.mouse = 0
        self.days = [[0,2],[0,4],[0,5],[0,5],[0,5]]
        self.cells = [5, 31, 33, 48, 7]
        self.ylim = [-.1, 1]
        self.colors = ['lime', 'darkgreen', 'red', 'magenta']
        self.name = 'odor'
        self.include_water = False

class OFC_Context_Config(object):
    def __init__(self):
        self.condition = experimental_conditions.OFC_CONTEXT
        self.mouse = 0
        self.days = [[0,1],[0,1],[0,1]]
        self.cells = [0, 2, 6] #0, 2, 6
        self.ylim = [-.1, 1]
        self.colors = ['red', 'magenta', 'lime', 'darkgreen']
        self.name = 'odor'

class OFC_State_Config(object):
    def __init__(self):
        self.condition = experimental_conditions.OFC_STATE
        self.mouse = 0
        self.days = [[0,1],[0,1],[0,1]]
        self.cells = [3, 17, 48]
        self.ylim = [-.1, 1]
        self.colors = ['lime', 'darkgreen', 'red', 'magenta']
        self.name = 'odor'

class OFC_Reversal_Config(object):
    def __init__(self):
        self.condition = experimental_conditions.OFC_REVERSAL
        self.mouse = 0
        self.days = [[0, 2, 4],[0,2, 4],[1, 2, 4],[0,2 ,4]]
        self.cells = [0, 3, 8, 22]
        self.ylim = [-.1, 1]
        self.colors = ['lime', 'darkgreen', 'red', 'magenta']
        self.name = 'odor'

class BLA_Config(object):
    def __init__(self):
        self.condition = experimental_conditions.BLA
        self.mouse = 3
        self.days = [[1, 4], [1, 4], [1, 4], [1, 4]]
        self.cells = [1, 12, 15, 18]
        self.ylim = [-.1, 1]
        self.colors = ['lime', 'darkgreen', 'red', 'magenta', 'turquoise']
        self.include_water = True
        self.name = 'odor'

config = PSTHConfig()
condition_config = OFC_Config()
condition = condition_config.condition
mouse = condition_config.mouse
days = condition_config.days
cells = condition_config.cells

data_path = os.path.join(Config.LOCAL_DATA_PATH, Config.LOCAL_DATA_TIMEPOINT_FOLDER, condition.name)
save_path = os.path.join(Config.LOCAL_FIGURE_PATH, 'OTHER', 'PSTH',  condition.name, 'CELL')

res = analysis.load_data(data_path)
analysis.add_indices(res)
analysis.add_time(res)
odors = condition.odors[mouse]
if condition_config.include_water:
    odors.append('water')

for j, cell in enumerate(cells):
    res_mouse = filter.filter(res, filter_dict={'mouse': mouse, 'day': days[j]})
    for i in range(len(res_mouse['day'])):
        data = utils.reshape_data(res_mouse['data'][i],
                                                        nFrames= res_mouse['TRIAL_FRAMES'][i],
                                                        cell_axis=0, trial_axis=1, time_axis=2)
        odor_on = res_mouse['DAQ_O_ON_F'][i]
        odor_trials = res_mouse['ODOR_TRIALS'][i]
        time_odor_on = 0
        time_odor_off = res_mouse['DAQ_O_OFF'][i] - res['DAQ_O_ON'][i]
        time_water_on = res_mouse['DAQ_W_ON'][i] - res['DAQ_O_ON'][i]
        time = res_mouse['time'][i]

        fig = plt.figure(figsize=(2, 1.5))
        ax = fig.add_axes([.2, .25, .7, .65])

        for j, odor in enumerate(odors):
            ix = odor == odor_trials
            cur_data = data[cell, ix, :]
            cur_data = subtract_baseline(cur_data, config.baseline_start, odor_on - config.baseline_end)

            mean = np.mean(cur_data, axis=0)
            err = sem(cur_data, axis=0)

            ax.plot(time, mean, color=condition_config.colors[j], linewidth=config.linewidth)
            ax.fill_between(time, mean - err, mean + err, zorder=0, lw=0, alpha=config.fill_alpha,
                            color=condition_config.colors[j])

        plt.xticks([time_odor_on, time_odor_off, time_water_on], labels=['ON', 'OFF', 'US'])
        plt.yticks([])
        plt.ylim(condition_config.ylim)
        if condition_config.name == 'standard':
            legends = ['CS+1','CS+2','CS-1','CS-2']
        else:
            legends = [odor.upper() for odor in odors]
        plt.legend(legends, frameon=False)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(r'$\Delta$ F/F')
        draw_scale_line_xy(ax, length=[5, .5], offset=[0, 0], linewidth= config.scale_linewidth)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

        plot._easy_save(save_path,
                        'mouse_' + str(mouse) +
                        '_cell_' + str(cell) +
                        '_day_' + str(i))



