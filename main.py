import init.load_matlab
import experiment
import analysis
import filter
import plot
from tools import experiment_tools
import os

import numpy as np
from CONSTANTS.config import Config
import CONSTANTS.conditions as experimental_conditions

#inputs
argTest = True
condition = experimental_conditions.OFC
data_path = os.path.join(Config.LOCAL_EXPERIMENT_PATH, 'Valence', condition.name)
save_path = os.path.join(Config.LOCAL_FIGURE_PATH, 'Valence', condition.name)

#load files from matlab
init.load_matlab.load_condition(condition)

#run exp
experiment_tools.perform(experiment=experiment.decode_odor_as_label,
                         condition=condition,
                         experiment_configs=experiment.vary_neuron_valence(argTest=argTest),
                         path=data_path)

#analyze exp
res = analysis.load_results(data_path)
analysis.analyze_results(res)

#prepare files for plotting
last_day_per_mouse = filter.get_last_day_per_mouse(res)
res_lastday = filter.filter_days_per_mouse(res, days_per_mouse=last_day_per_mouse)

## plotting

#neurons vs decoding performance
xkey = 'neurons'
ykey = 'max'
loopkey = 'mouse'
plot_dict = {'yticks':[.4, .6, .8, 1.0], 'ylim':[.35, 1.05]}
plot.plot_results(res_lastday, xkey, ykey, loopkey, select_dict=None, path= save_path, ax_args=plot_dict)

#decoding performance wrt time for each mouse, comparing 1st and last day
xkey = 'time'
ykey = 'mean'
loopkey = 'day'
plot_dict = {'yticks':[.4, .6, .8, 1.0], 'ylim':[.35, 1.05]}

mice = np.unique(res['mouse'])
for i, mouse in enumerate(mice):
    select_dict = {'neurons':20, 'mouse': mouse, 'day':[0, last_day_per_mouse[i]]}
    plot.plot_results(res, xkey, ykey, loopkey, select_dict, save_path, plot_dict)