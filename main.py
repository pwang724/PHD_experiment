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

#
core_experiments = ['vary_neuron', 'vary_shuffle', 'vary_decoding_style']
experiments = ['vary_decode_style']
EXPERIMENT = True
ANALYZE = True
argTest = True

#inputs
condition = experimental_conditions.OFC

#load files from matlab
# init.load_matlab.load_condition(condition)

if 'vary_decode_style' in experiments:
    data_path = os.path.join(Config.LOCAL_DATA_PATH, Config.LOCAL_DATA_TIMEPOINT_FOLDER, condition.name)
    experiment_path = os.path.join(Config.LOCAL_EXPERIMENT_PATH, 'vary_decoding_style', condition.name)
    save_path = os.path.join(Config.LOCAL_FIGURE_PATH, 'vary_decoding_style', condition.name)
    if EXPERIMENT:
        experiment_tools.perform(experiment=experiment.decode_odor_as_label,
                                 condition=condition,
                                 experiment_configs=experiment.vary_decode_style(argTest=argTest),
                                 data_path=data_path,
                                 save_path=experiment_path)
    if ANALYZE:
        res = analysis.load_results(experiment_path)
        analysis.analyze_results(res)
        last_day_per_mouse = filter.get_last_day_per_mouse(res)
        res_lastday = filter.filter_days_per_mouse(res, days_per_mouse=last_day_per_mouse)

        #decoding, plot for each condition: mouse + decode style
        xkey = 'time'
        ykey = 'mean'
        loopkey = ['shuffle','day']
        plot_dict = {'yticks': [.4, .6, .8, 1.0], 'ylim': [.1, 1.05]}
        decode_styles = np.unique(res['decode_style'])
        mice = np.unique(res['mouse'])
        for i, mouse in enumerate(mice):
            for j, dc in enumerate(decode_styles):
                select_dict = {'mouse':mouse, 'decode_style': dc}
                plot.plot_results(res, xkey, ykey, loopkey, select_dict, save_path, plot_dict)

        #decoding, comparing last day of each mouse, shuffle and no shuffle
        last_day_per_mouse = filter.get_last_day_per_mouse(res)
        res_lastday = filter.filter_days_per_mouse(res, days_per_mouse=last_day_per_mouse)
        xkey = 'time'
        ykey = 'mean'
        loopkey = ['shuffle','decode_style']
        plot_dict = {'yticks': [.4, .6, .8, 1.0], 'ylim': [.1, 1.05]}
        mice = np.unique(res['mouse'])
        for i, mouse in enumerate(mice):
            select_dict = {'mouse':mouse}
            plot.plot_results(res_lastday, xkey, ykey, loopkey, select_dict, save_path, plot_dict)



if 'vary_shuffle' in experiments:
    data_path = os.path.join(Config.LOCAL_DATA_PATH, Config.LOCAL_DATA_TIMEPOINT_FOLDER, condition.name)
    experiment_path = os.path.join(Config.LOCAL_EXPERIMENT_PATH, 'vary_shuffle', condition.name)
    save_path = os.path.join(Config.LOCAL_FIGURE_PATH, 'vary_shuffle', condition.name)
    if EXPERIMENT:
        experiment_tools.perform(experiment=experiment.decode_odor_as_label,
                                 condition=condition,
                                 experiment_configs=experiment.vary_shuffle(argTest=argTest),
                                 data_path=data_path,
                                 save_path=experiment_path)
    if ANALYZE:
        res = analysis.load_results(experiment_path)
        analysis.analyze_results(res)

        #shuffle vs non-shuffle. plot all days, one figure per mouse
        xkey = 'time'
        ykey = 'mean'
        loopkey = ['shuffle','day']
        plot_dict = {'yticks': [.4, .6, .8, 1.0], 'ylim': [.35, 1.05]}
        mice = np.unique(res['mouse'])
        for i, mouse in enumerate(mice):
            select_dict = {'mouse': mouse}
            plot.plot_results(res, xkey, ykey, loopkey, select_dict, save_path, plot_dict)

        #shuffle vs non-shuffle. plot last day for all mice
        last_day_per_mouse = filter.get_last_day_per_mouse(res)
        res_lastday = filter.filter_days_per_mouse(res, days_per_mouse=last_day_per_mouse)
        xkey = 'time'
        ykey = 'mean'
        loopkey = 'shuffle'
        plot_dict = {'yticks': [.4, .6, .8, 1.0], 'ylim': [.35, 1.05]}
        plot.plot_results(res_lastday, xkey, ykey, loopkey, select_dict=None, path=save_path, ax_args=plot_dict)

        loopkey = ['shuffle', 'mouse']
        plot_dict = {'yticks': [.4, .6, .8, 1.0], 'ylim': [.35, 1.05]}
        plot.plot_results(res_lastday, xkey, ykey, loopkey, select_dict=None, path=save_path, ax_args=plot_dict)


if 'vary_neuron' in experiments:
    data_path = os.path.join(Config.LOCAL_DATA_PATH, Config.LOCAL_DATA_TIMEPOINT_FOLDER, condition.name)
    experiment_path = os.path.join(Config.LOCAL_EXPERIMENT_PATH, 'vary_neuron', condition.name)
    save_path = os.path.join(Config.LOCAL_FIGURE_PATH, 'vary_neuron', condition.name)

    if EXPERIMENT:
        experiment_tools.perform(experiment=experiment.decode_odor_as_label,
                                 condition=condition,
                                 experiment_configs=experiment.vary_neuron(argTest=argTest),
                                 data_path=data_path,
                                 save_path=experiment_path)

    if ANALYZE:
        res = analysis.load_results(experiment_path)
        analysis.analyze_results(res)
        last_day_per_mouse = filter.get_last_day_per_mouse(res)
        res_lastday = filter.filter_days_per_mouse(res, days_per_mouse=last_day_per_mouse)

        # neurons vs decoding performance
        xkey = 'neurons'
        ykey = 'max'
        loopkey = 'mouse'
        plot_dict = {'yticks': [.4, .6, .8, 1.0], 'ylim': [.35, 1.05]}
        plot.plot_results(res_lastday, xkey, ykey, loopkey, select_dict=None, path=save_path, ax_args=plot_dict)

        # decoding performance wrt time for each mouse, comparing 1st and last day
        xkey = 'time'
        ykey = 'mean'
        loopkey = 'day'
        plot_dict = {'yticks': [.4, .6, .8, 1.0], 'ylim': [.35, 1.05]}
        mice = np.unique(res['mouse'])
        for i, mouse in enumerate(mice):
            select_dict = {'neurons': 20, 'mouse': mouse, 'day': [0, last_day_per_mouse[i]]}
            plot.plot_results(res, xkey, ykey, loopkey, select_dict, save_path, plot_dict)