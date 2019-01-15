from decode import experiment
import analysis
import filter
import reduce
import plot
from tools import experiment_tools
import os

import numpy as np
from CONSTANTS.config import Config
import CONSTANTS.conditions as experimental_conditions
import matplotlib.pyplot as plt

#
core_experiments = ['vary_neuron_odor', 'vary_decoding_style_odor', 'vary_decoding_style_days']
experiments = ['vary_decoding_style_days']
EXPERIMENT = False
ANALYZE = True
argTest = True

#inputs
condition = experimental_conditions.OFC

#load files from matlab
# init.load_matlab.load_condition(condition)

if 'vary_decoding_style_days' in experiments:
    data_path = os.path.join(Config.LOCAL_DATA_PATH, Config.LOCAL_DATA_TIMEPOINT_FOLDER, condition.name)
    experiment_path = os.path.join(Config.LOCAL_EXPERIMENT_PATH, 'vary_decoding_style_days', condition.name)
    save_path = os.path.join(Config.LOCAL_FIGURE_PATH, 'vary_decoding_style_days', condition.name)

    if EXPERIMENT:
        if condition.name == 'PIR' or condition.name == 'PIR_NAIVE':
            style = ('identity')
        elif condition.name == 'OFC':
            style = ('identity','csp_identity','csm_identity')
        else:
            raise ValueError('condition name not recognized')

        experiment_tools.perform(experiment=experiment.decode_day_as_label,
                                 condition=condition,
                                 experiment_configs=experiment.vary_decode_style(argTest=argTest, style=style),
                                 data_path=data_path,
                                 save_path=experiment_path)

    if ANALYZE:
        res = analysis.load_results(experiment_path)
        analysis.analyze_results(res)
        decode_styles = np.unique(res['decode_style'])

        #decoding, plot for each condition: mouse + decode style
        ax_args = {'yticks': [0, .2, .4, .6, .8, 1.0], 'ylim': [-.05, 1.05]}
        loopkey = ['shuffle']
        mice = np.unique(res['mouse'])
        for i, mouse in enumerate(mice):
            for j, dc in enumerate(decode_styles):
                select_dict = {'mouse':mouse, 'decode_style': dc}
                plot.plot_results(res, x_key='time', y_key='mean', loop_keys=loopkey,
                                  select_dict=select_dict, path=save_path, ax_args=ax_args)

        #summary for last day of each mouse
        nMouse = np.unique(res['mouse']).size
        ax_args = {'yticks': [0, .2, .4, .6, .8, 1.0], 'ylim': [0, 1.05],
                   'xticks': [0, 1]}
        for decode_style in decode_styles:
            cur_res = filter.filter(res, {'decode_style': decode_style})
            summary_res = reduce.filter_reduce(cur_res, 'shuffle','max')
            plot_args = {'alpha': .6, 'fill': False}
            plot.plot_results(summary_res, x_key='shuffle', y_key='max',
                              path = save_path, plot_function=plt.bar, plot_args=plot_args, ax_args=ax_args,
                              save=False)

            plot_args = {'alpha': .5, 'linewidth': 1, 'marker': 'o', 'markersize': 1}
            plot.plot_results(res, x_key='shuffle', y_key='max', loop_keys='mouse',
                              select_dict={'decode_style': decode_style},
                              colors = ['Black'] * nMouse,
                              path = save_path, plot_args= plot_args, ax_args=ax_args,
                              save=True, reuse=True)

if 'vary_decoding_style_odor' in experiments:
    data_path = os.path.join(Config.LOCAL_DATA_PATH, Config.LOCAL_DATA_TIMEPOINT_FOLDER, condition.name)
    experiment_path = os.path.join(Config.LOCAL_EXPERIMENT_PATH, 'vary_decoding_style', condition.name)

    save_path = os.path.join(Config.LOCAL_FIGURE_PATH, 'vary_decoding_style', condition.name)
    if EXPERIMENT:
        if condition.name == 'PIR':
            experiment_config = experiment.vary_decode_style
        elif condition.name == 'PIR_NAIVE':
            experiment_config = experiment.vary_decode_style_identity
        else:
            raise ValueError('condition name not recognized')
        experiment_tools.perform(experiment=experiment.decode_odor_as_label,
                                 condition=condition,
                                 experiment_configs=experiment_config(argTest=argTest),
                                 data_path=data_path,
                                 save_path=experiment_path)
    if ANALYZE:
        res = analysis.load_results(experiment_path)
        analysis.analyze_results(res)
        last_day_per_mouse = filter.get_last_day_per_mouse(res)
        res_lastday = filter.filter_days_per_mouse(res, days_per_mouse=last_day_per_mouse)
        decode_styles = np.unique(res['decode_style'])

        #decoding, plot for each condition: mouse + decode style
        ax_args = {'yticks': [0, .2, .4, .6, .8, 1.0], 'ylim': [-.05, 1.05]}
        loopkey = ['shuffle','day']
        mice = np.unique(res['mouse'])
        for i, mouse in enumerate(mice):
            for j, dc in enumerate(decode_styles):
                select_dict = {'mouse':mouse, 'decode_style': dc}
                plot.plot_results(res, x_key='time', y_key='mean', loop_keys=loopkey,
                                  select_dict=select_dict, path=save_path, ax_args=ax_args)

        #decoding, comparing last day of each mouse, shuffle and no shuffle
        ax_args = {'yticks': [0, .2, .4, .6, .8, 1.0], 'ylim': [-.05, 1.05]}
        loopkey = ['shuffle','decode_style']
        mice = np.unique(res['mouse'])
        for j, decode_style in enumerate(decode_styles):
            for i, mouse in enumerate(mice):
                select_dict = {'mouse':mouse, 'decode_style': decode_style}
                plot.plot_results(res_lastday, x_key='time', y_key='mean', loop_keys=loopkey,
                                  select_dict=select_dict, path=save_path, ax_args=ax_args)

        # summary of performance for each day, line plot
        loopkey = ['mouse']
        mice = np.unique(res['mouse'])
        ax_args = {'yticks': [0, .2, .4, .6, .8, 1.0], 'ylim': [0, 1.05]}
        plot_args = {'alpha':.5, 'linewidth':1, 'marker':'o', 'markersize':1}
        for j, decode_style in enumerate(decode_styles):
            select_dict = {'decode_style': decode_style, 'shuffle':False}
            plot.plot_results(res, x_key='day', y_key='max', loop_keys=loopkey,
                              colors= ['Black'] * mice.size,
                              select_dict= select_dict, path=save_path,
                              ax_args=ax_args, plot_args=plot_args)

        #summary for last day of each mouse
        nMouse = np.unique(res_lastday['mouse']).size
        ax_args = {'yticks': [0, .2, .4, .6, .8, 1.0], 'ylim': [0, 1.05]}
        plot_args = {'marker': 'o', 's': 10, 'facecolors': 'none', 'alpha': .4}
        select_dict = {'shuffle':False}
        plot.plot_results(res_lastday, x_key='decode_style', y_key='max', select_dict=select_dict,
                          path = save_path, plot_function=plt.scatter, plot_args= plot_args, ax_args=ax_args,
                          save=False)

        res_nonshuffle = filter.filter(res_lastday, filter_dict={'shuffle':False})
        summary_res_nonshuffle = reduce.filter_reduce(res_nonshuffle, 'decode_style', 'max')
        plot_args = {'alpha': .6, 'fill': False}
        plot.plot_results(summary_res_nonshuffle, x_key='decode_style', y_key='max', select_dict=select_dict,
                          path = save_path, plot_function=plt.bar, plot_args=plot_args,
                          save=True, reuse=True)


if 'vary_neuron_odor' in experiments:
    data_path = os.path.join(Config.LOCAL_DATA_PATH, Config.LOCAL_DATA_TIMEPOINT_FOLDER, condition.name)
    experiment_path = os.path.join(Config.LOCAL_EXPERIMENT_PATH, 'vary_neuron', condition.name)
    save_path = os.path.join(Config.LOCAL_FIGURE_PATH, 'vary_neuron', condition.name)

    if EXPERIMENT:
        if condition.name == 'PIR' or condition.name == 'PIR_NAIVE':
            style = ['identity']
        elif condition.name == 'OFC':
            style = ['identity','valence']
        else:
            raise ValueError('condition name not recognized')

        experiment_tools.perform(experiment=experiment.decode_odor_as_label,
                                 condition=condition,
                                 experiment_configs=experiment.vary_neuron(
                                     argTest=argTest, neurons=condition.min_neurons, style=style),
                                 data_path=data_path,
                                 save_path=experiment_path)

    if ANALYZE:
        res = analysis.load_results(experiment_path)
        analysis.analyze_results(res)
        last_day_per_mouse = filter.get_last_day_per_mouse(res)
        res_lastday = filter.filter_days_per_mouse(res, days_per_mouse=last_day_per_mouse)

        # neurons vs decoding performance. plot separately for each shuffle + decode style
        xkey = 'neurons'
        ykey = 'max'
        loopkey = 'mouse'
        ax_args = {'yticks': [0, .2, .4, .6, .8, 1.0], 'ylim': [-.05, 1.05]}
        plot_args = {'alpha': .5, 'linewidth': 1, 'marker': 'o', 'markersize': 1}
        mice = np.unique(res['mouse'])

        decode_styles = np.unique(res['decode_style'])
        shuffles = np.unique(res['shuffle'])
        for i, shuffle in enumerate(shuffles):
            for j, dc in enumerate(decode_styles):
                select_dict = {'shuffle':shuffle, 'decode_style': dc}
                plot.plot_results(res_lastday, x_key=xkey, y_key=ykey, loop_keys=loopkey,
                                  select_dict=select_dict, colors= ['Black'] * mice.size,
                                  path=save_path, ax_args=ax_args, plot_args=plot_args)

        # decoding performance wrt time for each mouse, comparing 1st and last day
        # plot separately for each mouse and each decode style
        xkey = 'time'
        ykey = 'mean'
        loopkey = 'day'
        ax_args = {'yticks': [0, .2, .4, .6, .8, 1.0], 'ylim': [-.05, 1.05]}
        mice = np.unique(res['mouse'])
        for i, mouse in enumerate(mice):
            for j, dc in enumerate(decode_styles):
                select_dict = {'decode_style': dc, 'shuffle': False,
                               'neurons': 40, 'mouse': mouse, 'day': [0, last_day_per_mouse[i]]}
                plot.plot_results(res, x_key=xkey, y_key=ykey, loop_keys=loopkey, select_dict=select_dict,
                                  path=save_path, ax_args=ax_args)