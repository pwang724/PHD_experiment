import analysis
import behavior.behavior_analysis
import filter
import reduce
import plot
import os

from pca.experiment import do_PCA, load_pca, easy_shuffle, analyze_pca, normalize_per_mouse
import copy
import numpy as np
from _CONSTANTS.config import Config
import _CONSTANTS.conditions as experimental_conditions
from behavior.behavior_analysis import get_days_per_mouse

import matplotlib.pyplot as plt


class PCAConfig(object):
    def __init__(self):
        self.n_principal_components = 3
        #style can be 'all', 'csp', or 'csm'
        self.style = 'csp'
        self.average = True
        self.shuffle_iterations = 100
        self.start_at_training = False

experiments = [
    'shuffle',
]

condition = experimental_conditions.OFC_COMPOSITE
config = PCAConfig()
config.style = 'csp'
config.n_principal_components = 5
config.shuffle_iterations = 50
config.average = False

data_path = os.path.join(Config.LOCAL_DATA_PATH, Config.LOCAL_DATA_TIMEPOINT_FOLDER, condition.name)
save_path = os.path.join(Config.LOCAL_EXPERIMENT_PATH, 'PCA_' + config.style, condition.name)
figure_path = os.path.join(Config.LOCAL_FIGURE_PATH, 'PCA', config.style, condition.name)

EXPERIMENT = False
ANALYZE = True
shuffle = True

ax_args = {'yticks': [0, .2, .4, .6, .8, 1.0], 'ylim': [-.05, 1.05]}
fill_args = {'zorder': 0, 'lw': 0, 'alpha': 0.3}
line_args = {'alpha': .5, 'linewidth': 1, 'marker': 'o', 'markersize': 1}
behavior_line_args = {'alpha': .5, 'linewidth': 1, 'marker': '.', 'markersize': 0, 'linestyle': '--'}
error_args = {'fmt':'.', 'capsize':2, 'elinewidth':1, 'markersize':2, 'alpha': .5}
scatter_args = {'marker':'o', 's':5, 'facecolors': 'none', 'alpha':.5}


if 'shuffle' in experiments:
    if EXPERIMENT:
        do_PCA(condition, config, data_path, save_path)

    if ANALYZE:
        shuffle_keys = ['mouse']
        res = load_pca(save_path)
        analysis.add_indices(res)
        analysis.add_time(res)
        analysis.add_odor_value(res, condition)

        # if hasattr(condition, 'training_start_day') and config.start_at_training:
        #     list_of_start_days = condition.training_start_day
        # else:
        #     list_of_start_days = [0] * np.unique(res['mouse']).size
        # list_of_days = []
        # for i, start_day in enumerate(list_of_start_days):
        #     list_of_days.append(np.arange(start_day, dt_last_day[i] + 1))
        # res = filter.filter_days_per_mouse(res, list_of_days)

        res = filter.filter(res, {'odor_valence':['CS+', 'CS-', 'PT CS+']})
        res['shuffle'] = np.zeros(res['odor'].shape, dtype=int)

        res = easy_shuffle(res, shuffle_keys=shuffle_keys, n = config.shuffle_iterations)
        analyze_pca(res)
        summary_res = reduce.new_filter_reduce(res, ['mouse','day','odor_valence','shuffle'], reduce_key='PCA Distance')
        normalize_per_mouse(summary_res, 'PCA Distance')

        lick_res = behavior.behavior_analysis.get_licks_per_day(data_path, condition)
        analysis.add_odor_value(lick_res, condition)
        # lick_res = filter.filter_days_per_mouse(lick_res, list_of_days)
        lick_res = filter.filter(lick_res, {'odor_valence':['CS+', 'CS-', 'PT CS+']})
        lick_res = reduce.new_filter_reduce(lick_res, ['odor_valence','day','mouse'], reduce_key='lick_boolean')

        def _helper(lick_res, summary_res, mouse, odor_valence):
            behavior_colors = [colors_dict[x] for x in odor_valence]
            colors = [colors_dict[x] for x in odor_valence]

            if 'COMPOSITE' in condition.name:
                days = behavior.behavior_analysis._get_days_per_condition(data_path, condition)
                res_dt_naive = filter.filter_days_per_mouse(summary_res, [x[0] for x in days])
                res_dt_train = filter.filter_days_per_mouse(summary_res, [x[1:] for x in days])
                list_of_res = [res_dt_naive, res_dt_train]
                lick_dt_naive = filter.filter_days_per_mouse(lick_res, [x[0] for x in days])
                lick_dt_train = filter.filter_days_per_mouse(lick_res, [x[1:] for x in days])
                list_of_lick_res = [lick_dt_naive, lick_dt_train]
            else:
                list_of_res = [summary_res]
                list_of_lick_res = [lick_res]

            # plot.plot_results(summary_res, x_key='day', y_key='PCA Distance',
            #                   select_dict={'mouse':mouse, 'shuffle': 1, 'odor_valence':odor_valence},
            #                   colors=['gray']*5,
            #                   ax_args=ax_args, plot_args=line_args,
            #                   path=figure_path, reuse=True, save=False)
            #
            # plot.plot_results(summary_res, x_key='day', y_key='PCA Distance', error_key='PCA Distance_sem',
            #                   select_dict={'mouse':mouse, 'shuffle': 1, 'odor_valence':odor_valence},
            #                   colors=['gray']*5, plot_function=plt.fill_between,
            #                   ax_args=ax_args, plot_args=fill_args,
            #                   path=figure_path, reuse=True, save=False)
            for i, res in enumerate(list_of_res):
                reuse = True
                if i == 0:
                    reuse = False
                save = False
                if i == len(list_of_res) -1:
                    save = True
                l_res = list_of_lick_res[i]
                plot.plot_results(l_res, x_key='day', y_key='lick_boolean', loop_keys='odor_valence',
                                  select_dict={'mouse': mouse, 'odor_valence': odor_valence},
                                  colors=behavior_colors,
                                  ax_args=ax_args, plot_args=behavior_line_args,
                                  path=figure_path, save=False, sort=True, reuse=reuse)
                plot.plot_results(res, x_key='day', y_key='PCA Distance', loop_keys='odor_valence',
                                  select_dict={'mouse':mouse, 'shuffle':0, 'odor_valence':odor_valence},
                                  colors=colors,
                                  ax_args=ax_args, plot_args=line_args,
                                  path=figure_path, reuse=True, save=save)


        mice = np.unique(res['mouse'])
        colors_dict = {'CS+':'green', 'CS-':'red','PT CS+':'orange'}
        for i, mouse in enumerate(mice):
            _helper(lick_res, summary_res, mouse, odor_valence=['CS+', 'PT CS+'])

        plot_cs = False
        summary_ax_args = copy.copy(ax_args)
        summary_line_args = {'alpha': .3, 'linewidth': .5, 'linestyle': '--', 'marker': 'o', 'markersize': 1}
        summary_res.pop('PCA Distance_std')
        summary_res.pop('PCA Distance_sem')
        summary_res = filter.filter(summary_res, filter_dict={'shuffle': 0})
        if plot_cs:
            select_dict = {'odor_valence': 'CS+'}
        else:
            select_dict = None

        if condition.name == 'OFC_LONGTERM':
            dt_learned_day, dt_last_day = get_days_per_mouse(data_path, condition)
            filtered_res = filter.filter_days_per_mouse(summary_res, list(zip(dt_learned_day, dt_last_day)))
            analysis.add_naive_learned(filtered_res, dt_learned_day, dt_last_day, 'Learned', 'Over-trained')
            mean_std_res = reduce.new_filter_reduce(filtered_res, reduce_key='PCA Distance',
                                                    filter_keys=['training_day', 'odor_valence'])
            plot.plot_results(filtered_res, x_key='training_day', y_key = 'PCA Distance', loop_keys=['odor_valence','mouse'],
                              select_dict=select_dict,
                              colors=['Green'] * mice.size + ['Red'] * mice.size,
                              ax_args= summary_ax_args, plot_function=plt.plot, plot_args=summary_line_args,
                              path=figure_path, save=False)

            plot.plot_results(mean_std_res, x_key='training_day', y_key='PCA Distance', loop_keys='odor_valence',
                              select_dict=select_dict,
                              colors=['Green','Red'], plot_function=plt.plot,
                              ax_args=ax_args, plot_args=line_args,
                              path=figure_path, reuse=True, save=False)

            plot.plot_results(mean_std_res, x_key='training_day', y_key='PCA Distance', loop_keys='odor_valence',
                              select_dict=select_dict,
                              error_key='PCA Distance_sem',
                              colors=['Green','Red'], plot_function=plt.errorbar,
                              ax_args=summary_ax_args, plot_args=error_args,
                              path=figure_path, reuse=True, save=True, legend=False)



        if condition.name == 'OFC_STATE' or condition.name == 'OFC_CONTEXT':
            if condition.name == 'OFC_STATE':
                summary_ax_args = {'yticks': [0, .2, .4, .6, .8, 1.0], 'ylim': [-.05, 1.05],
                           'xlim':[-.5, 1.5], 'xticks':[0, 1], 'xticklabels':['Thirsty','Sated']}
            if condition.name == 'OFC_CONTEXT':
                summary_ax_args = {'yticks': [0, .2, .4, .6, .8, 1.0], 'ylim': [-.05, 1.05],
                           'xlim':[-.5, 1.5], 'xticks':[0, 1], 'xticklabels':['+ Port','- Port']}

            mean_std_res = reduce.new_filter_reduce(summary_res, reduce_key='PCA Distance',
                                                    filter_keys=['day', 'odor_valence'])

            plot.plot_results(summary_res, x_key='day', y_key='PCA Distance', loop_keys=['odor_valence','mouse'],
                              select_dict=select_dict,
                              colors=['Green'] * mice.size + ['Red'] * mice.size,
                              ax_args=summary_ax_args, plot_function=plt.plot, plot_args= summary_line_args,
                              path=figure_path, save=False)

            plot.plot_results(mean_std_res, x_key='day', y_key='PCA Distance', loop_keys='odor_valence',
                              select_dict=select_dict,
                              colors=['Green','Red'], plot_function=plt.plot,
                              ax_args=ax_args, plot_args=line_args,
                              path=figure_path, reuse=True, save=False)

            plot.plot_results(mean_std_res, x_key='day', y_key='PCA Distance', loop_keys='odor_valence',
                              select_dict=select_dict,
                              error_key='PCA Distance_sem',
                              colors=['Green','Red'], plot_function=plt.errorbar,
                              ax_args=summary_ax_args, plot_args=error_args,
                              path=figure_path, reuse=True, save=True)

        if condition.name == 'MPFC_COMPOSITE' or condition.name == 'OFC_COMPOSITE':
            pt_start_day = np.array(condition.training_start_day)
            dt_start_day = np.array(condition.naive_dt_day)
            pt_learned_day, pt_last_day = get_days_per_mouse(data_path, condition, odor_valence='PT CS+')
            dt_learned_day, dt_last_day = get_days_per_mouse(data_path, condition, odor_valence='CS+')

            # if condition.name == 'MPFC_COMPOSITE':
                # dt_learned_day[1] = 5

            if condition.name == 'OFC_COMPOSITE':
                pass

            print(dt_start_day)
            print(pt_last_day)
            print(dt_learned_day)
            print(dt_last_day)

            day_pt_start = filter.filter_days_per_mouse(summary_res, pt_start_day)
            day_pt_start = filter.filter(day_pt_start, {'odor_valence': 'PT CS+'})
            day_pt_start['title'] = np.array(['PT_Start'] * len(day_pt_start['mouse']))

            day_pt_learned = filter.filter_days_per_mouse(summary_res, pt_last_day)
            day_pt_learned = filter.filter(day_pt_learned, {'odor_valence': 'PT CS+'})
            day_pt_learned['title'] = np.array(['PT_Learned'] * len(day_pt_learned['mouse']))

            day_dt_start = filter.filter_days_per_mouse(summary_res, days_per_mouse=dt_start_day)
            day_dt_start = filter.filter(day_dt_start, {'odor_valence': 'CS+'})
            day_dt_start['title'] = np.array(['DT_Start'] * len(day_dt_start['mouse']))

            day_dt_learned = filter.filter_days_per_mouse(summary_res, days_per_mouse=dt_learned_day)
            day_dt_learned = filter.filter(day_dt_learned, {'odor_valence': 'CS+'})
            day_dt_learned['title'] = np.array(['DT_Learned'] * len(day_dt_learned['mouse']))

            day_dt_last = filter.filter_days_per_mouse(summary_res, days_per_mouse=dt_last_day)
            day_dt_last = filter.filter(day_dt_last, {'odor_valence': 'CS+'})
            day_dt_last['title'] = np.array(['DT_End'] * len(day_dt_last['mouse']))

            def _helper(plot_res):
                mean_std_res = reduce.new_filter_reduce(plot_res, reduce_key='PCA Distance',
                                                        filter_keys='title')
                titles = np.unique(plot_res['title'])
                summary_ax_args_ = copy.copy(summary_ax_args)
                summary_ax_args_.update({'xlim':[-1, len(titles)]})

                plot.plot_results(plot_res, x_key='title', y_key='PCA Distance', loop_keys='mouse',
                                  select_dict= {'title': titles},
                                  colors=['Green'] * mice.size,
                                  ax_args=summary_ax_args_, plot_function=plt.plot, plot_args= summary_line_args,
                                  path=figure_path, save=False)
                plot.plot_results(mean_std_res, x_key='title', y_key='PCA Distance',
                                  select_dict={'title': titles},
                                  colors=['Green'], plot_function=plt.plot,
                                  ax_args=ax_args, plot_args=line_args,
                                  path=figure_path, reuse=True, save=False)

                plot.plot_results(mean_std_res, x_key='title', y_key='PCA Distance',
                                  select_dict={'title': titles},
                                  error_key='PCA Distance_sem',
                                  colors=['Green'], plot_function=plt.errorbar,
                                  ax_args=summary_ax_args_, plot_args=error_args,
                                  path=figure_path, reuse=True, save=True)

            plot_res = reduce.chain_defaultdicts(day_pt_learned, day_dt_learned, copy_dict=True)
            _helper(plot_res)
            plot_res = reduce.chain_defaultdicts(day_dt_start, day_dt_learned, copy_dict=True)
            _helper(plot_res)
            plot_res = reduce.chain_defaultdicts(day_dt_learned, day_dt_last, copy_dict=True)
            _helper(plot_res)

            plot_res = reduce.chain_defaultdicts(day_pt_start, day_pt_learned, copy_dict=True)
            plot_res = reduce.chain_defaultdicts(plot_res, day_dt_start, copy_dict=True)
            plot_res = reduce.chain_defaultdicts(plot_res, day_dt_learned, copy_dict=True)
            plot_res = reduce.chain_defaultdicts(plot_res, day_dt_last, copy_dict=True)
            _helper(plot_res)
