import analysis
import behavior.behavior_analysis
import filter
import reduce
import plot
from tools import experiment_tools
import os
import glob
import time
from tools import utils
import copy
import tools.file_io as fio
import numpy as np
from _CONSTANTS.config import Config
import _CONSTANTS.conditions as experimental_conditions
import sklearn.decomposition as skdecomp
from scipy import stats as sstats
from behavior.behavior_analysis import get_days_per_mouse

import matplotlib.pyplot as plt
from collections import defaultdict

class PCAConfig(object):
    def __init__(self):
        self.n_principal_components = 3
        #style can be 'all', 'csp', or 'csm'
        self.style = 'csp'
        self.average = True
        self.shuffle_iterations = 100

def do_PCA(condition, PCAConfig, data_path, save_path):
    '''
    Run PCA experiment. For each mouse, data should be in format of (trials * time) X neurons,
    stacked for all odor conditions

    :param condition:
    :param PCAConfig:
    :param data_path:
    :param save_path:
    :return:
    '''
    data_pathnames = sorted(glob.glob(os.path.join(data_path, '*' + Config.mat_ext)))
    config_pathnames = sorted(glob.glob(os.path.join(data_path, '*' + Config.cons_ext)))

    list_of_all_data = np.array([Config.load_mat_f(d) for d in data_pathnames])
    list_of_all_cons = np.array([Config.load_cons_f(d) for d in config_pathnames])
    mouse_names_per_file = np.array([cons.NAME_MOUSE for cons in list_of_all_cons])
    mouse_names, list_of_mouse_ix = np.unique(mouse_names_per_file, return_inverse=True)

    if mouse_names.size != len(condition.paths):
        raise ValueError("res has {0:d} mice, but filter has {1:d} mice".
                         format(mouse_names.size, len(condition.paths)))

    for i, mouse_name in enumerate(mouse_names):
        ix = mouse_name == mouse_names_per_file
        if hasattr(condition, 'training_start_day'):
            start_day = condition.training_start_day[i]
        else:
            start_day = 0

        start_day = 0
        list_of_cons = list_of_all_cons[ix][start_day:]
        list_of_data = list_of_all_data[ix][start_day:]
        for cons in list_of_cons:
            assert cons.NAME_MOUSE == mouse_name, 'Wrong mouse file!'

        if hasattr(condition, 'odors'):
            odor = condition.odors[i]
            csp = condition.csp[i]
        else:
            odor = condition.dt_odors[i] + condition.pt_csp[i]
            csp = condition.dt_csp[i] + condition.pt_csp[i]
        #
        # if PCAConfig.style == 'all':
        #     csp = None
        # else:
        #     csp = condition.csp[i]

        name = list_of_cons[0].NAME_MOUSE
        out, variance_explained = PCA(list_of_cons, list_of_data, odor, csp, PCAConfig)
        fio.save_pickle(save_path=save_path, save_name=name, data=out)
        print("Analyzed: {}. Variance explained: {}".format(name, variance_explained))

def PCA(list_of_cons, list_of_data, odor, csp, pca_config):
    def _get_trial_index(cons, pca_style, odor, csp):
        if pca_style == 'all':
            chosen_odors = odor
        elif pca_style == 'csp':
            chosen_odors = csp
        elif pca_style == 'csm':
            chosen_odors = [x for x in odor if not np.isin(x, csp)]
        else:
            raise ValueError('pca style is not recognized {}'.format(pca_style))
        labels = cons.ODOR_TRIALS
        list_of_ixs = []
        for o in chosen_odors:
            trial_ixs = np.isin(labels, o)
            list_of_ixs.append(trial_ixs)
        return list_of_ixs, chosen_odors

    pca_nPCs = pca_config.n_principal_components
    pca_style = pca_config.style
    pca_average = pca_config.average

    list_of_reshaped_data = []
    for cons, data in zip(list_of_cons, list_of_data):
        list_of_ixs, chosen_odors = _get_trial_index(cons, pca_style, odor, csp)
        data_cell_trial_time = utils.reshape_data(data, trial_axis=1, cell_axis=0, time_axis=2,
                                                  nFrames=cons.TRIAL_FRAMES)
        for ix in list_of_ixs:
            cur = data_cell_trial_time[:,ix,:]
            if pca_average:
                cur = np.mean(cur, axis=1, keepdims=True)
            cur_reshaped = cur.reshape([cur.shape[0],
                                        cur.shape[1] * cur.shape[2]])
            list_of_reshaped_data.append(cur_reshaped.transpose())
    pca_data = np.concatenate(list_of_reshaped_data, axis=0)

    pca = skdecomp.PCA(n_components=pca_nPCs)
    pca.fit(pca_data)
    variance_explained = pca.explained_variance_ratio_
    components = pca.components_
    variance = pca.explained_variance_

    res = defaultdict(list)
    for cons, data in zip(list_of_cons, list_of_data):
        list_of_ixs, chosen_odors = _get_trial_index(cons, pca_style, odor, csp)
        data_trial_time_cell = utils.reshape_data(data, trial_axis=0, cell_axis=2, time_axis=1,
                                                  nFrames=cons.TRIAL_FRAMES)
        for i, ix in enumerate(list_of_ixs):
            if any(ix):
                o = chosen_odors[i]
                cur = data_trial_time_cell[ix]
                transformed_data = []
                for trial in range(cur.shape[0]):
                    transformed_data.append(pca.transform(cur[trial]))
                transformed_data = np.stack(transformed_data, axis=0)

                # data is in format of trials X time X cells
                res['data'].append(transformed_data)
                res['odor'].append(o)
                res['variance_explained'].append(variance_explained)
                # res['components'].append(components)
                # res['variance'].append(variance)
                for key, val in pca_config.__dict__.items():
                    res[key].append(val)
                for key, val in cons.__dict__.items():
                    if type(val) != list and type(val) != np.ndarray:
                        res[key].append(val)

    for key, val in res.items():
        res[key] = np.array(val)
    return res, variance_explained

def load_pca(save_path):
    res = defaultdict(list)
    experiment_dirs = sorted([os.path.join(save_path, d) for d in os.listdir(save_path)])
    for d in experiment_dirs:
        temp_res = fio.load_pickle(d)
        reduce.chain_defaultdicts(res, temp_res)
    return res

def shuffle_data(res, shuffle_keys):
    out = copy.deepcopy(res)
    data = out['data']

    if isinstance(shuffle_keys, str):
        shuffle_keys = [shuffle_keys]
    unique_combinations, combination_ixs = filter.retrieve_unique_entries(out, shuffle_keys)

    for ixs in combination_ixs:
        current_data = data[ixs]
        trials_per_condition = [x.shape[0] for x in current_data]
        concatenated_data = np.concatenate(current_data, axis=0)
        shuffled_ixs = np.random.permutation(concatenated_data.shape[0])
        shuffled_ixs_per_condition = np.split(shuffled_ixs, np.cumsum(trials_per_condition))[:-1]
        for i in range(len(shuffled_ixs_per_condition)):
            assert(trials_per_condition[i] == len(shuffled_ixs_per_condition[i])), 'bad partitioning of shuffled ixs'

        new_data = np.zeros(len(ixs), dtype=object)
        for i, ix in enumerate(shuffled_ixs_per_condition):
            new_data[i] = concatenated_data[ix]
        out['data'][ixs] = new_data
    return out

def easy_shuffle(res, shuffle_keys, n):
    shuffled_res = defaultdict(list)
    for i in range(n):
        temp = shuffle_data(res, shuffle_keys=shuffle_keys)
        reduce.chain_defaultdicts(shuffled_res, temp)
    shuffled_res['shuffle'] = np.ones(len(shuffled_res['odor']), dtype=int)
    reduce.chain_defaultdicts(res, shuffled_res)
    return res


def analyze_pca(res):
    '''
    first average across all trials, then
    :param res:
    :return:
    '''
    list_of_data = res['data']
    O_on = res['DAQ_O_ON_F'].astype(np.int)
    O_off = res['DAQ_O_OFF_F'].astype(np.int)
    W_on = res['DAQ_W_ON_F'].astype(np.int)
    variance_explained = res['variance_explained']
    for i, data in enumerate(list_of_data):
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        sem = sstats.sem(data, axis=0)
        distances = mean * variance_explained[i]
        distance = np.max(distances[O_on[i]:W_on[i]]) - np.mean(distances[:O_on[i]])

        res['mean'].append(mean)
        res['std'].append(std)
        res['sem'].append(sem)
        res['PCA Distance'].append(distance)

        # for j in range(data.shape[0]):
        #     for k in range(data.shape[1]):
        #         data[j,k,] *= variance_explained[i]
        #
        # distance = np.linalg.norm(data, axis=2)
        # mean = np.mean(distance, axis=0)
        # std = np.std(distance, axis=0)
        # sem = sstats.sem(distance, axis=0)
        # distance_score = np.max(mean[O_on[i]:W_on[i]]) - np.mean(mean[:O_on[i]])
        # res['mean'].append(mean)
        # res['std'].append(std)
        # res['sem'].append(sem)
        # res['PCA Distance'].append(distance_score)
    for key, val in res.items():
        res[key] = np.array(val)

def normalize_per_mouse(res, key):
    mice, mouse_ix = np.unique(res['mouse'], return_inverse=True)
    for i, mouse in enumerate(mice):
        ix = mouse_ix == i
        data = res[key][ix]
        max = np.max(data)
        min = np.min(data)
        data = (data - min)/(max - min)
        res[key][ix] = data

        res[key + '_sem'][ix] = res[key + '_sem'][ix] / (max-min)
        res[key + '_std'][ix] = res[key + '_std'][ix] / (max-min)

experiments = [
    'shuffle',
]

condition = experimental_conditions.OFC_LONGTERM
config = PCAConfig()
config.style = 'csp'
config.n_principal_components = 5
config.shuffle_iterations = 50
config.average = False

data_path = os.path.join(Config.LOCAL_DATA_PATH, Config.LOCAL_DATA_TIMEPOINT_FOLDER, condition.name)
save_path = os.path.join(Config.LOCAL_EXPERIMENT_PATH, 'PCA_' + config.style, condition.name)
figure_path = os.path.join(Config.LOCAL_FIGURE_PATH, 'PCA', config.style, condition.name)

EXPERIMENT = True
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
        learned_day_per_mouse, last_day_per_mouse = get_days_per_mouse(data_path, condition)
        print(learned_day_per_mouse)
        analysis.add_indices(res)
        analysis.add_time(res)
        analysis.add_aligned_days(res, last_days=last_day_per_mouse, learned_days=learned_day_per_mouse)
        analysis.add_odor_value(res, condition)
        if hasattr(condition, 'training_start_day'):
            list_of_start_days = condition.training_start_day
        else:
            list_of_start_days = [0] * np.unique(res['mouse']).size
        list_of_days = []
        for i, start_day in enumerate(list_of_start_days):
            list_of_days.append(np.arange(start_day, last_day_per_mouse[i]+1))
        res= filter.filter_days_per_mouse(res, list_of_days)
        res = filter.filter(res, {'odor_valence':['CS+', 'CS-', 'PT CS+']})
        res['shuffle'] = np.zeros(res['odor'].shape, dtype=int)

        res = easy_shuffle(res, shuffle_keys=shuffle_keys, n = config.shuffle_iterations)
        analyze_pca(res)
        summary_res = reduce.new_filter_reduce(res, ['mouse','day','odor_valence','shuffle'], reduce_key='PCA Distance')
        normalize_per_mouse(summary_res, 'PCA Distance')

        lick_res = behavior.behavior_analysis.get_licks_per_day(data_path, condition)
        analysis.add_odor_value(lick_res, condition)
        lick_res = filter.filter_days_per_mouse(lick_res, list_of_days)
        lick_res = filter.filter(lick_res, {'odor_valence':['CS+', 'CS-', 'PT CS+']})
        lick_res = reduce.new_filter_reduce(lick_res, ['odor_valence','day','mouse'], reduce_key='lick_boolean')

        valences = np.unique(res['odor_valence'])
        mice = np.unique(res['mouse'])
        colors = ['Green', 'Red']
        for i, mouse in enumerate(mice):
            plot.plot_results(lick_res, x_key='day', y_key='lick_boolean', loop_keys='odor_valence',
                              select_dict={'mouse':mouse},
                              colors = colors,
                              ax_args=ax_args, plot_args=behavior_line_args,
                              path=figure_path, save=False, sort=True)

            plot.plot_results(summary_res, x_key='day', y_key='PCA Distance',
                              select_dict={'mouse':mouse, 'shuffle': 1},
                              colors=['gray']*5,
                              ax_args=ax_args, plot_args=line_args,
                              path=figure_path, reuse=True, save=False)

            plot.plot_results(summary_res, x_key='day', y_key='PCA Distance', error_key='PCA Distance_sem',
                              select_dict={'mouse':mouse, 'shuffle': 1},
                              colors=['gray']*5, plot_function=plt.fill_between,
                              ax_args=ax_args, plot_args=fill_args,
                              path=figure_path, reuse=True, save=False)

            plot.plot_results(summary_res, x_key='day', y_key='PCA Distance', loop_keys='odor_valence',
                              select_dict={'mouse':mouse, 'shuffle':0},
                              colors=colors,
                              ax_args=ax_args, plot_args=line_args,
                              path=figure_path, reuse=True, save=True)



            plot.plot_results(summary_res, x_key='day', y_key='PCA Distance',
                              select_dict={'mouse':mouse, 'shuffle': 1},
                              colors=['gray']*5,
                              ax_args=ax_args, plot_args=line_args,
                              path=figure_path, reuse=False, save=False)

            plot.plot_results(summary_res, x_key='day', y_key='PCA Distance', error_key='PCA Distance_sem',
                              select_dict={'mouse':mouse, 'shuffle': 1},
                              colors=['gray']*5, plot_function=plt.fill_between,
                              ax_args=ax_args, plot_args=fill_args,
                              path=figure_path, reuse=True, save=False)

            plot.plot_results(summary_res, x_key='day', y_key='PCA Distance', loop_keys='odor_valence',
                              select_dict={'mouse':mouse, 'shuffle':0, 'odor_valence': 'CS+'},
                              colors=colors,
                              ax_args=ax_args, plot_args=line_args,
                              path=figure_path, reuse=True, save=True)

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
            learned_day_per_mouse, last_day_per_mouse = get_days_per_mouse(data_path, condition)
            filtered_res = filter.filter_days_per_mouse(summary_res, list(zip(learned_day_per_mouse, last_day_per_mouse)))
            analysis.add_naive_learned(filtered_res, learned_day_per_mouse, last_day_per_mouse, 'Learned','Over-trained')
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