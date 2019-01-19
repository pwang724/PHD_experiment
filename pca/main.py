import analysis
import filter
import reduce
import plot
from analysis import align_days
from tools import experiment_tools
import os
import glob
import time
from tools import utils
from behavior.behavior_analysis import get_days_per_mouse
import copy
import tools.file_io as fio
import numpy as np
from _CONSTANTS.config import Config
import _CONSTANTS.conditions as experimental_conditions
import sklearn.decomposition as skdecomp
from scipy import stats as sstats

import matplotlib.pyplot as plt
from collections import defaultdict

class PCAConfig(object):
    def __init__(self):
        self.n_principal_components = 3
        #style can be 'all', 'csp', or 'csm'
        self.style = 'csp'
        self.average = True

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

        list_of_cons = list_of_all_cons[ix][start_day:]
        list_of_data = list_of_all_data[ix][start_day:]
        for cons in list_of_cons:
            assert cons.NAME_MOUSE == mouse_name, 'Wrong mouse file!'

        odor = condition.odors[i]
        if PCAConfig.style == 'all':
            csp = None
        else:
            csp = condition.csp[i]

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

    res = defaultdict(list)
    for cons, data in zip(list_of_cons, list_of_data):
        list_of_ixs, chosen_odors = _get_trial_index(cons, pca_style, odor, csp)
        data_trial_time_cell = utils.reshape_data(data, trial_axis=0, cell_axis=2, time_axis=1,
                                                  nFrames=cons.TRIAL_FRAMES)
        list_transformed_data = []
        for ix in list_of_ixs:
            cur = data_trial_time_cell[ix]
            transformed_data = []
            for trial in range(cur.shape[0]):
                transformed_data.append(pca.transform(cur[trial]))

            transformed_data = np.stack(transformed_data, axis=0)
            list_transformed_data.append(transformed_data)

        for o, data in zip(chosen_odors, list_transformed_data):
            # data is in format of trials X time X cells
            res['data'].append(data)
            res['odor'].append(o)
            res['variance_explained'].append(variance_explained)
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
        utils.chain_defaultdicts(res, temp_res)
    return res

def analyze_pca(res):
    list_of_data = res['data']
    O_on = res['DAQ_O_ON_F'].astype(np.int)
    O_off = res['DAQ_O_OFF_F'].astype(np.int)
    W_on = res['DAQ_W_ON_F'].astype(np.int)
    variance_explained = res['variance_explained']
    for i, data in enumerate(list_of_data):
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        sem = sstats.sem(data, axis=0)
        score = np.max(mean[O_on[i]:W_on[i],:], axis=0) - np.mean(mean[:O_on[i]], axis=0)
        distance = np.linalg.norm(score * variance_explained[i])
        res['mean'].append(mean)
        res['std'].append(std)
        res['sem'].append(sem)
        res['score'].append(score)
        res['distance'].append(distance)
    for key, val in res.items():
        res[key] = np.array(val)

def normalize_per_mouse(res, key):
    mice, mouse_ix = np.unique(res['mouse'], return_inverse=True)
    for i, mouse in enumerate(mice):
        ix = mouse_ix == i
        data = res[key][ix]
        data /= np.max(data)
        res[key][ix] = data


condition = experimental_conditions.OFC_LONGTERM
config = PCAConfig()
config.style = 'all'
config.n_principal_components = 5
config.average = False
data_path = os.path.join(Config.LOCAL_DATA_PATH, Config.LOCAL_DATA_TIMEPOINT_FOLDER, condition.name)
save_path = os.path.join(Config.LOCAL_EXPERIMENT_PATH, 'PCA_' + config.style, condition.name)
figure_path = os.path.join(Config.LOCAL_FIGURE_PATH, 'PCA', config.style, condition.name)
do_PCA(condition, config, data_path, save_path)

res = load_pca(save_path)
analysis.add_indices(res)
analysis.add_time(res)
analysis.add_odor_value(res, condition)
analyze_pca(res)

summary_res = defaultdict(list)
valences = np.unique(res['odor_valence'])
mice = np.unique(res['mouse'])
for valence in valences:
    for mouse in mice:
        temp = filter.filter(res, filter_dict={'odor_valence': valence, 'mouse': mouse})
        reduced_temp = reduce.filter_reduce(temp, filter_key='day', reduce_key='distance')
        utils.chain_defaultdicts(summary_res, reduced_temp)

normalize_per_mouse(summary_res, 'distance')
for mouse in mice:
    plot.plot_results(summary_res, x_key='day', y_key='distance', loop_keys='odor_valence',
                      select_dict={'mouse':mouse},
                      path=figure_path)

# print(res['score'])
# data = res['mean']
# x_data = data[:,:,0]
# y_data = data[:,:,1]
# z_data = data[:,:,2]
# plt.plot(x_data.transpose(),y_data.transpose())
# plt.show()