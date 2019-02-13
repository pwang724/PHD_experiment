import copy
import glob
import os
from collections import defaultdict

import numpy as np
from scipy import stats as sstats
from sklearn import decomposition as skdecomp

import filter
import reduce
from _CONSTANTS.config import Config
from tools import file_io as fio, utils


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