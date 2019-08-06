import glob
import os
from collections import defaultdict

import numpy as np
from scipy import stats as sstats

import reduce
import tools.file_io
from _CONSTANTS.config import Config
from analysis import add_indices, add_time


def load_results_cv_scores(data_path):
    res = defaultdict(list)
    experiment_dirs = sorted([os.path.join(data_path, d) for d in os.listdir(data_path)])
    for exp_dir in experiment_dirs:
        data_dirs = sorted(glob.glob(os.path.join(exp_dir, '*' + Config.mat_ext)))
        config_dirs = sorted(glob.glob(os.path.join(exp_dir, '*.json')))
        for data_dir, config_dir in zip(data_dirs,config_dirs):
            config = tools.file_io.load_json(config_dir)
            data = tools.file_io.load_numpy(data_dir)

            # data is in format of experiment X time X CVfold X repeat
            res['data'].append(data)
            for key, val in config.items():
                res[key].append(val)

    for key, val in res.items():
        try:
            res[key] = np.array(val)
        except:
            arr = np.empty(len(val), dtype='object')
            for i, v in enumerate(val):
                arr[i] = v
            res[key] = arr
    return res

def load_results_train_test_scores(data_path):
    res = defaultdict(list)
    experiment_dirs = sorted([os.path.join(data_path, d) for d in os.listdir(data_path)])

    keys = ['DAQ_O_ON_F', 'DAQ_O_OFF_F', 'DAQ_W_ON_F',
            'DAQ_O_ON', 'DAQ_O_OFF', 'DAQ_W_ON',
            'TRIAL_FRAMES', 'TRIAL_PERIOD','NAME_MOUSE', 'NAME_PLANE', 'NAME_DATE',
            'decode_style', 'neurons','shuffle']
    for exp_dir in experiment_dirs:
        data_dirs = sorted(glob.glob(os.path.join(exp_dir, '*' + '.pkl')))
        config_dirs = sorted(glob.glob(os.path.join(exp_dir, '*.json')))
        for data_dir, config_dir in zip(data_dirs,config_dirs):
            config = tools.file_io.load_json(config_dir)
            cur_res = tools.file_io.load_pickle(data_dir)

            for k in keys:
                cur_res[k] = np.array([config[k]] * len(cur_res['scores']))
            reduce.chain_defaultdicts(res, cur_res)

    add_indices(res)
    add_time(res)

    for i, _ in enumerate(res['scores']):
        res['top_score'].append(np.max(res['scores'][i]))
    res['top_score'] = np.array(res['top_score'])
    return res

def add_decode_stats(res, condition, arg='max'):
    '''
    #TODO: assumes that there is only one timepoint
    :param res:
    :param condition:
    :param arg:
    :return:
    '''
    # decoding data is in format of experiment X time X CVfold X repeat
    datas = np.array(res['data'])
    O_on = res['DAQ_O_ON_F'].astype(np.int)
    O_off = res['DAQ_O_OFF_F'].astype(np.int)
    W_on = res['DAQ_W_ON_F'].astype(np.int)
    for i, data in enumerate(datas):
        data_reshaped = np.reshape(data.transpose([0, 2, 1]),
                                   [data.shape[0], data.shape[1] * data.shape[2]])
        mean_over_repeats = np.mean(data_reshaped, axis=1)
        sem = sstats.sem(data_reshaped, axis=1)

        if arg == 'max':
            max = np.max(mean_over_repeats)
        elif arg == 'mean':
            max = np.mean(mean_over_repeats)
            # max = np.mean(mean[O_off[i]: W_on[i]])
        else:
            raise ValueError('Argument for calculating summary decoding metric, max, is not known: {}'.format(arg))

        res['mean'].append(mean_over_repeats)
        res['sem'].append(sem)
        res['max'].append(max)

    res['mean'] = np.array(res['mean'])
    res['sem'] = np.array(res['sem'])
    res['max'] = np.array(res['max'])