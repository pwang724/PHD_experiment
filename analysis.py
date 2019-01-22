from _CONSTANTS.config import Config
import numpy as np
import os
import glob
import tools.file_io
from collections import defaultdict
from scipy import stats as sstats

def load_all_cons(data_path):
    res = defaultdict(list)

    config_pathnames = sorted(glob.glob(os.path.join(data_path, '*' + Config.cons_ext)))
    for i, config_pn in enumerate(config_pathnames):
        cons = Config.load_cons_f(config_pn)
        for key, val in cons.__dict__.items():
            if isinstance(val, list):
                res[key].append(np.array(val))
            else:
                res[key].append(val)
    for key, val in res.items():
        if key == 'DAQ_DATA':
            arr = np.empty(len(val), dtype='object')
            for i, v in enumerate(val):
                arr[i] = v
            res[key] = arr
        else:
            res[key] = np.array(val)
    return res

def load_results(data_path):
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

def analyze_results(res, condition, arg='different'):
    add_indices(res)
    add_time(res)
    add_decode_stats(res, condition, arg)

def add_indices(res):
    from scipy.stats import rankdata

    list_of_dates = res['NAME_DATE']
    list_of_mice = res['NAME_MOUSE']
    days = np.zeros_like(list_of_dates, dtype=int)
    _, mouse_ixs = np.unique(list_of_mice, return_inverse=True)
    for mouse_ix in np.unique(mouse_ixs):
        ixs = mouse_ixs == mouse_ix
        mouse_dates = list_of_dates[ixs]
        sorted_dates = rankdata(mouse_dates, 'dense') - 1
        days[ixs] = sorted_dates
    res['mouse'] = mouse_ixs
    res['day'] = days

def add_time(res):
    nExperiments = res['DAQ_O_ON'].size
    for i in range(nExperiments):
        nF = res['TRIAL_FRAMES'][i]
        period = res['TRIAL_PERIOD'][i]
        O_on = res['DAQ_O_ON'][i]
        O_off = res['DAQ_O_OFF'][i]
        W_on = res['DAQ_W_ON'][i]
        time = np.arange(0, nF) * period - O_on
        xticks = np.asarray([O_on, O_off, W_on]) - O_on
        xticks = np.round(xticks, 1)
        res['time'].append(time)
        res['xticks'].append(xticks)
    res['time'] = np.array(res['time'])
    res['xticks'] = np.array(res['xticks'])

def add_aligned_days(res, last_days, learned_days):
    list_of_days = learned_days
    new_days = np.zeros_like(res['day'])
    mice, ix = np.unique(res['mouse'], return_inverse=True)
    for i, mice in enumerate(mice):
        current_ix = ix == i
        days = res['day'][current_ix]
        new_days[current_ix] = days - list_of_days[i]
    res['day_aligned'] = new_days


#add relevant stats
def add_decode_stats(res, condition, arg='different'):
    '''

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
        mean = np.mean(data_reshaped, axis=1)
        sem = sstats.sem(data_reshaped, axis=1)

        if arg == 'different':
            if condition.name == 'PIR' or condition.name == 'PIR_NAIVE':
                max = np.max(mean[O_on[i]: W_on[i]])
            else:
                max = np.mean(mean[O_off[i]: W_on[i]])
        elif arg == 'same':
            max = np.mean(mean[O_off[i]: W_on[i]])
        else:
            raise ValueError('Argument for calculating summary decoding metric, max, is not known: {}'.format(arg))
        res['mean'].append(mean)
        res['sem'].append(sem)
        res['max'].append(max)

    res['mean'] = np.array(res['mean'])
    res['sem'] = np.array(res['sem'])
    res['max'] = np.array(res['max'])


