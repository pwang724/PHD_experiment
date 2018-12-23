from CONSTANTS.config import Config
import numpy as np
import os
import glob
import tools.file_io
from collections import defaultdict
from scipy import stats as sstats


def load_results(data_path):
    res = defaultdict(list)
    experiment_dirs = [os.path.join(data_path, d) for d in os.listdir(data_path)]
    for exp_dir in experiment_dirs:
        data_dirs = glob.glob(os.path.join(exp_dir, '*' + Config.mat_ext))
        config_dirs = glob.glob(os.path.join(exp_dir, '*.json'))
        for data_dir, config_dir in zip(data_dirs,config_dirs):
            config = tools.file_io.load_json(config_dir)
            data = tools.file_io.load_numpy(data_dir)

            # data is in format of experiment X time X CVfold X repeat
            res['data'].append(data)
            for key, val in config.items():
                res[key].append(val)

    for key, val in res.items():
        res[key] = np.array(val)
    return res

def analyze_results(res):
    _add_days(res)
    _add_time(res)
    _add_stats(res)

# add days
def _add_days(res):
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

# add time
def _add_time(res):
    data = np.array(res['data'])
    for i in range(data.shape[0]):
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

#add relevant stats
def _add_stats(res):
    # data is in format of experiment X time X CVfold X repeat
    # TODO: ask Fabio if joining CV scores and repetitions is legitimate
    datas = np.array(res['data'])
    O_on = res['DAQ_O_ON_F'].astype(np.int)
    W_on = res['DAQ_W_ON_F'].astype(np.int)
    for i, data in enumerate(datas):
        data_reshaped = np.reshape(data.transpose([0, 2, 1]),
                                   [data.shape[0], data.shape[1] * data.shape[2]])
        mean = np.mean(data_reshaped, axis=1)
        max = np.max(mean[O_on[i]: W_on[i]])
        res['mean'].append(mean)
        res['sem'].append(sstats.sem(data_reshaped, axis=1))
        res['max'].append(max)

    res['mean'] = np.array(res['mean'])
    res['sem'] = np.array(res['sem'])
    res['max'] = np.array(res['max'])

