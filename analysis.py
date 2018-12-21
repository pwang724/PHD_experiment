from CONSTANTS.config import Config
from CONSTANTS import conditions
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import tools.file_io
from collections import defaultdict
from scipy import stats as sstats
import plot_decode

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

# add days
def _add_days(res):
    from scipy.stats import rankdata

    list_of_dates = res['NAME_DATE']
    list_of_mice = res['NAME_MOUSE']
    xdata_new = []
    _, mouse_ixs = np.unique(list_of_mice, return_inverse=True)
    for mouse_ix in np.unique(mouse_ixs):
        mouse_dates = list_of_dates[mouse_ixs == mouse_ix]
        sorted_dates = rankdata(mouse_dates, 'dense') - 1
        xdata_new.append(sorted_dates)
    res['mouse'] = mouse_ixs
    res['day'] = np.array(xdata_new).flatten()

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
    data = np.array(res['data'])
    data_reshaped = np.reshape(data.transpose([0, 1, 3, 2]),
                               [data.shape[0], data.shape[1], data.shape[2] * data.shape[3]])
    data_mean = np.mean(data_reshaped, axis=2)
    data_sem = sstats.sem(data_reshaped, axis=2)
    res['mean'] = data_mean
    res['sem'] = data_sem

condition = conditions.OFC
data_path = os.path.join(Config.LOCAL_EXPERIMENT_PATH, 'Valence', condition.name)
save_path = os.path.join(Config.LOCAL_FIGURE_PATH, 'Valence', condition.name)

res = load_results(data_path)
_add_days(res)
_add_time(res)
_add_stats(res)

# plotting
days = np.unique(res['day'])
xkey = 'time'
ykey = 'mean'
loopkey = 'mouse'

for day in days:
    select_dict = {'neurons':10, 'day': day}
    plot_decode.plot_results(res, xkey, ykey, loopkey, select_dict, save_path)