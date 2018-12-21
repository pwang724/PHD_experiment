from CONSTANTS.config import Config
from CONSTANTS import conditions
import tools.plots as pt
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import tools.file_io
from collections import defaultdict
from scipy import stats as sstats

def load_all_results(data_path):
    res = defaultdict(list)
    experiment_dirs = [os.path.join(data_path, d) for d in os.listdir(data_path)]
    for i, exp_dir in enumerate(experiment_dirs):
        config = tools.file_io.load_json(os.path.join(exp_dir, Config.DECODE_CONFIG_JSON))
        mice_dirs = [os.path.join(exp_dir, d) for d in os.listdir(exp_dir)
                     if os.path.isdir(os.path.join(exp_dir, d))]
        for j, mice_dir in enumerate(mice_dirs):
            data_dirs = glob.glob(os.path.join(mice_dir, '*' + Config.mat_ext))
            for k, data_dir in enumerate(data_dirs):
                data = Config.load_mat_f(data_dir)
                res['mouse'].append(j)
                res['day'].append(k)
                res['data'].append(data)

                for key, val in config.items():
                    res[key].append(val)
    for key, val in res.items():
        res[key] = np.array(val)
    return res

condition = conditions.OFC
data_path = os.path.join(Config.LOCAL_EXPERIMENT_PATH, 'Valence', condition.name)

res = load_all_results(data_path)

#analysis
data = np.array(res['data'])
#TODO: ask Fabio if joining CV scores and repetitions is legitimate
data_reshaped = np.reshape(data.transpose([0,1,3,2]), [data.shape[0], data.shape[1],data.shape[2]*data.shape[3]])
data_mean = np.mean(data_reshaped, axis=2)
data_sem = sstats.sem(data_reshaped, axis=2)
res['mean'] = data_mean
res['sem'] = data_sem