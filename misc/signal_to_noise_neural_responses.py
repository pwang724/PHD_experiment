import os

from _CONSTANTS.config import Config
import filter
import reduce
import numpy as np
import tools.file_io as fio
from collections import defaultdict
import matplotlib.pyplot as plt
import _CONSTANTS.conditions as conditions
import plot
import seaborn as sns

conditions = [
    conditions.PIR,
    conditions.OFC,
    conditions.OFC_LONGTERM,
    conditions.OFC_COMPOSITE,
    conditions.MPFC_COMPOSITE
]

save_path = os.path.join(Config.LOCAL_FIGURE_PATH, 'MISC', 'SNR')
out = defaultdict(list)

psth = False

for condition in conditions:
    data_path = os.path.join(Config.LOCAL_EXPERIMENT_PATH, 'COUNTING', condition.name)
    res = fio.load_pickle(os.path.join(data_path, 'dict.pkl'))
    mice = np.unique(res['mouse'])
    res = filter.filter_days_per_mouse(res, len(mice) * [0])

    if psth:
        for i, d in enumerate(res['data']):
            res['data'][i] = np.mean(d, axis=1)


    for mouse in mice:
        temp = filter.filter(res, {'mouse':mouse})

        data = np.concatenate(temp['data'], axis=1)
        out['condition'].append(condition.name)
        out['mouse'].append(mouse)
        out['data'].append(data)

        x = data.reshape(data.shape[0], -1)
        max_each_cell = np.max(x, axis=1)
        mean_max = np.mean(max_each_cell)
        out['max_each_cell'].append(max_each_cell)
        out['max'].append(mean_max)

        mean_each_cell = np.mean(x, axis=1)
        mean_mean = np.mean(mean_each_cell)
        out['mean_each_cell'].append(mean_each_cell)
        out['mean'].append(mean_mean)

        #todo: 95% percentile response

for k, v in out.items():
    out[k] = np.array(v)

print(out['condition'])
print(out['mouse'])

ykey = 'max'
if psth:
    ax_args = {'yticks': np.arange(1,1.5,.1), 'ylim': [1, 1.5], 'xlim': [-1, len(conditions)]}
else:
    ax_args = {'yticks': [1, 1.5, 2], 'ylim': [1, 2], 'xlim': [-1, len(conditions)]}

swarm_args = {'marker': '.', 'size': 5, 'facecolors': 'none', 'alpha': .5, 'palette': ['black'], 'jitter': .1}
error_args = {'fmt': '.', 'capsize': 2, 'elinewidth': 1, 'markersize': 0, 'alpha': .6}

mean_std = reduce.new_filter_reduce(out, filter_keys=['condition'],reduce_key=ykey)
path, name = plot.plot_results(out, x_key='condition', y_key=ykey,
                               ax_args=ax_args,
                               plot_function=sns.stripplot,
                               plot_args=swarm_args,
                               save=False,
                               path=save_path)

names_list = [x.name for x in conditions]

save_name_str = '_PSTH' if psth else ''
for i, name in enumerate(names_list):
    save = False if i != len(names_list) - 1 else True
    plot.plot_results(mean_std, x_key='condition', y_key=ykey, error_key=ykey + '_sem',
                      select_dict={'condition':name},
                      ax_args=ax_args,
                      plot_function=plt.errorbar,
                      plot_args=error_args,
                      path=save_path, reuse=True, save=save, name_str=save_name_str)