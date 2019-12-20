import copy
from collections import defaultdict
import numpy as np
import reduce
import plot
from format import *
import filter
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

def _get_power_1(data):
    s = 25
    e = 44
    data = np.mean(data, axis=1)
    data = data - np.mean(data[:, :s], axis=1, keepdims=True)
    y_ = np.mean(data[:, s:e], axis=1)
    ix = y_ > 0.0
    y = np.mean(np.abs(data[ix, :]), axis=0)
    out = np.max(y[s:e])
    return out

def get_cory(res, temp_res, exclude_naive, figure_path):
    res = copy.copy(res)
    if exclude_naive:
        pass
          # get rid of naive responses, only for OFC
    res = filter.filter(res, {'odor_valence': ['CS+']})  # filter only CS+ responses
    names, list_of_ixs = filter.retrieve_unique_entries(res, ['mouse', 'odor_standard'])
    out = defaultdict(list)

    for i, ixs in enumerate(list_of_ixs):
        mouse = names[i][0]
        odor_standard = names[i][1]

        out['mouse'].append(mouse)
        out['odor_standard'].append(odor_standard)
        out['odor_valence'].append(odor_standard[:-1])

        d = res['data'][ixs]
        print([x.shape for x in d])
        catdata = np.concatenate(d, axis=1)
        trials = catdata.shape[1]
        window = 9
        x = []
        for i in range(trials - window):
            temp = _get_power_1(catdata[:, i:i + window, :])
            if i == 0:
                for _ in range(1 + window // 2):
                    x.append(temp)
            else:
                x.append(temp)
        while len(x) != trials:
            x.append(temp)
        x = savgol_filter(x, 3, 0)
        out['power'].append(np.array(x))
        out['trials'].append(np.arange(trials))

        ix_lick = np.logical_and(temp_res['odor_valence'] == odor_standard[:-1], temp_res['mouse'] == mouse)
        out['lick'].append(temp_res['boolean_smoothed'][ix_lick][0])
    for k, v in out.items():
        out[k] = np.array(v)

    out = reduce.new_filter_reduce(out, filter_keys=['mouse', 'odor_valence'], reduce_key='power')
    for i, power in enumerate(out['power']):
        min = np.min(power)
        max = np.max(power)
        out['power'][i] = (power - min) / (max - min)

    trace_args = {'alpha': .5, 'linewidth': 1}
    trace_args_bhv = {'alpha': .5, 'linewidth': 1, 'linestyle':':'}
    out['lick'] /= 100
    ax_lim = {'xlim':[0,50], 'yticks':[0, .5, 1]}
    plot.plot_results(out, x_key='trials', y_key='power', loop_keys='mouse', select_dict={'mouse':1},
                      plot_args=trace_args, path=figure_path,
                      save=False, colors=['k'], ax_args = ax_lim)
    plot.plot_results(out, x_key='trials', y_key='lick', loop_keys='mouse', select_dict={'mouse':1},
                      plot_args=trace_args_bhv, path=figure_path,
                      reuse=True, save=True, colors=['k'], ax_args= ax_lim, twinax=True)

    plot.plot_results(out, x_key='trials', y_key='power', loop_keys='mouse',
                      plot_args=trace_args, path=figure_path,
                      save=True, ax_args = ax_lim)
    plot.plot_results(out, x_key='trials', y_key='lick', loop_keys='mouse',
                      plot_args=trace_args_bhv, path=figure_path,
                      save=True, ax_args= ax_lim, twinax=True)

    for i, pow in enumerate(out['power']):
        pow[:10] = 0
        half_pow = np.argwhere(pow > 0.5)[0][0]
        half_lick = np.argwhere(out['lick'][i] > 0.5)[0][0]
        print(half_lick, half_pow)
        out['half_power'].append(half_pow)
        out['half_lick'].append(half_lick)
    for k, v in out.items():
        out[k] = np.array(v)

    from sklearn import linear_model
    from sklearn.metrics import mean_squared_error, r2_score
    regr = linear_model.LinearRegression()
    regr.fit(out['half_lick'].reshape(-1,1), out['half_power'].reshape(-1,1))

    y_pred = regr.predict(out['half_lick'].reshape(-1,1))
    score = regr.score(out['half_lick'].reshape(-1,1), out['half_power'].reshape(-1,1))


    lim = [10, 50]
    ax_lim = {'xlim': lim, 'ylim': lim}
    a, b= plot.plot_results(out, x_key='half_lick', y_key='half_power',
                      plot_args=scatter_args, plot_function=plt.scatter, path=figure_path,
                      save=False, ax_args= ax_lim)

    plt.plot(lim, lim, '--', color='red', alpha=.5, linewidth=1)
    plt.text(25, lim[1], 'R = {:.2f}'.format(score), fontsize=5)
    plot._easy_save(a, b, pdf=True)

