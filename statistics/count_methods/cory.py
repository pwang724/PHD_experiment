import copy
from collections import defaultdict
import numpy as np
import reduce
import plot
from format import *
import filter
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import scipy.signal

# TODO: have option for no behavioral plotting
# TODO: normalize trial number

resample = True
resample_trials = 100
ykey_b = 'trials_with_licks'
xkey_b = 'behavioral_trials'
trace_args = {'alpha': .5, 'linewidth': .75}
trace_args_bhv = {'alpha': .5, 'linewidth': .75, 'linestyle': 'dashed'}

def _power(data, s, e, excitatory = True):
    data = np.mean(data, axis=1)
    data = data - np.mean(data[:, :s], axis=1, keepdims=True)
    y_ = np.mean(data[:, s:e], axis=1)
    if excitatory:
        ix = y_ > 0.0
        y = np.mean(data[ix, :], axis=0)
        out = np.max(y[s:e])
    else:
        ix = y_ < 0.0
        y = np.mean(data[ix, :], axis=0)
        out = np.min(y[s:e])
    return out

def _windowed_analysis(neural_res, behavior_res, window = 13, smooth_length = 3, excitatory = True, valence = 'CS+'):
    def _moving_window(catdata, window):
        n_trials = catdata.shape[1]
        x = []
        for i in range(n_trials - window):
            temp = _power(catdata[:, i:i + window, :], s, e, excitatory)
            if i == 0:
                for _ in range(1 + window // 2):
                    x.append(temp)
            else:
                x.append(temp)
        while len(x) != n_trials:
            x.append(temp)
        return x

    neural_res = copy.copy(neural_res)
    neural_res = filter.filter(neural_res, {'odor_valence': [valence]})  # filter only CS+ responses
    names, list_of_ixs = filter.retrieve_unique_entries(neural_res, ['mouse', 'odor_standard'])
    out = defaultdict(list)

    for i, ixs in enumerate(list_of_ixs):
        mouse = names[i][0]
        odor_standard = names[i][1]
        out['mouse'].append(mouse)
        out['odor_standard'].append(odor_standard)
        out['odor_valence'].append(odor_standard[:-1])

        #neural analysis
        if len(np.unique(neural_res['DAQ_O_ON_F'])) > 1:
            print('Odor times not the same')

        if len(np.unique(neural_res['DAQ_W_ON_F'])) > 1:
            print('Water times not the same')

        s = np.min(neural_res['DAQ_O_ON_F'][ixs])
        e = np.min(neural_res['DAQ_W_ON_F'][ixs])
        d = neural_res['data'][ixs]
        catdata = np.concatenate(d, axis=1)
        n_trials = catdata.shape[1]
        x = _moving_window(catdata, window)
        x = savgol_filter(x, smooth_length, 0)

        if excitatory:
            out['power'].append(np.array(x))
        else:
            out['power'].append(np.array(-1 * x))
        out['trials'].append(np.arange(n_trials))

        #behavior analysis
        ix_lick = np.logical_and(behavior_res['odor_standard'] == odor_standard, behavior_res['mouse'] == mouse)
        assert np.sum(ix_lick) == 1, '{},{},{}'.format(odor_standard, mouse, ix_lick)
        y = behavior_res['boolean_smoothed'][ix_lick][0]
        out[ykey_b].append(y)

        # both
        temp = (x - np.min(x)) / (np.max(x) - np.min(x))
        temp[:10] = 0
        half_pow = np.argwhere(temp > 0.5)[0][0]
        if np.any(y>50):
            half_lick = np.argwhere(y > 50)[0][0]
        else:
            half_lick = -1
        out['half_power'].append(half_pow)
        out['half_lick'].append(half_lick)

    for k, v in out.items():
        out[k] = np.array(v)

    #average by odor
    out = reduce.new_filter_reduce(out, filter_keys=['mouse', 'odor_valence'], reduce_key='power')
    for i, power in enumerate(out['power']):
        min = np.min(power)
        max = np.max(power)
        out['power'][i] = (power - min) / (max - min)

    temp = reduce.new_filter_reduce(out, filter_keys=['mouse', 'odor_valence'], reduce_key=ykey_b)
    for i in range(len(out['power'])):
        bhv = temp[ykey_b][i]
        neural = out['power'][i]
        if len(neural) > len(bhv): # when there is naive day but no training / behavioral data
            bhv_trials = np.arange(len(neural) - len(bhv), len(neural))
        else:
            bhv_trials = np.arange(len(neural))
        out[xkey_b].append(bhv_trials)
    out[xkey_b] = np.array(out[xkey_b])

    # resample
    # f = lambda a: ((resample_trials - a[0]) / (a[-1] - a[0])) * (a - a[0]) + a[0]
    # for i in range(len(out[xkey_b])):
    #     out[xkey_b][i] = f(out[xkey_b][i])
    #     out['trials'][i] = f(out['trials'][i])
    return out

def main(neural_res, behavior_res, figure_path, excitatory=True, valence='CS+'):
    out = _windowed_analysis(neural_res, behavior_res, excitatory=excitatory, valence=valence)
    _plot_power_each_mouse(out, figure_path, excitatory=excitatory, valence=valence)
    _plot_power_every_mouse(out, figure_path, excitatory, valence)
    _plot_power_mean_sem(out, figure_path, excitatory, valence)

def _plot_power_each_mouse(res, figure_path, excitatory, valence):
    res[ykey_b] /= 100
    ax_lim = {'yticks':[0, .5, 1],'ylim':[0, 1.05]}
    name_str = '_E' if excitatory else '_I'
    name_str += '_' + valence
    for mouse in np.unique(res['mouse']):
        path, name = plot.plot_results(res, x_key=xkey_b, y_key=ykey_b, loop_keys='mouse',
                           select_dict={'mouse':mouse},
                           plot_args=trace_args_bhv,
                           path=figure_path,
                           save=False,
                           colors=['k'],
                           ax_args= ax_lim)

        plot.plot_results(res, x_key='trials', y_key='power', loop_keys='mouse',
                          select_dict={'mouse':mouse},
                          plot_args=trace_args,
                          path=figure_path,
                          reuse=True, save=False,
                          colors=['k'],
                          ax_args = ax_lim)
        plt.gca().set_xlim(left=0)
        plt.gca().set_ylim(bottom=-0.05)
        plt.legend(['neural','behavior'], frameon=False, loc=0)
        plot._easy_save(path, name + name_str)

def _plot_power_every_mouse(res, figure_path, excitatory, valence):
    ax_lim = {'yticks':[0, .5, 1],'ylim':[0, 1.05]}
    name_str = '_E' if excitatory else '_I'
    name_str += '_' + valence
    # plot.plot_results(res, x_key=xkey_b, y_key=ykey_b, loop_keys='mouse',
    #                   plot_args=trace_args_bhv, path=figure_path,
    #                   save=False, reuse=False,
    #                   ax_args = ax_lim,
    #                   name_str=name_str)
    plt.gca().set_xlim(left=0)
    plt.gca().set_ylim(bottom=-0.05)
    plot.plot_results(res, x_key='trials', y_key='power', loop_keys='mouse',
                      plot_args=trace_args, path=figure_path,
                      save=True, reuse=False,
                      ax_args = ax_lim,
                      legend= False,
                      name_str=name_str)

def _plot_power_mean_sem(res, figure_path, excitatory, valence):
    color = 'red' if valence == 'CS-' else 'green'
    res.pop('power_sem')
    ax_lim = {'yticks': [0, .5, 1], 'ylim': [0, 1.05], 'xticks':np.arange(0, 100, 25), 'xlim':[0, 85]}
    name_str = '_E' if excitatory else '_I'
    name_str += '_' + valence


    mean_std_power = reduce.new_filter_reduce(res, filter_keys='odor_valence', reduce_key='power', regularize='max')
    mean_std_bhv = reduce.new_filter_reduce(res, filter_keys='odor_valence', reduce_key=ykey_b, regularize='max')
    plot.plot_results(mean_std_bhv, x_key=xkey_b, y_key=ykey_b,
                      plot_args=trace_args, path=figure_path, rect=(.2, .25, .6, .6),
                      save=False, ax_args=ax_lim)
    plot.plot_results(mean_std_power, x_key='trials', y_key='power',
                      colors=color,
                      plot_args=trace_args, path=figure_path,
                      reuse=True, save=False,
                      ax_args=ax_lim)
    plot.plot_results(mean_std_bhv, x_key=xkey_b, y_key=ykey_b, error_key=ykey_b + '_sem',
                      plot_function=plt.fill_between,
                      plot_args=fill_args,
                      path=figure_path,
                      reuse=True, save=False,
                      ax_args=ax_lim)
    plt.legend(['behavior','neural'], frameon=False)
    plot.plot_results(mean_std_power, x_key='trials', y_key='power', error_key='power_sem',
                      plot_function=plt.fill_between,
                      plot_args=fill_args,
                      colors=color,
                      path=figure_path,
                      reuse=True, save=True, twinax=True,
                      ax_args=ax_lim,
                      name_str=name_str)

def _plot_power_statistics(res_, figure_path, excitatory, valence):
    name_str = '_E' if excitatory else '_I'
    name_str += '_' + valence

    _ = reduce.new_filter_reduce(res_, filter_keys=['mouse', 'odor_valence'], reduce_key='half_lick')
    res = reduce.new_filter_reduce(res_, filter_keys=['mouse', 'odor_valence'], reduce_key='half_power')
    res['half_lick'] = _['half_lick']
    from sklearn import linear_model
    from sklearn.metrics import mean_squared_error, r2_score
    regr = linear_model.LinearRegression()
    regr.fit(res['half_lick'].reshape(-1,1), res['half_power'].reshape(-1,1))

    y_pred = regr.predict(res['half_lick'].reshape(-1,1))
    score = regr.score(res['half_lick'].reshape(-1,1), res['half_power'].reshape(-1,1))

    lim = [10, 50]
    ax_lim = {'xlim': lim, 'ylim': lim}
    a, b= plot.plot_results(res, x_key='half_lick', y_key='half_power',
                      plot_args=scatter_args, plot_function=plt.scatter, path=figure_path,
                      save=False, ax_args= ax_lim)

    plt.plot(lim, lim, '--', color='red', alpha=.5, linewidth=1)
    plt.text(25, lim[1], 'R = {:.2f}'.format(score), fontsize=5)
    plot._easy_save(a, b + name_str, pdf=True)

