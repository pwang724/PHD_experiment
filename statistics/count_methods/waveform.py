import os

import numpy as np
import scipy.stats
from matplotlib import pyplot as plt
import filter
import reduce
from plot import _easy_save
from collections import defaultdict
import plot
from format import *
from statistics.count_methods.power import _power, _max_dff
import copy

def behavior_vs_neural_power(neural_res, behavior_res, start, end, figure_path, neural_arg ='power', behavior_arg ='onset'):
    neural_res = copy.copy(neural_res)

    neural_key = 'neural'
    behavior_key = 'behavior'
    if behavior_arg == 'onset':
        behavior_data_key = 'time_first_lick_raw'
    elif behavior_arg == 'magnitude':
        behavior_data_key = 'lick_5s'
    elif behavior_arg == 'com':
        behavior_data_key = 'lick_com_raw'
    else:
        raise ValueError('bad key')
    neural_data_key = 'max_power'

    _power(neural_res, excitatory=True)
    for i, p in enumerate(neural_res['Power']):
        s, e = neural_res['DAQ_O_ON_F'][i], neural_res['DAQ_W_ON_F'][i]
        mp = np.max(p[s:e]) - np.mean(p[:s])
        neural_res[neural_data_key].append(mp)
    neural_res[neural_data_key] = np.array(neural_res[neural_data_key])

    list_of_days = [np.arange(s, e+1) for s, e in zip(start,end)]
    print(list_of_days)
    neural_res_filtered = filter.filter_days_per_mouse(neural_res, days_per_mouse=list_of_days)
    neural_res_filtered = filter.filter(neural_res_filtered, filter_dict={'odor_valence':'CS+'})
    behavior_res_filtered = filter.filter(behavior_res, filter_dict={'odor_valence':'CS+'})

    names_neu, ixs_neu = filter.retrieve_unique_entries(neural_res_filtered, ['mouse','day','odor_standard'])
    out = defaultdict(list)
    for ix, names in zip(ixs_neu, names_neu):
        mouse = names[0]
        day = names[1]
        odor_standard = names[2]

        assert len(ix) == 1
        neural = neural_res_filtered[neural_data_key][ix[0]]

        temp = filter.filter(behavior_res_filtered, {'mouse': mouse, 'odor_standard': odor_standard})
        assert len(temp[behavior_data_key]) == 1
        ix_day = [x == day for x in temp['day'][0]]
        lick = temp[behavior_data_key][0][ix_day]
        lick = lick[lick>-1]

        out['mouse'].append(mouse)
        out['day'].append(day)
        out['odor_standard'].append(odor_standard)
        out[neural_key + neural_arg].append(np.mean(neural))
        out[behavior_key + behavior_arg].append(np.mean(lick))
        out['neural_raw'].append(neural)
        out['lick_raw'].append(lick)

    for k, v in out.items():
        out[k] = np.array(v)

    ## versus
    if behavior_arg in ['onset', 'com']:
        xlim = [0, 5]
        xticks = [0, 2, 5]
        xticklabels = ['ON', 'OFF', 'US']
    else:
        xlim = [0, 35]
        xticks = [0, 10, 20, 30]
        xticklabels = [0, 10, 20, 30]

    ax_args = {'xlim': xlim, 'ylim': [0, .1], 'xticks': xticks, 'yticks': [0, .05, .1],
               'xticklabels': xticklabels}
    path, name = plot.plot_results(out, x_key=behavior_key + behavior_arg, y_key=neural_key + neural_arg,
                                   loop_keys='mouse',
                                   plot_function=plt.scatter,
                                   plot_args=scatter_args,
                                   ax_args=ax_args,
                                   colormap='jet',
                                   path=figure_path,
                                   save=False)

    res = out
    from sklearn import linear_model
    from sklearn.metrics import mean_squared_error, r2_score
    regr = linear_model.LinearRegression()
    x = res[behavior_key + behavior_arg].reshape(-1, 1)
    y = res[neural_key + neural_arg].reshape(-1, 1)
    regr.fit(x, y)

    y_pred = regr.predict(x)
    score = regr.score(x, y)

    xlim = plt.xlim()
    ylim = plt.ylim()
    plt.text(xlim[1], ylim[1], 'R = {:.2f}'.format(score))
    name += '_' + behavior_arg
    plot._easy_save(path, name)




def behavior_vs_neural_onset(neural_res, behavior_res, start, end, figure_path, neural_arg ='onset', behavior_arg ='onset'):
    neural_key = 'neural'
    behavior_key = 'behavior'
    
    if behavior_arg == 'onset':
        behavior_data_key = 'time_first_lick_raw'
    elif behavior_arg == 'magnitude':
        behavior_data_key = 'lick_5s'
    elif behavior_arg == 'com':
        behavior_data_key = 'lick_com_raw'
    else:
        raise ValueError('bad key')
    
    if neural_arg == 'onset':
        neural_data_key = 'onset'
    elif neural_arg == 'magnitude':
        neural_data_key = 'dff'
    else:
        raise ValueError('bad key')
    
    list_of_days = [np.arange(s, e+1) for s, e in zip(start,end)]
    neural_res_filtered = filter.filter_days_per_mouse(neural_res, days_per_mouse=list_of_days)
    neural_res_filtered = filter.filter(neural_res_filtered, filter_dict={'odor_valence':'CS+'})
    behavior_res_filtered = filter.filter(behavior_res, filter_dict={'odor_valence':'CS+'})

    names_neu, ixs_neu = filter.retrieve_unique_entries(neural_res_filtered, ['mouse','day','odor_standard'])
    out = defaultdict(list)
    for ix, names in zip(ixs_neu, names_neu):
        mouse = names[0]
        day = names[1]
        odor_standard = names[2]

        assert len(ix) == 1
        neural = neural_res_filtered[neural_data_key][ix[0]]
        neural = neural[neural > -1] * .229

        temp = filter.filter(behavior_res_filtered, {'mouse': mouse, 'odor_standard': odor_standard})
        assert len(temp[behavior_data_key]) == 1
        ix_day = [x == day for x in temp['day'][0]]
        lick = temp[behavior_data_key][0][ix_day]
        lick = lick[lick>-1]

        out['mouse'].append(mouse)
        out['day'].append(day)
        out['odor_standard'].append(odor_standard)
        out[neural_key + neural_arg].append(np.mean(neural))
        out[behavior_key + behavior_arg].append(np.mean(lick))
        out['neural_raw'].append(neural)
        out['lick_raw'].append(lick)

    for k, v in out.items():
        out[k] = np.array(v)

    ## distribution
    def _helper(real, label, bin, range, ax):
        density, bins = np.histogram(real, bins=bin, density=True, range= range)
        unity_density = density / density.sum()
        widths = bins[:-1] - bins[1:]
        ax.bar(bins[1:], unity_density, width=widths, alpha=.5, label=label)

    all_neural = np.concatenate(out['neural_raw'])
    all_licks = np.concatenate(out['lick_raw'])
    bins = 20
    range = [0, 5]
    xticks = [0, 2, 5]
    xticklabels = ['Odor ON', 'Odor OFF', 'US']
    fig = plt.figure(figsize=(2, 1.5))
    ax = fig.add_axes([0.2, 0.2, 0.7, 0.7])
    _helper(all_neural, neural_key + neural_arg, bins, range, ax)
    _helper(all_licks, behavior_key + behavior_arg, bins, range, ax)
    plt.xticks(xticks, xticklabels)
    plt.xlim([range[0]-0.5, range[1] + .5])
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    folder = 'onset_distribution_lick_neural_'
    name = behavior_arg + '_' + ','.join([str(x) for x in start]) + '_' + ','.join([str(x) for x in end])
    plt.legend(frameon=False)
    _easy_save(os.path.join(figure_path, folder), name, dpi=300, pdf=True)

    ## versus
    if behavior_arg in ['onset', 'com']:
        xlim = [0, 5]
        xticks = [0, 2, 5]
        xticklabels = ['ON','OFF', 'US']
    else:
        xlim = [0, 35]
        xticks = [0, 10, 20, 30]
        xticklabels = [0, 10, 20, 30]

    ax_args = {'xlim':xlim, 'ylim':[0, 3.5], 'xticks':xticks, 'yticks':[0, 2, 5],
               'xticklabels':xticklabels, 'yticklabels': ['ON','OFF','US']}
    path, name = plot.plot_results(out, x_key=behavior_key + behavior_arg, y_key=neural_key + neural_arg, loop_keys='mouse',
                      plot_function=plt.scatter,
                      plot_args= scatter_args,
                                   ax_args=ax_args,
                                   colormap='jet',
                      path = figure_path,
                      save=False)

    res = out
    from sklearn import linear_model
    from sklearn.metrics import mean_squared_error, r2_score
    regr = linear_model.LinearRegression()
    x = res[behavior_key + behavior_arg].reshape(-1,1)
    y = res[neural_key + neural_arg].reshape(-1,1)
    regr.fit(x, y)

    y_pred = regr.predict(x)
    score = regr.score(x, y)

    xlim = plt.xlim()
    ylim = plt.ylim()
    plt.text(xlim[1] - .5, ylim[1]-.5, 'R = {:.2f}'.format(score))
    name += '_' + behavior_arg
    plot._easy_save(path, name)




def distribution(res, start, end, data_arg, figure_path, save):
    list_of_days = [np.arange(s, e+1) for s, e in zip(start,end)]
    res_ = filter.filter_days_per_mouse(res, days_per_mouse=list_of_days)
    res_ = filter.filter(res_, filter_dict={'odor_valence':'CS+'})
    mice = np.unique(res_['mouse'])

    real = []
    for mouse in mice:
        res_mouse_ = filter.filter(res_, {'mouse': mouse})
        days = np.unique(res_mouse_['day'])
        for day in days:
            res_mouse_day = filter.filter(res_mouse_, {'day': day})
            assert res_mouse_day['onset'].size == 2, 'should be two entries corresponding to two CS+ per mouse'
            a = res_mouse_day[data_arg][0]
            b = res_mouse_day[data_arg][1]
            a = a[a>-1]
            b = b[b>-1]
            real.append(a)
            real.append(b)

    flatten = lambda l: [item for sublist in l for item in sublist]
    real = np.array(flatten(real), dtype=float)

    bin = 20
    if data_arg == 'amplitude':
        xlim = 1.5
        hist_range = 1.5
    else:
        period = .229
        xlim = np.ceil(bin * period)
        hist_range = bin * period
        real *= period

    if not save:
        fig = plt.figure(figsize=(2, 1.5))
        ax = fig.add_axes([0.2, 0.2, 0.7, 0.7])
    else:
        ax = plt.gca()

    def _helper(real, label):
        density, bins = np.histogram(real, bins=bin, density=True, range=[0, hist_range])
        unity_density = density / density.sum()
        widths = bins[:-1] - bins[1:]
        ax.bar(bins[1:], unity_density, width=widths, alpha=.5, label=label)

    _helper(real, 'Data')
    ax.set_xlabel((data_arg.capitalize()))
    ax.set_ylabel('Density')

    if data_arg != 'amplitude':
        plt.xticks([0, 2, 5], ['Odor On', 'Off', 'US'])
        plt.xlim([-0.5, 5.5])
    else:
        plt.xlim([0, xlim])

    # xticks = np.arange(xlim)
    # ax.set_xticks(xticks)
    # ax.set_xticklabels(xticks)
    # yticks = np.array([0, .5, 1])
    # ax.set_yticks(yticks)
    # plt.ylim([0, 1])

    mean = np.mean(real)
    median = np.median(real)
    ylim = ax.get_ylim()
    xlim = ax.get_xlim()
    x = (xlim[1] - xlim[0]) / 2
    y = (ylim[1] - ylim[0]) / 2
    if save:
        y -= .05
    t = 'Median = {:.3f}, mean = {:.3f}'.format(median, mean)
    plt.text(x, y, t)

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    name = 'distribution ' + data_arg + '_' + ','.join([str(x) for x in start]) + '_' + ','.join([str(x) for x in end])

    if save:
        plt.legend(frameon=False)
        _easy_save(os.path.join(figure_path, data_arg), name, dpi=300, pdf=True)

    #statistics
    if data_arg == 'onset':
        odor = np.sum(real < 2) / real.size
        delay = np.sum(real > 2) / real.size
        print('Fraction of cells with onset during odor presentation: {}'.format(odor))
    return real



def compare_to_shuffle(res, start, end, data_arg, figure_path):
    '''

    :param res:
    :param start:
    :param end:
    :param data_arg: 'onset', 'duration', 'amplitude'
    :param figure_path:
    :return:
    '''
    n_shuffle = 1000

    list_of_days = [np.arange(s, e+1) for s, e in zip(start,end)]
    res_ = filter.filter_days_per_mouse(res, days_per_mouse=list_of_days)
    res_ = filter.filter(res_, filter_dict={'odor_valence':'CS+'})
    mice = np.unique(res_['mouse'])

    real = []
    shuffled = []
    for mouse in mice:
        res_mouse_ = filter.filter(res_, {'mouse': mouse})
        days = np.unique(res_mouse_['day'])
        for day in days:
            res_mouse_day = filter.filter(res_mouse_, {'day': day})

            assert res_mouse_day['onset'].size == 2, 'should be two entries corresponding to two CS+ per mouse'
            a = res_mouse_day[data_arg][0]
            b = res_mouse_day[data_arg][1]
            responsive_to_both = np.array([x>-1 and y>-1 for x, y in zip(a,b)])
            a_ = a[responsive_to_both]
            b_ = b[responsive_to_both]
            real_diff = np.abs(a_ - b_)
            real.append(real_diff)

            for n in range(n_shuffle):
                a_shuffled = np.random.permutation(a_)
                b_shuffled = np.random.permutation(b_)
                shuffled_diff = np.abs(a_shuffled - b_shuffled)
                shuffled.append(shuffled_diff)

    flatten = lambda l: [item for sublist in l for item in sublist]
    real = np.array(flatten(real), dtype=float)
    shuffled = np.array(flatten(shuffled), dtype=float)

    bin = 20
    if data_arg == 'amplitude':
        xlim = 1
        hist_range = 1
    else:
        period = .229
        xlim = np.ceil(bin * period)
        hist_range = bin * period
        real *= period
        shuffled *= period

    p = scipy.stats.ranksums(real, shuffled)[1]

    fig = plt.figure(figsize=(2.5, 2))
    ax = fig.add_axes([0.2, 0.2, 0.7, 0.7])

    def _helper(real, label):
        density, bins = np.histogram(real, bins=bin, density=True, range=[0, hist_range])
        unity_density = density / density.sum()
        widths = bins[:-1] - bins[1:]
        ax.bar(bins[1:], unity_density, width=widths, alpha = .5, label=label)

    _helper(real, 'Data')
    _helper(shuffled, 'Shuffled')

    ax.legend(frameon=False)
    ax.set_xlabel((r'$\Delta$' + ' ' + data_arg.capitalize()))
    ax.set_ylabel('Density')
    plt.xlim([0, xlim])

    # xticks = np.arange(xlim)
    # ax.set_xticks(xticks)
    # ax.set_xticklabels(xticks)
    # yticks = np.array([0, .5, 1])
    # ax.set_yticks(yticks)
    # plt.ylim([0, 1])

    ylim = ax.get_ylim()
    xlim = ax.get_xlim()
    x = (xlim[1] - xlim[0])/2
    y = (ylim[1] - ylim[0])/2
    t = 'P = {:.3e}'.format(p)
    plt.text(x, y, t)

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    _easy_save(os.path.join(figure_path, data_arg), 'difference_' + data_arg, dpi=300, pdf=True)