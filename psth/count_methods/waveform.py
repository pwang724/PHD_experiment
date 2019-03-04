import os

import numpy as np
import scipy.stats
from matplotlib import pyplot as plt
import filter
from plot import _easy_save

def distribution(res, start, end, data_arg, figure_path):
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

    fig = plt.figure(figsize=(2.5, 2))
    ax = fig.add_axes([0.2, 0.2, 0.7, 0.7])

    def _helper(real, label):
        density, bins = np.histogram(real, bins=bin, density=True, range=[0, hist_range])
        unity_density = density / density.sum()
        widths = bins[:-1] - bins[1:]
        ax.bar(bins[1:], unity_density, width=widths, alpha=.5, label=label)

    _helper(real, 'Data')
    ax.set_xlabel((data_arg.capitalize()))
    ax.set_ylabel('Density')
    plt.xlim([0, xlim])

    # xticks = np.arange(xlim)
    # ax.set_xticks(xticks)
    # ax.set_xticklabels(xticks)
    # yticks = np.array([0, .5, 1])
    # ax.set_yticks(yticks)
    # plt.ylim([0, 1])

    median = np.median(real)
    ylim = ax.get_ylim()
    xlim = ax.get_xlim()
    x = (xlim[1] - xlim[0]) / 2
    y = (ylim[1] - ylim[0]) / 2
    t = 'Median = {:.3f}'.format(median)
    plt.text(x, y, t)

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    _easy_save(os.path.join(figure_path, data_arg), 'distribution_' + data_arg, dpi=300, pdf=True)



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