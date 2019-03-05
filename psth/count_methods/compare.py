from collections import defaultdict
import numpy as np
from matplotlib import pyplot as plt
import filter
import plot
from plot import _easy_save
from psth.format import *


def _compare_dff(res, loop_keys, arg):
    new = defaultdict(list)

    combinations, list_of_ixs = filter.retrieve_unique_entries(res, loop_keys=loop_keys)
    for ixs in list_of_ixs:
        assert len(ixs) == 2, 'more than two entries for days'
        assert res['day'][ixs[0]] < res['day'][ixs[1]], 'not the first day as reference'
        mask_a = res['sig'][ixs[0]]
        mask_b = res['sig'][ixs[1]]
        if arg == 'all':
            mask = np.array([a or b for a, b in zip(mask_a, mask_b)]).astype(bool)
        elif arg == 'first':
            mask = np.array(mask_a).astype(bool)
        elif arg == 'last':
            mask = np.array(mask_b).astype(bool)
        else:
            raise ValueError('arg not recognized')

        amplitudes = []
        for ix in ixs:
            s = res['DAQ_O_ON_F'][ix]
            e = res['DAQ_W_ON_F'][ix]
            dff = res['dff'][ix][mask]
            amplitude = np.max(dff[:, s:e], axis=1)
            amplitudes.append(amplitude)
        new['day_0'].append(np.array(amplitudes[0]))
        new['day_1'].append(np.array(amplitudes[1]))
        update_keys = loop_keys.copy()
        update_keys.append('odor_valence')
        for update_key in update_keys:
            new[update_key].append(res[update_key][ixs[0]])

        # amplitudes = np.transpose(amplitudes)
        # update_keys = loop_keys.copy()
        # update_keys.append('odor_valence')
        # for amplitude_tup in amplitudes:
        #     for update_key in update_keys:
        #         new[update_key].append(res[update_key][ixs[0]])
        #     new['data'].append(amplitude_tup)
        #     new['day'].append([0, 1])
        #     new['ix'].append(tup_ix)
        #     tup_ix += 1
    for k, v in new.items():
        new[k] = np.array(v)
    return new

def plot_compare_dff(res, start_days, end_days, arg, valence, more_stats, figure_path,
                     lim = (-.05, 1.2), ticks=(0, .5, 1)):
    list_of_days = list(zip(start_days,end_days))
    res = filter.filter_days_per_mouse(res, days_per_mouse=list_of_days)
    new = _compare_dff(res, loop_keys = ['mouse', 'odor'], arg = arg)
    new = filter.filter(new, {'odor_valence':valence})

    #analysis
    xs, ys = new['day_0'], new['day_1']
    a, b = [], []
    for x, y in zip(xs, ys):
        a.append(np.sum(x > y))
        b.append(np.sum(y > x))

    fraction = np.sum(a) / (np.sum(a) + np.sum(b))

    ax_args_copy = ax_args.copy()
    ax_args_copy.update({'ylim':lim, 'yticks':ticks, 'xlim':lim, 'xticks':ticks})
    scatter_args_copy = scatter_args.copy()
    scatter_args_copy.update({'marker':',', 's':1, 'alpha':.2})

    colors = ['Black']
    # if valence == 'CS+':
    #     colors = ['Green']
    # if valence == 'CS-':
    #     colors = ['darkred']
    colors *= 300
    path, name = plot.plot_results(new, select_dict={'odor_valence': valence},
                                   x_key='day_0', y_key='day_1', loop_keys=['mouse','odor'],
                                   path=figure_path, plot_args=scatter_args_copy, ax_args=ax_args_copy,
                                   plot_function=plt.scatter,
                                   colors = colors, legend=False,
                                   fig_size=(2, 1.5), save=False)
    plt.plot(lim, lim, '--', color='red', alpha=.5, linewidth=1)
    plt.title(valence)
    if valence == 'CS+':
        plt.text(0, lim[1]-.1, '{:.1f}% diminished'.format(fraction * 100))
    if valence == 'CS-':
        plt.text(0, lim[1]-.1, '{:.1f}% increased'.format(100 - fraction * 100))

    if more_stats:
        new = _compare_dff(res, loop_keys=['mouse', 'odor'], arg='first')
        new = filter.filter(new, {'odor_valence': valence})
        xs, ys = new['day_0'], new['day_1']
        a, b, c = [], [], []
        for x, y in zip(xs, ys):
            a.append(np.sum(x > y))
            b.append(np.sum(y > x))
            c.append(np.sum(y < 0.1))
        lost_fraction = np.sum(c) / (np.sum(a) + np.sum(b))
        plt.text(0, lim[1]-.2, 'Of those, {:.1f}% are gone'.format(100 * lost_fraction))
    _easy_save(path, name, pdf=True)