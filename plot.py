from filter import filter
from tools import plot_utils
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
import itertools
from sklearn import preprocessing

mpl.rcParams['font.size'] = 5

def nice_names(key):
    nice_name_dict = {
        'trial': 'Trials',
        'lick_smoothed': 'Number of Licks',
        'boolean_smoothed': '% Trials with Licks',
        'lick': 'Number of Licks',
        'odor': 'Odor',
        'mouse': 'Mouse',
        'half_max': 'Learning Rate',
        'odor_standard': 'Odor'
    }
    if key in nice_name_dict.keys():
        out = nice_name_dict[key]
    else:
        out = key
    return out

def _easy_save(path, name, dpi=300, pdf=True):
    '''
    convenience function for saving figs while taking care of making folders
    :param path: save path
    :param name: save name
    :param dpi: save dpi for .png format
    :param pdf: boolean, save in another pdf or not
    :return:
    '''
    os.makedirs(path, exist_ok=True)
    figname = os.path.join(path, name)
    plt.savefig(os.path.join(figname + '.png'), dpi=dpi)

    if pdf:
        plt.savefig(os.path.join(figname + '.pdf'), transparent=True)
    plt.close()

def _loop_key_filter(res, loop_keys):
    unique_entries_per_loopkey = []
    for x in loop_keys:
        a = res[x]
        indexes = np.unique(a, return_index=True)[1]
        unique_entries_per_loopkey.append([a[index] for index in sorted(indexes)])

    unique_entry_combinations = list(itertools.product(*unique_entries_per_loopkey))
    nlines = len(unique_entry_combinations)

    list_of_ind = []
    for x in range(nlines):
        list_of_ixs = []
        cur_combination = unique_entry_combinations[x]
        for i, val in enumerate(cur_combination):
            list_of_ixs.append(val == res[loop_keys[i]])
        list_of_ind.append(np.all(list_of_ixs, axis=0))
    return unique_entry_combinations, list_of_ind


def plot_results(res, x_key, y_key, loop_keys, select_dict=None, path=None, colors= None, plot_function= plt.plot, ax_args={}, plot_args={},
                 save = True, reuse = False):
    '''

    :param res: flattened dict of results
    :param x_key:
    :param y_key:
    :param loop_key:
    :param select_dict:
    :param path: save path
    :param ax_args: additional args to pass to ax, such as ylim, etc. in dictionary format
    :return:
    '''

    if select_dict is not None:
        res = filter(res, select_dict)

    # process data for plotting
    xdata = res[x_key]
    ydata = res[y_key]

    if reuse:
        ax = plt.gca()
    else:
        fig = plt.figure(figsize=(2, 1.5))
        rect = [.2, .2, .7, .7]
        ax = plt.axes(rect, **ax_args)

    if loop_keys:
        if isinstance(loop_keys, str):
            loop_keys = [loop_keys]
        unique_entry_combinations, list_of_ind = _loop_key_filter(res, loop_keys)
        nlines = len(unique_entry_combinations)
        if colors is None:
            cmap = plt.get_cmap('cool')
            colors = [cmap(i) for i in np.linspace(0, 1, nlines)]

        for x in range(nlines):
            ind = list_of_ind[x]
            cur_combination = unique_entry_combinations[x]
            x_plot = xdata[ind]
            y_plot = ydata[ind]
            if save:
                label = str(','.join(str(e) for e in cur_combination))
            else:
                label = None

            if xdata.dtype == 'O' and ydata.dtype == 'O':
                for i in range(x_plot.shape[0]):
                    plot_function(x_plot[i], y_plot[i], color= colors[x], label=label, **plot_args)
                    # if y_key == 'mean':
                    #     sem_plot = res['sem'][ind]
                    #     ax.fill_between(x_plot[i], y_plot[i] - sem_plot[i], y_plot[i] + sem_plot[i],
                    #                     color = colors[x], zorder=0, lw=0, alpha=0.3)
            else:
                plot_function(x_plot, y_plot, color=colors[x], label=label, **plot_args)
    else:
        if colors is None:
            colors = 'black'
        if type(xdata[0]) == str:
            x_index = np.unique(xdata, return_index=True)[1]
            x_inverse = np.unique(xdata, return_inverse=True)[1]

            x_ticks = np.unique(x_inverse)
            x_labels = [xdata[index] for index in sorted(x_index)]
            x_data = sorted(x_inverse)
            plot_function(x_data, ydata, color=colors, **plot_args)
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_labels)
        else:
            plot_function(xdata, ydata, color=colors, **plot_args)


    #format
    ax.set_ylabel(nice_names(y_key), fontsize = 5)
    ax.set_xlabel(nice_names(x_key), fontsize = 5)
    if x_key == 'time':
        xticks = res['xticks'][0]
        xticklabels = ['On', 'Off', 'US']
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
    plot_utils.nicer_plot(ax)

    if save:
        if loop_keys:
            nice_loop_str = '+'.join([nice_names(x) for x in loop_keys])
            l = ax.legend(fontsize=4)
            l.set_title(nice_loop_str)
            plt.setp(l.get_title(), fontsize=4)

        if select_dict is None:
            name = 'figure'
        else:
            name = ''
            for k, v in select_dict.items():
                name += k + '_' + str(v) + '_'

        folder_name = y_key + '_vs_' + x_key
        if loop_keys:
            loop_str = '+'.join(loop_keys)
            folder_name += '_vary_' + loop_str
        save_path = os.path.join(path, folder_name)
        _easy_save(save_path, name, dpi=300, pdf=False)