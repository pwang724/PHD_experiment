from filter import filter
from tools import plot_utils
import matplotlib.pyplot as plt
import numpy as np
import os
import itertools



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

def plot_results(res, x_key, y_key, loop_keys, select_dict=None, path=None, ax_args=None):
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
    if isinstance(loop_keys, str):
        loop_keys = [loop_keys]

    unique_entries_per_loopkey =  [np.unique(res[x]) for x in loop_keys]
    unique_entry_combinations = list(itertools.product(*unique_entries_per_loopkey))
    nlines = len(unique_entry_combinations)

    cmap = plt.get_cmap('cool')
    colors = [cmap(i) for i in np.linspace(0, 1, nlines)]

    fig = plt.figure(figsize=(2.5, 2))
    ax = plt.axes(**ax_args)
    for x in range(nlines):
        list_of_ixs = []
        cur_combination = unique_entry_combinations[x]
        for i, val in enumerate(cur_combination):
            list_of_ixs.append(val == res[loop_keys[i]])
        ind = np.all(list_of_ixs, axis=0)

        x_plot = xdata[ind]
        y_plot = ydata[ind]
        label = str(','.join(str(e) for e in cur_combination))

        if xdata.dtype == 'O' and ydata.dtype == 'O':
            for i in range(x_plot.shape[0]):
                ax.plot(x_plot[i], y_plot[i], color= colors[x], label=label)
                if y_key == 'mean':
                    sem_plot = res['sem'][ind]
                    ax.fill_between(x_plot[i], y_plot[i] - sem_plot[i], y_plot[i] + sem_plot[i],
                                    color = colors[x], zorder=0, lw=0, alpha=0.3)
        else:
            ax.plot(x_plot, y_plot,
                    color=colors[x], label=label)




    #format
    loop_str = '+'.join(loop_keys)
    if loop_keys:
        l = ax.legend(fontsize = 4)
        l.set_title(loop_str)
        plt.setp(l.get_title(), fontsize=4)

    if x_key == 'time':
        xticks = res['xticks'][0]
        xticklabels = ['On', 'Off', 'US']
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
    plot_utils.nicer_plot(ax)

    if select_dict is None:
        name = 'figure'
    else:
        name = ''
        for k, v in select_dict.items():
            name += k + '_' + str(v) + '_'

    folder_name = y_key + '_vs_' + x_key
    if loop_keys:
        folder_name += '_vary_' + loop_str
    save_path = os.path.join(path, folder_name)
    _easy_save(save_path, name, dpi=300, pdf=False)