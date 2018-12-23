from filter import filter
from tools import plot_utils
import matplotlib.pyplot as plt
import numpy as np
import os


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

def plot_results(res, x_key, y_key, loop_key, select_dict=None, path=None, ax_args=None):
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
    loopdata = res[loop_key]

    cmap = plt.get_cmap('cool')
    colors = [cmap(i) for i in np.linspace(0, 1, np.unique(loopdata).size)]

    fig = plt.figure(figsize=(2.5, 2))
    ax = plt.axes(**ax_args)
    ax.set_color_cycle(colors)
    for x in np.unique(loopdata):
        ind = loopdata == x
        x_plot = xdata[ind]
        y_plot = ydata[ind]
        if xdata.dtype == 'O':
            x_plot = x_plot[0]
        if ydata.dtype == 'O':
            y_plot = y_plot[0]
        if y_key == 'mean':
            x_plot = x_plot.transpose()
            y_plot = y_plot.transpose()
            sem_plot = res['sem'][ind][0].transpose()
            ax.fill_between(x_plot, y_plot - sem_plot, y_plot + sem_plot, zorder=0, lw=0, alpha=0.3)
        ax.plot(x_plot, y_plot, label=str(x))

    #format
    if loop_key:
        # l = ax.legend(loc=1, bbox_to_anchor=(1.0, 0.5))
        l = ax.legend()
        l.set_title(loop_key)

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
    if loop_key:
        folder_name += '_vary_' + loop_key
    save_path = os.path.join(path, folder_name)
    _easy_save(save_path, name, dpi=300, pdf=False)