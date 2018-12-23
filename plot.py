from filter import _filter_results
from tools import plot_utils
import matplotlib.pyplot as plt
import numpy as np
import os


# filter
#TODO: refactor somewhere else

def _easy_save(path, name, dpi=300, pdf=True):
    os.makedirs(path, exist_ok=True)
    figname = os.path.join(path, name)
    plt.savefig(os.path.join(figname + '.png'), dpi=dpi)

    if pdf:
        plt.savefig(os.path.join(figname + '.pdf'), transparent=True)
    plt.close()

def plot_results(res, xkey, ykey, loop_key, select_dict=None, path=None, kwargs=None):
    if select_dict is not None:
        res = _filter_results(res, select_dict)

    # process data for plotting
    xdata = res[xkey]
    ydata = res[ykey]
    loopdata = res[loop_key]

    cmap = plt.get_cmap('cool')
    colors = [cmap(i) for i in np.linspace(0, 1, np.unique(loopdata).size)]

    fig = plt.figure(figsize=(3, 3))
    ax = plt.axes(**kwargs)
    ax.set_color_cycle(colors)
    for x in np.unique(loopdata):
        ind = loopdata == x
        x_plot = xdata[ind]
        y_plot = ydata[ind]
        if xdata.dtype == 'O':
            x_plot = x_plot[0]
        if ydata.dtype == 'O':
            y_plot = y_plot[0]
        if ykey == 'mean':
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

    if xkey == 'time':
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

    folder_name = ykey + '_vs_' + xkey
    if loop_key:
        folder_name += '_vary_' + loop_key
    save_path = os.path.join(path, folder_name)
    _easy_save(save_path, name, dpi=300, pdf=False)

# figpath = os.path.join(constants.LOCAL_FIGURE_PATH, condition_name)
# figname = decode_style
# if not os.path.exists(figpath):
#     os.makedirs(figpath)
# figpathname = os.path.join(figpath, figname)
# plt.savefig(figpathname + '.png', dpi=300)