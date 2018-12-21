from tools import plot_utils
import matplotlib.pyplot as plt
import numpy as np
import copy
import os


# filter
def _filter_results(res, select_dict):
    out = copy.copy(res)
    list_of_ixs = []
    for key, val in select_dict.items():
        list_of_ixs.append(res[key] == val)
    select_ixs = np.all(list_of_ixs, axis=0)
    for key, value in res.items():
        out[key] = value[select_ixs]
    return out

def _easy_save(path, name, dpi=300, pdf=True):
    os.makedirs(path, exist_ok=True)
    figname = os.path.join(path, name)
    plt.savefig(os.path.join(figname + '.png'), dpi=dpi)

    if pdf:
        plt.savefig(os.path.join(figname + '.pdf'), transparent=True)
    plt.close()

def plot_results(res, xkey, ykey, loop_key, select_dict, path):
    res = _filter_results(res, select_dict)

    # process data for plotting
    xdata = res[xkey]
    ydata = res[ykey]
    loopdata = res[loop_key]

    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_axes([0.2, 0.2, 0.7, 0.7])
    for x in np.unique(loopdata):
        ind = loopdata == x
        x_plot = xdata[ind]
        y_plot = ydata[ind]
        if ykey == 'mean' or ykey == 'sem':
            ax.plot(x_plot.transpose(), y_plot.transpose(), label=str(x))
        else:
            ax.plot(x_plot, y_plot, 'o-', label=str(x))

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

    yticks = np.linspace(0,1,5)
    ax.set_yticks(yticks)
    ax.set_title(select_dict)
    plot_utils.nicer_plot(ax)

    name = ''
    for k, v in select_dict.items():
        name += k + '_' + str(v) + '_'

    name += '_' + ykey + '_vs_' + xkey
    if loop_key:
        name += '_vary_' + loop_key
    _easy_save(path, name, dpi=300, pdf=False)

# figpath = os.path.join(constants.LOCAL_FIGURE_PATH, condition_name)
# figname = decode_style
# if not os.path.exists(figpath):
#     os.makedirs(figpath)
# figpathname = os.path.join(figpath, figname)
# plt.savefig(figpathname + '.png', dpi=300)