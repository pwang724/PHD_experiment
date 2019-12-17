import filter
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
from collections import OrderedDict, defaultdict
import seaborn as sns

plt.style.use('default')
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.size'] = 7
mpl.rcParams['font.family'] = 'arial'

def nice_names(key):
    nice_name_dict = {
        'trial': 'Trials',
        'lick_smoothed': 'Number of Licks',
        'boolean_smoothed': '% Trials with Licks',
        'lick': 'Number of Licks',
        'odor': 'Odor',
        'mouse': 'Mouse',
        'half_max': 'Learning Rate',
        'odor_standard': 'Odor',
        'condition_name': 'Experimental Condition',
        'OFC_JAWS': r'OFC$_{\rm INH}$',
        'BLA_JAWS': r'BLA$_{\rm INH}$',
        'OFC_LONGTERM': r'OFC$_{\rm LT}$',
        'BLA_LONGTERM': r'BLA$_{\rm LT}$',
        'odor_valence': 'Odor Valence',
        'csp_identity': 'CS+ ID',
        'csm_identity': 'CS- ID',
        'identity': 'ID',
        'valence': 'Valence',
        'False': '0',
        'True': '1',
        'day':'Day',
        'BEHAVIOR_OFC_JAWS_PRETRAINING': 'PT IH',
        'BEHAVIOR_OFC_JAWS_DISCRIMINATION': 'DT IH',
        'BEHAVIOR_OFC_YFP': 'YFP',
    }

    if key in nice_name_dict.keys():
        out = nice_name_dict[key]
    else:
        out = key
    return out

def significance_str(x, y, val):
    if val < .05 and val >= .01:
        str = 'P < .05'
    elif val < .01 and val > .001:
        str = 'P < .01'
    elif val < .001:
        str = 'P < .001'
    else:
        str = 'P = {:.2f}'.format(val)
    plt.text(x, y, str)
    return str


def _easy_save(path, name, dpi=300, pdf=True, close=True):
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
    print('figure saved in {}'.format(figname))
    plt.savefig(os.path.join(figname + '.png'), dpi=dpi)

    if pdf:
        plt.savefig(os.path.join(figname + '.pdf'), transparent=True)

    if close:
        plt.close()


def _string_to_index(xdata):
    x_index = np.unique(xdata, return_index=True)[1]
    labels = [xdata[index] for index in sorted(x_index)]
    indices = np.zeros_like(xdata, dtype= int)
    for i, label in enumerate(labels):
        indices[label == xdata] = i
    nice_labels = [nice_names(key) for key in labels]
    return indices, nice_labels

def _plot(plot_function, x, y, color, label, plot_args, xjitter= 0):
    if x.dtype == 'O' and y.dtype == 'O':
        # print('plotted O')
        for i in range(x.shape[0]):
            # xj = x[i] + np.random.uniform(low=-xjitter,high=xjitter, size=x[i].shape)
            plot_function(x[i], y[i], color=color, label=label, **plot_args)
    else:
        x = np.squeeze(x)
        # xj = x + np.random.uniform(low=-xjitter, high=xjitter, size=x.shape)
        y = np.squeeze(y)
        plot_function(x, y, color=color, label=label, **plot_args)

def _plot_error(plot_function, x, y, err, color, label, plot_args):
    if x.dtype == 'O' and y.dtype == 'O':
        # print('plotted O')
        for i in range(x.shape[0]):
            plot_function(x[i], y[i], err[i], color=color, label=label, **plot_args)
    else:
        x_ = np.squeeze(x)
        y_ = np.squeeze(y)
        err_ = np.squeeze(err)

        if y_.size == 1:
            x_ = x
            y_ = y
            err_ = err

        plot_function(x_, y_, err_, color=color, label=label, **plot_args)

def _plot_fill(plot_function, x, y, err, color, label, plot_args):
    if x.dtype == 'O' and y.dtype == 'O':
        # print('plotted O')
        for i in range(x.shape[0]):
            plot_function(x[i], y[i]-err[i], y[i] + err[i], color=color, label=label, **plot_args)
    else:
        x = np.squeeze(x)
        y = np.squeeze(y)
        err = np.squeeze(err)
        plot_function(x, y-err, y+err, color=color, label=label, **plot_args)

def plot_results(res, x_key, y_key, loop_keys =None,
                 select_dict=None, path=None, colors= None, colormap='cool',
                 plot_function= plt.plot, ax_args={}, plot_args={},
                 xjitter = 0,
                 save = True, reuse = False, twinax = False, sort = False, error_key = '_sem',
                 fig_size = (2, 1.5), rect = (.2, .2, .6, .6), legend = True, name_str = ''):
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
        res = filter.filter(res, select_dict)

    if reuse:
        ax = plt.gca()
        if twinax:
            ax = ax.twinx()
    else:
        fig = plt.figure(figsize=fig_size)
        ax = fig.add_axes(rect, **ax_args)

    if sort:
        ind_sort = np.argsort(res[x_key])
        for key, val in res.items():
            res[key] = val[ind_sort]

    if loop_keys != None:
        if isinstance(loop_keys, str):
            loop_keys = [loop_keys]
        loop_combinations, loop_indices = filter.retrieve_unique_entries(res, loop_keys)
        if save:
            labels = [str(','.join(str(e) for e in cur_combination)) for cur_combination in loop_combinations]
        else:
            labels = [None] * len(loop_combinations)

        if colormap is None:
            cmap = plt.get_cmap('cool')
        else:
            cmap = plt.get_cmap(colormap)
        if colors is None:
            colors = [cmap(i) for i in np.linspace(0, 1, len(loop_combinations))]
        loop_lines = len(loop_combinations)
    else:
        loop_lines = 1
        loop_indices = [np.arange(len(res[y_key]))]
        if colors is None:
            colors = ['black']
        else:
            colors = [colors]
        labels = [None]

    for i in range(loop_lines):
        color = colors[i]
        label = labels[i]
        plot_ix = loop_indices[i]
        x_plot = res[x_key][plot_ix]
        if type(x_plot[0]) != np.ndarray:
            x_plot = np.array([nice_names(x) for x in list(x_plot)])
        y_plot = res[y_key][plot_ix]

        if plot_function == plt.errorbar:
            error_plot = res[error_key][plot_ix]
            _plot_error(plot_function, x_plot, y_plot, error_plot, color=color, label=label, plot_args=plot_args)
        elif plot_function == plt.fill_between:
            error_plot = res[error_key][plot_ix]
            _plot_fill(plot_function, x_plot, y_plot, error_plot, color=color, label=label, plot_args=plot_args)
        elif plot_function == sns.swarmplot:
            t = defaultdict(list)
            for k, v in res.items():
                t[k] = res[k][plot_ix]
            sns.swarmplot(x = x_key, y = y_key, hue=loop_keys[0], data=t, **plot_args)
            ax.get_legend().remove()
        else:
            _plot(plot_function, x_plot, y_plot, color=color, label=label, plot_args=plot_args, xjitter=xjitter)

    #format
    # plt.xticks(rotation=45)
    ax.set_ylabel(nice_names(y_key), fontsize = 7)
    ax.set_xlabel(nice_names(x_key), fontsize = 7)
    if x_key == 'time':
        xticks = res['xticks'][0]
        xticklabels = ['On', 'Off', 'US']
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)

    if not twinax:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
    else:
        ax.spines['top'].set_visible(False)

    if loop_keys and legend:
        nice_loop_str = '+'.join([nice_names(x) for x in loop_keys])

        handles, labels = ax.get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        l = ax.legend(by_label.values(), by_label.keys(), ncol = 2, fontsize = 4, frameon=False)
        try:
            for handle in l.legendHandles:
                handle.set_sizes([5])
        except:
            pass
        # l.set_title(nice_loop_str)
        # plt.setp(l.get_title(), fontsize=4)

    if select_dict is None:
        name = 'figure'
    else:
        name = ''
        for k, v in select_dict.items():
            name += k + '_' + str(v) + '_'
    name += name_str

    folder_name = y_key + '_vs_' + x_key
    if loop_keys:
        loop_str = '+'.join(loop_keys)
        folder_name += '_vary_' + loop_str
    save_path = os.path.join(path, folder_name)
    if save:
        _easy_save(save_path, name, dpi=300, pdf=True)
    else:
        return save_path, name


def plot_weight(summary_res, x_key, y_key, val_key, title, vmin, vmax,
                xticklabel =None, yticklabel=None,
                label ='Accuracy', save_path = None, text='', mask=False):
    x_len = len(np.unique(summary_res[x_key]))
    y_len = len(np.unique(summary_res[y_key]))

    x = summary_res[x_key]
    y = summary_res[y_key]
    z = summary_res[val_key]
    w_plot = np.zeros((x_len, y_len))
    w_plot[y, x] = z

    rect = [0.2, 0.2, 0.6, 0.6]
    rect_cb = [0.82, 0.2, 0.02, 0.65]
    fig = plt.figure(figsize=(2.2, 2.2))
    ax = fig.add_axes(rect)

    if mask:
        m = np.tri(w_plot.shape[0], k=0).astype(bool)
        w_plot = np.ma.array(w_plot, mask=np.invert(m))  # mask out the lower triangle
        cmap = plt.get_cmap()
        cmap.set_bad('w')  # default value is 'k'

    im = plt.pcolor(np.flipud(w_plot), cmap='jet', vmin=vmin, vmax=vmax)
    # im = plt.imshow(w_plot, cmap='jet', vmin=vmin, vmax=vmax, origin='upper')

    def _show_values(pc, fmt="%.2f", **kw):
        pc.update_scalarmappable()
        for p, color, value in zip(pc.get_paths(), pc.get_facecolors(), pc.get_array()):
            x, y = p.vertices[:-2, :].mean(0)
            if (value - vmin)/(vmax-vmin) > .2:
                color = (0.0, 0.0, 0.0)
            else:
                color = (1.0, 1.0, 1.0)
            ax.text(x, y, fmt % value, ha="center", va="center", color=color, fontsize=5, **kw)

    _show_values(im)

    # for i, j, k in zip(x, y, z):
    #     plt.text(i-.15, j-.1, np.round(k,2))
    # import seaborn as sns
    # im = sns.heatmap(w_plot, annot=True)

    plt.title(title, fontsize=7)
    ax.set_xlabel(x_key, labelpad=2)
    ax.set_ylabel(y_key, labelpad=2)
    plt.axis('tight')
    for loc in ['bottom', 'top', 'left', 'right']:
        ax.spines[loc].set_visible(False)
    ax.tick_params('both', length=0)
    xticks = np.arange(0, w_plot.shape[1]) + .5
    yticks = np.arange(0, w_plot.shape[0]) + .5
    ax.set_xticks(xticks)
    ax.set_yticks(yticks[::-1])

    if xticklabel == None:
        ax.set_xticklabels((xticks + .5).astype(int), fontsize = 7)
        ax.set_yticklabels((yticks + .5).astype(int), fontsize = 7)
    else:
        ax.set_xticklabels(xticklabel, fontsize = 7)
        ax.set_yticklabels(yticklabel, fontsize = 7)
    plt.axis('tight')

    ax = fig.add_axes(rect_cb)
    cb = plt.colorbar(cax=ax, ticks=[vmin, vmax])
    cb.outline.set_linewidth(0.5)
    cb.set_label(label, fontsize=7, labelpad=-5)
    plt.tick_params(axis='both', which='major', labelsize=7)
    plt.axis('tight')

    folder_name = x_key + '_and_' + y_key + '_vs_' + val_key + '_' + text
    p = os.path.join(save_path, folder_name)
    _easy_save(p, 'figure', dpi=300, pdf=True)