import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import sem
from format import ax_args
from plot import _easy_save

def _plot_graph(data, lim, ticks, figure_path, name):
    fig_size = (2, 1.5)
    fig = plt.figure(figsize=fig_size)
    rect = [.2, .3, .6, .6]
    ax = fig.add_axes(rect, **ax_args)

    plt.plot(data[0], data[1], '.', color='k', markersize=2, alpha=0.5)

    plt.xlim(lim)
    plt.ylim(lim)
    plt.yticks(ticks, ticks)
    plt.xticks(ticks, ticks)
    plt.xlabel('Firing Rate (Laser OFF)')
    plt.ylabel('Firing Rate (Laser ON)')
    plt.text(-.2, lim[1], 'Baseline: {0:.2f}, Inhibition: {1:.2f}'.format(np.mean(data[0]), np.mean(data[1])))

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    _easy_save(figure_path, name=name, pdf=True)

def _plot_summary(data, ylim, yticks, figure_path, name):
    fig_size = (2, 1.5)
    fig = plt.figure(figsize=fig_size)
    rect = [.2, .2, .6, .6]
    ax = fig.add_axes(rect, **ax_args)

    plt.errorbar([0,1], [np.mean(data[0]), np.mean(data[1])], [sem(data[0]), sem(data[1])], color='k',
                 elinewidth=1, linewidth=1, capsize=1)
    plt.plot(data, color='gray', linestyle='--', alpha=.5, linewidth = .5)
    plt.xlim([-.75, 1.75])
    plt.ylim(ylim)
    plt.yticks(yticks)
    plt.ylabel('Firing Rate')
    plt.xticks([0, 1], labels=['Baseline', 'Inhibition'])

    plt.text(-.2, ylim[1], 'Baseline: {0:.2f}, Inhibition: {1:.2f}'.format(np.mean(data[0]), np.mean(data[1])))

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    _easy_save(figure_path, name=name, pdf=True)


def _plot_firing_rate(data, ylim, figure_path, name, config):
    fig_size = (3, 1.5)
    fig = plt.figure(figsize=fig_size)
    rect = [.2, .2, .6, .6]
    ax = fig.add_axes(rect, **ax_args)

    plt.plot(data, 'k', linewidth=1)
    plt.fill_between([config.t_base, config.t_base + config.t_stim], ylim[0], ylim[1], color= config.color, alpha = .75, linewidth=0)
    plt.xlim([0, len(data)])
    plt.xticks([config.t_base, config.t_base + config.t_stim], labels=['On', 'Off'])
    if ylim is not None:
        plt.ylim(ylim)
    plt.yticks(np.arange(ylim[0], ylim[1] + 1, 10))
    plt.ylabel('Firing Rate')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    _easy_save(figure_path, name=name, pdf=True)


def _raster_plot(data, figure_path, name, config):
    s = config.t_base // config.raster_bin
    e = (config.t_base + config.t_stim) // config.raster_bin
    ylim = [-1, 40]

    fig_size = (3, 1.5)
    fig = plt.figure(figsize=fig_size)
    rect = [.2, .2, .6, .6]
    ax = fig.add_axes(rect, **ax_args)

    plt.imshow(data, cmap='binary')
    plt.axis('tight')
    plt.fill_between([s, e], ylim[0], ylim[1], color= config.color, alpha=.75, linewidth=0)

    plt.xticks([s, e], labels=['On', 'Off'])
    plt.yticks([])
    plt.ylabel('Trials')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.tick_params(axis=u'both', which=u'both', length=0)
    _easy_save(figure_path, name=name, pdf=True)