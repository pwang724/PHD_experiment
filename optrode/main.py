import numpy as np
import glob
import os
import scipy.io as io
import matplotlib.pyplot as plt
import matplotlib as mpl
from psth.format import *
from plot import _easy_save
from scipy.stats import sem

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.size'] = 10
mpl.rcParams['font.family'] = 'arial'

conversion = 1000
t_base = 10 * conversion
t_stim = 5 * conversion
t_after = 10 * conversion
t_total = t_base + t_stim + t_after
downsample = 1
moving_average_bin = 1000

d = r'C:\Users\P\Desktop\PHILIP\Peter Data\2017.02.05_Y2_1861'
eventFile = os.path.join(d, 'events_ChR2.mat')
events = io.loadmat(eventFile)
events = events['events']
events = np.round(events[:,0] * 1000).astype(int)

shankDirs = sorted(glob.glob(os.path.join(d, 'shank*')))

#data is in format of cell, event, time
mat = []
for shankDir in shankDirs:
    spikeFile = os.path.join(shankDir, 'kwik_spikes.mat')
    data = io.loadmat(spikeFile)
    data = data['spikeIDtiming']
    cell_id = data[:, 1]
    spike_times = data[:, 2]

    unique_cells = np.unique(cell_id)
    max_time = np.max(spike_times) + 1
    for cell in unique_cells:
        idx = cell == cell_id
        spike_time = spike_times[idx]
        spike_vec = np.zeros(max_time)
        spike_vec[spike_time] = 1

        cell_data = []
        for event in events:
            s = event - t_base
            e = event + t_stim + t_after
            cell_data.append(spike_vec[s:e])
        mat.append(cell_data)
mat = np.array(mat)

firing_rate = np.mean(mat, axis=1)
for i in range(firing_rate.shape[0]):
    firing_rate[i,:] = np.convolve(firing_rate[i,:], np.ones(moving_average_bin)/moving_average_bin, 'same')
firing_rate *= conversion / downsample

mean_before = np.mean(firing_rate[:, :t_base-500], axis=1)
mean_after = np.mean(firing_rate[:, 500+t_base:t_base+t_stim-500], axis=1)
data = [mean_before, mean_after]
norm_data = [np.ones_like(mean_before), mean_after/mean_before]


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

def _plot_firing_rate(data, ylim, figure_path, name):
    fig_size = (3, 1.5)
    fig = plt.figure(figsize=fig_size)
    rect = [.2, .2, .6, .6]
    ax = fig.add_axes(rect, **ax_args)

    plt.plot(data, 'k', linewidth=1)
    plt.fill_between([t_base, t_base + t_stim], ylim[0], ylim[1], color='salmon', alpha = .75, linewidth=0)
    plt.xlim([0, len(data)])
    plt.xticks([t_base, t_base + t_stim], labels=['On', 'Off'])
    plt.ylim(ylim)
    plt.yticks(np.arange(ylim[0], ylim[1] + 1, 10))
    plt.ylabel('Firing Rate')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    _easy_save(figure_path, name=name, pdf=True)

raster_bin = 20
def _raster_convert(mat):
    a = []
    for i in range(mat.shape[0]):
        b = []
        for j in range(mat.shape[1]):
            temp = np.convolve(mat[i, j, :], np.ones(raster_bin), 'same')
            out = temp[::raster_bin] > 0
            b.append(out)
        a.append(b)
    return a

def _raster_plot(data, figure_path, name):
    s = t_base // raster_bin
    e = (t_base + t_stim) // raster_bin
    ylim = [-1, 10]

    fig_size = (3, 1.5)
    fig = plt.figure(figsize=fig_size)
    rect = [.2, .2, .6, .6]
    ax = fig.add_axes(rect, **ax_args)

    plt.imshow(data, cmap='binary')
    plt.axis('tight')
    plt.fill_between([s, e], ylim[0], ylim[1], color='salmon', alpha=.75, linewidth=0)

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

figure_path = r'C:\Users\P\Desktop\PYTHON\PHD_experiment\_FIGURES\OTHER\OPTRODE'

# _plot_summary(data, ylim = [0, 22], yticks = [0, 10, 20], figure_path=figure_path, name ='spike_rate')
# _plot_summary(norm_data, ylim = [0, 1.3], yticks = [0, .5, 1], figure_path=figure_path, name ='spike_rate_normalized')

ixs = [5, 10, 12]
ylim_max = [30, 30, 40]
for i, ix in enumerate(ixs):
    data = firing_rate[ix]
    _plot_firing_rate(data, [0, ylim_max[i]], figure_path=figure_path, name= 'firing_rate_{}'.format(ix))

mat_raster = _raster_convert(mat)
for i, ix in enumerate(ixs):
    data = mat_raster[ix]
    _raster_plot(data, figure_path=figure_path, name= 'raster_image_{}'.format(ix))
