import numpy as np
import glob
import os
import scipy.io as io
import matplotlib as mpl
import matplotlib.pyplot as plt
from optrode.helper_functions import _plot_firing_rate, _raster_convert, _raster_plot, _plot_summary

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.size'] = 10
mpl.rcParams['font.family'] = 'arial'

class Base_Config(object):
    def __init__(self):
        self.conversion = 1000
        self.downsample = 1
        self.moving_average_bin = 1000
        self.raster_bin = 20
        self.spike_file = 'kwik_spikes.mat'
        self.base_path = '/Users/pwang/Desktop/GITHUB_PROJECTS/PHD_experiment/_FIGURES'

class JAWS_Config(Base_Config):
    def __init__(self):
        super(JAWS_Config, self).__init__()
        self.t_base = 10 * self.conversion
        self.t_stim = 5 * self.conversion
        self.t_after = 10 * self.conversion
        self.t_total = self.t_base + self.t_stim + self.t_after
        self.figure_path = os.path.join(self.base_path, 'OPTRODE','JAWS')
        self.directory = 'blank' #TODO
        self.event_file = 'events_ChR2.mat'
        self.color = 'salmon'

class HALO_Config(Base_Config):
    def __init__(self):
        super(HALO_Config, self).__init__()
        self.t_base = 10 * self.conversion
        self.t_stim = 10 * self.conversion
        self.t_after = 10 * self.conversion
        self.t_total = self.t_base + self.t_stim + self.t_after
        self.figure_path = os.path.join(self.base_path, 'OPTRODE','HALO')
        self.directory = '/Users/pwang/Dropbox/TEMP_DATA/test6'
        self.event_file = 'events_HALO.mat'
        self.color = 'yellow'
        self.cells = np.arange(40)
        self.ylim = [40] * 8
        self.moving_average_bin = 1000

def _parse_data(spikeFile, events):
    mat = []
    data = io.loadmat(spikeFile)
    data = data['spikeIDtiming']
    cell_id = data[:, 1]
    spike_times = data[:, 2]

    unique_cells = np.unique(cell_id)
    print(len(unique_cells))
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
    return mat


config = HALO_Config()
t_base = config.t_base
t_stim = config.t_stim
t_after = config.t_after
figure_path = config.figure_path

# for JAWS
# eventFile = os.path.join(config.directory, config.event_file)
# events = io.loadmat(eventFile)
# events = events['events']
# events = np.round(events[:,0] * 1000).astype(int)
#
# #data is in format of cell, event, time
# mat = []
# shankDirs = sorted(glob.glob(os.path.join(config.directory, 'shank*')))
# for shankDir in shankDirs:
#     spikeFile = os.path.join(shankDir, config.spike_file)
#     temp = _parse_data(spikeFile, events)
#     mat += temp
# mat = np.array(mat)


#for HALO
mat = []

eventFile = os.path.join(config.directory, config.event_file)
events = io.loadmat(eventFile)
events = events['events']

diff = np.diff(events[:,0])
ixs = np.where(np.all([diff > 20, diff < 21], axis=0))[0]
events = np.round(events[:,0] * 1000).astype(int)
events = events[ixs]
events = events[events - 10 * config.conversion > 0]

spikeFile = os.path.join(config.directory, config.spike_file)
temp = _parse_data(spikeFile, events)
mat += temp
mat = np.array(mat)

#
firing_rate = np.mean(mat, axis=1)
for i in range(firing_rate.shape[0]):
    firing_rate[i,:] = np.convolve(firing_rate[i,:], np.ones(config.moving_average_bin)/config.moving_average_bin, 'same')
firing_rate *= config.conversion / config.downsample

ixs = config.cells
for i, ix in enumerate(ixs):
    data = firing_rate[ix]
    _plot_firing_rate(data, [0, config.ylim[i]], figure_path=figure_path, name='firing_rate_{}'.format(ix),
                      config=config)

raster = _raster_convert(mat, raster_bin = config.raster_bin)
for i, ix in enumerate(ixs):
    data = raster[ix]
    _raster_plot(data, figure_path=figure_path, name='raster_image_{}'.format(ix),
                 config=config)

# mean_before = np.mean(firing_rate[:, :t_base-500], axis=1)
# mean_after = np.mean(firing_rate[:, 500+t_base:t_base+t_stim-500], axis=1)
# data = [mean_before, mean_after]
# norm_data = [np.ones_like(mean_before), mean_after/mean_before]
#
# _plot_summary(data, ylim = [0, 22], yticks = [0, 10, 20], figure_path=figure_path, name ='spike_rate')
# _plot_summary(norm_data, ylim = [0, 1.3], yticks = [0, .5, 1],
#               figure_path=figure_path, name ='spike_rate_normalized')