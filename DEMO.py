import numpy as np
import os
import CONSTANTS.conditions as conditions
import decoding as dc
import file_io
import glob

condition = conditions.OFC
decode_style = 'valence'
path = r'C:\Users\Peter\PycharmProjects\phd_project\DATA\timepoint\OFC'
mouse_files = [os.path.join(path, o) for o in os.listdir(path)
                    if os.path.isdir(os.path.join(path,o))]

#decode
condition_odors = condition.odors
condition_csps = condition.csp
all_decode_data = []
for i, (mouse_file, odors, csps) in enumerate(zip(mouse_files, condition_odors, condition_csps)):
    data_pathnames = glob.glob(os.path.join(mouse_file, '*.txt'))
    config_pathnames = glob.glob(os.path.join(mouse_file, '*.pkl'))

    mouse_decode_data = []
    for j, data_pathname, config_pathname in enumerate(zip(data_pathnames, config_pathnames)):
        config = file_io.load_pickle(config_pathname)
        data = file_io.load_text(data_pathname)
        scores = dc.decode(config, data, odors, csps, arg=decode_style)
        mouse_decode_data.append((scores, config))
    all_decode_data.append(mouse_decode_data)

# plot
from CONSTANTS.constants import constants
import plots as pt
import matplotlib.pyplot as plt

condition_name = condition.condition
nMouse = len(condition.paths)

r, c = (nMouse, 7)
f, axs = plt.subplots(r,c)
f.set_size_inches(8,6)
ylim = [-5, 105]
yticks = np.arange(0,101,20)
for i, mouse_decode_data in enumerate(all_decode_data):
    for j, (score, config) in enumerate(mouse_decode_data):
        time = np.arange(0, config.TRIAL_FRAMES) * config.TRIAL_PERIOD - config.DAQ_O_ON
        xticks = np.asarray([config.DAQ_O_ON, config.DAQ_O_OFF, config.DAQ_W_ON]) - config.DAQ_O_ON
        xticks = np.round(xticks, 1)
        pt.plot_decoding_performance(time, score, ax=axs[i,j], add_labels=False)
        axs[i,j].set_ylim(ylim)
        axs[i,j].set_yticks(yticks)
        axs[i,j].set_xticks(xticks)
        axs[i,j].set_xticklabels(['','',''])

figpath = os.path.join(constants.LOCAL_FIGURE_PATH, condition_name)
figname = decode_style
if not os.path.exists(figpath):
    os.makedirs(figpath)
figpathname = os.path.join(figpath, figname)
plt.savefig(figpathname + '.png', dpi=300)