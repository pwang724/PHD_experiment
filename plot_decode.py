from CONSTANTS.config import Config
from CONSTANTS import conditions
import tools.plots as pt
import matplotlib.pyplot as plt
import numpy as np
import os

condition = conditions.OFC
condition_name = condition.name
decode_style = 'valence'
save_path = os.path.join(Config.LOCAL_ANALYSIS_PATH, condition.name, decode_style)

nMouse = len(condition.paths)
r, c = (nMouse, 7)
f, axs = plt.subplots(r,c)
f.set_size_inches(8,6)
ylim = [-5, 105]
yticks = np.arange(0,101,20)
for i, mouse_decode_data in enumerate(all_decode_data):
    for j, (score, cons) in enumerate(mouse_decode_data):
        time = np.arange(0, cons.TRIAL_FRAMES) * cons.TRIAL_PERIOD - cons.DAQ_O_ON
        xticks = np.asarray([cons.DAQ_O_ON, cons.DAQ_O_OFF, cons.DAQ_W_ON]) - cons.DAQ_O_ON
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