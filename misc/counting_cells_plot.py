import numpy as np
import _CONSTANTS.conditions as conditions
import _CONSTANTS.config as config
import os
from tools import file_io
from scipy.ndimage.measurements import center_of_mass
import matplotlib.pyplot as plt
import plot
import matplotlib as mpl

plt.style.use('default')
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.size'] = 7
mpl.rcParams['font.family'] = 'arial'

mouse = 1
condition = conditions.OFC_LONGTERM
pad = 20


data_directory = config.Config().LOCAL_DATA_PATH
data_directory = os.path.join(data_directory, 'registration','ROI', condition.name)
load_path = os.path.join(data_directory, str(mouse) + '.npy')
save_path = os.path.join(config.Config().LOCAL_FIGURE_PATH, 'MISC', 'COUNTING_CELLS')

rois = file_io.load_numpy(load_path)
#ROIT has dimensions YDIM, XDIM, NROI, DAYS
roiT = np.transpose(rois, (1, 2, 3, 0)).astype('float32')

ydim, xdim, nroi, ndays = roiT.shape

for r in range(nroi):
    fig, axs = plt.subplots(1, ndays, figsize = (6, 1))
    for d in range(ndays):
        roi = roiT[:,:,r, d]
        comy, comx = np.round(center_of_mass(roi)).astype(int)
        subset = roi[comy-pad:comy+pad, comx-pad:comx+pad]
        im = axs[d].imshow(subset)
        axs[d].set_title('Day {}'.format(1 + d * 2), fontsize = 7)
        axs[d].axis('off')

        cm = im.get_cmap()
        cm.colors[0] = [0, 0, 0]
        im = axs[d].imshow(subset, cmap=cm)
        fig.suptitle(str(r))

    try:
        plot._easy_save(path=save_path, name=str(r))
    except:
        plt.close()








