from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import _CONSTANTS.conditions as conditions
import _CONSTANTS.config as config
import os
import glob
from collections import defaultdict
from tools import file_io
import skimage.io._plugins.tifffile_plugin as fi
from itertools import combinations
import plot

def within_corr_metric(imstack):
    imR = imstack.reshape(imstack.shape[0], -1)
    meanImR = np.mean(imR, keepdims=True, axis=0)
    cov = np.corrcoef(meanImR, imR)
    corr_vec = cov[0,1:]
    return corr_vec

def within_crisp_metric(im):
    grads = np.gradient(im)
    grad_vec = np.sqrt(np.square(grads[0]) + np.square(grads[1]))
    crispness = np.linalg.norm(grad_vec, ord='fro')
    return crispness

def across_corr_metric(imstack):
    imR = imstack.reshape(imstack.shape[0], -1)
    cov = np.corrcoef(imR)
    return cov

test = True
#PIR: M199 has an issue = .6

#TODO: have not analyzed piriform naive, ofc reversal

condition = conditions.OFC_COMPOSITE

X_exclude, Y_exclude = 15, 15

if test:
    numImages = 3
else:
    numImages = 10

data_directory = config.Config().LOCAL_DATA_PATH
data_directory = os.path.join(data_directory, 'registration','motion_metrics')

ds = condition.paths
for d in ds:
    image_directories = glob.glob(os.path.join(d, 'imRAW', '*__*'))
    imData = []
    for image_directory in image_directories:
        images = glob.glob(os.path.join(image_directory, '*.tif'))
        ims_per_day = []
        for i in range(numImages):
            temp = io.imread(images[i])
            if X_exclude > 0 and Y_exclude >0:
                temp = temp[:, X_exclude:-X_exclude, Y_exclude:-Y_exclude]
            ims_per_day.append(temp)
        imData.append(ims_per_day)

    #imData has dimensions DAYS X N_IMAGES_PER_DAY X (FRAMES X DIM_X X DIM_Y)
    imData = np.array(imData)

    #within day metric
    within_day_corrs = []
    within_day_crisp = []
    for ims in imData:
        ims = np.stack(ims, axis=0)
        ims = ims.reshape(-1, ims.shape[2],ims.shape[3])
        corr = within_corr_metric(ims)
        crisp = within_crisp_metric(ims.mean(axis=0))
        within_day_corrs.append(corr)
        within_day_crisp.append(crisp)
    within_day_corr_average = [np.mean(x) for x in within_day_corrs]

    within_day_mean_corrs = []
    for ims in imData:
        ims = np.stack(ims, axis=0)
        mean_ims = np.mean(ims, axis=1) #average across frames, get mean image
        cov = across_corr_metric(mean_ims)
        cov_average = cov[~np.eye(cov.shape[0], dtype=bool)].mean()
        within_day_mean_corrs.append(cov_average)

    #across-day metric
    for i in range(numImages):
        mean_ims = []
        for ims in imData:
            ims = np.stack(ims, axis=0)
            ims = ims[i]
            mean_ims.append(ims.mean(axis=0))
        mean_ims = np.stack(mean_ims, axis=0)
        across_day_mean_corrs = across_corr_metric(mean_ims)
        across_day_mean_corrs_average = across_day_mean_corrs[~np.eye(across_day_mean_corrs.shape[0], dtype=bool)].mean()

    imdir = os.path.join(d,'python_reg.tif')
    fi.imsave(imdir, mean_ims.astype('uint16'), photometric='minisblack')
    for i, im in enumerate(mean_ims):
        plt.imshow(im, cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plot._easy_save(d, str(i))

    #saving
    p, experiment = os.path.split(d)
    p, mouse = os.path.split(p)
    save_dir = os.path.join(data_directory, condition.name)
    res = defaultdict(list)

    res['within_day_corrs'].append(within_day_corrs)
    res['within_day_corrs_average'].append(within_day_corr_average)
    res['within_day_crisp'].append(within_day_crisp)
    res['within_day_crisp_average'].append(np.mean(within_day_crisp))
    res['across_day_mean_corrs'].append(across_day_mean_corrs)
    res['across_day_mean_corrs_average'].append(across_day_mean_corrs_average)
    # res['within_day_mean_corrs'].append(within_day_mean_corrs)
    res['mouse'].append(mouse)
    res['experiment'].append(condition.name)

    for k, v in res.items():
        res[k] = np.array(v)
        if k in ['mouse', 'within_day_crisp', 'across_day_mean_corrs_average']:
            print('{}: {}'.format(k, v))

    file_io.save_pickle(save_path=save_dir, save_name=mouse, data=res)





