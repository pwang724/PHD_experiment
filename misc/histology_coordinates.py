import os
import glob
from skimage import io
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from collections import defaultdict
import itertools
import plot
import seaborn as sns
from _CONSTANTS.config import Config
import filter
import reduce
import copy

plt.style.use('default')
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.size'] = 7
mpl.rcParams['font.family'] = 'arial'

def _string_splitter(str, length = 6):
    str = str.replace(')(', ',')
    # str = str.replace('),(', ',')
    str = str.replace('(', '')
    str = str.replace(')', '')
    str = str.split(',')
    assert len(str) == length, 'did not retrieve 6 numbers'
    numbers = np.array(str).astype(np.float)
    return numbers

def filter_subset(dict, key, value, exclude=False):
    res = copy.copy(dict)
    ix = [value in x for x in res[key]]
    ix = np.array(ix)
    if exclude:
        ix = np.invert(ix)
    for k, v in res.items():
        res[k] = v[ix]
    return res

def _add_jitter(x, jitter_range):
    x = x + np.random.uniform(0, jitter_range, size= x.shape)
    return x

d = r'C:\Users\P\Dropbox\LAB\MANUSCRIPTS\master_mouse_list.xlsx'
brain_regions = ['OFC','MPFC']

save_path = os.path.join(Config.LOCAL_FIGURE_PATH, 'MISC', 'RNA_SCOPE')
ws = pd.read_excel(d, 0)
dict = defaultdict(list)
for name in ws.columns:
    data = np.array(ws[name])
    for d in data:
        dict[name].append(d)
for k,v in dict.items():
    dict[k] = np.array(v)

arg_type = 'both' #moving, headfixed, both
arg = 'OFC' #OFC, MPFC, OUTPUT

res = filter_subset(dict, 'commentary', 'EXCLUDED', exclude=True)
res = filter_subset(res, 'experiment', 'IMAGING', exclude=True)
res = filter_subset(res, 'experiment', 'JAWS', exclude=True)
res = filter_subset(res, 'coordinates fiber (AP, ML, DV)', 'CRISTIAN', exclude=True) #TODO

if arg_type == 'moving':
    res = filter_subset(res, 'experiment', 'FREELY MOVING')
elif arg_type == 'headfixed':
    res = filter_subset(res, 'experiment', 'FREELY MOVING', exclude=True)
elif arg_type == 'both':
    pass

if arg != 'OUTPUT':
    res = filter_subset(res, 'experiment', arg)
    res = filter_subset(res, 'experiment', 'OUTPUT', exclude=True)
elif arg == 'OUTPUT':
    res = filter_subset(res, 'experiment', 'OUTPUT')


if arg == 'MPFC':
    min, max = 1.6, 2.1
    ylim = [-2.5, -1]
    yticks = [0, -1, -2, -3]
    xlim = [-.6, .6]
    xticks = [-0.6, -.3, 0, .3, .6]
    length = 6
elif arg == 'OFC':
    min, max = 2.1, 2.7
    ylim = [-3, -2]
    yticks = [0, -1, -2, -3]
    xlim = [-.6, .6]
    xticks = [-1.5, -1, -.5,  0, .5, 1, 1.5]
    length = 6
elif arg == 'OUTPUT':
    min, max = 2.3, 2.7
    ylim = [-3, -1]
    yticks = [-1, -2, -3]
    xlim = [0, 2]
    xticks = [0, .5, 1, 1.5, 2]
    length = 3
else:
    raise ValueError('')

for i, d in enumerate(res['coordinates fiber (AP, ML, DV)']):
    try:
        numbers = _string_splitter(d, length= length)
        res['ap_l'].append(numbers[0])
        res['ml_l'].append(numbers[1])
        res['dv_l'].append(numbers[2])
        if length > 3:
            res['ap_r'].append(numbers[3])
            res['ml_r'].append(numbers[4])
            res['dv_r'].append(numbers[5])
    except:
        raise ValueError(res['experiment'][i], res['mouse'][i])

for k, v in res.items():
    res[k] = np.array(v)

print(res.keys())
print(res['experiment'])
fig = plt.figure(figsize=[3,2])
ax = fig.add_axes([.2, .2, .6, .6])
ax1 = fig.add_axes([0.82, 0.2, 0.02, 0.6])

hemispheres = [['ml_l', 'dv_l', 'ap_l'], ['ml_r', 'dv_r', 'ap_r']]
hemispheres = hemispheres if length == 6 else [hemispheres[0]]
for side in hemispheres:
    x = res[side[0]]
    y = - res[side[1]]
    z = res[side[2]]
    x = _add_jitter(x, 0.1)
    y = _add_jitter(y, 0.1)
    z = _add_jitter(z, 0.1)
    print(z.min())
    print(z.max())
    scaled_z = (z - min) / (max-min)
    colors = plt.cm.cool(scaled_z)
    ax.scatter(x, y, edgecolors = '', c = colors, s = 5, alpha = 0.7)

ax.set_ylim(ylim)
ax.set_yticks(yticks)
ax.set_xlim(xlim)
ax.set_xticks(xticks)
ax.set_ylabel('DV')
ax.set_xlabel('ML')
plt.axis('tight')
for loc in ['top', 'right']:
    ax.spines[loc].set_visible(False)
ax.tick_params('in', length=0.25)

cmap = mpl.cm.cool
norm = mpl.colors.Normalize(vmin=min, vmax= max)
cb = mpl.colorbar.ColorbarBase(ax1, cmap=cmap, norm=norm)
cb.set_ticks(np.arange(min,max+.01,.1))
cb.outline.set_linewidth(0.5)
cb.set_label('AP', fontsize=7, labelpad=0)
plt.tick_params(axis='both', which='major', labelsize=7)
plt.axis('tight')

save_path = os.path.join(Config.LOCAL_FIGURE_PATH, 'MISC', 'HISTOLOGY')
plot._easy_save(save_path, arg_type + '_' + arg)

if arg == 'OUTPUT':
    min, max = 1.0, 1.8
    ylim = [-5, -3]
    yticks = [-3, -4, -5]
    xlim = [0, 4]
    xticks = [0, 1, 2, 3, 4]

    for i, d in enumerate(res['coordinates virus (AP, ML, DV)']):
        try:
            numbers = _string_splitter(d, length=length)
            res['ap_r_v'].append(numbers[0])
            res['ml_r_v'].append(numbers[1])
            res['dv_r_v'].append(numbers[2])
        except:
            raise ValueError(res['experiment'][i], res['mouse'][i])

    for k, v in res.items():
        res[k] = np.array(v)

    print(res.keys())
    print(res['experiment'])
    fig = plt.figure(figsize=[3, 2])
    ax = fig.add_axes([.2, .2, .6, .6])
    ax1 = fig.add_axes([0.82, 0.2, 0.02, 0.6])

    hemispheres = [['ml_r_v', 'dv_r_v', 'ap_r_v']]
    for side in hemispheres:
        x = res[side[0]]
        y = - res[side[1]]
        z = res[side[2]]
        x = _add_jitter(x, 0.1)
        y = _add_jitter(y, 0.1)
        z = _add_jitter(z, 0.1)
        print(z.min())
        print(z.max())
        scaled_z = (z - min) / (max - min)
        colors = plt.cm.cool(scaled_z)
        ax.scatter(x, y, edgecolors='', c=colors, s=5, alpha=0.7)

    ax.set_yticks(yticks)
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    ax.set_xticks(xticks)
    ax.set_ylabel('DV')
    ax.set_xlabel('ML')
    plt.axis('tight')
    for loc in ['top', 'right']:
        ax.spines[loc].set_visible(False)
    ax.tick_params('in', length=0.25)

    cmap = mpl.cm.cool
    norm = mpl.colors.Normalize(vmin=min, vmax=max)
    cb = mpl.colorbar.ColorbarBase(ax1, cmap=cmap, norm=norm)
    cb.set_ticks(np.arange(min, max + .01, .2))
    cb.outline.set_linewidth(0.5)
    cb.set_label('AP', fontsize=7, labelpad=0)
    plt.tick_params(axis='both', which='major', labelsize=7)
    plt.axis('tight')

    save_path = os.path.join(Config.LOCAL_FIGURE_PATH, 'MISC', 'HISTOLOGY')
    plot._easy_save(save_path, arg_type + '_' + arg + '_virus')
