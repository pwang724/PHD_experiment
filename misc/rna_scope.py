import os
import glob
from skimage import io
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import defaultdict
import itertools
import plot
import seaborn as sns
from format import *


# d = r'I:\MANUSCRIPT_DATA\RNA_SCOPE\values.xlsx'
d = '/Users/pwang/Desktop/values.xlsx'

name_str ='RNA_SCOPE'
names = ['GCAMP','VGLUT1','VGLUT2']
brain_regions = ['PIR','OFC','MPFC']
xlim = [-.5, 1.5]

# name_str = 'NEUROTRACE'
# names = ['GCAMP', 'NEUROTRACE']
# brain_regions = ['OFC_NEUROTRACE','MPFC_NEUROTRACE']
# xlim = [-1, 1]

permutations = list(itertools.permutations(names, 2))

dict = defaultdict(list)
#load data
for i in range(len(brain_regions)):
    dict['condition'].append(brain_regions[i])

    ws = pd.read_excel(d, brain_regions[i])
    for name in names:
        data = np.array(ws[name]) == 'Y'
        data = data.astype(int)
        dict[name].append(data)

#get overlap
stats_dict = defaultdict(list)
for i in range(len(dict['condition'])):
    for pair in permutations:
        name = pair[0] + '/' + pair[1]
        numerator = dict[pair[0]][i]
        denominator = dict[pair[1]][i]
        mask = denominator.astype(np.bool)
        relevant = numerator[mask]
        fraction = relevant.sum() / relevant.size
        stats_dict['name'].append(name)
        stats_dict['numerator'].append(pair[0])
        stats_dict['denominator'].append(pair[1])
        stats_dict['value'].append(fraction)
        stats_dict['condition'].append(dict['condition'][i])

for k,v in stats_dict.items():
    stats_dict[k] = np.array(v)

ax_args = {'ylim':[0, 1.1], 'yticks':[0, .5, 1], 'xlim':xlim}
cat_args = {'alpha': .5, 'hue':'condition'}
denominator = 'GCAMP'
plot.plot_results(stats_dict, x_key= 'name', y_key='value',
                  select_dict= {'denominator':denominator},
                  plot_function=sns.barplot,
                  plot_args = cat_args,
                    fig_size=[2.5, 2],
                  ax_args=ax_args,
                  path = 'test', name_str=name_str)

numerator = 'GCAMP'
cat_args = {'alpha': .5, 'hue':'condition'}
plot.plot_results(stats_dict, x_key= 'name', y_key='value',
                  select_dict= {'numerator':numerator},
                  plot_function=sns.barplot,
                  plot_args = cat_args,
                    fig_size=[2.5, 2],
                  ax_args=ax_args,
                  path = 'test', name_str=name_str)

numerator = 'VGLUT1'
denominator = 'VGLUT2'
cat_args = {'alpha': .5, 'hue':'condition'}
ax_args.update({'xlim':[-1, 1]})
plot.plot_results(stats_dict, x_key= 'name', y_key='value',
                  select_dict= {'numerator':numerator,'denominator':denominator},
                  plot_function=sns.barplot,
                  plot_args = cat_args,
                    fig_size=[2.5, 2],
                  ax_args=ax_args,
                  path = 'test', name_str=name_str)




