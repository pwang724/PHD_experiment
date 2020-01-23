import os
import glob
from skimage import io
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import defaultdict
import itertools



d = r'I:\MANUSCRIPT_DATA\RNA_SCOPE\values.xlsx'

names = ['GCAMP','VGLUT1','VGLUT2']
brain_regions = ['PIR','OFC','MPFC']
sheets = [0, 1, 2]

permutations = list(itertools.permutations(names, 2))

dict = defaultdict(list)
#load data
for i in range(len(brain_regions)):
    dict['condition'].append(brain_regions[i])

    ws = pd.read_excel(d, sheet = sheets[i])
    for name in names:
        data = np.array(ws[name]) == 'Y'
        data = data.astype(int)
        dict[name].append(data)

#get overlap
stats_dict = defaultdict(list)
for i in range(len(brain_regions)):
    for pair in permutations:
        name = pair[0] + '/' + pair[1]
        numerator = dict[pair[0]][i]
        denominator = dict[pair[1]][i]
        mask = denominator.astype(np.bool)
        relevant = numerator[mask]
