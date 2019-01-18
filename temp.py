import os
from collections import defaultdict

import filter
from CONSTANTS import conditions as experimental_conditions
from CONSTANTS.config import Config
from behavior.behavior_analysis import analyze_behavior
from reduce import filter_reduce
from tools.utils import chain_defaultdicts
import plot
import matplotlib.pyplot as plt
import numpy as np
import analysis

core_experiments = ['individual', 'individual_half_max', 'summary','basic_3']
# experiments = ['individual', 'individual_half_max', 'basic_3']
conditions = [experimental_conditions.PIR, experimental_conditions.OFC, experimental_conditions.BLA,
              experimental_conditions.OFC_LONGTERM, experimental_conditions.BLA_LONGTERM,
              experimental_conditions.OFC_JAWS, experimental_conditions.BLA_JAWS]

experiments = core_experiments
condition = experimental_conditions.PIR


data_path = os.path.join(Config.LOCAL_DATA_PATH, Config.LOCAL_DATA_TIMEPOINT_FOLDER, condition.name)
plot_res = analyze_behavior(data_path, condition)

filtered_res = filter.filter(plot_res, {'odor_valence': 'CS+'})
summary_res = filter_reduce(filtered_res, filter_key='mouse', reduce_key='learned_day')
mice, ix = np.unique(summary_res['mouse'], return_inverse=True)
learned_days = summary_res['learned_day'][ix]
print(learned_days)