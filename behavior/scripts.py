from collections import defaultdict

import numpy as np

import filter
from reduce import reduce_by_mean
from tools.utils import append_defaultdicts


def get_summary(res, condition):
    # summarize data
    new_res = defaultdict(list)
    mice = np.unique(res['mouse'])
    for i, mouse in enumerate(mice):
        filter_dict = {'mouse': mouse, 'odor': condition.csp[i]}
        cur_res = filter.filter(res, filter_dict)
        temp_res = reduce_by_mean(cur_res, 'half_max')
        append_defaultdicts(new_res, temp_res)
    for key, val in new_res.items():
        new_res[key] = np.array(val)
    return new_res
