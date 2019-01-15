import numpy as np
from collections import defaultdict
from behavior.behavior_config import behaviorConfig
import filter
from scipy.signal import savgol_filter

from reduce import reduce_by_concat
from tools.utils import append_defaultdicts
import copy

def convert(res, condition):
    '''

    :param res:
    :param condition:
    :return:
    '''
    def _parseLick(mat, start, end):
        mask = mat > 1
        on_off = np.diff(mask, n=1)
        n_licks = np.sum(on_off[start:end] > 0)
        return n_licks

    config = behaviorConfig()
    new_res = defaultdict(list)
    toConvert = ['day', 'mouse']
    res_odorTrials = res['ODOR_TRIALS']
    res_data = res['DAQ_DATA']
    for i, odorTrials in enumerate(res_odorTrials):
        mouse = res['mouse'][i]
        relevant_odors = condition.odors[mouse]
        csps = condition.csp[mouse]
        csms = [x for x in relevant_odors if not np.isin(x, csps)]

        for j, odor in enumerate(odorTrials):
            if odor in relevant_odors:
                start = int(res['DAQ_O_ON'][i] * res['DAQ_SAMP'][i])
                end = int(res['DAQ_W_ON'][i] * res['DAQ_SAMP'][i])
                lick_data = res_data[i][:,res['DAQ_L'][i],j]
                if odor in csms:
                    end += int(config.extra_csm_time * res['DAQ_SAMP'][i])
                n_licks = _parseLick(lick_data, start, end)

                if odor in csms:
                    ix = csms.index(odor)
                    new_res['odor_valence'].append('CS-')
                    new_res['odor_standard'].append('CS-' + str(ix + 1))
                else:
                    ix = csps.index(odor)
                    new_res['odor_valence'].append('CS+')
                    new_res['odor_standard'].append('CS+' + str(ix + 1))

                new_res['odor'].append(odor)
                new_res['lick'].append(n_licks)
                new_res['ix'].append(j)
                for names in toConvert:
                    new_res[names].append(res[names][i])
    for key, val in new_res.items():
        new_res[key] = np.array(val)
    return new_res

def agglomerate_days(res, condition, first_day, last_day):
    mice = np.unique(res['mouse'])
    out = defaultdict(list)

    for i, mouse in enumerate(mice):
        for odor in condition.odors[mouse]:
            filter_dict = {'mouse': mouse, 'day': np.arange(first_day[i], last_day[i]), 'odor': odor}
            filtered_res = filter.filter(res, filter_dict)
            temp_res = reduce_by_concat(filtered_res, 'lick', rank_keys=['day', 'ix'])
            temp_res['trial'] = np.arange(len(temp_res['lick']))
            append_defaultdicts(out, temp_res)
    for key, val in out.items():
        out[key] = np.array(val)
    return out

def analyze_behavior(res, condition, arg = 'normal'):
    '''

    :param res:
    :return:
    '''

    def _half_max_up(vec):
        config = behaviorConfig()
        vec_binary = vec > config.halfmax_up_threshold

        if np.all(vec_binary):
            half_max = 0
        else:
            last_ix_below_threshold = np.where(vec_binary == 0)[0][-1]
            vec_binary[:last_ix_below_threshold] = 0
            if np.any(vec_binary):
                half_max = np.where(vec_binary == 1)[0][0]
            else:
                half_max = None
        return half_max

    def _half_max_down(vec):
        config = behaviorConfig()
        vec_binary = vec < config.halfmax_down_threshold

        if np.all(vec_binary):
            half_max = 0
        else:
            last_ix_below_threshold = np.where(vec_binary == 0)[0][-1]
            vec_binary[:last_ix_below_threshold] = 0
            if np.any(vec_binary):
                half_max = np.where(vec_binary == 1)[0][0]
            else:
                half_max = None
        return half_max

    config = behaviorConfig()
    res['lick_smoothed'] = [savgol_filter(y, config.smoothing_window, config.polynomial_degree)
                            for y in res['lick']]
    res['boolean_smoothed'] = [
        100 * savgol_filter(y > 0, config.smoothing_window_boolean, config.polynomial_degree)
        for y in res['lick']]

    up = []
    for x in res['boolean_smoothed']:
        up.append(_half_max_up(x))

    down = []
    for x in res['boolean_smoothed']:
        down.append(_half_max_down(x))

    for i, odor in enumerate(res['odor']):
        mouse = res['mouse'][i]
        is_csp = np.isin(odor, condition.csp[mouse])
        if arg == 'normal':
            if is_csp:
                res['half_max'].append(up[i])
            else:
                res['half_max'].append(down[i])
        elif arg == 'reversed':
            if is_csp:
                res['half_max'].append(down[i])
            else:
                res['half_max'].append(up[i])
        else:
            raise ValueError('Did not recognize keyword {} for determining half_max'.format(arg))

    for key, val in res.items():
        res[key] = np.array(val)






