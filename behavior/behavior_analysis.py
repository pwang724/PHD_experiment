import numpy as np
from collections import defaultdict

from analysis import add_odor_value
from behavior.behavior_config import behaviorConfig
import filter
import reduce
from scipy.signal import savgol_filter
from reduce import append_defaultdicts, reduce_by_concat
import analysis

def _get_days_per_condition(data_path, condition, odor_valence = None):
    res = analysis.load_all_cons(data_path)
    analysis.add_indices(res)
    lick_res = convert(res, condition)
    add_odor_value(lick_res, condition)
    if odor_valence is not None:
        lick_res = filter.filter(lick_res, {'odor_valence': odor_valence})

    days = []
    list_of_dates = lick_res['day']
    list_of_mice = lick_res['mouse']
    _, mouse_ixs = np.unique(list_of_mice, return_inverse=True)
    for mouse_ix in np.unique(mouse_ixs):
        mouse_dates = list_of_dates[mouse_ixs == mouse_ix]
        counts = np.unique(mouse_dates)
        days.append(counts)
    return days

def get_days_per_mouse(data_path, condition, odor_valence ='CS+'):
    res = analysis.load_all_cons(data_path)
    analysis.add_indices(res)
    days_per_mouse = np.array(_get_days_per_condition(data_path, condition, odor_valence))
    first_day_per_mouse = np.array([x[0] for x in days_per_mouse])
    last_day_per_mouse = np.array([x[-1] for x in days_per_mouse])

    if hasattr(condition, 'csp') or hasattr(condition, 'pt_csp'):
        res_behavior = analyze_behavior(data_path, condition)
        res_behavior_csp = filter.filter(res_behavior, {'odor_valence': odor_valence})
        res_behavior_summary = reduce.filter_reduce(res_behavior_csp, filter_key='mouse', reduce_key='learned_day')
        mice, ix = np.unique(res_behavior_summary['mouse'], return_inverse=True)
        temp = res_behavior_summary['learned_day'][ix]
        temp[temp == None] = last_day_per_mouse[temp == None]
        temp[np.isnan(temp)] = last_day_per_mouse[np.isnan(temp)]
        learned_days_per_mouse = np.ceil(temp)
    else:
        learned_days_per_mouse = last_day_per_mouse
    return learned_days_per_mouse, last_day_per_mouse

def get_licks_per_day(data_path, condition):
    res = analysis.load_all_cons(data_path)
    analysis.add_indices(res)
    analysis.add_time(res)
    lick_res = convert(res, condition)
    lick_res['lick_boolean'] = np.array([y > 0 for y in lick_res['lick']])
    out = reduce.new_filter_reduce(lick_res, ['odor','day','mouse'], 'lick_boolean')
    return out


def analyze_behavior(data_path, condition):
    res = analysis.load_all_cons(data_path)
    analysis.add_indices(res)
    analysis.add_time(res)
    lick_res = convert(res, condition)
    days_per_mouse = _get_days_per_condition(data_path, condition)
    last_day_per_mouse = np.array([x[-1] for x in days_per_mouse])
    plot_res = agglomerate_days(lick_res, condition, condition.training_start_day, last_day_per_mouse)
    add_odor_value(plot_res, condition)
    add_behavior_stats(plot_res)
    return plot_res

def convert(res, condition, includeRaw = False):
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

        if hasattr(condition, 'csp'):
            #standard
            relevant_odors = condition.odors[mouse]
            csps = condition.csp[mouse]
            csms = [x for x in relevant_odors if not np.isin(x, csps)]
        else:
            #composite
            relevant_odors = condition.dt_odors[mouse] + condition.pt_odors[mouse]
            csps = condition.pt_csp[mouse] + condition.dt_csp[mouse]
            csms = [x for x in relevant_odors if not np.isin(x, csps)]

        for j, odor in enumerate(odorTrials):
            if odor in relevant_odors:
                start = int(res['DAQ_O_ON'][i] * res['DAQ_SAMP'][i])
                start = int((res['DAQ_W_ON'][i]-1) * res['DAQ_SAMP'][i])
                end = int(res['DAQ_W_ON'][i] * res['DAQ_SAMP'][i])
                lick_data = res_data[i][:,res['DAQ_L'][i],j]

                if odor in csms:
                    end += int(config.extra_csm_time * res['DAQ_SAMP'][i])
                n_licks = _parseLick(lick_data, start, end)

                new_res['odor'].append(odor)
                new_res['lick'].append(n_licks)
                new_res['ix'].append(j)
                if includeRaw:
                    new_res['lick_raw_data'].append(lick_data)
                for names in toConvert:
                    new_res[names].append(res[names][i])
    for key, val in new_res.items():
        new_res[key] = np.array(val)
    return new_res

def agglomerate_days(res, condition, first_day, last_day):
    mice = np.unique(res['mouse'])
    out = defaultdict(list)

    for i, mouse in enumerate(mice):
        if hasattr(condition, 'csp'):
            odors = condition.odors[mouse]
        else:
            odors = condition.dt_odors[mouse] + condition.pt_odors[mouse]
        for odor in odors:
            filter_dict = {'mouse': mouse, 'day': np.arange(first_day[i], last_day[i]+1), 'odor': odor}
            filtered_res = filter.filter(res, filter_dict)
            temp_res = reduce_by_concat(filtered_res, 'lick', rank_keys=['day', 'ix'])
            temp_res['day'] = np.array(sorted(filtered_res['day']))
            temp_res['trial'] = np.arange(len(temp_res['lick']))
            append_defaultdicts(out, temp_res)
    for key, val in out.items():
        out[key] = np.array(val)
    return out


def add_behavior_stats(res, arg ='normal'):
    '''

    :param res:
    :return:
    '''
    def _criterion_up(vec, threshold):
        vec_binary = vec > threshold
        if np.all(vec_binary):
            out = 0
        else:
            if np.any(vec_binary):
                out = np.where(vec_binary == 1)[0][0]
            else:
                out = len(vec_binary)
        return out

    def _criterion_down(vec, threshold):
        vec_binary = vec < threshold
        if np.all(vec_binary):
            out = 0
        else:
            if np.any(vec_binary):
                out = np.where(vec_binary == 1)[0][0]
            else:
                out = len(vec_binary)
        return out

    def _half_max_up(vec, threshold):
        vec_binary = vec > threshold

        if np.all(vec_binary):
            half_max = 0
        else:
            last_ix_below_threshold = np.where(vec_binary == 0)[0][-1]
            vec_binary[:last_ix_below_threshold] = 0
            if np.any(vec_binary):
                half_max = np.where(vec_binary == 1)[0][0]
            else:
                half_max = len(vec_binary)
        return half_max

    def _half_max_down(vec, threshold):
        vec_binary = vec < threshold

        if np.all(vec_binary):
            half_max = 0
        else:
            last_ix_below_threshold = np.where(vec_binary == 0)[0][-1]
            vec_binary[:last_ix_below_threshold] = 0
            if np.any(vec_binary):
                half_max = np.where(vec_binary == 1)[0][0]
            else:
                half_max = len(vec_binary)
        return half_max

    config = behaviorConfig()
    res['lick_smoothed'] = [savgol_filter(y, config.smoothing_window, config.polynomial_degree)
                            for y in res['lick']]
    res['boolean_smoothed'] = [
        100 * savgol_filter(y > 0, config.smoothing_window_boolean, config.polynomial_degree)
        for y in res['lick']]
    for x in res['boolean_smoothed']:
        x[x>100] = 100
    res['false_negative'] = [100 - x for x in res['boolean_smoothed']]

    up_half = []
    up_criterion = []
    for x in res['boolean_smoothed']:
        up_half.append(_half_max_up(x, config.halfmax_up_threshold))
        up_criterion.append(_criterion_up(x, config.fully_learned_threshold_up))

    down_half = []
    down_criterion = []
    for x in res['boolean_smoothed']:
        down_half.append(_half_max_down(x, config.halfmax_down_threshold))
        down_criterion.append(_criterion_down(x, config.fully_learned_threshold_down))

    for i, odor_valence in enumerate(res['odor_valence']):
        is_csp = odor_valence == 'CS+' or odor_valence == 'PT CS+'
        if arg == 'normal':
            if is_csp:
                res['half_max'].append(up_half[i])
                res['criterion'].append(up_criterion[i])
            else:
                res['half_max'].append(down_half[i])
                res['criterion'].append(down_criterion[i])
        elif arg == 'reversed':
            if is_csp:
                res['half_max'].append(down_half[i])
                res['criterion'].append(down_criterion[i])
            else:
                res['half_max'].append(up_half[i])
                res['criterion'].append(up_criterion[i])
        else:
            raise ValueError('Did not recognize keyword {} for determining half_max'.format(arg))


    for i, half_max in enumerate(res['half_max']):
        odor_valence = res['odor_valence'][i]
        y = res['lick'][i] > 0
        days = res['day'][i]
        unique_days, ixs = np.unique(days, return_inverse=True)
        mean_licks_per_day = []
        for i, day in enumerate(unique_days):
            ix = ixs == i
            mean_licks_per_day.append(np.mean(y[ix]))
        mean_licks_per_day = np.array(mean_licks_per_day) * 100

        if odor_valence == 'CS+' or odor_valence == 'PT CS+':
            if any(mean_licks_per_day > config.fully_learned_threshold_up):
                ix = np.argmax(mean_licks_per_day > config.fully_learned_threshold_up)
                day = unique_days[ix]
            else:
                day = None
        elif odor_valence == 'CS-':
            if any(mean_licks_per_day < config.fully_learned_threshold_down):
                ix = np.argmax(mean_licks_per_day < config.fully_learned_threshold_down)
                day = unique_days[ix]
            else:
                day=None
        elif odor_valence == 'PT Naive':
                day=None
        else:
            raise ValueError('Odor Valence is not known {}'.format(odor_valence))
        res['learned_day'].append(day)

    for key, val in res.items():
        res[key] = np.array(val)


