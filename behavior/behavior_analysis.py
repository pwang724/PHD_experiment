import numpy as np
from collections import defaultdict
from behavior.behavior_config import behaviorConfig
import filter
import reduce
from scipy.signal import savgol_filter
from reduce import append_defaultdicts, reduce_by_concat
import analysis

def _get_last_day_per_mouse(res):
    '''
    returns the value of the last day of imaging per mouse
    :param res: flattened dict of results
    :return: list of last day per each mouse
    '''
    out = []
    list_of_dates = res['NAME_DATE']
    list_of_mice = res['NAME_MOUSE']
    _, mouse_ixs = np.unique(list_of_mice, return_inverse=True)
    for mouse_ix in np.unique(mouse_ixs):
        mouse_dates = list_of_dates[mouse_ixs == mouse_ix]
        counts = np.unique(mouse_dates).size - 1
        out.append(counts)
    return out

def get_days_per_mouse(data_path, condition):
    res = analysis.load_all_cons(data_path)
    last_day_per_mouse = np.array(_get_last_day_per_mouse(res))

    if hasattr(condition, 'csp'):
        res_behavior = analyze_behavior(data_path, condition)
        res_behavior_csp = filter.filter(res_behavior, {'odor_valence': 'CS+'})
        res_behavior_summary = reduce.filter_reduce(res_behavior_csp, filter_key='mouse', reduce_key='learned_day')
        mice, ix = np.unique(res_behavior_summary['mouse'], return_inverse=True)
        temp = res_behavior_summary['learned_day'][ix]
        temp[temp == None] = last_day_per_mouse[temp == None]
        learned_days_per_mouse = np.ceil(temp.astype(int))
    else:
        learned_days_per_mouse = last_day_per_mouse
    return learned_days_per_mouse, last_day_per_mouse

def analyze_behavior(data_path, condition):
    res = analysis.load_all_cons(data_path)
    analysis.add_indices(res)
    analysis.add_time(res)
    lick_res = convert(res, condition)
    plot_res = agglomerate_days(lick_res, condition, condition.training_start_day,
                                _get_last_day_per_mouse(res))
    add_odor_value(plot_res, condition)
    add_behavior_stats(plot_res)
    return plot_res

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
        y = res['boolean_smoothed'][i]
        days = res['day'][i]
        y_ = np.copy(y)
        y_[:half_max] = 0

        if odor_valence == 'CS+' or odor_valence == 'PT CS+':
            if any(y_ > config.fully_learned_threshold_up):
                ix = np.argmax(y_ > config.fully_learned_threshold_up)
                day = days[ix]
            else:
                day = None
        elif odor_valence == 'CS-':
            if any(y_ < config.fully_learned_threshold_down):
                ix = np.argmax(y_ < config.fully_learned_threshold_down)
                day = days[ix]
            else:
                day=None
        elif odor_valence == 'PT Naive':
                day=None
        else:
            raise ValueError('Odor Valence is not known {}'.format(odor_valence))
        res['learned_day'].append(day)

    for key, val in res.items():
        res[key] = np.array(val)


def add_odor_value(res, condition):
    mice, ix = np.unique(res['mouse'], return_inverse=True)
    valence_array = np.zeros_like(res['odor']).astype(object)
    standard_array = np.zeros_like(res['odor']).astype(object)

    for i, mouse in enumerate(mice):
        if hasattr(condition, 'odors'):
            odors = condition.odors[i]
            csps = condition.csp[i]
            csms = [x for x in odors if not np.isin(x, csps)]
            standard_dict = {}
            valence_dict = {}
            j=1
            for csp in csps:
                standard_dict[csp] = 'CS+' + str(j)
                valence_dict[csp] = 'CS+'
                j+=1
            j=1
            for csm in csms:
                standard_dict[csm] = 'CS-' + str(j)
                valence_dict[csm] = 'CS-'
                j+=1
        else:
            dt_odors = condition.dt_odors[i]
            dt_csp = condition.dt_csp[i]
            dt_csm = [x for x in dt_odors if not np.isin(x, dt_csp)]
            pt_odors = condition.pt_odors[i]
            pt_csp = condition.pt_csp[i]
            pt_naive = [x for x in pt_odors if not np.isin(x, pt_csp)]
            assert len(pt_naive) <= 1, 'More than 1 pt naive odor'
            assert len(pt_csp) <= 1, 'More than 1 pt CS+ odor'
            standard_dict = {}
            valence_dict = {}
            j=1
            for csp in dt_csp:
                standard_dict[csp] = 'CS+' + str(j)
                valence_dict[csp] = 'CS+'
                j+=1
            j=1
            for csm in dt_csm:
                standard_dict[csm] = 'CS-' + str(j)
                valence_dict[csm] = 'CS-'
                j+=1
            if len(pt_naive):
                standard_dict[pt_naive[0]] = 'PT Naive'
                valence_dict[pt_naive[0]] = 'PT Naive'
            standard_dict[pt_csp[0]] = 'PT CS+'
            valence_dict[pt_csp[0]] = 'PT CS+'
            j += 1

        mouse_ix = ix == i
        mouse_odors = res['odor'][mouse_ix]
        valence_array[mouse_ix] = [valence_dict[o] for o in mouse_odors]
        standard_array[mouse_ix] = [standard_dict[o] for o in mouse_odors]
    res['odor_valence'] = valence_array
    res['odor_standard'] = standard_array