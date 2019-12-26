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
    if odor_valence is not None:
        add_odor_value(lick_res, condition)
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
        res_behavior_summary = reduce.filter_reduce(res_behavior_csp, filter_keys='mouse', reduce_key='learned_day')
        mice, ix = np.unique(res_behavior_summary['mouse'], return_inverse=True)
        temp = res_behavior_summary['learned_day'][ix]
        temp[temp == None] = last_day_per_mouse[temp == None]
        temp[np.isnan(temp)] = last_day_per_mouse[np.isnan(temp)]
        learned_days_per_mouse = np.ceil(temp)
    else:
        learned_days_per_mouse = last_day_per_mouse
    return learned_days_per_mouse, last_day_per_mouse

def get_licks_per_day(data_path, condition, return_raw=False):
    res = analysis.load_all_cons(data_path)
    analysis.add_indices(res)
    analysis.add_time(res)
    lick_res = convert(res, condition)
    lick_res['lick_boolean'] = np.array([y > 0 for y in lick_res['lick']])
    out = reduce.new_filter_reduce(lick_res, ['odor','day','mouse'], 'lick_boolean')
    if return_raw:
        add_odor_value(lick_res, condition)
        return lick_res
    else:
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
    add_behavior_stats(plot_res, condition)
    return plot_res

def convert(res, condition, includeRaw = False):
    '''

    :param res:
    :param condition:
    :return:
    '''
    def _get_number_of_licks(mat, start, end):
        mask = mat > 1
        on_off = np.diff(mask, n=1)
        n_licks = np.sum(on_off[start:end] > 0)
        return n_licks

    def _get_time_of_first_lick(mat, start, end, sample_rate):
        mask = mat > 1
        on_off = np.diff(mask, n=1)[start:end]
        if any(on_off):
            ix = np.argwhere(on_off)[0][0]
            ix /= sample_rate
        else:
            ix = -1
        return ix

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
        elif hasattr(condition, 'dt_csp'):
            #composite
            relevant_odors = condition.dt_odors[mouse] + condition.pt_odors[mouse]
            csps = condition.pt_csp[mouse] + condition.dt_csp[mouse]
            csms = [x for x in relevant_odors if not np.isin(x, csps)]
        else:
            raise ValueError('cannot find odors')
            # relevant_odors = condition.odors[mouse]
            # csps = relevant_odors[:2]
            # csms = relevant_odors[2:]

        for j, odor in enumerate(odorTrials):
            if odor in relevant_odors:
                # start = int(res['DAQ_O_OFF'][i] * res['DAQ_SAMP'][i])
                # start = int((res['DAQ_O_ON'][i]) * res['DAQ_SAMP'][i])
                start_odor = int((res['DAQ_O_ON'][i]) * res['DAQ_SAMP'][i])
                start = int((res['DAQ_W_ON'][i]-1) * res['DAQ_SAMP'][i])
                end = int(res['DAQ_W_ON'][i] * res['DAQ_SAMP'][i])
                end_coll = int((res['DAQ_W_ON'][i] + 1) * res['DAQ_SAMP'][i])
                end_coll_time = int((res['DAQ_W_ON'][i] + 5) * res['DAQ_SAMP'][i])
                lick_data = res_data[i][:,res['DAQ_L'][i],j]

                if odor in csms:
                    end += int(config.extra_csm_time * res['DAQ_SAMP'][i])

                time_first_lick = _get_time_of_first_lick(lick_data, start_odor, end, res['DAQ_SAMP'][i])
                time_first_lick_collection = _get_time_of_first_lick(lick_data, end, end_coll_time, res['DAQ_SAMP'][i])
                n_licks_baseline = _get_number_of_licks(lick_data, 0, start_odor)
                n_licks = _get_number_of_licks(lick_data, start, end)
                n_licks_coll = _get_number_of_licks(lick_data, end, end_coll)

                new_res['time_first_lick_collection'].append(time_first_lick_collection)
                new_res['time_first_lick'].append(time_first_lick)
                new_res['odor'].append(odor)
                new_res['lick_baseline'].append(n_licks_baseline)
                new_res['lick'].append(n_licks)
                new_res['lick_collection'].append(n_licks_coll)
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
            temp_res_ = reduce_by_concat(filtered_res, 'lick_collection', rank_keys=['day', 'ix'])
            temp_res__ = reduce_by_concat(filtered_res, 'lick_baseline', rank_keys=['day', 'ix'])
            temp_res___ = reduce_by_concat(filtered_res, 'time_first_lick', rank_keys=['day', 'ix'])
            temp_res____ = reduce_by_concat(filtered_res, 'time_first_lick_collection', rank_keys=['day', 'ix'])
            temp_res['time_first_lick_collection'] = temp_res____['time_first_lick_collection']
            temp_res['time_first_lick'] = temp_res___['time_first_lick']
            temp_res['lick_baseline'] = temp_res__['lick_baseline']
            temp_res['lick_collection'] = temp_res_['lick_collection']
            temp_res['day'] = np.array(sorted(filtered_res['day']))
            temp_res['trial'] = np.arange(len(temp_res['lick']))
            if len(temp_res['lick']):
                append_defaultdicts(out, temp_res)
    for key, val in out.items():
        out[key] = np.array(val)
    return out

def get_roc(res):
    def _dprime(a, b):
        u1, u2 = np.mean(a), np.mean(b)
        s1, s2 = np.std(a), np.std(b)
        return (u1 - u2) / np.sqrt(.5 * (np.square(s1) + np.square(s2)))

    def _roc(a, b):
        import sklearn.metrics
        data = np.concatenate((a,b))
        labels = np.concatenate((np.ones_like(a), np.zeros_like(b))).astype('bool')
        roc = sklearn.metrics.roc_auc_score(labels, data)
        return roc

    def _rolling_window(a, window):
        shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
        strides = a.strides + (a.strides[-1],)
        return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

    key = 'lick'
    print_key = 'roc'
    x_key = 'roc_trial'
    window = 20
    res[print_key] = np.copy(res[key])
    res[x_key] = np.copy(res['trial'])

    res = filter.exclude(res, {'odor_valence':'US'})
    res_ = reduce.new_filter_reduce(res, filter_keys=['mouse','odor_valence'], reduce_key=key)
    combinations, list_of_ixs = filter.retrieve_unique_entries(res_, ['mouse'])

    for i, ixs in enumerate(list_of_ixs):
        assert len(ixs) == 2
        assert res_['odor_valence'][ixs[0]] == 'CS+'
        assert res_['odor_valence'][ixs[1]] == 'CS-'

        a = _rolling_window(res_[key][ixs[0]], window)
        b = _rolling_window(res_[key][ixs[1]], window)
        dprimes = np.array([_roc(x, y) for x, y in zip(a, b)])

        res_[print_key][ixs[0]] = dprimes
        res_[print_key][ixs[1]] = dprimes
        res_[x_key][ixs[0]] = np.arange(len(dprimes))
        res_[x_key][ixs[1]] = np.arange(len(dprimes))
    return res_


def add_behavior_stats(res, condition, arg ='normal'):
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
            last_ix_below_threshold = np.where(vec_binary == 0)[0][-1]
            vec_binary[:last_ix_below_threshold] = 0
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

    def _filter(vec, smoothing_window):
        if smoothing_window < len(vec):
            out = savgol_filter(vec, smoothing_window, config.polynomial_degree)
        else:
            window = smoothing_window // 2
            if window % 2 == 0:
                window += 1
            out = savgol_filter(vec, window, config.polynomial_degree)
        return out


    config = behaviorConfig()
    if 'PT CS+' in res['odor_valence']:
        rules_lick = config.rules_two_phase_lick
        rules_boolean = config.rules_two_phase_boolean
    else:
        rules_lick = config.rules_single_phase_lick
        rules_boolean = config.rules_single_phase_boolean

    if 'OUTPUT' in condition.name:
        rules_lick = config.rules_output_lick
        rules_boolean = config.rules_output_boolean

    time_filter = config.smoothing_window_first_lick
    res['time_first_lick_raw'] = np.copy(res['time_first_lick'])
    for i in range(len(res['time_first_lick'])):
        v = res['time_first_lick'][i]
        res['time_first_lick'][i] = v[v > -1]
        res['time_first_lick_trial'].append(res['trial'][i][v > -1])
        if len(res['time_first_lick'][i]) < time_filter:
            res['time_first_lick_smoothed'].append(res['time_first_lick'][i])
        else:
            res['time_first_lick_smoothed'].append(_filter(res['time_first_lick'][i], time_filter))

        v = res['time_first_lick_collection'][i]
        res['time_first_lick_collection'][i] = v[v > -1]
        res['time_first_lick_collection_trial'].append(res['trial'][i][v > -1])
        if len(res['time_first_lick_collection'][i]) < time_filter:
            res['time_first_lick_collection_smoothed'].append(res['time_first_lick_collection'][i])
        else:
            res['time_first_lick_collection_smoothed'].append(_filter(res['time_first_lick_collection'][i], time_filter))

    for i, v in enumerate(res['odor_valence']):
        smoothing_window_lick = rules_lick[v]
        smoothing_window_boolean = rules_boolean[v]

        lick = res['lick'][i]
        boolean = 100 * (lick > 0)
        res['lick_smoothed'].append(_filter(lick, smoothing_window_lick))
        res['boolean_smoothed'].append(_filter(boolean, smoothing_window_boolean))
        lick_collection = res['lick_collection'][i]
        boolean_collection = 100 * (lick_collection > 0)
        res['lick_collection_smoothed'].append(_filter(lick_collection, smoothing_window_lick))
        res['boolean_collection_smoothed'].append(_filter(boolean_collection, smoothing_window_boolean))

    res['boolean'] = [100 * y > 0 for y in res['lick']]
    for x in res['boolean_smoothed']:
        x[x>100] = 100

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


