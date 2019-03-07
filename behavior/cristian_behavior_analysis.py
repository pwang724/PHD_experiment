import os
from collections import defaultdict
import numpy as np
import filter
from scipy.signal import savgol_filter

class Indices(object):
    def __init__(self):
        self.session = 0
        self.trials = 2
        self.reversal_trials = 3
        self.time = 4
        self.stimulus_identity = 5
        self.bin_ant_3 = 6
        self.bin_ant_2 = 8
        self.bin_ant_1 = 7
        self.bin_col_1 = 11
        self.bin_col_2 = 12
        self.bin_samp = 9
        self.bin_ir = 10


class Constants(object):
    def __init__(self):
        self.pretraining_directory = 'Pretraining'
        self.discrimination_directory = 'Discrimination'
        self.time_per_bin = .6
        self.csp_file_prefix = 'plus'
        self.csm_file_prefix = 'minus'
        self.detail_file_prefix = 'details'
        self.csp_stimulus_identity = 1
        self.csm_stimulus_identity = 2
        self.halo = 'H'
        self.yfp = 'Y'


class Analysis_Config():
    def __init__(self):
        indices = Indices()
        self.anticipatory_bins = [indices.bin_ant_2, indices.bin_ant_3]
        self.filter_window = 19
        self.filter_order = 0
        self.criterion_filter_window = 49
        self.criterion_threshold = .8
        self.half_max_threshold = .5

def parse(files, experiment, condition, phase):
    def _names(f, is_csp):
        if is_csp:
            prefix = constants.csp_file_prefix
        else:
            prefix = constants.csm_file_prefix

        mouse = os.path.split(f)[1]
        if mouse[-2:].isdigit():
            data_name = prefix + mouse[:-2] + mouse[-1] + '.npy'
            detail_name = constants.detail_file_prefix + mouse[:-2] + mouse[-1] + '.npy'
        else:
            data_name = prefix + mouse + '.npy'
            detail_name = constants.detail_file_prefix + mouse + '.npy'
            mouse = mouse[0] + '0' + mouse[1:]
        detail_name = os.path.join(f, detail_name)
        data_name = os.path.join(f, data_name)
        return mouse, data_name, detail_name

    def _add_details(res, detail_name):
        detail_data = np.load(detail_name)
        dates = [x.decode('utf-8') for x in detail_data[:,3]]
        time = [x.decode('utf-8') for x in detail_data[:,4]]
        res['session_date'].append(dates)
        res['session_time'].append(time)

    def _add_data(res, data_name, is_csp):
        data = np.load(data_name)
        indices_dict = indices.__dict__
        for k, v in indices_dict.items():
            res[k].append(data[:, v])
        res['mouse'].append(mouse)
        res['phase'].append(phase)
        res['condition'].append(condition)
        if is_csp:
            valence = 'CS+'
        else:
            valence = 'CS-'
        res['odor_valence'].append(valence)

    def _fix_trials(res):
        import filter
        combinations, ixs = filter.retrieve_unique_entries(res, loop_keys=['mouse'])
        for ix in ixs:
            for i in ix:
                trials = res['trials'][i]
                new_trials = np.arange(1,len(trials)+1)
                res['trials'][i] = new_trials

    constants = Constants()
    indices = Indices()
    res = defaultdict(list)
    for f in files:
        is_csp = [True, False]
        for csp in is_csp:
            mouse, data_name, detail_name = _names(f, is_csp=csp)
            _add_details(res, detail_name)
            _add_data(res, data_name, csp)
    for k, v in res.items():
        res[k] = np.array(v)
    _fix_trials(res)
    return res


def analyze(res):
    def _smooth(vector, filter_window, keyword):
        if len(vector) < filter_window:
            smoothed = vector
        else:
            smoothed = savgol_filter(vector, window_length= filter_window,
                                     polyorder=analysis_config.filter_order)
        res[keyword].append(smoothed)
        return smoothed

    def _pass_criterion(data, threshold, valence, keyword):
        if valence == 'CS+':
            mask = data >= threshold
        else:
            mask = data <= (1 - threshold)

        if np.any(mask):
            ix = np.where(mask)[0][0]
        else:
            ix = len(mask)
        res[keyword].append(ix)
        return ix

    def _pass_half_max(data, threshold, valence, keyword):
        if valence == 'CS+':
            mask = data >= threshold
        else:
            mask = data <= (1 - threshold)

        if np.any(np.invert(mask)):
            last_zero = np.where(mask == 0)[0][-1]
            mask[:last_zero] = 0

        if np.any(mask):
            ix = np.where(mask)[0][0]
        else:
            ix = len(mask)
        res[keyword].append(ix)
        return ix


    def _trials_per_day(data):
        trials_per_mouse = []
        max_day = int(np.max(data))
        days = np.arange(1, max_day+1)
        for i in days:
            trials_per_day = np.sum(data == i)
            trials_per_mouse.append(trials_per_day)
        return days, np.array(trials_per_mouse)


    indices = Indices()
    analysis_config = Analysis_Config()
    ind_dict = indices.__dict__

    ant3 = res['bin_ant_3']
    ant2 = res['bin_ant_2']
    ant23 = [(a + b)/2 for a, b in zip(ant2, ant3)]
    res['bin_ant_23'] = np.array(ant23)
    ind_dict.update({'bin_ant_23':''})
    ind_dict = {'bin_ant_23':''}
    for key in ind_dict.keys():
        if key[:3] == 'bin':
            data = res[key]
            valences = res['odor_valence']
            for i, vector in enumerate(data):
                valence = valences[i]

                # if valence == 'CS+':
                #     print([res['mouse'][i], res['phase'][i], valence])

                #smooth
                _smooth(vector, analysis_config.filter_window, key + '_smooth')

                #boolean
                bool_vector = vector > 0
                _smooth(bool_vector, analysis_config.filter_window, key + '_boolean')

                #criterion
                bool_vector = vector > 0
                vector_criterion = _smooth(bool_vector, analysis_config.criterion_filter_window, key + '_boolean_criterion')

                #passed criterion
                _pass_criterion(vector_criterion, analysis_config.criterion_threshold, valence, key + '_trials_to_criterion')

                #passed halfmax
                _pass_half_max(vector_criterion, analysis_config.half_max_threshold, valence, key + '_trials_to_half_max')

    data = res['session']
    for d in data:
        days, trials_per_day = _trials_per_day(d)
        res['trials_per_day'].append(trials_per_day)
        res['days'].append(days)

    for k, v in res.items():
        res[k] = np.array(v)

def shift_discrimination_index(res):
    combs, ixs = filter.retrieve_unique_entries(res, ['mouse'])
    res['together_trials'] = np.copy(res['trials'])
    res['together_days'] = np.copy(res['days'])

    for comb, ix in zip(combs, ixs):
        phases = res['phase'][ix]
        pt_mask = ix[phases == 'Pretraining']
        dt_mask = ix[phases == 'Discrimination']

        pt_trials = res['trials'][pt_mask]
        total_pt_trials = np.max([len(x) for x in pt_trials])
        dt_trials = res['trials'][dt_mask]
        modified_dt_trials = [x + total_pt_trials for x in dt_trials]
        res['together_trials'][dt_mask] = modified_dt_trials

        pt_days = res['days'][pt_mask]
        total_pt_days = np.sum([x[-1] for x in pt_days])
        dt_days = res['days'][dt_mask]
        modified_dt_days = [x + total_pt_days for x in dt_days]
        res['together_days'][dt_mask] = modified_dt_days
    for k, v in res.items():
        res[k] = np.array(v)