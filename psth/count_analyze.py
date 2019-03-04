import os

import psth.junk
from _CONSTANTS import conditions as experimental_conditions
from _CONSTANTS.config import Config
import analysis
import psth.psth_helper
from collections import defaultdict
from tools import utils as utils
import copy
import numpy as np
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
import time
import tools.file_io as fio

class Base_Config(object):
    def __init__(self):
        self.condition = None
        self.mouse = None
        self.include_water = None
        self.start_at_training = None

        self.window = 3
        self.p_threshold = 0.001
        self.p_window = 8
        self.m_threshold = 0.00
        self.m_window = 1
        self.colors = ['lime', 'darkgreen', 'red', 'magenta']

class OFC_Config(Base_Config):
    def __init__(self):
        super(OFC_Config, self).__init__()
        self.condition = experimental_conditions.OFC
        self.include_water = True
        self.start_at_training = False
        self.mouse = 0

class OFC_LONGTERM_Config(Base_Config):
    def __init__(self):
        super(OFC_LONGTERM_Config, self).__init__()
        self.condition = experimental_conditions.OFC_LONGTERM
        self.include_water = False
        self.start_at_training = False

class OFC_COMPOSITE_Config(Base_Config):
    def __init__(self):
        super(OFC_COMPOSITE_Config, self).__init__()
        self.condition = experimental_conditions.OFC_COMPOSITE
        self.include_water = False
        self.start_at_training = True

class MPFC_COMPOSITE_Config(Base_Config):
    def __init__(self):
        super(MPFC_COMPOSITE_Config, self).__init__()
        self.condition = experimental_conditions.MPFC_COMPOSITE
        self.include_water = False
        self.start_at_training = True

class BLA_Config(Base_Config):
    def __init__(self):
        super(BLA_Config, self).__init__()
        self.condition = experimental_conditions.BLA
        self.include_water = True
        self.start_at_training = False

class BLA_LONGTERM_Config(Base_Config):
    def __init__(self):
        super(BLA_LONGTERM_Config, self).__init__()
        self.condition = experimental_conditions.BLA_LONGTERM
        self.include_water = False
        self.start_at_training = False

class PIR_Config(Base_Config):
    def __init__(self):
        super(PIR_Config, self).__init__()
        self.condition = experimental_conditions.PIR
        self.include_water = True
        self.start_at_training = False
        self.m_threshold = 0.1

class PIR_NAIVE_Config(Base_Config):
    def __init__(self):
        super(PIR_NAIVE_Config, self).__init__()
        self.condition = experimental_conditions.PIR_NAIVE
        self.start_at_training = False
        self.m_threshold = 0.1


class OFC_State_Config(Base_Config):
    def __init__(self):
        super(OFC_State_Config, self).__init__()
        self.condition = experimental_conditions.OFC_STATE
        self.include_water = False
        self.start_at_training = False
        self.m_threshold = 0.1

class OFC_Reversal_Config(Base_Config):
    def __init__(self):
        super(OFC_Reversal_Config, self).__init__()
        self.condition = experimental_conditions.OFC_REVERSAL
        self.include_water = False
        self.start_at_training = False
        self.m_threshold = 0.03

class OFC_Context_Config(Base_Config):
    def __init__(self):
        super(OFC_Context_Config, self).__init__()
        self.condition = experimental_conditions.OFC_CONTEXT
        self.include_water = False
        self.start_at_training = False
        self.m_threshold = 0.1

def convert(res, condition_config):
    d = defaultdict(list)
    toConvert = ['day', 'mouse', 'DAQ_O_ON_F', 'DAQ_O_OFF_F', 'DAQ_W_ON_F', 'TRIAL_FRAMES']
    for i in range(len(res['mouse'])):
        data = res['data'][i]
        data = utils.reshape_data(data,
                                  nFrames=res['TRIAL_FRAMES'][i],
                                  cell_axis=0, trial_axis=1, time_axis=2)
        odor_trials = res['ODOR_TRIALS'][i]
        assert data.shape[1] == len(odor_trials), 'number of trials in cons does not match trials in data'
        mouse = res['mouse'][i]

        condition = condition_config.condition
        if hasattr(condition, 'odors'):
            relevant_odors = copy.copy(condition.odors[mouse])
        else:
            relevant_odors = copy.copy(condition.pt_odors[mouse] + condition.dt_odors[mouse])
        if condition_config.include_water:
            relevant_odors.append('water')

        for odor in relevant_odors:
            ix = odor == odor_trials
            if np.any(ix):
                current_data = data[:, ix, :]
                d['data'].append(current_data)
                d['odor'].append(odor)
                for names in toConvert:
                    d[names].append(res[names][i])

    for key, val in d.items():
        d[key] = np.array(val)
    return d

def parse_data(res):
    list_odor_on = res['DAQ_O_ON_F']
    list_odor_off = res['DAQ_O_OFF_F']
    list_water_on = res['DAQ_W_ON_F']
    list_data = res['data']

    for i, data in enumerate(list_data):
        start_time = time.time()
        p_list = []
        dff_list = []
        for cell_data in data:
            baseline = cell_data[:, config.baseline_start:list_odor_on[i] - config.baseline_end]
            baseline = baseline.flatten()
            data_trial_time_window = _rolling_window(cell_data, window=condition_config.p_window)
            data_time_trial_window = np.transpose(data_trial_time_window, [1, 0, 2])
            data_time_pixels = data_time_trial_window.reshape(data_time_trial_window.shape[0],-1)

            #statistical significance
            f = lambda y: mannwhitneyu(baseline, y, use_continuity=True, alternative='less')[-1]
            p = []
            for y in data_time_pixels:
                try:
                    out = f(y)
                except:
                    out = 1
                p.append(out)
            while (len(p) < cell_data.shape[-1]):
                p.append(p[-1])

            #magnitude significance
            dff = psth.psth_helper.subtract_baseline(cell_data, config.baseline_start, list_odor_on[i] - config.baseline_end)
            dff = np.mean(dff, axis=0)

            p_list.append(np.array(p))
            dff_list.append(np.array(dff))
        res['dff'].append(np.array(dff_list))
        res['p'].append(np.array(p_list))
        elapsed = time.time() - start_time
        print('analyzed mouse {}, day {}, odor {}, in {} seconds'.format(res['mouse'][i], res['day'][i], res['odor'][i], elapsed))
    for key, val in res.items():
        res[key] = np.array(val)
    res.pop('data')
    fio.save_pickle(save_path=save_path, save_name= 'dict', data= res)

def analyze_data(res, condition_config, m_threshold=None):
    list_odor_on = res['DAQ_O_ON_F']
    list_water_on = res['DAQ_W_ON_F']
    res['ssig'] = []
    res['msig'] = []
    res['sig'] = []

    for i in range(len(list_odor_on)):
        p_list = res['p'][i]
        dff_list = res['dff'][i]
        ssig_list = []
        msig_list = []
        sig_list = []
        onset_list = []
        amplitude_list = []
        duration_list = []
        odor = res['odor'][i]
        for p, dff in zip(p_list, dff_list):
            ssig = np.array(p) < condition_config.p_threshold
            reached_ssig = [np.all(x) for x in _rolling_window(ssig, condition_config.p_window)]
            if odor == 'water':
                s, e = list_water_on[i], list_water_on[i] + 4
            else:
                s, e = list_odor_on[i], list_water_on[i]
            if np.any(reached_ssig[s:e]):
                statistical_significance = True
            else:
                statistical_significance = False

            if m_threshold == None:
                magnitude_threshold = condition_config.m_threshold
            else:
                magnitude_threshold = m_threshold
            msig = dff > magnitude_threshold
            reached_msig = np.array([np.all(x) for x in _rolling_window(msig, condition_config.m_window)])
            if np.any(reached_msig[s:e]):
                mag_significance = True
            else:
                mag_significance = False

            if statistical_significance and mag_significance:
                onset = np.where(reached_ssig[s:e])[0][0]
                onset_list.append(onset)

                duration = np.sum(reached_msig[s:e])
                duration_list.append(duration)

                amplitude = np.max(dff[s:e])
                amplitude_list.append(amplitude)
            else:
                onset_list.append(-1)
                duration_list.append(-1)
                amplitude_list.append(-1)


            ssig_list.append(statistical_significance)
            msig_list.append(mag_significance)
            sig_list.append(statistical_significance and mag_significance)
        res['ssig'].append(np.array(ssig_list).astype(int))
        res['msig'].append(np.array(msig_list).astype(int))
        res['sig'].append(np.array(sig_list).astype(int))

        res['onset'].append(np.array(onset_list).astype(int))
        res['duration'].append(np.array(duration_list).astype(int))
        res['amplitude'].append(np.array(amplitude_list))
    for key, val in res.items():
        res[key] = np.array(val)

def _rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


if __name__ == '__main__':
    config = psth.psth_helper.PSTHConfig()
    condition_config = OFC_Reversal_Config()
    condition = condition_config.condition

    data_path = os.path.join(Config.LOCAL_DATA_PATH, Config.LOCAL_DATA_TIMEPOINT_FOLDER, condition.name)
    save_path = os.path.join(Config.LOCAL_EXPERIMENT_PATH, 'COUNTING', condition.name)
    figure_path = os.path.join(Config.LOCAL_FIGURE_PATH, 'OTHER', 'PSTH',  condition.name, 'CELL')

    res = analysis.load_data(data_path)
    analysis.add_indices(res, arg_plane=True)
    analysis.add_time(res)
    odor_res = convert(res, condition_config)
    analysis.add_odor_value(odor_res, condition_config.condition)
    parse_data(odor_res)

    # odor_res = fio.load_pickle(pickle_path=os.path.join(save_path,'dict.pkl'))
    # analyze_data(odor_res, condition_config)

