import numpy as np
import os
from decode import decode_config
from _CONSTANTS.config import Config
import _CONSTANTS.conditions as experimental_conditions
from collections import OrderedDict

from decode.organizer import organizer_decode_odor_within_day
from tools.experiment_tools import perform

def test_fp_fn(argTest = True, neurons = 40, style = ('identity'), no_end_time=True, start_day = [], end_day = []):
    decodeConfig = decode_config.DecodeConfig()
    decodeConfig.repeat = 100
    decodeConfig.neurons = neurons
    decodeConfig.average_time = True
    decodeConfig.no_end_time = no_end_time
    decodeConfig.shuffle = False
    decodeConfig.start_day = start_day
    decodeConfig.end_day = end_day
    hp_ranges = OrderedDict()
    hp_ranges['decode_style'] = style

    if argTest:
        decodeConfig.repeat = 50
    return decodeConfig, hp_ranges

def test_across_days(argTest = True, neurons = 40, style = ('identity'), no_end_time=True):
    decodeConfig = decode_config.DecodeConfig()
    decodeConfig.repeat = 100
    decodeConfig.neurons = neurons
    decodeConfig.average_time = True
    decodeConfig.no_end_time = no_end_time
    hp_ranges = OrderedDict()
    hp_ranges['decode_style'] = style
    hp_ranges['shuffle'] = [True, False]

    if argTest:
        decodeConfig.repeat = 50
    return decodeConfig, hp_ranges

def vary_neuron(argTest = True, neurons = 50, style = ('valence')):
    decodeConfig = decode_config.DecodeConfig()
    decodeConfig.shuffle = False
    decodeConfig.repeat = 100
    decodeConfig.average_time = True
    hp_ranges = OrderedDict()
    hp_ranges['neurons'] = np.arange(5, neurons, 5)
    hp_ranges['decode_style'] = style
    hp_ranges['shuffle'] = [False, True]
    if argTest:
        decodeConfig.repeat = 50
        hp_ranges['neurons'] = np.arange(10, neurons, 10)
    return decodeConfig, hp_ranges

def vary_shuffle(argTest = True):
    decodeConfig = decode_config.DecodeConfig()
    decodeConfig.decode_style = 'valence'

    decodeConfig.repeat = 10
    decodeConfig.neurons = 40
    decodeConfig.average_time = True

    hp_ranges = OrderedDict()
    hp_ranges['shuffle'] = [False, True]
    if argTest:
        decodeConfig.repeat = 5
    return decodeConfig, hp_ranges

def vary_decode_style(argTest = True, style = ('identity','csp_identity','csm_identity', 'valence')):
    decodeConfig = decode_config.DecodeConfig()
    decodeConfig.repeat = 100
    decodeConfig.neurons = 40
    decodeConfig.average_time = True

    hp_ranges = OrderedDict()
    hp_ranges['shuffle'] = [False, True]
    hp_ranges['decode_style'] = style
    if argTest:
        decodeConfig.repeat = 50
    return decodeConfig, hp_ranges


if __name__ == '__main__':
    argTest = True
    condition = experimental_conditions.OFC
    save_path = os.path.join(Config.LOCAL_EXPERIMENT_PATH, 'Valence', condition.name)
    data_path = os.path.join(Config.LOCAL_DATA_PATH, Config.LOCAL_DATA_TIMEPOINT_FOLDER,
                             condition.name)
    perform(experiment=organizer_decode_odor_within_day,
            condition =condition,
            experiment_configs=vary_neuron(argTest= argTest),
            data_path = data_path,
            save_path= save_path)