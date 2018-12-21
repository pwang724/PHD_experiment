import numpy as np
import os
import decoding
import glob
from CONSTANTS.config import Config
import decode_config
import time
import tools.file_io
import CONSTANTS.conditions as experimental_conditions
from collections import OrderedDict
from tools.experiment_tools import perform

def vary_neuron_valence(argTest = True):
    decodeConfig = decode_config.DecodeConfig()
    decodeConfig.shuffle = False
    decodeConfig.decode_style = 'valence'
    decodeConfig.repeat = 10

    if argTest:
        decodeConfig.repeat = 2

    hp_ranges = OrderedDict()
    hp_ranges['neurons'] = [10, 20]
    return decodeConfig, hp_ranges

def decode_experiment(condition, decodeConfig, save_path):
    data_path = os.path.join(Config.LOCAL_DATA_PATH, Config.LOCAL_DATA_TIMEPOINT_FOLDER,
                             condition.name)
    mouse_files = [os.path.join(data_path, o) for o in os.listdir(data_path)
                   if os.path.isdir(os.path.join(data_path, o))]
    tools.file_io.save_json(save_path, Config.DECODE_CONFIG_JSON, decodeConfig)

    for i, (mouse_file, odors, csps) in enumerate(zip(mouse_files, condition.odors, condition.csp)):
        start_time = time.time()
        data_pathnames = glob.glob(os.path.join(mouse_file, '*' + Config.mat_ext))
        config_pathnames = glob.glob(os.path.join(mouse_file, '*' + Config.cons_ext))
        for j, (data_pathname, config_pathname) in enumerate(zip(data_pathnames, config_pathnames)):
            cons = Config.load_cons_f(config_pathname)
            data = Config.load_mat_f(data_pathname)

            scores = decoding.decode_odor_labels(cons, data, odors, csps, decodeConfig)

            save_folder = os.path.split(mouse_file)[1]
            save_name = os.path.splitext(os.path.split(data_pathname)[1])[0]
            tools.file_io.save_numpy(save_path=os.path.join(save_path, save_folder),
                              save_name=save_name, data=scores)
        print("Analyzed: {0:s} in {1:.2f} seconds".format(mouse_file, time.time()-start_time))

argTest = True
condition = experimental_conditions.OFC
save_path = os.path.join(Config.LOCAL_EXPERIMENT_PATH, 'Valence', condition.name)
perform(experiment=decode_experiment,
        condition =condition,
        experiment_configs=vary_neuron_valence(argTest = argTest),
        path= save_path)