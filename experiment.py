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
        decodeConfig.repeat = 5

    hp_ranges = OrderedDict()
    hp_ranges['neurons'] = [5, 10, 20, 30, 40, 50]
    return decodeConfig, hp_ranges

def decode_experiment(condition, decodeConfig, save_path):
    data_path = os.path.join(Config.LOCAL_DATA_PATH, Config.LOCAL_DATA_TIMEPOINT_FOLDER,
                             condition.name)
    mouse_files = [os.path.join(data_path, o) for o in os.listdir(data_path)
                   if os.path.isdir(os.path.join(data_path, o))]

    for i, (mouse_file, odors, csps) in enumerate(zip(mouse_files, condition.odors, condition.csp)):
        start_time = time.time()
        data_pathnames = glob.glob(os.path.join(mouse_file, '*' + Config.mat_ext))
        config_pathnames = glob.glob(os.path.join(mouse_file, '*' + Config.cons_ext))
        for j, (data_pathname, config_pathname) in enumerate(zip(data_pathnames, config_pathnames)):
            cons = Config.load_cons_f(config_pathname)
            data = Config.load_mat_f(data_pathname)

            cons_dict = cons.__dict__
            for key, value in cons_dict.items():
                if isinstance(value, list) or isinstance(value, np.ndarray):
                    pass
                else:
                    setattr(decodeConfig, key, value)
            scores = decoding.decode_odor_labels(cons, data, odors, csps, decodeConfig)

            mouse = os.path.split(mouse_file)[1]
            date_plane = os.path.splitext(os.path.split(data_pathname)[1])[0]
            name = mouse + '__' + date_plane
            tools.file_io.save_json(save_path=save_path, save_name=name, config=decodeConfig)
            tools.file_io.save_numpy(save_path=save_path, save_name=name, data=scores)
        print("Analyzed: {0:s} in {1:.2f} seconds".format(mouse_file, time.time()-start_time))

if __name__ == '__main__':
    argTest = True
    condition = experimental_conditions.OFC
    save_path = os.path.join(Config.LOCAL_EXPERIMENT_PATH, 'Valence', condition.name)
    perform(experiment=decode_experiment,
            condition =condition,
            experiment_configs=vary_neuron_valence(argTest= argTest),
            path= save_path)