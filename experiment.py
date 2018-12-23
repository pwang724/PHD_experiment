import numpy as np
import os
import decoding
import glob
from CONSTANTS.config import Config
import decode_config
import time
import tools.file_io as fio
import CONSTANTS.conditions as experimental_conditions
from collections import OrderedDict
from tools.experiment_tools import perform

def vary_neuron_valence(argTest = True):
    decodeConfig = decode_config.DecodeConfig()
    decodeConfig.shuffle = False
    decodeConfig.decode_style = 'valence'
    decodeConfig.repeat = 100

    if argTest:
        decodeConfig.repeat = 10

    hp_ranges = OrderedDict()
    hp_ranges['neurons'] = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    return decodeConfig, hp_ranges

def decode_odor_as_label(condition, decodeConfig, save_path):
    '''
    :param condition:
    :param decodeConfig:
    :param save_path:
    :return:
    '''
    #TODO: see if it works
    data_path = os.path.join(Config.LOCAL_DATA_PATH, Config.LOCAL_DATA_TIMEPOINT_FOLDER,
                             condition.name)
    data_pathnames = glob.glob(os.path.join(data_path, '*' + Config.mat_ext))
    config_pathnames = glob.glob(os.path.join(data_path, '*' + Config.cons_ext))
    mouse_names_per_file = [Config.load_cons_f(d).NAME_MOUSE for d in config_pathnames]
    mouse_names, list_of_mouse_ix = np.unique(mouse_names_per_file, return_inverse=True)

    if mouse_names.size != len(condition.paths):
        raise ValueError("res has {0:d} mice, but filter has {1:d} mice".
                         format(mouse_names.size, len(condition.paths)))

    for i, data_pn in enumerate(data_pathnames):
        start_time = time.time()
        cons = Config.load_cons_f(config_pathnames[i])
        data = Config.load_mat_f(data_pn)

        cons_dict = cons.__dict__
        for key, value in cons_dict.items():
            if isinstance(value, list) or isinstance(value, np.ndarray):
                pass
            else:
                setattr(decodeConfig, key, value)

        odor = condition.odors[list_of_mouse_ix[i]]
        csp = condition.csp[list_of_mouse_ix[i]]
        scores = decoding.decode_odor_labels(cons, data, odor, csp, decodeConfig)
        mouse = cons.NAME_MOUSE
        date = cons.NAME_DATE
        plane = cons.NAME_PLANE
        name = mouse + '__' + date + '__' + plane
        fio.save_json(save_path=save_path, save_name=name, config=decodeConfig)
        fio.save_numpy(save_path=save_path, save_name=name, data=scores)
        print("Analyzed: {0:s} in {1:.2f} seconds".format(data_pn, time.time() - start_time))

if __name__ == '__main__':
    argTest = True
    condition = experimental_conditions.OFC
    save_path = os.path.join(Config.LOCAL_EXPERIMENT_PATH, 'Valence', condition.name)
    perform(experiment=decode_odor_as_label,
            condition =condition,
            experiment_configs=vary_neuron_valence(argTest= argTest),
            path= save_path)