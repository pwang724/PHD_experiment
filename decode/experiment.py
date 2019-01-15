import numpy as np
import os
from decode import decoding, decode_config
import glob
from CONSTANTS.config import Config
import time
import tools.file_io as fio
import CONSTANTS.conditions as experimental_conditions
from collections import OrderedDict
from tools.experiment_tools import perform

def vary_neuron(argTest = True, neurons = 50, style = ('valence')):
    decodeConfig = decode_config.DecodeConfig()
    decodeConfig.shuffle = False
    decodeConfig.repeat = 25
    hp_ranges = OrderedDict()
    hp_ranges['neurons'] = np.arange(5, neurons, 5)
    hp_ranges['decode_style'] = style
    if argTest:
        decodeConfig.repeat = 10
        hp_ranges['neurons'] = np.arange(10, neurons, 10)
    return decodeConfig, hp_ranges

def vary_shuffle(argTest = True):
    decodeConfig = decode_config.DecodeConfig()
    decodeConfig.decode_style = 'valence'

    decodeConfig.repeat = 10
    decodeConfig.neurons = 40

    hp_ranges = OrderedDict()
    hp_ranges['shuffle'] = [False, True]
    if argTest:
        decodeConfig.repeat = 5
    return decodeConfig, hp_ranges

def vary_decode_style(argTest = True):
    decodeConfig = decode_config.DecodeConfig()
    decodeConfig.repeat = 10
    decodeConfig.neurons = 40

    hp_ranges = OrderedDict()
    hp_ranges['shuffle'] = [False, True]
    hp_ranges['decode_style'] = ['identity','csp_identity','csm_identity', 'valence']
    if argTest:
        decodeConfig.repeat = 5
    return decodeConfig, hp_ranges

def vary_decode_style_identity(argTest = True):
    decodeConfig = decode_config.DecodeConfig()
    decodeConfig.repeat = 10
    decodeConfig.neurons = 40

    hp_ranges = OrderedDict()
    hp_ranges['shuffle'] = [False, True]
    hp_ranges['decode_style'] = ['identity']
    if argTest:
        decodeConfig.repeat = 5
    return decodeConfig, hp_ranges


def decode_odor_as_label(condition, decodeConfig, data_path, save_path):
    '''
    Run decoding experiments with labels based on the odor that was presented. Data is contained

    :param condition: experimental condition. For example, condition OFC.
    Must contain fields: name, paths, odors, csp
    :param decodeConfig: class config, contains as fields relevant parameters to run decoding experiment
    :param data_path:
    :param save_path:
    :return:
    '''
    #TODO: see if it works when odors are different for each animal
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
        if decodeConfig.decode_style == 'identity':
            csp = None
        else:
            csp = condition.csp[list_of_mouse_ix[i]]
        scores = decoding.decode_odor_labels(cons, data, odor, csp, decodeConfig)
        mouse = cons.NAME_MOUSE
        date = cons.NAME_DATE
        plane = cons.NAME_PLANE
        name = mouse + '__' + date + '__' + plane
        fio.save_json(save_path=save_path, save_name=name, config=decodeConfig)
        fio.save_numpy(save_path=save_path, save_name=name, data=scores)
        print("Analyzed: {0:s} in {1:.2f} seconds".format(data_pn, time.time() - start_time))

def decode_day_as_label(condition, decodeConfig, data_path, save_path):
    '''
    Run decoding experiments with labels based on the day of presentation. Data is contained

    :param condition: experimental condition. For example, condition OFC.
    Must contain fields: name, paths, odors, csp
    :param decodeConfig: class config, contains as fields relevant parameters to run decoding experiment
    :param data_path:
    :param save_path:
    :return:
    '''
    data_pathnames = glob.glob(os.path.join(data_path, '*' + Config.mat_ext))
    config_pathnames = glob.glob(os.path.join(data_path, '*' + Config.cons_ext))

    list_of_all_data = np.array([Config.load_mat_f(d) for d in data_pathnames])
    list_of_all_cons = np.array([Config.load_cons_f(d) for d in config_pathnames])
    mouse_names_per_file = np.array([cons.NAME_MOUSE for cons in list_of_all_cons])
    mouse_names, list_of_mouse_ix = np.unique(mouse_names_per_file, return_inverse=True)

    if mouse_names.size != len(condition.paths):
        raise ValueError("res has {0:d} mice, but filter has {1:d} mice".
                         format(mouse_names.size, len(condition.paths)))

    for i, mouse_name in enumerate(mouse_names):
        start_time = time.time()
        ix = mouse_name == mouse_names_per_file
        list_of_cons = list_of_all_cons[ix]
        list_of_data = list_of_all_data[ix]
        for cons in list_of_cons:
            assert(cons.NAME_MOUSE == mouse_name, 'Wrong mouse file!')

        cons = list_of_cons[0]
        cons_dict = cons.__dict__
        for key, value in cons_dict.items():
            if isinstance(value, list) or isinstance(value, np.ndarray):
                pass
            else:
                setattr(decodeConfig, key, value)
        odor = condition.odors[list_of_mouse_ix[i]]
        if decodeConfig.decode_style == 'identity':
            csp = None
        else:
            csp = condition.csp[list_of_mouse_ix[i]]

        scores = decoding.decode_odor_labels(list_of_cons, list_of_data, odor, csp, decodeConfig)
        name = cons.NAME_MOUSE
        fio.save_json(save_path=save_path, save_name=name, config=decodeConfig)
        fio.save_numpy(save_path=save_path, save_name=name, data=scores)
        print("Analyzed: {0:s} in {1:.2f} seconds".format(name, time.time() - start_time))

if __name__ == '__main__':
    argTest = True
    condition = experimental_conditions.OFC
    save_path = os.path.join(Config.LOCAL_EXPERIMENT_PATH, 'Valence', condition.name)
    data_path = os.path.join(Config.LOCAL_DATA_PATH, Config.LOCAL_DATA_TIMEPOINT_FOLDER,
                             condition.name)
    perform(experiment=decode_odor_as_label,
            condition =condition,
            experiment_configs=vary_neuron(argTest= argTest),
            data_path = data_path,
            save_path= save_path)