import glob
import os
import time

import numpy as np

from _CONSTANTS.config import Config
from decode import decoding
from tools import file_io as fio


def organizer_decode_odor_within_day(condition, decodeConfig, data_path, save_path):
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
    data_pathnames = sorted(glob.glob(os.path.join(data_path, '*' + Config.mat_ext)))
    config_pathnames = sorted(glob.glob(os.path.join(data_path, '*' + Config.cons_ext)))
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

        if hasattr(condition, 'odors'):
            odor = condition.odors[list_of_mouse_ix[i]]
        else:
            odor = condition.dt_odors[list_of_mouse_ix[i]]

        if decodeConfig.decode_style == 'identity':
            csp = None
        else:
            if hasattr(condition, 'odors'):
                csp = condition.csp[list_of_mouse_ix[i]]
            else:
                csp = condition.dt_csp[list_of_mouse_ix[i]]

        if odor[0] in cons.ODOR_UNIQUE:
            scores = decoding.decode_odor_labels(cons, data, odor, csp, decodeConfig)
            mouse = cons.NAME_MOUSE
            date = cons.NAME_DATE
            plane = cons.NAME_PLANE
            name = mouse + '__' + date + '__' + plane
            fio.save_json(save_path=save_path, save_name=name, config=decodeConfig)
            fio.save_numpy(save_path=save_path, save_name=name, data=scores)
            print("Analyzed: {0:s} in {1:.2f} seconds".format(data_pn, time.time() - start_time))

def organizer_test_fp_fn(condition, decodeConfig, data_path, save_path):
    '''

    :param condition:
    :param decodeConfig:
    :param data_path:
    :param save_path:
    :return:
    '''
    data_pathnames = sorted(glob.glob(os.path.join(data_path, '*' + Config.mat_ext)))
    config_pathnames = sorted(glob.glob(os.path.join(data_path, '*' + Config.cons_ext)))

    list_of_all_data = np.array([Config.load_mat_f(d) for d in data_pathnames])
    list_of_all_cons = np.array([Config.load_cons_f(d) for d in config_pathnames])
    mouse_names_per_file = np.array([cons.NAME_MOUSE for cons in list_of_all_cons])
    mouse_names, list_of_mouse_ix = np.unique(mouse_names_per_file, return_inverse=True)

    if mouse_names.size != len(condition.paths):
        raise ValueError("res has {0:d} mice, but filter has {1:d} mice".
                         format(mouse_names.size, len(condition.paths)))

    # learned_days = np.array([4, 3, 3, 3, 5])
    # last_days = np.array([5, 5, 3, 4, 5])

    learned_days = decodeConfig.start_day
    last_days = decodeConfig.end_day

    days_per_mouse = []
    for x, y in zip(learned_days, last_days):
        days_per_mouse.append(np.arange(x, y+1))
    print(days_per_mouse)


    for i, mouse_name in enumerate(mouse_names):
        start_time = time.time()
        ix = mouse_name == mouse_names_per_file
        list_of_cons_ = list_of_all_cons[ix]
        list_of_data_ = list_of_all_data[ix]
        for cons in list_of_cons_:
            assert cons.NAME_MOUSE == mouse_name, 'Wrong mouse file!'

        if len(days_per_mouse[i]):
            list_of_cons = list_of_cons_[days_per_mouse[i]]
            list_of_data = list_of_data_[days_per_mouse[i]]

            cons = list_of_cons[0]
            cons_dict = cons.__dict__
            for key, value in cons_dict.items():
                if isinstance(value, list) or isinstance(value, np.ndarray):
                    pass
                else:
                    setattr(decodeConfig, key, value)

            if hasattr(condition, 'odors'):
                odor = condition.odors[i]
            else:
                odor = condition.dt_odors[i]
                cons_odors = [cons.ODOR_UNIQUE for cons in list_of_cons]
                ix = [i for i, unique_odors in enumerate(cons_odors) if odor[0] in unique_odors]
                list_of_cons = [list_of_cons[i] for i in ix]
                list_of_data = [list_of_data[i] for i in ix]
            if decodeConfig.decode_style == 'identity':
                csp = None
            else:
                if hasattr(condition, 'odors'):
                    csp = condition.csp[i]
                else:
                    csp = condition.dt_csp[i]

            scores_res = decoding.test_fp_fn(list_of_cons, list_of_data, odor, csp, decodeConfig)
            name = cons.NAME_MOUSE
            fio.save_json(save_path=save_path, save_name=name, config=decodeConfig)
            fio.save_pickle(save_path=save_path, save_name=name, data=scores_res)
            print("Analyzed: {0:s} in {1:.2f} seconds".format(name, time.time() - start_time))

def organizer_test_odor_across_day(condition, decodeConfig, data_path, save_path):
    '''

    :param condition:
    :param decodeConfig:
    :param data_path:
    :param save_path:
    :return:
    '''
    data_pathnames = sorted(glob.glob(os.path.join(data_path, '*' + Config.mat_ext)))
    config_pathnames = sorted(glob.glob(os.path.join(data_path, '*' + Config.cons_ext)))

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
            assert cons.NAME_MOUSE == mouse_name, 'Wrong mouse file!'

        cons = list_of_cons[0]
        cons_dict = cons.__dict__
        for key, value in cons_dict.items():
            if isinstance(value, list) or isinstance(value, np.ndarray):
                pass
            else:
                setattr(decodeConfig, key, value)

        if hasattr(condition, 'odors'):
            odor = condition.odors[i]
        else:
            odor = condition.dt_odors[i]
            cons_odors = [cons.ODOR_UNIQUE for cons in list_of_cons]
            ix = [i for i, unique_odors in enumerate(cons_odors) if odor[0] in unique_odors]
            list_of_cons = [list_of_cons[i] for i in ix]
            list_of_data = [list_of_data[i] for i in ix]
        if decodeConfig.decode_style == 'identity':
            csp = None
        else:
            if hasattr(condition, 'odors'):
                csp = condition.csp[i]
            else:
                csp = condition.dt_csp[i]

        scores_res = decoding.test_odors_across_days(list_of_cons, list_of_data, odor, csp, decodeConfig)
        name = cons.NAME_MOUSE
        fio.save_json(save_path=save_path, save_name=name, config=decodeConfig)
        fio.save_pickle(save_path=save_path, save_name=name, data=scores_res)
        print("Analyzed: {0:s} in {1:.2f} seconds".format(name, time.time() - start_time))


def organizer_decode_day(condition, decodeConfig, data_path, save_path):
    #TODO: obsolete right now. need to fix before using
    '''
    Run decoding experiments with labels based on the day of presentation. Data is contained

    :param condition: experimental condition. For example, condition OFC.
    Must contain fields: name, paths, odors, csp
    :param decodeConfig: class config, contains as fields relevant parameters to run decoding experiment
    :param data_path:
    :param save_path:
    :return:
    '''
    data_pathnames = sorted(glob.glob(os.path.join(data_path, '*' + Config.mat_ext)))
    config_pathnames = sorted(glob.glob(os.path.join(data_path, '*' + Config.cons_ext)))

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
            assert cons.NAME_MOUSE == mouse_name, 'Wrong mouse file!'

        cons = list_of_cons[0]
        cons_dict = cons.__dict__
        for key, value in cons_dict.items():
            if isinstance(value, list) or isinstance(value, np.ndarray):
                pass
            else:
                setattr(decodeConfig, key, value)
        odor = condition.odors[i]
        if decodeConfig.decode_style == 'identity':
            csp = None
        else:
            csp = condition.csp[i]

        scores = decoding.decode_day_labels(list_of_cons, list_of_data, odor, csp, decodeConfig)
        name = cons.NAME_MOUSE
        fio.save_json(save_path=save_path, save_name=name, config=decodeConfig)
        fio.save_numpy(save_path=save_path, save_name=name, data=scores)
        print("Analyzed: {0:s} in {1:.2f} seconds".format(name, time.time() - start_time))