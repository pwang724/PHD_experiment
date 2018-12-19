from init.cons import Config
import matlab.engine
import matlab
import time
import numpy as np
import os
import glob
from CONSTANTS.constants import constants
import CONSTANTS.conditions as conditions
import pickle


def copy_config_from_matlab(path):
    c = Config()
    eng = matlab.engine.start_matlab()

    obj = eng.load(path)['m']
    eng.workspace["obj"] = obj
    fieldnames = eng.eval("fieldnames(obj.constants)")
    for name in fieldnames:
        if name != 'DAQ_DATA':
            val = eng.eval("obj.constants." + name)
            setattr(c, name, val)
        #cannot do this for the key of DAQ_DATA
    return c

def _easy_save(save_path, save_name, data):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_pathname = os.path.join(save_path, save_name + '.pkl')
    with open(save_pathname, "wb") as f:
        pickle.dump(data, f)

def load_calcium_traces_from_matlab(data_path, eng):
    temp= eng.load(data_path)
    if list(temp.keys())[0] != 'm':
        print('Key value of path {} is: {}'.format(data_path, list(temp.keys())[0]))
    obj= list(temp.values())[0]
    obj_name = "obj"
    eng.workspace[obj_name] = obj
    matlab_mat = eng.eval(obj_name + ".roiCell('norm')")
    mat = np.asarray(matlab_mat).squeeze()
    return mat, obj_name

def load_single_from_matlab(data_path, save = True):
    start_time = time.time()

    eng = matlab.engine.start_matlab()
    config = Config(data_path)
    mat, _ = load_calcium_traces_from_matlab(config.STORAGE_DATA, eng)
    print('Time taken to load matfiles: {0:3.3f}'.format(time.time() - start_time))
    print(mat.shape)
    if save == True:
        mouse = config.NAME_MOUSE
        date = config.NAME_DATE
        plane = config.NAME_PLANE
        save_path = os.path.join(constants.LOCAL_DATA_PATH, constants.LOCAL_DATA_SINGLE_FOLDER, mouse)
        save_name = date + '_' + plane
        _easy_save(save_path, save_name, data = (config, mat))
    return mat, config

def load_timepoint_from_matlab(path, condition, save = True):
    eng = matlab.engine.start_matlab()
    data_path = os.path.join(path, 'data')
    data_wildcard = os.path.join(data_path, '*.mat')
    matfile_paths = glob.glob(data_wildcard)
    mats, configs = [],[]
    for p in matfile_paths:
        start_time = time.time()
        mat, obj_name = load_calcium_traces_from_matlab(p, eng)
        dir = eng.eval(obj_name + ".constants.DIR")
        config = Config(dir)

        mats.append(mat)
        configs.append(config)
        print('[***] LOADED {0:<50s} in: {1:3.3f} seconds'.format(p, time.time() - start_time))

        if save == True:
            save_path = os.path.join(constants.LOCAL_DATA_PATH, constants.LOCAL_DATA_TIMEPOINT_FOLDER,
                                     condition, config.NAME_MOUSE)
            save_name = config.NAME_DATE
            _easy_save(save_path, save_name, data=(config, mat))

    if save == True:
        save_path = os.path.join(constants.LOCAL_DATA_PATH, constants.LOCAL_DATA_TIMEPOINT_FOLDER, condition)
        save_name = configs[0].NAME_MOUSE
        _easy_save(save_path, save_name, data=(configs, mats))
    return mats, configs

def load_pickle(pickle_path):
    with open(pickle_path, "rb") as f:
        e = pickle.load(f)
    return e[0], e[1]

def load_condition(condition):
    name = condition.condition
    paths = condition.paths
    for path in paths:
        load_timepoint_from_matlab(path, name, save=True)

if __name__ == '__main__':
    # example_path = 'E:/IMPORTANT DATA/DATA_2P/M187_ofc/7-19-2016/420'
    # mat, config = load_single_from_matlab(example_path)

    # condition = 'OFC'
    # path = 'E:/IMPORTANT DATA/DATA_2P/M187_ofc/training_LEARNING'
    # load_timepoint_from_matlab(path, condition, save=True)

    condition = conditions.OFC
    load_condition(condition)