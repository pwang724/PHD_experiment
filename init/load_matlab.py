from init.cons import Cons
import matlab.engine
import matlab
import time
import numpy as np
import os
import glob
import CONSTANTS.conditions as conditions
from CONSTANTS.config import Config


def copy_config_from_matlab(path):
    c = Cons()
    eng = matlab.engine.start_matlab()

    obj = eng.load(path)['m']
    eng.workspace["obj"] = obj
    fieldnames = eng.eval("fieldnames(obj.Config)")
    for name in fieldnames:
        if name != 'DAQ_DATA':
            val = eng.eval("obj.Config." + name)
            setattr(c, name, val)
        #cannot do this for the key of DAQ_DATA
    return c

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
    cons = Cons(data_path)
    mat, _ = load_calcium_traces_from_matlab(cons.STORAGE_DATA, eng)
    print('Time taken to load matfiles: {0:3.3f}'.format(time.time() - start_time))
    print(mat.shape)
    if save == True:
        mouse = cons.NAME_MOUSE
        date = cons.NAME_DATE
        plane = cons.NAME_PLANE
        save_path = os.path.join(Config.LOCAL_DATA_PATH, Config.LOCAL_DATA_SINGLE_FOLDER, mouse)
        save_name = date + '_' + plane
        Config.save_mat_f(save_path, save_name, data=mat)
        Config.save_cons_f(save_path, save_name, data=cons)
    return mat, cons

def load_timepoint_from_matlab(path, condition, save = True):
    eng = matlab.engine.start_matlab()
    data_path = os.path.join(path, 'data')
    data_wildcard = os.path.join(data_path, '*.mat')
    matfile_paths = glob.glob(data_wildcard)
    list_of_mats, list_of_cons = [],[]
    for p in matfile_paths:
        start_time = time.time()
        mat, obj_name = load_calcium_traces_from_matlab(p, eng)
        dir = eng.eval(obj_name + ".Config.DIR")
        cons = Cons(dir)
        print('[***] LOADED {0:<50s} in: {1:3.3f} seconds'.format(p, time.time() - start_time))

        if save == True:
            save_path = os.path.join(Config.LOCAL_DATA_PATH, Config.LOCAL_DATA_TIMEPOINT_FOLDER,
                                     condition, cons.NAME_MOUSE)
            save_name = cons.NAME_DATE + '__' + cons.NAME_PLANE
            Config.save_mat_f(save_path, save_name, data=mat)
            Config.save_cons_f(save_path, save_name, data=cons)
    return list_of_mats, list_of_cons

def load_condition(condition):
    name = condition.name
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