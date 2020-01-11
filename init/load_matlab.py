from init.cons import Cons
# import matlab.engine
import matlab
import time
import numpy as np
import os
import glob
import _CONSTANTS.conditions as conditions
from _CONSTANTS.config import Config


def copy_config_from_matlab(path):
    c = Cons()
    eng = matlab.engine.start_matlab()

    obj = eng.load(path)['m']
    eng.workspace["obj"] = obj
    fieldnames = eng.eval("fieldnames(obj.constants)")
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

def load_timepoint_from_matlab(path, name, timing_override = False):
    eng = matlab.engine.start_matlab()
    data_path = os.path.join(path, 'data')
    data_wildcard = os.path.join(data_path, '*.mat')
    matfile_paths = glob.glob(data_wildcard)
    for p in matfile_paths:
        start_time = time.time()
        mat, obj_name = load_calcium_traces_from_matlab(p, eng)
        dir = eng.eval(obj_name + ".constants.DIR")
        dir = 'I' + dir[1:]
        cons = Cons(dir, timing_override)
        print('[***] LOADED {0:<50s} in: {1:3.3f} seconds'.format(p, time.time() - start_time))

        save_path = os.path.join(Config.LOCAL_DATA_PATH, Config.LOCAL_DATA_TIMEPOINT_FOLDER,
                                 name)
        save_name = cons.NAME_MOUSE + '__' + cons.NAME_DATE + '__' + cons.NAME_PLANE
        Config.save_mat_f(save_path, save_name, data=mat)
        Config.save_cons_f(save_path, save_name, data=cons)

def load_behavior_folders_from_matlab(path, name, timing_override = False):
    date_dirs = [os.path.join(path,x) for x in os.listdir(path) if os.path.isdir(os.path.join(path, x))]
    for date_dir in date_dirs:
        start_time = time.time()
        dirs = [os.path.join(date_dir, x) for x in os.listdir(date_dir) if 'cycle' not in x]
        for dir in dirs:
            cons = Cons(dir, timing_override)
            print('[***] LOADED {0:<50s} in: {1:3.3f} seconds'.format(dir, time.time() - start_time))
            save_path = os.path.join(Config.LOCAL_DATA_PATH, Config.LOCAL_DATA_BEHAVIOR_FOLDER,
                                     name)
            save_name = cons.NAME_MOUSE + '__' + cons.NAME_DATE + '__' + cons.NAME_PLANE
            Config.save_cons_f(save_path, save_name, data=cons)



def load_condition(condition, arg = 'timepoint'):
    name = condition.name
    paths = condition.paths
    for i, path in enumerate(paths):
        if arg == 'timepoint':
            load_timepoint_from_matlab(path, name, condition.timing_override[i])
        elif arg == 'behavior':
            load_behavior_folders_from_matlab(path, name, condition.timing_override[i])
        else:
            raise ValueError('argument for loading matlab files is not recognized')



if __name__ == '__main__':
    # example_path = 'E:/IMPORTANT _DATA/DATA_2P/M187_ofc/7-19-2016/420'
    # mat, config = load_single_from_matlab(example_path)

    # condition = 'OFC'
    # path = 'E:/IMPORTANT _DATA/DATA_2P/M187_ofc/training_LEARNING'
    # load_timepoint_from_matlab(path, condition, save=True)

    # for condition in conditions.all_conditions():
    #     load_condition(condition)

    # condition = conditions.PIR_CONTEXT
    # load_condition(condition)

    condition = conditions.BEHAVIOR_OFC_MUSH_YFP
    load_condition(condition, arg = 'behavior')

    # condition = conditions.BEHAVIOR_OFC_MUSH_JAWS_HALO
    # condition.paths = [condition.paths[0]]
    # load_condition(condition, arg = 'timepoint')