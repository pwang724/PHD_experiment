from init.cons import Cons
import matlab
import init.load_matlab

import _CONSTANTS.conditions as experimental_conditions
import os
import glob
from _CONSTANTS.config import Config

def _look_at_timing(cons):
    on = cons.DAQ_O_ON_F
    off = cons.DAQ_O_OFF_F
    us = cons.DAQ_W_ON_F
    dir = cons.DIR
    odors = cons.ODOR_UNIQUE
    str= 'Odor ON: {0:d}, Odor OFF: {1:d}, US: {2:d}, DIR: {3:s}, ODORS: {4:s}'.format(on, off, us, dir, '|'.join(odors))
    print(str)

# condition = experimental_conditions.BEHAVIOR_OFC_MUSH_HALO
# data_path = os.path.join(Config.LOCAL_DATA_PATH, Config.LOCAL_DATA_TIMEPOINT_FOLDER, condition.name)
condition = experimental_conditions.BEHAVIOR_OFC_MUSH_HALO
data_path = os.path.join(Config.LOCAL_DATA_PATH, Config.LOCAL_DATA_BEHAVIOR_FOLDER, condition.name)

# load cons
# paths = condition.paths
# import matlab.engine
# eng = matlab.engine.start_matlab()
# for path in paths:
#     data_path = os.path.join(path, 'data')
#     data_wildcard = os.path.join(data_path, '*.mat')
#     matfile_paths = glob.glob(data_wildcard)
#     list_of_mats, list_of_cons = [],[]
#     for i, p in enumerate(matfile_paths):
#         mat, obj_name = init.load_matlab.load_calcium_traces_from_matlab(p, eng)
#         dir = eng.eval(obj_name + ".constants.DIR")
#         cons = Cons(dir, condition.timing_override[i])
#         _look_at_timing(cons)


#look at cons once loaded
config_pathnames = glob.glob(os.path.join(data_path, '*' + Config.cons_ext))

for i, config_pn in enumerate(config_pathnames):
    cons = Config.load_cons_f(config_pn)
    _look_at_timing(cons)
    # print(cons.DIR + '__' + str(cons.TRIAL_FRAMES))
    # print(cons.DIR + '__' + str(cons.ODOR_TRIALS))
