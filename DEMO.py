import numpy as np
import os
import decoding
import glob
from CONSTANTS.config import Config
import decode_config
import time

dc = decode_config.DecodeConfig()

mouse_files = [os.path.join(dc.data_path, o) for o in os.listdir(dc.data_path)
               if os.path.isdir(os.path.join(dc.data_path, o))]

#decode
condition_odors = dc.condition.odors
condition_csps = dc.condition.csp
all_decode_data = []
for i, (mouse_file, odors, csps) in enumerate(zip(mouse_files, condition_odors, condition_csps)):
    start_time = time.time()
    data_pathnames = glob.glob(os.path.join(mouse_file, '*.txt'))
    config_pathnames = glob.glob(os.path.join(mouse_file, '*.pkl'))
    for j, (data_pathname, config_pathname) in enumerate(zip(data_pathnames, config_pathnames)):
        cons = Config.load_cons_f(config_pathname)
        data = Config.load_mat_f(data_pathname)

        scores = decoding.decode(cons, data, odors, csps, dc)

        save_folder = os.path.split(mouse_file)[1]
        save_name = os.path.splitext(os.path.split(data_pathname)[1])[0]
        Config.save_mat_f(save_path=os.path.join(dc.save_path, save_folder), save_name=save_name, data=scores)
    print("Analyzed: {0:s} in {1:.2f} seconds".format(mouse_file, time.time()-start_time))
