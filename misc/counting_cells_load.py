import numpy as np
import _CONSTANTS.conditions as conditions
import _CONSTANTS.config as config
import os
import glob
from tools import file_io
import matlab.engine
import time

mouse = 2
condition = conditions.OFC_COMPOSITE

d = os.path.join(condition.paths[mouse], 'data')
mat_files = glob.glob(os.path.join(d,'*.mat'))
eng = matlab.engine.start_matlab()

rois = []
for mat_file in mat_files:
    start_time = time.time()
    x = eng.load(mat_file)
    obj = list(x.values())[0]
    obj_name = "obj"
    eng.workspace[obj_name] = obj

    roi_m = eng.eval(obj_name + ".roi")
    roi = np.asarray(roi_m).squeeze()
    print('Time taken: {}'.format(time.time() - start_time))

    rois.append(roi)
rois = np.stack(rois)
print(rois.shape)

rois = rois.astype('float32')
data_directory = config.Config().LOCAL_DATA_PATH
data_directory = os.path.join(data_directory, 'registration','ROI', condition.name)
file_io.save_numpy(data_directory, str(mouse), rois)

