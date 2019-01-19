from tools import file_io
import os

class Config:
    save_mat_f = file_io.save_numpy
    load_mat_f = file_io.load_numpy
    mat_ext = '.npy'

    save_cons_f = file_io.save_pickle
    load_cons_f = file_io.load_pickle
    cons_ext = '.pkl'

    DECODE_CONFIG_JSON = 'decodeConfig'

    cwd = r'C:\Users\P\Desktop\PYTHON\PHD_experiment'
    # cwd = '/Users/pwang/Desktop/GITHUB_PROJECTS/PHD_experiment'
    LOCAL_DATA_PATH = os.path.join(cwd,'_DATA')
    LOCAL_DATA_SINGLE_FOLDER = 'single'
    LOCAL_DATA_TIMEPOINT_FOLDER = 'timepoint'

    LOCAL_FIGURE_PATH = os.path.join(cwd,'_FIGURES')
    LOCAL_EXPERIMENT_PATH = os.path.join(cwd,'_EXPERIMENTS')
    LOCAL_ANALYSIS_PATH = os.path.join('_ANALYSIS')