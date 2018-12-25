import os
import glob
from shutil import copyfile
from distutils.dir_util import copy_tree
import time
import numpy as np

def _easy_copytree(old_name, new_name):
    st = time.time()
    if os.path.exists(old_name):
        if not os.path.exists(new_name):
            os.makedirs(new_name)
        copy_tree(old_name, new_name)
        duration = np.round(time.time() - st,1)
        print('finished copying {} in {} seconds'.format(old_name, duration))
    else:
        raise ValueError('did not copy {}'.format(old_name))

def _easy_copy(old, new):
    st = time.time()
    if os.path.isfile(old):
        copyfile(old, new)
        duration = np.round(time.time() - st, 1)
        print('finished copying {} in {} seconds'.format(old, duration))
    else:
        raise ValueError('did not copy {}'.format(old))

efty_dir = r'E:\IMPORTANT DATA\STORAGE_EFTY'
roi_dir = r'E:\IMPORTANT DATA\STORAGE_ROI'
data_dir = r'E:\IMPORTANT DATA\STORAGE_DATA'
x_dir = r'E:\IMPORTANT DATA\DATA_X'

old_efty_str = '__EFTYf'
new_efty_str = '_EFTY_F'

old_efty_kalman_str = '__EFTYfk'
new_efty_kalman_str = '_EFTY_FK'

raw_str = '_'
z_str = '_z'
old_mask_str = 'm'
new_mask_str = '_m.tif'

base_dir = r'E:\IMPORTANT DATA\DATA_2P\M187_ofc'
date_dirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if d[0].isdigit()]

for current_dir in date_dirs:
    processed_folder = 'PROCESSED'
    processed_dir = os.path.join(base_dir, os.path.split(current_dir)[1], processed_folder)
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)

    p, date = os.path.split(current_dir)
    pp, mouse = os.path.split(p)
    plane_folders = []
    for p in os.listdir(current_dir):
        if p[-5:] != 'cycle' and p[-5:] != 'ESSED':
            plane_folders.append(p)

    for plane in plane_folders:
        print('[***] Starting with: {}'.format(os.path.join(current_dir, plane)))
        file = mouse + '_' + date + '_' + plane

        mask_files = [n for n in os.listdir(os.path.join(x_dir, mouse, date)) if n[-5:] == 'm.tif']
        if len(mask_files) > 1:
            for mask_file in mask_files:
                old_m_name = os.path.join(x_dir, mouse, date, mask_file)
                new_m_name = os.path.join(processed_dir, mouse + '_' + date + '_' + mask_file)
                _easy_copy(old_m_name, new_m_name)
        else:
            old_m_name = os.path.join(x_dir, mouse, date, mask_files[0])
            new_m_name = os.path.join(processed_dir, file + new_mask_str)
            _easy_copy(old_m_name, new_m_name)

        old_raw_name = os.path.join(x_dir, mouse, date, plane + raw_str)
        new_raw_name = os.path.join(processed_dir, file + raw_str)
        _easy_copytree(old_raw_name, new_raw_name)

        old_efty_name = os.path.join(x_dir, mouse, date, plane + old_efty_str)
        new_efty_name = os.path.join(processed_dir, file + new_efty_str)
        _easy_copytree(old_efty_name, new_efty_name)

        old_efty_kalman_name = os.path.join(x_dir, mouse, date, plane + old_efty_kalman_str)
        new_efty_kalman_name = os.path.join(processed_dir, file + new_efty_kalman_str)
        _easy_copytree(old_efty_kalman_name, new_efty_kalman_name)

        old_z_name = os.path.join(x_dir, mouse, date, plane + z_str)
        new_z_name = os.path.join(processed_dir, file + z_str)
        _easy_copytree(old_z_name, new_z_name)

        efty_file = os.path.join(efty_dir, file + '.mat')
        new_efty_file = os.path.join(processed_dir, file + '_EFTY.mat')
        _easy_copy(efty_file, new_efty_file)

        roi_file = os.path.join(roi_dir, file + '.mat')
        new_roi_file = os.path.join(processed_dir, file + '_ROI.mat')
        _easy_copy(roi_file, new_roi_file)

        data_file = os.path.join(data_dir, file + '.mat')
        new_data_file = os.path.join(processed_dir, file + '_DATA.mat')
        _easy_copy(data_file, new_data_file)

        t = 5
        print('[***] Done with: {}'.format(os.path.join(current_dir,plane)))
        print('[***] Sleeping for {}'.format(t))
        time.sleep(t)
