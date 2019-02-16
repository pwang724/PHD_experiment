from _CONSTANTS.config import Config
import numpy as np
import os
import glob
import tools.file_io
from collections import defaultdict
from scipy import stats as sstats
import reduce

def load_data(data_path):
    res = load_all_cons(data_path)
    data_pathnames = sorted(glob.glob(os.path.join(data_path, '*' + Config.mat_ext)))
    list_of_all_data = np.array([Config.load_mat_f(d) for d in data_pathnames])
    res['data'] = list_of_all_data
    assert len(list_of_all_data) == len(res['TRIAL_FRAMES']), 'number of data files does not equal number of cons files'
    return res

def load_all_cons(data_path):
    res = defaultdict(list)

    config_pathnames = sorted(glob.glob(os.path.join(data_path, '*' + Config.cons_ext)))
    for i, config_pn in enumerate(config_pathnames):
        cons = Config.load_cons_f(config_pn)
        for key, val in cons.__dict__.items():
            if isinstance(val, list):
                res[key].append(np.array(val))
            else:
                res[key].append(val)
    for key, val in res.items():
        if key == 'DAQ_DATA':
            arr = np.empty(len(val), dtype='object')
            for i, v in enumerate(val):
                arr[i] = v
            res[key] = arr
        else:
            res[key] = np.array(val)
    return res

def load_results_scores(data_path):
    res = defaultdict(list)
    experiment_dirs = sorted([os.path.join(data_path, d) for d in os.listdir(data_path)])
    for exp_dir in experiment_dirs:
        data_dirs = sorted(glob.glob(os.path.join(exp_dir, '*' + Config.mat_ext)))
        config_dirs = sorted(glob.glob(os.path.join(exp_dir, '*.json')))
        for data_dir, config_dir in zip(data_dirs,config_dirs):
            config = tools.file_io.load_json(config_dir)
            data = tools.file_io.load_numpy(data_dir)

            # data is in format of experiment X time X CVfold X repeat
            res['data'].append(data)
            for key, val in config.items():
                res[key].append(val)

    for key, val in res.items():
        try:
            res[key] = np.array(val)
        except:
            arr = np.empty(len(val), dtype='object')
            for i, v in enumerate(val):
                arr[i] = v
            res[key] = arr
    return res

def load_results_train_test_scores(data_path):
    res = defaultdict(list)
    experiment_dirs = sorted([os.path.join(data_path, d) for d in os.listdir(data_path)])

    keys = ['DAQ_O_ON_F', 'DAQ_O_OFF_F', 'DAQ_W_ON_F',
            'DAQ_O_ON', 'DAQ_O_OFF', 'DAQ_W_ON',
            'TRIAL_FRAMES', 'TRIAL_PERIOD','NAME_MOUSE', 'NAME_PLANE', 'NAME_DATE',
            'decode_style', 'neurons','shuffle']
    for exp_dir in experiment_dirs:
        data_dirs = sorted(glob.glob(os.path.join(exp_dir, '*' + '.pkl')))
        config_dirs = sorted(glob.glob(os.path.join(exp_dir, '*.json')))
        for data_dir, config_dir in zip(data_dirs,config_dirs):
            config = tools.file_io.load_json(config_dir)
            cur_res = tools.file_io.load_pickle(data_dir)

            for k in keys:
                cur_res[k] = np.array([config[k]] * len(cur_res['scores']))
            reduce.chain_defaultdicts(res, cur_res)

    add_indices(res)
    add_time(res)

    for i, _ in enumerate(res['scores']):
        start =res['DAQ_O_ON_F'][i]
        # res['top_score'].append(np.max(res['scores'][i][start:]))
        res['top_score'].append(np.max(res['scores'][i]))
    res['top_score'] = np.array(res['top_score'])
    return res

def analyze_results(res, condition, arg='different'):
    add_indices(res)
    add_time(res)
    add_decode_stats(res, condition, arg)

def add_indices(res, arg_plane = True):
    #TODO: fix issue for multiple planes in the same day. add 'effective days'
    from scipy.stats import rankdata

    list_of_dates = res['NAME_DATE']
    list_of_planes = res['NAME_PLANE']
    list_of_mice = res['NAME_MOUSE']

    if arg_plane:
        list_of_dates = list(list_of_dates)
        for i in range(len(list_of_dates)):
            list_of_dates[i] = list_of_dates[i] +'_' + list_of_planes[i]
        list_of_dates = np.array(list_of_dates)

    days = np.zeros_like(list_of_dates, dtype=int)
    _, mouse_ixs = np.unique(list_of_mice, return_inverse=True)
    for mouse_ix in np.unique(mouse_ixs):
        ixs = mouse_ixs == mouse_ix
        mouse_dates = list_of_dates[ixs]
        sorted_dates = rankdata(mouse_dates, 'dense') - 1
        days[ixs] = sorted_dates
    res['mouse'] = mouse_ixs
    res['day'] = days

def add_time(res):
    nExperiments = res['DAQ_O_ON'].size
    for i in range(nExperiments):
        nF = res['TRIAL_FRAMES'][i]
        period = res['TRIAL_PERIOD'][i]
        O_on = res['DAQ_O_ON'][i]
        O_off = res['DAQ_O_OFF'][i]
        W_on = res['DAQ_W_ON'][i]
        time = np.arange(0, nF) * period - O_on
        xticks = np.asarray([O_on, O_off, W_on]) - O_on
        xticks = np.round(xticks, 1)
        res['time'].append(time)
        res['xticks'].append(xticks)
    res['time'] = np.array(res['time'])
    res['xticks'] = np.array(res['xticks'])

def add_aligned_days(res, last_days, learned_days):
    list_of_days = learned_days
    new_days = np.zeros_like(res['day'])
    mice, ix = np.unique(res['mouse'], return_inverse=True)
    for i, mice in enumerate(mice):
        current_ix = ix == i
        days = res['day'][current_ix]
        new_days[current_ix] = days - list_of_days[i]
    res['day_aligned'] = new_days


#add relevant stats
def add_decode_stats(res, condition, arg='different'):
    '''

    :param res:
    :param condition:
    :param arg:
    :return:
    '''
    # decoding data is in format of experiment X time X CVfold X repeat
    datas = np.array(res['data'])
    O_on = res['DAQ_O_ON_F'].astype(np.int)
    O_off = res['DAQ_O_OFF_F'].astype(np.int)
    W_on = res['DAQ_W_ON_F'].astype(np.int)
    for i, data in enumerate(datas):
        data_reshaped = np.reshape(data.transpose([0, 2, 1]),
                                   [data.shape[0], data.shape[1] * data.shape[2]])
        mean = np.mean(data_reshaped, axis=1)
        sem = sstats.sem(data_reshaped, axis=1)

        if arg == 'different':
            if condition.name == 'PIR' or condition.name == 'PIR_NAIVE':
                # max = np.max(mean[O_on[i]: W_on[i]])
                max = np.max(mean)
            else:
                # max = np.mean(mean[O_off[i]: W_on[i]])
                max = np.mean(mean)
        elif arg == 'same':
            # max = np.mean(mean[O_off[i]: W_on[i]])
            max = np.mean(mean)
        else:
            raise ValueError('Argument for calculating summary decoding metric, max, is not known: {}'.format(arg))
        res['mean'].append(mean)
        res['sem'].append(sem)
        res['max'].append(max)

    res['mean'] = np.array(res['mean'])
    res['sem'] = np.array(res['sem'])
    res['max'] = np.array(res['max'])


def add_odor_value(res, condition):
    mice, ix = np.unique(res['mouse'], return_inverse=True)
    valence_array = np.zeros_like(res['odor']).astype(object)
    standard_array = np.zeros_like(res['odor']).astype(object)

    for i, mouse in enumerate(mice):
        if hasattr(condition, 'odors'):
            odors = condition.odors[i]
            # if condition.name == 'OFC_CONTEXT' or condition.name == 'BLA_CONTEXT':
            #     csms = condition.csp[i]
            #     csps = [x for x in odors if not np.isin(x, csms)]
            # else:
            csps = condition.csp[i]
            csms = [x for x in odors if not np.isin(x, csps)]
            standard_dict = {}
            valence_dict = {}
            j=1
            for csp in csps:
                standard_dict[csp] = 'CS+' + str(j)
                valence_dict[csp] = 'CS+'
                j+=1
            j=1
            for csm in csms:
                standard_dict[csm] = 'CS-' + str(j)
                valence_dict[csm] = 'CS-'
                j+=1
            standard_dict['water'] = 'US'
            valence_dict['water'] = 'US'
        else:
            dt_odors = condition.dt_odors[i]
            dt_csp = condition.dt_csp[i]
            dt_csm = [x for x in dt_odors if not np.isin(x, dt_csp)]
            pt_odors = condition.pt_odors[i]
            pt_csp = condition.pt_csp[i]
            pt_naive = [x for x in pt_odors if not np.isin(x, pt_csp)]
            assert len(pt_naive) <= 1, 'More than 1 pt naive odor'
            assert len(pt_csp) <= 1, 'More than 1 pt CS+ odor'
            standard_dict = {}
            valence_dict = {}
            j=1
            for csp in dt_csp:
                standard_dict[csp] = 'CS+' + str(j)
                valence_dict[csp] = 'CS+'
                j+=1
            j=1
            for csm in dt_csm:
                standard_dict[csm] = 'CS-' + str(j)
                valence_dict[csm] = 'CS-'
                j+=1
            if len(pt_naive):
                standard_dict[pt_naive[0]] = 'PT Naive'
                valence_dict[pt_naive[0]] = 'PT Naive'
            standard_dict[pt_csp[0]] = 'PT CS+'
            valence_dict[pt_csp[0]] = 'PT CS+'
            j += 1

        mouse_ix = ix == i
        mouse_odors = res['odor'][mouse_ix]
        valence_array[mouse_ix] = [valence_dict[o] for o in mouse_odors]
        standard_array[mouse_ix] = [standard_dict[o] for o in mouse_odors]
    res['odor_valence'] = valence_array
    res['odor_standard'] = standard_array


def add_naive_learned(res, start_day_per_mouse, learned_day_per_mouse, str1 = 'Naive', str2='Learned'):
    for i in range(len(res['day'])):
        day = res['day'][i]
        mouse = res['mouse'][i]
        if start_day_per_mouse[mouse] == day:
            res['training_day'].append(str1)
        elif learned_day_per_mouse[mouse] == day:
            res['training_day'].append(str2)
        else:
            raise ValueError('day is not either start day or learned day')
    res['training_day'] = np.array(res['training_day'])