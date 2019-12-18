import os
from collections import defaultdict

import filter
from _CONSTANTS import conditions as experimental_conditions
from _CONSTANTS.config import Config
from behavior.behavior_analysis import analyze_behavior
from reduce import chain_defaultdicts
import plot
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import reduce
from format import *
from scipy.stats import ranksums
from scipy.stats import wilcoxon
import behavior.behavior_config

plt.style.use('default')
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.size'] = 5
mpl.rcParams['font.family'] = 'arial'

experiments = [
    'summary'
]

conditions = [
    experimental_conditions.OFC_COMPOSITE,
    experimental_conditions.MPFC_COMPOSITE,
    # experimental_conditions.BEHAVIOR_OFC_YFP_PRETRAINING,
    # experimental_conditions.BEHAVIOR_OFC_JAWS_PRETRAINING,
    # experimental_conditions.BEHAVIOR_OFC_HALO_PRETRAINING,
    # experimental_conditions.BEHAVIOR_OFC_YFP_DISCRIMINATION,
    # experimental_conditions.BEHAVIOR_OFC_JAWS_DISCRIMINATION,
    # experimental_conditions.BEHAVIOR_OFC_MUSH_HALO,
    # experimental_conditions.BEHAVIOR_OFC_MUSH_JAWS,
    # experimental_conditions.BEHAVIOR_OFC_MUSH_YFP,
    # experimental_conditions.OFC,
    # experimental_conditions.PIR,
    # experimental_conditions.OFC_LONGTERM,
    # experimental_conditions.BLA_LONGTERM,
    # experimental_conditions.BEHAVIOR_OFC_JAWS_MUSH,
    # experimental_conditions.BEHAVIOR_OFC_HALO_MUSH,
    # experimental_conditions.BEHAVIOR_OFC_JAWS_MUSH_UNUSED,
    # experimental_conditions.BEHAVIOR_OFC_MUSH_YFP,
    # experimental_conditions.BLA,
    # experimental_conditions.BLA_JAWS,
    # experimental_conditions.OFC_REVERSAL,
    # experimental_conditions.OFC_STATE
]

collapse_arg = 'condition'
def _collapse_conditions(res, control_condition, str):
    conditions = res['condition'].copy().astype('<U20')
    control_ix = conditions == control_condition
    conditions[control_ix] = 'YFP'
    conditions[np.invert(control_ix)] = 'INH'
    res[str] = conditions

def custom_convert(res, condition, start_time_before_water = 1, end_time_before_water = 0):
    from behavior.behavior_config import behaviorConfig
    '''

    :param res:
    :param condition:
    :return:
    '''
    def _get_number_of_licks(mat, start, end):
        mask = mat > 1
        on_off = np.diff(mask, n=1)
        n_licks = np.sum(on_off[start:end] > 0)
        return n_licks

    def _get_time_of_first_lick(mat, start, end, sample_rate):
        mask = mat > 1
        on_off = np.diff(mask, n=1)[start:end]
        if any(on_off):
            ix = np.argwhere(on_off)[0][0]
            ix /= sample_rate
        else:
            ix = -1
        return ix

    config = behaviorConfig()
    new_res = defaultdict(list)
    toConvert = ['day', 'mouse']
    res_odorTrials = res['ODOR_TRIALS']
    res_data = res['DAQ_DATA']
    for i, odorTrials in enumerate(res_odorTrials):
        mouse = res['mouse'][i]

        if hasattr(condition, 'csp'):
            relevant_odors = condition.odors[mouse]
            csps = condition.csp[mouse]
            csms = [x for x in relevant_odors if not np.isin(x, csps)]
        elif hasattr(condition, 'dt_csp'):
            relevant_odors = condition.dt_odors[mouse] + condition.pt_odors[mouse]
            csps = condition.pt_csp[mouse] + condition.dt_csp[mouse]
            csms = [x for x in relevant_odors if not np.isin(x, csps)]
        else:
            raise ValueError('cannot find odors')

        for j, odor in enumerate(odorTrials):
            if odor in relevant_odors:
                start_odor = int((res['DAQ_O_ON'][i]) * res['DAQ_SAMP'][i])
                start = int((res['DAQ_W_ON'][i] - start_time_before_water) * res['DAQ_SAMP'][i])
                end = int((res['DAQ_W_ON'][i] - end_time_before_water) * res['DAQ_SAMP'][i])
                end_coll = int((res['DAQ_W_ON'][i] + 1) * res['DAQ_SAMP'][i])
                lick_data = res_data[i][:,res['DAQ_L'][i],j]
                if odor in csms:
                    end += int(config.extra_csm_time * res['DAQ_SAMP'][i])
                n_licks = _get_number_of_licks(lick_data, start, end)
                n_licks_baseline = _get_number_of_licks(lick_data, 0, start_odor)
                n_licks_coll = _get_number_of_licks(lick_data, end, end_coll)
                time_first_lick = _get_time_of_first_lick(lick_data, start_odor, end, res['DAQ_SAMP'][i])
                new_res['odor'].append(odor)
                new_res['lick'].append(n_licks)
                new_res['ix'].append(j)
                new_res['time_first_lick'].append(time_first_lick)
                new_res['lick_baseline'].append(n_licks_baseline)
                new_res['lick_collection'].append(n_licks_coll)
                for names in toConvert:
                    new_res[names].append(res[names][i])
    for key, val in new_res.items():
        new_res[key] = np.array(val)
    return new_res

def custom_analyze_behavior(data_path, condition, start_time, end_time):
    import analysis
    from behavior import behavior_analysis
    res = analysis.load_all_cons(data_path)
    analysis.add_indices(res)
    analysis.add_time(res)
    lick_res = custom_convert(res, condition, start_time, end_time)
    days_per_mouse = behavior_analysis._get_days_per_condition(data_path, condition)
    last_day_per_mouse = np.array([x[-1] for x in days_per_mouse])
    plot_res = behavior_analysis.agglomerate_days(lick_res, condition, condition.training_start_day, last_day_per_mouse)
    analysis.add_odor_value(plot_res, condition)
    behavior_analysis.add_behavior_stats(plot_res, condition)
    return plot_res

def custom_get_res(start_time, end_time):
    list_of_res = []
    for i, condition in enumerate(conditions):
        if any(s in condition.name for s in ['YFP', 'HALO', 'JAWS']):
            data_path = os.path.join(Config.LOCAL_DATA_PATH, Config.LOCAL_DATA_BEHAVIOR_FOLDER, condition.name)
        else:
            data_path = os.path.join(Config.LOCAL_DATA_PATH, Config.LOCAL_DATA_TIMEPOINT_FOLDER, condition.name)
        res = custom_analyze_behavior(data_path, condition, start_time=start_time, end_time=end_time)

        if condition.name == 'OFC_LONGTERM':
            res = filter.exclude(res, {'mouse':3})

        if 'YFP' in condition.name:
            res['condition'] = np.array(['YFP'] * len(res['mouse']))
        elif 'JAWS' in condition.name:
            res['condition'] = np.array(['JAWS'] * len(res['mouse']))
        elif 'HALO' in condition.name:
            res['condition'] = np.array(['HALO'] * len(res['mouse']))
        else:
            res['condition'] = np.array([condition.name] * len(res['mouse']))
        list_of_res.append(res)
    all_res = defaultdict(list)
    for res, condition in zip(list_of_res, conditions):
        reduce.chain_defaultdicts(all_res, res)
    return all_res

names = []
for i, condition in enumerate(conditions):
    names.append(condition.name)
directory_name = ','.join(names)
save_path = os.path.join(Config.LOCAL_FIGURE_PATH, 'BEHAVIOR', directory_name)

color_dict_valence = {'PT CS+': 'C1', 'CS+': 'green', 'CS-': 'red'}
color_dict_condition = {'HALO': 'C1', 'JAWS':'red','YFP':'black'}
bool_ax_args = {'yticks': [0, 50, 100], 'ylim': [-5, 105], 'xticks': [0, 50, 100, 150, 200],
                'xlim': [0, 200]}
ax_args_mush = {'yticks': [0, 5, 10], 'ylim': [-1, 12],'xticks': [0, 50, 100],'xlim': [0, 100]}
bool_ax_args_mush = {'yticks': [0, 50, 100], 'ylim': [-5, 105], 'xticks': [0, 100], 'xlim': [0, 100]}
ax_args_dt = {'yticks': [0, 5, 10], 'ylim': [-1, 12],'xticks': [0, 50],'xlim': [0, 50]}
bool_ax_args_dt = {'yticks': [0, 50, 100], 'ylim': [-5, 105], 'xticks': [0, 50], 'xlim': [0, 50]}
ax_args_pt = {'yticks': [0, 5, 10], 'ylim': [-1, 12], 'xticks': [0, 50, 100, 150, 200], 'xlim': [0, 200]}
bool_ax_args_pt = {'yticks': [0, 50, 100], 'ylim': [-5, 105], 'xticks': [0, 50, 100, 150, 200], 'xlim': [0, 200]}
ax_args_output = {'yticks': [0, 5, 10], 'ylim': [-1, 12], 'xticks': [0, 100, 200, 300], 'xlim': [0, 300]}
bool_ax_args_output = {'yticks': [0, 50, 100], 'ylim': [-5, 105], 'xticks': [0, 100, 200, 300], 'xlim': [0, 300]}
bar_args = {'alpha': .6, 'fill': False}
scatter_args = {'marker': '.', 's': 4, 'alpha': .6}

collection = False
if collection:
    lick = 'lick_collection'
    lick_smoothed = 'lick_collection_smoothed'
    boolean_smoothed = 'boolean_collection_smoothed'
    boolean_sem = 'boolean_collection_smoothed_sem'
    lick_sem = 'lick_collection_smoothed_sem'
else:
    lick = 'lick'
    lick_smoothed = 'lick_smoothed'
    boolean_smoothed = 'boolean_smoothed'
    boolean_sem = 'boolean_smoothed_sem'
    lick_sem = 'lick_smoothed_sem'

if 'summary' in experiments:
    full = defaultdict(list)
    list_of_res = []
    for time in np.arange(.5, 5, .5):
        start_time = time
        end_time = 0
        all_res = custom_get_res(start_time= start_time, end_time= end_time)
        all_res = filter.filter(all_res, {'odor_valence':['CS+','CS-', 'PT CS+']})
        all_res_lick = reduce.new_filter_reduce(all_res, filter_keys=['condition', 'odor_valence','mouse'], reduce_key=lick)

        for i, x in enumerate(all_res_lick[lick]):
            all_res_lick['training_end_licks'].append(np.mean(x[-20:]))
            all_res_lick['start_time'].append(start_time)
            all_res_lick['end_time'].append(end_time)
        for k, v in all_res_lick.items():
            all_res_lick[k]= np.array(v)
        list_of_res.append(all_res_lick)

    for res in list_of_res:
        reduce.chain_defaultdicts(full, res)

    line_args_copy = line_args.copy()
    line_args_copy.update({'marker': '.', 'linewidth':.5, 'markersize':.5})

    valences = np.unique(full['odor_valence'])
    for valence in valences:
        color = [color_dict_valence[valence]]

        mean_std_res = reduce.new_filter_reduce(full, filter_keys=['odor_valence','start_time'],
                                                reduce_key='training_end_licks')

        plot.plot_results(full, x_key='start_time', y_key='training_end_licks', loop_keys=['mouse','condition'],
                          colors =color * 50,
                          select_dict={'odor_valence':valence},
                          plot_args= line_args_copy,
                          legend=False,
                          path=save_path)

        mean_std_res['rho_over_mu'] = mean_std_res['training_end_licks_std'] / mean_std_res['training_end_licks']

        plot.plot_results(mean_std_res, x_key='start_time', y_key= 'rho_over_mu',
                          select_dict={'odor_valence': valence},
                          plot_function= plt.scatter,
                          plot_args= scatter_args,
                          fig_size=[2, 1.5],
                          path=save_path)


        stats = filter.filter(mean_std_res, {'odor_valence':valence})
        print(stats['start_time'])
        print(stats['training_end_licks'])
        print(stats['training_end_licks_std'])

