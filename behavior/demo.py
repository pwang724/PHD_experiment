import CONSTANTS.conditions as experimental_conditions
import os
import numpy as np
from CONSTANTS.config import Config
from collections import defaultdict
import analysis
from behavior.behavior_config import behaviorConfig
import matplotlib.pyplot as plt
import filter
import plot
from scipy.signal import savgol_filter

from reduce import reduce_by_concat, reduce_by_mean
from tools.utils import append_defaultdicts, chain_defaultdicts

def convert(res, condition):
    '''

    :param res:
    :param condition:
    :return:
    '''
    def _parseLick(mat, start, end):
        mask = mat > 1
        on_off = np.diff(mask, n=1)
        n_licks = np.sum(on_off[start:end] > 0)
        return n_licks

    config = behaviorConfig()
    new_res = defaultdict(list)
    toConvert = ['day', 'mouse']
    res_odorTrials = res['ODOR_TRIALS']
    res_data = res['DAQ_DATA']
    for i, odorTrials in enumerate(res_odorTrials):
        mouse = res['mouse'][i]
        relevant_odors = condition.odors[mouse]
        csps = condition.csp[mouse]
        csms = list(set(relevant_odors) - set(csps))

        for j, odor in enumerate(odorTrials):
            if odor in relevant_odors:
                start = int(res['DAQ_O_ON'][i] * res['DAQ_SAMP'][i])
                end = int(res['DAQ_W_ON'][i] * res['DAQ_SAMP'][i])
                lick_data = res_data[i][:,res['DAQ_L'][i],j]
                if odor in csms:
                    end += int(config.extra_csm_time * res['DAQ_SAMP'][i])
                n_licks = _parseLick(lick_data, start, end)

                if odor in csms:
                    ix = csms.index(odor)
                    new_res['odor_valence'].append('CS-')
                    new_res['odor_standard'].append('CS-' + str(ix + 1))
                else:
                    ix = csps.index(odor)
                    new_res['odor_valence'].append('CS+')
                    new_res['odor_standard'].append('CS+' + str(ix + 1))

                new_res['odor'].append(odor)
                new_res['lick'].append(n_licks)
                new_res['ix'].append(j)
                for names in toConvert:
                    new_res[names].append(res[names][i])
    for key, val in new_res.items():
        new_res[key] = np.array(val)
    return new_res

def agglomerate_days(res, condition, first_day, last_day):
    mice = np.unique(res['mouse'])
    out = defaultdict(list)

    for i, mouse in enumerate(mice):
        for odor in condition.odors[mouse]:
            filter_dict = {'mouse': mouse, 'day': np.arange(first_day[i], last_day[i]), 'odor': odor}
            filtered_res = filter.filter(res, filter_dict)
            temp_res = reduce_by_concat(filtered_res, 'lick', rank_keys=['day', 'ix'])
            temp_res['trial'] = np.arange(len(temp_res['lick']))
            append_defaultdicts(out, temp_res)
    for key, val in out.items():
        out[key] = np.array(val)
    return out

def analyze_behavior(res):
    '''

    :param res:
    :return:
    '''

    def _half_max_up(vec):
        config = behaviorConfig()
        vec_binary = vec > config.halfmax_up_threshold

        if np.all(vec_binary):
            half_max = 0
        else:
            last_ix_below_threshold = np.where(vec_binary == 0)[0][-1]
            vec_binary[:last_ix_below_threshold] = 0
            if np.any(vec_binary):
                half_max = np.where(vec_binary == 1)[0][0]
            else:
                half_max = None
        return half_max

    # TODO: implement find half-max last down
    def _half_max_down(vec):
        pass

    config = behaviorConfig()
    res['lick_smoothed'] = [savgol_filter(y, config.smoothing_window, config.polynomial_degree)
                            for y in res['lick']]
    res['boolean_smoothed'] = [
        100 * savgol_filter(y > 0, config.smoothing_window_boolean, config.polynomial_degree)
        for y in res['lick']]

    for x in res['boolean_smoothed']:
        res['half_max'].append(_half_max_up(x))

    for key, val in res.items():
        res[key] = np.array(val)

#inputs
def get_behavior_analysis(data_path):
    res = analysis.load_all_cons(data_path)
    analysis.add_indices(res)
    analysis.add_time(res)
    lick_res = convert(res, condition)
    plot_res = agglomerate_days(lick_res, condition, condition.training_start_day,
                                filter.get_last_day_per_mouse(res))
    analyze_behavior(plot_res)
    return plot_res

def get_summary(plot_res):
    # summarize data
    summary_res = defaultdict(list)
    mice = np.unique(plot_res['mouse'])
    for i, mouse in enumerate(mice):
        filter_dict = {'mouse': mouse, 'odor': condition.csp[i]}
        cur_res = filter.filter(plot_res, filter_dict)
        temp_res = reduce_by_mean(cur_res, 'half_max')
        append_defaultdicts(summary_res, temp_res)
    return summary_res

def plot_individual(plot_res, summary_res, save_path):
    #plot
    colors = ['green','lime','red','maroon']
    plot_args = {'marker':'o', 'markersize':1, 'alpha':.6, 'linewidth':1}
    ax_args = {'yticks':[0, 10, 20, 30, 40], 'ylim':[-1, 41], 'xticks':[0, 20, 40, 60, 80, 100], 'xlim':[0, 100]}

    mice = np.unique(plot_res['mouse'])
    for i, mouse in enumerate(mice):
        select_dict = {'mouse':mouse}
        plot.plot_results(plot_res, x_key='trial', y_key = 'lick_smoothed', loop_keys = 'odor',
                          select_dict= select_dict, colors=colors, ax_args=ax_args, plot_args=plot_args, path = save_path)
        plot.plot_results(plot_res, x_key='trial', y_key = 'lick', loop_keys = 'odor',
                          select_dict= select_dict, colors=colors, ax_args=ax_args, plot_args=plot_args, path = save_path)

    ax_args = {'yticks':[0, 50, 100], 'ylim':[-5, 105], 'xticks':[0, 20, 40, 60, 80, 100], 'xlim':[0, 100]}
    mice = np.unique(plot_res['mouse'])
    for i, mouse in enumerate(mice):
        select_dict = {'mouse':mouse}
        plot.plot_results(plot_res, x_key='trial', y_key = 'boolean_smoothed', loop_keys = 'odor',
                          select_dict= select_dict, colors=colors, ax_args=ax_args, plot_args=plot_args, path = save_path)

    #bar plot
    csp_plot_res = filter.filter_odors_per_mouse(plot_res, condition.csp)
    colors = ['black','black']
    select_dict = {'odor_valence':'CS+'}
    ax_args = {'yticks':[0, 20, 40, 60, 80], 'ylim':[0, 80]}
    plot_args = {'marker':'o', 's':10, 'facecolors': 'none', 'alpha':.6}
    plot.plot_results(plot_res, x_key='mouse', y_key = 'half_max', loop_keys='odor_standard', colors = colors,
                      select_dict= select_dict, path=save_path, plot_function= plt.scatter, plot_args=plot_args,
                      ax_args=ax_args, save = False)

    plot_args = {'alpha':.6, 'fill': False}
    plot.plot_results(summary_res, x_key='mouse', y_key = 'half_max', loop_keys=None,
                      path=save_path, plot_function= plt.bar, plot_args=plot_args,
                      ax_args=ax_args, save = True, reuse= True)

conditions = [experimental_conditions.PIR, experimental_conditions.OFC, experimental_conditions.BLA]
# conditions = [experimental_conditions.PIR]

summary_all = defaultdict(list)
for condition in conditions:
    data_path = os.path.join(Config.LOCAL_DATA_PATH, Config.LOCAL_DATA_TIMEPOINT_FOLDER, condition.name)
    save_path = os.path.join(Config.LOCAL_FIGURE_PATH, 'BEHAVIOR', condition.name)
    plot_res = get_behavior_analysis(data_path)
    summary_res = get_summary(plot_res)
    # plot_individual(plot_res, summary_res, save_path)

    #fix this
    summary_res['name'] = [condition.name] * len(summary_res['half_max'])
    chain_defaultdicts(summary_all, summary_res)

save_path = os.path.join(Config.LOCAL_FIGURE_PATH, 'BEHAVIOR', 'TEST')
plot_args = {'marker':'o', 's':10, 'facecolors': 'none', 'alpha':.6}
plot.plot_results(summary_all, x_key= 'name', y_key= 'half_max', loop_keys=None, path=save_path,
                  plot_function= plt.scatter, plot_args= plot_args)






