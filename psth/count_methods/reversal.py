import copy
from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt

import filter
import plot
import reduce
from format import *

def plot_reversal(res, start_days, end_days, figure_path):
    ax_args_copy = ax_args.copy()
    ax_args_copy.update({'ylim':[0, .6]})
    res = copy.copy(res)
    list_of_days = list(zip(start_days, end_days))
    start_end_day_res = filter.filter_days_per_mouse(res, days_per_mouse=list_of_days)
    reversal_res, stats_res = get_reversal_sig(start_end_day_res)
    filter.assign_composite(reversal_res, loop_keys=['day','odor_valence'])

    mean_res = reduce.new_filter_reduce(reversal_res, filter_keys=['day','odor_valence'], reduce_key='Fraction')
    plot.plot_results(mean_res,
                      x_key='day_odor_valence', y_key='Fraction', error_key='Fraction_sem',
                      path=figure_path,
                      plot_function=plt.errorbar, plot_args=error_args, ax_args=ax_args_copy,
                      fig_size=(2, 1.5), save=False)
    plt.plot([1.5, 1.5], plt.ylim(), '--', color='gray', linewidth=2)
    plot.plot_results(reversal_res,
                      x_key='day_odor_valence', y_key='Fraction', loop_keys= 'day_odor_valence',
                      path=figure_path,
                      colors = ['Green','Red','Green','Red'],
                      plot_function=plt.scatter, plot_args=scatter_args, ax_args=ax_args_copy,
                      fig_size=(2, 1.5), reuse=True, save=True,
                      legend=False)
    print(mean_res['day_odor_valence'])
    print(mean_res['Fraction'])


    titles = ['','CS+', 'CS-', 'None']
    conditions = [['none-p','p-m','p-none', 'p-p'], ['p-m','p-none', 'p-p'],['m-m','m-none', 'm-p'],['none-m','none-none', 'none-p']]
    labels = [['Added','Reversed','Lost','Retained'], ['Reversed','Lost','Retained'], ['Retained','Lost','Reversed'], ['to CS-','Retained','to CS+']]
    for i, title in enumerate(titles):
        mean_stats = reduce.new_filter_reduce(stats_res, filter_keys='condition', reduce_key= 'Fraction')
        ax_args_copy.update({'ylim':[-.1, 1], 'yticks':[0, .5, 1], 'xticks':[0,1,2,3],
                             'xticklabels':labels[i]})
        plot.plot_results(mean_stats,
                          select_dict={'condition': conditions[i]},
                          x_key='condition', y_key='Fraction', loop_keys='mouse', error_key='Fraction_sem',
                          sort=True,
                          path=figure_path,
                          colors=['Black'] * 10,
                          plot_function=plt.errorbar, plot_args=error_args, ax_args=ax_args_copy,
                          fig_size=(2, 1.5), save=False)
        plt.title(title)

        plot.plot_results(stats_res, select_dict={'condition': conditions[i]},
                          x_key='condition', y_key='Fraction', loop_keys='mouse', sort=True,
                          path = figure_path,
                          colors=['Black']*10,
                          plot_function=plt.scatter, plot_args=scatter_args, ax_args=ax_args_copy,
                          fig_size=(2, 1.5),
                          legend=False, save=True, reuse=True)

        print(mean_stats['Fraction'])

    print(mean_res['Fraction'])


def get_reversal_sig(res):
    key = 'ssig'
    def _helper(res):
        assert res['odor_valence'][0] == 'CS+', 'wrong odor'
        assert res['odor_valence'][1] == 'CS-', 'wrong odor'
        on = res['DAQ_O_ON_F'][0]
        off = res['DAQ_W_ON_F'][0]
        sig_p = res[key][0]
        sig_m = res[key][1]
        dff_p = res['dff'][0]
        dff_m = res['dff'][1]
        sig_p_mask = sig_p == 1
        sig_m_mask = sig_m == 1
        dff_mask = dff_p - dff_m
        dff_mask = np.mean(dff_mask[:, on:off], axis=1)
        p = [a and b for a, b in zip(sig_p_mask, dff_mask>0)]
        m = [a and b for a, b in zip(sig_m_mask, dff_mask<0)]
        return np.array(p), np.array(m)

    mice = np.unique(res['mouse'])
    res = filter.filter(res, filter_dict={'odor_valence':['CS+','CS-']})
    sig_res = reduce.new_filter_reduce(res, reduce_key=key, filter_keys=['mouse','day','odor_valence'])
    dff_res = reduce.new_filter_reduce(res, reduce_key='dff', filter_keys=['mouse','day','odor_valence'])
    sig_res['dff'] = dff_res['dff']

    reversal_res = defaultdict(list)
    day_strs = ['Lrn','Rev']
    for mouse in mice:
        mouse_res = filter.filter(sig_res, filter_dict={'mouse':mouse})
        days = np.unique(mouse_res['day'])
        p_list = []
        m_list = []
        for i, day in enumerate(days):
            mouse_day_res = filter.filter(mouse_res, filter_dict={'day':day})
            p, m = _helper(mouse_day_res)
            reversal_res['mouse'].append(mouse)
            reversal_res['mouse'].append(mouse)
            reversal_res['day'].append(day_strs[i])
            reversal_res['day'].append(day_strs[i])
            reversal_res['odor_valence'].append('CS+')
            reversal_res['odor_valence'].append('CS-')
            reversal_res[key].append(p)
            reversal_res[key].append(m)
            reversal_res['Fraction'].append(np.mean(p))
            reversal_res['Fraction'].append(np.mean(m))
            p_list.append(p)
            m_list.append(m)
    for k, val in reversal_res.items():
        reversal_res[k] = np.array(val)

    stats_res = defaultdict(list)
    for mouse in mice:
        mouse_res = filter.filter(reversal_res, filter_dict={'mouse':mouse})
        combinations, list_of_ixs = filter.retrieve_unique_entries(mouse_res, ['day','odor_valence'])

        assert len(combinations) == 4, 'not equal to 4'
        assert combinations[0][-1] == 'CS+'
        assert combinations[1][-1] == 'CS-'
        assert combinations[2][-1] == 'CS+'
        assert combinations[3][-1] == 'CS-'
        assert combinations[0][0] == day_strs[0]
        assert combinations[1][0] == day_strs[0]
        assert combinations[2][0] == day_strs[1]
        assert combinations[3][0] == day_strs[1]

        p_before = mouse_res[key][0]
        m_before = mouse_res[key][1]
        n_before = np.invert([a or b for a, b in zip(p_before, m_before)])
        p_after = mouse_res[key][2]
        m_after = mouse_res[key][3]
        n_after = np.invert([a or b for a, b in zip(p_after, m_after)])

        list_before = [p_before, m_before, n_before]
        list_after = [p_after, m_after, n_after]
        str = ['p', 'm', 'none']
        for i, before in enumerate(list_before):
            for j, after in enumerate(list_after):
                ix_intersect =np.intersect1d(np.where(before)[0], np.where(after)[0])
                fraction = len(ix_intersect) / np.sum(before)
                stats_res['mouse'].append(mouse)
                stats_res['condition'].append(str[i]  + '-' + str[j])
                stats_res['Fraction'].append(fraction)
    for key, val in stats_res.items():
        stats_res[key] = np.array(val)
    return reversal_res, stats_res