from collections import defaultdict
import numpy as np
from matplotlib import pyplot as plt
import filter
import plot
from plot import _easy_save
from format import *
import copy
import reduce
from analysis import add_naive_learned

def plot_compare_responsive(res, figure_path):
    ax_args_copy = ax_args.copy()
    ax_args_copy.update({'ylim':[0, .65], 'yticks':[0, .2, .4, .6], 'xticks':list(range(20))})
    res = copy.copy(res)
    res = filter.filter(res, {'odor_valence':['CS+','CS-']})
    res_ = get_compare_responsive_sig(res)

    line_args_copy = line_args.copy()
    line_args_copy.update({'marker':'.', 'linestyle':'--', 'linewidth':.5, 'alpha': .75})

    plot.plot_results(res_,
                      x_key='day', y_key='Fraction', loop_keys=['mouse','odor_valence'],
                      colors=['green','red']*10,
                      path=figure_path, plot_args=line_args_copy, ax_args=ax_args_copy,
                      fig_size=(2, 1.5), legend=False)


def get_compare_responsive_sig(res):
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

    new_res = defaultdict(list)
    for mouse in mice:
        mouse_res = filter.filter(sig_res, filter_dict={'mouse':mouse})
        days = np.unique(mouse_res['day'])
        p_list = []
        m_list = []
        for i, day in enumerate(days):
            mouse_day_res = filter.filter(mouse_res, filter_dict={'day':day})
            p, m = _helper(mouse_day_res)
            new_res['mouse'].append(mouse)
            new_res['mouse'].append(mouse)
            new_res['day'].append(day)
            new_res['day'].append(day)
            new_res['odor_valence'].append('CS+')
            new_res['odor_valence'].append('CS-')
            new_res[key].append(p)
            new_res[key].append(m)
            new_res['Fraction'].append(np.mean(p))
            new_res['Fraction'].append(np.mean(m))
            p_list.append(p)
            m_list.append(m)
    for key, val in new_res.items():
        new_res[key] = np.array(val)
    return new_res

def plot_responsive_difference_odor_and_water(res, odor_start_days, end_days, use_colors= True, figure_path = None,
                                              normalize = False, ylim = .6):
    key = 'Change in Fraction'
    if normalize:
        key = 'Norm. Fraction'

    def _helper(start_end_day_res):
        combs, list_of_ixs = filter.retrieve_unique_entries(start_end_day_res, ['mouse','odor_valence'])
        for i, comb in enumerate(combs):
            ixs = list_of_ixs[i]
            assert len(ixs) == 2

            if start_end_day_res['training_day'][0] == 'Naive':
                ref = ixs[0]
                test = ixs[1]
            elif start_end_day_res['training_day'][0] == 'Learned':
                ref = ixs[1]
                test = ixs[0]
            else:
                raise ValueError('cannot find ref day')

            if normalize:
                start_end_day_res[key][test] = start_end_day_res['Fraction'][test] / \
                                               start_end_day_res['Fraction'][ref]
                start_end_day_res[key][ref] = 1
            else:

                start_end_day_res[key][test] = start_end_day_res['Fraction'][test] - \
                                               start_end_day_res['Fraction'][ref]
                start_end_day_res[key][ref] = 0

    ax_args_copy = ax_args.copy()
    res = get_compare_responsive_sig(res)
    list_of_days = list(zip(odor_start_days, end_days))
    mice = np.unique(res['mouse'])
    res[key] = np.zeros_like(res['Fraction'])
    start_end_day_res = filter.filter_days_per_mouse(res, days_per_mouse= list_of_days)
    start_end_day_res = filter.filter(start_end_day_res, {'odor_valence': ['CS+','CS-']})
    add_naive_learned(start_end_day_res, odor_start_days, end_days)
    _helper(start_end_day_res)
    start_end_day_res = filter.filter(start_end_day_res, {'training_day':'Learned'})
    summary_res = reduce.new_filter_reduce(start_end_day_res, filter_keys='odor_valence', reduce_key= key)

    dict = {'CS+':'Green', 'CS-':'Red'}
    if use_colors:
        colors = [dict[key] for key in np.unique(start_end_day_res['odor_valence'])]
    else:
        colors = ['Black'] * 6
    ax_args_copy = ax_args_copy.copy()
    n_valence = len(np.unique(summary_res['odor_valence']))
    ax_args_copy.update({'xlim':[-1, n_valence], 'ylim':[-ylim, ylim], 'yticks': np.arange(-1, 1, .2)})

    if normalize:
        ax_args_copy.update({'xlim': [-1, n_valence], 'ylim': [-.1, 1.5], 'yticks':[0, .5, 1, 1.5]})
    scatter_args_copy = scatter_args.copy()
    scatter_args_copy.update({'s':8})

    odors = ['CS+', 'CS-']
    for i, odor in enumerate(odors):
        reuse = True
        if i == 0:
            reuse=False
        plot.plot_results(start_end_day_res, loop_keys='odor_valence', select_dict={'odor_valence':odor},
                          x_key='odor_valence', y_key=key,
                          colors= [dict[odor]] * len(mice),
                          path =figure_path, plot_args=scatter_args_copy, plot_function=plt.scatter, ax_args= ax_args_copy,
                          save= False, reuse=reuse,
                          fig_size=(2, 1.5), legend=False, name_str = ','.join([str(x) for x in odor_start_days]))

    if not normalize:
        plt.plot(plt.xlim(), [0, 0], '--', color = 'gray', linewidth = 1, alpha = .5)

    plot.plot_results(summary_res,
                      x_key='odor_valence', y_key=key, error_key = key + '_sem',
                      colors= 'black',
                      path =figure_path, plot_args=error_args, plot_function=plt.errorbar, ax_args= ax_args_copy,
                      save= True, reuse=True,
                      fig_size=(2, 1.5), legend=False)