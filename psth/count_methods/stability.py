import copy
from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt

import filter
import plot
import reduce
from format import *

def plot_stability_across_days(res, start_days_per_mouse, learned_days_per_mouse, figure_path):
    strengthen_starting_threshold = .1
    strengthen_absolute_threshold = .2
    strengthen_relative_threshold = 2
    weaken_starting_threshold = .3
    weaken_absolute_threshold = .15
    weaken_relative_threshold = .5

    res = copy.copy(res)
    res = filter.filter(res, {'odor_valence':['CS+', 'CS-']})
    list_odor_on = res['DAQ_O_ON_F']
    list_water_on = res['DAQ_W_ON_F']
    for i in range(len(list_odor_on)):
        dff_list = res['dff'][i]
        dff_max = np.max(dff_list[:,list_odor_on[i]: list_water_on[i]],axis=1)
        res['dff_max'].append(dff_max)
    res['dff_max'] = np.array(res['dff_max'])

    out = defaultdict(list)
    combinations, ixs = filter.retrieve_unique_entries(res, ['mouse','odor'])
    for combination, ix in zip(combinations, ixs):
        odor_valence = np.unique(res['odor_valence'][ix])
        mouse = np.unique(res['mouse'][ix])
        days = res['day'][ix]
        assert len(days) == len(np.unique(days))
        assert len(mouse) == 1
        sort_ix = np.argsort(days)
        list_of_dff_max = res['dff_max'][ix][sort_ix]
        list_of_ssig = res['sig'][ix][sort_ix]
        data = [x * y for x, y in zip(list_of_ssig, list_of_dff_max)]
        data = np.array(data)

        strengthen_mask = data[0,:] > strengthen_starting_threshold
        strengthen_overall_threshold = np.max((data[0,:] * strengthen_relative_threshold, data[0,:] + strengthen_absolute_threshold), axis=0)
        strengthen_passed_mask = strengthen_overall_threshold < data
        strengthen_any = np.any(strengthen_passed_mask[1:,:], axis=0)
        n_strengthen_passed = np.sum(strengthen_any * strengthen_mask)
        n_strengthen_denom = np.sum(strengthen_mask)

        new_mask = np.invert(strengthen_mask)
        n_new_passed = np.sum(strengthen_any * new_mask)
        n_new_denom = np.sum(new_mask)

        weaken_mask = data[0,:] > weaken_starting_threshold
        weaken_overall_threshold = np.min((data[0,:] * weaken_relative_threshold, data[0,:] - weaken_absolute_threshold), axis=0)
        weaken_passed_mask = weaken_overall_threshold > data
        weaken_any = np.any(weaken_passed_mask[1:,:], axis=0)
        n_weaken_passed = np.sum(weaken_any * weaken_mask)
        n_weaken_denom = np.sum(weaken_mask)

        strs = ['strengthen_denom', 'strengthen_pass', 'new_denom', 'new_pass','weaken_denom','weaken_pass']
        vals = [n_strengthen_denom, n_strengthen_passed, n_new_denom, n_new_passed, n_weaken_denom, n_weaken_passed]

        for k, v in zip(strs,vals):
            out['Type'].append(k)
            out['Count'].append(v)
            out['mouse'].append(mouse[0])
            out['odor_valence'].append(odor_valence[0])
    for k, v in out.items():
        out[k] = np.array(v)

    def _helper(res):
        out = defaultdict(list)
        strs = [['strengthen_denom', 'strengthen_pass'], ['new_denom', 'new_pass'], ['weaken_denom', 'weaken_pass']]
        out_strs = ['up', 'new', 'down']
        for i, keys in enumerate(strs):
            ix_denom = res['Type'] == keys[0]
            ix_numer = res['Type'] == keys[1]
            numer = np.sum(res['Count'][ix_numer])
            denom = np.sum(res['Count'][ix_denom])
            fraction =  numer/ denom
            out['Type'].append(out_strs[i])
            out['Fraction'].append(fraction)
            out['numer'].append(numer)
            out['denom'].append(denom)
        for k, v in out.items():
            out[k] = np.array(v)
        return out

    ax_args_copy = ax_args.copy()
    ax_args_copy.update({'ylim':[0, 0.5], 'yticks':[0, .1, .2, .3, .4, .5]})
    overall = copy.copy(out)
    csm_res = filter.filter(overall, {'odor_valence':'CS-'})
    csp_res = filter.filter(overall, {'odor_valence':'CS+'})

    csp_stats = _helper(csp_res)
    csp_stats['Condition'] = np.array(['CS+'] * len(csp_stats['Type']))
    csm_stats = _helper(csm_res)
    csm_stats['Condition'] = np.array(['CS-'] * len(csm_stats['Type']))
    overall_stats = reduce.chain_defaultdicts(csp_stats, csm_stats, copy_dict=True)
    filter.assign_composite(overall_stats, ['Type', 'Condition'])

    bar_args = {'alpha': 1, 'edgecolor':'black','linewidth':1}
    colors = ['Green','Red']
    save_path, name = plot.plot_results(overall_stats, x_key='Type_Condition', y_key='Fraction', sort=True,
                                        colors=colors,
                                        path=figure_path,
                                        plot_function=plt.bar, plot_args=bar_args, ax_args=ax_args_copy,
                                        save=False, reuse=False)

    strs = ['down','new','up']
    fontsize = 4
    for i in range(3):
        ix = csp_stats['Type'] == strs[i]
        numer = csp_stats['numer'][ix][0]
        denom = csp_stats['denom'][ix][0]

        x = i * 2
        y = numer/denom
        y_offset = .025
        plt.text(x, y + y_offset, str(numer) + '/' + str(denom), horizontalalignment = 'center', fontsize= fontsize)

    strs = ['down','new','up']
    for i in range(3):
        ix = csm_stats['Type'] == strs[i]
        numer = csm_stats['numer'][ix][0]
        denom = csm_stats['denom'][ix][0]

        x = i * 2 + 1
        y = numer/denom
        y_offset = .025
        plt.text(x, y + y_offset, str(numer) + '/' + str(denom), horizontalalignment = 'center', fontsize= fontsize)
    plot._easy_save(save_path, name, pdf=True)

    overall_stats_no_distinction = _helper(overall)
    save_path, name = plot.plot_results(overall_stats_no_distinction, x_key='Type', y_key='Fraction', sort=True,
                      colors=['Grey']*10,
                      path=figure_path,
                      plot_function=plt.bar, plot_args=bar_args, ax_args=ax_args_copy,
                      save=False, reuse=False)
    strs = ['down','new','up']
    for i in range(3):
        ix = overall_stats_no_distinction['Type'] == strs[i]
        numer = overall_stats_no_distinction['numer'][ix][0]
        denom = overall_stats_no_distinction['denom'][ix][0]

        x = i
        y = numer/denom
        y_offset = .025
        plt.text(x, y + y_offset, str(numer) + '/' + str(denom), horizontalalignment = 'center', fontsize= fontsize)
    plot._easy_save(save_path, name, pdf=True)