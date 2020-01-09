import copy
import itertools
from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt

import filter
import plot
import reduce
from analysis import add_naive_learned
from format import *

def plot_overlap_odor(res, start_days, end_days, delete_non_selective = False, figure_path = None, excitatory=True):
    ax_args_copy = overlap_ax_args.copy()
    res = copy.copy(res)
    res = _get_overlap_odor(res, delete_non_selective)
    list_of_days = list(zip(start_days, end_days))
    mice = np.unique(res['mouse'])
    start_end_day_res = filter.filter_days_per_mouse(res, days_per_mouse= list_of_days)
    start_end_day_res = reduce.new_filter_reduce(start_end_day_res, filter_keys=['mouse', 'day', 'condition'],
                                                 reduce_key='Overlap')
    add_naive_learned(start_end_day_res, start_days, end_days)

    filter.assign_composite(start_end_day_res, loop_keys=['condition', 'training_day'])
    odor_list = ['+:+', '-:-','+:-']
    colors = ['Green','Red','Gray']
    name_str = '_E' if excitatory else '_I'
    ax_args_copy.update({'xlim': [-1, 6]})
    for i, odor in enumerate(odor_list):
        save_arg = False
        reuse_arg = True
        if i == 0:
            reuse_arg = False
        if i == len(odor_list) -1:
            save_arg = True

        temp = filter.filter(start_end_day_res, {'condition':odor})
        name = ','.join([str(x) for x in start_days]) + '_' + ','.join([str(x) for x in end_days])
        name += name_str
        plot.plot_results(temp,
                          x_key='condition_training_day', y_key='Overlap', loop_keys='mouse',
                          colors= [colors[i]]*len(mice),
                          path =figure_path, plot_args=line_args, ax_args= ax_args_copy,
                          save=save_arg, reuse=reuse_arg, name_str=name,
                          fig_size=(2, 1.5), legend=False)

        b = filter.filter(temp, {'training_day':'Learned'})
        print(odor)
        print(np.mean(b['Overlap']))

    start_end_day_res = filter.filter_days_per_mouse(res, days_per_mouse=list_of_days)
    add_naive_learned(start_end_day_res, start_days, end_days, str1='0', str2='1')
    start_end_day_res.pop('Overlap_sem',None)
    summary_res = reduce.new_filter_reduce(start_end_day_res, filter_keys='training_day', reduce_key='Overlap')

    ax_args_copy.update({'xlim': [-1, 2], 'ylim':[0, .5], 'yticks':[0, .1, .2, .3, .4, .5]})
    plot.plot_results(summary_res, x_key= 'training_day', y_key='Overlap',
                      path= figure_path, plot_args= bar_args, ax_args=ax_args_copy, plot_function=plt.bar,
                      fig_size=(2, 1.5), legend=False,
                      reuse=False, save=False)
    plot.plot_results(summary_res, x_key='training_day', y_key='Overlap', error_key='Overlap_sem',
                      path=figure_path,
                      plot_function= plt.errorbar, plot_args= error_args, ax_args = ax_args, save=True, reuse=True,
                      name_str=name_str)

    before_odor = filter.filter(start_end_day_res, filter_dict={'training_day':'0', 'condition':'+:+'})
    after_odor = filter.filter(start_end_day_res, filter_dict={'training_day':'1', 'condition':'+:+'})
    before_csp = filter.filter(start_end_day_res, filter_dict={'training_day':'0', 'condition':'+:-'})
    after_csp = filter.filter(start_end_day_res, filter_dict={'training_day':'1', 'condition':'+:-'})
    before_csm = filter.filter(start_end_day_res, filter_dict={'training_day':'0', 'condition':'-:-'})
    after_csm = filter.filter(start_end_day_res, filter_dict={'training_day':'1', 'condition':'-:-'})

    from scipy.stats import ranksums, wilcoxon, kruskal

    print('Before ++: {}'.format(np.mean(before_odor['Overlap'])))
    print('After ++: {}'.format(np.mean(after_odor['Overlap'])))
    print('Wilcoxin:{}'.format(wilcoxon(before_odor['Overlap'], after_odor['Overlap'])))

    print('Before +-: {}'.format(np.mean(before_csp['Overlap'])))
    print('After +-: {}'.format(np.mean(after_csp['Overlap'])))
    print('Wilcoxin:{}'.format(wilcoxon(before_csp['Overlap'], after_csp['Overlap'])))

    print('Before --: {}'.format(np.mean(before_csm['Overlap'])))
    print('After --: {}'.format(np.mean(after_csm['Overlap'])))
    print('Wilcoxin:{}'.format(wilcoxon(before_csm['Overlap'], after_csm['Overlap'])))


def plot_overlap_water(res, start_days, end_days, figure_path):
    ax_args_copy = overlap_ax_args.copy()
    res = copy.copy(res)
    mice = np.unique(res['mouse'])
    res = filter.filter_days_per_mouse(res, days_per_mouse= end_days)
    add_naive_learned(res, start_days, end_days)
    ax_args_copy.update({'xlim': [-1, 2]})
    y_keys = ['US/CS+', 'CS+/US']
    summary_res = defaultdict(list)
    for arg in y_keys:
        _get_overlap_water(res, arg = arg)
        new_res = reduce.new_filter_reduce(res, filter_keys=['mouse','day','odor_valence'],
                                             reduce_key='Overlap')
        new_res['Type'] = np.array([arg] * len(new_res['training_day']))
        reduce.chain_defaultdicts(summary_res, new_res)

    summary_res.pop('Overlap_sem')
    summary_res.pop('Overlap_std')
    summary_res = filter.filter(summary_res, {'odor_valence':'CS+'})
    mean_std_res = reduce.new_filter_reduce(summary_res, filter_keys='Type', reduce_key='Overlap')
    types = np.unique(summary_res['Type'])
    scatter_args_copy = scatter_args.copy()
    scatter_args_copy.update({'s':2,'alpha':.6})
    for i, type in enumerate(types):
        reuse_arg = True
        if i == 0:
            reuse_arg = False
        temp = filter.filter(summary_res, {'Type':type})
        plot.plot_results(temp,
                          x_key='Type', y_key='Overlap', loop_keys='mouse',
                          colors=['Black'] * len(mice),
                          plot_function= plt.scatter,
                          path=figure_path, plot_args=scatter_args_copy, ax_args=ax_args_copy,
                          save=False, reuse=reuse_arg,
                          fig_size=(1.5, 1.5), rect = (.25, .25, .6, .6), legend = False)

    plot.plot_results(mean_std_res,
                      x_key='Type', y_key='Overlap', error_key='Overlap_sem',
                      path=figure_path, plot_function=plt.errorbar, plot_args=error_args, ax_args=ax_args,
                      save=True, reuse=True,
                      fig_size=(1.5, 1.5), legend=False)
    print(mean_std_res['Overlap'])


def _overlap(ix1, ix2, arg = 'max'):
    if arg == 'max':
        size = np.max((ix1.size, ix2.size))
    elif arg == 'over':
        size = ix2.size
    else:
        raise ValueError('overlap arg not recognized')
    intersect = float(len(np.intersect1d(ix1, ix2)))
    return intersect / size


def _get_overlap_water(res, arg):
    def _helper(list_of_name_ix_tuple, desired_tuple):
        for tuple in list_of_name_ix_tuple:
            if tuple[0] == desired_tuple:
                ix = tuple[1]
                assert len(ix) == 1, 'more than 1 unique entry'
                return ix[0]

    res['Overlap'] = np.zeros(res['day'].shape)
    names, ixs = filter.retrieve_unique_entries(res, ['mouse', 'day', 'odor_standard'])
    list_of_name_ix_tuples = list(zip(names, ixs))

    mice =np.unique(res['mouse'])
    for mouse in mice:
        mouse_res = filter.filter(res, filter_dict={'mouse':mouse})
        days = np.unique(mouse_res['day'])
        for day in days:
            mouse_day_res = filter.filter(mouse_res, filter_dict={'day':day})
            odors = np.unique(mouse_day_res['odor_standard'])
            if 'US' in odors:
                us_ix = _helper(list_of_name_ix_tuples, (mouse, day, 'US'))
                us_cells = np.where(res['sig'][us_ix])[0]
                for odor in odors:
                    odor_ix = _helper(list_of_name_ix_tuples, (mouse, day, odor))
                    odor_cells = np.where(res['sig'][odor_ix])[0]
                    if arg == 'US/CS+':
                        overlap = _overlap(us_cells, odor_cells, arg='over')
                    elif arg == 'CS+/US':
                        overlap = _overlap(odor_cells, us_cells, arg='over')
                    else:
                        raise ValueError('overlap arg not recognized')
                    res['Overlap'][odor_ix] = overlap


def _get_overlap_odor(res, delete_non_selective):
    def _subsets(S, m):
        return set(itertools.combinations(S, m))
    new = defaultdict(list)
    mice =np.unique(res['mouse'])
    for mouse in mice:
        mouse_res = filter.filter(res, filter_dict={'mouse':mouse})
        days = np.unique(mouse_res['day'])
        for day in days:
            mouse_day_res = filter.filter(mouse_res,
                                          filter_dict={'day':day, 'odor_valence':['CS+','CS-']})

            odors, odor_ix = np.unique(mouse_day_res['odor_standard'], return_index= True)
            assert len(odor_ix) == 4, 'Number of odors does not equal 4'
            all_comparisons = _subsets(odor_ix, 2)
            for comparison in all_comparisons:
                mask1 = mouse_day_res['sig'][comparison[0]]
                mask2 = mouse_day_res['sig'][comparison[1]]

                if delete_non_selective:
                    non_selective_mask = _respond_to_all(mouse_day_res['sig'])
                    mask1 = np.all([mask1, np.invert(non_selective_mask)], axis=0).astype(int)
                    mask2 = np.all([mask2, np.invert(non_selective_mask)], axis=0).astype(int)

                overlap = _overlap(np.where(mask1)[0], np.where(mask2)[0])
                new['Overlap'].append(overlap)
                new['mouse'].append(mouse)
                new['day'].append(day)
                if comparison == (0,1):
                    new['condition'].append('+:+')
                elif comparison == (2,3):
                    new['condition'].append('-:-')
                else:
                    new['condition'].append('+:-')
    for key, val in new.items():
        new[key] = np.array(val)
    return new


def _respond_to_all(list_of_masks):
    arr = np.stack(list_of_masks, axis=1)
    non_selective_mask = np.all(arr, axis=1)
    return non_selective_mask