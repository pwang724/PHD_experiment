import copy
import numpy as np
import filter
import plot
import reduce
from analysis import add_naive_learned
from format import *
import matplotlib.pyplot as plt
from collections import defaultdict

def plot_summary_odor_and_water(res, odor_start_days, water_start_days, end_days, use_colors= True, excitatory = True,
                                arg = 'odor_valence', figure_path = None):
    include_water = True

    ax_args_copy = ax_args.copy()
    res = copy.copy(res)
    get_responsive_cells(res)
    mice = np.unique(res['mouse'])

    list_of_days = list(zip(odor_start_days, end_days))
    start_end_day_res = filter.filter_days_per_mouse(res, days_per_mouse= list_of_days)
    start_end_day_res = filter.exclude(start_end_day_res, {'odor_valence':'US'})
    add_naive_learned(start_end_day_res, odor_start_days, end_days, 'a','b')

    if include_water:
        list_of_days = list(zip(water_start_days, end_days))
        start_end_day_res_water = filter.filter_days_per_mouse(res, days_per_mouse= list_of_days)
        start_end_day_res_water = filter.filter(start_end_day_res_water, {'odor_valence':'US'})
        add_naive_learned(start_end_day_res_water, water_start_days, end_days, 'a', 'b')
        reduce.chain_defaultdicts(start_end_day_res, start_end_day_res_water)

    ax_args_copy = ax_args_copy.copy()
    if arg == 'odor_valence':
        start_end_day_res = reduce.new_filter_reduce(start_end_day_res, filter_keys=['training_day','mouse','odor_valence'],
                                                 reduce_key='Fraction Responsive')
        odor_list = ['CS+', 'CS-']
        ax_args_copy.update({'xlim': [-1, 6], 'ylim': [0, .6], 'yticks': [0, .1, .2, .3, .4, .5]})
        colors = ['Green', 'Red']
    elif arg == 'naive':
        arg = 'odor_valence'
        start_end_day_res = reduce.new_filter_reduce(start_end_day_res,
                                                     filter_keys=['training_day', 'mouse', 'odor_valence'],
                                                     reduce_key='Fraction Responsive')
        odor_list = ['CS+']
        ax_args_copy.update({'xlim': [-1, 4], 'ylim': [0, .6], 'yticks': [0, .1, .2, .3, .4, .5]})
        colors = ['GoldenRod']
    else:
        odor_list = ['CS+1', 'CS+2','CS-1', 'CS-2']
        colors = ['Green', 'Green', 'Red', 'Red']
        ax_args_copy.update({'xlim':[-1, 10], 'ylim':[0, .6], 'yticks':[0, .1, .2, .3, .4, .5]})

    filter.assign_composite(start_end_day_res, loop_keys=[arg, 'training_day'])
    if not use_colors:
        colors = ['Black'] * 4

    name_str = '_E' if excitatory else '_I'
    for i, odor in enumerate(odor_list):
        reuse_arg = True
        if i == 0:
            reuse_arg = False
        plot.plot_results(start_end_day_res,
                          select_dict= {arg:odor},
                          x_key=arg + '_training_day', y_key='Fraction Responsive', loop_keys='mouse',
                          colors= [colors[i]]*len(mice),
                          path =figure_path, plot_args=line_args, ax_args= ax_args_copy,
                          save= False, reuse=reuse_arg,
                          fig_size=(2.5, 1.5), legend=False, name_str = ','.join([str(x) for x in odor_start_days]))


    plot.plot_results(start_end_day_res, select_dict={'odor_standard': 'US'},
                      x_key='training_day', y_key='Fraction Responsive', loop_keys='mouse',
                      colors= ['Turquoise']*len(mice),
                      path =figure_path, plot_args=line_args, ax_args= ax_args_copy,
                      fig_size=(1.6, 1.5), legend=False,
                      reuse=True, save=True, name_str=name_str)

    before_odor = filter.filter(start_end_day_res, filter_dict={'training_day':'a', 'odor_valence':['CS+','CS-']})
    after_odor = filter.filter(start_end_day_res, filter_dict={'training_day':'b', 'odor_valence':['CS+','CS-']})
    before_csp = filter.filter(start_end_day_res, filter_dict={'training_day':'a', 'odor_valence':'CS+'})
    after_csp = filter.filter(start_end_day_res, filter_dict={'training_day':'b', 'odor_valence':'CS+'})
    before_csm = filter.filter(start_end_day_res, filter_dict={'training_day':'a', 'odor_valence':'CS-'})
    after_csm = filter.filter(start_end_day_res, filter_dict={'training_day':'b', 'odor_valence':'CS-'})
    before_water = filter.filter(start_end_day_res, filter_dict={'training_day':'a', 'odor_valence':'US'})
    after_water = filter.filter(start_end_day_res, filter_dict={'training_day':'b', 'odor_valence':'US'})

    try:
        from scipy.stats import ranksums, wilcoxon, kruskal

        print('Before Odor: {}'.format(np.mean(before_odor['Fraction Responsive'])))
        print('After Odor: {}'.format(np.mean(after_odor['Fraction Responsive'])))
        print('Wilcoxin:{}'.format(wilcoxon(before_odor['Fraction Responsive'], after_odor['Fraction Responsive'])))

        print('Before CS+: {}'.format(np.mean(before_csp['Fraction Responsive'])))
        print('After CS+: {}'.format(np.mean(after_csp['Fraction Responsive'])))
        print('Wilcoxin:{}'.format(wilcoxon(before_csp['Fraction Responsive'], after_csp['Fraction Responsive'])))

        print('Before CS-: {}'.format(np.mean(before_csm['Fraction Responsive'])))
        print('After CS-: {}'.format(np.mean(after_csm['Fraction Responsive'])))
        print('Wilcoxin:{}'.format(wilcoxon(before_csm['Fraction Responsive'], after_csm['Fraction Responsive'])))

        print('Before US: {}'.format(np.mean(before_water['Fraction Responsive'])))
        print('After US: {}'.format(np.mean(after_water['Fraction Responsive'])))
        print('Wilcoxin:{}'.format(wilcoxon(before_water['Fraction Responsive'], after_water['Fraction Responsive'])))
    except:
        print('stats didnt work')

def plot_responsive_difference_odor_and_water(res, odor_start_days, water_start_days, end_days, use_colors= True, figure_path = None,
                                              include_water = True, normalize = False, pt_start = None, pt_learned = None,
                                              average = True, ylim = .22,
                                              reuse_arg = False, save_arg = True):
    key = 'Change in Fraction Responsive'
    if normalize:
        key = 'Norm. Fraction Responsive'

    def _helper(start_end_day_res):
        combs, list_of_ixs = filter.retrieve_unique_entries(start_end_day_res, ['mouse','odor_standard'])
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
                start_end_day_res[key][test] = start_end_day_res['Fraction Responsive'][test] / \
                                               start_end_day_res['Fraction Responsive'][ref]
                start_end_day_res[key][ref] = 1
            else:

                start_end_day_res[key][test] = start_end_day_res['Fraction Responsive'][test] - \
                                               start_end_day_res['Fraction Responsive'][ref]
                start_end_day_res[key][ref] = 0

    ax_args_copy = ax_args.copy()
    res = copy.copy(res)
    get_responsive_cells(res)
    list_of_days = list(zip(odor_start_days, end_days))
    mice = np.unique(res['mouse'])
    res[key] = np.zeros_like(res['Fraction Responsive'])
    start_end_day_res = filter.filter_days_per_mouse(res, days_per_mouse= list_of_days)
    start_end_day_res = filter.filter(start_end_day_res, {'odor_valence': ['CS+','CS-','Naive']})
    add_naive_learned(start_end_day_res, odor_start_days, end_days)

    odors = ['CS+', 'CS-', 'Naive']
    if 'PT CS+' in np.unique(res['odor_valence']):
        odors = ['PT CS+'] + odors
        list_of_days = list(zip(pt_start, pt_learned))
        start_end_day_res_pt = filter.filter_days_per_mouse(res, days_per_mouse= list_of_days)
        start_end_day_res_pt = filter.filter(start_end_day_res_pt, {'odor_valence':'PT CS+'})
        add_naive_learned(start_end_day_res_pt, pt_start, pt_learned)
        reduce.chain_defaultdicts(start_end_day_res, start_end_day_res_pt)

    if include_water:
        odors += ['US']
        list_of_days = list(zip(water_start_days, end_days))
        start_end_day_res_water = filter.filter_days_per_mouse(res, days_per_mouse= list_of_days)
        start_end_day_res_water = filter.filter(start_end_day_res_water, {'odor_valence':'US'})
        add_naive_learned(start_end_day_res_water, water_start_days, end_days)
        reduce.chain_defaultdicts(start_end_day_res, start_end_day_res_water)

    filter.assign_composite(start_end_day_res, loop_keys=['odor_standard', 'training_day'])
    if average:
        start_end_day_res = reduce.new_filter_reduce(start_end_day_res,
                                                     filter_keys=['odor_valence', 'mouse', 'training_day'], reduce_key=key)
        start_end_day_res.pop(key + '_sem')
    _helper(start_end_day_res)
    start_end_day_res = filter.filter(start_end_day_res, {'training_day':'Learned'})
    summary_res = reduce.new_filter_reduce(start_end_day_res, filter_keys='odor_valence', reduce_key= key)

    dict = {'CS+':'Green', 'CS-':'Red','US':'Turquoise', 'PT CS+':'Orange', 'Naive':'Gray'}
    if use_colors:
        colors = [dict[key] for key in np.unique(start_end_day_res['odor_valence'])]
    else:
        colors = ['Black'] * 6

    ax_args_copy = ax_args_copy.copy()
    n_valence = len(np.unique(summary_res['odor_valence']))
    ax_args_copy.update({'xlim':[-.5, 3.5], 'ylim':[-ylim, ylim], 'yticks':[-.3, -.2, -.1, 0, .1, .2, .3]})
    if normalize:
        ax_args_copy.update({'xlim': [-.5, 3.5], 'ylim': [-.1, 1.5], 'yticks':[0, .5, 1, 1.5]})
    error_args_ = {'fmt': '.', 'capsize': 2, 'elinewidth': 1, 'markersize': 2, 'alpha': .75}
    scatter_args_copy = scatter_args.copy()
    scatter_args_copy.update({'s':3})

    for i, odor in enumerate(odors):
        reuse = True
        if i == 0:
            reuse= reuse_arg
        plot.plot_results(start_end_day_res, loop_keys='odor_valence', select_dict={'odor_valence':odor},
                          x_key='odor_valence', y_key=key,
                          colors= [dict[odor]] * len(mice),
                          path =figure_path, plot_args=scatter_args_copy, plot_function=plt.scatter, ax_args= ax_args_copy,
                          save= False, reuse=reuse,
                          fig_size=(2, 1.5), rect = (.25, .2, .6, .6), legend=False, name_str = ','.join([str(x) for x in odor_start_days]))

    if not normalize:
        plt.plot(plt.xlim(), [0, 0], '--', color = 'gray', linewidth = 1, alpha = .5)

    plot.plot_results(summary_res,
                      x_key='odor_valence', y_key=key, error_key = key + '_sem',
                      colors= 'black',
                      path =figure_path, plot_args=error_args_, plot_function=plt.errorbar, ax_args= ax_args_copy,
                      save= save_arg, reuse=True,
                      fig_size=(2, 1.5), legend=False)


def plot_summary_odor(res, start_days, end_days, use_colors= True, figure_path = None, reuse = False, excitatory=True):
    ax_args_copy = ax_args.copy()
    res = copy.copy(res)
    get_responsive_cells(res)
    list_of_days = list(zip(start_days, end_days))
    mice = np.unique(res['mouse'])
    start_end_day_res = filter.filter_days_per_mouse(res, days_per_mouse= list_of_days)
    add_naive_learned(start_end_day_res, start_days, end_days,'a','b')
    filter.assign_composite(start_end_day_res, loop_keys=['odor_valence', 'training_day'])
    start_end_day_res = reduce.new_filter_reduce(start_end_day_res,
                                                 filter_keys=['training_day', 'mouse', 'odor_valence'],
                                                 reduce_key='Fraction Responsive')

    odor_list = ['CS+','CS-']
    if use_colors:
        colors = ['Green','Red']
    else:
        colors = ['Black'] * 2
    ax_args_copy = ax_args_copy.copy()
    ax_args_copy.update({'xlim':[-1, 8], 'ylim':[0, .4], 'yticks':[0, .1, .2, .3, .4]})
    name_str = '_E' if excitatory else '_I'
    for i, odor in enumerate(odor_list):
        save_arg = False
        reuse_arg = True
        if i == 0 and not reuse:
            reuse_arg = False
        if i == len(odor_list) -1:
            save_arg = True

        plot.plot_results(start_end_day_res,
                          select_dict= {'odor_valence':odor},
                          x_key='odor_valence_training_day', y_key='Fraction Responsive', loop_keys='mouse',
                          colors= [colors[i]]*len(mice),
                          path =figure_path, plot_args=line_args, ax_args= ax_args_copy,
                          save=save_arg, reuse=reuse_arg,
                          fig_size=(2.5, 1.5),legend=False,
                          name_str = ','.join([str(x) for x in start_days]) + name_str)

    start_end_day_res = filter.filter_days_per_mouse(res, days_per_mouse=list_of_days)
    add_naive_learned(start_end_day_res, start_days, end_days, 'a', 'b')
    filter.assign_composite(start_end_day_res, loop_keys=['odor_valence', 'training_day'])
    start_end_day_res = reduce.new_filter_reduce(start_end_day_res,
                                                 filter_keys=['training_day', 'mouse', 'odor_standard'],
                                                 reduce_key='Fraction Responsive')

    before_csp = filter.filter(start_end_day_res, filter_dict={'training_day':'a', 'odor_valence':'CS+'})
    after_csp = filter.filter(start_end_day_res, filter_dict={'training_day':'b', 'odor_valence':'CS+'})
    before_csm = filter.filter(start_end_day_res, filter_dict={'training_day':'a', 'odor_valence':'CS-'})
    after_csm = filter.filter(start_end_day_res, filter_dict={'training_day':'b', 'odor_valence':'CS-'})

    try:
        from scipy.stats import ranksums, wilcoxon, kruskal
        print('Before CS+: {}'.format(np.mean(before_csp['Fraction Responsive'])))
        print('After CS+: {}'.format(np.mean(after_csp['Fraction Responsive'])))
        print('Wilcoxin:{}'.format(wilcoxon(before_csp['Fraction Responsive'], after_csp['Fraction Responsive'])))

        print('Before CS-: {}'.format(np.mean(before_csm['Fraction Responsive'])))
        print('After CS-: {}'.format(np.mean(after_csm['Fraction Responsive'])))
        print('Wilcoxin:{}'.format(wilcoxon(before_csm['Fraction Responsive'], after_csm['Fraction Responsive'])))
    except:
        print('stats didnt work')

def plot_summary_odor_pretraining(res, start_days, end_days, arg_naive, figure_path, save, excitatory = True):
    ax_args_copy = ax_args.copy()
    res = copy.copy(res)
    list_of_days = list(zip(start_days, end_days))
    mice = np.unique(res['mouse'])
    start_end_day_res = filter.filter_days_per_mouse(res, days_per_mouse= list_of_days)
    get_responsive_cells(start_end_day_res)

    if arg_naive:
        day_start = filter.filter(start_end_day_res, {'odor_standard': 'PT Naive'})
        day_start['odor_standard'] = np.array(['PT CS+'] * len(day_start['odor_standard']))
        day_end = filter.filter_days_per_mouse(start_end_day_res, days_per_mouse= end_days)
        day_end = filter.filter(day_end, {'odor_standard': 'PT CS+'})
        reduce.chain_defaultdicts(day_start, day_end)
        start_end_day_res = day_start
    else:
        start_end_day_res = filter.exclude(start_end_day_res, {'odor_standard': 'PT Naive'})

    add_naive_learned(start_end_day_res, start_days, end_days, 'a','b')
    filter.assign_composite(start_end_day_res, loop_keys=['odor_standard', 'training_day'])

    odor_list = ['PT CS+']
    colors = ['Orange']
    ax_args_copy = ax_args_copy.copy()
    ax_args_copy.update({'xlim':[-1, 10], 'ylim':[0, .4], 'yticks':[0, .1, .2, .3, .4]})
    for i, odor in enumerate(odor_list):
        save_arg = False
        reuse_arg = True
        if i == 0:
            reuse_arg = False

        if save and i == len(odor_list) -1:
            save_arg = True

        plot.plot_results(start_end_day_res,
                          select_dict= {'odor_standard':odor},
                          x_key='odor_standard_training_day', y_key='Fraction Responsive', loop_keys='mouse',
                          colors= [colors[i]]*len(mice),
                          path =figure_path, plot_args=line_args, ax_args= ax_args_copy,
                          save=save_arg, reuse=reuse_arg,
                          fig_size=(2.5, 1.5),legend=False, name_str = '_E' if excitatory else '_I')

    before_csm = filter.filter(start_end_day_res, filter_dict={'training_day':'a', 'odor_standard':'PT CS+'})
    after_csm = filter.filter(start_end_day_res, filter_dict={'training_day':'b', 'odor_standard':'PT CS+'})
    from scipy.stats import ranksums, wilcoxon, kruskal

    print('Before PT CS+: {}'.format(np.mean(before_csm['Fraction Responsive'])))
    print('After PT CS+: {}'.format(np.mean(after_csm['Fraction Responsive'])))
    print('Wilcoxin:{}'.format(wilcoxon(before_csm['Fraction Responsive'], after_csm['Fraction Responsive'])))

def plot_summary_water(res, start_days, end_days, figure_path):
    ax_args_copy = ax_args.copy()
    res = copy.copy(res)
    get_responsive_cells(res)
    list_of_days = list(zip(start_days, end_days))
    mice = np.unique(res['mouse'])
    start_end_day_res = filter.filter_days_per_mouse(res, days_per_mouse= list_of_days)
    add_naive_learned(start_end_day_res, start_days, end_days, 'a','b')
    odor_list = ['US']
    colors = ['Turquoise']
    ax_args_copy.update({'xlim':[-1, 2]})
    for i, odor in enumerate(odor_list):
        plot.plot_results(start_end_day_res, select_dict={'odor_standard':odor},
                          x_key='training_day', y_key='Fraction Responsive', loop_keys='mouse',
                          colors= [colors[i]]*len(mice),
                          path =figure_path, plot_args=line_args, ax_args= ax_args_copy,
                          fig_size=(1.6, 1.5), legend=False)

    before_csm = filter.filter(start_end_day_res, filter_dict={'training_day':'a', 'odor_standard':'US'})
    after_csm = filter.filter(start_end_day_res, filter_dict={'training_day':'b', 'odor_standard':'US'})

    from scipy.stats import ranksums, wilcoxon, kruskal
    print('Before PT CS+: {}'.format(np.mean(before_csm['Fraction Responsive'])))
    print('After PT CS+: {}'.format(np.mean(after_csm['Fraction Responsive'])))
    print('Wilcoxin:{}'.format(wilcoxon(before_csm['Fraction Responsive'], after_csm['Fraction Responsive'])))

def get_responsive_cells(res):
    list_of_data = res['sig']
    for data in list_of_data:
        res['Fraction Responsive'].append(np.mean(data))
    res['Fraction Responsive'] = np.array(res['Fraction Responsive'])


def plot_individual(res, lick_res, figure_path):
    ax_args_copy = ax_args.copy()
    ax_args_copy.update({'ylim':[0,.65], 'yticks':[0, .3, .6]})
    overlap_ax_args_copy = overlap_ax_args.copy()
    res = copy.copy(res)
    get_responsive_cells(res)
    summary_res = reduce.new_filter_reduce(res, reduce_key= 'Fraction Responsive',
                                           filter_keys=['odor_valence', 'mouse','day'])

    summary_res = filter.filter(summary_res, {'odor_valence':'CS+'})
    lick_res = filter.filter(lick_res, {'odor_valence':'CS+'})
    for mouse in np.unique(summary_res['mouse']):
        select_dict = {'mouse':mouse}

        plot.plot_results(summary_res, x_key='day', y_key='Fraction Responsive', loop_keys='odor_valence',
                          colors=['green','red','turquoise'],
                          select_dict=select_dict, path=figure_path, ax_args=ax_args_copy, plot_args = line_args,
                          save=False, sort=True)

        plot.plot_results(lick_res, x_key='day', y_key='lick_boolean', loop_keys='odor_valence',
                          select_dict={'mouse': mouse},
                          colors=['green','red'],
                          ax_args=overlap_ax_args_copy, plot_args=behavior_line_args,
                          path=figure_path,
                          reuse=True, save=True, twinax=True)