import copy
import numpy as np
import filter
import plot
import reduce
from analysis import add_naive_learned
from format import *
import matplotlib.pyplot as plt

def magnitude_histogram(res, learned_days, end_days, use_colors= True, figure_path = None, reuse = False):
    def _helper(res):
        list_odor_on = res['DAQ_O_ON_F']
        list_water_on = res['DAQ_W_ON_F']
        list_of_dff = res['dff']
        for i, dff in enumerate(list_of_dff):
            s = list_odor_on[i]
            e = list_water_on[i]
            max = np.max(dff[:, s:e], axis=1)
            res['max_dff'].append(max)
        res['max_dff'] = np.array(res['max_dff'])

    def _helper1(res):
        for data in res['amplitude']:
            res['max_dff'].append(data[data > 0])
        res['max_dff'] = np.array(res['max_dff'])

    res = copy.copy(res)
    # list_of_days = list(zip(learned_days, end_days))
    list_of_days = end_days
    start_end_day_res = filter.filter_days_per_mouse(res, days_per_mouse= list_of_days)
    _helper(start_end_day_res)
    add_naive_learned(start_end_day_res, learned_days, end_days)
    filter.assign_composite(start_end_day_res, loop_keys=['odor_standard', 'training_day'])
    csp = filter.filter(start_end_day_res, filter_dict={'odor_valence':'CS+'})
    csm = filter.filter(start_end_day_res, filter_dict={'odor_valence':'CS-'})

    csp_data = np.hstack(csp['max_dff'].flat)
    csm_data = np.hstack(csm['max_dff'].flat)

    fig = plt.figure(figsize = (3,2))
    ax = fig.add_axes([.2, .2, .7, .7])
    x_upper = 1.01
    y_upper = .5
    bins = 20
    xrange = [0, x_upper]
    yrange = [0, y_upper]
    xticks = np.arange(0, x_upper, .2)
    yticks = np.arange(0, y_upper, 3)
    legends = ['CS+','CS-']
    colors = ['green','red']

    for i, data in enumerate([csp_data,csm_data]):
        plt.hist(data, bins= bins, range= xrange, density=True, color= colors[i], alpha = .5)
    plt.legend(legends, loc=1, bbox_to_anchor=(1.05, .4), fontsize=5)
    ax.set_xlabel('DF/F')
    ax.set_ylabel('Fraction')
    ax.set_xticks(xticks)
    # ax.set_yticks(yticks)
    plt.xlim(xrange)
    # plt.ylim(yrange)
    plt.show()




    # ax_args_copy = ax_args.copy()
    # odor_list = ['CS+1', 'CS+2','CS-1', 'CS-2']
    # if use_colors:
    #     colors = ['Green','Green','Red','Red']
    # else:
    #     colors = ['Black'] * 4
    # ax_args_copy = ax_args_copy.copy()
    # ax_args_copy.update({'xlim':[-1, 8], 'ylim':[0, .4], 'yticks':[0, .1, .2, .3, .4]})
    # for i, odor in enumerate(odor_list):
    #     save_arg = False
    #     reuse_arg = True
    #     if i == 0 and not reuse:
    #         reuse_arg = False
    #     if i == len(odor_list) -1:
    #         save_arg = True
    #
    #     plot.plot_results(start_end_day_res,
    #                       select_dict= {'odor_standard':odor},
    #                       x_key='odor_standard_training_day', y_key='Fraction Responsive', loop_keys='mouse',
    #                       colors= [colors[i]]*len(mice),
    #                       path =figure_path, plot_args=line_args, ax_args= ax_args_copy,
    #                       save=save_arg, reuse=reuse_arg,
    #                       fig_size=(2.5, 1.5),legend=False, name_str = ','.join([str(x) for x in start_days]))