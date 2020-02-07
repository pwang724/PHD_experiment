from collections import defaultdict
import numpy as np
from matplotlib import pyplot as plt
import filter
import plot
from plot import _easy_save
from format import *
import reduce
import seaborn as sns
import matplotlib as mpl
import psth.psth_helper

plt.style.use('default')

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.size'] = 5
mpl.rcParams['font.family'] = 'arial'

def _correlation(res, loop_keys, shuffle, odor_end = True, direction = 0):
    res = filter.exclude(res, {'odor_valence':'US'})
    for i, dff in enumerate(res['dff']):
        s = res['DAQ_O_ON_F'][i]
        e = res['DAQ_W_ON_F'][i]
        if odor_end:
            amplitude = np.max(dff[:, s:e], axis=1)
        else:
            amplitude = np.max(dff[:, s:], axis=1)
        res['corr_amp'].append(amplitude)
    res['corr_amp'] = np.array(res['corr_amp'])

    combinations, list_of_ixs = filter.retrieve_unique_entries(res, loop_keys=loop_keys)

    loop_keys_ = loop_keys +  ['odor_valence', 'odor_standard']

    corrcoefs = defaultdict(list)
    for ixs in list_of_ixs:
        data = res['corr_amp'][ixs]
        for i, data_1 in enumerate(data):
            for j, data_2 in enumerate(data):
                if shuffle:
                    n_iter = 10
                    corrcoef = 0
                    for k in range(n_iter):
                        corrcoef += np.corrcoef((np.random.permutation(data_1), np.random.permutation(data_2)))[0, 1]
                    corrcoef /= (n_iter * 1.0)
                else:
                    if i != j:
                        datas = res['data'][ixs[i]]
                        s = res['DAQ_O_ON_F'][ixs[i]]
                        e = res['DAQ_W_ON_F'][ixs[i]]
                        ds = []
                        for cell_data in datas:
                            config = psth.psth_helper.PSTHConfig()
                            d = psth.psth_helper.subtract_baseline(cell_data, config.baseline_start,
                                                                   s - config.baseline_end)
                            ds.append(d)
                        datas_i = np.array(ds)

                        datas = res['data'][ixs[j]]
                        s = res['DAQ_O_ON_F'][ixs[j]]
                        e = res['DAQ_W_ON_F'][ixs[j]]
                        ds = []
                        for cell_data in datas:
                            config = psth.psth_helper.PSTHConfig()
                            d = psth.psth_helper.subtract_baseline(cell_data, config.baseline_start,
                                                                   s - config.baseline_end)
                            ds.append(d)
                        datas_j = np.array(ds)

                        corrcoefs_ = []
                        for rep in np.arange(100):
                            s_ix_a = np.random.choice(datas_i.shape[1], datas_i.shape[1]//2, replace=False)
                            s_ix_b = np.random.choice(datas_j.shape[1], datas_j.shape[1]//2, replace=False)
                            dffa = np.mean(datas_i[:,s_ix_a,:], axis=1)
                            dffb = np.mean(datas_j[:,s_ix_b,:], axis=1)
                            if odor_end:
                                dffa = dffa[:, s:e]
                                dffb = dffb[:, s:e]
                            else:
                                dffa = dffa[:,s:]
                                dffb = dffb[:,s:]

                            if direction == 1:
                                dffa[dffa < 0] = 0
                                dffb[dffb < 0] = 0
                                amplitudea = np.max(dffa, axis=1)
                                amplitudeb = np.max(dffb, axis=1)
                            elif direction == -1:
                                dffa[dffa > 0] = 0
                                dffb[dffb > 0] = 0
                                amplitudea = np.min(dffa, axis=1)
                                amplitudeb = np.min(dffb, axis=1)
                            elif direction == 0:
                                amplitudea = np.max(dffa, axis=1)
                                amplitudeb = np.max(dffb, axis=1)
                            else:
                                raise ValueError('no direction given')

                            # if odor_end:
                            #     amplitudea = np.max(dffa[:, s:e], axis=1)
                            #     amplitudeb = np.max(dffb[:, s:e], axis=1)
                            # else:
                            #     amplitudea = np.max(dffa[:, s:], axis=1)
                            #     amplitudeb = np.max(dffb[:, s:], axis=1)
                            corrcoefs_.append(np.corrcoef(amplitudea, amplitudeb)[0,1])
                        corrcoef = np.mean(corrcoefs_)

                        # corrcoef = np.corrcoef((data_1, data_2))[0, 1]
                    else:
                        # corrcoef = np.corrcoef((data_1, data_2))[0, 1]

                        datas = res['data'][ixs[i]]
                        s = res['DAQ_O_ON_F'][ixs[i]]
                        e = res['DAQ_W_ON_F'][ixs[i]]
                        ds = []
                        for cell_data in datas:
                            config = psth.psth_helper.PSTHConfig()
                            d = psth.psth_helper.subtract_baseline(cell_data, config.baseline_start,
                                                                   s - config.baseline_end)
                            ds.append(d)
                        datas = np.array(ds)

                        corrcoefs_ = []
                        for rep in np.arange(100):
                            s_ix_a = np.random.choice(datas.shape[1], datas.shape[1]//2, replace=False)
                            s_ix_b = [x for x in np.arange(datas.shape[1]) if x not in s_ix_a]
                            dffa = np.mean(datas[:,s_ix_a,:], axis=1)
                            dffb = np.mean(datas[:,s_ix_b,:], axis=1)
                            # if odor_end:
                            #     amplitudea = np.max(dffa[:, s:e], axis=1)
                            #     amplitudeb = np.max(dffb[:, s:e], axis=1)
                            # else:
                            #     amplitudea = np.max(dffa[:, s:], axis=1)
                            #     amplitudeb = np.max(dffb[:, s:], axis=1)
                            if direction == 1:
                                dffa[dffa < 0] = 0
                                dffb[dffb < 0] = 0
                                amplitudea = np.max(dffa, axis=1)
                                amplitudeb = np.max(dffb, axis=1)
                            elif direction == -1:
                                dffa[dffa > 0] = 0
                                dffb[dffb > 0] = 0
                                amplitudea = np.min(dffa, axis=1)
                                amplitudeb = np.min(dffb, axis=1)
                            elif direction == 0:
                                amplitudea = np.max(dffa, axis=1)
                                amplitudeb = np.max(dffb, axis=1)
                            else:
                                raise ValueError('no direction given')
                            corrcoefs_.append(np.corrcoef(amplitudea, amplitudeb)[0,1])
                        corrcoef = np.mean(corrcoefs_)

                corrcoefs['corrcoef'].append(corrcoef)
                corrcoefs['Odor_A'].append(i)
                corrcoefs['Odor_B'].append(j)
                for loop_key in loop_keys_:
                    corrcoefs[loop_key].append(res[loop_key][ixs[0]])

    for key, value in corrcoefs.items():
        corrcoefs[key] = np.array(value)
    return corrcoefs

def plot_correlation(res, start_days, end_days, figure_path, odor_end = True, linestyle = '-', direction = 0,
                     arg=False,
                     save=False, reuse=False, color = 'black'):
    def _get_ixs(r, arg):
        A = r['Odor_A']
        B = r['Odor_B']
        l = []
        for i, a in enumerate(A):
            b = B[i]
            if arg == 'opposing':
                if a < 2 and b > 1:
                    l.append(i)
            elif arg == 'CS+':
                if a == 0 and b == 1:
                    l.append(i)
            elif arg == 'CS-':
                if a == 2 and b == 3:
                    l.append(i)
        return np.array(l)

    res_before = filter.filter_days_per_mouse(res, start_days)
    corr_before = _correlation(res_before, ['mouse'], shuffle=False, odor_end=odor_end, direction=direction)
    corr_before['day'] = np.array(['A'] * len(corr_before['Odor_A']))
    res_after = filter.filter_days_per_mouse(res, end_days)
    corr_after = _correlation(res_after, ['mouse'], shuffle=False, odor_end=odor_end, direction=direction)
    corr_after['day'] = np.array(['B'] * len(corr_after['Odor_A']))
    corr = defaultdict(list)
    reduce.chain_defaultdicts(corr, corr_before)
    reduce.chain_defaultdicts(corr, corr_after)

    ix_same = np.equal(corr['Odor_A'], corr['Odor_B'])
    ix_different = np.invert(ix_same)
    for k, v in corr.items():
        corr[k] = v[ix_different]

    if arg is not False:
        ixs = _get_ixs(corr, arg)
        for k, v in corr.items():
            corr[k] = v[ixs]


    mouse_corr = reduce.new_filter_reduce(corr, filter_keys=['mouse','day'], reduce_key='corrcoef')
    mouse_corr.pop('corrcoef_sem')
    mouse_corr.pop('corrcoef_std')
    average_corr = reduce.new_filter_reduce(mouse_corr, filter_keys=['day'], reduce_key='corrcoef')

    error_args = {'capsize': 2, 'elinewidth': 1, 'markersize': 2, 'alpha': .5}
    error_args.update({'linestyle':linestyle})
    plot.plot_results(average_corr, x_key='day', y_key='corrcoef', error_key='corrcoef_sem',
                      plot_args= error_args,
                      plot_function=plt.errorbar,
                      colors= color,
                      ax_args={'ylim':[0, 1], 'xlim':[-.5, 1.5]},
                      save=save, reuse=reuse,
                      path = figure_path, name_str=str(direction) + '_' + str(arg))

    #stats
    before_odor = filter.filter(corr, filter_dict={'day':'A'})
    after_odor = filter.filter(corr, filter_dict={'day':'B'})
    from scipy.stats import ranksums, wilcoxon, kruskal
    print('direction: {}'.format(direction))
    print('Data Before: {}'.format(before_odor['corrcoef']))
    print('Data After: {}'.format(after_odor['corrcoef']))
    print('Before: {}'.format(np.mean(before_odor['corrcoef'])))
    print('After: {}'.format(np.mean(after_odor['corrcoef'])))
    print('Wilcoxin:{}'.format(wilcoxon(before_odor['corrcoef'], after_odor['corrcoef'])))


def plot_correlation_matrix(res, days, loop_keys, shuffle, figure_path, odor_end = True, direction = 0):
    res = filter.filter_days_per_mouse(res, days)
    res_ = _correlation(res, loop_keys, shuffle, odor_end, direction=direction)
    res = reduce.new_filter_reduce(res_, filter_keys= ['Odor_A', 'Odor_B'], reduce_key='corrcoef')
    if shuffle:
        s = '_shuffled'
    else:
        s = ''
    plot.plot_weight(res, x_key='Odor_A', y_key='Odor_B', val_key='corrcoef', title='Correlation', label='Correlation',
                     vmin = 0, vmax = 1, mask=True,
                     xticklabel= ['CS+1', 'CS+2', 'CS-1', 'CS-2'], yticklabel = ['CS+1', 'CS+2', 'CS-1', 'CS-2'],
                     save_path=figure_path, text= ','.join([str(x) for x in days]) + s + '_direction_' + str(direction))
    return res_

def plot_correlation_across_days(res, days, loop_keys, shuffle, figure_path, reuse, save, analyze, plot_bool, odor_end = True):
    if analyze:
        res_ = defaultdict(list)
        for day_list in days:
            d = list(zip(day_list[0], day_list[1]))
            res_temp = filter.filter_days_per_mouse(res, d)
            corr_res = _correlation(res_temp, loop_keys, shuffle, odor_end=odor_end)
            reduce.chain_defaultdicts(res_, corr_res)

        res_ = filter.filter(res_, {'Odor_A': 0, 'Odor_B':1})
        res_ = reduce.new_filter_reduce(res_, filter_keys=['mouse','odor_standard'], reduce_key='corrcoef')
        res_.pop('corrcoef_sem')
        return res_

    if plot_bool:
        res_ = res
        if shuffle:
            s = '_shuffled'
        else:
            s = ''

        ax_args_copy = ax_args.copy()
        ax_args_copy.update({'xlim':[-.5, 2.5], 'ylim':[0, 1.05], 'yticks':np.arange(0, 1.1, .2)})

        swarm_args_copy = swarm_args.copy()
        swarm_args_copy.update({'palette': ['green', 'red','gray']})

        plot.plot_results(res_, x_key='odor_valence', y_key='corrcoef',
                               path = figure_path, plot_args=swarm_args_copy, plot_function=sns.stripplot, ax_args=ax_args_copy,
                               reuse= reuse, save=False, sort=True, name_str= s)

        summary = reduce.new_filter_reduce(res_, filter_keys=['odor_valence'], reduce_key='corrcoef')
        plot.plot_results(summary,
                               x_key='odor_valence', y_key= 'corrcoef', error_key = 'corrcoef_sem',
                               colors= 'black',
                               path =figure_path, plot_args=error_args, plot_function=plt.errorbar,
                               save= save, reuse=True, legend=False, name_str=s)

        from scipy.stats import ranksums
        print(summary['corrcoef'])

def plot_correlation_over_time(res, start_days, end_days, figure_path):
    pass