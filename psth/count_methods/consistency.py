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
import analysis

plt.style.use('default')

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.size'] = 5
mpl.rcParams['font.family'] = 'arial'

def _correlation(res):
    res = filter.exclude(res, {'odor_valence': 'US'})
    # for i, dff in enumerate(res['data']):
    #     s = res['DAQ_O_ON_F'][i]
    #     e = res['DAQ_W_ON_F'][i]
    #     amplitude = np.mean(dff[:,:, s:e], axis=2)
    #     corrcoefs = np.corrcoef(amplitude.T)
    #     mask = ~np.eye(corrcoefs.shape[0], dtype=bool)
    #     corrcoef = np.mean(corrcoefs[mask])
    #     res['consistency_corrcoef'].append(corrcoef)

    for i, dff in enumerate(res['data']):
        cell_mask = res['msig'][i]
        s = res['DAQ_O_ON_F'][i]
        e = res['DAQ_W_ON_F'][i]
        dff = dff[:,:,s:e]
        corrcoef = []
        for dff_per_cell in dff:
            corrcoefs = np.corrcoef(dff_per_cell)
            diagmask = ~np.eye(corrcoefs.shape[0], dtype=bool)
            nanmask = ~np.isnan(corrcoefs)
            mask = np.logical_and(diagmask, nanmask)
            corrcoef_per_cell = np.mean(corrcoefs[mask])
            corrcoef.append(corrcoef_per_cell)
        corrcoef = np.array(corrcoef)
        mask = ~np.isnan(corrcoef, dtype=bool)
        corrcoef_ = np.mean(corrcoef[mask])
        res['consistency_corrcoef'].append(corrcoef_)
    res['consistency_corrcoef'] = np.array(res['consistency_corrcoef'])
    return res

def plot_consistency_within_day(res, start, end, shuffle, pretraining, figure_path):
    d = list(zip(start, end))
    res_temp = filter.filter_days_per_mouse(res, d)
    if pretraining:
        res_temp = filter.filter(res_temp, {'odor_valence': ['PT CS+']})
    else:
        res_temp = filter.filter(res_temp, {'odor_valence': ['CS+','CS-']})
    corr_res = _correlation(res_temp)
    corr_res.pop('data')

    analysis.add_naive_learned(corr_res, start, end, '0','1')
    res_ = reduce.new_filter_reduce(corr_res, filter_keys=['mouse','odor_standard','training_day'],
                                    reduce_key='consistency_corrcoef')
    res_.pop('consistency_corrcoef_sem')
    filter.assign_composite(res_, loop_keys=['training_day','odor_valence'])

    if shuffle:
        s = '_shuffled'
    else:
        s = ''

    ax_args_copy = ax_args.copy()
    ax_args_copy.update({'xlim':[-.5, 2.5], 'ylim':[0, .55], 'yticks':np.arange(0, 1.1, .1)})

    swarm_args_copy = swarm_args.copy()
    if pretraining:
        swarm_args_copy.update({'palette': ['gray', 'orange','green','red']})
    else:
        swarm_args_copy.update({'palette': ['gray','gray','green', 'red']})

    ix = res_['training_day_odor_valence'] == '1_PT CS+'
    res_['training_day_odor_valence'][ix] = '1_APT CS+'
    plot.plot_results(res_, x_key='training_day_odor_valence', y_key='consistency_corrcoef',
                      path = figure_path, plot_args=swarm_args_copy, plot_function=sns.stripplot, ax_args=ax_args_copy,
                      reuse= False, save=False, sort=True, name_str= s)

    summary = reduce.new_filter_reduce(res_, filter_keys='training_day_odor_valence', reduce_key='consistency_corrcoef')
    plot.plot_results(summary,
                      x_key='training_day_odor_valence', y_key= 'consistency_corrcoef',
                      error_key = 'consistency_corrcoef_sem',
                      colors= 'black',
                      path =figure_path, plot_args=error_args, plot_function=plt.errorbar,
                      save= True, reuse=True, legend=False, name_str=s)

    print(summary['consistency_corrcoef'])

    ix_a = res_['training_day_odor_valence'] == '0_CS+'
    ix_b = res_['training_day_odor_valence'] == '0_CS-'
    ix_c = res_['training_day_odor_valence'] == '1_CS+'
    ix_d = res_['training_day_odor_valence'] == '1_CS-'
    a = res_['consistency_corrcoef'][ix_a]
    b = res_['consistency_corrcoef'][ix_b]
    c = res_['consistency_corrcoef'][ix_c]
    d = res_['consistency_corrcoef'][ix_d]

    from scipy.stats import ranksums, wilcoxon, kruskal
    import scikit_posthocs

    print(kruskal(a, b, c))
    x =scikit_posthocs.posthoc_dunn(a = [a, b, c, d], p_adjust=None)
    print(x)
