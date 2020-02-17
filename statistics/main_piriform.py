import os

from _CONSTANTS.config import Config
import filter
import reduce
import numpy as np
import tools.file_io as fio
from behavior.behavior_analysis import get_days_per_mouse
import behavior
import analysis
import scikit_posthocs

import statistics.analyze
import statistics.count_methods.cory as cory
import statistics.count_methods.power as power
import statistics.count_methods.compare as compare
import statistics.count_methods.correlation as correlation
import statistics.count_methods.responsive as responsive
import statistics.count_methods.stability as stability
from scipy.stats import ranksums, wilcoxon, kruskal

condition_config = statistics.analyze.PIR_Config()
condition = condition_config.condition
data_path = os.path.join(Config.LOCAL_DATA_PATH, Config.LOCAL_DATA_TIMEPOINT_FOLDER, condition.name)
save_path = os.path.join(Config.LOCAL_EXPERIMENT_PATH, 'COUNTING', condition.name)
figure_path = os.path.join(Config.LOCAL_FIGURE_PATH, 'OTHER', 'COUNTING',  condition.name)

# res.pop('data')
# retrieving relevant days
learned_day_per_mouse, last_day_per_mouse = get_days_per_mouse(data_path, condition)

if condition_config.start_at_training and hasattr(condition, 'training_start_day'):
    start_days_per_mouse = condition.training_start_day
else:
    start_days_per_mouse = [0] * len(condition_config.condition.paths)
training_start_day_per_mouse = condition.training_start_day

lick_res = behavior.behavior_analysis.get_licks_per_day(data_path, condition)
analysis.add_odor_value(lick_res, condition)
lick_res = filter.filter(lick_res, {'odor_valence': ['CS+', 'CS-', 'PT CS+']})
lick_res = reduce.new_filter_reduce(lick_res, ['odor_valence', 'day', 'mouse'], reduce_key='lick_boolean')

temp_res = behavior.behavior_analysis.analyze_behavior(data_path, condition)
temp_res = filter.filter(temp_res, {'odor_valence': ['CS+']})
# temp_res = reduce.new_filter_reduce(temp_res, ['odor_valence', 'mouse'], reduce_key='boolean_smoothed')

if condition.name == 'PIR_CONTEXT':
    res = statistics.analyze.analyze_data(save_path, condition_config)

    # compare.plot_compare_dff(res, start_days_per_mouse, last_day_per_mouse,
    #                          arg='all', valence='CS+', more_stats=False, figure_path= figure_path)
    # compare.plot_compare_dff(res, start_days_per_mouse, last_day_per_mouse,
    #                          arg='all', valence='CS-', more_stats=False, figure_path= figure_path)
    #
    # power.plot_power(res, start_days_per_mouse, last_day_per_mouse, figure_path, odor_valence=['CS+'],
    #                  colors_before = {'CS+':'Gray','CS-':'Gray'}, colors_after = {'CS+':'Green','CS-':'Red'}, ylim=0.15)
    # power.plot_power(res, start_days_per_mouse, last_day_per_mouse, figure_path, odor_valence=['CS-'],
    #                  colors_before = {'CS+':'Gray','CS-':'Gray'}, colors_after = {'CS+':'Green','CS-':'Red'}, ylim=0.15)

    shuffle = False
    odor_end = False
    days = [[[0,0],[1,1]]]
    temp = correlation.plot_correlation_across_days(res, days, loop_keys=['mouse', 'odor'], shuffle=shuffle, figure_path = figure_path,
                                                    reuse=False, save=True, analyze=True, plot_bool=False, odor_end=odor_end)
    correlation.plot_correlation_across_days(temp, days, loop_keys=['mouse', 'odor'], shuffle=shuffle, figure_path = figure_path,
                                             reuse=False, save=True, analyze=False, plot_bool=True, odor_end=odor_end)
    ixa = temp['odor_valence'] == 'CS+'
    ixb = temp['odor_valence'] == 'CS-'
    a = temp['corrcoef'][ixa]
    b = temp['corrcoef'][ixb]
    print(ranksums(a,b))

if condition.name == 'PIR':
    naive_config = statistics.analyze.PIR_NAIVE_Config()
    data_path_ = os.path.join(Config.LOCAL_DATA_PATH, Config.LOCAL_DATA_TIMEPOINT_FOLDER, naive_config.condition.name)
    save_path_ = os.path.join(Config.LOCAL_EXPERIMENT_PATH, 'COUNTING', naive_config.condition.name)
    res_naive = fio.load_pickle(os.path.join(save_path_, 'dict.pkl'))
    learned_day_per_mouse_, last_day_per_mouse_ = get_days_per_mouse(data_path_, naive_config.condition)
    #
    res = statistics.analyze.analyze_data(save_path, condition_config, m_threshold= .1)
    # res_naive = statistics.analyze.analyze_data(save_path_, naive_config, m_threshold= .1)
    # res_naive['odor_valence'] = np.array(['Naive'] * len(res_naive['day']))

    # excitatory = [True, False]
    # thresholds = [0.1, -0.05]
    # for i, sign in enumerate(excitatory):
    #     res = statistics.analyze.analyze_data(save_path, condition_config, m_threshold= thresholds[i], excitatory=sign)
    #     responsive.plot_summary_odor_and_water(res, start_days_per_mouse, training_start_day_per_mouse, last_day_per_mouse,
    #                                        figure_path=figure_path, excitatory= sign, arg='odor_standard')

    # responsive.plot_individual(res, lick_res, figure_path = figure_path)
    # overlap.plot_overlap_odor(res, start_days_per_mouse, learned_day_per_mouse,
    #                                              delete_non_selective=True, figure_path= figure_path)
    # stability.plot_stability_across_days(res, start_days_per_mouse, learned_day_per_mouse, figure_path = figure_path)
    # waveform.distribution(res, start=learned_day_per_mouse, end=last_day_per_mouse, data_arg='onset', figure_path=figure_path)
    # waveform.distribution(res, start=learned_day_per_mouse, end=last_day_per_mouse, data_arg='amplitude', figure_path=figure_path)
    # waveform.distribution(res, start=learned_day_per_mouse, end=last_day_per_mouse, data_arg='duration', figure_path=figure_path)

    # responsive.plot_summary_odor_and_water(res, start_days_per_mouse, training_start_day_per_mouse, last_day_per_mouse,
    #                                        figure_path=figure_path)

    # power.plot_power(res, start_days_per_mouse, last_day_per_mouse, figure_path, odor_valence=['CS+'],
    #                  colors_before = {'CS+':'Gray','CS-':'Gray'}, colors_after = {'CS+':'Green','CS-':'Red'},
    #                  excitatory=True, ylim=[-0.005, .15])
    # power.plot_power(res, start_days_per_mouse, last_day_per_mouse, figure_path, odor_valence=['CS+'],
    #                  colors_before = {'CS+':'Gray','CS-':'Gray'}, colors_after = {'CS+':'Green','CS-':'Red'},
    #                  excitatory=False, ylim=[-0.1, .005])
    #
    # power.plot_power(res, start_days_per_mouse, last_day_per_mouse, figure_path, odor_valence=['CS-'],
    #                  colors_before = {'CS+':'Gray','CS-':'Gray'}, colors_after = {'CS+':'Green','CS-':'Red'},
    #                  excitatory=True, ylim=[-0.005, .15])
    # power.plot_power(res, start_days_per_mouse, last_day_per_mouse, figure_path, odor_valence=['CS-'],
    #                  colors_before = {'CS+':'Gray','CS-':'Gray'}, colors_after = {'CS+':'Green','CS-':'Red'},
    #                  excitatory=False, ylim=[-0.1, .005])

    # cory.main(res, temp_res, figure_path, excitatory=True)
    # cory.main(res, temp_res, figure_path, excitatory=False)



    # overlap.plot_overlap_odor(res, start_days_per_mouse, last_day_per_mouse, figure_path = figure_path)

    # responsive.plot_responsive_difference_odor_and_water(res, start_days_per_mouse, training_start_day_per_mouse, last_day_per_mouse,
    #                                        figure_path=figure_path, average=False, reuse_arg=False, save_arg=False)
    #
    # responsive.plot_responsive_difference_odor_and_water(res_, [0,0,0,0], [0,0,0,0], last_day_per_mouse_, use_colors=False,
    #                                                      include_water=False, normalize=False,
    #                                                      figure_path=figure_path, average=False, reuse_arg=True, save_arg=True)

    # odor_end = False
    # for d in [-1, 0, 1]:
    #     a = correlation.plot_correlation_matrix(res, start_days_per_mouse, loop_keys=['mouse'], shuffle=False,
    #                                             figure_path = figure_path, odor_end=odor_end, direction=d)
    #     b = correlation.plot_correlation_matrix(res, last_day_per_mouse, loop_keys=['mouse'], shuffle=False,
    #                                             figure_path = figure_path, odor_end=odor_end, direction=d)

    odor_end = False
    args = [False, 'CS+', 'CS-']
    for arg in args:
        correlation.plot_correlation(res, start_days_per_mouse, last_day_per_mouse, figure_path=figure_path,
                                     odor_end=odor_end, arg=arg,
                                     direction=1, color='green', save=False, reuse=False)
        correlation.plot_correlation(res, start_days_per_mouse, last_day_per_mouse, figure_path=figure_path,
                                     odor_end=odor_end, arg=arg,
                                     direction=-1, color='red', save=False, reuse=True)
        correlation.plot_correlation(res, start_days_per_mouse, last_day_per_mouse, figure_path=figure_path,
                                     odor_end=odor_end, linestyle='--', arg=arg,
                                     direction=0, color='black', save=True, reuse=True)

    # correlation.plot_correlation_matrix(res, start_days_per_mouse, loop_keys=['mouse'], shuffle=True, figure_path = figure_path)
    # correlation.plot_correlation_matrix(res, last_day_per_mouse, loop_keys=['mouse'], shuffle=True, figure_path = figure_path)
    # print(last_day_per_mouse)

    # shuffle = True
    # odor_end = False
    # days = [[[0,0,0,0,0,0],[3,2,3,3,3,3]]]
    # out1 = correlation.plot_correlation_across_days(res, days, loop_keys=['mouse', 'odor'], shuffle=shuffle, figure_path = figure_path,
    #                                                 reuse=False, save=True, analyze=True, plot_bool=False, odor_end=odor_end)
    # days_naive = [[[0,0,0,0],[3,3,3,3]]]
    # out2 = correlation.plot_correlation_across_days(res_naive, days_naive, loop_keys=['mouse', 'odor'], shuffle=shuffle, figure_path = figure_path,
    #                                                 reuse=True, save=True, analyze=True, plot_bool=False, odor_end=odor_end)
    # out2['odor_valence'] = np.array(['Naive'] * len(out2['odor_valence']))
    # out = reduce.chain_defaultdicts(out1, out2, copy_dict=True)
    # correlation.plot_correlation_across_days(out, days, loop_keys=['mouse', 'odor'], shuffle=shuffle, figure_path = figure_path,
    #                                          reuse=False, save=True, analyze=False, plot_bool=True, odor_end=odor_end)
    #
    # ixa = out1['odor_valence'] == 'CS+'
    # ixb = out1['odor_valence'] == 'CS-'
    # ixc = out2['odor_valence'] == 'Naive'
    # a = out1['corrcoef'][ixa]
    # b = out1['corrcoef'][ixb]
    # c = out2['corrcoef'][ixc]
    # print(kruskal(a,b,c))
    # print(ranksums(a,b))
    # print(ranksums(a,c))
    # print(ranksums(b,c))
    #
    # x =scikit_posthocs.posthoc_dunn(a = [a, b, c], p_adjust=None)
    # print(x)


if condition.name == 'PIR_NAIVE':
    days = [3, 3, 2, 2]
    res = statistics.analyze.analyze_data(save_path, condition_config)

    # excitatory = [True, False]
    # thresholds = [0.1, -0.05]
    # for i, sign in enumerate(excitatory):
    #     res = statistics.analyze.analyze_data(save_path, condition_config, m_threshold= thresholds[i], excitatory=sign)
    #     ix = res['odor_valence'] == 'CS-'
    #     res['odor_valence'][ix] = 'CS+'
    #     responsive.plot_summary_odor_and_water(res, start_days_per_mouse, training_start_day_per_mouse, last_day_per_mouse,
    #                                        figure_path=figure_path, excitatory= sign, arg = 'naive')
    #
    # ac = a['corrcoef']
    # bc = b['corrcoef']
    # ac = ac[a['Odor_A'] != a['Odor_B']]
    # bc = bc[b['Odor_A'] != b['Odor_B']]
    # x = wilcoxon(ac, bc)
    # print(x)
    # print(np.mean(ac))
    # print(np.mean(bc))

    # responsive.plot_summary_odor(res, start_days_per_mouse, days, use_colors=False, figure_path = figure_path)
    # overlap.plot_overlap_odor(res, start_days_per_mouse, days,
    #                                              delete_non_selective=True, figure_path= figure_path)
    # stability.plot_stability_across_days(res, start_days_per_mouse, learned_day_per_mouse, figure_path = figure_path)

    # res['odor_valence'] = np.array(['CS'] * np.size(res['odor_valence']))
    # temp_res['odor_valence'] = np.array(['CS'] * np.size(temp_res['odor_valence']))
    # cory.main(res, temp_res, figure_path, excitatory=True, valence='CS+')
    # cory.main(res, temp_res, figure_path, excitatory=False)

    # odor_end = False
    # for d in [-1, 0, 1]:
    #     a = correlation.plot_correlation_matrix(res, start_days_per_mouse, loop_keys=['mouse'], shuffle=False,
    #                                             figure_path = figure_path, odor_end=odor_end, direction=d)
    #     b = correlation.plot_correlation_matrix(res, last_day_per_mouse, loop_keys=['mouse'], shuffle=False,
    #                                             figure_path = figure_path, odor_end=odor_end, direction=d)
    #

    odor_end = False
    correlation.plot_correlation(res, start_days_per_mouse, last_day_per_mouse, figure_path=figure_path, odor_end=odor_end,
                                 direction=1, color='green', save=False, reuse=False)
    correlation.plot_correlation(res, start_days_per_mouse, last_day_per_mouse, figure_path=figure_path, odor_end=odor_end,
                                 direction=-1, color='red', save=False, reuse=True)
    correlation.plot_correlation(res, start_days_per_mouse, last_day_per_mouse, figure_path=figure_path, odor_end=odor_end,
                                 direction=0, color='black', save=True, reuse=True, linestyle='--',)


    # res['odor_valence'] = np.array(['CS'] * np.size(res['odor_valence']))
    # # res = filter.exclude(res, {'odor':'2pe'})
    # power.plot_power(res, start_days_per_mouse, last_day_per_mouse, figure_path, odor_valence=['CS'],
    #                  colors_before = {'CS':'Gray'}, colors_after = {'CS':'GoldenRod'},
    #                  excitatory=True, ylim=[-0.005, .15])
    # power.plot_power(res, start_days_per_mouse, last_day_per_mouse, figure_path, odor_valence=['CS'],
    #                  colors_before={'CS': 'Gray'}, colors_after={'CS': 'GoldenRod'},
    #                  excitatory=False, ylim=[-0.1, .005])
