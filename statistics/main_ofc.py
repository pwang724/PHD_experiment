import os

from _CONSTANTS.config import Config
import filter
import reduce
import numpy as np
import tools.file_io as fio
from behavior.behavior_analysis import get_days_per_mouse
import behavior
import analysis
# import scikit_posthocs

import statistics.analyze
import statistics.count_methods.overlap as overlap
import statistics.count_methods.waveform as waveform
import statistics.count_methods.power as power
import statistics.count_methods.compare as compare
import statistics.count_methods.correlation as correlation
import statistics.count_methods.cory as cory
import statistics.count_methods.responsive as responsive
from scipy.stats import ranksums, wilcoxon, kruskal

condition_config = statistics.analyze.OFC_Config()
condition = condition_config.condition
data_path = os.path.join(Config.LOCAL_DATA_PATH, Config.LOCAL_DATA_TIMEPOINT_FOLDER, condition.name)
save_path = os.path.join(Config.LOCAL_EXPERIMENT_PATH, 'COUNTING', condition.name)
figure_path = os.path.join(Config.LOCAL_FIGURE_PATH, 'OTHER', 'COUNTING',  condition.name)

# retrieving relevant days
learned_day_per_mouse, last_day_per_mouse = get_days_per_mouse(data_path, condition)

if condition_config.start_at_training and hasattr(condition, 'training_start_day'):
    start_days_per_mouse = condition.training_start_day
else:
    start_days_per_mouse = [0] * len(condition_config.condition.paths)
training_start_day_per_mouse = condition.training_start_day

#behavior
lick_res = behavior.behavior_analysis.get_licks_per_day(data_path, condition)
analysis.add_odor_value(lick_res, condition)
lick_res = filter.filter(lick_res, {'odor_valence': ['CS+', 'CS-', 'PT CS+']})
lick_res = reduce.new_filter_reduce(lick_res, ['odor_valence', 'day', 'mouse'], reduce_key='lick_boolean')
temp_res = behavior.behavior_analysis.analyze_behavior(data_path, condition)

if condition.name == 'OFC' or condition.name == 'BLA':
    if condition.name == 'OFC':
        last_day_per_mouse = [5, 5, 3, 4, 4]

    res = statistics.analyze.analyze_data(save_path, condition_config, m_threshold=0.04)

    # naive_config = statistics.analyze.OFC_LONGTERM_Config()
    # data_path_ = os.path.join(Config.LOCAL_DATA_PATH, Config.LOCAL_DATA_TIMEPOINT_FOLDER, naive_config.condition.name)
    # save_path_ = os.path.join(Config.LOCAL_EXPERIMENT_PATH, 'COUNTING', naive_config.condition.name)
    # res_naive = statistics.analyze.analyze_data(save_path_, condition_config, m_threshold=0.04)
    # res_naive = filter.exclude(res_naive, {'mouse': 3})
    # res_naive['mouse'] += 5
    # temp_res_naive = behavior.behavior_analysis.analyze_behavior(data_path_, naive_config.condition)
    # temp_res_naive = filter.filter(temp_res_naive, {'odor_valence': ['CS+']})
    # temp_res_naive = filter.exclude(temp_res_naive, {'mouse': 3})
    # temp_res_naive['mouse'] += 5
    res = filter.exclude(res, {'day': 0})
    # reduce.chain_defaultdicts(res, res_naive)
    # reduce.chain_defaultdicts(temp_res, temp_res_naive)
    # learned_days_combined = [3, 3, 2, 3, 3, 3, 2, 2]
    # last_days_combined = [5, 5, 3, 4, 4, 8, 7, 5]

    cory.main(res, temp_res, figure_path, excitatory=True,valence='CS+')
    # cory.main(res, temp_res, figure_path, excitatory=False,valence='CS+')
    # cory.main(res, temp_res, figure_path, excitatory=True,valence='CS-')
    # cory.main(res, temp_res, figure_path, excitatory=False,valence='CS-')

    # waveform.behavior_vs_neural_onset(res, temp_res, learned_day_per_mouse, last_day_per_mouse, figure_path,
    #                                   behavior_arg='onset')
    # waveform.behavior_vs_neural_onset(res, temp_res, learned_days_combined, last_days_combined, figure_path, behavior_arg='com')
    # waveform.behavior_vs_neural_onset(res, temp_res, learned_days_combined, last_days_combined, figure_path, behavior_arg='onset')
    # waveform.behavior_vs_neural_onset(res, temp_res, learned_days_combined, last_days_combined, figure_path, behavior_arg='magnitude')
    # waveform.behavior_vs_neural_power(res, temp_res, learned_days_combined, last_days_combined, figure_path, behavior_arg='com')
    # waveform.behavior_vs_neural_power(res, temp_res, learned_days_combined, last_days_combined, figure_path, behavior_arg='magnitude')
    # waveform.behavior_vs_neural_power(res, temp_res, learned_days_combined, last_days_combined, figure_path, behavior_arg='onset')

    # excitatory = [True, False]
    # thresholds = [0.04, -0.04]
    # for i, sign in enumerate(excitatory):
    #     res = statistics.analyze.analyze_data(save_path, condition_config, m_threshold= thresholds[i], excitatory=sign)
    #     res.pop('data')
        # responsive.plot_individual(res, lick_res, figure_path=figure_path)
        # responsive.plot_summary_odor_and_water(res, start_days_per_mouse, training_start_day_per_mouse, last_day_per_mouse,
        #                                    figure_path=figure_path, excitatory= sign)
        # overlap.plot_overlap_odor(res, start_days_per_mouse, last_day_per_mouse, figure_path = figure_path, excitatory=sign)
    # overlap.plot_overlap_water(res, training_start_day_per_mouse, last_day_per_mouse, figure_path = figure_path)

    # odor_end = True
    # opposing_valence = True
    # correlation.plot_correlation(res, start_days_per_mouse, last_day_per_mouse, figure_path=figure_path,
    #                              odor_end=odor_end, opposing_valence=opposing_valence,
    #                              direction=1, color='green', save=False, reuse=False)
    # correlation.plot_correlation(res, start_days_per_mouse, last_day_per_mouse, figure_path=figure_path,
    #                              odor_end=odor_end, opposing_valence=opposing_valence,
    #                              direction=-1, color='red', save=False, reuse=True)
    # correlation.plot_correlation(res, start_days_per_mouse, last_day_per_mouse, figure_path=figure_path,
    #                              odor_end=odor_end, linestyle='--', opposing_valence=opposing_valence,
    #                              direction=0, color='black', save=True, reuse=True)

    # waveform.compare_to_shuffle(res, start= learned_day_per_mouse, end = last_day_per_mouse, data_arg='onset', figure_path=figure_path)
    # waveform.compare_to_shuffle(res, start= learned_day_per_mouse, end = last_day_per_mouse, data_arg='amplitude', figure_path=figure_path)
    # waveform.compare_to_shuffle(res, start= learned_day_per_mouse, end = last_day_per_mouse, data_arg='duration', figure_path=figure_path)
    # waveform.distribution(res, start=learned_day_per_mouse, end=last_day_per_mouse, data_arg='onset', figure_path=figure_path)
    # waveform.distribution(res, start=learned_day_per_mouse, end=last_day_per_mouse, data_arg='amplitude', figure_path=figure_path)
    # waveform.distribution(res, start=learned_day_per_mouse, end=last_day_per_mouse, data_arg='duration', figure_path=figure_path)
    # compare.plot_compare_dff(res, start_days_per_mouse, learned_day_per_mouse,
    #                          arg='all', valence='CS+', more_stats=False, figure_path=figure_path)

    # histogram.magnitude_histogram(res, learned_day_per_mouse, last_day_per_mouse, figure_path=figure_path)
    # power.plot_mean_dff(res, learned_day_per_mouse, last_day_per_mouse, figure_path=figure_path)
    # power.plot_max_dff(res, learned_day_per_mouse, last_day_per_mouse, figure_path=figure_path)
    # responsive.plot_responsive_difference_odor_and_water(res, start_days_per_mouse, training_start_day_per_mouse, last_day_per_mouse,
    #                                        figure_path=figure_path)
    # valence_responsive.plot_compare_responsive(res, figure_path)
    # valence_responsive.plot_responsive_difference_odor_and_water(res, start_days_per_mouse, last_day_per_mouse,
    #                                                              figure_path=figure_path, normalize=False, ylim=.65)

    # for d in [-1, 0, 1]:
    #     a = correlation.plot_correlation_matrix(res, start_days_per_mouse, loop_keys=['mouse'], odor_end=True,
    #                                             shuffle=False, figure_path = figure_path, direction=d)
    #     b = correlation.plot_correlation_matrix(res, last_day_per_mouse, loop_keys=['mouse'], odor_end=True,
    #                                             shuffle=False, figure_path = figure_path, direction=d)
    #
    # def _get_ixs(r):
    #     A = r['Odor_A']
    #     B = r['Odor_B']
    #     l = []
    #     for i, a in enumerate(A):
    #         b = B[i]
    #         if a < 2 and b > 1:
    #             l.append(i)
    #     return l
    # ixs = _get_ixs(a)
    # before = a['corrcoef'][ixs]
    # ixs = _get_ixs(b)
    # after = b['corrcoef'][ixs]
    #
    # x = wilcoxon(before, after)
    # print(x)
    # print(np.mean(before))
    # print(np.mean(after))

    # consistency.plot_consistency_within_day(res, start_days_per_mouse, last_day_per_mouse, shuffle=False, pretraining=False, figure_path = figure_path)

    # days = [
    #     [[3, 3, 2, 3, 3],[4,4,3,4,4]]
    #     ]
    # shuffle = False
    # out = correlation.plot_correlation_across_days(res, days, loop_keys=['mouse', 'odor'], shuffle=shuffle,
    #                                                 figure_path = figure_path,
    #                                                 reuse=False, save=False, analyze=True, plot_bool=False)
    # correlation.plot_correlation_across_days(out, days, loop_keys=['mouse', 'odor'], shuffle=shuffle,
    #                                          figure_path = figure_path,
    #                                          reuse=False, save=True, analyze=False, plot_bool=True)
    #
    # power.plot_power(res, start_days_per_mouse, last_day_per_mouse, figure_path, odor_valence=['CS+','CS-'],
    #                  colors_before = {'CS+':'Gray','CS-':'Gray'}, colors_after = {'CS+':'Green','CS-':'Red'},
    #                  excitatory=True, ylim = [-0.01, .1])
    #
    # power.plot_power(res, start_days_per_mouse, last_day_per_mouse, figure_path, odor_valence=['CS+','CS-'],
    #                  colors_before = {'CS+':'Gray','CS-':'Gray'}, colors_after = {'CS+':'Green','CS-':'Red'},
    #                  excitatory=False, ylim= [-.06, 0.01])


    # power.plot_max_dff_days(res, [start_days_per_mouse, last_day_per_mouse], ['CS+', 'CS+'], normalize= True,
    #                         save=False, reuse=False, day_pad= 0, figure_path = figure_path,
    #                         colors = ['green'], ylim=6)
    # power.plot_max_dff_days(res, [start_days_per_mouse, last_day_per_mouse], ['CS-', 'CS-'], normalize= True,
    #                         save=False, reuse=True, day_pad= 0, figure_path = figure_path,
    #                         colors = ['red'])
    # power.plot_bar(res, [start_days_per_mouse, last_day_per_mouse],
    #                ['CS+', 'CS+'], normalize = True,
    #                day_pad=0, save=False, reuse=True, figure_path=figure_path)
    # power.plot_bar(res, [start_days_per_mouse, last_day_per_mouse],
    #                ['CS-', 'CS-'], normalize = True,
    #                day_pad=0, save=True, reuse=True, figure_path=figure_path)

if condition.name == 'OFC_LONGTERM':
    #fully learned thres 70%
    res = statistics.analyze.analyze_data(save_path, condition_config, m_threshold= 0.04)
    # res.pop('data')
    # start_day = np.array([0,0,0,0])
    # learned_day_per_mouse = np.array([3, 2, 2, 3])
    # last_day_per_mouse = np.array([8, 7, 5, 5])

    res = filter.exclude(res, {'mouse': 3})
    temp_res = filter.exclude(temp_res, {'mouse': 3})
    start_day_per_mouse = [1,0,0]
    learned_day_per_mouse = np.array([3, 2, 2])
    last_day_per_mouse = np.array([8, 7, 7])

    # onset_learned = waveform.distribution(res, start=learned_day_per_mouse, end=learned_day_per_mouse + 1, data_arg='onset',
    #                       figure_path=figure_path, save = False)
    # onset_late = waveform.distribution(res, start=last_day_per_mouse - 1, end=last_day_per_mouse, data_arg='onset',
    #                       figure_path=figure_path, save = True)
    # print('ranksum between distributions of odor onsets early and late in learning {}'.format(
    #     ranksums(onset_learned,onset_late)[-1]))

    # waveform.behavior_vs_neural_onset(res, temp_res, learned_day_per_mouse, last_day_per_mouse, figure_path, behavior_arg='onset')
    # waveform.behavior_vs_neural_onset(res, temp_res, learned_day_per_mouse, last_day_per_mouse, figure_path, behavior_arg='magnitude')
    # waveform.behavior_vs_neural_onset(res, temp_res, learned_day_per_mouse, last_day_per_mouse, figure_path, behavior_arg='com')
    # #
    # waveform.behavior_vs_neural_power(res, temp_res, learned_day_per_mouse, last_day_per_mouse, figure_path, behavior_arg='magnitude')
    # waveform.behavior_vs_neural_power(res, temp_res, learned_day_per_mouse, last_day_per_mouse, figure_path, behavior_arg='onset')
    # waveform.behavior_vs_neural_power(res, temp_res, learned_day_per_mouse, last_day_per_mouse, figure_path, behavior_arg='com')
    #
    # excitatory = [True, False]
    # thresholds = [0.04, -0.04]
    # for i, sign in enumerate(excitatory):
    #     res = statistics.analyze.analyze_data(save_path, condition_config, m_threshold= thresholds[i], excitatory=sign)
    #     res = filter.exclude(res, {'mouse': 3})
    #     responsive.plot_summary_odor(res, start_day_per_mouse, last_day_per_mouse, figure_path=figure_path, excitatory=sign)
    #     responsive.plot_summary_odor(res, learned_day_per_mouse, last_day_per_mouse,
    #                                  figure_path=figure_path, excitatory=sign)

    # cory.main(res, temp_res, figure_path, excitatory=True)
    # cory.main(res, temp_res, figure_path, excitatory=False)

    # responsive.plot_individual(res, lick_res, figure_path= figure_path)
    # responsive.plot_summary_odor(res, learned_day_per_mouse, last_day_per_mouse, figure_path=figure_path)
    # responsive.plot_summary_odor_and_water(res, learned_day_per_mouse, learned_day_per_mouse, last_day_per_mouse,
    #                                        figure_path=figure_path, arg='odor_valence')
    # responsive.plot_responsive_difference_odor_and_water(res, learned_day_per_mouse, training_start_day_per_mouse, last_day_per_mouse,
    #                                        figure_path=figure_path, normalize=True)
    # valence_responsive.plot_compare_responsive(res, figure_path)
    # valence_responsive.plot_responsive_difference_odor_and_water(res, learned_day_per_mouse, last_day_per_mouse,
    #                                                              figure_path=figure_path, normalize=True, ylim=.5)
    # power.plot_max_dff_days(res, [learned_day_per_mouse, last_day_per_mouse], ['CS+', 'CS+'], save=True, reuse=False, day_pad=0,
    #                         ylim=.17, figure_path=figure_path)



    # power.plot_power(res, start_day, learned_day_per_mouse, figure_path, odor_valence=['CS+'],
    #                  colors_before = {'CS+':'Gray','CS-':'Gray'}, colors_after = {'CS+':'Green','CS-':'Red'}, ylim=.05)
    # power.plot_power(res, start_days_per_mouse, last_day_per_mouse, figure_path, odor_valence=['CS+'],
    #                  colors_before = {'CS+':'Gray','CS-':'Gray'}, colors_after = {'CS+':'Green','CS-':'Red'}, ylim=.05)
    # power.plot_power(res, start_days_per_mouse, learned_day_per_mouse, figure_path, odor_valence=['CS-'],
    #                  colors_before = {'CS+':'Gray','CS-':'Gray'}, colors_after = {'CS+':'Green','CS-':'Red'}, ylim=.05)
    # power.plot_power(res, start_days_per_mouse, last_day_per_mouse, figure_path, odor_valence=['CS-'],
    #                  colors_before = {'CS+':'Gray','CS-':'Gray'}, colors_after = {'CS+':'Green','CS-':'Red'}, ylim=.05)

    signs = [True, False]
    ylims = [[-.005, 0.06], [-.05, 0.01]]
    for sign, ylim in zip(signs, ylims):
        power.plot_power(res, learned_day_per_mouse, start_day_per_mouse, figure_path, odor_valence=['CS+'],
                         colors_before = {'CS+':'greenyellow'}, colors_after = {'CS+':'Gray'}, ylim=ylim, excitatory=sign)
        power.plot_power(res, learned_day_per_mouse, start_days_per_mouse, figure_path, odor_valence=['CS-'],
                         colors_before = {'CS-':'Red'}, colors_after = {'CS-':'Gray'}, ylim=ylim, excitatory=sign)

        power.plot_power(res, learned_day_per_mouse, last_day_per_mouse, figure_path, odor_valence=['CS+'],
                         colors_before = {'CS+':'greenyellow'}, colors_after = {'CS+':'darkgreen'}, ylim=ylim, excitatory=sign)
        power.plot_power(res, learned_day_per_mouse, last_day_per_mouse, figure_path, odor_valence=['CS-'],
                         colors_before = {'CS-':'mistyrose'}, colors_after = {'CS-':'darkred'}, ylim=ylim, excitatory=sign)

        power.plot_power(res, start_day_per_mouse, last_day_per_mouse, figure_path, odor_valence=['CS+'],
                         colors_before = {'CS+':'Gray'}, colors_after = {'CS+':'darkgreen'}, ylim=ylim, excitatory=sign)
        power.plot_power(res, start_day_per_mouse, last_day_per_mouse, figure_path, odor_valence=['CS-'],
                         colors_before = {'CS-':'Gray'}, colors_after = {'CS-':'darkred'}, ylim=ylim, excitatory=sign)