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
import statistics.count_methods.power as power
import statistics.count_methods.compare as compare
import statistics.count_methods.correlation as correlation
import statistics.count_methods.cory as cory
from scipy.stats import ranksums, wilcoxon, kruskal

condition_config = statistics.analyze.MPFC_COMPOSITE_Config()
condition = condition_config.condition
data_path = os.path.join(Config.LOCAL_DATA_PATH, Config.LOCAL_DATA_TIMEPOINT_FOLDER, condition.name)
save_path = os.path.join(Config.LOCAL_EXPERIMENT_PATH, 'COUNTING', condition.name)
figure_path = os.path.join(Config.LOCAL_FIGURE_PATH, 'OTHER', 'COUNTING',  condition.name)

#analysis
res = fio.load_pickle(os.path.join(save_path, 'dict.pkl'))
# res.pop('data')
learned_day_per_mouse, last_day_per_mouse = get_days_per_mouse(data_path, condition)

if condition_config.start_at_training and hasattr(condition, 'training_start_day'):
    start_days_per_mouse = condition.training_start_day
else:
    start_days_per_mouse = [0] * len(np.unique(res['mouse']))
training_start_day_per_mouse = condition.training_start_day

lick_res = behavior.behavior_analysis.get_licks_per_day(data_path, condition)
analysis.add_odor_value(lick_res, condition)
lick_res = filter.filter(lick_res, {'odor_valence': ['CS+', 'CS-', 'PT CS+']})
lick_res = reduce.new_filter_reduce(lick_res, ['odor_valence', 'day', 'mouse'], reduce_key='lick_boolean')

temp_res = behavior.behavior_analysis.analyze_behavior(data_path, condition)
temp_res = filter.filter(temp_res, {'odor_valence': ['CS+']})
temp_res = reduce.new_filter_reduce(temp_res, ['odor_valence', 'mouse'], reduce_key='boolean_smoothed')

if condition.name == 'OFC_COMPOSITE':
    statistics.analyze.analyze_data(res, condition_config, m_threshold = 0.04)

    pt_start = [1, 1, 1, 1]
    pt_learned = [4, 4, 4, 3]
    dt_naive = [0, 0, 0, 0]
    dt_start = [4, 4, 6, 4]
    dt_learned = [5, 5, 9, 5]
    dt_end = [8, 9, 10, 8]
    dt_last = [x for x in last_day_per_mouse]
    dt_res = filter.filter(res, filter_dict={'odor_valence': ['CS+', 'CS-']})
    pt_res = filter.filter(res, filter_dict={'odor_valence':['PT CS+','PT Naive']})

    # valence_responsive.plot_compare_responsive(dt_res, figure_path)
    # valence_responsive.plot_responsive_difference_odor_and_water(res, dt_naive, dt_learned,
    #                                                              figure_path=figure_path, normalize=False, ylim=.3)

    # overlap.plot_overlap_odor(dt_res, dt_naive, dt_start, figure_path = figure_path)
    # overlap.plot_overlap_odor(dt_res, dt_start, dt_learned, figure_path = figure_path)

    # consistency.plot_consistency_within_day(res, pt_start, pt_learned, shuffle=False, pretraining=True,
    #                                         figure_path=figure_path)
    # consistency.plot_consistency_within_day(res, dt_start, dt_learned, shuffle=False, pretraining=False,
    #                                         figure_path=figure_path)

    # responsive.plot_summary_odor_pretraining(res, pt_start, pt_learned, arg_naive=True, figure_path = figure_path, save=False)
    # responsive.plot_summary_odor(res, dt_naive, dt_end, figure_path=figure_path, reuse=True)
    # responsive.plot_responsive_difference_odor_and_water(res, dt_naive, None, dt_end,
    #                                                      pt_start= pt_start, pt_learned= pt_learned,
    #                                                      include_water=False, figure_path=figure_path)
    # power.plot_max_dff_days(res, [dt_naive], ['CS+'], save=False, reuse=False, day_pad= 0, figure_path = figure_path)

    # power.plot_max_dff_days(res, [dt_naive], ['CS-'],
    #                         save=False, reuse=False, day_pad=0, figure_path = figure_path)
    # power.plot_max_dff_days(res, [dt_naive], ['CS+'],
    #                         save=False, reuse=True, day_pad= 0, figure_path = figure_path)

    # power.plot_max_dff_days(res, [pt_start, pt_learned], ['PT CS+', 'PT CS+'], save=False, reuse=False, ylim=.17, day_pad= 1, figure_path = figure_path)
    # power.plot_max_dff_days(res, [pt_learned, dt_start], ['PT CS+', 'CS+'], save=False, reuse=True, day_pad= 2, figure_path = figure_path,
    #                         colors = ['black'])
    # power.plot_max_dff_days(res, [dt_start, dt_learned, dt_end], ['CS+','CS+','CS+'],
    #                         save=False, reuse=True, day_pad= 3, figure_path = figure_path)
    # power.plot_bar(res, [pt_start, pt_learned, dt_start, dt_learned, dt_end],
    #                ['PT CS+', 'PT CS+', 'CS+', 'CS+', 'CS+'],
    #                day_pad=1, save=True, reuse=True, figure_path=figure_path)
    #
    # power.plot_max_dff_days(res, [pt_start, pt_learned], ['PT CS+', 'PT CS+'], save=False, reuse=False, day_pad= 1, ylim=.17, figure_path = figure_path)
    # power.plot_max_dff_days(res, [pt_learned, dt_start], ['PT CS+', 'CS-'], save=False, reuse=True, day_pad= 2, figure_path = figure_path,
    #                         colors = ['black'])
    # power.plot_max_dff_days(res, [dt_start, dt_learned, dt_end], ['CS-','CS-','CS-'],
    #                         save=False, reuse=True, day_pad= 3, figure_path = figure_path)
    # power.plot_bar(res, [pt_start, pt_learned, dt_start, dt_learned, dt_end],
    #                ['PT CS+', 'PT CS+', 'CS-', 'CS-', 'CS-'],
    #                day_pad=1, save=True, reuse=True, figure_path=figure_path)

    # power.plot_power(pt_res, pt_start, pt_learned, figure_path, odor_valence=['PT CS+'], naive=False,
    #                  colors_before = {'PT CS+':'Gray'}, colors_after = {'PT CS+':'Orange'})
    # power.plot_power(res, dt_naive, dt_learned, figure_path, odor_valence=['CS+'],
    #                  colors_before = {'CS+':'Gray'}, colors_after = {'CS+':'Green'})
    # power.plot_power(res, dt_naive, dt_start, figure_path, odor_valence=['CS+'],
    #                  colors_before = {'CS+':'Gray'}, colors_after = {'CS+':'Green'})
    # power.plot_power(res, dt_naive, dt_end, figure_path, odor_valence=['CS+'],
    #                  colors_before = {'CS+':'Gray'}, colors_after = {'CS+':'Green'})
    # power.plot_power(res, dt_naive, dt_learned, figure_path, odor_valence=['CS-'],
    #                  colors_before = {'CS-':'Gray'}, colors_after = {'CS+':'Green', 'CS-':'Red'})
    # power.plot_power(res, dt_naive, dt_start, figure_path, odor_valence=['CS-'],
    #                  colors_before = {'CS-':'Gray'}, colors_after = {'CS+':'Green', 'CS-':'Red'})
    # power.plot_power(res, dt_naive, dt_end, figure_path, odor_valence=['CS-'],
    #                  colors_before = {'CS-':'Gray'}, colors_after = {'CS+':'Green', 'CS-':'Red'})
    # power.plot_power(res, dt_start, dt_end, figure_path, odor_valence=['CS+'],
    #                  colors_before = {'CS+':'Green'}, colors_after = {'CS+':'Black'})
    # power.plot_power(res, dt_start, dt_end, figure_path, odor_valence=['CS-'],
    #                  colors_before = {'CS-':'Red'}, colors_after = {'CS-':'Black'})

    # a = correlation.plot_correlation_matrix(dt_res, dt_naive, loop_keys=['mouse'], shuffle=False, figure_path = figure_path)
    # b = correlation.plot_correlation_matrix(dt_res, dt_start, loop_keys=['mouse'], shuffle=False, figure_path = figure_path)
    # c = correlation.plot_correlation_matrix(dt_res, dt_learned, loop_keys=['mouse'], shuffle=False, figure_path = figure_path)
    # d = correlation.plot_correlation_matrix(dt_res, dt_end, loop_keys=['mouse'], shuffle=False, figure_path = figure_path)

    # def _get_ixs(r):
    #     A = r['Odor_A']
    #     B = r['Odor_B']
    #     l = []
    #     for i, a in enumerate(A):
    #         b = B[i]
    #         if a < 2 and b > 1:
    #             l.append(i)
    #     return l
    # ixs = _get_ixs(b)
    # before = b['corrcoef'][ixs]
    # ixs = _get_ixs(c)
    # after = c['corrcoef'][ixs]
    # print(ranksums(before,after))
    # print(np.mean(before))
    # print(np.mean(after))

if condition.name == 'MPFC_COMPOSITE':
    statistics.analyze.analyze_data(res, condition_config, m_threshold=0.03)
    pt_start = [1,1,1,1]
    pt_learned = [3, 3, 3, 3]
    dt_naive = [0, 0, 0, 0]
    dt_start = [3, 3, 4, 4]
    dt_learned = [4, 4, 5, 5]
    dt_end = [8, 8, 5, 8]
    dt_res = filter.filter(res, filter_dict={'odor_valence':['CS+','CS-']})


    # valence_responsive.plot_compare_responsive(dt_res, figure_path)
    # valence_responsive.plot_responsive_difference_odor_and_water(res, dt_naive, dt_end,
    #                                                              figure_path=figure_path, normalize=False, ylim=.3)

    # responsive.plot_summary_odor_and_water(res, start_days_per_mouse, training_start_day_per_mouse, last_day_per_mouse,
    #                                                                               figure_path=figure_path)

    # responsive.plot_summary_odor_pretraining(res, pt_start, pt_learned, arg_naive=False, figure_path = figure_path, save=False)
    # responsive.plot_summary_odor(res, dt_naive, dt_end, figure_path=figure_path, reuse=True)
    # responsive.plot_summary_odor(res, dt_learned, dt_end, figure_path=figure_path)
    # overlap.plot_overlap_odor(dt_res, dt_naive, dt_start, figure_path = figure_path)
    # overlap.plot_overlap_odor(dt_res, dt_start, dt_last, figure_path = figure_path)
    # responsive.plot_summary_water(res, dt_start, dt_learned, figure_path=figure_path)
    # responsive.plot_responsive_difference_odor_and_water(res, dt_naive, None, dt_end,
    #                                                      pt_start= pt_start, pt_learned= pt_learned,
    #                                                      ylim=.3,
    #                                                      include_water=False, figure_path=figure_path)

    # consistency.plot_consistency_within_day(res, pt_start, pt_learned, shuffle=False, pretraining=True,
    #                                         figure_path=figure_path)

    # power.plot_max_dff_days(res, [dt_naive], ['CS-'],
    #                         save=False, reuse=False, day_pad=0, figure_path = figure_path)
    # power.plot_max_dff_days(res, [dt_naive], ['CS+'],
    #                         save=False, reuse=True, day_pad= 0, figure_path = figure_path)

    # power.plot_max_dff_days(res, [pt_start, pt_learned], ['PT CS+', 'PT CS+'], save=False, reuse=False, day_pad= 1, ylim=.12, figure_path = figure_path)
    # power.plot_max_dff_days(res, [pt_learned, dt_start], ['PT CS+', 'CS+'], save=False, reuse=True, day_pad= 2, figure_path = figure_path,
    #                         colors = ['black'])
    # power.plot_max_dff_days(res, [dt_start, dt_learned, dt_end], ['CS+','CS+','CS+'],
    #                         save=False, reuse=True, day_pad= 3, figure_path = figure_path)
    # power.plot_bar(res, [pt_start, pt_learned, dt_start, dt_learned, dt_end],
    #                ['PT CS+', 'PT CS+', 'CS+', 'CS+', 'CS+'],
    #                day_pad=1, save=True, reuse=True, figure_path=figure_path)
    #
    # power.plot_max_dff_days(res, [pt_start, pt_learned], ['PT CS+', 'PT CS+'], save=False, reuse=False, day_pad= 1, ylim=.12, figure_path = figure_path)
    # power.plot_max_dff_days(res, [pt_learned, dt_start], ['PT CS+', 'CS-'], save=False, reuse=True, day_pad= 2, figure_path = figure_path,
    #                         colors = ['black'])
    # power.plot_max_dff_days(res, [dt_start, dt_learned, dt_end], ['CS-','CS-','CS-'],
    #                         save=False, reuse=True, day_pad= 3, figure_path = figure_path)
    # power.plot_bar(res, [pt_start, pt_learned, dt_start, dt_learned, dt_end],
    #                ['PT CS+', 'PT CS+', 'CS-', 'CS-', 'CS-'],
    #                day_pad=1, save=True, reuse=True, figure_path=figure_path)

    # power.plot_power(res, pt_start, pt_learned, figure_path, odor_valence=['PT CS+'],
    #                  colors_before = {'PT CS+':'Gray'}, colors_after = {'PT CS+':'Orange'})
    # power.plot_power(res, dt_naive, dt_start, figure_path, odor_valence=['CS+'],
    #                  colors_before = {'CS+':'Gray'}, colors_after = {'CS+':'Green'})
    # power.plot_power(res, dt_naive, dt_learned, figure_path, odor_valence=['CS+'],
    #                  colors_before = {'CS+':'Gray'}, colors_after = {'CS+':'Green'})
    # power.plot_power(res, dt_naive, dt_end, figure_path, odor_valence=['CS+'],
    #                  colors_before = {'CS+':'Gray'}, colors_after = {'CS+':'Green'})
    # power.plot_power(res, dt_naive, dt_learned, figure_path, odor_valence=['CS-'],
    #                  colors_before = {'CS-':'Gray'}, colors_after = {'CS+':'Green', 'CS-':'Red'})
    # power.plot_power(res, dt_naive, dt_start, figure_path, odor_valence=['CS-'],
    #                  colors_before = {'CS-':'Gray'}, colors_after = {'CS+':'Green', 'CS-':'Red'})
    # power.plot_power(res, dt_naive, dt_end, figure_path, odor_valence=['CS-'],
    #                  colors_before = {'CS-':'Gray'}, colors_after = {'CS+':'Green', 'CS-':'Red'})
    # power.plot_power(res, dt_learned, dt_end, figure_path, odor_valence=['CS+'],
    #                  colors_before = {'CS+':'Green'}, colors_after = {'CS+':'Black'})

    a = correlation.plot_correlation_matrix(dt_res, dt_naive, loop_keys=['mouse'], shuffle=False, figure_path = figure_path)
    b = correlation.plot_correlation_matrix(dt_res, dt_start, loop_keys=['mouse'], shuffle=False, figure_path = figure_path)
    c = correlation.plot_correlation_matrix(dt_res, dt_learned, loop_keys=['mouse'], shuffle=False, figure_path = figure_path)
    d = correlation.plot_correlation_matrix(dt_res, dt_end, loop_keys=['mouse'], shuffle=False, figure_path = figure_path)
    #
    def _get_ixs(r):
        A = r['Odor_A']
        B = r['Odor_B']
        l = []
        for i, a in enumerate(A):
            b = B[i]
            if a != b:
                l.append(i)
        return l

    testX = a
    testY = b
    ixs = _get_ixs(testX)
    before = testX['corrcoef'][ixs]
    ixs = _get_ixs(testY)
    after = testY['corrcoef'][ixs]
    print(wilcoxon(before,after))
    print(np.mean(before))
    print(np.mean(after))



