import os

from _CONSTANTS.config import Config
import filter
import reduce
import numpy as np
import tools.file_io as fio
from behavior.behavior_analysis import get_days_per_mouse
import behavior
import analysis

import psth.count_analyze
import psth.count_methods.overlap as overlap
import psth.count_methods.stability as stability
import psth.count_methods.responsive as responsive
import psth.count_methods.waveform as waveform
import psth.count_methods.reversal as reversal
import psth.count_methods.power as power
import psth.count_methods.compare as compare
import psth.count_methods.histogram as histogram
import psth.count_methods.valence_responsive as valence_responsive

condition_config = psth.count_analyze.OFC_Reversal_Config()

config = psth.psth_helper.PSTHConfig()
condition = condition_config.condition
data_path = os.path.join(Config.LOCAL_DATA_PATH, Config.LOCAL_DATA_TIMEPOINT_FOLDER, condition.name)
save_path = os.path.join(Config.LOCAL_EXPERIMENT_PATH, 'COUNTING', condition.name)
figure_path = os.path.join(Config.LOCAL_FIGURE_PATH, 'OTHER', 'COUNTING',  condition.name)

#analysis
res = fio.load_pickle(os.path.join(save_path, 'dict.pkl'))

# retrieving relevant days
learned_day_per_mouse, last_day_per_mouse = get_days_per_mouse(data_path, condition)

if condition_config.start_at_training and hasattr(condition, 'training_start_day'):
    start_days_per_mouse = condition.training_start_day
else:
    start_days_per_mouse = [0] * len(np.unique(res['mouse']))
training_start_day_per_mouse = condition.training_start_day

print(start_days_per_mouse)
print(last_day_per_mouse)

lick_res = behavior.behavior_analysis.get_licks_per_day(data_path, condition)
analysis.add_odor_value(lick_res, condition)
lick_res = filter.filter(lick_res, {'odor_valence': ['CS+', 'CS-', 'PT CS+']})
lick_res = reduce.new_filter_reduce(lick_res, ['odor_valence', 'day', 'mouse'], reduce_key='lick_boolean')

if condition.name == 'PIR':
    psth.count_analyze.analyze_data(res, condition_config)
    # responsive.plot_individual(res, lick_res, figure_path = figure_path)
    responsive.plot_summary_odor_and_water(res, start_days_per_mouse, training_start_day_per_mouse, learned_day_per_mouse,
                                           figure_path=figure_path)
    responsive.plot_responsive_difference_odor_and_water(res, start_days_per_mouse, training_start_day_per_mouse, last_day_per_mouse,
                                           figure_path=figure_path, average=False)
    # overlap.plot_overlap_odor(res, start_days_per_mouse, learned_day_per_mouse,
    #                                              delete_non_selective=True, figure_path= figure_path)
    # stability.plot_stability_across_days(res, start_days_per_mouse, learned_day_per_mouse, figure_path = figure_path)
    # waveform.distribution(res, start=learned_day_per_mouse, end=last_day_per_mouse, data_arg='onset', figure_path=figure_path)
    # waveform.distribution(res, start=learned_day_per_mouse, end=last_day_per_mouse, data_arg='amplitude', figure_path=figure_path)
    # waveform.distribution(res, start=learned_day_per_mouse, end=last_day_per_mouse, data_arg='duration', figure_path=figure_path)


if condition.name == 'PIR_NAIVE':
    days = [3, 3, 2, 2]
    psth.count_analyze.analyze_data(res, condition_config)
    responsive.plot_summary_odor(res, start_days_per_mouse, days, use_colors=False, figure_path = figure_path)
    responsive.plot_responsive_difference_odor_and_water(res, start_days_per_mouse, training_start_day_per_mouse, last_day_per_mouse,
                                           figure_path=figure_path, average=False)
    # overlap.plot_overlap_odor(res, start_days_per_mouse, days,
    #                                              delete_non_selective=True, figure_path= figure_path)
    # stability.plot_stability_across_days(res, start_days_per_mouse, learned_day_per_mouse, figure_path = figure_path)

if condition.name == 'OFC' or condition.name == 'BLA':
    if condition.name == 'OFC':
        last_day_per_mouse = [5, 5, 3, 4, 3]
    psth.count_analyze.analyze_data(res, condition_config, m_threshold=0.05)
    # responsive.plot_individual(res, lick_res, figure_path=figure_path)
    # responsive.plot_summary_odor_and_water(res, start_days_per_mouse, training_start_day_per_mouse, last_day_per_mouse,
    #                                        figure_path=figure_path)
    # overlap.plot_overlap_odor(res, start_days_per_mouse, last_day_per_mouse, figure_path = figure_path)
    # overlap.plot_overlap_water(res, training_start_day_per_mouse, last_day_per_mouse, figure_path = figure_path)

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
    valence_responsive.plot_responsive_difference_odor_and_water(res, start_days_per_mouse, last_day_per_mouse,
                                                                 figure_path=figure_path, normalize=False, ylim=.65)

if condition.name == 'OFC_REVERSAL':
    psth.count_analyze.analyze_data(res, condition_config, m_threshold= 0.05)
    # responsive.plot_summary_odor(res, start_days_per_mouse, last_day_per_mouse, figure_path= figure_path)
    # compare.plot_compare_dff(res, start_days_per_mouse, last_day_per_mouse,
    #                                             arg='first', valence='CS+', more_stats=False, figure_path= figure_path)
    # compare.plot_compare_dff(res, start_days_per_mouse, last_day_per_mouse,
    #                                             arg='last', valence='CS-', more_stats=False, figure_path= figure_path)
    reversal.plot_reversal(res, start_days_per_mouse, last_day_per_mouse, figure_path= figure_path)

if condition.name == 'OFC_STATE':
    psth.count_analyze.analyze_data(res, condition_config, m_threshold=.05)
    # responsive.plot_summary_odor(res, start_days_per_mouse, last_day_per_mouse, figure_path)
    power.plot_power(res, start_days_per_mouse, last_day_per_mouse, figure_path)
    # compare.plot_compare_dff(res, start_days_per_mouse, last_day_per_mouse,
    #                                             arg='first', valence='CS+', more_stats=True, figure_path= figure_path)
    start = [0,0,0,0,0]
    end = [1,1,1,1,1]
    # valence_responsive.plot_responsive_difference_odor_and_water(res, start, end,
    #                                                              figure_path=figure_path, normalize=False, ylim=.4)
    # power.plot_max_dff_days(res, [start, end], ['CS+', 'CS+'], save=False, reuse=False, day_pad= 0, figure_path = figure_path, ylim=.2)
    # power.plot_max_dff_days(res, [start, end], ['CS-','CS-'],save=False, reuse=True, day_pad= 0, figure_path = figure_path, ylim=.2)
    # power.plot_bar(res, [start, end], ['CS-', 'CS-'], color='darkred',
    #                day_pad=0, save=False, reuse=True, figure_path=figure_path)
    # power.plot_bar(res, [start, end], ['CS+', 'CS+'], color='darkgreen',
    #                day_pad=0, save=True, reuse=True, figure_path=figure_path)

if condition.name == 'OFC_CONTEXT':
    psth.count_analyze.analyze_data(res, condition_config, m_threshold=.05)
    # responsive.plot_summary_odor(res, start_days_per_mouse, last_day_per_mouse, figure_path)
    power.plot_power(res, start_days_per_mouse, last_day_per_mouse, figure_path)
    # compare.plot_compare_dff(res, start_days_per_mouse, last_day_per_mouse,
    #                                             arg='first', valence='CS+', more_stats=True, figure_path= figure_path)
    start = [0,0,0,0]
    end = [1,1,1,1]
    # valence_responsive.plot_responsive_difference_odor_and_water(res, [0,0,0,0], [1,1,1,1],
    #                                                              figure_path=figure_path, normalize=False, ylim=.65)
    # power.plot_max_dff_days(res, [start, end], ['CS+', 'CS+'], save=False, reuse=False, day_pad= 0, figure_path = figure_path, ylim=.2)
    # power.plot_max_dff_days(res, [start, end], ['CS-','CS-'],save=False, reuse=True, day_pad= 0, figure_path = figure_path, ylim=.2)
    # power.plot_bar(res, [start, end], ['CS-', 'CS-'], color='darkred',
    #                day_pad=0, save=False, reuse=True, figure_path=figure_path)
    # power.plot_bar(res, [start, end], ['CS+', 'CS+'], color='darkgreen',
    #                day_pad=0, save=True, reuse=True, figure_path=figure_path)

if condition.name == 'OFC_LONGTERM':
    #fully learned thres 70%
    psth.count_analyze.analyze_data(res, condition_config, m_threshold= 0.05)
    learned_day_per_mouse = np.array([3, 4, 2, 3])
    last_day_per_mouse = np.array([8, 7, 5, 5])
    # responsive.plot_individual(res, lick_res, figure_path= figure_path)
    # responsive.plot_summary_odor_and_water(res, learned_day_per_mouse, learned_day_per_mouse, last_day_per_mouse,
    #                                        figure_path=figure_path, arg='odor_valence')
    responsive.plot_responsive_difference_odor_and_water(res, learned_day_per_mouse, training_start_day_per_mouse, last_day_per_mouse,
                                           figure_path=figure_path, normalize=True)
    valence_responsive.plot_compare_responsive(res, figure_path)
    valence_responsive.plot_responsive_difference_odor_and_water(res, learned_day_per_mouse, last_day_per_mouse,
                                                                 figure_path=figure_path, normalize=True, ylim=.5)

if condition.name == 'OFC_COMPOSITE':
    psth.count_analyze.analyze_data(res, condition_config, m_threshold = 0.05)
    pt_start = [1, 1, 1, 1]
    pt_learned = [4, 4, 4, 3]
    dt_naive = [0, 0, 0, 0]
    dt_start = [4, 4, 6, 4]
    dt_learned = [5, 5, 9, 5]
    dt_end = [8, 9, 10, 8]
    dt_last = [x for x in last_day_per_mouse]
    dt_res = filter.filter(res, filter_dict={'odor_valence': ['CS+', 'CS-']})
    # valence_responsive.plot_compare_responsive(dt_res, figure_path)
    # valence_responsive.plot_responsive_difference_odor_and_water(res, dt_naive, dt_learned,
    #                                                              figure_path=figure_path, normalize=False, ylim=.3)

    # overlap.plot_overlap_odor(dt_res, dt_naive, dt_start, figure_path = figure_path)
    # overlap.plot_overlap_odor(dt_res, dt_start, dt_learned, figure_path = figure_path)

    # responsive.plot_summary_odor_pretraining(res, pt_start, pt_learned, arg_naive=True, figure_path = figure_path, save=False)
    # responsive.plot_summary_odor(res, dt_naive, dt_learned, figure_path=figure_path, reuse=True)
    # responsive.plot_responsive_difference_odor_and_water(res, dt_naive, None, dt_end,
    #                                                      pt_start= pt_start, pt_learned= pt_learned,
    #                                                      include_water=False, figure_path=figure_path)
    # power.plot_max_dff_days(res, [dt_naive], ['CS+'], save=False, reuse=False, day_pad= 0, figure_path = figure_path)

    # power.plot_max_dff_days(res, [dt_naive], ['CS-'],
    #                         save=False, reuse=False, day_pad=0, figure_path = figure_path)
    # power.plot_max_dff_days(res, [dt_naive], ['CS+'],
    #                         save=False, reuse=True, day_pad= 0, figure_path = figure_path)

    power.plot_max_dff_days(res, [pt_start, pt_learned], ['PT CS+', 'PT CS+'], save=False, reuse=False, ylim=.17, day_pad= 1, figure_path = figure_path)
    power.plot_max_dff_days(res, [pt_learned, dt_start], ['PT CS+', 'CS+'], save=False, reuse=True, day_pad= 2, figure_path = figure_path,
                            colors = ['black'])
    power.plot_max_dff_days(res, [dt_start, dt_learned, dt_end], ['CS+','CS+','CS+'],
                            save=False, reuse=True, day_pad= 3, figure_path = figure_path)
    power.plot_bar(res, [pt_start, pt_learned, dt_start, dt_learned, dt_end],
                   ['PT CS+', 'PT CS+', 'CS+', 'CS+', 'CS+'],
                   day_pad=1, save=True, reuse=True, figure_path=figure_path)

    power.plot_max_dff_days(res, [pt_start, pt_learned], ['PT CS+', 'PT CS+'], save=False, reuse=False, day_pad= 1, ylim=.17, figure_path = figure_path)
    power.plot_max_dff_days(res, [pt_learned, dt_start], ['PT CS+', 'CS-'], save=False, reuse=True, day_pad= 2, figure_path = figure_path,
                            colors = ['black'])
    power.plot_max_dff_days(res, [dt_start, dt_learned, dt_end], ['CS-','CS-','CS-'],
                            save=False, reuse=True, day_pad= 3, figure_path = figure_path)
    power.plot_bar(res, [pt_start, pt_learned, dt_start, dt_learned, dt_end],
                   ['PT CS+', 'PT CS+', 'CS-', 'CS-', 'CS-'],
                   day_pad=1, save=True, reuse=True, figure_path=figure_path)


if condition.name == 'MPFC_COMPOSITE':
    psth.count_analyze.analyze_data(res, condition_config, m_threshold=0.05)
    pt_start = [1,1,1,1]
    pt_learned = [3, 3, 3, 3]
    dt_naive = [0, 0, 0, 0]
    dt_start = [3, 3, 4, 4]
    dt_learned = [4, 4, 5, 5]
    dt_end = [8, 8, 5, 8]
    dt_last = [x for x in last_day_per_mouse]
    dt_res = filter.filter(res, filter_dict={'odor_valence':['CS+','CS-']})
    # valence_responsive.plot_compare_responsive(dt_res, figure_path)
    # valence_responsive.plot_responsive_difference_odor_and_water(res, dt_naive, dt_end,
    #                                                              figure_path=figure_path, normalize=False, ylim=.3)


    # responsive.plot_summary_odor_pretraining(res, pt_start, pt_learned, arg_naive=False, figure_path = figure_path, save=False)
    # responsive.plot_summary_odor(res, dt_naive, dt_learned, figure_path=figure_path, reuse=True)
    # responsive.plot_summary_odor(res, dt_learned, dt_end, figure_path=figure_path)
    # overlap.plot_overlap_odor(dt_res, dt_naive, dt_start, figure_path = figure_path)
    # overlap.plot_overlap_odor(dt_res, dt_start, dt_last, figure_path = figure_path)
    # responsive.plot_summary_water(res, dt_start, dt_learned, figure_path=figure_path)
    # responsive.plot_responsive_difference_odor_and_water(res, dt_naive, None, dt_end,
    #                                                      pt_start= pt_start, pt_learned= pt_learned,
    #                                                      ylim=.3,
    #                                                      include_water=False, figure_path=figure_path)

    # power.plot_max_dff_days(res, [dt_naive], ['CS-'],
    #                         save=False, reuse=False, day_pad=0, figure_path = figure_path)
    # power.plot_max_dff_days(res, [dt_naive], ['CS+'],
    #                         save=False, reuse=True, day_pad= 0, figure_path = figure_path)

    power.plot_max_dff_days(res, [pt_start, pt_learned], ['PT CS+', 'PT CS+'], save=False, reuse=False, day_pad= 1, ylim=.12, figure_path = figure_path)
    power.plot_max_dff_days(res, [pt_learned, dt_start], ['PT CS+', 'CS+'], save=False, reuse=True, day_pad= 2, figure_path = figure_path,
                            colors = ['black'])
    power.plot_max_dff_days(res, [dt_start, dt_learned, dt_end], ['CS+','CS+','CS+'],
                            save=False, reuse=True, day_pad= 3, figure_path = figure_path)
    power.plot_bar(res, [pt_start, pt_learned, dt_start, dt_learned, dt_end],
                   ['PT CS+', 'PT CS+', 'CS+', 'CS+', 'CS+'],
                   day_pad=1, save=True, reuse=True, figure_path=figure_path)

    power.plot_max_dff_days(res, [pt_start, pt_learned], ['PT CS+', 'PT CS+'], save=False, reuse=False, day_pad= 1, ylim=.12, figure_path = figure_path)
    power.plot_max_dff_days(res, [pt_learned, dt_start], ['PT CS+', 'CS-'], save=False, reuse=True, day_pad= 2, figure_path = figure_path,
                            colors = ['black'])
    power.plot_max_dff_days(res, [dt_start, dt_learned, dt_end], ['CS-','CS-','CS-'],
                            save=False, reuse=True, day_pad= 3, figure_path = figure_path)
    power.plot_bar(res, [pt_start, pt_learned, dt_start, dt_learned, dt_end],
                   ['PT CS+', 'PT CS+', 'CS-', 'CS-', 'CS-'],
                   day_pad=1, save=True, reuse=True, figure_path=figure_path)


# if condition.name == 'BLA_LONGTERM':
#     psth.count_analyze.analyze_data(res, condition_config)
#     plot_individual(res, lick_res, figure_path)
#     plot_summary_odor(res, start_days_per_mouse, learned_day_per_mouse, figure_path)
#     plot_overlap_odor(res, start_days_per_mouse, learned_day_per_mouse, figure_path)
#
#     dt_start = [x+1 for x in condition.last_pt_day]
#     dt_end = [x+1 for x in learned_day_per_mouse]
#     dt_last = [x for x in last_day_per_mouse]
#     plot_summary_odor(res, [0, 0, 0, 0], dt_end, figure_path)



