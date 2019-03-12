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

condition_config = psth.count_analyze.OFC_Context_Config()

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
    # responsive.plot_summary_odor_and_water(res, start_days_per_mouse, training_start_day_per_mouse, learned_day_per_mouse,
    #                                        figure_path=figure_path)
    # overlap.plot_overlap_odor(res, start_days_per_mouse, learned_day_per_mouse,
    #                                              delete_non_selective=True, figure_path= figure_path)
    # stability.plot_stability_across_days(res, start_days_per_mouse, learned_day_per_mouse, figure_path = figure_path)
    # waveform.distribution(res, start=learned_day_per_mouse, end=last_day_per_mouse, data_arg='onset', figure_path=figure_path)
    # waveform.distribution(res, start=learned_day_per_mouse, end=last_day_per_mouse, data_arg='amplitude', figure_path=figure_path)
    # waveform.distribution(res, start=learned_day_per_mouse, end=last_day_per_mouse, data_arg='duration', figure_path=figure_path)
    compare.plot_compare_dff(res, [1]*6, [2]*6,
                                                arg='all', valence='CS+', more_stats=False, figure_path= figure_path)
    compare.plot_compare_dff(res, [1]*6, [2]*6,
                                                arg='all', valence='CS-', more_stats=False, figure_path= figure_path)


if condition.name == 'PIR_NAIVE':
    days = [3, 3, 2, 2]
    psth.count_analyze.analyze_data(res, condition_config)
    # responsive.plot_summary_odor(res, start_days_per_mouse, days, use_colors=False, figure_path = figure_path)
    # overlap.plot_overlap_odor(res, start_days_per_mouse, days,
    #                                              delete_non_selective=True, figure_path= figure_path)
    # stability.plot_stability_across_days(res, start_days_per_mouse, learned_day_per_mouse, figure_path = figure_path)
    compare.plot_compare_dff(res, [1]*4, [2]*4,
                                                arg='all', valence='CS+', more_stats=False, figure_path= figure_path)
    compare.plot_compare_dff(res, [1]*4, [2]*4,
                                                arg='all', valence='CS-', more_stats=False, figure_path= figure_path)

if condition.name == 'OFC' or condition.name == 'BLA':
    psth.count_analyze.analyze_data(res, condition_config, m_threshold=0.03)
    # responsive.plot_individual(res, lick_res, figure_path=figure_path)
    # responsive.plot_summary_odor_and_water(res, start_days_per_mouse, training_start_day_per_mouse, learned_day_per_mouse,
    #                                        figure_path=figure_path)
    # overlap.plot_overlap_odor(res, start_days_per_mouse, learned_day_per_mouse, figure_path = figure_path)
    # overlap.plot_overlap_water(res, training_start_day_per_mouse, learned_day_per_mouse, figure_path = figure_path)
    # waveform.compare_to_shuffle(res, start= learned_day_per_mouse, end = last_day_per_mouse, data_arg='onset', figure_path=figure_path)
    # waveform.compare_to_shuffle(res, start= learned_day_per_mouse, end = last_day_per_mouse, data_arg='amplitude', figure_path=figure_path)
    # waveform.compare_to_shuffle(res, start= learned_day_per_mouse, end = last_day_per_mouse, data_arg='duration', figure_path=figure_path)
    # waveform.distribution(res, start=learned_day_per_mouse, end=last_day_per_mouse, data_arg='onset', figure_path=figure_path)
    # waveform.distribution(res, start=learned_day_per_mouse, end=last_day_per_mouse, data_arg='amplitude', figure_path=figure_path)
    # waveform.distribution(res, start=learned_day_per_mouse, end=last_day_per_mouse, data_arg='duration', figure_path=figure_path)
    compare.plot_compare_dff(res, start_days_per_mouse, learned_day_per_mouse,
                             arg='all', valence='CS+', more_stats=False, figure_path=figure_path)

if condition.name == 'OFC_REVERSAL':
    psth.count_analyze.analyze_data(res, condition_config, m_threshold=.03)
    responsive.plot_summary_odor(res, start_days_per_mouse, last_day_per_mouse, figure_path= figure_path)
    reversal.plot_reversal(res, start_days_per_mouse, last_day_per_mouse, figure_path= figure_path)
    compare.plot_compare_dff(res, start_days_per_mouse, last_day_per_mouse,
                                                arg='first', valence='CS+', more_stats=False, figure_path= figure_path)
    compare.plot_compare_dff(res, start_days_per_mouse, last_day_per_mouse,
                                                arg='last', valence='CS-', more_stats=False, figure_path= figure_path)

if condition.name == 'OFC_STATE':
    psth.count_analyze.analyze_data(res, condition_config, m_threshold=.05)
    # responsive.plot_summary_odor(res, start_days_per_mouse, last_day_per_mouse, figure_path)
    power.plot_power(res, start_days_per_mouse, last_day_per_mouse, figure_path)
    compare.plot_compare_dff(res, start_days_per_mouse, last_day_per_mouse,
                                                arg='first', valence='CS+', more_stats=True, figure_path= figure_path)

if condition.name == 'OFC_CONTEXT':
    psth.count_analyze.analyze_data(res, condition_config, m_threshold=.05)
    # responsive.plot_summary_odor(res, start_days_per_mouse, last_day_per_mouse, figure_path)
    power.plot_power(res, start_days_per_mouse, last_day_per_mouse, figure_path)
    compare.plot_compare_dff(res, start_days_per_mouse, last_day_per_mouse,
                                                arg='first', valence='CS+', more_stats=True, figure_path= figure_path)

if condition.name == 'OFC_LONGTERM':
    #fully learned thres 70%
    psth.count_analyze.analyze_data(res, condition_config)
    learned_day_per_mouse = np.array([3, 2, 2, 3])
    last_day_per_mouse = np.array([8, 7, 5, 5])
    # responsive.plot_individual(res, lick_res, figure_path= figure_path)
    # responsive.plot_summary_odor(res, learned_day_per_mouse, last_day_per_mouse, figure_path=figure_path)
    compare.plot_compare_dff(res, learned_day_per_mouse, last_day_per_mouse,
                             arg = 'first', valence='CS+', more_stats=True, figure_path= figure_path,
                             lim=[-.05, .6], ticks = [0, .5])


if condition.name == 'BLA_LONGTERM':
    psth.count_analyze.analyze_data(res, condition_config)
    plot_individual(res, lick_res, figure_path)
    plot_summary_odor(res, start_days_per_mouse, learned_day_per_mouse, figure_path)
    plot_overlap_odor(res, start_days_per_mouse, learned_day_per_mouse, figure_path)

if condition.name == 'OFC_COMPOSITE':
    psth.count_analyze.analyze_data(res, condition_config, m_threshold = 0.03)
    pt_learned = [3, 3, 4, 3]
    pt_start = condition.naive_pt_day
    plot_summary_odor_pretraining(res, pt_start, pt_learned, arg_naive=True, figure_path = figure_path)

    dt_start = [x+1 for x in condition.last_pt_day]
    dt_end = [x+1 for x in learned_day_per_mouse]
    dt_last = [x for x in last_day_per_mouse]
    plot_summary_odor(res, [0, 0, 0, 0], dt_end, figure_path)

if condition.name == 'MPFC_COMPOSITE':
    psth.count_analyze.analyze_data(res, condition_config, m_threshold=0.03)
    pt_learned = [3, 3, 3, 3]
    pt_start = [1,1,1,1]

    dt_start = [x+1 for x in condition.last_pt_day]
    dt_end = [x+1 for x in learned_day_per_mouse]
    dt_last = [x for x in last_day_per_mouse]
    dt_res = filter.filter(res, filter_dict={'odor_valence':['CS+','CS-']})

    plot_summary_odor_pretraining(res, pt_start, pt_learned, arg_naive=False, figure_path= figure_path)
    # plot_summary_odor(res, [0,0,0,0], dt_last)
    # plot_summary_odor(res, dt_start, dt_last)
    # plot_overlap(dt_res, dt_start, dt_last)



