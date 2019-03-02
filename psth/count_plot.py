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
import psth.count_methods.power as power
import psth.count_methods.responsive as responsive
import psth.count_methods.reversal as reversal

condition_config = psth.count_analyze.PIR_Config()

config = psth.psth_helper.PSTHConfig()
condition = condition_config.condition
data_path = os.path.join(Config.LOCAL_DATA_PATH, Config.LOCAL_DATA_TIMEPOINT_FOLDER, condition.name)
save_path = os.path.join(Config.LOCAL_EXPERIMENT_PATH, 'COUNTING', condition.name)
figure_path = os.path.join(Config.LOCAL_FIGURE_PATH, 'OTHER', 'COUNTING',  condition.name)

#analysis
res = fio.load_pickle(os.path.join(save_path, 'dict.pkl'))

# retrieving relevant days
learned_day_per_mouse, last_day_per_mouse = get_days_per_mouse(data_path, condition)
print(learned_day_per_mouse)
if condition_config.start_at_training and hasattr(condition, 'training_start_day'):
    start_days_per_mouse = condition.training_start_day
else:
    start_days_per_mouse = [0] * len(np.unique(res['mouse']))
training_start_day_per_mouse = condition.training_start_day

lick_res = behavior.behavior_analysis.get_licks_per_day(data_path, condition)
analysis.add_odor_value(lick_res, condition)
lick_res = filter.filter(lick_res, {'odor_valence': ['CS+', 'CS-', 'PT CS+']})
lick_res = reduce.new_filter_reduce(lick_res, ['odor_valence', 'day', 'mouse'], reduce_key='lick_boolean')

if condition.name == 'OFC' or condition.name == 'BLA':
    psth.count_analyze.analyze_data(res, condition_config, m_threshold=0.03)
    plot_individual(res, lick_res, figure_path=figure_path)
    plot_summary_odor(res, start_days_per_mouse, last_day_per_mouse, figure_path)
    plot_summary_water(res, training_start_day_per_mouse, learned_day_per_mouse, figure_path)
    psth.count_analyze.analyze_data(res, condition_config, m_threshold=0.03)
    plot_overlap_water(res, training_start_day_per_mouse, learned_day_per_mouse, figure_path)
    plot_overlap_odor(res, start_days_per_mouse, learned_day_per_mouse, figure_path)

if condition.name == 'BLA_LONGTERM':
    psth.count_analyze.analyze_data(res, condition_config)
    plot_individual(res, lick_res, figure_path)
    plot_summary_odor(res, start_days_per_mouse, learned_day_per_mouse, figure_path)
    plot_overlap_odor(res, start_days_per_mouse, learned_day_per_mouse, figure_path)

if condition.name == 'OFC_LONGTERM':
    #fully learned thres 70%
    psth.count_analyze.analyze_data(res, condition_config, m_threshold=0.03)
    plot_individual(res, lick_res, figure_path)
    plot_summary_odor(res, learned_day_per_mouse, last_day_per_mouse, figure_path)

if condition.name == 'PIR':
    psth.count_analyze.analyze_data(res, condition_config)
    responsive.plot_individual(res, lick_res, figure_path)
    responsive.plot_summary_odor(res, start_days_per_mouse, learned_day_per_mouse, figure_path)
    responsive.plot_summary_water(res, training_start_day_per_mouse, learned_day_per_mouse, figure_path)
    # overlap.plot_overlap_odor(res, start_days_per_mouse, learned_day_per_mouse,
    #                                              delete_non_selective=True, figure_path= figure_path)
    # stability.plot_stability_across_days(res, start_days_per_mouse, learned_day_per_mouse, figure_path)

if condition.name == 'OFC_STATE':
    psth.count_analyze.analyze_data(res, condition_config)
    plot_summary_odor(res, start_days_per_mouse, last_day_per_mouse, figure_path)
    plot_power(res, start_days_per_mouse, last_day_per_mouse, figure_path)

if condition.name == 'OFC_REVERSAL':
    psth.count_analyze.analyze_data(res, condition_config)
    plot_summary_odor(res, start_days_per_mouse, last_day_per_mouse, figure_path)
    plot_reversal(res, start_days_per_mouse, last_day_per_mouse, figure_path)


if condition.name == 'OFC_CONTEXT':
    psth.count_analyze.analyze_data(res, condition_config)
    plot_summary_odor(res, start_days_per_mouse, last_day_per_mouse, figure_path)
    plot_power(res, start_days_per_mouse, last_day_per_mouse, figure_path)

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



