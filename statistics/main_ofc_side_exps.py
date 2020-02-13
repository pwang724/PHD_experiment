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
import matplotlib.pyplot as plt
import matplotlib as mpl

import statistics.analyze
import statistics.count_methods.power as power
import statistics.count_methods.compare as compare
import statistics.count_methods.correlation as correlation
import statistics.count_methods.cory as cory
import statistics.count_methods.reversal as reversal
from scipy.stats import ranksums, wilcoxon, kruskal

plt.style.use('default')
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.size'] = 6
mpl.rcParams['font.family'] = 'arial'

condition_config = statistics.analyze.OFC_Reversal_Config()
condition = condition_config.condition
data_path = os.path.join(Config.LOCAL_DATA_PATH, Config.LOCAL_DATA_TIMEPOINT_FOLDER, condition.name)
save_path = os.path.join(Config.LOCAL_EXPERIMENT_PATH, 'COUNTING', condition.name)
figure_path = os.path.join(Config.LOCAL_FIGURE_PATH, 'OTHER', 'COUNTING',  condition.name)

#analysis
res = fio.load_pickle(os.path.join(save_path, 'dict.pkl'))
# res.pop('data')
# retrieving relevant days
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

if condition.name == 'OFC_REVERSAL':
    res = statistics.analyze.analyze_data(save_path, condition_config, m_threshold= 0.04)
    #1 0 1 1
    start_days_per_mouse = [1, 1, 1, 1, 1]
    # responsive.plot_summary_odor(res, start_days_per_mouse, last_day_per_mouse, figure_path= figure_path)
    # compare.plot_compare_dff(res, start_days_per_mouse, last_day_per_mouse,
    #                                             arg='first', valence='CS+', more_stats=False, figure_path= figure_path)
    # compare.plot_compare_dff(res, start_days_per_mouse, last_day_per_mouse,
    #                                             arg='last', valence='CS-', more_stats=False, figure_path= figure_path)
    # reversal.plot_reversal(res, start_days_per_mouse, last_day_per_mouse, figure_path= figure_path)
    # power.plot_power(res, [1, 0, 1, 1, 1], [1, 0, 1, 1, 1], figure_path, odor_valence=['CS+', 'CS-'],
    #                  colors_before = {'CS+':'Green','CS-':'Red'}, colors_after = {'CS-':'Red','CS+':'Green'},
    #                  ylim = [-0.005, .1])
    power.plot_power(res, last_day_per_mouse, last_day_per_mouse, figure_path, odor_valence=['CS+', 'CS-'],
                     colors_before = {'CS+':'Green','CS-':'Red'}, colors_after = {'CS-':'Red','CS+':'Green'},
                     ylim = [-0.005, .1])

    before_csp = [ 0.12609179,  0.11384058,  0.07161805,  0.06145226,  0.07293244,  0.05645741, 0.04455081,  0.03228587,  0.04068222,  0.04597894]
    before_csm = [ 0.05445334,  0.03700504,  0.03669895,  0.04474983,  0.03151948,  0.03421161, 0.04146117,  0.03686822,  0.0434961,   0.02564307]

    after_csp = [0.05620492, 0.04649211,  0.05401214,  0.04357546,  0.03712788,  0.03253676 , 0.01846068 , 0.0207211 ,  0.02797289 , 0.02952]
    after_csm = [0.11979772 , 0.12339208,  0.06531946 , 0.0946324  , 0.0653834 ,  0.09284357, 0.03751425 , 0.03192471 , 0.04160611,  0.04502207]

    print(wilcoxon(before_csp, after_csp))
    print(wilcoxon(before_csm, after_csm))

if condition.name == 'OFC_STATE':
    res = statistics.analyze.analyze_data(save_path, condition_config, m_threshold=0.04)
    # responsive.plot_summary_odor(res, start_days_per_mouse, last_day_per_mouse, figure_path=figure_path)
    # power.plot_power(res, start_days_per_mouse, last_day_per_mouse, figure_path, odor_valence=['CS+'],
    #                  colors_before = {'CS+':'Green','CS-':'Red'}, colors_after = {'CS+':'Gray','CS-':'Gray'},
    #                  ylim = [-0.005, .1])
    # compare.plot_compare_dff(res, start_days_per_mouse, last_day_per_mouse,
    #                                             arg='first', valence='CS+', more_stats=True, figure_path= figure_path)
    compare.distribution_dff(res, start_days_per_mouse, last_day_per_mouse, arg='first', valence='CS+',
                             figure_path=figure_path)
    # start = [0,0,0,0,0]
    # end = [1,1,1,1,1]
    # valence_responsive.plot_responsive_difference_odor_and_water(res, start, end,
    #                                                              figure_path=figure_path, normalize=False, ylim=.4)
    # power.plot_max_dff_days(res, [start, end], ['CS+', 'CS+'], save=False, reuse=False, day_pad= 0, figure_path = figure_path, ylim=.2)
    # power.plot_max_dff_days(res, [start, end], ['CS-','CS-'],save=False, reuse=True, day_pad= 0, figure_path = figure_path, ylim=.2)
    # power.plot_bar(res, [start, end], ['CS-', 'CS-'], color='darkred',
    #                day_pad=0, save=False, reuse=True, figure_path=figure_path)
    # power.plot_bar(res, [start, end], ['CS+', 'CS+'], color='darkgreen',
    #                day_pad=0, save=True, reuse=True, figure_path=figure_path)

if condition.name == 'OFC_CONTEXT':
    res = statistics.analyze.analyze_data(save_path, condition_config, m_threshold=0.04)
    # responsive.plot_summary_odor(res, start_days_per_mouse, last_day_per_mouse, figure_path=figure_path)
    # power.plot_power(res, start_days_per_mouse, last_day_per_mouse, figure_path, odor_valence=['CS+'],
    #                  colors_before = {'CS+':'Green','CS-':'Red'}, colors_after = {'CS+':'Gray','CS-':'Gray'},
    #                  ylim = [-0.005, .1])
    # compare.plot_compare_dff(res, start_days_per_mouse, last_day_per_mouse,
    #                          arg='all', valence='CS+', more_stats=True, figure_path= figure_path)
    compare.distribution_dff(res, start_days_per_mouse, last_day_per_mouse, arg='first', valence='CS+',
                             figure_path=figure_path)
    # start = [0,0,0,0]
    # end = [1,1,1,1]
    # valence_responsive.plot_responsive_difference_odor_and_water(res, [0,0,0,0], [1,1,1,1],
    #                                                              figure_path=figure_path, normalize=False, ylim=.65)
    # power.plot_max_dff_days(res, [start, end], ['CS+', 'CS+'], save=False, reuse=False, day_pad= 0, figure_path = figure_path, ylim=.2)
    # power.plot_max_dff_days(res, [start, end], ['CS-','CS-'],save=False, reuse=True, day_pad= 0, figure_path = figure_path, ylim=.2)
    # power.plot_bar(res, [start, end], ['CS-', 'CS-'], color='darkred',
    #                day_pad=0, save=False, reuse=True, figure_path=figure_path)
    # power.plot_bar(res, [start, end], ['CS+', 'CS+'], color='darkgreen',
    #                day_pad=0, save=True, reuse=True, figure_path=figure_path)