import numpy as np
import os
import glob
from _CONSTANTS.config import Config
import  behavior.cristian_behavior_analysis as analysis
import filter
import reduce
import plot
import matplotlib.pyplot as plt
import matplotlib as mpl
from format import *
from collections import defaultdict
from scipy.stats import ranksums
import seaborn as sns
import behavior.behavior_config

plt.style.use('default')
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.size'] = 7
mpl.rcParams['font.family'] = 'arial'

ax_args_copy = ax_args.copy()
ax_args_copy.update({'ylim':[-5, 65], 'yticks':[0, 30, 60]})
bool_ax_args_copy = ax_args.copy()
bool_ax_args_copy.update({'ylim':[-5, 105], 'yticks':[0, 50, 100]})

class OFC_DT_Config():
    path = 'I:\MANUSCRIPT_DATA\FREELY MOVING\OFC_Discrimination_with_lick_data'
    name = 'OFC_PT'

indices = analysis.Indices()
constants = analysis.Constants()
config = Config()

experiments = [OFC_DT_Config]

names = ','.join([x.name for x in experiments])
save_path = os.path.join(Config.LOCAL_FIGURE_PATH, 'BEHAVIOR_CRISTIAN', 'LICKING_CORRESPONDENCE', names)
directories = [constants.pretraining_directory, constants.discrimination_directory]

color_dict = {'Pretraining_CS+': 'C1', 'Discrimination_CS+':'green', 'Discrimination_CS-':'red'}
res = defaultdict(list)
for experiment in experiments:
    for directory in directories:
        halo_files = sorted(glob.glob(os.path.join(experiment.path, directory, constants.halo + '*')))
        res1 = analysis.parse(halo_files, experiment=experiment, condition=constants.halo, phase = directory)
        res1['experiment'] = np.array([experiment.name] * len(res1['odor_valence']))
        yfp_files = sorted(glob.glob(os.path.join(experiment.path, directory, constants.yfp + '*')))
        res2 = analysis.parse(yfp_files, experiment=experiment, condition=constants.yfp, phase = directory)
        res2['experiment'] = np.array([experiment.name] * len(res2['odor_valence']))

        # if experiment.name == 'MPFC_DT':
        #     res1 = filter.exclude(res1, {'mouse':['H01']})
        #     res2 = filter.exclude(res2, {'mouse':['H01']})
        # if experiment.name == 'MPFC_PT':
        #     res1 = filter.exclude(res1, {'mouse': ['Y01']})
        #     res2 = filter.exclude(res2, {'mouse': ['Y01']})
        reduce.chain_defaultdicts(res, res1)
        reduce.chain_defaultdicts(res, res2)

# analysis
def _get_number_of_licks(mat, start, end):
    on_off = np.diff(mat, n=1)
    n_licks = np.sum(on_off[start:end,:] > 0, axis=1)
    return n_licks

for i in range(len(res['bin_ant_1_raw'])):
    a, b = res['bin_ir_raw'][i], res['bin_samp_raw'][i]
    c, d, e = res['bin_ant_1_raw'][i], res['bin_ant_2_raw'][i], res['bin_ant_3_raw'][i]
    f, g = res['bin_col_1_raw'][i], res['bin_col_2_raw'][i]

    raw = np.concatenate([c, d, e, f, g], axis=1)
    res['bin_ant_raw'].append(raw)
    n_licks = _get_number_of_licks(raw, 0, raw.shape[0] + 1)
    res['n_licks'].append(n_licks)
    a, b = res['bin_ir'][i], res['bin_samp'][i]
    c, d, e = res['bin_ant_1'][i], res['bin_ant_2'][i], res['bin_ant_3'][i]
    f, g = res['bin_col_1'][i], res['bin_col_2'][i]
    time_on_port = c + d + e +f + g
    res['time_on_port'].append(time_on_port)

for k, v in res.items():
    res[k] = np.array(v)

def _raw_plot(res, i):
    res = filter.filter(res, {'condition':'Y'})
    print(res['mouse'][i])
    print(res['session'][i])
    a = res['bin_ir_raw'][i]
    b = res['bin_samp_raw'][i]
    x = res['bin_ant_raw'][i]

    data = np.concatenate([a, b, x], axis=1)

    fig = plt.figure(figsize=(3,3))
    ax = fig.add_axes((.25, .25, .6, .6))
    plt.imshow(data, cmap='gray')
    xticks = [0, 120, 220, 300]
    xticklabels = ['Nosepoke','Odor', '1 S', 'US']
    plt.xticks(xticks, xticklabels)
    plt.xlim([120, 300])
    plt.xlabel('Time')
    plt.ylabel('Trial')
    plot._easy_save(path=os.path.join(save_path, 'example_licks'), name='Y_{}'.format(i))

_raw_plot(res, 0)

xname = 'time_on_port'
yname = 'n_licks'
xlim = [0, 270]
ylim = [0, 17]
xticks = [0, 100, 200]
yticks = [0, 5, 10, 15]

def _get_density(x, y):
    from scipy.stats import gaussian_kde
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)
    z = (z - np.min(z)) / (np.max(z) - np.min(z))

    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    return x, y, z

def _add_jitter(x, jitter_range):
    x = x + np.random.uniform(0, jitter_range, size= x.shape)
    return x

def _filter_session(temp, session_filter):
    for i, sess in enumerate(temp['session']):
        ix = sess == session_filter
        temp[xname][i] = temp[xname][i][ix]
        temp[yname][i] = temp[yname][i][ix]

def _filter_time_on_port(temp, minimum_time, maximum_time):
    for i, time_port in enumerate(temp[xname]):
        ix = (time_port > minimum_time) * (time_port < maximum_time)
        temp[xname][i] = temp[xname][i][ix]
        temp[yname][i] = temp[yname][i][ix]

def _plot(x, y, z, xlim, ylim, xticks, yticks, name, color=True):
    fig = plt.figure(figsize=(3,2))
    ax = fig.add_axes((.2, .2, .6, .6))
    if not color:
        z = np.ones_like(x)
    plt.scatter(x, y, c= z, s=3, edgecolor='')
    plt.xlabel(xname)
    plt.ylabel(yname)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xticks(xticks)
    plt.yticks(yticks)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    folder = yname + '_vs_' + xname
    name = name

    #first method
    from sklearn.linear_model import LinearRegression
    x = x.reshape(-1,1)
    y = y.reshape(-1,1)
    model = LinearRegression().fit(x, y)
    r2 = model.score(x, y)
    slope = model.coef_[0][0]
    intercept = model.intercept_[0]
    y_pred = model.predict(x)

    # second method
    # slope, residual, _, _ = np.linalg.lstsq(x, y, rcond=-1)
    # slope = slope[0][0]
    # r2= 1 - residual[0] / (x.size * y.var())
    # intercept = 0
    # y_pred = slope * x

    plt.plot(x, y_pred, 'r-', linewidth=1, alpha=.7)

    xlim = plt.xlim()
    ylim = plt.ylim()
    plt.text(x= .1 * (xlim[-1] - xlim[0]), y=.8 * (ylim[-1] - ylim[0]), s= 'R = {0:.3f}'.format(r2))
    plt.title('$y= {0:.3f} x+{1:.2f}$'.format(slope, intercept))
    plot._easy_save(os.path.join(save_path, folder), name)

# early
temp = filter.filter(res, {'odor_valence':'CS+'})
_filter_session(temp, 1)
_filter_time_on_port(temp, 1, 200)
x = np.concatenate(temp[xname])
y = np.concatenate(temp[yname])
x, y, z = _get_density(x, y)
x_jitter = _add_jitter(x, 5)
y_jitter = _add_jitter(y, 1)
_plot(x_jitter, y_jitter, z, xlim=xlim, ylim=ylim, xticks=xticks, yticks=yticks, name='early')

# late
temp = filter.filter(res, {'odor_valence':'CS+'})
_filter_session(temp, 2)
_filter_time_on_port(temp, 1, 1000)
x = np.concatenate(temp[xname])
y = np.concatenate(temp[yname])
x, y, z = _get_density(x, y)
x_jitter = _add_jitter(x, 5)
y_jitter = _add_jitter(y, 1)
_plot(x_jitter, y_jitter, z, xlim=xlim, ylim=ylim, xticks=xticks, yticks=yticks,name='late')

#individual
tuples, _ = filter.retrieve_unique_entries(res, loop_keys=['condition', 'phase','mouse'])
for tuple in tuples:
    condition = tuple[0]
    phase = tuple[1]
    mouse = tuple[2]
    temp = filter.filter(res, {'odor_valence': 'CS+', 'condition': condition, 'phase': phase, 'mouse': mouse})

    sessions = np.unique(temp['session'][0])
    for session in sessions:
        temp = filter.filter(res, {'odor_valence': 'CS+', 'phase': phase, 'mouse': mouse, 'condition': condition})
        _filter_session(temp, session)
        _filter_time_on_port(temp, 1, 1000)
        x = np.concatenate(temp[xname])
        y = np.concatenate(temp[yname])
        # x, y, z = _get_density(x, y)
        x_jitter = _add_jitter(x, 5)
        y_jitter = _add_jitter(y, 1)
        print(x_jitter.shape)
        print(y_jitter.shape)

        if len(x_jitter):
            _plot(x_jitter, y_jitter, [], xlim=xlim, ylim=ylim, xticks=xticks,
                  yticks=yticks, name='{}_{}_{}'.
                  format(mouse, phase, session), color=False)