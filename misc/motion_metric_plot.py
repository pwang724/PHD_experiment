import numpy as np
import matplotlib.pyplot as plt
from _CONSTANTS.config import Config
import os
import glob
from collections import defaultdict
from tools import file_io
import filter
import reduce
import plot

data_directory = Config().LOCAL_DATA_PATH
data_directory = os.path.join(data_directory, 'registration','motion_metrics')

conditions = glob.glob(os.path.join(data_directory, '*/'))

res = defaultdict(list)
for c in conditions:
    mouse_pickles = glob.glob(os.path.join(c, '*.pkl'))
    for pickle in mouse_pickles:
        temp = file_io.load_pickle(pickle)
        reduce.chain_defaultdicts(res, temp)

res['within_day_crisp_average'] = []
for v in res['within_day_crisp']:
    res['within_day_crisp_average'].append(np.mean(v))

for k, v in res.items():
    res[k] = np.array(v)


save_path = os.path.join(Config.LOCAL_FIGURE_PATH, 'MISC', 'MOTION_METRICS')

mouse = 'M4_OFC'

vmin, vmax = 0.6, 1
ix = res['mouse'] == mouse
im = res['across_day_mean_corrs'][ix][0]

rect = [0.15, 0.2, 0.6, 0.6]
rect_cb = [0.77, 0.2, 0.02, 0.6]
fig = plt.figure(figsize=(2.2, 2.2))
ax = fig.add_axes(rect)
plt.imshow(im, cmap='jet', vmin=vmin, vmax= vmax)
ax.set_xlabel('Day', labelpad=2)
ax.set_ylabel('Day', labelpad=2)
plt.axis('tight')
for loc in ['bottom', 'top', 'left', 'right']:
    ax.spines[loc].set_visible(False)
ax.tick_params('both', length=0)
xticks = np.arange(0, im.shape[1]) + .5
yticks = np.arange(0, im.shape[0]) + .5
ax.set_xticks(xticks)
ax.set_yticks(yticks[::-1])
ax.set_xticklabels((xticks + .5).astype(int), fontsize=7)
ax.set_yticklabels((yticks + .5).astype(int), fontsize=7)

ax = fig.add_axes(rect_cb)
cb = plt.colorbar(cax=ax, ticks=np.arange(vmin, vmax + 0.01, 0.1))
cb.outline.set_linewidth(0.5)
cb.set_label('Correlation', fontsize=7, labelpad=2)
plt.tick_params(axis='both', which='major', labelsize=7)
plt.axis('tight')
plot._easy_save(os.path.join(save_path, 'matrix', mouse), 'across_correlation_matrix')

res = filter.exclude(res, {'mouse':'M241_ofc'})
list = ['PIR','OFC','OFC_LONGTERM','OFC_COMPOSITE','MPFC_COMPOSITE']
scatter_args = {'marker':'.', 's':8, 'alpha': .5}
error_args = {'fmt': '', 'capsize': 2, 'elinewidth': 1, 'markersize': 2, 'alpha': .5}
reduce_keys = ['within_day_crisp_average', 'across_day_mean_corrs_average']
xkey = 'experiment'
for reduce_key in reduce_keys:
    res_reduce = reduce.new_filter_reduce(res, filter_keys=['experiment'], reduce_key= reduce_key)

    for i, element in enumerate(list):
        reuse = True if i > 0 else False
        save = False if i != len(list)-1 else True
        plot.plot_results(res, x_key=xkey, y_key= reduce_key,
                          select_dict={xkey:element},
                           plot_function=plt.scatter,
                           plot_args=scatter_args,
                           fig_size=[3, 2],
                          reuse = reuse, save = False,
                           path=save_path)

        plot.plot_results(res_reduce, x_key=xkey, y_key=reduce_key,
                          error_key=reduce_key + '_sem',
                          select_dict={xkey:element},
                          plot_function=plt.errorbar,
                          plot_args=error_args,
                          fig_size=[3, 2],
                          path=save_path, reuse=True, save=save)


    for i in range(len(res['mouse'])):
        print(res['mouse'][i],res[reduce_key][i])