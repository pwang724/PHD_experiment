ax_args = {'yticks': [0, .2, .4, .6, .8], 'ylim': [0, .85]}
overlap_ax_args = {'yticks': [0, .5, 1], 'ylim': [0, 1.05]}
trace_ax_args = {'yticks': [0, .1, .2, .3], 'ylim': [0, .3]}

bar_args = {'alpha': .6, 'fill': False}
trace_args = {'alpha':1, 'linewidth':1}
line_args = {'alpha': .5, 'linewidth': 1, 'marker': 'o', 'markersize': 2}
scatter_args = {'marker':'.', 's':12, 'alpha': .75}
error_args = {'fmt': '.', 'capsize': 2, 'elinewidth': 1, 'markersize': 0, 'alpha': .6}
fill_args = {'zorder': 0, 'lw': 0, 'alpha': 0.5}
behavior_line_args = {'alpha': .5, 'linewidth': 1, 'marker': '.', 'markersize': 0, 'linestyle': '--'}
swarm_args = {'marker':'.', 'size':8, 'facecolors': 'none', 'alpha': .5, 'jitter': 0.2}