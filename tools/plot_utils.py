#!/usr/bin/env python
# coding: utf-8

import numpy as np
from scipy import stats as sstats

import pylab as pl
from matplotlib import lines
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec
pl.rcParams['figure.dpi'] = 300
pl.rcParams['figure.figsize'] = (3, 2)
pl.rcParams['figure.facecolor'] = 'w'
pl.rcParams['axes.labelsize'] = 7
pl.rcParams['xtick.labelsize'] = 5
pl.rcParams['ytick.labelsize'] = 5
pl.rcParams['font.size'] = 7
pl.rcParams['figure.titlesize'] = 7
pl.rcParams['axes.titlesize'] = 7
pl.rcParams['legend.fontsize'] = 5
pl.rcParams['xtick.major.size'] = 2
pl.rcParams['ytick.major.size'] = 2
pl.rcParams['axes.linewidth'] = 1


def plot_cell(time_ax, data, offset=0, ax=None, color='k',
              plot_line=False, add_labels=True):
    if ax is None:
        fig, ax = pl.subplots(1, 1, figsize=(3, 2))

    m = np.mean(data, 0) + offset
    s = sstats.sem(data, 0)

    ax.plot(time_ax, m, lw=0.5, color=color)
    ax.fill_between(time_ax, m - s, m + s, zorder=0, lw=0, alpha=0.1, color=color)

    if plot_line:
        ax.vlines(0, 0.99, 1.1, color='r', lw=1, zorder=0)

    if add_labels:
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(r'$\Delta$ F/F')

    return ax


def plot_decoding_performance(time_ax, scores, ax=None, color='k', add_labels=True):
    if ax is None:
        fig, ax = pl.subplots(1, 1, figsize=(3, 2))

    m = np.mean(scores, 1) * 100
    s = sstats.sem(scores, 1) * 100

    ax.plot(time_ax, m, lw=0.5, color=color)
    ax.fill_between(time_ax, m - s, m + s, zorder=0, lw=0, alpha=0.3, color=color)

    if add_labels:
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Decoding performance')
    return ax

def plot_add_border(ax, offset=[0, 0], **rec_args):
    xmin, xmax, ymin, ymax = ax.axis()
    xmin, xmax, ymin, ymax = xmin-offset[0], xmax+offset[0], ymin-offset[1], ymax+offset[1]
    rec = Rectangle((xmin, ymin),
                    (xmax-xmin),
                    (ymax-ymin),
                    fill=False, zorder=19, **rec_args)
    rec = ax.add_patch(rec)
    rec.set_clip_on(False)


def remove_plot_axes(ax):
    ax.set_xticks(())
    ax.set_yticks(())
    for s in ax.spines.values():
        s.set_visible(False)


def nicer_plot(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')


def draw_scale_line_xy(ax, length=(10, 10), offset=(0, 0), **line_args):
    xmin, xmax, ymin, ymax = ax.axis()
    xdata = (xmin+offset[0], xmin+offset[0], xmin+offset[0]+length[0])
    ydata = (ymin+offset[1]+length[1], ymin+offset[1], ymin+offset[1])
    if 'color' not in line_args.keys():
        line_args.update({'color':'k'})
    l = lines.Line2D(xdata, ydata, **line_args)
    l.set_clip_on(False)
    ax.add_line(l)


def make_subplots(rows=1, cols=1, wratios=None, hratios=None, fig=None):
    if fig is None:
        fig = pl.figure()
    gs = GridSpec(rows, cols, width_ratios=wratios, height_ratios=hratios)
    return [subplot(g) for g in gs]


def pvalue_to_stars(pvalue, significance_thresholds=(0.05, 0.01, 0.001)):
    return ('n.s.' if pvalue>significance_thresholds[0]
            else '*' if pvalue>significance_thresholds[1]
            else '**' if pvalue>significance_thresholds[2]
            else '***')


def plot_significance_bar(ax, centers, height, with_ticks=False, tick_len=1, color='k', lw=1, pvalues=None):
    # with_ticks = False
    if with_ticks:
        linez = [lines.Line2D([c1, c1, c2, c2], [height-tick_len, height, height, height-tick_len], lw=lw, color=color)
                 for c1, c2 in zip(np.arange(4)+centers[0], np.arange(4)+centers[1])]
    else:
        linez = [lines.Line2D([c1, c2], [height, height], lw=lw, color=color)
                 for c1, c2 in zip(np.arange(4)+centers[0], np.arange(4)+centers[1])]

    for l in linez:
        l.set_clip_on(False)
        ax.add_line(l)
    
    if pvalues is not None:
        stars = [pvalue_to_stars(p) for p in pvalues]
        
        [ax.text((c1+c2)/2, height, s, fontsize=5, horizontalalignment='center')
         for c1, c2, s in zip(np.arange(4)+centers[0], np.arange(4)+centers[1], stars)]

    return ax

def suplabel(fig, axis, label, labelpad=5, **label_prop):
    ''' Add super ylabel or xlabel to the figure
    Similar to matplotlib.suptitle
    fig        - the matplotlib figure 
    axis       - string: "x" or "y"
    label      - string
    label_prop - keyword dictionary for Text
    labelpad   - padding from the axis (default: 5)
    ha         - horizontal alignment (default: "center")
    va         - vertical alignment (default: "center")
    '''
    xmin = []
    ymin = []
    for ax in fig.axes:
        xmin.append(ax.get_position().xmin)
        ymin.append(ax.get_position().ymin)
    xmin,ymin = min(xmin),min(ymin)
    dpi = fig.dpi
    if axis.lower() == "y":
        rotation=90.
        x = xmin - float(labelpad)/dpi
        y = 0.5
    elif axis.lower() == 'x':
        rotation = 0.
        x = 0.5
        y = ymin - float(labelpad)/dpi
    else:
        raise Exception("Unexpected axis: x or y")
    if label_prop is None: 
        label_prop = dict()
    pl.text(x,y,label,rotation=rotation,
               transform=fig.transFigure,
               **label_prop)


def plot_bars(errs_list, xpos=None, colors=None, ax=None, light=False, stderr=False):
    if ax is None:
        fig, ax = pl.subplots(1, 1, figsize=(1.5, 2))
    
    if colors is None:
        colors = [ax._get_lines.prop_cycler.next()['color'] for i in xrange(len(errs_list))]
    
    if xpos is None:
        xpos = range(len(errs_list))
    for pos, errs, color in zip(xpos, errs_list, colors):
        if light:
            ax.plot([pos]*len(errs), errs, 'o', mfc=color, ms=2, mec=(0,0,0,0), alpha=0.5)
            ax.plot([pos-0.4, pos+0.4], [np.mean(errs)]*2, 'k-', lw=1)
        else:
            ax.bar([pos], np.mean(errs), color=(0, 0, 0, 0), lw=1, edgecolor=color)
            norm = 1 if not stderr else np.sqrt(len(errs)-1)
            ax.errorbar([pos], np.mean(errs), np.std(errs)/norm, color=color, lw=1)

    return ax


def plot_cumulative(values, norm=True, bins=20, ax=None, **args):
    if ax is None:
        fig, ax = pl.subplots(1, 1)
    y, x = np.histogram(values, bins=bins)
    n = 1./np.sum(y) if norm else 1
    ax.step(x[1:], 1.*n*np.cumsum(y), where='pre', **args)
    return ax


def add_significance_bar(ax, x, y, values1, values2, test_func=None, ps=[0.05, 0.01, 0.001], dy=0.3, rotate=False,
                         **test_func_args):
    ls = [lines.Line2D(x, y, lw=0.7, color='k'),
            ]
    if test_func is None:
        test_func = lambda x, y: sstats.mannwhitneyu(x, y, alternative='two-sided')
    p = test_func(values1, values2, **test_func_args)[-1]
    [ax.add_line(l) for l in ls];
    if rotate:
        ax.text(x[0]+dy, np.mean(y), pvalue_to_stars(p),
                ha='center', rotation=90, fontsize=5)
    else:
        ax.text(np.mean(x), y[0]+dy, pvalue_to_stars(p),
                ha='center', fontsize=5)

def plot_error_shaded(x, y, color, ax=None, stderr=False, **args):
        
    if ax is None:
        fig, ax = pl.subplots(1, 1, figsize=(3, 2))
        
    m = np.r_[[np.mean(yy) for yy in y]]
    s = np.r_[[np.std(yy) if stderr==False else sstats.sem(yy) for yy in y]]
    ax.plot(x, m, '-', color=color, **args)
    ax.fill_between(x, m-s, m+s, color=color, lw=0, alpha=0.1)
    
    return ax


def add_legend(ax, labels, colors, types=None, **legend_args):
    if types==None:
        types = ['-']*len(labels)
    markers = [lines.Line2D('', '', linestyle='-', color=c, lw=1)
               for t, c, l in zip(types, colors, labels)]
    return ax.legend(markers, labels, **legend_args)


def add_chance(ax, level='25', xmin=-5.725, xmax=11.221):
    ax.hlines(level, xmin, xmax, linestyle='dashed', lw=0.5, color='k')


def plot_heatmap(time_ax, data, trials=None, ax=None, cmap=pl.cm.viridis, add_labels=True,
                 **im_args):
    
    if ax is None:
        fig, ax = pl.subplots(1, 1, figsize=(3, 2))
        
    if trials is None:
        trials = [True]*data.shape[0]

    m = data.mean(0)

    im = ax.imshow(data[trials].mean(0), origin='lower', cmap=cmap,
                   extent=(time_ax[0], time_ax[-1], 0, data.shape[1]), **im_args)
    

    ax.set_yticks((0, data.shape[1]))
    ax.set_yticklabels((1, data.shape[1]))

    if add_labels:

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Cell #')

    return ax, im


def nicer_plot(ax, bounds=True):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
