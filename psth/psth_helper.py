import os
import filter
from _CONSTANTS import conditions as experimental_conditions
from _CONSTANTS.config import Config
import glob
import behavior.behavior_analysis
import plot
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import analysis
from matplotlib.colors import LinearSegmentedColormap
import tools.utils
from matplotlib import lines
from scipy import stats as sstats

class PSTHConfig(object):
    def __init__(self):
        self.baseline_start = 5
        self.baseline_end = 3
        self.linewidth = 1
        self.scale_linewidth = 2
        self.fill_alpha = .3

def subtract_baseline(data, baseline_start, baseline_end):
    '''
    data is in format of trial X time
    :param data:
    :return:
    '''
    mean = np.mean(data, axis=0)
    baseline_mean = np.mean(mean[baseline_start:baseline_end])
    return data - baseline_mean


def draw_scale_line_xy(ax, length=(1, .2), offset=(0, 0), **line_args):
    xmin, xmax, ymin, ymax = ax.axis()
    x = xmax+offset[0]
    y = ymin+offset[1]
    xdata = (x, x, x-length[0])
    ydata = (y+length[1], y, y)
    if 'color' not in line_args.keys():
        line_args.update({'color':'k'})
    l = lines.Line2D(xdata, ydata, **line_args)
    l.set_clip_on(False)
    ax.add_line(l)
    ax.text(x - length[0]/2, y - (ymax-ymin)/10, str(length[0]))
    ax.text(x + (xmax-xmin)/25, y + length[1]/2, str(length[1]))