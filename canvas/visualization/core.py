# This file contains configurations for matplotlib plotting parameters
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import font_manager

from cycler import cycler

colors = cycler('color',
                ['#4285F4', # Blue
                 '#DB4437', # Red
                 '#F4B400', # Yellow
                 '#0F9D58', # Green
                 '#904EE8', '#6B72B9',
                 '#E377C3'])

#config = {'font.family' : 'Arial',
config = {
          'font.size'   : 7,
          'image.resample' : False,
          'figure.dpi' : 600,
          'xtick.major.size' : 2,
          'ytick.major.size' : 2,
          'xtick.major.width' : 0.5,
          'ytick.major.width' : 0.5,
          'axes.linewidth' : 0.5,
          'lines.linewidth' : 0.5,
          'savefig.dpi' : 600,
          'savefig.transparent' : False,
          'savefig.bbox' : 'tight',
          'axes.prop_cycle' : colors,
          'scatter.edgecolors' : 'none',
          'legend.frameon' : False,
          }

#font_manager.fontManager.addfont('/gpfs/home/jt3545/fonts/Arial.ttf')

mpl.rcParams.update(config)

def figure(figsize=(8, 6), font_size = 7, **kwargs):
    """
    Create a figure with the given figsize and kwargs.
    figsize: tuple of (width, height) in millimeters
    """
    figsize = (figsize[0] / 25.4, figsize[1] / 25.4)
    mpl.rcParams['font.size'] = font_size
    return plt.figure(figsize=figsize, **kwargs)

def subplots(nrows=1, ncols=1, figsize=(8, 6), font_size = 7, **kwargs):
    """
    Create a figure with the given figsize and kwargs.
    figsize: tuple of (width, height) in millimeters
    """
    figsize = (figsize[0] / 25.4, figsize[1] / 25.4)
    mpl.rcParams['font.size'] = font_size
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, **kwargs)

    # Remove the top and right spines
    if nrows == 1 and ncols == 1:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
    elif nrows == 1:
        for a in ax:
            a.spines['right'].set_visible(False)
            a.spines['top'].set_visible(False)
    elif ncols == 1:
        for a in ax:
            a.spines['right'].set_visible(False)
            a.spines['top'].set_visible(False)
    else:
        for row in range(nrows):
            for col in range(ncols):
                ax[row, col].spines['right'].set_visible(False)
                ax[row, col].spines['top'].set_visible(False)
    return fig, ax
