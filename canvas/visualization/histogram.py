import numpy as np
import visualization.core as core

def multipanel_hist(data_dict, save_path, bins=100, title=None, xlabel=None, ylabel=None, max_cols = 10, log_y = False):
    markers = list(data_dict.keys())
    rows = len(markers) // max_cols + 1
    fig, axs = core.subplots(rows, max_cols, figsize=(max_cols*50, rows*50))
    for i, marker in enumerate(markers):
        ax = axs[i//max_cols, i%max_cols]
        ax.hist(data_dict[marker], bins=bins)
        ax.set_title(marker)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if log_y:
            ax.set_yscale('log')
    # Remove empty subplots
    for i in range(len(markers), rows*max_cols):
        fig.delaxes(axs[i//max_cols, i%max_cols])
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(f'{save_path}.png')
