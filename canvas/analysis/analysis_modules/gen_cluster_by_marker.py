import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.io import imsave
import pdb

#IF HEATMAP IS HARD TO READ/TOO SMALL, PLAY AROUND WITH cellSizePixels and fontsize in fixedWidthClusterMap=

def gen_cluster_by_marker(tile_embedding_path, kmeans_path ,output_path, channel_names):
  
    os.makedirs(output_path, exist_ok = True)
    output_path = os.path.join(output_path, "heatmap.npy")

    image_mean_path = os.path.join(tile_embedding_path, 'image_mean.npy') 
    cluster_path = os.path.join(kmeans_path, 'clusters.npy')

    mean_intensities = np.load(image_mean_path)
    clusters = np.load(cluster_path)
    n_cluster = len(np.unique(clusters))

    # Normalized channel intensity to [0, 1]
    t_min, t_max = np.percentile(mean_intensities, [2, 98], axis = 0)
    mean_intensities = np.clip((mean_intensities - t_min) / t_max, 0, 1)

    # Overall mean intensity
    overall_mean_intensity = np.mean(mean_intensities, axis = 0)
    np.save(output_path.replace('.npy', '_overall_mean_intensity.npy'), overall_mean_intensity)

    cluster_mat_list = []


    for cluster_i in range(n_cluster):
        cluster_data = mean_intensities[clusters == cluster_i, :]
        plot_df = pd.DataFrame(cluster_data, columns = channel_names)
        cluster_n = len(cluster_data)

        cluster_mat_list.append(np.nanmean(cluster_data, axis = 0))

        print(f'n = {cluster_n}')

    cluster_mat = np.array(cluster_mat_list)
    cluster_mat = np.nan_to_num(cluster_mat)
    np.save(output_path, cluster_mat)

    # Set font size
    plt.rcParams.update({'font.size': 8})
    # Set font family
    # from matplotlib import font_manager
    # font = '/gpfs/home/jt3545/fonts/Arial.ttf'
    # font_manager.fontManager.addfont(font)
    # plt.rcParams['font.family'] = 'Arial'

    fixedWidthClusterMap(cluster_mat, yticklabels = [f'{x}' for x in range(n_cluster)], xticklabels = channel_names,  cmap = 'Reds', row_cluster=False, col_cluster = False)
    
    #sns.clustermap(cluster_mat, yticklabels = [f'{x}' for x in range(n_cluster)], xticklabels = channel_names,  cmap = 'Reds', row_cluster=False, col_cluster = False, figsize = (2, 5))

    plt.yticks(rotation=45)
    plt.xticks(rotation=90)
    print(output_path)
    plt.savefig(output_path.replace('.npy', '.png'), bbox_inches = 'tight')
    plt.clf()


def fixedWidthClusterMap(dataFrame, cellSizePixels=20, **args):
    # Calulate the figure size, this gets us close, but not quite to the right place
    import matplotlib
    dpi = matplotlib.rcParams['figure.dpi']
    marginWidth = matplotlib.rcParams['figure.subplot.right']-matplotlib.rcParams['figure.subplot.left']
    marginHeight = matplotlib.rcParams['figure.subplot.top']-matplotlib.rcParams['figure.subplot.bottom']
    Ny,Nx = dataFrame.shape
    figWidth = (Nx*cellSizePixels/dpi)/0.8/marginWidth
    figHeigh = (Ny*cellSizePixels/dpi)/0.8/marginHeight 

    # do the actual plot
    grid = sns.clustermap(dataFrame, figsize=(figWidth, figHeigh), **args)

    # calculate the size of the heatmap axes
    axWidth = (Nx*cellSizePixels)/(figWidth*dpi)
    axHeight = (Ny*cellSizePixels)/(figHeigh*dpi)

    # resize heatmap
    ax_heatmap_orig_pos = grid.ax_heatmap.get_position()
    grid.ax_heatmap.set_position([ax_heatmap_orig_pos.x0, ax_heatmap_orig_pos.y0, 
                                  axWidth, axHeight])

    # resize dendrograms to match
    ax_row_orig_pos = grid.ax_row_dendrogram.get_position()
    grid.ax_row_dendrogram.set_position([ax_row_orig_pos.x0, ax_row_orig_pos.y0, 
                                         ax_row_orig_pos.width, axHeight])
    ax_col_orig_pos = grid.ax_col_dendrogram.get_position()
    grid.ax_col_dendrogram.set_position([ax_col_orig_pos.x0, ax_heatmap_orig_pos.y0+axHeight,
                                         axWidth, ax_col_orig_pos.height])
    return grid # return ClusterGrid object

if __name__ == '__main__':
    main()
