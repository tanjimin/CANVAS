import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.io import imsave
import pdb

def gen_cluster_vs_sample(kmeans_path, tile_embedding_path, cluster_save_path):

    cluster_path = os.path.join(kmeans_path, 'clusters.npy')
    sample_path = os.path.join(tile_embedding_path, 'sample_name.npy')
    csv_save_path = os.path.join(cluster_save_path, 'cluster_vs_sample.csv')
    output_cluster_sample_enrichment = os.path.join(cluster_save_path, 'cluster_with_sample_enrichment.png')

    clusters = np.load(cluster_path)
    samples = np.load(sample_path)

    # Plot distribution of samples for each cluster
    unique_clusters = np.unique(clusters)
    unique_samples = np.unique(samples)

    plot_heatmap(unique_clusters, unique_samples, clusters, samples, output_cluster_sample_enrichment, csv_save_path)
    plot_bar_charts(unique_clusters, unique_samples, clusters, samples, output_cluster_sample_enrichment)

def plot_heatmap(unique_clusters, unique_samples, clusters, samples, save_path, csv_save_path):
    fig, ax = plt.subplots(len(unique_clusters), figsize=(len(unique_clusters) * 2, 40))

    cluster_mat_list = []
    for i, cluster in enumerate(unique_clusters):
        # Generate counts for each unique sample
        print(f'Generating cluster {i} data')
        # Get all samples in a cluster
        cluster_samples = samples[clusters == cluster]
        # Generate counts for each unique sample
        cluster_samples, counts = np.unique(cluster_samples, return_counts=True)
        counts_all_samples = np.zeros(len(unique_samples))
        cluster_idx = np.where(np.in1d(unique_samples, cluster_samples))[0]
        counts_all_samples[cluster_idx] = counts
        cluster_mat_list.append(counts_all_samples.copy())

    cluster_mat = np.array(cluster_mat_list)
    cluster_mat = np.nan_to_num(cluster_mat).astype(int)

    # Generate csv file
    df = pd.DataFrame(cluster_mat, columns=unique_samples)
    csv_save_path_dir = os.path.dirname(csv_save_path)
    #pdb.set_trace()
    os.makedirs(csv_save_path_dir, exist_ok = True)
    df.to_csv(csv_save_path)

    # Save all counts
    fixedWidthClusterMap(cluster_mat, yticklabels = [f'cluster {x}' for x in unique_clusters], xticklabels = unique_samples,  cmap = 'Reds', annot = True, fmt='g', row_cluster = False, col_cluster = False)
    plt.yticks(rotation=45)
    plt.xticks(rotation=45)
    plt.savefig(save_path.replace('.png', '_heatmap.png'))
    plt.close()

    # all counts clustered
    fixedWidthClusterMap(cluster_mat, yticklabels = [f'cluster {x}' for x in unique_clusters], xticklabels = unique_samples,  cmap = 'Reds', annot = True, fmt='g', row_cluster = True, col_cluster = True)
    plt.yticks(rotation=45)
    plt.xticks(rotation=45)
    plt.savefig(save_path.replace('.png', '_heatmap_clustered.png'))
    plt.close()

    # Save marginalized counts
    marg_cluster_mat = np.concatenate([cluster_mat, np.sum(cluster_mat, axis=0).reshape(1, -1)], axis=0)
    marg_cluster_mat = np.concatenate([marg_cluster_mat, np.sum(marg_cluster_mat, axis=1).reshape(-1, 1)], axis=1)
    fixedWidthClusterMap(marg_cluster_mat, yticklabels = [f'cluster {x}' for x in unique_clusters] + ['total'], xticklabels = unique_samples.tolist() + ['total'],  cmap = 'Reds', annot = True, fmt='g', row_cluster = False, col_cluster = False, vmax = 160)
    plt.yticks(rotation=45)
    plt.xticks(rotation=45)
    plt.savefig(save_path.replace('.png', '_marginalized_heatmap.png'))
    plt.close()

    # Save column normalized counts
    col_norm_cluster_mat = cluster_mat / np.sum(cluster_mat, axis=0)
    col_norm_cluster_mat = np.nan_to_num(col_norm_cluster_mat)
    fixedWidthClusterMap(col_norm_cluster_mat, yticklabels = [f'cluster {x}' for x in unique_clusters], xticklabels = unique_samples,  cmap = 'Reds', annot = True, fmt='.2f', row_cluster = False, col_cluster = False)
    plt.yticks(rotation=45)
    plt.xticks(rotation=45)
    plt.savefig(save_path.replace('.png', '_col_normalized_heatmap.png'))
    plt.close()

    # Save row normalized counts
    row_norm_cluster_mat = cluster_mat / np.sum(cluster_mat, axis=1).reshape(-1, 1)
    row_norm_cluster_mat = np.nan_to_num(row_norm_cluster_mat)
    fixedWidthClusterMap(row_norm_cluster_mat, yticklabels = [f'cluster {x}' for x in unique_clusters], xticklabels = unique_samples,  cmap = 'Reds', annot = True, fmt='.2f', row_cluster = False, col_cluster = False)
    plt.yticks(rotation=45)
    plt.xticks(rotation=45)
    plt.savefig(save_path.replace('.png', '_row_normalized_heatmap.png'))
    plt.close()



def plot_bar_charts(unique_clusters, unique_samples, clusters, samples, save_path):
    fig, ax = plt.subplots(len(unique_clusters), figsize=(len(unique_clusters) * 5, 100))
    for i, cluster in enumerate(unique_clusters):
        print(f'Plotting cluster {i}')
        # Get all samples in a cluster
        cluster_samples = samples[clusters == cluster]
        # Generate counts for each unique sample
        cluster_samples, counts = np.unique(cluster_samples, return_counts=True)
        counts_all_samples = np.zeros(len(unique_samples))
        cluster_idx = np.where(np.in1d(unique_samples, cluster_samples))[0]
        counts_all_samples[cluster_idx] = counts
        # Plot counts of each sample
        sns.barplot(x=unique_samples, y=counts_all_samples, ax=ax[i])
        ax[i].set_title('Cluster {}'.format(cluster))

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def fixedWidthClusterMap(dataFrame, cellSizePixels=50, **args):
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
