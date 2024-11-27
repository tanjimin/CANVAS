import os
import numpy as np
import matplotlib.pyplot as plt

def color_umap(save_path, umap_coord, clusters, cluster_color, output_suffix = 'umap'):
    output_path = f'{save_path}/{output_suffix}'
    save_file = os.path.join(output_path, 'umap_color.pdf')
    os.makedirs(output_path, exist_ok = True)
    if os.path.exists(save_file):
        print('UMAP color already exist, skipping')
        #return 
    #cluster_color = np.clip(cluster_color / 255 * 2, 0, 1)
    cluster_color = cluster_color
    fig, ax = plt.subplots()
    for i in range(clusters.max() + 1):
        mask = clusters == i
        ax.scatter(umap_coord[mask, 0], umap_coord[mask, 1], c = cluster_color[i], label = f'cluster {i}', s = 0.3)
    ax.legend()
    plt.savefig(save_file)

def compare_with_legacy_cluster(save_path, umap_coord, clusters = None, cluster_color = None):
    if clusters is None:
        clusters = '/gpfs/data/tsirigoslab/home/jt3545/CODEX/codex-imaging/src/model/feature_extraction/features/imc/s4-1_cluster_kmeans/thumbnail_scale_16/64/checkpoint-1999/default/50/clusters.npy'
    if cluster_color is None:
        cluster_color = '/gpfs/data/tsirigoslab/home/jt3545/CODEX/codex-imaging/src/model/feature_extraction/features/imc/s4-1_cluster_kmeans/thumbnail_scale_16/64/checkpoint-1999/default/50/cluster_plot_color_rgb.npy'
    umap_coord = np.load(umap_coord)
    clusters = np.load(clusters)
    cluster_color = np.load(cluster_color)
    color_umap(save_path, umap_coord, clusters, cluster_color)

def save_sankey(save_path, grouping1, grouping2, title = 'Sankey Diagram', output_suffix = 'sankey'):
    output_path = f'{save_path}/{output_suffix}'
    save_file = os.path.join(output_path, 'sankey.pdf')
    os.makedirs(output_path, exist_ok = True)
    if os.path.exists(save_file):
        print('Sankey already exist, skipping')
        #return 
    fig, ax = plt.subplots()
    sankey = ax.sankey(grouping1, grouping2, scale = 1, head_angle = 180, shoulder = 0, gap = 0.1)
    ax.set_title(title)
    plt.savefig(save_file)

def plot_sankey(ax, grouping1, grouping2):
    pass
