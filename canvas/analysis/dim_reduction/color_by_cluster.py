import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    umap_emb_path = sys.argv[1]
    cluster_label_path = sys.argv[2]
    plot_path = sys.argv[3]
    plot_umap_by_cluster(umap_emb_path, cluster_label_path, plot_path)

def plot_umap_by_cluster(umap_path, cluster_label_path, plot_path):
        
    umap_embedding = np.load(umap_path)
    cluster_names = np.load(cluster_label_path)
    unique_clusters = np.unique(cluster_names)

    # Set font size
    plt.rcParams.update({'font.size': 3})
    # Set font family
    from matplotlib import font_manager
    font = '/gpfs/home/jt3545/fonts/Arial.ttf'
    font_manager.fontManager.addfont(font)
    plt.rcParams['font.family'] = 'Arial'
    # Set color map
    #cmap = plt.get_cmap('Spectral')
    cmap = plt.get_cmap('gist_rainbow')

    fig, ax = plt.subplots(figsize=(5, 5))
    for cluster in unique_clusters:
        cluster_mask = cluster_names == cluster
        color = cmap(cluster / len(unique_clusters))
        ax.scatter(umap_embedding[cluster_mask, 0], umap_embedding[cluster_mask, 1], s = 0.5, label = cluster, c = color)
    ax.set_title(f'UMAP by cluster')
    # Put a legend at the bottom with 5 rows and 10 columns, text at the bottom
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=10, columnspacing = 0.5)

    #ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=10)
    plt.savefig(plot_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.savefig(plot_path.replace('.png', 'fake.png'), bbox_inches='tight', dpi=300)

if __name__ == '__main__':
    main()
