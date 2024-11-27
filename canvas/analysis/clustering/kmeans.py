import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pdb
import sklearn.cluster as cluster

def kmeans(emb_path, umap_path, n_clusters, save_path):
    os.makedirs(save_path, exist_ok = True)
    emb_path = os.path.join(emb_path, 'embedding_mean.npy')
    umap_path = os.path.join(umap_path, 'coord.npy')
    save_path = os.path.join(save_path, 'clusters.npy')
    kmeans_clustering(emb_path, umap_path, n_clusters, save_path)

def plot_cluster_color(umap_path, kmeans_labels_path, cluster_colors_path, save_path):
    print('Plotting clusters on UMAP')
    umap_embedding = np.load(umap_path)
    kmeans_labels = np.load(kmeans_labels_path)
    cluster_colors = np.load(cluster_colors_path)
    palette = sns.color_palette(cluster_colors)
    n_clusters = len(palette)
    fig, ax = plt.subplots(figsize = (10, 10))
    #sns.scatterplot(x = umap_embedding[:, 0], y = umap_embedding[:, 1], size = 1, hue=kmeans_labels, palette = palette, legend='full')
    # Create a scatter plot with a colormap using matplotlib
    plt.scatter(
        umap_embedding[:, 0], 
        umap_embedding[:, 1], 
        c=kmeans_labels, 
        cmap='tab20'  # Use cmap for continuous color mapping
    )
    plt.colorbar()  # Add a colorbar if needed
    plt.title(f'Kmeans {n_clusters} clusters')
    plt.legend(bbox_to_anchor=(1.1, 1.05))
    plt.savefig(save_path)
    plt.clf()

def kmeans_clustering(emb_path, umap_path, n_clusters, save_path):
    embedding = np.load(emb_path)
    print('Embedding loaded')
    kmeans = cluster.KMeans(n_clusters=n_clusters, random_state=0).fit(embedding)
    kmeans_labels = kmeans.labels_
    kmeans_inertia = kmeans.inertia_
    np.save(save_path, kmeans_labels)
    print('Kmeans saved')
    with open(save_path.split('.')[0] + '_inertia.txt', 'w') as f:
        f.write(str(kmeans_inertia))
    print('Kmeans inertia saved')
    print('Plotting clusters on UMAP')
    umap_embedding = np.load(umap_path)
    fig, ax = plt.subplots(figsize = (10, 10))
    if n_clusters <= 20:
        palette = 'tab20'
    else:
        palette = 'Spectral'
    #sns.scatterplot(x = umap_embedding[:, 0], y = umap_embedding[:, 1], size = 1, hue=kmeans_labels, palette = palette, legend='full', cmap='tab20') #MH added cmap='tab20'
    
    # Create a scatter plot with a colormap using matplotlib
    plt.scatter(
        umap_embedding[:, 0], 
        umap_embedding[:, 1], 
        c=kmeans_labels, 
        cmap='tab20'  # Use cmap for continuous color mapping
    )
    
    plt.colorbar()  # Add a colorbar if needed
    plt.title(f'Kmeans {n_clusters} clusters')
    plt.legend(bbox_to_anchor=(1.1, 1.05))
    plt.savefig(save_path.replace('.npy', '.png'))
    plt.clf()

if __name__ == '__main__':
    main()
