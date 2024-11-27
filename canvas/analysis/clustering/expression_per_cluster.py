import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore

def main():
    matrix_path = sys.argv[1] #image_mean.npy
    labels_path = sys.argv[2] #nclusters.npy
    save_path = sys.argv[3]
    n_clusters = int(sys.argv[4])
    channel_path = sys.argv[5] #'/media/ssd02/mh6486/Endometrial/CANVAS_v2/canvas/out_256/data'
    

    #data_path = '/media/ssd02/mh6486/Endometrial/as18894/cell_phenotyping/out'
    #matrix_path = f'{data_path}/normalized_tog/endo_matrix_normal.npy'
    #labels_path = f'{data_path}/normalized_tog/endo_20_labels.npy'
    #n_clusters = 20
    #save_path = f'{data_path}/normalized_tog'

    #channel_path = '/media/ssd02/mh6486/Endometrial/CANVAS_v2/canvas/out_256/data'
    channel_path = '/gpfs/data/proteomics/projects/mh6486/FenyoLab/Endometrial/CANVAS_v2/canvas/out_256/data'
    channel_names = [channel.strip() for channel in open(f'{channel_path}/common_channels.txt', 'r')]

    #print("hi")
    mean_df = create_mean_cluster_matrix(matrix_path, labels_path, n_clusters, save_path, channel_names)
    plot_heatmap(mean_df, save_path, n_clusters)

def create_mean_cluster_matrix(matrix_path, labels_path, n_clusters, save_path, channel_names = None):
    matrix = np.load(matrix_path)
    print(matrix.shape)

    kmeans_labels = np.load(labels_path)
    unique_labels = np.unique(kmeans_labels)

    mean_df = pd.DataFrame(index=channel_names, columns=unique_labels)

    for label in unique_labels:
        indices = np.where(kmeans_labels == label)[0]
        cluster = matrix[indices]
        cluster_means = np.mean(cluster, axis=0)
        mean_df[label] = cluster_means
    
    mean_df.to_csv(f'{save_path}/{n_clusters}_expression_per_cluster.csv' )
    return mean_df

def plot_heatmap(mean_df, save_path, n_clusters):
    
    if n_clusters == 50:
        plt.figure(figsize=(40, 10))
    else:
        plt.figure(figsize=(20, 10))

    # Normalize the data within each cluster to [0, 1]
    normalized_mean_df = mean_df.apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=0)
    
    sns.heatmap(mean_df, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
    plt.title(f'Channel Expression Means of {n_clusters} KMeans Clusters')
    plt.xlabel('Clusters')
    plt.ylabel('Channel Names')
    plt.tight_layout()
    plt.savefig(f'{save_path}/{n_clusters}_expression_per_cluster.png')
    print(f"expression per cluster fig saved for {n_clusters} clusters")
    plt.clf()

    sns.heatmap(normalized_mean_df, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
    plt.title(f'Channel Expression Means of {n_clusters} KMeans Clusters')
    plt.xlabel('Clusters')
    plt.ylabel('Channel Names')
    plt.tight_layout()
    plt.savefig(f'{save_path}/{n_clusters}_expression_per_cluster_normalized.png')
    print(f"expression per cluster fig saved for {n_clusters} clusters")
    plt.clf()


if __name__ == '__main__':
    main()