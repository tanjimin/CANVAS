import sys
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pdb

#python sample_per_cluster.py "/media/ssd02/mh6486/Endometrial/CANVAS-Dev/out_512_artrm/analysis/epoch_1350/tile_embedding/sample_name.npy" "/media/ssd02/mh6486/Endometrial/CANVAS-Dev/out_512_artrm/analysis/epoch_1350/kmeans/5clusters.npy" "/media/ssd02/mh6486/Endometrial/CANVAS-Dev/out_512_artrm/analysis/epoch_1350" 5

def main():
    sample_names_path = sys.argv[1]
    labels_path = sys.argv[2]
    save_path = sys.argv[3]
    n_clusters = int(sys.argv[4])

    #data_path = '/media/ssd02/mh6486/Endometrial/as18894/cell_phenotyping/out'
    #sample_names_path = f'{data_path}/normal_matrix/endo_cell_sample_names_filtered.npy'
    #labels_path = f'{data_path}/clustering/endo_20_labels.npy'
    #save_path = f'{data_path}/clustering'
    #n_clusters = 20
    slides = ["4G", "2G", "3K", "4C-1", "1J-1", "3H-1", "1O", "1P", "1H", "3S", "3P", "4I", "IT", "1I", "6H", "3D", "1E"]

    proportion_per_cluster(sample_names_path, labels_path, save_path, n_clusters, slides)

def proportion_per_cluster(sample_names_path, labels_path, save_path, n_clusters, slides):
    sample_names = np.load(sample_names_path)
    unique_samples = np.unique(sample_names)
    
    labels = np.load(labels_path) 
    unique_labels = np.unique(labels) #this is the number of unique kmeans 

    proportion_matrix = np.zeros((len(unique_samples), len(unique_labels))) #row,column

    for i, sample in enumerate(unique_samples):
        print(i,sample)
        sample_indices = np.where(sample_names == sample)[0] #this tells us the index of each tile that is the sample we care about 
        
        for j, label in enumerate(unique_labels):
            label_count = np.count_nonzero(labels[sample_indices] == label) #this counts the number of tiles (sample_indices) that have the label we care about 
            proportion_matrix[i, j] = label_count / len(sample_indices) #this is the number of tiles in the sample we care about and the kmeans cluster we care about / total number of tiles for that sample

    if n_clusters == 50:
        plt.figure(figsize=(25, 10))
    else:
        plt.figure(figsize=(20, 10))
    

    #pdb.set_trace()
    sns.heatmap(proportion_matrix, linewidths =1, linecolor='black',cmap='Blues', annot=True, fmt=".2f", cbar_kws={'label': 'Proportion'}, vmin = 0, vmax = 1)

    plt.xlabel('Cluster')
    plt.ylabel('Sample')
    plt.xticks(ticks=np.arange(len(unique_labels)) + 0.5, labels=np.arange(1, len(unique_labels) + 1))
    plt.yticks(ticks=[i + 0.5 for i in range(len(slides))], labels=slides)
    plt.title('Proportion of Samples in Each Cluster')
    plt.tight_layout()
    plt.savefig(f'{save_path}/{n_clusters}_sample_per_cluster.png')
    print(f'sample per cluster fig saved for {n_clusters} clusters')
    plt.clf()

if __name__ == '__main__':
    main()