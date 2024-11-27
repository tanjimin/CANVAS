import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm

def main():
    umap_emb_path = sys.argv[1]
    sample_label_path = sys.argv[2]
    plot_path = sys.argv[3]
    plot(umap_emb_path, sample_label_path, plot_path)

def plot(umap_path, sample_label_path, plot_path, cols = 4):
    print("cols",cols)
    umap_embedding = np.load(umap_path)
    print(umap_embedding.shape)
    sample_names = np.load(sample_label_path)
    unique_samples = np.unique(sample_names)
    print(f'Loaded {len(sample_names)} samples, {len(unique_samples)} unique samples')

    rows = (len(unique_samples) - 1) // cols + 1
    print("rows,cols: ", rows, cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))

    for sample_i in tqdm(range(len(unique_samples))):
        print("sample i : ", sample_i)
        sample_name = unique_samples[sample_i]
        print("sample name : ", sample_name)
        x = sample_i // cols
        y = sample_i % cols
        print("x,y",x,y)
        #sample_color = plt.cm.get_cmap('viridis')(sample_i / len(unique_samples))
        sample_color = np.array([94, 47, 235, 255]) / 255
        sample_idices = sample_names == sample_name
     
        axes[x, y].scatter(umap_embedding[:, 0],
                           umap_embedding[:, 1], s = 1, c = np.array([241, 209, 97, 255]) / 255, label = 'all')
        axes[x, y].scatter(umap_embedding[sample_idices, 0],
                           umap_embedding[sample_idices, 1], s = 1, c = sample_color, label = sample_name)

        axes[x, y].set_title(f'{sample_name} (n = {np.sum(sample_idices)})')
        axes[x, y].set_xticks([])
        axes[x, y].set_yticks([])
        axes[x, y].legend()

    # Clean up empty axes
    for i in range(len(unique_samples), rows * cols):
        axes[i // cols, i % cols].axis('off')

    for ax in axes.flat:
        ax.label_outer()
    plt.savefig(plot_path, bbox_inches='tight')
    plt.clf()

if __name__ == '__main__':
    main()
