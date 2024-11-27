import sys
import os
import numpy as np
import matplotlib.pyplot as plt

def main():
    umap_emb_path = sys.argv[1]
    celltype_by_tile_path = sys.argv[2]
    output_path = sys.argv[3]
    celltype_by_tile = np.load(celltype_by_tile_path)

    umap_embedding = np.load(umap_emb_path)

    cellcounts_per_tile = [len(celltype_by_tile[str(idx)]) for idx in range(len(celltype_by_tile))]

    np.save(output_path, umap_embedding)

    # Plot
    plt.scatter(umap_embedding[:, 0], umap_embedding[:, 1], s = 0.3, c = cellcounts_per_tile, cmap = 'viridis')
    plt.colorbar()
    plt.savefig(output_path.replace('.npy', '.png'))

if __name__ == '__main__':
    main()
