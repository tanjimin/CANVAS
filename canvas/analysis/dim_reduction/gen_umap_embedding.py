import sys
import os
import numpy as np
import umap.umap_ as umap
import matplotlib.pyplot as plt

def main():
    emb_path = sys.argv[1]
    umap_emb_path = sys.argv[2]
    plot_umap(emb_path, umap_emb_path)

def plot_umap(emb_path, umap_emb_path):
    if type(emb_path) == str:
        embedding = np.load(emb_path)
    else:
        embedding = emb_path
    print(f'Embedding shape: {embedding.shape}')

    print('Generating umap')
    # PCA init
    from sklearn.decomposition import PCA
    if embedding.shape[1] > 1000:
        n_components = 1000
    else:
        n_components = embedding.shape[1]
    pca = PCA(n_components=n_components)
    data_pc = pca.fit_transform(embedding)
    print(f'PCA shape: {data_pc.shape}')
    umap_embedding = umap.UMAP(random_state=2021, init = 'spectral').fit_transform(data_pc)
    np.save(umap_emb_path, umap_embedding)

    plt.scatter(umap_embedding[:, 0], umap_embedding[:, 1], s = 0.3)
    plt.savefig(umap_emb_path.replace('.npy', '.png'))
    plt.clf()

if __name__ == '__main__':
    main()
