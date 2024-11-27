import numpy as np

def main():
    cluster_label_path = '/gpfs/scratch/jt3545/projects/CODEX/analysis/kidney/analysis/clustering/labels.npy'
    image_mean_path = '/gpfs/scratch/jt3545/projects/CODEX/analysis/kidney/analysis/tile_embedding/image_mean.npy'
    plot_color_path = '/gpfs/scratch/jt3545/projects/CODEX/analysis/kidney/analysis/clustering/plot_color_rgb.npy'
    gen_color(cluster_label_path, image_mean_path, plot_color_path)

    '''
    plot_color_rgb = np.load('/gpfs/scratch/jt3545/projects/CODEX/analysis/kidney/analysis/clustering/plot_color_rgb.npy')

    all_umap_embedding = np.load('/gpfs/scratch/jt3545/projects/CODEX/analysis/kidney/analysis/umap/coord.npy')
    cluster_label = np.load('/gpfs/scratch/jt3545/projects/CODEX/analysis/kidney/analysis/clustering/labels.npy')


    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize = (10, 10))
    unique_clusters = plot_color_rgb.shape[0]
    for cluster in range(unique_clusters):
        umap_embedding = all_umap_embedding[cluster_label == cluster]
        ax.scatter(umap_embedding[:, 0], umap_embedding[:, 1], s = 1, c = plot_color_rgb[cluster], alpha = 0.5)

    plt.title('Kmeans clusters')
    plt.legend(bbox_to_anchor=(1.1, 1.05))
    plt.savefig('test.pdf')
    plt.clf()
    '''

def gen_color(cluster_label_path, image_mean_path, plot_color_path):
    cluster_names = np.load(cluster_label_path)
    unique_clusters = np.unique(cluster_names)
    image_mean = np.load(image_mean_path)
    n_c = image_mean.shape[-1]
    color = np.zeros((len(unique_clusters), n_c))
    color_rgb = np.zeros((len(unique_clusters), 3))
    for cluster in unique_clusters:
        cluster_mask = cluster_names == cluster
        color[cluster] = image_mean[cluster_mask].mean(axis = 0)
    
        import sys
        sys.path.append('/gpfs/data/tsirigoslab/home/jt3545/CODEX/codex-imaging/src/model/feature_extraction/imaging_modality/imc/modules/stories')
        from no_marker.gen_sample_images import vis_codex
        color_img = np.repeat(color[cluster].reshape(1, 1, n_c), 10, axis = 0)
        color_img = np.repeat(color_img, 10, axis = 1)
        color_rgb[cluster] = vis_codex(color_img)[0, 0]

    plotting_colors = np.zeros((len(unique_clusters), 3))

    for cluster in unique_clusters:
        cluster_mask = cluster_names == cluster
        color = color_rgb[cluster] / 255
        #mean_color = color_rgb.mean(axis = 0) / 255
        #std_color = color_rgb.std(axis = 0) / 255
        #color = (color - mean_color) / (std_color)
        #color = np.log1p(np.log1p(color))
        # Add noise
        np.random.seed(cluster)
        color = color + np.random.normal(0, 1, 3) * (color.std(axis = 0) * 0.5)
        #color = (color - color.mean()) / (color.std() * 2+ 0.01) + 0.4
        color = color / (color.std() + 0.01)
        color = color + np.random.normal(0, 0.01, 3)
        color = color / (color.max() + 0.01)
        color = np.clip(color, 0, 1)
        color = color * 0.7 + 0.2
        plotting_colors[cluster] = color

    np.save(plot_color_path, plotting_colors)
    return plotting_colors

if __name__ == '__main__':
    main()
