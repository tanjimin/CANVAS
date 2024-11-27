import os
import numpy as np
from utils.plotting import color_umap
import pdb

def gen_umap(data_root, save_path, emb_path, input_pixel_per_um, inference_window_um):
    os.makedirs(save_path, exist_ok = True)
    # Embedding visualization
    embedding_mean_path = os.path.join(emb_path,'embedding_mean.npy')
    sample_label_path = os.path.join(emb_path,'sample_name.npy')
    data_base_path = os.path.join(data_root, 'processed_data/data')
    image_mean_path = os.path.join(emb_path, 'image_mean.npy')
    umap_visualization(embedding_mean_path, save_path, sample_label_path, data_base_path, image_mean_path)

def umap_visualization(embedding_mean_path, save_path, sample_label_path, data_base_path, image_mean_path):
    #save_path = f'{save_path}/umap'
    # Umap coordinates
    #os.makedirs(embedding_mean_path, exist_ok = True)
    umap_coord_path = get_umap_coord(save_path, embedding_mean_path) # umap
    #plot_mosaic(umap_coord_path, save_path, data_path, input_pixel_per_um, inference_window_um)
    save_path = f'{save_path}/umap'
    color_by_sample(umap_coord_path, save_path, sample_label_path)
    color_by_marker(umap_coord_path, save_path, data_base_path, image_mean_path)

def plot_mosaic(umap_coord_path, save_path, data_path, input_pixel_per_um, inference_window_um):
    # Load dataset to extract images
    from canvas.inference.infer import load_dataset
    tile_size = input_pixel_per_um * inference_window_um
    print("tile size:", tile_size)
    dataloader = load_dataset(data_path, tile_size)
    from canvas.analysis.dim_reduction.gen_umap_mosaic import plot_umap_mosaic
    from canvas.visualization.utils import vis_multiplex
    save_file = os.path.join(save_path, 'umap_mosaic.pdf')
    # Plot example tiles on UMAP
    plot_umap_mosaic(umap_coord_path, save_file, dataloader, vis_multiplex, run_config)

def color_by_sample(umap_coord_path, save_path, sample_label_path):
    from analysis.dim_reduction.color_by_sample import plot
    plot_path = os.path.join(save_path, 'umap_sample.png')
    plot(umap_coord_path, sample_label_path, plot_path, cols = 4)

def color_by_marker(umap_coord_path, save_path, data_base_path, image_mean_path):
    from analysis.dim_reduction.color_by_marker import plot
    plot_path = os.path.join(save_path, 'umap_marker.png')
    from utils.helper import read_channel_file
    common_channels = read_channel_file(data_base_path + '/common_channels.txt')
    plot(umap_coord_path, image_mean_path, plot_path, common_channels, cols = 4)

def get_umap_coord(save_path, embedding_mean_path, output_suffix = 'umap'):
    output_path = f'{save_path}/{output_suffix}'
    print("umap output path: ", output_path)
    save_file = os.path.join(output_path, 'coord.npy')
    os.makedirs(output_path, exist_ok = True)
    if os.path.exists(save_file):
        print('UMAP coord already exist, skipping')
        return save_file
    from analysis.dim_reduction.gen_umap_embedding import plot_umap
    plot_umap(embedding_mean_path, save_file)
    return save_file

if __name__ == '__main__':
    main()
