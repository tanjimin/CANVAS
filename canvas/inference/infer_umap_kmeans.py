from utils import helper
from types import SimpleNamespace
import pdb
import numpy as np
import os

def run_infer_umap_kmeans(config):
    # import python files
    from canvas.inference.infer import infer
    from canvas.analysis.main import gen_umap
    from canvas.analysis.clustering.kmeans import kmeans
    
    out_dir_analysis = os.path.join(config.data_root, config.analysis_dir)
    kmeans_path = os.path.join(out_dir_analysis, 'kmeans', str(config.n_clusters))
    tile_embedding_path = os.path.join(out_dir_analysis, 'tile_embedding') 
    umap_path = os.path.join(out_dir_analysis, 'umap')
    model_path = os.path.join(config.data_root, config.model_dir, f'checkpoint-{config.ckpt_num}.pth')

    # Run inference
    post_normalization_scaling_strategy = config.post_normalization_scaling_strategy if hasattr(config, 'post_normalization_scaling_strategy') else None
    cap_cutoff = config.cap_cutoff if hasattr(config, 'cap_cutoff') else None
    perc_thres = config.perc_thres if hasattr(config, 'perc_thres') else None

    infer(data_root = config.data_root, processed_data_path = config.processed_data_dir, post_normalization_scaling_strategy = post_normalization_scaling_strategy, 
          cap_cutoff = cap_cutoff, perc_thres = perc_thres, 
          tiles_dir = config.tiles_dir, model_path = model_path, save_path = out_dir_analysis, 
          input_pixel_per_um = config.input_pixel_per_um,  inference_window_um = config.inference_window_um) 

    gen_umap(data_root = config.data_root, save_path = out_dir_analysis, emb_path = tile_embedding_path, input_pixel_per_um = config.input_pixel_per_um, 
             inference_window_um= config.inference_window_um) 

    kmeans(umap_path = umap_path, emb_path = tile_embedding_path, n_clusters = config.n_clusters, save_path = kmeans_path)

