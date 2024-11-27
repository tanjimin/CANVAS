from utils import helper
from types import SimpleNamespace
import pdb
import numpy as np
import os
import pandas as pd

from canvas.analysis.analysis_modules.gen_sample_images import gen_sample_images
from canvas.analysis.analysis_modules.gen_cluster_color import gen_cluster_color
from canvas.analysis.analysis_modules.gen_cluster_sample_map import gen_cluster_sample_map

def run_mapping_to_imgs(config):
    out_dir = os.path.join(config.data_root, config.processed_data_dir)
    out_dir_analysis = os.path.join(config.data_root, config.analysis_dir)
    kmeans_path = os.path.join(out_dir_analysis, 'kmeans', str(config.n_clusters))
    tile_embedding_path = os.path.join(out_dir_analysis, 'tile_embedding') 
    out_path_sample_visualization = os.path.join(out_dir_analysis, 'sample_visualization', 
                                  str(config.n_clusters))

    common_channels_open = open(os.path.join(out_dir, 'data', 'common_channels.txt'), "r")
    common_channels_read = common_channels_open.read() 
    common_channels = common_channels_read.split("\n") 

    color_map = helper.load_channel_yaml_file(os.path.join(config.config_root, config.selected_channel_color_file))
    color_strength = helper.load_channel_yaml_file(os.path.join(config.config_root, config.channel_strength_file))
    color_map_tuple = (color_map, color_strength)

    gen_sample_images(out_dir = out_dir, analysis_sample_vis_path = out_path_sample_visualization,
                     tile_embedding_path= tile_embedding_path, common_channels = common_channels, color_map = color_map_tuple,
                      WSI_subset_regions = config.WSI_subset_regions_file, subset_region_w = config.image_sub_w, 
                      subset_region_h = config.image_sub_h,local_region = config.local_region)
    
    gen_cluster_color(tile_embedding_path = tile_embedding_path, kmeans_path= kmeans_path,
                      num_channels = config.num_channels, common_channels = common_channels, 
                      color_map = color_map_tuple, n_clusters = config.n_clusters)

    
    gen_cluster_sample_map(sample_vis_path = out_path_sample_visualization, n_clusters = config.n_clusters, 
                           tile_size = config.tile_size, kmeans_path = kmeans_path, tile_embedding_path = tile_embedding_path,
                           image_type = config.data_type, WSI_subset_regions = config.WSI_subset_regions_file, 
                           subset_region_w = config.image_sub_w, subset_region_h = config.image_sub_h,
                           local_region = config.local_region)
    

