from utils import helper
from types import SimpleNamespace
import pdb
import numpy as np
import os
import pandas as pd

from canvas.analysis.analysis_modules.gen_core_graph import gen_core_graph
from canvas.analysis.analysis_modules.gen_core_stats import gen_core_stats

def run_graph_based_analysis(config):
    out_dir = os.path.join(config.data_root, config.processed_data_dir)
    out_dir_analysis = os.path.join(config.data_root, config.analysis_dir)
    kmeans_path = os.path.join(out_dir_analysis, 'kmeans', str(config.n_clusters))
    tile_embedding_path = os.path.join(out_dir_analysis, 'tile_embedding') 
    out_path_graph = os.path.join(out_dir_analysis, 'sample_graph', 
                                  str(config.n_clusters))

    gen_core_graph(kmeans_path = kmeans_path, tile_embedding_path = tile_embedding_path,
                   analysis_graph_dir = out_path_graph, WSI_subset_regions = config.WSI_subset_regions_file, 
                   subset_region_w = config.image_sub_w, subset_region_h = config.image_sub_h,
                   local_region = config.local_region, tile_size = config.tile_size) 
    gen_core_stats(analysis_graph_dir = out_path_graph, tile_embedding_path = tile_embedding_path,
                   WSI_subset_regions = config.WSI_subset_regions_file, local_region = config.local_region) 
