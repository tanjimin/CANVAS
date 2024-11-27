from utils import helper
from types import SimpleNamespace
import pdb
import numpy as np
import os

from canvas.analysis.analysis_modules.gen_cluster_by_marker import gen_cluster_by_marker
from canvas.analysis.analysis_modules.gen_clinical_corr import gen_clinical_corr
from canvas.analysis.analysis_modules.gen_cluster_plot_marker_celltype_clinical_embedding_based import gen_cluster_plot_marker_celltype_clinical_embedding_based
from canvas.analysis.analysis_modules.gen_cluster_color import gen_cluster_color

def run_signature_characterization(config):
    out_dir = os.path.join(config.data_root, config.processed_data_dir)
    out_dir_analysis = os.path.join(config.data_root, config.analysis_dir)
    kmeans_path = os.path.join(out_dir_analysis, 'kmeans', str(config.n_clusters))
    tile_embedding_path = os.path.join(out_dir_analysis, 'tile_embedding') 
    out_path_marker_celltype_clinical = os.path.join(out_dir_analysis, 'marker_celltype_clinical', 
                                  str(config.n_clusters))

    #open channel_names
    channel_names = os.path.join(out_dir, 'data', 'common_channels.txt')
    common_channels_open = open(channel_names, "r")
    common_channels_read = common_channels_open.read() 
    common_channels = common_channels_read.split("\n")

    #OUTPUT PATHS
    output_path_cluster_by_marker = os.path.join(out_dir_analysis, 'cluster_by_marker')
    output_path_clinical_corr = os.path.join(out_dir_analysis, 'clinical_corr', str(config.n_clusters))

    #functions
    gen_cluster_color(tile_embedding_path = tile_embedding_path, kmeans_path = kmeans_path, 
                      num_channels = config.num_channels, common_channels = common_channels, 
                      color_map = config.color_map, n_clusters = config.n_clusters)
    gen_cluster_by_marker(tile_embedding_path = tile_embedding_path, kmeans_path = kmeans_path,
                          output_path = output_path_cluster_by_marker, channel_names = common_channels) 
    gen_clinical_corr(kmeans_path = kmeans_path, tile_embedding_path = tile_embedding_path, clinical_df_path = config.Clinical_Data_path, save_path = output_path_clinical_corr)
    gen_cluster_plot_marker_celltype_clinical_embedding_based(clinical_path = config.Clinical_Data_path, clinical_corr_path = output_path_clinical_corr, 
                                                              marker_heatmap_path = output_path_cluster_by_marker, kmeans_path = kmeans_path, 
                                                              channel_names = common_channels, output_path = out_path_marker_celltype_clinical,
                                                              cell_type_info_exists = False, clinical_info_exists = False,
                                                              cell_counts_heatmap_path = False)
    
    #only run these if you have cell type information!!! (not sure if I should delete)
    # output_position_path = os.path.join(out_base_dir_cell_vs_cluster, out_base_dir_ext, "position_by_tile.npz")
    # output_celltype_path = os.path.join(out_base_dir_cell_vs_cluster, out_base_dir_ext, "celltype_by_tile.npz")
    #gen_cell_position(root_path = config.root_path, mask_id = config.mask_id, output_position_path = output_position_path, output_celltype_path = output_celltype_path)
    #gen_cell_vs_cluster_analysis(position_by_tile_path, celltype_by_tile_path, cluster_path, output_file, output_csv_file) 



   