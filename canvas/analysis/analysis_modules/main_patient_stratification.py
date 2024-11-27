from utils import helper
from types import SimpleNamespace
import pdb
import numpy as np
import os
import sys

from canvas.analysis.analysis_modules.gen_patient_grouping_kmeans import gen_patient_grouping_kmeans
from canvas.analysis.analysis_modules.gen_cluster_vs_sample import gen_cluster_vs_sample

def run_patient_stratification(config):
    out_dir = os.path.join(config.data_root, config.processed_data_dir)
    out_dir_analysis = os.path.join(config.data_root, config.analysis_dir)
    kmeans_path = os.path.join(out_dir_analysis, 'kmeans', str(config.n_clusters))
    tile_embedding_path = os.path.join(out_dir_analysis, 'tile_embedding') 
    out_path_cluster_vs_sample = os.path.join(out_dir_analysis, 'cluster_vs_sample', 
                                             str(config.n_clusters))

    #OUTPUT PATHS
    #gen_cluster_vs_sample.py
    # out_base_dir_cluster_vs_sample = os.path.join(analysis_base_path, config.cluster_vs_sample_dir)
    # out_base_dir_ext = os.path.join(config.mask_id, str(config.tile_size), config.model_weights, config.augmentation, str(config.n_clusters)) #this is the rest of the path

    # save_path = os.path.join(out_base_dir_cluster_vs_sample, out_base_dir_ext)
    # output_cluster_sample_enrichment = os.path.join(out_base_dir_cluster_vs_sample, out_base_dir_ext, 'cluster_with_sample_enrichment.png')
    # output_cluster_sample_csv = os.path.join(out_base_dir_cluster_vs_sample, out_base_dir_ext, 'cluster_with_sample.csv')

    gen_cluster_vs_sample(kmeans_path = kmeans_path, tile_embedding_path = tile_embedding_path, cluster_save_path = out_path_cluster_vs_sample)
    gen_patient_grouping_kmeans(save_path = out_path_cluster_vs_sample, clinical_path = config.Clinical_Data_path, n_clusters = config.n_clusters, clinical_var_labels = config.clinical_var_labels)