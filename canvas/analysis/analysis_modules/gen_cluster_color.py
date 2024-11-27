import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import pdb

def gen_cluster_color(tile_embedding_path, kmeans_path, num_channels, common_channels, color_map, n_clusters):
    
    cluster_label_path = os.path.join(kmeans_path, 'clusters.npy') 
    image_mean_path = os.path.join(tile_embedding_path, 'image_mean.npy') 
    color_path = os.path.join(kmeans_path, 'cluster_color.npy') 
    rgb_color_path = os.path.join(kmeans_path, 'cluster_color_rgb.npy') 
    plot_color_path = os.path.join(kmeans_path, 'cluster_plot_color_rgb.npy') 
    
    color, color_rgb = gen_color(cluster_label_path, image_mean_path, color_path, rgb_color_path, num_channels, common_channels, color_map)
    plot_umap_by_cluster(tile_embedding_path, cluster_label_path, color_path, color_rgb, plot_color_path, n_clusters)

def gen_color(cluster_label_path, image_mean_path, color_path, rgb_color_path, num_channels, common_channels, color_map):
    color_map, color_strength = color_map
    #pdb.set_trace()
    cluster_names = np.load(cluster_label_path)
    unique_clusters = np.unique(cluster_names)
    image_mean = np.load(image_mean_path)

    # Scale image mean to 0-1
    #color = np.zeros((len(unique_clusters), 18))
    color = np.zeros((len(unique_clusters), num_channels))
    color_rgb = np.zeros((len(unique_clusters), 3))
    for cluster in unique_clusters:
        #pdb.set_trace()
        cluster_mask = cluster_names == cluster
        color[cluster] = image_mean[cluster_mask].mean(axis = 0)
    
        color_img = np.repeat(color[cluster].reshape(1, 1, num_channels), 10, axis = 0)
        color_img = np.repeat(color_img, 10, axis = 1)
        #from canvas.analysis.analysis_modules.gen_cluster_sample_map import vis_codex_values
        #color_rgb[cluster] = vis_codex_values(color_img, common_channels, color_map)[0, 0]
        import visualization.utils as utils
        color_img = np.transpose(color_img, (2, 0, 1))  # [ x_dim, y_dim, num_channels] --> [num_channels, x_dim, y_dim] 
        color_rgb[cluster] = utils.vis_multiplex(color_img, common_channels, color_map, color_strength)[0, 0]
    np.save(color_path, color)
    np.save(rgb_color_path, color_rgb)
    return color, color_rgb

def plot_umap_by_cluster(tile_embedding_path, cluster_label_path, color_path, color_rgb, plot_color_path, n_clusters):
    umap_embedding = np.load(os.path.join(tile_embedding_path,"embedding_mean.npy"))
    cluster_names = np.load(cluster_label_path)
    unique_clusters = np.unique(cluster_names)

    # Set color map
    plotting_colors = np.zeros((len(unique_clusters), 3))

    for cluster in unique_clusters:
        cluster_mask = cluster_names == cluster
        color = color_rgb[cluster] / 255
        mean_color = color_rgb.mean(axis = 0) / 255
        std_color = color_rgb.std(axis = 0) / 255
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