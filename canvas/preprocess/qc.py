import os
import zarr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import visualization.histogram as hist 
import utils.helper as utils

def sample_hist(root_path, sample_name, qc_path, bins=100, sample_max = 10000):
    marker_stats_dict = get_channel_data(root_path, sample_name, sample_max)
    qc_save_path = f'{qc_path}/{sample_name}'
    os.makedirs(qc_save_path, exist_ok=True)
    hist.multipanel_hist(marker_stats_dict, save_path = f'{qc_save_path}/channel_hist', bins=bins, title=sample_name, xlabel='Marker Intensity', ylabel='Frequency')
    hist.multipanel_hist(marker_stats_dict, save_path = f'{qc_save_path}/channel_hist_log', bins=bins, title=sample_name, xlabel='Marker Intensity', ylabel='log(Frequency)', log_y = True)

def global_hist(root_path, sample_list, qc_save_path, bins=100, sample_max = 100000):
    print('Generating global histogram')
    qc_save_path = f'{qc_save_path}/global'
    os.makedirs(qc_save_path, exist_ok=True)
    merged_marker_stats_dict = gen_global_dist_data(root_path, sample_list, qc_save_path, sample_max)

def plot_hist(merged_marker_stats_dict, save_path, bins=100, title='Global', xlabel='Marker Intensity', ylabel='Frequency'):
    hist.multipanel_hist(merged_marker_stats_dict, save_path = save_path, bins=bins, title=title, xlabel=xlabel, ylabel=ylabel)
    hist.multipanel_hist(merged_marker_stats_dict, save_path = save_path + '_log', bins=bins, title=title + ' log frequency', xlabel=xlabel, ylabel='log(Frequency)', log_y = True)

def calculate_normalization_stats(root_path, sample_list, qc_save_path, sample_max = 100000, tile_size = None):
    qc_save_path = f'{qc_save_path}/normalization'
    os.makedirs(qc_save_path, exist_ok=True)
    data_dict = gen_global_dist_data(root_path, sample_list, qc_save_path, tile_size = tile_size)
    plot_hist(data_dict, save_path = f'{qc_save_path}/channel_hist_tiled_{tile_size}')
    data_dict, norm_stats_df = normalize_channels(data_dict)
    norm_stats_df.to_csv(f'{qc_save_path}/normalization_stats.csv', index=False)
    plot_hist(data_dict, save_path = f'{qc_save_path}/channel_hist_tiled_{tile_size}_normalized')

def normalize_channels(data_dict):
    '''
    Normalize data_dict to have mean 0 and std 1
    '''
    norm_stats_list = []
    for key in data_dict.keys():
        data = data_dict[key]
        mean = np.mean(data)
        std = np.std(data)
        data_dict[key] = (data - mean) / std
        norm_stats_list.append({'marker': key, 'mean': mean, 'std': std})
    return data_dict, pd.DataFrame(norm_stats_list)

def gen_global_dist_data(root_path, sample_list, qc_save_path, sample_max = 100000, tile_size = None):
    if tile_size is None:
        save_file_path = f'{qc_save_path}/global_hist.npz'
    else:
        save_file_path = f'{qc_save_path}/global_hist_tiled_{tile_size}.npz'
    if os.path.exists(save_file_path):
        print('Loading global histogram data as a dictionary')
        data = np.load(save_file_path, allow_pickle=True)
        data_dict = {}
        for key in data.files:
            data_dict[key] = data[key]
        return data_dict
    merged_marker_stats_dict = {}
    total_samples = len(sample_list)
    for sample_name in tqdm(sample_list):
        sub_sample_max = sample_max // total_samples
        marker_stats_dict = get_channel_data(root_path, sample_name, sub_sample_max, tile_size)
        for marker in marker_stats_dict.keys():
            if marker not in merged_marker_stats_dict:
                merged_marker_stats_dict[marker] = []
            merged_marker_stats_dict[marker].extend(marker_stats_dict[marker])
    # Save global distribution data as npz
    np.savez(save_file_path, **merged_marker_stats_dict)
    plot_hist(merged_marker_stats_dict, save_path = f'{qc_save_path}/channel_hist')
    return merged_marker_stats_dict

def get_channel_data(root_path, sample_name, sample_max, tile_size):
    '''
    Return data stats for all channels in the zarr file
    '''
    data, channels = utils.load_zarr_w_channel(root_path, sample_name)
    marker_channel_dict = dict(zip(channels['marker'], channels['channel']))
    marker_stats_dict = {}
    for marker in marker_channel_dict.keys():
        channel = marker_channel_dict[marker]
        if tile_size is None:
            data_channel = data[channel, :, :].flatten()
        else:
            data_channel = get_tile_region(data[channel, :, :], root_path, sample_name, tile_size)
        if len(data_channel) > sample_max: # Sample data if too large
            data_channel = np.random.choice(data_channel, sample_max, replace=False)
        marker_stats_dict[marker] = data_channel
    return marker_stats_dict

def get_tile_region(channel_img, root_path, sample_name, tile_size):
    # Load tile information
    tile_df = utils.load_tile_info(root_path, sample_name, tile_size)
    tile_regions = []
    for index, row in tile_df.iterrows():
        x, y = row['h'], row['w']
        tile_region = channel_img[y:y+tile_size, x:x+tile_size].flatten()
        tile_regions.append(tile_region)
    if len(tile_regions) > 0:
        tile_regions_flat = np.concatenate(tile_regions).flatten()
    else:
        tile_regions_flat = []
    return tile_regions_flat