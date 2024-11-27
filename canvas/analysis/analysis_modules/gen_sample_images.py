import sys
import os
import numpy as np
import pandas as pd
import skimage.io as io
import zarr
from utils import helper
from types import SimpleNamespace
import pdb

def gen_sample_images(out_dir, analysis_sample_vis_path, tile_embedding_path, 
                      common_channels, color_map, WSI_subset_regions, 
                      subset_region_w, subset_region_h, local_region = False):

    sample_name_path = os.path.join(tile_embedding_path, 'sample_name.npy')
    sample_names = np.load(sample_name_path, mmap_mode='r')

    for sample_id in np.unique(sample_names):
        print(sample_id)
    
        
        # Construct directory path
        img_sample_path = os.path.join(analysis_sample_vis_path, sample_id)

        # Create the directory if it does not exist
        os.makedirs(img_sample_path, exist_ok=True)

        # Construct full file paths
        color_image_path = os.path.join(img_sample_path, 'color.png')
        intensity_image_path = os.path.join(img_sample_path, 'intensity.png')

        #instead of loading in the numpy file, lets load in the zarr!!
        data_path = f'{out_dir}/data/{sample_id}/data.zarr'
        zarr_file = zarr.open(data_path, mode='r')

        channel_path = data_path.replace('data.zarr', 'channels.csv')
        channels_csv = pd.read_csv(channel_path)

        common_channels_ids = []
        for index,row in channels_csv.iterrows():
            if row['marker'] in common_channels:
                common_channels_ids.append(row['channel'])

        markers = channels_csv['marker'][common_channels_ids]

        if local_region:
            WSI_subset_regions_samplewise = (WSI_subset_regions[WSI_subset_regions['Sample'] == sample_id]).reset_index(drop=True)
            for index,row in WSI_subset_regions_samplewise.iterrows():
                core_zarr = zarr_file[common_channels_ids,WSI_subset_regions_samplewise['h1'][index]:WSI_subset_regions_samplewise['h1'][index]+subset_region_h,WSI_subset_regions_samplewise['w1'][index]:WSI_subset_regions_samplewise['w1'][index]+subset_region_w]
                core = (core_zarr.astype(np.float64) / 255.0) #* scaling factor 
                gen_color(core_zarr, color_image_path, markers, color_map)
                gen_intensity(core_zarr, intensity_image_path)
        else:
            core_zarr = zarr_file[common_channels_ids,:,:]
            core = (core_zarr.astype(np.float64) / 255.0) #* scaling factor 
            gen_color(core_zarr, color_image_path, markers, color_map)
            gen_intensity(core_zarr, intensity_image_path)

    
def gen_color(core, color_image_path, common_channels, color_map):
    color_map, color_strength = color_map
    import visualization.utils as utils
    core_color = utils.vis_multiplex(core, common_channels, color_map, color_strength)
    color_image_dir = os.path.dirname(color_image_path)
    os.makedirs(color_image_dir, exist_ok=True)
    io.imsave(color_image_path, core_color)

def gen_intensity(core, intensity_image_path):
    core = np.transpose(core, (1, 2, 0))  # [num_channels, x_dim, y_dim]
    brightness_factor = .1
    core_intensity = np.clip(np.mean(core * brightness_factor, axis = 2), 0, 1)
    print(intensity_image_path)
    io.imsave(intensity_image_path, (core_intensity*255).astype(np.uint8))

if __name__ == '__main__':
    main()
