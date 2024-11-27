import os
import numpy as np
import visualization.utils as utils
import skimage.io as io

def visualize_sample(root_path, sample_name, color_dict, strength_dict, data_type, downsample = None):
    save_path = f'{root_path}/{sample_name}/visualization'
    os.makedirs(save_path, exist_ok=True)
    if data_type == 'WSI' and downsample is None:
        print(f'Downsample not provided for WSI {sample_name}, setting to 20')
        downsample = 20
    if data_type == 'TMA' and downsample is None:
        print(f'Downsample not provided for TMA {sample_name}, setting to 1')
        downsample = 1

    data, channels = utils.load_zarr_w_channel(root_path, sample_name)
    
    # Downsample the image
    if downsample != 1:
        assert type(downsample) == int, 'Downsample must be an integer'
        data = data[:, ::downsample, ::downsample]

    # Rearrange data according to channel order
    channel_order = channels['channel'].values
    data = data[channel_order]

    image = utils.vis_multiplex(data, channels['marker'].values, color_dict, strength_dict = strength_dict)
    io.imsave(f'{save_path}/sample.png', image)
