import numpy as np
import pandas as pd
import zarr

def load_zarr_w_channel(root_path, sample_name):
    sample_zarr_path = f'{root_path}/{sample_name}/data.zarr'
    channel_path = f'{root_path}/{sample_name}/channels.csv'
    data = zarr.open(sample_zarr_path, mode='r')
    channels = pd.read_csv(channel_path)
    return data, channels

def vis_multiplex(data, channels, color_dict, strength_dict = None):
    # Initialize black image
    image = np.zeros((data.shape[1], data.shape[2], 3))
    data = np.array(data).astype(np.float32) / 255
    for channel, color in color_dict.items():
        channel_index = np.where(np.array(channels) == channel)[0][0]
        if strength_dict:
            strength = strength_dict[channel]
        else:
            strength = 1
        # From 0-255 to 0-1
        color = np.array(color) / 255
        for channel_i in range(3):
            image[:, :, channel_i] = np.maximum(image[:, :, channel_i], data[channel_index] * color[channel_i] * strength)
    image = (image * 255).clip(0, 255).astype(np.uint8)
    return image