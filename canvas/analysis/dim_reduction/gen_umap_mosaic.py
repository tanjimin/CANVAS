import sys
import torch
from tqdm import tqdm
import numpy as np

from PIL import Image
from canvas.utils.helper import read_channel_file, load_channel_yaml_file


def plot_umap_mosaic(umap_data, plot_path, dataloader, vis_multiplex, run_config, img_size = 64):
    common_channels = read_channel_file(run_config['output_path'] + '/data/common_channels.txt')
    color_dict = load_channel_yaml_file(run_config['selected_channel_color_file'])

    # Load umap embeddings
    if isinstance(umap_data, str):
        embeddings = np.load(umap_data)
    else:
        embeddings = umap_data

    # Load dataset and visualizer
    dataset = dataloader.dataset
    dataset.use_normalization = False

    # Normalize frames
    xs = embeddings[:, 0]
    ys = embeddings[:, 1]
    x_range = xs.min(), xs.max() 
    y_range = ys.min(), ys.max() 
    margin = 0.1
    x_norm = (xs - x_range[0]) / (x_range[1] - x_range[0]) * (1 - 2 * margin) + margin
    y_norm = (ys - y_range[0]) / (y_range[1] - y_range[0]) * (1 - 2 * margin) + margin

    # Get frame from umap
    frame_x = 3000
    frame_y = 3000

    x_coord = x_norm * frame_x
    y_coord = y_norm * frame_y
    frame = np.zeros((frame_x, frame_y, 3)).astype(np.uint8)# + 255

    # For each grid get a image and paste on the frame
    bin_size = int(img_size * 1.5)
    r = int(img_size / 2)

    for idx_x in tqdm(range(frame_x // bin_size)):
        for idx_y in range(frame_y // bin_size):
            patch_x_range = idx_x * bin_size, (idx_x + 1) * bin_size
            patch_y_range = idx_y * bin_size, (idx_y + 1) * bin_size
            x_index_in_range = np.logical_and(x_coord >= patch_x_range[0], x_coord < patch_x_range[1])
            y_index_in_range = np.logical_and(y_coord >= patch_y_range[0], y_coord < patch_y_range[1])
            qualified_patch_bool = np.logical_and(x_index_in_range, y_index_in_range)
            qualified_patches = np.where(qualified_patch_bool)[0]
            if len(qualified_patches) > 0:
                np.random.seed(0)
                selected_sample_idx = selected_real_idx = qualified_patches[np.random.choice(range(len(qualified_patches)), 1)[0]]

                img_tensor = dataset[selected_real_idx][0]
                img_array = np.array(img_tensor.detach().cpu())
                #img_array = vis_codex(img_array * 8)
                img_array = vis_multiplex(img_array, common_channels, color_dict)
                img_array = img_array.astype(np.uint8)
                img = Image.fromarray(img_array)

                img = np.array(img.resize((img_size, img_size)))
                coord = int(x_coord[selected_sample_idx]), int(y_coord[selected_sample_idx])
                frame[coord[0] - r : coord[0] + r, coord[1] -r : coord[1] + r] = img[:2 * r, :2 * r] # Make sure it work for odd resolutions
                # Add a white border
                # Add top border
                frame[coord[0] - r : coord[0] - r + 1, coord[1] -r : coord[1] + r] = 255
                # Add bottom border
                frame[coord[0] + r - 1 : coord[0] + r, coord[1] -r : coord[1] + r] = 255
                # Add left border
                frame[coord[0] - r : coord[0] + r, coord[1] -r : coord[1] - r + 1] = 255
                # Add right border
                frame[coord[0] - r : coord[0] + r, coord[1] + r - 1 : coord[1] + r] = 255

    # save image
    #frame = np.moveaxis(frame, 0, 1)
    frame = np.rot90(frame)
    img = Image.fromarray(frame)

    img.save(plot_path)
