import os
from tqdm import tqdm
import utils.helper as utils
import pdb
from utils import helper
from types import SimpleNamespace
import canvas.visualization.multiplex_image as multi_img

def run_preprocess(config):
    output_path = os.path.join(config.data_root, 'processed_data')
    data_path = f'{output_path}/data'
    qc_path = f'{output_path}/qc'
    image_path = os.path.join(config.data_root, config.input_path, config.raw_image_path)
    dummy_input_path = os.path.join(config.data_root, config.input_path, 'dummy_input')
    os.makedirs(dummy_input_path, exist_ok=True)

    # Step 1: Convert tiff to zarr and QC
    zarr_conversion(config.data_root, image_path, config.input_ext, data_path, qc_path, dummy_input_path)

    # Step 2: Choose channels with colormap and visualize individual images
    visualize_samples(config.data_type, dummy_input_path, config.input_ext, output_path, 
                      config.config_root, config.selected_channel_color_file, 
                      config.channel_strength_file, data_path)

    # Step 3: Generate tiles
    ROI_path = config.ROI_path if hasattr(config, 'ROI_path') else None
    selected_region = config.selected_region if hasattr(config, 'selected_region') else None
    tiling(dummy_input_path, config.input_ext, data_path, ROI_path, 
           config.inference_window_um, 
           config.input_pixel_per_um, 
           selected_region, 
           config.ref_channel)

    # Step 4: QC
    normalization(dummy_input_path, config.input_ext, data_path, config.inference_window_um, config.input_pixel_per_um, qc_path)

    # Step 5: Copy common_channels.txt to processed_data/data
    import shutil
    common_channels_path = os.path.join(config.data_root, 'raw_data', 'common_channels.txt')
    shutil.copy(common_channels_path, os.path.join(data_path, 'common_channels.txt'))

def normalization(input_path, input_ext, data_path, inference_window_um, input_pixel_per_um, qc_path):
    from canvas.preprocess import qc
    file_names = utils.get_file_name_list(input_path, input_ext)
    qc_save_path = f'{qc_path}/global'
    os.makedirs(qc_save_path, exist_ok=True)
    training_window_um = inference_window_um * 2
    training_window_pixel = training_window_um * input_pixel_per_um
    qc.calculate_normalization_stats(data_path, file_names, qc_save_path, tile_size = training_window_pixel)

def tiling(input_path, input_ext, data_path, ROI_path, inference_window_um, input_pixel_per_um, selected_region, ref_channel):
    from canvas.preprocess.tile import gen_tiles
    file_names = utils.get_file_name_list(input_path, input_ext)
    training_window_um = inference_window_um * 2
    training_window_pixel = training_window_um * input_pixel_per_um
    inference_window_pixel = inference_window_um * input_pixel_per_um
    for file_name in tqdm(file_names):
        gen_tiles(data_path, file_name, ref_channel, ROI_path, training_window_pixel, selected_region)
        gen_tiles(data_path, file_name, ref_channel, ROI_path, inference_window_pixel, selected_region)

def visualize_samples(data_type, input_path, input_ext, output_path, config_root, selected_channel_color_file, channel_strength_file, data_path):
    selected_channel_color_file = os.path.join(config_root, selected_channel_color_file)
    channel_strength_file = os.path.join(config_root, channel_strength_file)
    utils.visualize_color_yaml_file(selected_channel_color_file, output_path) # Confirm color
    color_dict = utils.load_channel_yaml_file(selected_channel_color_file)
    strength_dict = utils.load_channel_yaml_file(channel_strength_file)
    file_names = utils.get_file_name_list(input_path, input_ext)
    for file_name in tqdm(file_names):
        multi_img.visualize_sample(data_path, file_name, color_dict, strength_dict, data_type)

def zarr_conversion(data_root, image_path, input_ext, data_path, qc_path, dummy_input_path):
    from canvas.preprocess import io 
    from canvas.preprocess import qc
    # Start preprocessing
    file_names = utils.get_file_name_list(image_path, input_ext)
    print("files names: ", file_names)
    print('Converting tiff to zarr')
    for file_name in tqdm(file_names):
        print(file_name)
        if input_ext in ['qptiff', 'tiff', 'tif']:
            io.tiff_to_zarr(image_path, data_path, dummy_input_path, file_name, input_ext = input_ext)
        elif input_ext == 'mcd':
            io.mcd_to_zarr(image_path, data_path, dummy_input_path, file_name)
        else:
            raise ValueError(f'Unsupported input extension: {input_ext}')

    # Plot global QC histogram
    print('Generating global QC histogram')
    qc_file_names = utils.get_file_name_list(dummy_input_path, f'dummy_{input_ext}')
    qc.global_hist(data_path, qc_file_names, qc_path)