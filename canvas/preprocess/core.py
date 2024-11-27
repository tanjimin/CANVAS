import os
from tqdm import tqdm
import canvas.utils.helper as utils

import canvas.visualization.multiplex_image as multi_img

def main():
    # Define parameters
    data_type = 'TMA' # 'TMA' or 'WSI'
    input_path = '/gpfs/scratch/jt3545/projects/CODEX/data/lung_imc'
    input_ext = 'tif'
    input_pixel_per_um = 1
    inference_window_um = 64
    common_channel_file = '/gpfs/scratch/jt3545/projects/CODEX/data/lung_imc_metadata/all_channels.txt'
    output_path = '/gpfs/scratch/jt3545/projects/CODEX/data_processed/lung_imc'
    selected_channel_color_file = '/gpfs/scratch/jt3545/projects/CODEX/data/lung_imc_metadata/selected_channels_w_color.yaml'
    channel_strength_file = '/gpfs/scratch/jt3545/projects/CODEX/data/lung_imc_metadata/channels_vis_strength.yaml'
    #common_channels = ["CD117", "CD11c", "CD14", "CD163", "CD16", "CD20", "CD31", "CD3", "CD4", "CD68", "CD8a", "CD94", "DNA1", "FoxP3", "HLA-DR", "MPO", "Pancytokeratin", "TTF1"]

    data_path = f'{output_path}/data'
    qc_path = f'{output_path}/qc'

    # Step 1: Convert tiff to zarr and QC
    #zarr_conversion(common_channel_file, input_path, input_ext, output_path, data_path, qc_path)

    # Step 2: Choose channels with colormap and visualize individual images
    #visualize_samples(data_type, input_path, input_ext, output_path, selected_channel_color_file, channel_strength_file, data_path)

    # Step 3: Generate tiles
    #tiling(input_path, input_ext, data_path, output_path, inference_window_um, input_pixel_per_um)

    # Step 4: QC
    #normalization(input_path, input_ext, data_path, output_path, inference_window_um, input_pixel_per_um, qc_path)

def normalization(input_path, input_ext, data_path, output_path, inference_window_um, input_pixel_per_um, qc_path):
    from canvas.preprocess import qc
    file_names = utils.get_file_name_list(input_path, input_ext)
    qc_save_path = f'{qc_path}/global'
    os.makedirs(qc_save_path, exist_ok=True)
    training_window_um = inference_window_um * 2
    training_window_pixel = training_window_um * input_pixel_per_um
    qc.calculate_normalization_stats(data_path, file_names, qc_save_path, tile_size = training_window_pixel)

def tiling(input_path, input_ext, data_path, output_path, inference_window_um, input_pixel_per_um):
    from canvas.preprocess.tile import gen_tiles
    file_names = utils.get_file_name_list(input_path, input_ext)
    training_window_um = inference_window_um * 2
    training_window_pixel = training_window_um * input_pixel_per_um
    inference_window_pixel = inference_window_um * input_pixel_per_um
    for file_name in tqdm(file_names):
        gen_tiles(data_path, file_name, training_window_pixel)
        gen_tiles(data_path, file_name, inference_window_pixel)

def visualize_samples(data_type, input_path, input_ext, output_path, selected_channel_color_file, channel_strength_file, data_path):
    utils.visualize_color_yaml_file(selected_channel_color_file, output_path) # Confirm color
    color_dict = utils.load_channel_yaml_file(selected_channel_color_file)
    strength_dict = utils.load_channel_yaml_file(channel_strength_file)
    file_names = utils.get_file_name_list(input_path, input_ext)
    for file_name in tqdm(file_names):
        #file_name = 'LUAD_D360'
        multi_img.visualize_sample(data_path, file_name, color_dict, strength_dict, data_type)

def zarr_conversion(common_channel_file, input_path, input_ext, output_path, data_path, qc_path):
    from canvas.preprocess import io 
    from canvas.preprocess import qc
    # Start preprocessing
    common_channels = utils.read_channel_file(common_channel_file)
    file_names = utils.get_file_name_list(input_path, input_ext)
    #file_names = ['LUAD_D301', 'LUAD_D360', 'LUAD_D373']

    print('Converting tiff to zarr')
    for file_name in tqdm(file_names):
        io.tiff_to_zarr(input_path, data_path, file_name, input_ext, common_channels)
    # Plot global QC histogram
    print('Generating global QC histogram')
    qc.global_hist(data_path, file_names, qc_path)
        
if __name__ == '__main__':
    main()
