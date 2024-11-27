import argparse
from utils import helper
from types import SimpleNamespace
import pdb
import sys
import os

# import python files
from canvas.preprocess.preprocess import run_preprocess

def get_args_parser():
    parser = argparse.ArgumentParser('CANVAS Preprocessing Pipeline', add_help=False)
    parser.add_argument('--config_root', type=str, help='Root directory of the config files')
    parser.add_argument('--data_root', type=str, help='Root directory of all the data')
    parser.add_argument('--data_type', type=str, help='Type of data being processed')
    parser.add_argument('--input_ext', type=str, help='Extension of the input files')
    parser.add_argument('--input_pixel_per_um', type=int, help='Resolution of the input images in pixels per micrometer')
    parser.add_argument('--inference_window_um', type=int, help='Size of the inference window in micrometers')
    parser.add_argument('--ref_channel', type=int, help='channel used for background removal. Ideally DAPI')
    parser.add_argument('--ROI_path', type=str, help='relative to the Region of Interest (ROI) file with annotations')
    parser.add_argument('--selected_region', type=str, help='names of region in ROI_path for analysis')
    parser.add_argument('--raw_image_path', type=str, default='image_files', help='relative path to the raw image files')
    parser.add_argument('--input_path', type=str, default='raw_data', help='relative path to the input data')
    parser.add_argument('--selected_channel_color_file', type=str, help='Full path to the file containing selected channel colors')
    parser.add_argument('--channel_strength_file', type=str, help='Full path to the file containing channel strength information')
    parser.add_argument('--tiles_dir', type=str, default='tiles', help='Directory name containing image tiles')
    return parser

def update_config_from_args(config, args):
    # Update the in-memory config object with command-line arguments
    for key, value in vars(args).items():
        if value is not None:
            setattr(config, key, value)

def main():
    #Parse command-line arguments
    parser = get_args_parser()
    args = parser.parse_args()

    #load the default configuration
    config_yaml= os.path.join(args.config_root, 'config.yaml')
    run_config = helper.load_yaml_file(config_yaml)
    config = SimpleNamespace(**run_config)
    update_config_from_args(config, args)

    run_preprocess(config)

if __name__ == "__main__":
    main()