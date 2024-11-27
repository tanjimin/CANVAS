import pdb
import argparse
from utils import helper
from types import SimpleNamespace
import sys
import os

from canvas.analysis.analysis_modules.main_mapping_to_imgs import run_mapping_to_imgs
from canvas.analysis.analysis_modules.main_patient_stratification import run_patient_stratification
from canvas.analysis.analysis_modules.main_signature_characterization import run_signature_characterization
from canvas.analysis.analysis_modules.main_graph_based_analysis import run_graph_based_analysis

def get_args_parser():
    parser = argparse.ArgumentParser('CODEX Analysis', add_help=False)
    parser.add_argument('--config_root', type=str, help='Root directory of the config files')
    parser.add_argument('--data_root', type=str, help='Root directory of the project')
    parser.add_argument('--processed_data_dir', type=str, default='processed_data', help='Directory with processed data')
    parser.add_argument('--analysis_dir', type=str, default='analysis', help='Directory where analysis results will be stored')
    parser.add_argument('--n_clusters', type=int, default=10, help='Number of clusters for clustering algorithms')
    parser.add_argument('--Clinical_Data_path', type=str, help='relative to the clinical data file')
    parser.add_argument('--WSI_subset_regions_file', type=str, help='File containing subset regions of Whole Slide Images (WSI)')
    parser.add_argument('--local_region', type=int, default=False, help='Flag to indicate if local region analysis is to be performed... for WSIs')
    parser.add_argument('--image_sub_w', type=int, default=None, help='Width of the image subset if local region analysis is to be performed')
    parser.add_argument('--image_sub_h', type=int, default=None, help='Height of the image subset if local region analysis is to be performed')
    return parser 

def update_config_from_args(config, args):
    # Update the in-memory config object with command-line arguments
    for key, value in vars(args).items():
        if value is not None:
            setattr(config, key, value)

def main():
    parser = get_args_parser()
    args = parser.parse_args()
    run_config = helper.load_yaml_file(f'{args.config_root}/config.yaml')
    config = SimpleNamespace(**run_config)
    update_config_from_args(config, args)

    # Number of channels is the number of lines in the common_channels.txt file
    with open(os.path.join(config.data_root, config.processed_data_dir, 'data', 'common_channels.txt'), 'r') as file:
        setattr(config, 'num_channels', sum(1 for _ in file))
    setattr(config, 'tile_size', config.input_pixel_per_um * config.inference_window_um)
    if not hasattr(config, 'WSI_subset_regions_file'):
        setattr(config, 'WSI_subset_regions_file', None)
    if not hasattr(config, 'image_sub_w'):
        setattr(config, 'image_sub_w', None)
    if not hasattr(config, 'image_sub_h'):
        setattr(config, 'image_sub_h', None)
    if not hasattr(config, 'Clinical_Data_path'):
        setattr(config, 'Clinical_Data_path', None)

    #Analysis Scripts
    run_mapping_to_imgs(config)
    #run_graph_based_analysis(config)
    #run_signature_characterization(config)

#run main() when this analysis.py is run 
if __name__ == "__main__":
    main()