import argparse
from utils import helper
from types import SimpleNamespace
import pdb
import sys
import os

# import python files
from canvas.inference.infer_umap_kmeans import run_infer_umap_kmeans

def get_args_parser():
    parser = argparse.ArgumentParser('CANVAS Analysis', add_help=False)
    parser.add_argument('--config_root', type=str, help='Root directory of the config files')
    parser.add_argument('--data_root', type=str, help='Root directory of the project')
    parser.add_argument('--processed_data_dir', type=str, default='processed_data', help='Directory with processed data')
    parser.add_argument('--data_type', type=str, help='Type of data being processed')
    parser.add_argument('--input_path', type=str, help='Directory with raw data')
    parser.add_argument('--input_ext', type=str, help='Extension of input data')
    parser.add_argument('--input_pixel_per_um', type=str, help='Pixel per micrometer ratio of the input data')
    parser.add_argument('--inference_window_um', type=str, help='Size of the inference window in micrometers')
    parser.add_argument('--n_clusters', type=int, default=10, help='Number of clusters for k-means')
    parser.add_argument('--model_dir', type=str, default='model_ckpt/ckpts', help='relative path of the checkpoint file')
    parser.add_argument('--ckpt_num', type=str, help='relative path of the checkpoint file')
    parser.add_argument('--analysis_dir', type=str, default='analysis', help='Directory to save analysis results')
    parser.add_argument('--post_normalization_scaling_strategy', type=str, default=None, help='Strategy for post-normalization scaling')
    parser.add_argument('--cap_cutoff', type=float, default=None, help='Cap cutoff value for normalization')
    parser.add_argument('--perc_thres', type=float, default=None, help='Percentage threshold for filtering')
    parser.add_argument('--tiles_dir', type=str, default='tiles', help='Directory to save tiles')
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

    run_infer_umap_kmeans(config)

#run main() when this analysis.py is run 
if __name__ == "__main__":
    main()