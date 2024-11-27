import argparse
from utils import helper
from types import SimpleNamespace
import os
import subprocess

def get_args_parser():
    parser = argparse.ArgumentParser('CANVAS Pretraining Pipeline', add_help=False)
    parser.add_argument('--ngpus', type=int, default=1, help='Number of GPUs to use')
    parser.add_argument('--config_root', type=str, help='Root directory of the config files')
    parser.add_argument('--data_root', type=str, help='Root directory of all the data')
    parser.add_argument('--input_pixel_per_um', type=int, help='Resolution of the input images in pixels per micrometer')
    parser.add_argument('--inference_window_um', type=int, help='Size of the inference window in micrometers')
    parser.add_argument('--epoch', type=int, default=2000, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--output_path', type=str, default='model_ckpt', help='Directory to save the model checkpoints')
    return parser

def update_config_from_args(config, args):
    # Update the in-memory config object with command-line arguments
    for key, value in vars(args).items():
        if value is not None:
            setattr(config, key, value)

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

    # Get current directory
    current_path = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(config.data_root, config.output_path)
    ckpt_path = os.path.join(output_path, 'ckpts')
    log_path = os.path.join(output_path, 'log_dir')
    data_path = os.path.join(config.data_root, 'processed_data', 'data')
    tile_size = config.inference_window_um * config.input_pixel_per_um * 2 # 2x to compensate for augmentations
    # Run training script
    subprocess.run(f'torchrun --standalone --nnodes=1 --nproc_per_node={config.ngpus} {current_path}/model/main_pretrain.py --epoch {config.epoch} --batch_size {config.batch_size} --tile_size {tile_size} --output_dir {ckpt_path} --log_dir {log_path} --data_path {data_path}', shell=True)

if __name__ == '__main__':
    main()