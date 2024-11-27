import argparse
from canvas.utils import helper
from types import SimpleNamespace
import pdb
import sys
import os

#signature characterization 
from codex_imaging.main_patient_stratification import run_patient_stratification
from codex_imaging.main_signature_characterization import run_signature_characterization
from codex_imaging.main_mapping_to_imgs import run_mapping_to_imgs
from codex_imaging.main_graph_based_analysis import run_graph_based_analysis

def get_args_parser():
    parser = argparse.ArgumentParser('CODEX Analysis', add_help=False)
    parser.add_argument('--n_clusters', type=int, help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus)')
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
    config_yaml= "/gpfs/data/proteomics/data/Endometrial_mIF/canvas_files/yaml_files/config_codeximaging.yaml"
    run_config = helper.load_yaml_file(config_yaml)
    config = SimpleNamespace(**run_config)

    update_config_from_args(config, args)

    # Add the directory containing your analysis scripts to the Python path
    # this is where the following analysis scripts are located OR change all the module paths in the analysis scripts
    sys.path.append(os.path.abspath("/gpfs/data/proteomics/projects/mh6486/FenyoLab/Endometrial/CANVAS_v2/canvas/analysis/codex_imaging"))
    
    #Analysis Scripts
    run_patient_stratification(config_yaml)
    run_signature_characterization(config_yaml)
    run_mapping_to_imgs(config_yaml)
    run_graph_based_analysis(config_yaml)

#run main() when this analysis.py is run 
if __name__ == "__main__":
    main()