# CANVAS

# Instructions for running the CANVAS analysis framework for multiplexed images.


## Overview:
CANVAS pipeline requires 4 steps:
1. Preprocess Images: this step converts raw images IMC/tiff or other images into zarr format and specify the channels of interest.
2. Train CANVAS: this step trains the CANVAS model with the zarr data and save the trained model.
3. Extract features with trained model: this step extracts features from the trained model and save the features.
4. Run downstream analysis: this step runs the downstream analysis with the extracted features.

## Step 1: Preprocess Images
Follow instruction located at canvas/preprocess

Provide the root of the config files path

Organize the data with the following directory structure, each image file (<image>.ext) should have a corresponding channel text file (<image>.txt).

**You need a common channel text file to limit the channels for only those used in the analysis. This `common_channels.txt` file should be placed under the '<data_root>/raw_data' directory.**

```
dataset_root
    - config_root
        - config.yaml # Config file for the dataset
        - preprocess
            - selected_channels_w_color.yaml # List of channels and colors to be used for visualization
            - channels_vis_strength.yaml # Specify the color strength for each channel
    data_root
        - raw_data
            - common_channels.txt # List of all channels in the dataset
            - image_files
                - image1.ext
                - image1.txt # Channel file for each image file, each line is a channel name
                - image2.ext
                ...
        - processed_data
        - model_ckpt
        - analysis
```
Example of structured data set is on zenodo: https://zenodo.org/record/

Run preprocessing function: `python run_preprocess.py --config_root <config_root_path> --data_root <data_root_path>`

Example:
```
python run_preprocess.py --config_root /home/epoch/Documents/Jimin/CANVAS_v2/config_files --data_root /home/epoch/Documents/Jimin/CANVAS_v2_data
```

Overall directory structure should be like the following:
```
data_root
    - raw_data
        - common_channels.txt # List of all channels in the dataset
        - image_files
            - image1.ext
            - image1.txt # Channel file for each image file, each line is a channel name
            - image2.ext
            ...
        - dummy_input
            - image1_acquisition_0.dummy_ext # The dummy file is used as reference for the zarr samples
            - image2_acquisition_0.dummy_ext
            ...
    - processed_data
        - data
        - qc
    - model_ckpt
        - ckpts
        - log_dir
```

Things to check:
1. Sample image visualization: `data_root/processed_data/data/<image_name>/visualization/sample.png`.
2. Check the normalized image intensity distribution and check if it is in a reasonable range (-5 to 30 are normal).


## Step 2: Train CANVAS
You need a GPU for step 2 training and 3 inference.

Note: if this doesn't work the first time, try running it again. It is generating common channels.

Run training script: `python run_training.py --config_root <config_root_path> --data_root <data_root_path> --epoch <epochs>`

## Step 3: Inference and feature extraction using trained model
Adjust hyperparameters and data directories in canvas/inference/infer.py and run infer.py.

Run `python run_inference.py --config_root <config_root_path> --data_root <data_root_path> --ckpt_num <ckpt_num>` to extract features using a specific checkpoint.

This script will also run UMAP and KMeans clustering to generate initial clusters.

## Step 4: Run downstream analysis
The analysis python script is located at canvas/analysis/main.py. Functions can be slected to run only desired analysis.
