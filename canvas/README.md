
# Project Title

This project goes through preprocessing, training, inference, and analysis for running CANVAS on either whole slide tissue images or tissue microarrays 

#install the canvas package into the CANVAS environment: within the CANVAS folder you are working with, pip install -e . 

## Table of Contents
1. [Preprocessing](#Preprocessing)
2. [Training](#Training)
3. [Inference](#Inference)
4. [Analysis](#Analysis)

## Preprocessing 

run CANVAS_v2/canvas/run_preprocess.py

### Parameters: 
- data_type = whole slide image ('WSI') or tissue microarray ('TMA')
- input_path = This is the location of the qptiff files for all your images. All qptiffs should be in this folder
- input_ext = 'tif' or 'qptiff'
- input_pixel_per_um = this is the number of pixels per um
- inference_window_um = this is the tile size that you want your inference tiles to be
- output_path = this is the location of your out directory. ie: 'CANVAS_v2/canvas/out'
- common_channel_file = Must generate this text file which is a list of all the channels you plan to use in your analysis. Must be common across all slides. ie: 'CANVAS_v2/canvas/out/data/common_channels.txt'
- selected_channel_color_file = this is a yaml file with each marker and your desired color for visualization ie: DAPI: [0,0,255]. These colors are used when visualizing your samples. ie: 'CANVAS_v2/canvas/out/selected_channels_w_color.yaml'
- channel_strength_file = this is a yaml file with each marker and how strong the marker should show up. When the marker is typically pretty weak, you may want to put a strength greater than 1 but if you want to keep the brightness the same, set the marker to 1 ie: DAPI: 1. ie: 'CANVAS_v2/canvas/out/selected_channels_w_color.yaml'

zarr_conversion(): Generate zarr files for each TIFF image. (Statisitics are visualized for each zarr and for all channels)
- converts qptiffs to zarr
- Within the out/data directory: Make a file called "common_channels.txt" which is a file including all the channels you would like to include in your analysis. Each line is a separate channel. The channels should be in the same order as your 
- zarr files for each sample should be stored in each samples folder within 'out/data/'
- generates a global histogram npz file within out/qc/global/normalization/
    - generates global_hist_tiled_tilesize.npz files if tiled and if not tiled global_hist.npz --> this is basically a dictionary of all the tiles for each marker and the mean marker intensity for each tile  
     - !!! need to confirm whether this subset is the total number of pixels per sample or total number of tiles!!!

visualize_samples()
- Ensure that you generate a channel_strength_file.yaml and channel_strength_file.yaml prior to running this script
- Visualization is of certain channels that are specified in the yaml files
- generates a sample.png within the visualization folder within each sample within the data directory. This png is a downsampled visual of your entire slide or TMA for easy visualization. 

tiling()
- generates tiles for both training and inference... if training tiles are size 256, inference tiles will be size 128 
- all output will be per slide in out/data/{slide}/*.png
- gen_tiles() in tile.py **use tile_v2.py**
    - reads in the zarr file
    - gen_thumbnail() - generates and saves a thumbail which is a lower resolution greyscale image of your slide
        - the user can decide how you want background removal done: cache = block_reduce(slide[0], block_size=(scaling_factor, scaling_factor), func=np.mean) --> slide[0] refers to the first channel in your zarr file which in our case is DAPI. Therefore, background removal is done based on the DAPI stain. You can also decide to do it based on the mean of all channels.
    - gen_mask() - generates a mask of the slide which is based on the thumbnail. The user can set a threshold here per slide if needed or can set the same threshold for all slides which will determine the value that is set to background vs tissue. Anything that is background is set to 0 and then tissue is set to 1. You can set different thresholds for each slide. 
    - gen_tile_positions() - generates a csv file here: out/data/{slide}/positions_{tile_size}.csv. The 1st column is the height of the tile location and the 2nd column is the width of the tile location (top left position of each tile in the slide). It only generates tiles where the mask is set to 1, not 0, thereby avoiding the background. 

normalization()
    - generates normalization_stats.csv : marker, mean, std
        - these are the normalization stats on a subset of tiles. The number of tiles to use as a subset for each sample is specified in qc.py
    - plot_hist(): generates histograms across all channels on a subset of tiles. ???Confirm if this is all the pixels within each tile???
        - qc/global/normalization/channel_hist_tiled_{tile_size}_log.png
        - qc/global/normalization/channel_hist_tiled_{tile_size}_normalized_log.png
        - qc/global/normalization/channel_hist_tiled_{tile_size}_normalized.png
       
## Training 
CANVAS_v2/canvas/model/main_pretrain.py

### Parameters:
- batch_size = Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
- epochs = this is the number of epochs you want to train your model for. Default= 2000
- accum_iter = Accumulate gradient iterations (for increasing the effective batch size under memory constraints)

### Model parameters
- model = default - mae_vit_large_patch16 - Name of model to train
- input_size = default=224, images input size
- mask_ratio = default=0.75, Masking ratio (percentage of removed patches for masked autoencoding) 
- norm_pix_loss =  Use (per-patch) normalized pixels as targets for computing loss
    
### Optimizer parameters
- weight_decay =  default=0.05,weight decay (default: 0.05)
- lr = learning rate (absolute lr)
- blr = base learning rate: absolute_lr = base_lr * total_batch_size / 256
- min_lr = default=0 lower lr bound for cyclic schedulers that hit 0
- warmup_epochs = epochs to warmup LR

### Dataset parameters
- data_path = This is where the Zarr files are located 
- tile_size = Sample tile size
- output_dir = path where to save the checkpoints (.pth files), empty for no saving
- log_dir = path where to tensorboard log
- device = default='cuda', device to use for training / testing
- seed = default=0
- resume = this is the checkpoint file (.pth) you want to resume training from 
- start_epoch = default=0, starting epoch
- num_workers = default=10
- pin_mem = Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.
- no_pin_mem
    
### distributed training parameters
- world_size = default=1, type=int,help='number of distributed processes
- local_rank = default=-1, type=int
- dist_on_itp
- dist_url = url used to set up distributed training

### normalization adjustments parameters
- post_normalization_scaling_strategy = The data is Z score normalized. After this normalization, very intense pixels that have intensity values much higher than the mean will result is very high post normalized values. These extreme normalized values prevent the model from successfully training. If this is happening in your data, you must pick an option on how to deal with these values. One option is to cap all intensity values for all biomarkers to a certain value which brings down the extreme values to 1 value. Another option is to pick a threshold. This brings down the extreme values for each biomarker separately. Ex) 35% means to bring all the data in the top 35% of the data down to the value of the top 35%. Each biomarker may have a different top 35% so every biomarker gets adjusted differentltly.  
- cap_cutoff = This is the cutoff value when --post_normalization_scaling_strategy = cap'
- perc_thres = this is the threshold used for normalization when --post_normalization_scaling_strategy = threshold'


1. Run run_pretraining.sh
    - most necessary parameters: 
        - epoch 
        - batch_size
        - tile_size
        - output_dir 
        - log_dir 
        - data_path
2. Outputs: out/model_ckpt/checkpoint-X.pth and out/model_ckpt/log.txt
    - checkpoint-X.pth: use these checkpoints to visualize on tensorboard. It shows you what your loss is doing per iteration. 
    - log.txt: summary of training learning rate and loss per epoch. Do not blindly trust these values. It's better to visualize what the loss is doing after each iteration and understand the trend on tensorboard before assuming your model is training correctly.


## Inference

config file: run_infer_umap_kmeans.yaml
run CANVAS_v2/canvas/run_inference.py --> this script runs inference, generates the umap of the inference output, and then runs kmeans. The user can specify how many clusters to run for kmeans

## General Parameters 
- data_type =  WSI # 'TMA' or 'WSI'
- input_path =  /gpfs/data/proteomics/projects/mh6486/FenyoLab/Endometrial/dat/qptiff_dat #dir with qptiffs 
- input_ext =  qptiff
- input_pixel_per_um =  2 # .5um = 1 pixel, 1um = 2 pixels 
- inference_window_um =  128 #64
- n_clusters = 17
- common_channel_file = /gpfs/data/proteomics/projects/mh6486/FenyoLab/Endometrial/CANVAS_v2/canvas/out_256/data/common_channels.txt 
- output_path =  /gpfs/data/proteomics/projects/mh6486/FenyoLab/Endometrial/CANVAS_v2/canvas/out_256 #this is where the analysis dir will be 
- selected_channel_color_file =  /gpfs/data/proteomics/projects/mh6486/FenyoLab/Endometrial/CANVAS_v2/canvas/out_256/selected_channels_w_color.yaml 
- channel_strength_file = /gpfs/data/proteomics/projects/mh6486/FenyoLab/Endometrial/CANVAS_v2/canvas/out_256/channels_vis_strength.yaml
- model_dir = model_ckpt_cap35perc_endo/checkpoint-1550.pth #this is the latest training .pth output file 
- base_path = /gpfs/data/proteomics/projects/mh6486/FenyoLab/Endometrial/CANVAS_v2/canvas/out_256
- analysis_base_path = analysis/epoch_1550_cap35perc_endo
- data_base_path = /gpfs/data/proteomics/data/Endometrial_mIF/out_256 #this is the base to where the actual slide data is 

## Directory Parameters
- clusters_npy = 17clusters.npy

## SlidesDataset Arguments
- post_normalization_scaling_strategy = threshold #this is the type of scaling (cap vs threshold)
- cap_cutoff = None #if cap, what cutoff to use
- perc_thres = 35 
- tiles_dir = endometrium_tiles 

run_infer_umap_kmeans(config_yaml)
 - infer: inference/infer.py
    - load_dataset(): This uses the dataloader that was also used for model training to load the dataset. 
    - load_model(): This loads the model ??????
    - get_tile_embedding(): outputs are saved here: out/{analysis_base_path}/tile_embedding/*.npy
        - image_mean.n6py : ????
        - embedding_mean.npy : ????
        - tile_location.npy : these are the coordinates of the top left corner of each tile 
        - sample_name.npy : these are the sample names of all the tiles
        **These are all the same dimensions because they are using all the same tiles**
 - gen_umap: analysis/main.py
    - this generates a umap using the embedding generated during training 
 - kmeans: analysis/clustering/kmeans.py
    - must set the number of clusters you want to separate your tiles into 
    - kmeans uses the information in the latent space to determine what tiles should be in what clusters

## Analysis 

run CANVAS_v2/canvas/run_analysis.py
config: config_codeximaging.yaml

### Config Parameters
basic info about images 
- mask_id: thumbnail_scale_16
- image_type: WSI
- tile_size: 256
- num_channels: 36
- n_clusters: This is the number of clusters that you used when running kmeans.py
- model_weights: This is the checkpoint (.pth) file that will be used when doing these analyses
- augmentation: default
- local_region: True vs False. This is whether you want to look at a subset of the slide rather than the entire slide. This is most necessary to be True with WSIs. 

### Directory Paths
- data_base_path: This is the path to the actual slide data  
- base_path: This is the path to where you want to output your analysis ex) CANVAS_v2/canvas/out 
- analysis_base_path: this is the folder with the kmeans, tile_embedding, and umap directories. analysis/{analysis_name} 
- canvas_model_data_path: CANVAS_v2/canvas/model/data

### sys.path.append paths 
- module_path: /gpfs/data/proteomics/projects/mh6486/FenyoLab/Endometrial/CANVAS_v2/canvas/analysis/codex-imaging
- plt_fig_path: /gpfs/data/proteomics/projects/mh6486/FenyoLab/Endometrial/CANVAS_v2/canvas/plt-figure

### directory names
clusters_npy: 17clusters.npy

### output from gen_sample_images.py
stats_dir: The name of the folder within out/{stats_dir_name}
model_ckpt_file: path to model checkpoint directory + .pth file: model_ckpt/checkpoint-{X}.pth

### OUTPUT DIRECTORY NAMES - separated by the main.py file used 
#Signature characterization:
cluster_by_marker_dir: s5-1_cluster_by_marker
clinical_corr_dir: s6-1_clinical_corr
cluster_by_marker_celltype_clinical_dir: s5-2_cluster_by_marker_celltype_clinical
#Patient stratification:
cluster_vs_sample_dir: s4-2_cluster_vs_sample
#Mapping to images
sample_visualization_dir: s7_2_sample_visualization
#Graph based analysis:
sample_graph_visualiation_dir: s4-4_sample_graph_visualization

### assign colors for codex_imaging/gen_cluster_sample_map.py
color_map: 
  magenta : Pan-Cytokeratin
  blue : MPO
  yellow : HLA-ABC
  orange : CD68
  orange1 : CD163
                 
### Zarr subset regions... in numpy, to access a x,y coordinate in omero, use: img[y,x]
image_sub_w: 2000 #this is the width of the local region subset
image_sub_h: 2000 #this is the height of the local region subset

#signature characterization 
Clinical_Data_path: /gpfs/data/proteomics/data/Endometrial_mIF/canvas_files/dat/EC_Clinical_Data.csv
- The path to the ClinicalData.csv file which must be created prior to running analyses. 

WSI_subset_regions_file: /gpfs/data/proteomics/data/Endometrial_mIF/canvas_files/dat/WSI_subset_regions.csv
- This is a csv file where slideID, Height, and Width. The height and width refer to the top left corner of the subset of the WSI that we are using for the graph analysis. 

#convert column values to binary: [value in Clinical_Data_path : value you want to show up in legend on heat map]
- In the clinical data, if 0's and 1's are used to denote specific amounts, replace them with their actual values so that it's easier to understand on a figure
clinical_var_labels:
  Age:
    0 : <70 
    1 : ≥70 
  BMI:
    0 : <30 
    1 : ≥30 
  IO response:
    0 : 'No'
    1 : 'Yes'
  MLH1 promoter hypermethylation?:
    1 : 'Yes'

### codex_imaging analysis output explanations
run_patient_stratification(config_yaml): 
- s4-2_cluster_vs_sample : Heatmaps to interpret cluster enrichment 
  - cluster_labels.csv
  - cluster_with_sample_enrichment_col_normalized_heatmap.pdf
  - cluster_with_sample_enrichment_heatmap_cluster.pdf
  - cluster_with_sample_enrichment_heatmap.pdf
  - cluster_with_sample_enrichment_marginalized_heatmap.pdf
  - cluster_with_sample_enrichment.pdf
  - cluster_with_sample.csv
  - clustermap.pdf

run_signature_characterization(config_yaml)
- s5-1_cluster_by_marker 
  - heatmap_overall_mean_intensity.npy
  - heatmap.npy
  - heatmap.pdf: Marker Enrichment per cluster
- s6-1_clinical_corr
  - clinical_corr.pdf : scatter plot visualizing clinical variables per cluster
  - counts_table.csv
- s5-2_cluster_by_marker_celltype_clinical
  - plot.pdf : marker cluster enrichment + counts of tiles per cluster + pvalues per clinical variable per cluster

run_mapping_to_imgs(config_yaml)
- s7_2_sample_visualization
  - per slide (specific section of slide): 
     - color.png : visualization of region with specified marker expression 
     - intensity.png : Black and white version of color.png
     - per cluster number: the region of interest or the whole slide sectioned into it's tiles 
        - cluster_on_color_by_cluster.png: color.png is split up by cluster and for each cluster, we can visualize what tiles within the region are assigned to what cluster
        - cluster_on_color.png: color.png but each tile is colored by its assigned cluster
        - cluster_on_intensity_by_cluster.png: intensity.png (B&W) with the tiles colored based on their assigned cluster. Separated by cluster. 
        - cluster_on_intensity.png: 1 intensity.png with each tile colored by its cluster assignment. 

run_graph_based_analysis(config_yaml)
- s4-4_sample_graph_visualization: separated by sample 
  - core_graph.pdf: this is the full slide. Each node (black dot) is a tile and is colored based on its assigned cluster. Each node is connected to its neighboring node. 
  - core_graph.pkl
  - stats directory
    - inbetweeness_array.csv
    - inbetweeness_hist.pdf
    - inbetweeness.pdf
  - Directory with region of interest h{image_sub_h}_w{image_sub_w}... image_sub_h and image_sub_w are specified by user in the config file. With larger images, the graphs in this directory are too computationally intensive. Therefore, it's better to do so on a smaller region of the whole slide image or larger slide! 
    - core_graph_free_snap_reduced.pdf : reduced version of core_graph.pdf of just the region specifed 
    - core_graph_reduced.pdf
    - core_graph_snap.pkl
    - core_graph.pdf
    - core_graph.pkl



