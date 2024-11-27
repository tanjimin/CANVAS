import sys
import os
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt

def main():
    '''
    sample_name = sys.argv[1]
    n_clusters = int(sys.argv[2])
    tile_size = int(sys.argv[3])
    intensity_image_path = sys.argv[4]
    color_image_path = sys.argv[5]
    all_samlpe_names_path = sys.argv[6]
    all_tile_positions_path = sys.argv[7]
    all_cluster_labels_path = sys.argv[8]
    cluster_color_path = sys.argv[9]
    output_path = sys.argv[10]
    '''

    
    sample_name = 'AMP_1150'
    n_clusters = 20
    tile_size = 128
    zarr_path = f'/gpfs/scratch/jt3545/projects/CODEX/analysis/kidney/data/{sample_name}/data.zarr'
    #intensity_image_path = sys.argv[4]
    #color_image_path = sys.argv[5]
    all_sample_names_path = '/gpfs/scratch/jt3545/projects/CODEX/analysis/kidney/analysis/tile_embedding/sample_name.npy'
    all_tile_positions_path = '/gpfs/scratch/jt3545/projects/CODEX/analysis/kidney/analysis/tile_embedding/tile_location.npy'
    all_cluster_labels_path = '/gpfs/scratch/jt3545/projects/CODEX/analysis/kidney/analysis/clustering/labels.npy'
    cluster_color_path = '/gpfs/scratch/jt3545/projects/CODEX/analysis/kidney/analysis/clustering/plot_color_rgb.npy'
    output_path = f'sample_{sample_name}_cluster_map.png'

    # Load data
    all_sample_names = np.load(all_sample_names_path)
    all_tile_positions = np.load(all_tile_positions_path, allow_pickle = True)
    all_cluster_labels = np.load(all_cluster_labels_path)

    # Get sample index
    sample_index = np.where(all_sample_names == sample_name)[0]

    # Get tile positions
    tile_positions = all_tile_positions[sample_index]
    tile_positions = np.array([pos.numpy() for pos in tile_positions])

    # Get cluster labels
    cluster_labels = all_cluster_labels[sample_index]

    # Get cluster colors
    cluster_colors = np.load(cluster_color_path)
    # Increase the sum of RGB to at least 1
    for color_i in range(cluster_colors.shape[0]):
        while np.sum(cluster_colors[color_i, :]) < 1:
            cluster_colors[color_i, :] += 0.2

    '''
    # Get color image
    color_image = io.imread(color_image_path) / 255
    plot_cluster_map(color_image, tile_positions, cluster_labels, n_clusters, tile_size, output_path.replace('intensity', 'color'), cluster_colors)
    '''

    import zarr
    img_zarr = zarr.open(zarr_path, mode = 'r')

    cache = img_zarr[:].mean(axis = 0)
    cache = np.clip(cache, 0, np.percentile(cache, 98))
    cache /= cache.max()
    intensity_image = np.clip(cache, 0, 1) * 0.7

    # Get intensity image
    #intensity_image = io.imread(intensity_image_path) / 255
    intensity_image = np.expand_dims(intensity_image, axis = 2)
    intensity_image = np.repeat(intensity_image, 3, axis = 2)
    plot_cluster_map(intensity_image, tile_positions, cluster_labels, n_clusters, tile_size, output_path, cluster_colors, bin_size = 8)
    
    # Gen example map, for figures only
    #save_cluster_map_and_data_for_figure(sample_name, color_image, tile_positions, cluster_labels, n_clusters, tile_size, output_path, cluster_colors)

def plot_cluster_map(image, tile_positions, cluster_labels, n_clusters, tile_size, output_path, cluster_colors, bin_size = 10):
    colors = cluster_colors

    # Get cluster map
    cluster_map = np.zeros(image.shape)
    for i in range(tile_positions.shape[0]):
        x = tile_positions[i, 0]
        y = tile_positions[i, 1]
        cluster_map[x : x + tile_size, y : y + tile_size, :] = colors[cluster_labels[i]][:3]

    # Overlay intensity image
    cluster_map = cluster_map * 0.7 + image * 0.3
    cluster_map = np.clip(cluster_map, 0, 1)


    from skimage.measure import block_reduce

    # Downsample
    reduced_cluster_map = block_reduce(cluster_map, block_size = (bin_size, bin_size, 1), func = np.mean)
    
    io.imsave(output_path, reduced_cluster_map)


    '''
    # Save overlay for each cluster
    cols = 10
    rows = int(np.ceil(n_clusters / cols))
    fig, axs = plt.subplots(rows, cols, figsize = (cols * 5, rows * 5))
    for i in range(n_clusters):
        cluster_map = np.zeros(image.shape)
        if 'color' in output_path:
            cluster_map += 0.7
        for j in range(tile_positions.shape[0]):
            x = tile_positions[j, 0]
            y = tile_positions[j, 1]
            if cluster_labels[j] == i:
                if 'color' in output_path:
                    color_i = np.array([0, 0, 0])
                else:
                    color_i = colors[i][:3]
                cluster_map[x : x + tile_size, y : y + tile_size, :] = color_i
        cluster_map = cluster_map * 0.5 + image
        cluster_map = np.clip(cluster_map, 0, 1)
        axes = axs[i // cols, i % cols]
        axes.imshow(cluster_map, cmap = 'gray')
        axes.set_title('Cluster {}'.format(i))
        axes.axis('off')
    fig.savefig(output_path.replace('.png', '_by_cluster.png'), bbox_inches = 'tight', dpi = 300)
    plt.close()
    '''

def save_cluster_map_and_data_for_figure(sample_name, image, tile_positions, cluster_labels, n_clusters, tile_size, output_path, cluster_colors):
    # Get continuous color palette
    colors = cluster_colors / 255

    output_path = output_path.replace('cluster_on_intensity.png', 'tiles')
    os.makedirs(output_path, exist_ok = True)

    # Load numpy file
    npy_path = f'/gpfs/data/tsirigoslab/home/jt3545/CODEX/codex-imaging/src/preprocess/imaging_modality/imc_luad/datalake/samples/{sample_name}/data/core.npy'
    core = np.load(npy_path)
    
    from skimage import io, transform

    new_img = image.copy()

    # Save each cluster as a separate image
    for i in range(n_clusters):
        for j in range(tile_positions.shape[0]):
            x = tile_positions[j, 0]
            y = tile_positions[j, 1]
            if cluster_labels[j] == i:
                tile_img = image[x : x + tile_size, y : y + tile_size, :]
                tile_img = transform.resize(tile_img, (224, 224))
                io.imsave(output_path + f'/tile_x_{x}_y_{y}.png', tile_img)
                # Save npy file
                tile_npy = core[x : x + tile_size, y : y + tile_size, :]
                tile_npy_dapi = transform.resize(tile_npy[:,:,12], (224, 224))
                io.imsave(output_path + f'/tile_x_{x}_y_{y}_npy_dapi.png', tile_npy_dapi)
                # Save npy file
                np.save(output_path + f'/tile_x_{x}_y_{y}.npy', tile_npy)

                
                # Model inference
                if x == 128 and y == 576:
                #if x == 128 and y == 128:
                    model_weights = 2000
                    model_path = f'/gpfs/data/tsirigoslab/home/jt3545/CODEX/codex-imaging/src/model/pretrain/imc/output_dir/checkpoint-{model_weights}.pth'
                    save_path = output_path + f'/tile_x_{x}_y_{y}_inference'
                    os.makedirs(save_path, exist_ok = True)
                    gen_example(tile_npy, model_path, save_path)

                # Add border to image
                fill_value = [1, 1, 1]
                width = 3
                # Add top border
                new_img[x - width : x + width, y : y + tile_size , :] = fill_value
                new_img[x : x + width, y : y + tile_size , :] = fill_value
                new_img[x - width : x + width, y - width : y + tile_size + width, :] = fill_value
                # Add bottom border
                new_img[x + tile_size - width : x + tile_size + width, y : y + tile_size, :] = fill_value
                new_img[x + tile_size - width : x + tile_size, y : y + tile_size, :] = fill_value
                new_img[x + tile_size - width : x + tile_size + width, y - width : y + tile_size + width, :] = fill_value
                # Add left border
                new_img[x : x + tile_size, y - width : y + width, :] = fill_value
                new_img[x : x + tile_size, y : y + width, :] = fill_value
                new_img[x - width : x + tile_size + width, y - width : y + width, :] = fill_value
                # Add right border
                new_img[x : x + tile_size, y + tile_size - width : y + tile_size + width, :] = fill_value
                new_img[x : x + tile_size, y + tile_size - width : y + tile_size , :] = fill_value
                new_img[x - width : x + tile_size + width, y + tile_size - width : y + tile_size + width, :] = fill_value

    io.imsave(output_path + f'/img_with_boarders.png', new_img)

class Args():
    input_size = 224
    model = 'mae_vit_large_patch16'
    norm_pix_loss = False
    device = 'cuda'
    mask_ratio = 0.75
    batch_size = 32
    num_workers = 4
    pin_mem = False

def load_model(model_path):
    # Load model
    args = Args()
    sys.path.append('/gpfs/data/tsirigoslab/home/jt3545/CODEX/codex-imaging/src/model/pretrain/imc')
    import models_mae
    model = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)

    import torch
    device = torch.device(args.device)
    model.to(device)
    print('Model initialized')

    state_dict = torch.load(model_path)['model']
    model.load_state_dict(state_dict)
    print('State dicts loaded')
    model.eval()
    return model

def gen_example(data_feature, model_path, save_path):

    model = load_model(model_path)

    vis_pair(data_feature, model, save_path)

def vis_pair(data_feature, model, save_path):
    import torch
    os.makedirs(save_path, exist_ok = True)

    # Load mean and std of dataset
    stats_path = '/gpfs/data/tsirigoslab/home/jt3545/CODEX/codex-imaging/src/preprocess/imaging_modality/imc_luad/datalake/stats'
    dataset_mean = np.load(f'{stats_path}/mean.npy')
    dataset_std = np.load(f'{stats_path}/std.npy')
    # Move channel axis to first dimension and convert to 0-1 range
    data_feature = np.moveaxis(data_feature, -1, 0) / 255
    img_tensor = (data_feature - dataset_mean) / dataset_std
    img_tensor = torch.tensor(img_tensor).unsqueeze(0).float().cuda()
    # Rescale data to 224x224
    img_tensor = torch.nn.functional.interpolate(img_tensor, size=(224, 224), mode='bilinear', align_corners=False)

    # Prediction
    with torch.no_grad():
        torch.manual_seed(2077)
        loss, pred, mask = model(img_tensor,
                                 mask_ratio = 0.75)

    img_pred = model.unpatchify(pred).squeeze(0)

    mask = mask.detach().cpu().unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2 *18)
    mask = model.unpatchify(mask).squeeze(0)
    img_tensor = img_tensor.detach().cpu().squeeze(0)
    img_masked = img_tensor * (1 - mask)

    img_pred = img_pred.detach().cpu()

    # Paint mask to image
    img_filled = img_pred.clone()
    img_filled[mask == 0] = img_tensor[mask == 0]
    save_img((img_filled * dataset_std + dataset_mean) * 8, f'{save_path}/img_pred_filled.png')


    save_img((img_tensor * dataset_std + dataset_mean) * 8, f'{save_path}/img_source.png')
    save_img((img_masked * dataset_std + dataset_mean) * 8, f'{save_path}/img_source_masked.png')
    save_img((img_pred * dataset_std + dataset_mean) * 8, f'{save_path}/img_pred.png')

def save_img(img_tensor, file_name):
    import torch
    image = vis_codex(torch.moveaxis(img_tensor, 0, 2).detach().cpu())
    from skimage import io
    io.imsave(file_name, image, dpi = (300, 300))

def vis_codex(image):

    color_pallete = {'white' : (np.array([255, 255, 255]) * 0.3).astype(np.uint8),
                     'red' : np.array([255, 0, 0]),
                     'magenta' : np.array([255, 0, 255]),
                     'blue' : np.array([0, 0, 255]),
                     'green' : np.array([0, 255, 0]),
                     'yellow' : np.array([255, 255, 0]),
                     'cyan' : np.array([0, 255, 255]),
                     'orange' : np.array([255, 127, 0]),
                     }
    color_pallete['white1'] = color_pallete['white']
    color_pallete['white2'] = color_pallete['white']
    color_pallete['magenta1'] = color_pallete['magenta']
    color_pallete['magenta2'] = color_pallete['magenta']
    color_pallete['cyan1'] = color_pallete['cyan']
    color_pallete['cyan2'] = color_pallete['cyan']
    color_pallete['orange1'] = color_pallete['orange']
    color_pallete['orange2'] = color_pallete['orange']
    channel_names = ["CD117", "CD11c", "CD14", "CD163", "CD16", "CD20", "CD31", "CD3", "CD4", "CD68", "CD8a", "CD94", "DNA1", "FoxP3", "HLA-DR", "MPO", "Pancytokeratin", "TTF1"]
    name_map = dict(zip(channel_names, range(len(channel_names))))
    color_map = {'white' : 'DNA1',
                 'red' : 'CD3',
                 'magenta' : 'Pancytokeratin',
                 'magenta1' : 'TTF1',
                 'blue' : 'MPO',
                 'green' : 'CD20',
                 'yellow' : 'HLA-DR',
                 'cyan' : 'CD14',
                 'cyan1' : 'CD16',
                 'orange' : 'CD68',
                 'orange1' : 'CD163',
                 }

    channel_weights = dict(zip(channel_names, np.ones(len(channel_names))))
    channel_weights['DNA1'] = 2
    channel_weights['HLA-DR'] = 1
    channel_weights['Pancytokeratin'] = 1

    rgb_image = np.zeros_like(image)[:, :, :3]
    for c_name, marker in color_map.items():
        marker_map_1d = image[:, :, name_map[marker]]
        marker_map_1d *= channel_weights[marker]
        marker_map = np.moveaxis(np.tile(marker_map_1d, (3, 1, 1)), 0, 2)
        final_map = (marker_map / 255) * color_pallete[c_name].reshape(1, 1, 3)
        rgb_image = np.maximum(rgb_image, final_map)
    return (rgb_image * 255 / 12 * 2).clip(0, 255).astype(np.uint8)

if __name__ == '__main__':
    main()

