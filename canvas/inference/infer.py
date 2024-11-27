import os
import json

import torch
import numpy as np
import pdb

from tqdm import tqdm
import inference.inference_utils as iutils

def infer(data_root, processed_data_path, post_normalization_scaling_strategy, cap_cutoff, perc_thres, tiles_dir, model_path, save_path, input_pixel_per_um, inference_window_um):
    inference_window_pixel = inference_window_um * input_pixel_per_um
    data_path = os.path.join(data_root, processed_data_path, 'data')
    dataloader = load_dataset(data_path, post_normalization_scaling_strategy, cap_cutoff, perc_thres, tiles_dir, tile_size = inference_window_pixel)
    model = load_model(model_path, dataloader)
    get_tile_embedding(save_path, dataloader, model)
    print('Done')

def load_dataset(data_path, post_normalization_scaling_strategy, cap_cutoff, perc_thres, tiles_dir, tile_size, batch_size = 32, num_workers = 8):
    # Predefined parameters
    input_size = 224
    from torchvision import transforms
    transform_codex = transforms.Compose([
            transforms.ToTensor(), #converts values 0,255 -> 0,1
            transforms.Resize(input_size, interpolation = 2),
            ])
    from model.data.imc_dataset import CANVASDatasetWithLocation, SlidesDataset
    print("data path: ",data_path)
    dataset = SlidesDataset(data_path, tile_size = tile_size, post_normalization_scaling_strategy = post_normalization_scaling_strategy, cap_cutoff = cap_cutoff, perc_thres = perc_thres, tiles_dir = tiles_dir, transform = transform_codex, dataset_class = CANVASDatasetWithLocation)

    dataloader= torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
    )
    return dataloader

def load_model(model_path, dataloader, norm_pix_loss = False, 
               model_name = 'mae_vit_large_patch16'):
    num_channels = len(dataloader.dataset.common_channel_names)
    from model import models_mae
    model = models_mae.__dict__[model_name](norm_pix_loss=norm_pix_loss, 
                                            in_chans = num_channels)
    model_device = iutils.get_device()
    model.to(model_device)
    print('Model initialized')
    state_dict = torch.load(model_path)['model']
    model.load_state_dict(state_dict)
    print('State dicts loaded')
    model.eval()
    return model

def get_tile_embedding(save_path,
                       dataloader, model,
                       output_suffix = 'tile_embedding',
                       save_image = False, save_full_emb = False):
    device = iutils.get_device()
    output_path = f'{save_path}/{output_suffix}'
    os.makedirs(output_path, exist_ok = True)
    embedding_path = os.path.join(output_path, 'embedding.npy')
    print("embedding path:", embedding_path)
    if os.path.exists(embedding_path):
        print('Embedding already exist, skipping')
        return 
    # Prediction
    data_size = len(dataloader.dataset)
    num_channels = len(dataloader.dataset.common_channel_names)
    if save_image == 'True':
        print('Saving images')
        image_tensor = np.zeros((data_size, num_channels, 224, 224))
    if save_full_emb == 'True':
        print('Saving full embeddings')
        embedding_tensor = np.zeros((data_size, 196, 1024)).astype(np.float16)
    image_mean_tensor = np.zeros((data_size, num_channels))
    embedding_mean_tensor = np.zeros((data_size, 1024))
    sample_name_list = []
    tile_location_list = []
    batch_size = dataloader.batch_size

    with torch.no_grad():
        for batch_idx, sample in enumerate(tqdm(dataloader)):
            img_tensor, (labels, locations) = sample
            data_idx = batch_idx * batch_size
            temp_size = img_tensor.shape[0]
            embedding = proc_embedding(img_tensor, model, device)
            sample_name_list.extend(labels)
            tile_location_list.extend(locations)
            image_mean_tensor[data_idx:data_idx + temp_size] = img_tensor.mean(axis = (2, 3))
            embedding_mean_tensor[data_idx:data_idx + temp_size] = embedding.mean(axis = 1)
            if  save_image == 'True':
                image_tensor[data_idx:data_idx + temp_size] = img_tensor.numpy()
            if save_full_emb == 'True':
                embedding_tensor[data_idx:data_idx + temp_size] = embedding

    # ensure that tile_location.npy elements are an array!! (not tensors)
    # Check if the elements are tensors and convert accordingly
    if isinstance(tile_location_list[0], torch.Tensor):
        # Convert tensors to a 2D NumPy array
        tile_location_list = np.vstack([tensor.numpy() for tensor in tile_location_list])
    else:
        # If not tensors, just convert to a regular NumPy array
        tile_location_list = np.array(tile_location_list)

    print('output path: ', output_path)
    np.save(os.path.join(output_path, 'image_mean.npy'), image_mean_tensor)
    np.save(os.path.join(output_path, 'embedding_mean.npy'), embedding_mean_tensor)
    np.save(os.path.join(output_path, 'tile_location.npy'), np.array(tile_location_list))
    np.save(os.path.join(output_path, 'sample_name.npy'), np.array(sample_name_list))
    if save_image == 'True':
        np.save(os.path.join(output_path, 'image.npy'), image_tensor)
    if  save_full_emb == 'True':
        np.save(os.path.join(output_path, 'embedding.npy'), embedding_tensor)

def proc_embedding(img_tensor, model, device):
    imgs = img_tensor.to(device).float()
    mask_ratio = 0
    with torch.no_grad():
        latent, mask, ids_restore = model.forward_encoder(imgs, mask_ratio)
        latent_no_cls = latent[:, 1:, :]
        restored_latent = torch.gather(latent_no_cls, dim = 1, index = ids_restore.unsqueeze(-1).repeat(1, 1, latent.shape[2])).detach().cpu().numpy()
    return restored_latent

if __name__ == '__main__':
    main()
