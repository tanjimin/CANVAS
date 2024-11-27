import os
import zarr
import numpy as np
from skimage.measure import block_reduce
from skimage.transform import resize
from skimage.io import imsave
import pandas as pd
from skimage.draw import polygon
from PIL import Image
import cv2
import pdb

def gen_tiles(data_path: str, slideID: str, ref_channel: int, ROI_path:str, tile_size: int = 128, selected_region: str = None) -> np.ndarray:
   
    print("slideID: ", slideID)
    ''' Generate tiles for a given slide '''

    output_path = os.path.join(f'{data_path}/{slideID}/tiles')
    os.makedirs(output_path, exist_ok=True)
    
    # Read slide
    print('Reading slide...')
    slide = f'{data_path}/{slideID}/data.zarr'
    if isinstance(slide, str):
        if os.path.exists(slide):
            slide = zarr.load(slide)
        else:
            print(f'{slideID} zarr file DNE')
            return None
    
    # Generate and save thumbnail
    thumbnail_path = os.path.join(output_path, f"thumbnail_{tile_size // 4}.png")
    thumbnail = gen_thumbnail(slide, slideID, ref_channel, scaling_factor=tile_size // 4)
    save_img(output_path, 'thumbnail', tile_size // 4, thumbnail)
    
    # Generate and save mask
    mask_path = os.path.join(output_path, f"mask_{tile_size // 4}.png")
    if os.path.exists(mask_path):
        print("Mask already exists, loading it")
        mask_img = Image.open(mask_path)
        mask = np.array(mask_img)
    else:
        print("Mask doesn't exist, generating it and saving")
        mask = gen_mask(thumbnail, slideID)
        save_img(output_path, 'mask', tile_size // 4, mask)
    
    # Generate and save tile positions
    positions_file = os.path.join(output_path, f'positions_{tile_size}.csv')
    if not os.path.exists(positions_file):
        print("Positions and mask with artifacts removed don't exist, generating it and saving")
        positions, tile_img = gen_tile_positions(output_path, slide, mask, mask_path, slideID, selected_region, ROI_path, tile_size=tile_size)
        #check if positions and/or tile_img are None type (not generated)... if so, move to the next sample! 
        save_img(output_path, 'tile_img', tile_size, tile_img)
        #save_img(output_path, 'mask', tile_size // 4, mask_artifactsrm)
        with open(positions_file, 'w') as f:
            f.write(' ,h,w\n')
            for i, (h, w) in enumerate(positions):
                f.write(f'{i},{h},{w}\n')
        print(f'Generated {len(positions)} tiles for slide with shape {slide.shape}')

def save_img(output_path: str, task: str, tile_size: int, img: np.ndarray):
    ''' Save image to output path '''
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    imsave(os.path.join(output_path, f'{task}_{tile_size}.png'), img)

def gen_thumbnail(slide: zarr, slideID: str, ref_channel: int, scaling_factor: int) -> np.ndarray:
    ''' Generate thumbnail for a given slide '''
    assert slide.shape[0] < slide.shape[1] and slide.shape[0] < slide.shape[2]
    cache = block_reduce(slide[ref_channel], block_size=(scaling_factor, scaling_factor), func=np.mean)
    cache = np.clip(cache, 0, np.percentile(cache, 95))
    cache /= cache.max()
    return np.clip(cache, 0, 1).squeeze()

def gen_mask(thumbnail: np.ndarray, slideID, threshold: int = .4) -> np.ndarray:
    ''' Generate mask for a given thumbnail '''
    return np.where(thumbnail > threshold, 1, 0)

def gen_tile_positions(output_path: str, slide: zarr, mask: np.ndarray, mask_path, slideID: str, selected_region: str = None, ROI_path: str = None, tile_size: int = 256, threshold: float = 0.1) -> np.ndarray:
    
    # first smooth the mask!!! Removing unwanted holes 
    # Load the mask image in grayscale mode
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Use a small kernel for morphological operations to preserve real data
    kernel = np.ones((5, 5), np.uint8)  # Smaller kernel to limit modifications
    # Apply morphological closing to fill small black spots
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    #Now remove the very small white dots, since they are typically in the background and shouldn't be included
    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Save the smoothed mask
    save_img(output_path, 'mask_smoothed', tile_size // 4, mask)
    
    ''' Generate tiles for a given slide and mask '''
    _, slide_height, slide_width = slide.shape

    grid_height, grid_width = slide_height // tile_size, slide_width // tile_size
    
     # Original dimensions
    original_width = slide.shape[1]
    original_height = slide.shape[2]

    # Mask dimensions
    mask_width = mask.shape[0]
    mask_height = mask.shape[1]

    # Calculate scaling factors
    scale_x = mask_width / original_width
    scale_y = mask_height / original_height

    # List all files in the ROI directory
    all_files = os.listdir(ROI_path)

    # Filter files containing slideID in their name
    ROIfile_name = [f for f in all_files if slideID in f]

    if not ROIfile_name:
        print(f"No ROI file found containing the slideID: {slideID}")
        print("Generating positions on entire slide and not removing artifacts")
        
    else:
        print(ROIfile_name)
        ROIs_path = os.path.join(ROI_path, ROIfile_name[0])
        ROIdata = pd.read_csv(ROIs_path)
 
        if selected_region is not None: #this is the region we want to include in the analysis
            #first keep only endometrium!! 
            data_subset = ROIdata.loc[ROIdata.Text == selected_region] 
            data_subset_coords = data_subset['all_points']

            data_subset_coords_list = data_subset_coords.str.split(' ', expand=False)
        
            ROIs_list = []
            for ROI in data_subset_coords_list:
                print(ROI)
                # Convert the list of strings to a NumPy array of floats (if it's in the format 'x,y')
                data_subset_coords_array = np.array([list(map(float, coord.split(','))) for coord in ROI])
                data_subset_coords_array_rescaled = data_subset_coords_array * [scale_x, scale_y]
                ROIs_list.append(data_subset_coords_array_rescaled)

            for ROI in ROIs_list:
                clipped_indices_x = np.clip(ROI[:,0].astype(int), 0, mask.shape[1] - 1)
                clipped_indices_y = np.clip(ROI[:,1].astype(int), 0, mask.shape[0] - 1)
                # Create a boolean mask where the mask values are not 0
                not_zero_mask = mask[clipped_indices_y, clipped_indices_x] != 0
                # Update mask only where the original values were not 0
                mask[clipped_indices_y[not_zero_mask], clipped_indices_x[not_zero_mask]] = 1
                cc, rr = polygon(clipped_indices_x, clipped_indices_y)
                not_zero_mask_rr_cc = mask[rr, cc] != 0
                mask[rr[not_zero_mask_rr_cc], cc[not_zero_mask_rr_cc]] = 1

            mask[mask == 255] = 0
            img = (np.clip(mask, 0, 1) * 255).astype(np.uint8)
            save_img(output_path, 'mask_selected_region', tile_size // 4, img)
        else:
            print("no selected region... utilizing entire slide for analysis")

        #Now remove Artifacts!!!
        #create a new column indicating whether or not the row was an integer
        ROIdata['Text_numeric'] = pd.to_numeric(ROIdata['Text'], errors='coerce')
        #these are just the artifact ROIs!!
        data_subset = ROIdata.dropna(subset=['Text_numeric'])

        data_subset_coords = data_subset['all_points']
        data_subset_coords_list = data_subset_coords.str.split(' ', expand=False)
        
        ROIs_list = []
        for ROI in data_subset_coords_list:
            print(ROI)
            # Convert the list of strings to a NumPy array of floats (if it's in the format 'x,y')
            data_subset_coords_array = np.array([list(map(float, coord.split(','))) for coord in ROI])
            data_subset_coords_array_rescaled = data_subset_coords_array * [scale_x, scale_y]
            ROIs_list.append(data_subset_coords_array_rescaled)

        for ROI in ROIs_list:
            clipped_indices_x = np.clip(ROI[:,0].astype(int), 0, mask.shape[1] - 1)
            clipped_indices_y = np.clip(ROI[:,1].astype(int), 0, mask.shape[0] - 1)
            mask[clipped_indices_y, clipped_indices_x] = 0
            cc, rr = polygon(clipped_indices_x, clipped_indices_y)
            mask[rr, cc] = 0
        
        img = (np.clip(mask, 0, 1) * 255).astype(np.uint8)
        save_img(output_path, 'mask_artrm', tile_size // 4, img)

    mask_pixellevel = resize(mask, (grid_height, grid_width), order=0, anti_aliasing=False)
    tile_img = np.where(mask_pixellevel > threshold, 1, 0)
    hs, ws = np.where(mask_pixellevel > threshold)
    positions = np.array(list(zip(hs, ws))) * tile_size

    return positions, tile_img
