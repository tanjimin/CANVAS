import os

import zarr
import numpy as np
import pandas as pd
import torch.utils.data as data
import torch
import math
import sys

from canvas.model.data.slide_dataset import SlideDataset

class NPYDataset(SlideDataset):

    def __init__(self, root_path = None, tile_size = None, tiles_dir = None, transform = None, lazy = True):
        super().__init__(root_path, tile_size, tiles_dir, transform)
        self.slide = self.read_slide(root_path, lazy)
        self.read_counter = 0

    def read_slide(self, file_path, lazy):
        ''' Read numpy file on disk mapped to memory '''
        numpy_path = f'{file_path}/data/core.npy'
        if lazy:
            slide = np.load(numpy_path, mmap_mode = 'r', allow_pickle = True)
        else:
            slide = np.load(numpy_path, allow_pickle = True)
        return slide

    def read_region(self, pos_x, pos_y, width, height):
        ''' Read a numpy slide region '''
        region_np = self.slide[:, pos_x:pos_x+width, pos_y:pos_y+height].copy()
        # Swap channel to last dimension
        region_np = region_np.swapaxes(0, 1).swapaxes(1, 2)
        region = region_np.swapaxes(0, 1) # Change to numpy format
        self.read_counter += 1
        return region

    def get_slide_dimensions(self):
        ''' Get slide dimensions '''
        return self.slide.shape[0:2]

    # Generate thumbnail
    def generate_thumbnail(self, scaling_factor):
        tile_cache_size = 50 * scaling_factor
        cache = self.reduce_by_tile(self.slide, tile_cache_size, scaling_factor)
        thumbnail = cache.swapaxes(0, 1).astype(np.uint8)
        return thumbnail

    def reduce_by_tile(self, slide, tile_size, scaling_factor):
        from skimage.measure import block_reduce
        from tqdm import tqdm
        dims = self.get_slide_dimensions()
        cache = np.zeros((dims[0] // scaling_factor, dims[1] // scaling_factor, 4), dtype = np.uint8)
        for x in tqdm(range(0, dims[0], tile_size)):
            for y in range(0, dims[1], tile_size):
                tile = self.read_region(x, y, tile_size, tile_size).swapaxes(0, 1)
                reduced_tile = block_reduce(tile, block_size=(scaling_factor, scaling_factor, 1), func=np.mean)
                x_reduced = x // scaling_factor
                y_reduced = y // scaling_factor
                x_end = min(x_reduced + tile_size // scaling_factor, cache.shape[0])
                y_end = min(y_reduced + tile_size // scaling_factor, cache.shape[1])
                cache[x_reduced:x_end, y_reduced:y_end, :] = reduced_tile[:x_end - x_reduced, :y_end - y_reduced, :]
                self.slide = self.read_slide(self.root_path, lazy = True)
        return cache

    def save_thumbnail(self, scaling_factor = 32):
        from skimage.io import imsave
        ''' Save thumbnail of the slide '''
        thumbnail = self.generate_thumbnail(scaling_factor)
        os.makedirs(f'{self.root_path}/thumbnails', exist_ok=True)
        imsave(f'{self.root_path}/thumbnails/npy_{scaling_factor}_bin.png', thumbnail)

class ZarrDataset(NPYDataset):

    def read_slide(self, file_path, lazy = True):
        ''' Read zarr file on disk '''
        zarr_path = f'{file_path}/data.zarr'
        slide = zarr.open(zarr_path, mode = 'r')
        return slide

class CANVASDataset(ZarrDataset):

    def __init__(self, root_path, tile_size, tiles_dir, common_channel_names : [str], transform = None, lazy = True):
        super().__init__(root_path, tile_size, tiles_dir, transform)
        self.tiles_dir = tiles_dir
        self.root_path = root_path
        self.slide = self.read_slide(root_path, lazy)
        self.read_counter = 0
        self.common_channel_names = common_channel_names
        self.channel_idx = self.get_channel_idx(common_channel_names)

    def __getitem__(self, index):
        image, label, x, y, img_id = super().__getitem__(index)
        # Move channel to first dimension
        if not self.channel_idx is None:
            image = image[self.channel_idx, :, :]
        dummy_label = self.root_path.split('/')[-1]
        return image, dummy_label

    def get_channel_idx(self, channel_names):
        ''' Get channel index from channel names '''
        channel_df = pd.read_csv(f'{self.root_path}/channels.csv')
        channel_dict = dict(zip(channel_df['marker'], channel_df['channel']))
        channel_idx = [channel_dict[channel_name] for channel_name in channel_names]
        return channel_idx

class CANVASDatasetWithLocation(CANVASDataset):

    def __getitem__(self, index):
        image, sample_label = super().__getitem__(index)
        location = self.tile_pos[index]
        return image, (sample_label, location)

class SlidesDataset(data.Dataset):
    ''' Dataset for a list of slides '''

    def __init__(self, slides_root_path = None, tile_size = None, post_normalization_scaling_strategy = None, cap_cutoff = None, perc_thres = None, tiles_dir = None, stats_dir = 'stats', transform = None, dataset_class = None, use_normalization = True):
        self.slides_root_path = slides_root_path
        self.tile_size = tile_size
        self.transform = transform
        self.tiles_dir = tiles_dir
        self.stats_dir = stats_dir
        print("tiles_dir: ",tiles_dir)
        
        # Get id and path for all slides
        slide_ids = self.get_slide_paths(slides_root_path)
        self.common_channel_names = self.get_common_channel_names(self.slides_root_path)
        self.slides_dict, self.lengths = self.get_slides(slide_ids, dataset_class, self.common_channel_names)
        self.mean = None
        self.std = None
        self.use_normalization = use_normalization

        self.post_normalization_scaling_strategy = post_normalization_scaling_strategy
        self.cap_cutoff = cap_cutoff 
        self.perc_thres = perc_thres
        #pdb.set_trace()

       

        #pick the method you want to calculate normalization stats- tile level or pixel level
        #self.mean, self.std = self.get_normalization_stats_pixel_level() #pixel level 
        if self.post_normalization_scaling_strategy == "threshold":
            self.mean, self.std, self.thres_norm, self.scalingmethod_norm = self.get_normalization_stats_tile_level()
        else:
            self.mean, self.std = self.get_normalization_stats_tile_level()

    def __getitem__(self, index):
        for slide_idx, (slide_id, slide) in enumerate(self.slides_dict.items()):
            if index < self.lengths[slide_idx]:
                image, label = slide[index]
                # Check if already initialized
                if not self.use_normalization:
                    return image, label
                if not self.mean is None:
                    image_norm = (image - self.mean) / self.std

                    if self.post_normalization_scaling_strategy == 'cap':
                        image = torch.clamp(image_norm, min = None, max = self.cap_cutoff)
                        print("min and max post capping: ", torch.min(image), torch.max(image))
                    
                    if self.post_normalization_scaling_strategy == 'threshold':
                        thres_norm = torch.tensor(self.thres_norm)

                        #reshape to 3D
                        thres_norm = thres_norm.reshape(thres_norm.shape[0],1,1)

                        scalingmethod_1_0 = np.where(self.scalingmethod_norm == 'nonlog', 0, 1)
                        scalingmethod_1_0 = torch.tensor(scalingmethod_1_0)
                        scalingmethod_1_0 = scalingmethod_1_0.reshape(scalingmethod_1_0.shape[0],1,1)
                    
                        cond = (image_norm > thres_norm)

                        def my_function(x):
                            #the default of torch.log is base e!!
                            scaling_res = torch.where(scalingmethod_1_0 == 1, torch.log(x - thres_norm + 1) + thres_norm, .1 * (x - thres_norm) + thres_norm)
                            #still needs the +1 because if x is only a little bit greater than the threshold, the value will be <1 and log will give a negative value!! Adding the 1 prevents this from happening. 
                            return scaling_res   
                    
                        #Apply the function only where the condition is True. This is only changing the values if they are greater than the threshold!! 
                        #scale after normalizing 
                        image_norm_scaled = torch.where(cond, my_function(image_norm), image_norm) #This is using [0,255] data!! 
         
                        #for i in range(len(self.common_channel_names)):
                            #print(self.common_channel_names[i])
                            #print(f"original min and max for channel {self.common_channel_names[i]}", torch.min(image[i]), torch.max(image[i]))
                            #print(f"pre-scaled normalized min and max:", torch.min(image_norm[i]), torch.max(image_norm[i]))
                            #print(f"scaled normalized min and max:", torch.min(image_norm_scaled[i]), torch.max(image_norm_scaled[i]))
                            #print(" ")
                        
                        image = image_norm_scaled

                    else:
                        image = image_norm

                return image, label
            else:
                index -= self.lengths[slide_idx] #index = index - self.lengths[slide_idx]

    def __len__(self):
        return sum(self.lengths)

    def get_common_channel_names(self, root_path):
        with open(f'{root_path}/common_channels.txt', 'r') as f:
            channel_names = f.read().splitlines()
        return channel_names

    def get_normalization_stats_pixel_level(self): #these mean and std values were generated based on random pixels
        import pandas as pd
        stats_path = f'{self.slides_root_path}/../../qc/global/normalization/normalization_stats.csv'
        df = pd.read_csv(stats_path)
        mean = np.array(df['mean'].values).astype(np.float32).reshape(-1, 1, 1)
        std = np.array(df['std'].values).astype(np.float32).reshape(-1, 1, 1)
        return mean, std

    def get_normalization_stats_tile_level(self): #this generates mean and std based on random tiles
        ''' Get normalization stats across samples '''
        from tqdm import tqdm
        import glob
        mean = 0
        std = 0
        stats_path = f'{self.slides_root_path}/../{self.stats_dir}'
        print("stats path: ", stats_path)
        print("total number of tiles across all slides: ", len(self))

        if self.post_normalization_scaling_strategy != 'threshold':
            if os.path.exists(f'{stats_path}/mean.npy') and os.path.exists(f'{stats_path}/std.npy'):
                mean = np.load(f'{stats_path}/mean.npy')
                std = np.load(f'{stats_path}/std.npy')
            else:
                # Generate random samples with seed
                rand_state = np.random.RandomState(42)
                rand_idices = rand_state.randint(0, len(self), size = 1000)

                n_samples = 0
                for i in tqdm(rand_idices):
                    image, label  = self.__getitem__(i)
                    mean += image.mean(axis = (1, 2))
                    std += image.std(axis = (1, 2))
                    n_samples += 1
                mean /= n_samples
                std /= n_samples
                mean = mean[:, np.newaxis, np.newaxis]
                std = std[:, np.newaxis, np.newaxis]
                # Save stats
                os.makedirs(stats_path, exist_ok = True)
                np.save(f'{stats_path}/mean.npy', mean)
                np.save(f'{stats_path}/std.npy', std)
            return mean, std

        if self.post_normalization_scaling_strategy == 'threshold':
            #load the max_top*perc_thres_norm.npy file if it exists
            thres_norm_files = glob.glob(f'{stats_path}/max_top*perc_thres_norm.npy')
            thres_norm_exists = bool(thres_norm_files)
            # Load mean and std if exists
            if os.path.exists(f'{stats_path}/mean.npy') and os.path.exists(f'{stats_path}/std.npy') and os.path.exists(f'{stats_path}/max_values_norm.npy') and thres_norm_exists:
                print("using previously generated mean and std numpy files")
                mean = np.load(f'{stats_path}/mean.npy')
                std = np.load(f'{stats_path}/std.npy')
                max_values_norm = np.load(f'{stats_path}/max_values_norm.npy')
                thres_norm_path = glob.glob(f'{stats_path}/max_top*perc_thres_norm.npy')
                thres_norm = np.load(thres_norm_path[0],allow_pickle=True)
            else:
                print("Generating new stats files")
                # Generate random samples with seed
                rand_state = np.random.RandomState(42) #, 43, 44
                rand_indices = rand_state.randint(0, len(self), size = 1000) 
                print("Total number of tiles used for generating stats: ", len(rand_indices))

                mean_total = torch.zeros(len(self.common_channel_names))
                std_total = torch.zeros(len(self.common_channel_names))

                max_top_perc_thres_prenorm = torch.zeros(len(self.common_channel_names))

                max_values_prenorm = torch.zeros(len(self.common_channel_names))
                min_values_prenorm = torch.zeros(len(self.common_channel_names))

                df_files_exist = False
                if os.path.exists(stats_path):
                    files = os.listdir(stats_path)
                    print(files)
                    df_files_exist = any(file.startswith("df_") for file in files)
                print("Do the df files exist?:", df_files_exist)
                # I only want to run the following if the df files don't exist or mean, std, or max_values_norm don't exist
                if not df_files_exist or not os.path.exists(f'{stats_path}/mean.npy') or not os.path.exists(f'{stats_path}/std.npy') or not os.path.exists(f'{stats_path}/max_values_norm.npy'):
                    n_samples = 0
                    #this checks if df_files_exist is False
                    for i in tqdm(rand_indices): #loop through each of the random tiles and calculate the mean and std
                        image, label  = self.__getitem__(i)
                        mean_total += image.mean(axis = (1, 2)) #calculating the mean along the height and width dimensions of each channel individually, resulting in 36 mean values, one for each channel.
                        std_total += image.std(axis = (1, 2))
                        top_perc_thres_prenorm = np.percentile(image, 100-self.perc_thres, axis=(1,2))
                        
                        #this is for calculating the MAX of the top 5% 
                        max_top_perc_thres_prenorm = torch.max(max_top_perc_thres_prenorm, torch.tensor(top_perc_thres_prenorm))

                        max_values_per_channel_prenorm = []
                        min_values_per_channel_prenorm = []
                        
                        #calculate the max pre and post norm
                        for chan in range(image.shape[0]):
                            # Compute the maximum value for the current channel
                            max_value_prenorm = torch.max(image[chan]) 
                            min_value_prenorm = torch.min(image[chan])

                            # Append the maximum value to the list
                            max_values_per_channel_prenorm.append(max_value_prenorm)
                            min_values_per_channel_prenorm.append(min_value_prenorm)

                        max_values_per_channel_prenorm = torch.tensor(max_values_per_channel_prenorm)
                        min_values_per_channel_prenorm = torch.tensor(min_values_per_channel_prenorm)
                                    
                        max_values_prenorm = torch.max(max_values_prenorm, max_values_per_channel_prenorm)
                        min_values_prenorm = torch.min(min_values_prenorm, min_values_per_channel_prenorm)

                        n_samples += 1

                    mean_avg = mean_total / n_samples
                    std_avg = std_total / n_samples

                    mean_avg = mean_avg[:, np.newaxis, np.newaxis]
                    std_avg = std_avg[:, np.newaxis, np.newaxis]

                    #turn the thresholds into 3 dimensions!!
                    max_top_perc_thres_prenorm = max_top_perc_thres_prenorm[:, np.newaxis, np.newaxis]

                    max_values_norm = ((max_values_prenorm.reshape(max_values_prenorm.shape[0],1,1)) - mean_avg)/std_avg
                    max_values_norm_1d = max_values_norm.reshape(-1)

                    min_values_norm = ((min_values_prenorm.reshape(max_values_prenorm.shape[0],1,1)) - mean_avg)/std_avg
                    min_values_norm_1d = min_values_norm.reshape(-1)
                    
                    mean_1d = mean_avg.reshape(-1)
                    std_1d = std_avg.reshape(-1)

                    max_top_perc_thres_prenorm_1d = max_top_perc_thres_prenorm.reshape(-1)

                    max_top_perc_thres_norm_1d = ((max_top_perc_thres_prenorm_1d - mean_1d) / std_1d)

                    df_scalingstrategies_threshold_perc = pd.DataFrame({'channel':self.common_channel_names,'mean':mean_1d,'std':std_1d, 'min_prenorm':min_values_prenorm, 'max_prenorm':max_values_prenorm, 'threshold_top_perc_prenorm_max': max_top_perc_thres_prenorm_1d, 'min_norm':min_values_norm_1d,'max_norm':max_values_norm_1d, f'threshold_top{self.perc_thres}perc_norm_max':max_top_perc_thres_norm_1d}) 
                    
                    os.makedirs(stats_path, exist_ok = True)

                    #check this csv file to ensure that the thresholds are what you want based on the threshold percentage you picked
                    df_scalingstrategies_threshold_perc.to_csv(f'{stats_path}/df_scalingstrategies_threshold_{self.perc_thres}perc.csv', index=False)  # Set index=False to exclude index from the CSV
                    
                    np.save(f'{stats_path}/mean.npy', mean_avg)
                    np.save(f'{stats_path}/std.npy', std_avg)
                    np.save(f'{stats_path}/max_values_norm.npy', max_values_norm)
                    mean = mean_avg
                    std = std_avg

                    print("mean: ", mean)
                    print("std: ", std)
                    
                #load the thres_prenorm and thres_norm from the df of the perc you care about!!
                df_thres = pd.read_csv(f'{stats_path}/df_scalingstrategies_threshold_{self.perc_thres}perc.csv')
                thres_norm = df_thres.loc[:,f'threshold_top{self.perc_thres}perc_norm_max']
                np.save(f'{stats_path}/max_top{self.perc_thres}perc_thres_norm.npy', thres_norm)

            if not os.path.exists(f'{stats_path}/scalingmethod.npy'):
                print("generating scaling method!!")
                max_values_norm = np.load(f'{stats_path}/max_values_norm.npy') #3D
                mean = np.load(f'{stats_path}/mean.npy')
                std = np.load(f'{stats_path}/std.npy')
                mean_1d = mean.reshape(-1)
                std_1d = std.reshape(-1)
                max_values_norm_1d = max_values_norm.reshape(-1)

                df = pd.read_csv(f'{stats_path}/df_scalingstrategies_threshold_{self.perc_thres}perc.csv')

                df_mean_std_chan = pd.DataFrame({'channel':self.common_channel_names,'mean':mean_1d,'std':std_1d, 'threshold_norm':thres_norm, 'max_norm':max_values_norm_1d}) 
                df_mean_std_chan['(max-threshold)_norm'] = (df_mean_std_chan['max_norm'] - df_mean_std_chan['threshold_norm']) 
                
                scaling_thres = np.percentile(df_mean_std_chan['(max-threshold)_norm'], 75) #only apply log to the most skewed markers (top 25%)
                scalingmethod_norm = np.where(df_mean_std_chan['(max-threshold)_norm'] > float(scaling_thres), 'log', 'nonlog')
                np.save(f'{stats_path}/scalingmethod.npy', scalingmethod_norm)

                df_mean_std_chan['scaling method'] = scalingmethod_norm

                df_mean_std_chan.to_csv(f'{stats_path}/mean_std_max_threshold_df.txt',sep=',', index=False)
                print("successfully generated summary stat file")
            
            else: 
                scalingmethod_norm = np.load(f'{stats_path}/scalingmethod.npy')
                
            return mean, std, thres_norm, scalingmethod_norm

    def get_slide_paths(self, slides_root_path):
        ''' Get slides from a directory '''
        slide_ids = []
        slide_channels = []
        slide_channel_dicts = []
        for slide_id in os.listdir(slides_root_path):
            if os.path.isdir(os.path.join(slides_root_path, slide_id)) and not slide_id.startswith('.') and 'V' not in slide_id:
                mat = zarr.open(f'{slides_root_path}/{slide_id}/data.zarr', mode = 'r')
                channel_df = pd.read_csv(f'{slides_root_path}/{slide_id}/channels.csv')
                channel_dict = dict(zip(channel_df['channel'], channel_df['marker']))
                slide_channels.append(mat.shape[0])
                slide_channel_dicts.append(channel_dict)
                slide_ids.append(slide_id)
        # Check if all slides have the same channels
        print(f'Found {len(slide_ids)} slides with {slide_channels} channels')

        common_channels_path = f'{slides_root_path}/common_channels.txt'
        if not os.path.exists(common_channels_path):
            common_channels = self.get_common_channels(slide_channel_dicts)
            # Save common channels as txt file
            with open(common_channels_path, 'w') as f:
                for channel in common_channels:
                    f.write(f'{channel}\n')
            if len(set(slide_channels)) > 1 or len(set([tuple(channel_dict.values()) for channel_dict in slide_channel_dicts])) > 1:
                raise Exception(f'All slides must have the same channels, common channel file is written to {common_channels_path}, PLEASE REVIEW')
            else:
                raise Exception(f'All slides DO have the same channels, common channel file is written to {common_channels_path}, PLEASE REVIEW and remove unnecessary channels')
        return slide_ids

    def get_common_channels(self, slide_channel_dicts):
        ''' Get common channels for a list of slides '''
        common_markers = [] # Channel dict: channel -> marker
        for channel_dict in slide_channel_dicts:
            common_markers.append(set(channel_dict.values()))
        common_markers = set.intersection(*common_markers)
        return common_markers

    def get_slides(self, slide_ids, dataset_class, common_channel_names):
        ''' Get slides from a list of slide ids '''
        from tqdm import tqdm
        slides_dict = {}
        lengths = []
        print('Loading slides...')
        for slide_id in tqdm(slide_ids):
            slide_path = os.path.join(self.slides_root_path, slide_id)
            slide = dataset_class(slide_path, self.tile_size, self.tiles_dir, common_channel_names, self.transform)
            slides_dict[slide_id] = slide
            lengths.append(len(slide))
        return slides_dict, lengths

