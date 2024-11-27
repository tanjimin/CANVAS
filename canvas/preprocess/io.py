import numpy as np
import os
import tifffile
import zarr

def tiff_to_zarr(input_path, output_path, dummy_input_path, file_name, input_ext='tiff', chunk_size=(None, 256, 256)):
    # Create zarr file and channel.csv for each sample
    input_file = f'{input_path}/{file_name}.{input_ext}'
    output_file_path = f'{output_path}/{file_name}'
    dummy_input_file_path = f'{dummy_input_path}/{file_name}'

    input_dummy_file = f'{dummy_input_file_path}.dummy_{input_ext}'

    # Create dummy file if not exists
    if not os.path.exists(input_dummy_file):
        # Create empty dummy file
        with open(input_dummy_file, 'w') as f:
            f.write('')

    os.makedirs(output_file_path, exist_ok=True)
    output_zarr = f'{output_file_path}/data.zarr'

    if os.path.exists(output_zarr): # Skip if already exists
        print(f'Zarr file already exists at {output_zarr}')
        return 

    # Process channel
    input_channel_file = f'{input_path}/{file_name}.txt'
    output_channel_file = f'{output_file_path}/channels.csv'
    if os.path.exists(input_channel_file): # Check if channel file exists
        with open(input_channel_file, 'r') as f:
            channels = f.read().splitlines()
    else:
        raise ValueError(f'Channel file does not exist at {input_channel_file}')

    # Read the TIFF file
    with tifffile.TiffFile(input_file) as tif:
        img_data = tif.asarray().astype(int)

    # Check if channels matches the number of channels given
    assert img_data.shape[0] == len(channels), f'Number of channels in the file does not match the number of channels provided. File has {len(img_data.shape)} channels and {len(channels)} channels were provided.'

    with open(output_channel_file, 'w') as file:
        file.write('channel,marker\n')
        for channel, marker in enumerate(channels):
            file.write(f'{channel},{marker}\n')

    # Convert to Zarr Array
    zarr.array(img_data, chunks=chunk_size, store=output_zarr)
    #print(f'Successfully converted {file_name}.{input_ext} to zarr')

def mcd_to_zarr(input_path, output_path, dummy_input_path, file_name, input_ext='mcd', chunk_size=(None, 256, 256)):
    # Create zarr file and channel.csv for each sample
    input_file = f'{input_path}/{file_name}.{input_ext}'
    output_file_path = f'{output_path}/{file_name}'
    dummy_input_file_path = f'{dummy_input_path}/{file_name}'

    # Read the mcd file
    import pyimc
    data = pyimc.Mcd.parse(input_file)
    acquisition_ids = data.acquisition_ids()
    for acquisition_id in acquisition_ids:
        img_data, label_list = extract_acquisition(data, acquisition_id)

        output_channel_file = f'{output_file_path}_acquisition_{acquisition_id}/channels.csv'
        output_zarr = f'{output_file_path}_acquisition_{acquisition_id}/data.zarr'
        input_dummy_file = f'{dummy_input_file_path}_acquisition_{acquisition_id}.dummy_{input_ext}'

        # Create dummy file if not exists
        if not os.path.exists(input_dummy_file):
            # Create empty dummy file
            with open(input_dummy_file, 'w') as f:
                f.write('')

        if os.path.exists(output_zarr): # Skip if already exists
            print(f'Zarr file already exists at {output_zarr}')
            return 

        os.makedirs(f'{output_file_path}_acquisition_{acquisition_id}', exist_ok=True)

        # Read the channel file
        input_channel_file = f'{input_path}/{file_name}.txt'
        if os.path.exists(input_channel_file): # Check if channel file exists
            with open(input_channel_file, 'r') as f:
                channels = f.read().splitlines()
        else:
            raise ValueError(f'Channel file does not exist at {input_channel_file}')

        # Check if channels matches the number of channels given
        assert img_data.shape[0] == len(channels), f'Number of channels in the file does not match the number of channels provided. File has {len(img_data.shape)} channels and {len(channels)} channels were provided.'

        with open(output_channel_file, 'w') as file:
            file.write('channel,marker\n')
            for channel, marker in enumerate(channels):
                file.write(f'{channel},{marker}\n')

        # Convert to Zarr Array
        zarr.array(img_data, chunks=chunk_size, store=output_zarr)
        print(f'Successfully converted {file_name}_acquisition_{acquisition_id} to zarr')

def extract_acquisition(data, acquisition_id):
    acquisition = data.acquisition(acquisition_id)
    channels = acquisition.channels()
    data_list = []
    label_list = []
    for channel in channels:
        data_list.append(acquisition.channel_data(channel) * 1.0)
        label_list.append(channel.label())
    data = np.stack(data_list, axis=0)
    return data, label_list