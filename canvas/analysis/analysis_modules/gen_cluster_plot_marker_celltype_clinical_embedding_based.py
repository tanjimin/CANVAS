import os
import sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.io import imsave
import pdb
from pathlib import Path

def gen_cluster_plot_marker_celltype_clinical_embedding_based(clinical_path, clinical_corr_path, marker_heatmap_path, kmeans_path,
                                                              channel_names, output_path, cell_type_info_exists = False, clinical_info_exists = False,
                                                              cell_counts_heatmap_path = False):
    
    marker_heatmap_path = os.path.join(marker_heatmap_path, 'heatmap.npy')
    marker_heatmap = np.load(marker_heatmap_path)
    num_clusters = marker_heatmap.shape[0]
    # Convert to dataframe
    marker_heatmap_df = pd.DataFrame(marker_heatmap, index=list(range(num_clusters)), columns = channel_names)
    count_path = os.path.join(clinical_corr_path, 'counts_table.csv')
    cluster_path = os.path.join(kmeans_path, 'clusters.npy')
    cluster_plot_color_path = os.path.join(kmeans_path, 'cluster_plot_color_rgb.npy')

    # Remove DNA1, CD94 and CD117
    '''
    marker_heatmap_df.drop(columns=['DNA1', 'CD94', 'CD117'], inplace=True)
    channel_names.remove('DNA1')
    channel_names.remove('CD94')
    channel_names.remove('CD117')
    '''
    # Reorder columns
    #channel_names_reordered = ['CD3', 'CD4', 'CD8a', 'FoxP3', 'CD20', 'CD14', 'CD16', 'CD68', 'CD163', 'CD11c', 'CD94', 'CD117', 'MPO', 'HLA-DR', 'CD31', 'PanCK', 'TTF1', 'DNA1']
    #marker_heatmap_df = marker_heatmap_df[channel_names_reordered]

    if cell_type_info_exists == True:
        channel_names = ["CD117", "CD11c", "CD14", "CD163", "CD16", "CD20", "CD31", "CD3", "CD4", "CD68", "CD8a", "CD94", "DNA1", "FoxP3", "HLA-DR", "MPO", "PanCK", "TTF1"]
        marker_heatmap = np.load(marker_heatmap_path)
        num_clusters = marker_heatmap.shape[0]
        # Convert to dataframe
        marker_heatmap_df = pd.DataFrame(marker_heatmap, index=list(range(num_clusters)), columns = channel_names)
        # Remove DNA1, CD94 and CD117
        '''
        marker_heatmap_df.drop(columns=['DNA1', 'CD94', 'CD117'], inplace=True)
        channel_names.remove('DNA1')
        channel_names.remove('CD94')
        channel_names.remove('CD117')
        '''
        # Reorder columns
        channel_names_reordered = ['CD3', 'CD4', 'CD8a', 'FoxP3', 'CD20', 'CD14', 'CD16', 'CD68', 'CD163', 'CD11c', 'CD94', 'CD117', 'MPO', 'HLA-DR', 'CD31', 'PanCK', 'TTF1', 'DNA1']
        marker_heatmap_df = marker_heatmap_df[channel_names_reordered]
        # Reorder rows
        Tcell = [0]
        Tcell_2 = [2, 39]
        Bcell = [7, 34, 38]
        Tcell_Mo = [20, 23, 43, 45, 28]
        Mo_DC = [24, 22, 13, 46]
        Mo_3 = [25, 31, 10, 42]
        Neutrophil = [49, 5, 37]
        HLA_DR = [21, 30, 8, 3, 35]
        CD31 = [32, 14, 6]
        tumor = [4, 11, 40, 16, 29, 36, 48, 33, 47, 17, 44]
        Nothing = [15, 27]
        empty = [26, 19, 12, 1, 9, 18, 41]
        new_index = np.concatenate([Tcell, Tcell_2, Bcell, Tcell_Mo, Mo_DC, 
                                    Mo_3, Neutrophil, HLA_DR, CD31, tumor, 
                                    Nothing, empty])

        marker_heatmap_df = marker_heatmap_df.reindex(new_index)
        # Index convert dict
        index_convert_dict = {}
        for i in range(len(new_index)):
            index_convert_dict[i] = np.where(new_index == i)[0][0]

        cell_counts_df = pd.read_csv(cell_counts_heatmap_path, index_col = 0).T
        # Normalize cell counts by cluster
        cell_counts_df = cell_counts_df.div(cell_counts_df.sum(axis=1), axis=0)
        # Rename columns with dictionary
        cell_counts_df.rename(columns={'Endothelial cell' : 'Endothelial'}, inplace=True)
        # Reorder rows
        cell_counts_df = cell_counts_df.reset_index().drop('index', axis = 1)
        cell_counts_df = cell_counts_df.reindex(new_index)
    
    if clinical_info_exists == True:
        plot_df, _, _ = proc_clinical(count_path, clinical_path)
    #pdb.set_trace()
    # Reset cluster index
    #plot_df['cluster'] = plot_df['cluster'].map(index_convert_dict)
    # Plot three panels

    # Set font size
    plt.rcParams.update({'font.size': 8})
    # Set font family
    from matplotlib import font_manager
    from visualization import core
    #font = '/gpfs/home/jt3545/fonts/Arial.ttf'
    #font_manager.fontManager.addfont(font)
    #plt.rcParams['font.family'] = 'Arial'
    
    #figure_size = (170/25.4,  150/25.4)
    #figure_size = (320/25.4,  300/25.4) #this works
    #figure_size = (len(reordered_data.columns) * 0.5, 10)
    
    #pdb.set_trace()

    from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
    # 1. Marker heatmap

    #cluster the markers so most similar ones are close to eachother
    # Calculate the correlation matrix
    corr = marker_heatmap_df.corr()
    # Perform hierarchical clustering
    linkage_matrix = linkage(corr, method='ward')
    # Plot the dendrogram to visualize the clustering
    plt.figure(figsize=(10, 5))
    dendro = dendrogram(linkage_matrix, labels=corr.columns, leaf_rotation=90)
    plt.show()
    # Get the order of columns based on the clustering
    ordered_columns = [corr.columns[i] for i in leaves_list(linkage_matrix)]
    # Reorder the DataFrame columns based on the clustering results
    marker_heatmap_df = marker_heatmap_df[ordered_columns]
   
    # Print the reordered DataFrame
    #print(marker_heatmap_df)
    
    #reorder the clusters to ensure a diagonal pattern in heatmap
    # Initialize a list to keep track of the ordered indices
    ordered_indices = []

    # Loop through each marker and find the index of the maximum value, ensuring no duplicates
    for marker in marker_heatmap_df.columns:
        # Get the index of the row with the maximum value for the current marker
        max_index = marker_heatmap_df[marker].idxmax()
        # Append this index to the list
        ordered_indices.append(max_index)

    # Ensure all indices are unique and maintain order
    ordered_indices = list(dict.fromkeys(ordered_indices))

    # Create a DataFrame to store the reordered rows
    reordered_data = pd.DataFrame(columns=marker_heatmap_df.columns)

    # Add the ordered rows to the new DataFrame
    for idx in ordered_indices:
        #reordered_data = reordered_data.append(marker_heatmap_df.loc[idx])
        reordered_data = pd.concat([reordered_data, marker_heatmap_df.loc[[idx]]])

    # Add any remaining rows that were not included in the ordered list
    remaining_indices = set(marker_heatmap_df.index) - set(ordered_indices)
    for idx in remaining_indices:
        #reordered_data = reordered_data.append(marker_heatmap_df.loc[idx])
        reordered_data = pd.concat([reordered_data, marker_heatmap_df.loc[[idx]]])

    # Print the reordered DataFrame
    print(reordered_data)

    # Check if each file or data exists
    available_data = {
        "marker_heatmap": marker_heatmap_df if ('marker_heatmap_df' in locals() or 'marker_heatmap_df' in globals()) else None,
        "cluster_counts": cluster_path if os.path.exists(cluster_path) else None,
        "cell_counts": cell_counts_df if cell_type_info_exists else None,
        "cell_density": cell_counts_heatmap_path if cell_type_info_exists else None,
        "clinical_data": clinical_path if clinical_info_exists else None,
    }

    # Check which files or data exist and prepare subplots accordingly
    existing_files = [key for key, path in available_data.items() if path is not None and (isinstance(path, str) and Path(path).exists() or not isinstance(path, str))]
    num_subplots = len(existing_files)

    # Define the number of subplots and width ratios based on available files
    num_subplots = len(existing_files)
    width_ratios = [1.6, 0.2, 1.5, 0.2, 1.2][:num_subplots]

    # Create the figure and axes dynamically
    figure_size = (10, 10)  # Adjust based on your needs
    fig, axes = plt.subplots(1, num_subplots, figsize=figure_size, gridspec_kw={'width_ratios': width_ratios})

    # Adjust indexing to work with the number of subplots
    for i, file_key in enumerate(existing_files):
        if file_key == "marker_heatmap":
            # Plot marker heatmap (example)
            sns.heatmap(marker_heatmap_df, ax=axes[i], cmap='RdBu_r', center=0, vmin=-1, vmax=1, cbar=False)
            axes[i].set_title('Marker expression')
            axes[i].set_ylabel('Clusters')
            axes[i].margins(y=0)
            # Customize xticks
            num_cols = len(marker_heatmap_df.columns)
            axes[i].set_xticks(np.arange(num_cols) + 0.5)
            axes[i].set_xticklabels(marker_heatmap_df.columns, rotation=90, ha='center')
            axes[i].tick_params(axis='both', which='both', length=0)

        elif file_key == "cluster_counts":
            # Plot cluster counts (example)
            cluster_labels = np.load(cluster_path)
            unique_clusters, cluster_counts = np.unique(cluster_labels, return_counts=True)
            cluster_counts_df = pd.DataFrame({'cluster': unique_clusters, 'count': cluster_counts})
            cluster_counts_df = cluster_counts_df.loc[reordered_data.index]
            palette = np.load(cluster_plot_color_path)
            axes[i].barh(cluster_counts_df['cluster'], cluster_counts_df['count'], color=palette)
            axes[i].margins(y=0)
            axes[i].invert_yaxis()
            axes[i].set_xlabel('Counts')
            axes[i].spines['top'].set_visible(False)
            axes[i].spines['right'].set_visible(False)
            # axes[i].set_xticks(np.arange(num_cols) + 0.5)
            #axes[i].set_xticklabels(cluster_counts_df['count'], rotation=45, ha='right')
            axes[i].tick_params(axis='y', which='both', length=0)
            axes[i].set_title('TME counts')
            axes[i].set_yticks(np.arange(len(cluster_counts_df)))
            axes[i].set_yticklabels(cluster_counts_df['cluster'])           

        elif file_key == "cell_counts":
            cell_list = ['B cell' ,'Cancer' , 'Cl MAC','Alt MAC','Cl Mo','Int Mo',
                         'Non-Cl Mo','DCs cell','Endothelial','Mast cell','NK cell',
                         'Neutrophils','Tc','Th','Treg','T other','None']

            cell_counts_df = cell_counts_df[cell_list]
            all_cell_types = cell_counts_df.columns
            cell_color_dict = get_cell_color_palette()
            cell_color_list = [cell_color_dict[cell_type] / 255 for cell_type in all_cell_types]
            list_cmap = matplotlib.colors.ListedColormap(cell_color_list)
            cell_counts_df.plot.barh(stacked=True, ax=axes[i], legend=False, cmap=list_cmap, width = 0.9)
            axes[i].set_title('Cell Composition')
            # Reverse y axis
            axes[i].set_ylim(-0.5, num_clusters - 0.5)
            axes[i].invert_yaxis()
            #axes[2].set_yticks([])
            axes[i].tick_params(axis='y', which='both', length=0)
            # Remove x axis
            axes[i].set_xticks([])
            #axes[i].tick_params(axis='y', which='both', length=0)
            # Remove margin
            # Remove wireframe
            axes[i].spines['top'].set_visible(False)
            axes[i].spines['right'].set_visible(False)
            axes[i].spines['bottom'].set_visible(False)
            axes[i].spines['left'].set_visible(False)
            # Add legend at bottom, 3 columns
            axes[i].legend(bbox_to_anchor=(0, 0.0, 1, 0), loc='upper center', ncol=3, mode='expand', borderaxespad=0, frameon=False)

        # Add further plotting conditions as needed for other file types
        elif file_key == "cell_density":
            cell_count_df = pd.read_csv(cell_counts_heatmap_path, index_col=0)
            # Read cluster labels
            cluster_labels = np.load(cluster_path)
            # Find number of tiles per cluster
            unique_clusters, cluster_counts = np.unique(cluster_labels, return_counts=True)

            # Transpose
            cell_count_df = cell_count_df.T
            # Reset index 
            cell_count_df = cell_count_df.reset_index().rename(columns={'index': 'sample_id'})

            # Sum all cell types by cluster
            cell_count_df['total'] = cell_count_df.sum(axis=1)

            # Add tile count according to unique_clusters
            cell_count_df['tile_count'] = cluster_counts

            # Find density
            cell_count_df['density'] = cell_count_df['total'] / cell_count_df['tile_count']
            # Reorder rows by new index
            cell_count_df = cell_count_df.reindex(new_index)
            axes[i].barh(cell_count_df['sample_id'], cell_count_df['density'], color='#C0C0C0')
            axes[i].margins(y=0)
            axes[i].invert_yaxis()
            # Remove frame
            axes[i].spines['top'].set_visible(False)
            axes[i].spines['right'].set_visible(False)
            # Remove ticks but keep line
            axes[i].tick_params(axis='y', which='both', length=0)
            axes[i].set_ylabel('')
            axes[i].set_xlabel('Counts per tile')
            axes[i].set_title('Cell Density')

        elif file_key == "clinical_data":
            #reorder plot_df to have the same order as the reordered clusters
            plot_df['cluster'] = pd.Categorical(plot_df['cluster'], categories=reordered_data.index, ordered=True)
            plot_df = plot_df.sort_values('cluster').reset_index(drop=True)

            #dot_plot = plot_df.plot.scatter(x='clinical', y='cluster', s='neg_log10pvalue_corrected', c='stats', colormap=colormap, ax=axes[i], legend=False, sharex=False)
            plot_df['neg_log10pvalue_corrected'] *= 5
            dot_plot1 = plot_df.plot.scatter(x='clinical', y='cluster', s='neg_log10pvalue_corrected', c='grey', ax=axes[i], legend=False, sharex=False)
            #dot_plot.collections[0].colorbar.remove()

            #pdb.set_trace()
            plot_df_significant = plot_df[plot_df['pvalue_corrected'] < 0.05]
            df_high = plot_df_significant[plot_df_significant['stats'] > 0]
            df_low = plot_df_significant[plot_df_significant['stats'] <= 0]
            #dot_plot2 = plot_df_significant.plot.scatter(x='clinical', y='cluster', s='neg_log10pvalue_corrected', c='stats_binary', colormap=colormap, ax=axes[i], legend=False, sharex=False)
            dot_plot2 = df_high.plot.scatter(x='clinical', y='cluster', s='neg_log10pvalue_corrected', c='#DB4437', ax=axes[i], legend=False, sharex=False, edgecolors = 'none')
            dot_plot3 = df_low.plot.scatter(x='clinical', y='cluster', s='neg_log10pvalue_corrected', c='#4285F4', ax=axes[i], legend=False, sharex=False)
            # Remove colorbar
            #sns.scatterplot(x='clinical', y='cluster', size='neg_log10pvalue', hue='stats', data=plot_df, palette = colormap, ax=axes[2], linewidth=0, size_norm=(0, 10), s=3)
            #sns.scatterplot(data=plot_df, x='PC1', y='PC2', hue='clinical', ax=axes[2], palette='tab20b', s=10, linewidth=0, legend=False)
            # Reverse y axis
            axes[i].invert_yaxis()
            axes[i].set_title('Clinical Variables')
            # Remove x y label
            axes[i].set_ylabel('')
            axes[i].set_xlabel('')
            # Remove margin
            margin_size = figure_size[1] * 0.2 / num_clusters / 2
            axes[i].margins(margin_size * 6, margin_size)
            # Turn x labels vertical
            plt.setp(axes[i].get_xticklabels(), rotation=90, ha='right')
            # Center x ticks
            for tick in axes[i].get_xticklabels():
                tick.set_horizontalalignment('center')

            # add grid
            axes[i].set_axisbelow(True)
            axes[i].grid(True, axis='x', which='major', color='lightgrey', linestyle='-', linewidth=0.5)
            axes[i].grid(True, axis='y', which='major', color='lightgrey', linestyle='-', linewidth=0.5)

            # Remove x tick marks
            axes[i].tick_params(axis='x', which='both', length=0)
            axes[i].tick_params(axis='y', which='both', length=0)
            axes[i].set_yticks(np.arange(num_clusters))
            axes[i].set_yticklabels(reordered_data.index.astype(str), rotation=0, ha='right')

            #axes[i].set_yticklabels([str(idx) for idx in new_index])
            # Hide y axis labels
            #axes[i].set_yticklabels([])
            #axes[i].get_yaxis().set_visible(False)
            # Remove wireframe
            axes[i].spines['top'].set_visible(False)
            axes[i].spines['right'].set_visible(False)
            #axes[i].spines['bottom'].set_visible(False)
            #axes[i].spines['left'].set_visible(False)

            # Plot legend
            # import Line2D
            from matplotlib.lines import Line2D
            legend_elements = [Line2D([0], [0], marker='s', color='w', label='Negative', markerfacecolor='#4285F4', markersize=4),
                            Line2D([0], [0], marker='s', color='w', label='Positive', markerfacecolor='#DB4437', markersize=4)]
            # Add legend for size   
            legend_elements.append(Line2D([0], [0], marker='o', color='w', label='P = 0.05', markerfacecolor='grey', markersize= 3.3))
            legend_elements.append(Line2D([0], [0], marker='o', color='w', label='P < 0.0001', markerfacecolor='grey', markersize= 5.3))
            # Put legend outside of plot at the bottom
            axes[i].legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=2, frameon=False, columnspacing=0.5)


    plt.tight_layout()
    plt.show()
    os.makedirs(output_path, exist_ok = True)
    plt.savefig(os.path.join(output_path, 'plot.png'))
    plt.clf()
   
def get_cell_color_palette():

    color_pallete = {'white' : (np.array([255, 255, 255]) * 0.3).astype(np.uint8),
                     'red' : np.array([255, 32, 32]),
                     'red_light' : np.array([255, 128, 128]),
                     'red_dark' : np.array([192, 16, 16]),
                     'red_darker' : np.array([128, 16, 16]),
                     'magenta' : np.array([255, 128, 255]),
                     'blue' : np.array([64, 64, 255]),
                     'green' : np.array([128, 255, 128]),
                     'yellow' : np.array([255, 255, 0]),
                     'yellow_light' : np.array([255, 255, 192]),
                     'cyan' : np.array([64, 255, 255]),
                     'cyan_light' : np.array([192, 255, 255]),
                     'cyan_dark' : np.array([64, 192, 192]),
                     'orange' : np.array([255, 128, 0]),
                     'orange_light' : np.array([255, 192, 64]),
                     'grey' : np.array([128, 128, 128]),
                     'grey_light' : np.array([192, 192, 192]),
                     'grey_lighter' : np.array([224, 224, 224]),
                     'grey_dark' : np.array([64, 64, 64]),
                     }

    cell_dict = {'B cell' : 'green',
                 'Cancer' : 'magenta',
                 'Cl MAC' : 'orange_light',
                 'Alt MAC' : 'orange',
                 'Cl Mo' : 'cyan_light',
                 'Int Mo' : 'cyan',
                 'Non-Cl Mo' : 'cyan_dark',
                 'DCs cell' : 'yellow_light',
                 'Endothelial cell' : 'yellow',
                 'Endothelial' : 'yellow',
                 'Mast cell' : 'grey_light',
                 'NK cell' : 'grey',
                 'Neutrophils' : 'blue',
                 'Tc' : 'red_light',
                 'Th' : 'red',
                 'Treg' : 'red_dark',
                 'T other' : 'red_darker',
                 'None' : 'grey_lighter'}

    cell_palette = {}
    for cell_type, color in cell_dict.items():
        cell_palette[cell_type] = darken_color(color_pallete[color])
    return cell_palette

def darken_color(color, amount=0.8):
    d_color = (color + 30) * amount
    return np.clip(d_color, 0, 255).astype(np.uint8)

def proc_clinical(count_path, clinical_path):
    #pdb.set_trace()
    # Read clinical data
    clinical_df = pd.read_csv(clinical_path)
    #clinical_df['Solid Tumor'] = clinical_df['Predominant histological pattern (Lepidic:1, Papillary: 2, Acinar: 3, Micropapillary: 4, Solid: 5)'] == 5
    keys = clinical_df['Key']

    #clinical_hist = clinical_df['Predominant histological pattern (Lepidic:1, Papillary: 2, Acinar: 3, Micropapillary: 4, Solid: 5)']
    #clinical_df.drop(columns=['Key', 'Predominant histological pattern (Lepidic:1, Papillary: 2, Acinar: 3, Micropapillary: 4, Solid: 5)'], inplace=True)
    n_clinical = len(clinical_df.columns) # Ignore histology

    # Read counts data
    plot_df = pd.read_csv(count_path)
    # Convert na to 0
    plot_df = plot_df.fillna(0)

    # Normalize counts by total sample counts
    sample_counts_dict = {}
    for sample in plot_df['sample'].unique():
        sample_counts_dict[sample] = plot_df[plot_df['sample'] == sample]['count'].sum()

    plot_df['count_ratio'] = plot_df.apply(lambda row: row['count'] / sample_counts_dict[row['sample']], axis=1)

    # Generate dot grid
    pvalue_grid = np.ones((len(np.unique(plot_df['cluster'])), n_clinical))
    effetive_size_grid = np.ones((len(np.unique(plot_df['cluster'])), n_clinical))
    plot_list = []

    #this shortens the column names and only keeps the part of the name before the parenthesis
    #clinical_var_renamed = [col.split('(')[0].strip() for col in clinical_df.columns]
    #clinical_df.columns = clinical_var_renamed

    # def convert_to_float_if_int(val):
    #     try:
    #         return float(val) if isinstance(val, int) else val
    #     except ValueError:
    #         return val

    #if the column is binary, convert to 0 and 1 to prepare to do stats. If the column is not binary, create a new column that compares one output vs other 
    def convert_binary_to_0_1(df):
        for col in clinical_df.columns:
            if col == 'Key':
                continue
            unique_values = list(df[col].dropna().unique()) #don't include NaN in the unique values!! We only care about the values we have!!
            if len(unique_values) == 2:
                if set(unique_values)  == {0, 1}:
                    print("no changes needed")
                else:
                    df[col] = df[col].replace({unique_values[0]: 0, unique_values[1]: 1})
                    print(f'{col} - converting {unique_values[0]}: 0 and {unique_values[1]}:1')
            elif len(unique_values) > 2:
                for i in range(len(unique_values)):
                    #create new binary columns 
                    df[f'{col}: {unique_values[i]} vs other'] = 0 #initialize the column
                    indices_of_interest = df.index[df[col] == unique_values[i]] #identify indices of interest
                    df.loc[indices_of_interest, f'{col}: {unique_values[i]} vs other'] = 1
                df.drop(columns=[col], inplace=True)
            elif len(unique_values) < 2: #if the column only has 1 unique var, then there are no stats to be done to compare!
                df.drop(columns=[col], inplace=True)
        return(df)

    for cluster in np.unique(plot_df['cluster']):
        cluster_df = plot_df[plot_df['cluster'] == cluster]
        cluster_df = convert_binary_to_0_1(cluster_df)
        # Columns to exclude
        exclude_columns = ['cluster', 'sample', 'count', 'sample_normalized_count', 'count_ratio']
        for c_i, c_name in enumerate(cluster_df.columns):
            if c_name in exclude_columns: continue
            if c_name == 'Key': continue
            c_i -= 1
            print(f'Computing {c_name}')
            c_idx = np.array(cluster_df[c_name]).astype(int)
            
            '''
            # Centroid score 
            c_score = np.load('/gpfs/data/tsirigoslab/home/jt3545/CODEX/codex-imaging/src/model/feature_extraction/features/imc/s4-1_cluster_kmeans/thumbnail_scale_16/64/checkpoint-1999/default/50/centroid_euclidean_score.npy')
            sample_dict = dict(zip(sorted(sample_counts_dict.keys()), range(len(sample_counts_dict))))
            sample_dict_r = sorted(sample_counts_dict.keys())

            c_score_tiles = c_score[:, cluster]

            # load sample names for each tile
            sample_names = np.load('/gpfs/data/tsirigoslab/home/jt3545/CODEX/codex-imaging/src/model/feature_extraction/features/imc/s7-1_annotated_samples/thumbnail_scale_16/64/checkpoint-1999/default/sample_name.npy')

            # Sort cluster df by sample name
            cluster_df['sample_idx'] = cluster_df['sample'].apply(lambda x: sample_dict[x])
            cluster_df.sort_values(by=['sample_idx'], inplace=True)
            cluster_df.drop(columns=['sample_idx'], inplace=True)

            # Generate df with sample names and values in c_name
            sample_df = pd.DataFrame({'sample' : sample_names})
            c_name_mapping = dict(zip(cluster_df['sample'], cluster_df[c_name]))
            sample_df[c_name] = sample_df['sample'].map(c_name_mapping)

            # Score average by sample
            sample_df['score'] = c_score_tiles
            #sample_df_grouped = sample_df.groupby(['sample']).mean().reset_index()

            # Compute correlation between clinical data and centroid score
            import scipy.stats as stats
            #score = c_score_tiles
            #value = sample_df[c_name]
            score = sample_df['score']
            value = sample_df[c_name]

            # Separate two groups based on score 10 percentile
            score_threshold = np.percentile(score, 98)
            score_low = score[score < score_threshold]
            score_high = score[score >= score_threshold]

            group1 = value[score >= score_threshold]
            group2 = value[score < score_threshold]

            # Compare group 1 and group 2 calculate statistic and p-value
            stats, pvalue = stats.mannwhitneyu(group1, group2, alternative='two-sided')
            # Two-sided ttest
            #stats, pvalue = stats.ttest_ind(group1, group2, equal_var=False)

            simulation_count = 3000
            #stats, pvalue = stats.ttest_ind(group1, group2, permutations=simulation_count)

            # Compute correlation and p-value
            #stats, pvalue = stats.pearsonr(score, value)
            #stats, pvalue = stats.spearmanr(score, value)

            #pvalue *= 50 * 9

            '''
            #pdb.set_trace()
            # Separate to two groups
            group1 = cluster_df.iloc[c_idx == 1]
            group2 = cluster_df.iloc[c_idx == 0]
            assert len(group1) + len(group2) == len(clinical_df)
            # Perform t-test
            from scipy.stats import ttest_ind, mannwhitneyu
            simulation_count = 3000
            #stats, pvalue = ttest_ind(group1['count_ratio'], group2['count_ratio'], permutations=simulation_count)
            #stats, ppvalue = ttest_ind(group1['count_ratio'], group2['count_ratio'])
            #log_fold_change = np.log(np.mean(group1['count_ratio']) / np.mean(group2['count_ratio']))
            mean_diff = np.mean(group1['count_ratio']) - np.mean(group2['count_ratio'])
            stats = mean_diff
            mstats, pvalue = mannwhitneyu(group1['count_ratio'], group2['count_ratio'], alternative='two-sided')
            #if pvalue < 0.05:
            #    print(f'pvalue: {pvalue}')

            neg_log10pvalue = - np.log10(pvalue + 1e-10)
            max_stats = 0.5
            if stats < -max_stats:
                stats = -max_stats
            elif stats > max_stats:
                stats = max_stats
            #if pvalue > 0.10 / (50):
            #    print(f'pvalue {pvalue} is not significant')
            #    stats = np.nan

            stats_binary = (stats > 0) * 1

            #pvalue_grid[cluster, c_i] = neg_log10pvalue
            #effetive_size_grid[cluster, c_i] = (group1['count'].mean() - group2['count'].mean()) / np.sqrt((group1['count'].std() ** 2 + group2['count'].std() ** 2) / 2)
            #effetive_size_grid[cluster, c_i] = stats_binary

            plot_dict = {'cluster' : cluster,
                         'clinical' : c_name,
                         'pvalue' : pvalue,
                         'neg_log10pvalue' : neg_log10pvalue,
                         'stats' : stats,
                         'stats_binary' : stats_binary,
                         'effect_size' : stats, #(group1['count'].mean() - group2['count'].mean()) / np.sqrt((group1['count'].std() ** 2 + group2['count'].std() ** 2) / 2),
                         }
            plot_list.append(plot_dict)
    plot_df = pd.DataFrame(plot_list)
    #plot_df.fillna(0, inplace=True)
    

    # FDR correction for multiple testing on p-values for each cluster
    from statsmodels.stats.multitest import multipletests
    '''
    # Select p-values for each cluster
    corrected_grid = np.zeros_like(pvalue_grid)
    for cluster in range(50):
        pvalues = pvalue_grid[:, cluster]
        # FDR correction
        reject, pvals_corrected, alphacSidak, alphacBonf = multipletests(pvalues, alpha=0.05, method='fdr_bh')
        corrected_grid[:, cluster] = pvals_corrected
    '''
    plot_df['pvalue_corrected'] = 0
    for cluster in plot_df['cluster'].unique():
        cluster_index = plot_df['cluster'] == cluster
        pvalues = plot_df.loc[cluster_index, 'pvalue']
        # FDR correction
        reject, pvals_corrected, alphacSidak, alphacBonf = multipletests(pvalues, alpha=0.05, method='fdr_bh')
        plot_df.loc[cluster_index, 'pvalue_corrected'] = pvals_corrected

    '''
    corrected_pvalue = multipletests(plot_df['pvalue'], alpha=0.05, method='fdr_bh')[1]
    plot_df['pvalue_corrected'] = corrected_pvalue
    '''

    plot_df['neg_log10pvalue_corrected'] = - np.log10(plot_df['pvalue_corrected'] + 1e-10)

    return plot_df, pvalue_grid, effetive_size_grid


if __name__ == '__main__':
    main()

