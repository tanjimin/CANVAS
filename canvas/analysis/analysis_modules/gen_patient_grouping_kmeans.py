from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

from utils import helper
from types import SimpleNamespace
import pdb

import os

def gen_patient_grouping_kmeans(save_path, clinical_path, n_clusters, clinical_var_labels):
  
    from canvas.visualization.core import subplots, figure
    
    os.makedirs(save_path, exist_ok = True)

    output_cluster_sample_csv = os.path.join(save_path, 'cluster_vs_sample.csv')

    data = pd.read_csv(output_cluster_sample_csv)

    # Drop the Unnamed: 0 column
    data = data.drop(columns=['Unnamed: 0']).T

    # Normalize the data so that each sample's cluster_ids add up to 1
    data_normalized = data.div(data.sum(axis=1), axis=0)

    # Perform KMeans clustering with k = 17 on the normalized data
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(data_normalized)

    # Get the cluster labels
    cluster_labels = kmeans.labels_

    # Save the cluster labels
    np.savetxt(save_path + '/cluster_labels.csv', cluster_labels, delimiter=',')

    # Reorder the data and cluster so that the samples are grouped by cluster
    data_reordered = data_normalized.copy()
    data_reordered['cluster_id'] = cluster_labels
    data_reordered = data_reordered.sort_values(by=['cluster_id'])
    cluster_labels_reordered = data_reordered['cluster_id']
    data_reordered = data_reordered.drop(columns=['cluster_id'])

    # Reorder columns for each group so that columns have enrichment in diagonal
    # Get unique cluster labels
    unique_cluster_labels = np.unique(cluster_labels_reordered)
    label_rows_dict = {}
    for cluster_label in unique_cluster_labels:
        label_rows_dict[cluster_label] = data_reordered.index[cluster_labels_reordered == cluster_label]

    # For each column find the cluster label that has the most enrichment in that column
    data_reordered_columns = data_reordered.columns
    column_cluster_labels = []
    for column in data_reordered_columns:
        column_enrichment = data_reordered[column]
        column_enrichment = column_enrichment.sort_values(ascending=False)
        # Find the maximum enrichment for this column in a cluster
        max_enrichment = 0
        for cluster_label in unique_cluster_labels:
            cluster_enrichment = column_enrichment[label_rows_dict[cluster_label]].mean()
            if cluster_enrichment > max_enrichment:
                max_enrichment = cluster_enrichment
                max_enrichment_cluster_label = cluster_label
        column_cluster_labels.append(max_enrichment_cluster_label)

    # Concate column names with cluster labels in a df and sort by cluster labels
    column_cluster_labels_df = pd.DataFrame({'column': data_reordered_columns, 'cluster_label': column_cluster_labels})
    column_cluster_labels_df = column_cluster_labels_df.sort_values(by=['cluster_label'])

    # Reorder the columns
    data_reordered = data_reordered[column_cluster_labels_df['column']]

    # Convert cluster labels to colors with a colormap
    import matplotlib.cm as cm
    cmap = cm.get_cmap('tab20')
    colors = cmap(cluster_labels_reordered)

    import seaborn as sns

    from scipy.cluster import hierarchy
    # Calculate hierarchical clustering on the cluster_id axis (rows)
    col_linkage = hierarchy.linkage(data_normalized.T, method='average')

    # Create a clustermap
    figure(figsize=(200, 150), font_size = 5)
    plt.rcParams.update({'font.size': 5})
    '''
    # Set default font size to be 2
    sns.set(font_scale=0.3)
    g = sns.clustermap(data_reordered, col_cluster = False, row_cluster=False, cmap="Reds",
                    row_colors = colors, 
                    linewidths=0, figsize=(18, 50), cbar_pos=None)
    '''
    pdb.set_trace()
    # Read clinical data
    if clinical_path is not None:
        clinical_df = pd.read_csv(clinical_path, index_col = 'Key')
    else:
        print('No clinical data provided: generate a clinical data table and rerun')
        return

    # Map cluster labels to clinical data
    clinical_df = clinical_df.join(cluster_labels_reordered, how='left')

    # Sort clinical data by cluster labels
    clinical_df = clinical_df.sort_values(by=['cluster_id'])

    clinical_df['Group'] = clinical_df['cluster_id'].astype(str)
    clinical_df.drop(columns=['cluster_id'], inplace=True)

    #this shortens the column names and only keeps the part of the name before the parenthesis
    clinical_var_renamed = [col.split('(')[0].strip() for col in clinical_df.columns]
    
    clinical_df.columns = clinical_var_renamed

    def convert_to_float_if_int(val):
        try:
            return float(val) if isinstance(val, int) else val
        except ValueError:
            return val

    

    #first replace the columns that don't just have Yes and No with their string values - stored in the config file 
    for key in clinical_var_labels.keys():
        unique_values = set(clinical_df[key].unique())
        for index in range(len(clinical_var_labels[key])):
            val = list(clinical_var_labels[key].keys())
            val = [convert_to_float_if_int(v) for v in clinical_var_labels[key].keys()]
            value_new = list(clinical_var_labels[key].values())
            clinical_df[key] = clinical_df[key].replace(val[index],value_new[index])

    # if the column only has 0s and 1s, replace them with with No and Yes
    for col in clinical_df.columns:
        unique_values = set(clinical_df[col].unique())
        if unique_values == {0, 1} or unique_values == {0} or unique_values == {1}:
            clinical_df[col].replace({0: 'No', 1: 'Yes'}, inplace=True)
 
    # Import Linear segmented colormap
    import matplotlib.colors as mcolors

    red_color = np.clip(np.array([219, 68, 55, 255]) / 256 + 0.3, 0, 1)
    blue_color = np.clip(np.array([66, 133, 244, 255]) / 256 + 0.3, 0, 1)


    # Determine the number of colors needed based on the number of columns with pairs
    num_clinical_var_pairs =  sum(clinical_df.nunique() == 2)

    tab20_colors = generate_distinct_colors(num_clinical_var_pairs * 2) #multiply the number of pairs by 2 since that is the total number of colors we need

   # Define your colormap list; ensure it matches the number of clinical variables that have pairs
    pair_cmaps = [extract_pair_colormap(i, tab20_colors) for i in range(num_clinical_var_pairs)]

    rb_cmap = mcolors.LinearSegmentedColormap.from_list('rb_cmap', [red_color, blue_color], N=2)
    br_cmap = mcolors.LinearSegmentedColormap.from_list('br_cmap', [blue_color, red_color], N=2)

    # Create 17 color colormap from spectral
    spectral_cmap = plt.cm.get_cmap('tab20', n_clusters)

    from PyComplexHeatmap import HeatmapAnnotation, ClusterMapPlotter, anno_label, anno_simple

    anno_h = 1.5
    #row_ha = HeatmapAnnotation(df = clinical_df, axis = 0)
    anno_legend_dict = {'fontsize' : 5, 'color_text' : False, 'frameon' : False}
    #pdb.set_trace()

    # Initialize a dictionary to store annotations
    annotations = {}

    # Iterate over columns and assign colormaps
    for idx, col in enumerate(clinical_df.columns):
        unique_values = set(clinical_df[col].unique())
        print(idx)
        if idx < len(pair_cmaps) and len(unique_values) <= 2:
            # Create annotation
            annotations[col] = anno_simple(clinical_df[col], legend_kws=anno_legend_dict, cmap=pair_cmaps[idx], height=anno_h)
        if len(unique_values) > 2: #if there's more than 2 options, then use spectral_cmap rather than pair_cmaps
            annotations[col] = anno_simple(clinical_df[col], legend_kws=anno_legend_dict, cmap=spectral_cmap, height=anno_h)

    # Add a default annotation for the axis
    annotations['axis'] = 1

    # Create HeatmapAnnotation with all annotations
    row_ha = HeatmapAnnotation(**annotations)

    clinical_df['group_int'] = clinical_df['Group'].astype(int)
    clinical_df['split_14'] = (clinical_df['group_int'] < 14) * 1
    clinical_df['split_14'] += (clinical_df['group_int'] == 14) * 2

    #cm = ClusterMapPlotter(data_reordered, left_annotation = row_ha, col_cluster = False, row_cluster=False, show_rownames = True, show_colnames = True, cmap = 'Reds')
    #cm = ClusterMapPlotter(data_reordered.T, col_split = clinical_df['split_14'], top_annotation = row_ha, col_cluster = False, row_cluster=False, show_rownames = True, show_colnames = False, cmap = 'Reds')
    cm = ClusterMapPlotter(data_reordered.T, top_annotation = row_ha, col_cluster = False, row_cluster=False, show_rownames = False, show_colnames = False, cmap = 'Reds')

    # Adjust the postion of the main colorbar for the heatmap
    #g.cax.set_position([.97, .2, .03, .45])

    # Save the figure
    plt.savefig(f'{save_path}/clustermap.png', dpi=600, bbox_inches='tight')

from matplotlib.colors import LinearSegmentedColormap


def generate_distinct_colors(num_colors):
    colors = []
    cmap1 = plt.get_cmap('Paired')
    cmap2 = plt.get_cmap('tab20c')

    # Calculate the number of colors needed from each colormap
    num_colors_cmap1 = min(num_colors // 2, cmap1.N)  # Avoid exceeding the available colors in cmap1
    num_colors_cmap2 = num_colors - num_colors_cmap1  # Remaining colors from cmap2

    # Generate colors from cmap1
    colors_cmap1 = cmap1(np.linspace(0, 1, num_colors_cmap1))

    # Generate colors from cmap2
    colors_cmap2 = cmap2(np.linspace(0, 1, num_colors_cmap2))

     # Append colors from cmap2 to the end of colors from cmap1
    colors = np.vstack([colors_cmap1, colors_cmap2])

    return np.array(colors)


def extract_pair_colormap(pair_index, tab20_colors):
    start_index = 2 * pair_index
    end_index = start_index + 1
    color1 = tab20_colors[start_index]
    color2 = tab20_colors[end_index]
    #pdb.set_trace()
    return LinearSegmentedColormap.from_list('custom', [color1, color2], N=2)

