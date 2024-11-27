import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pdb

def gen_clinical_corr(kmeans_path, tile_embedding_path, clinical_df_path, save_path):
    
    cluster_path = os.path.join(kmeans_path, 'clusters.npy')
    name_path = os.path.join(tile_embedding_path, 'sample_name.npy')
    count_path = os.path.join(save_path, 'counts_table.csv')

    # import clustring
    labels = np.load(cluster_path)
    samples = np.load(name_path)
    n_cluster = len(np.unique(labels))
    
    tile_df = pd.DataFrame({'cluster' : labels,
                            'sample' : samples})
    if clinical_df_path:
        # Clinical df
        clinical_df = pd.read_csv(clinical_df_path)
    else:
        print('No clinical data provided: generate a clinical data table and rerun')
        return

    # Data Cleaning
    # None

    # Generate counts table
    tile_df['count'] = 1
    counts_df = pd.DataFrame(tile_df.groupby(['cluster', 'sample']).count().reset_index())
    
    # Add missing entries
    all_sample_set = set(clinical_df['Key'])
    all_cluster_set = set(tile_df['cluster'])

    from tqdm import tqdm
    print('Adding 0 entries')
    for sample in tqdm(all_sample_set):
        for cluster in all_cluster_set:
            if not ((counts_df['sample'] == sample) & (counts_df['cluster'] == cluster)).any():
                # Create a new DataFrame with the row to be added
                new_row = pd.DataFrame([{'cluster': cluster, 'sample': sample, 'count': 0}])
                # Concatenate the new row to the existing DataFrame
                counts_df = pd.concat([counts_df, new_row], ignore_index=True)
                #counts_df = counts_df.append({'cluster' : cluster, 'sample' : sample, 'count' : 0}, ignore_index = True)

    counts_df['sample'] = counts_df['sample'].astype(str)
    #total_cluster_counts = pd.DataFrame(tile_df.groupby(['cluster']).count().drop(['id', 'x', 'y'], axis = 1).reset_index())
    #total_sample_counts = pd.DataFrame(tile_df.groupby(['sample']).count().drop(['id', 'x', 'y'], axis = 1).reset_index())
    # Normalize counts by total sample counts
    sample_counts = counts_df.groupby('sample').sum()['count']
    sample_normalized_counts = []
    for index, row in counts_df.iterrows():
        sample_normalized_counts.append(row['count'] / sample_counts[row['sample']])
    counts_df['sample_normalized_count'] = sample_normalized_counts

    # Attach clinical data
    plot_df = counts_df.merge(clinical_df, how = 'left', left_on = 'sample', right_on = 'Key').drop(['Key'], axis = 1)
    plot_df = plot_df[plot_df['sample'] != '0'] # Filter out zero sample group

    # Extract the directory path
    count_dir = os.path.dirname(count_path)
    os.makedirs(count_dir, exist_ok = True)

    # Save counts table
    plot_df.to_csv(count_path, index = False)

    import seaborn as sns
    import matplotlib.pyplot as plt
    #n_clinical = 10
    n_clinical = len(clinical_df.columns) - 1 
   
    fig, axs = plt.subplots(n_clinical, n_cluster, figsize=(n_cluster * 2, n_clinical * 2)) #adjust factor as needed 
    for c_i, c_name in enumerate(clinical_df.columns):
        print(f'Plotting {c_name}')
        if c_name == 'Key': continue
        c_i -= 1
        for cluster in range(n_cluster):

            cluster_df = plot_df[plot_df['cluster'] == cluster]
            #if isinstance(cluster_df[c_name], np.object):
            #    cluster_df[c_name] = cluster_df[c_name].astype(str)

            if c_name == 'Survival or loss to follow-up (years)':
                #sns.scatterplot(ax = axs[c_i, cluster], data = cluster_df, x = c_name, y = 'count')
                sns.scatterplot(ax = axs[c_i, cluster], data = cluster_df, x = c_name, y = 'sample_normalized_count')
            else:
                #sns.stripplot does not handle NaNs. therefore, replace NaNs with "Unkown"
                cluster_df = cluster_df.copy()
                cluster_df.loc[:, c_name] = cluster_df[c_name].fillna('Unknown')
                sns.stripplot(ax = axs[c_i, cluster], data = cluster_df, x = c_name, y = 'sample_normalized_count', color = 'black', alpha = 0.5)
            axs[c_i, cluster].set_ylabel(f'Counts {c_name}')
            #axs[c_i, cluster].scatter(cluster_df['count'], cluster_df[c_name], s= cluster_df['count'])
            #axs[c_i, cluster].set_xlim(0, 200)
            axs[c_i, cluster].set_title(f'cluster {cluster}, n = {len(cluster_df)}')
            #axs[c_i, cluster].get_legend().remove()
            if c_i < len(clinical_df.columns):
                axs[c_i, cluster].set_xlabel('')
            else:
                axs[c_i, cluster].set_xlabel('Tile counts')
    
    clinical_corr_path = os.path.join(save_path, 'clinical_corr.png')
    plt.savefig(clinical_corr_path, bbox_inches = 'tight')
    plt.clf()

if __name__ == '__main__':
    main()

