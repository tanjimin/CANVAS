import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import networkx as nx
import sys
import pdb
import pandas as pd

def gen_core_stats(analysis_graph_dir, tile_embedding_path, WSI_subset_regions, local_region = False):
    from canvas.visualization.core import figure

    sample_name_path = os.path.join(tile_embedding_path, 'sample_name.npy')
    sample_names = np.load(sample_name_path, mmap_mode = 'r')
    
    for sample in np.unique(sample_names):
        print(sample)
        output_path_sample = os.path.join(analysis_graph_dir, sample)
        
        if local_region:
            WSI_subset_regions = pd.read_csv(WSI_subset_regions)
            WSI_subset_regions_samplewise = (WSI_subset_regions[WSI_subset_regions['Sample'] == sample]).reset_index(drop=True)
            for index,row in WSI_subset_regions_samplewise.iterrows():
                #pdb.set_trace()
                h1_value = WSI_subset_regions_samplewise['h1'].iloc[index]
                w1_value = WSI_subset_regions_samplewise['w1'].iloc[index]  
                coords = 'h' + str(h1_value) + '_' + 'w' + str(w1_value)

                output_path_sample_coords = os.path.join(output_path_sample, coords)

                #loop through the specific positions!! 
                core_graph = os.path.join(output_path_sample_coords, "core_graph_snap.pkl")

                if not os.path.exists(core_graph):
                    print(f"Core graph file {core_graph} does not exist. Moving to next sample.")
                    continue  # Skip to the next sample

                with open(core_graph, 'rb') as f:
                    core_graph = pickle.load(f)
                stats_path = os.path.join(output_path_sample_coords, "stats")
                os.makedirs(stats_path, exist_ok = True)
                inbetweeness_graph = os.path.join(stats_path, 'inbetweeness.png')
                inbetweeness_dist = os.path.join(stats_path, 'inbetweeness_array.csv')

                #if there are no edges or no nodes in the core_graph, then move on to the next sample!
                if len(core_graph.edges) == 0 or len(core_graph.nodes) == 0:
                    print(f"{sample}'s graph does not have any nodes or edges... skipping plotting and moving on to next sample")
                    continue

                plot_centrality(core_graph, nx.betweenness_centrality(core_graph), inbetweeness_graph, inbetweeness_dist)
        else:
            core_graph = os.path.join(output_path_sample, "core_graph_snap.pkl")

            if not os.path.exists(core_graph):
                print(f"Core graph file {core_graph} does not exist. Moving to next sample.")
                continue  # Skip to the next sample

            with open(core_graph, 'rb') as f:
                core_graph = pickle.load(f)
            stats_path = os.path.join(output_path_sample, "stats")
            os.makedirs(stats_path, exist_ok = True)
            inbetweeness_graph = os.path.join(stats_path, 'inbetweeness.png')
            inbetweeness_dist = os.path.join(stats_path, 'inbetweeness_array.csv')
            
            #if there are no edges or no nodes in the core_graph, then move on to the next sample!
            if len(core_graph.edges) == 0 or len(core_graph.nodes) == 0:
                print(f"{sample}'s graph does not have any nodes or edges... skipping plotting and moving on to next sample")
                continue

            plot_centrality(core_graph, nx.betweenness_centrality(core_graph), inbetweeness_graph, inbetweeness_dist)

def plot_centrality(graph, centrality, save_path, dist_path):
    # Generate color by degree centrality
    color_dict = {}
    for node in graph.nodes():
        color_dict[node] = centrality[node]

    fig, ax = plt.subplots(figsize=(5, 5))
    plot_graph(graph, color_dict, ax, edge_color = 'black')
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.clf()

    # Plot distribution of centrality
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.hist(list(centrality.values()), bins = 100)
    # Annotate standard deviation
    ax.annotate('std: {:.2f}'.format(np.std(list(centrality.values()))), xy=(0.05, 0.95), xycoords='axes fraction')
    plt.savefig(save_path.replace('.png', '_hist.png'), dpi=600, bbox_inches='tight')

    # Save distribution
    with open(dist_path, 'w') as f:
        f.write('node,type,tile_counts,centrality\n')
        for node in centrality:
            f.write('{},{},{},{}\n'.format(node, graph.nodes[node]['type'], graph.nodes[node]['counts'], centrality[node]))

def plot_graph(graph, color_dict, ax, edge_color = 'white', structure = None):
    if 'coords' in graph.nodes[list(graph.nodes)[0]]:
        pos = dict(zip(list(graph.nodes()), [graph.nodes[i]['coords'] for i in graph.nodes]))
    else:
        pos = None

    # Seed all
    np.random.seed(0)
    if structure is not None:
        pos = nx.spring_layout(graph, pos = pos, weight = 'attraction', iterations= 100, seed=1234567)
        #pos = nx.multipartite_layout(graph)
    
    #pdb.set_trace()
    if 'weight' in graph.edges[list(graph.edges)[0]]:
        edge_width = [graph.edges[i]['weight'] ** 0.5 for i in graph.edges]
    else:
        edge_width = 1

    if 'counts' in graph.nodes[list(graph.nodes)[0]]:
        sizes = np.array([graph.nodes[i]['counts'] ** 0.5 for i in graph.nodes]) * 30
        size = np.clip(sizes, 0, 500)
    else:
        sizes = 50

    colors = [color_dict[i] for i in graph.nodes()]
    import matplotlib.colors as mcolors
    norm = mcolors.Normalize(vmin=0, vmax=1)
    normalized_colors = [plt.cm.jet(norm(color)) for color in colors]

    nx.draw(graph,
            pos,
            ax = ax,
            node_size = sizes,
            node_color = normalized_colors,
            edge_color = edge_color,
            #edge_color="k",
            width = edge_width,
            alpha = 1,
        )

    '''
    import matplotlib as mpl

    mpl.rcParams['font.size'] = 25
    mpl.rcParams['font.family'] = 'Arial'
    mpl.rcParams['axes.linewidth'] = 2.5
    mpl.rcParams['xtick.major.size'] = 10
    mpl.rcParams['xtick.major.width'] = 2.5
    mpl.rcParams['ytick.major.size'] = 2.5
    mpl.rcParams['ytick.major.width'] = 2.5
    mpl.rcParams['xtick.labelsize'] = 25

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.jet, norm=norm)
    sm.set_array([])
    # Set colorbar to be half the height of the figure
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=25)
    '''
    #plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)

if __name__ == '__main__':
    main()
