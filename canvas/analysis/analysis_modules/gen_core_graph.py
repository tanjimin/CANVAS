import sys
import os
import numpy as np
import pandas as pd
import skimage.io as io
import matplotlib
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d, ConvexHull
from shapely.geometry import Polygon, Point
import pickle
import networkx as nx
from libpysal import weights
from libpysal.cg import voronoi_frames
import pdb
import zarr
from tqdm import tqdm

def gen_core_graph(kmeans_path, tile_embedding_path,analysis_graph_dir, WSI_subset_regions, 
                   subset_region_w, subset_region_h,local_region, tile_size):

    print('Loading data...')

    cluster_path = os.path.join(kmeans_path, 'clusters.npy') 
    sample_name_path = os.path.join(tile_embedding_path, 'sample_name.npy')
    position_path = os.path.join(tile_embedding_path, 'tile_location.npy')

    # Load sample and cluster information
    clusters = np.load(cluster_path, mmap_mode='r')
    sample_names = np.load(sample_name_path, mmap_mode='r')
    positions = np.load(position_path, mmap_mode='r')

    for selected_sample in np.unique(sample_names):
        print(selected_sample)
        # Get cluster index selected sample
        selected_sample_index = np.where(sample_names == selected_sample)[0]

        # Get cluster and position of selected sample
        selected_sample_cluster = clusters[selected_sample_index]
        selected_sample_position = positions[selected_sample_index]
        selected_center = selected_sample_position + tile_size // 2

        # Get triangulation graph
        graph = gen_triangulation(selected_sample_position, selected_sample_cluster, fill_edges=True)

        output_path_sample = os.path.join(analysis_graph_dir, selected_sample)
        os.makedirs(output_path_sample, exist_ok=True)

        output_pkl_path_sample = os.path.join(output_path_sample, 'core_graph.pkl')

        # Save graph as pickle
        with open(output_pkl_path_sample, 'wb') as f:
            pickle.dump(graph, f)

        plot_color_path = os.path.join(kmeans_path, 'cluster_plot_color_rgb.npy')
        plot_color = np.load(plot_color_path, mmap_mode='r')
        color_dict = dict(zip(range(len(plot_color)), plot_color))
                
        output_graph_path_sample = os.path.join(output_path_sample, 'core_graph.png')

        print('Plotting graph...')
        fig, ax = plt.subplots(figsize=(20, 20))
        plot_graph(graph, color_dict, ax, edge_color='black')
        plt.savefig(output_graph_path_sample, bbox_inches='tight')
        plt.clf()

        if local_region:
            WSI_subset_regions = pd.read_csv(WSI_subset_regions)
            WSI_subset_regions_samplewise = WSI_subset_regions[WSI_subset_regions['Sample'] == selected_sample].reset_index(drop=True)
            
            for index, row in WSI_subset_regions_samplewise.iterrows():
                h1_value = row['h1']
                w1_value = row['w1']

                selected_sample_position_df = pd.DataFrame(selected_sample_position, columns=['h1', 'w1'])
                positions_keep = ((selected_sample_position_df['h1'] >= h1_value) & (selected_sample_position_df['h1'] <= (h1_value + subset_region_h))) & \
                                 ((selected_sample_position_df['w1'] >= w1_value) & (selected_sample_position_df['w1'] <= (w1_value + subset_region_w)))
                selected_sample_position_df_filtered = selected_sample_position_df[positions_keep]
                indices_to_keep = selected_sample_position_df_filtered.index

                selected_sample_cluster = selected_sample_cluster[indices_to_keep]
                selected_center = selected_center[indices_to_keep]
                selected_sample_position = selected_sample_position[indices_to_keep]

                graph = gen_triangulation(selected_sample_position, selected_sample_cluster, fill_edges=True)

                if graph is None:
                    print(f"Graph for sample {selected_sample} is NoneType, moving to next sample.")
                    continue

                coords = f'h{h1_value}_w{w1_value}'
                output_path_sample_coord = os.path.join(output_path_sample, coords)
                os.makedirs(output_path_sample_coord, exist_ok=True)
                
                output_pkl_dir_sample_coord = os.path.join(output_path_sample_coord, 'core_graph.pkl')

                with open(output_pkl_dir_sample_coord, 'wb') as f:
                    pickle.dump(graph, f)
                
                output_graph_path_sample_coord = os.path.join(output_path_sample_coord, 'core_graph.png')

                print('Plotting graph...')
                fig, ax = plt.subplots(figsize=(5, 5))
                plot_graph(graph, color_dict, ax, edge_color='black')
                plt.savefig(output_graph_path_sample_coord, bbox_inches='tight')
                plt.clf()

                output_graph_reduced_path_sample_coord = os.path.join(output_path_sample_coord, 'core_graph_reduced.png')

                print('Contracting graph...')
                contracted_graph = same_edge_contraction(graph)
                contracted_graph = update_coords(contracted_graph)
                add_weight_attribute_to_edge(contracted_graph, 'counts')
                contracted_graph.remove_edges_from(nx.selfloop_edges(contracted_graph))
                
                fig, ax = plt.subplots(figsize=(5, 5))
                plot_graph(contracted_graph, color_dict, ax, edge_color='black')
                plt.savefig(output_graph_reduced_path_sample_coord)
                plt.clf()

                output_graph_snap_reduced_free_path_sample_coord = os.path.join(output_path_sample_coord, 'core_graph_free_snap_reduced.png')
                output_graph_snap_reduced_free_pkl_sample_coord = os.path.join(output_path_sample_coord, 'core_graph_snap.pkl')

                print('Contracting graph with snap reduced free...')
                snap_graph = nx.snap_aggregation(contracted_graph, ('type',))
                update_supernode(snap_graph, contracted_graph)
                add_weight_attribute_to_edge(snap_graph, 'counts')
                
                fig, ax = plt.subplots(figsize=(5, 5))
                plot_graph(snap_graph, color_dict, ax, edge_color='black')
                plt.savefig(output_graph_snap_reduced_free_path_sample_coord)
                plt.clf()

                with open(output_graph_snap_reduced_free_pkl_sample_coord, 'wb') as f:
                    pickle.dump(snap_graph, f)
        else:
            print('Contracting graph...')
            contracted_graph = same_edge_contraction(graph)
            contracted_graph = update_coords(contracted_graph)
            add_weight_attribute_to_edge(contracted_graph, 'counts')
            contracted_graph.remove_edges_from(nx.selfloop_edges(contracted_graph))
            
            fig, ax = plt.subplots(figsize=(20, 20))
            output_graph_reduced_path_sample = os.path.join(output_path_sample, 'core_graph_reduced.png')
            plot_graph(contracted_graph, color_dict, ax, edge_color='black')
            plt.savefig(output_graph_reduced_path_sample)
            plt.clf()

            output_graph_snap_reduced_free_path_sample = os.path.join(output_path_sample, 'core_graph_free_snap_reduced.png')
            output_graph_snap_reduced_free_pkl_sample = os.path.join(output_path_sample, 'core_graph_snap.pkl')

            print('Contracting graph with snap reduced free...')
            snap_graph = nx.snap_aggregation(contracted_graph, ('type',))
            update_supernode(snap_graph, contracted_graph)
            add_weight_attribute_to_edge(snap_graph, 'counts')
            
            fig, ax = plt.subplots(figsize=(5, 5))
            plot_graph(snap_graph, color_dict, ax, edge_color='black')
            plt.savefig(output_graph_snap_reduced_free_path_sample)
            plt.clf()

            with open(output_graph_snap_reduced_free_pkl_sample, 'wb') as f:
                pickle.dump(snap_graph, f)

def gen_triangulation(coordinates, types, method_name = 'gabriel', fill_edges = False):

    # Check if enough coordinates
    if len(coordinates) < 4:
        print('unable to do triangulation because there are less than 4 coordinates')
        return None
    # convert npy coordinates to drawing coordinates
    coordinates = convert_positions(coordinates)
    # Triangulation
    if method_name == 'gabriel':
        graph = weights.Gabriel(coordinates).to_networkx()
    elif method_name == 'delaunay':
        graph = weights.Delaunay(coordinates).to_networkx()
    else:
        raise ValueError(f'Invalid method name {method_name}')

    if fill_edges:
        # Fill edges with equal distance
        print('Filling edges')
        graph = fill_adjacent_edges(graph, coordinates)
    
    # Give attributes to nodes
    types_dict = dict(zip(range(len(types)), types))
    coords_dict = dict(zip(range(len(coordinates)), coordinates))
    nx.set_node_attributes(graph, types_dict, 'type')
    nx.set_node_attributes(graph, coords_dict, 'coords')
    # Add edge attributes
    add_adjacency_attribute_to_edge(graph, 'type')
    return graph

def fill_adjacent_edges(graph, coordinates, grid_step = 64):
    for node_idx in range(len(coordinates)):
        distances = np.linalg.norm(coordinates - coordinates[node_idx], axis = 1)
        neighbors = np.logical_and(distances > 0.1, distances <= np.sqrt(2) * grid_step + 0.1)
        for neighbor_idx in np.where(neighbors)[0]:
            graph.add_edge(node_idx, neighbor_idx)
    return graph

def convert_positions(positions):
    graph_positions = positions.copy()
    # Swap x and y, and flip y
    graph_positions[:, 0] = positions[:, 1]
    graph_positions[:, 1] = positions[:, 0]
    max_y = np.max(graph_positions[:, 0])
    graph_positions[:, 1] = max_y - graph_positions[:, 1]
    return graph_positions

def add_adjacency_attribute_to_edge(graph, attribute_name):
    for edge in graph.edges():
        node1_idx = edge[0]
        node2_idx = edge[1]
        node1 = graph.nodes[node1_idx]
        node2 = graph.nodes[node2_idx]
        type1 = node1[attribute_name]
        type2 = node2[attribute_name]
        #graph.edges[(node1_idx, node2_idx)]['type'] = f'{type1}-{type2}'
        graph.edges[(node1_idx, node2_idx)]['type'] = 'default'
        graph.edges[(node1_idx, node2_idx)].pop('weight', None)

def plot_graph(graph, color_dict, ax, edge_color = 'white', structure = None):

    #default values
    sizes = 50
    pos = None
    edge_width = 1

    print('total number of nodes: ', len(graph.nodes))

    if graph is not None and list(graph.nodes): #added this to prevent this error: 'NoneType' object has no attribute 'nodes''
        if 'coords' in graph.nodes[list(graph.nodes)[0]]:
            pos = dict(zip(list(graph.nodes()), [graph.nodes[i]['coords'] for i in graph.nodes]))
        else:
            pos = None
    else:
        return

    np.random.seed(0)

    if structure is not None and list(graph.nodes):
        print("structure is not None")
        if len(graph.nodes) > 2000:
            pos = nx.spring_layout(graph, pos = pos, weight = 'attraction', iterations= 100, seed=1234567) #increase the number of iterations for the nodes to be more spread out
        else:
            pos = nx.spring_layout(graph, pos = pos, weight = 'attraction', iterations= 100, seed=1234567)

    if graph is not None and list(graph.edges):
        if 'weight' in graph.edges[list(graph.edges)[0]]:
            edge_width = [graph.edges[i]['weight'] ** 0.5 for i in graph.edges]
        else:
            edge_width = 1

    if graph is not None and list(graph.nodes):
        if 'counts' in graph.nodes[list(graph.nodes)[0]]:
            if len(graph.nodes) > 2000:
                sizes = np.array([graph.nodes[i]['counts'] ** 0.5 for i in graph.nodes]) * 2 #multiply by 2 rather than 30 to decrease node size
                sizes = np.clip(sizes, 0, 2) 
            if len(graph.nodes) < 50:
                sizes = np.array([graph.nodes[i]['counts'] ** 0.5 for i in graph.nodes]) * 50
                sizes = np.clip(sizes, 0, 300) 
            else:
                sizes = np.array([graph.nodes[i]['counts'] ** 0.5 for i in graph.nodes]) * 20
                sizes = np.clip(sizes, 0, 100) 

        else: #if counts is not in graph.nodes
            if len(graph.nodes) > 2000: #if there are a very large number of nodes, decrease the size
                sizes = np.array([2 for _ in graph.nodes])
            if len(graph.nodes) < 50: #if there are a very small number of nodes, increase the size
                sizes = np.array([500 for _ in graph.nodes])
            else:
                sizes = np.array([200 for _ in graph.nodes])

        print("node size:", np.unique(sizes))

    nx.draw(graph,
            pos,
            ax = ax,
            node_size = sizes,
            node_color = [color_dict[graph.nodes[i]['type']] for i in graph.nodes],
            edge_color = edge_color,
            #edge_color="k",
            width = edge_width,
            alpha = 1,
        )

# Contraction algorithm
def same_edge_contraction(graph):
    ''' This contracts edge if two nodes are adjacent and have the same type '''

    new_graph = graph.copy()
    attr_name = 'type'
    something_to_merge = True
    while something_to_merge:
        count = 0
        merged_nodes = set()
        for node in tqdm(new_graph.nodes()):
            if node in merged_nodes:
                continue
            node_attr = new_graph.nodes[node][attr_name]
            # Look at adjacent nodes
            for neighbor in new_graph[node]:
                if neighbor == node:
                    continue
                neighbor_attr = new_graph.nodes[neighbor][attr_name]
                if node_attr == neighbor_attr:
                    # Check if coords is a two dimensional array, if not, add dimension at 0
                    if len(new_graph.nodes[node]['coords']) == 1:
                        new_graph.nodes[node]['coords'] = np.array([new_graph.nodes[node]['coords']])
                    if len(new_graph.nodes[neighbor]['coords']) == 1:
                        new_graph.nodes[neighbor]['coords'] = np.array([new_graph.nodes[neighbor]['coords']])

                    # Check if counts attribute exists, if not, set to 1
                    if 'counts' not in new_graph.nodes[node]:
                        new_graph.nodes[node]['counts'] = 1
                    if 'counts' not in new_graph.nodes[neighbor]:
                        new_graph.nodes[neighbor]['counts'] = 1
                    # Add counts of merged nodes
                    new_graph.nodes[node]['counts'] += new_graph.nodes[neighbor]['counts']
                    # Concatenate coordinates
                    new_graph.nodes[node]['coords'] = np.concatenate((new_graph.nodes[node]['coords'], new_graph.nodes[neighbor]['coords']), axis = 0)
                    # Merge nodes
                    new_graph = nx.contracted_edge(new_graph, (node, neighbor), self_loops=True)
                    merged_nodes.add(neighbor)
                    count += 1
        if count == 0:
            something_to_merge = False
    return new_graph

def update_coords(graph):
    new_graph = graph.copy()
    for node in new_graph.nodes():
        current_node = new_graph.nodes[node]
        coords_stats = current_node['coords'].reshape(-1, 2)
        mean_coord = np.mean(coords_stats, axis=0)
        new_graph.nodes[node]['coords'] = list(mean_coord)
        if not 'counts' in current_node:
            new_graph.nodes[node]['counts'] = 1
    return new_graph

def get_list_of_contracted_nodes(node_id, node):
    contracted_list = [node_id]
    while 'contraction' in node:
        contracted_dict = node['contraction']
        for key, value in contracted_dict.items():
            contracted_list.append(key)
            node = value
    return contracted_list

def add_weight_attribute_to_edge(graph, attribute_name):
    for edge in graph.edges():
        node1_idx = edge[0]
        node2_idx = edge[1]
        node1 = graph.nodes[node1_idx]
        node2 = graph.nodes[node2_idx]
        type1 = node1[attribute_name]
        type2 = node2[attribute_name]
        if node1 != node2:
            graph.edges[(node1_idx, node2_idx)]['weight'] = min(type1, type2)
            graph.edges[(node1_idx, node2_idx)]['attraction'] = (type1 + type2) * 0.5
            #graph.edges[(node1_idx, node2_idx)]['attraction'] = 1 / (min(type1, type2)) * 0.1
        else:
            graph.edges[(node1_idx, node2_idx)]['weight'] = 1
            graph.edges[(node1_idx, node2_idx)]['attraction'] = 1

def update_supernode(snap_graph, original_graph):
    for super_node in snap_graph.nodes():
        node_group = snap_graph.nodes[super_node]['group']
        type_list = []
        coord_list = []
        count_list = []
        for node in node_group:
            type_list.append(original_graph.nodes[node]['type'])
            coord_list.append(original_graph.nodes[node]['coords'])
            count_list.append(original_graph.nodes[node]['counts'])
        coord_list = np.array(coord_list)
        if len(set(type_list)) != 1:
            raise ValueError('Super node has different types')
        snap_graph.nodes[super_node]['type'] = type_list[0]
        #snap_graph.nodes[super_node]['coords'] = list(np.mean(coord_list, axis=0))
        snap_graph.nodes[super_node]['counts'] = sum(count_list)

