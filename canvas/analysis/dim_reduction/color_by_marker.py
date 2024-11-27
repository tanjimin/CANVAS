import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import sys
#sys.path.append('/gpfs/data/tsirigoslab/home/jt3545/plt-figure')
#from python-core import subplots

def main():
    umap_emb_path = sys.argv[1]
    marker_path = sys.argv[2]
    plot_path = sys.argv[3]
    #plot_umap_markers_separate(umap_emb_path, marker_path, plot_path)
    plot(umap_emb_path, marker_path, plot_path)

def plot(umap_path, marker_path, plot_path, channel_names = None, cols = 5):
    umap_embedding = np.load(umap_path)
    markers = np.load(marker_path)
    
    rows = len(channel_names) // cols + 1
    #fig, axes = subplots(rows, cols, figsize=(cols * 50, rows * 40), font_size = 5)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 15, rows * 14))

    for marker_i in range(len(channel_names)):
        marker_color = markers[:, marker_i]
        
        m_min, m_max = np.percentile(marker_color, [5, 95])
        marker_color = np.clip(marker_color, m_min, m_max)
        marker_color = (marker_color - m_min) / (m_max - m_min + 1e-6)

        sc = axes[marker_i // cols, marker_i % cols].scatter(umap_embedding[:, 0], umap_embedding[:, 1], s = 0.3, c = marker_color, cmap = 'viridis')
        axes[marker_i // cols, marker_i % cols].set_title(f'Marker: {channel_names[marker_i]}').set_size(40)
        #axes[marker_i // cols, marker_i % cols].title.set_size(20) #added by MH
        fig.colorbar(sc, ax=axes[marker_i // cols, marker_i % cols])

        # Dark background
        #axes[marker_i // cols, marker_i % cols].set_facecolor('black')

    # Clean up empty axes
    for i in range(len(channel_names), rows * cols):
        axes[i // cols, i % cols].axis('off')

    for ax in axes.flat:
        ax.label_outer()
    plt.savefig(plot_path, bbox_inches='tight')
    plt.clf()


def plot_umap_markers_separate(umap_path, marker_path, plot_path):
    umap_embedding = np.load(umap_path)
    markers = np.load(marker_path)

    #channel_names = ["CD117", "CD11c", "CD14", "CD163", "CD16", "CD20", "CD31", "CD3", "CD4", "CD68", "CD8a", "CD94", "DNA1", "FoxP3", "HLA-DR", "MPO", "Pancytokeratin", "TTF1"]
    channel_names = ["MPO","PD-1","PARP1","E-cadherin","Ki67","ER","PR","pJAK1","CD4","CD56","MSH6","CD44","CD47","CD8","Her2","HLA-ABC","MSH2","MAL","PD-L1","ARID1A","CD163","DAPI","PI3KCA","STAT1","CD3e","PMS","Pan-Cytokeratin","CD68","GAL3","MLH1","IFNG","b-Catenin1","CD20","CD31","LAG3","TIM3"]

    for marker_i in range(len(channel_names)):
        marker_color = markers[:, marker_i]
        
        m_min, m_max = np.percentile(marker_color, [5, 95])
        marker_color = np.clip((marker_color - m_min) / m_max, 0, 1)

        fig, ax = plt.subplots(figsize=(20, 17), font_size = 5)


        if channel_names[marker_i] in ['CD4', 'CD8', 'CD3e', 'CD56']:
            cmap = 'Reds'
        elif channel_names[marker_i] in ['CD68', 'CD163']:
            cmap = 'Oranges'
        elif channel_names[marker_i] in ['Pan-Cytokeratin', 'E-cadherin']:
            cmap = 'Purples'
        elif channel_names[marker_i] == 'CD20':
            cmap = 'Greens'
        elif channel_names[marker_i] in ['MPO']:
            cmap = 'Blues'
        else:
            cmap = 'Greys'


        # if channel_names[marker_i] in ['CD3', 'CD4', 'CD8a', 'FoxP3']:
        #     cmap = 'Reds'
        # elif channel_names[marker_i] in ['CD68', 'CD163']:
        #     cmap = 'Oranges'
        # elif channel_names[marker_i] in ['Pancytokeratin', 'TTF1']:
        #     cmap = 'Purples'
        # elif channel_names[marker_i] == 'CD20':
        #     cmap = 'Greens'
        # elif channel_names[marker_i] in ['MPO', 'CD14', 'CD16']:
        #     cmap = 'Blues'
        # else:
        #     cmap = 'Greys'
        
        sc = ax.scatter(umap_embedding[:, 0], umap_embedding[:, 1], s = 0.1, c = marker_color, cmap = cmap, edgecolors = 'none')
        ax.set_title(f'{channel_names[marker_i]}')
        # Remove all frame
        ax.axis('off')

        fig.colorbar(sc, ax=ax)

        plot_path_marker = plot_path.replace('.pdf', f'_{channel_names[marker_i]}.pdf')
        plot_path_marker = plot_path.replace('.png', f'_{channel_names[marker_i]}.png')

        plt.savefig(plot_path_marker, bbox_inches='tight', dpi=900)
        plt.clf()


if __name__ == '__main__':
    main()

