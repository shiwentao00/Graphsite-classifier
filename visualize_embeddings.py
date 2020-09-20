"""
Generate embeddings and visualize them in 2-d plane
"""
import random
import argparse
import os
import numpy as np
import yaml
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import umap


def get_args():
    parser = argparse.ArgumentParser('python')

    parser.add_argument('-embedding_dir',
                        default='../embeddings/',
                        required=False,
                        help='text file to get the cluster labels')                        

    parser.add_argument('-run',
                        required=True,
                        type=int,
                        help='which experiment.')     

    return parser.parse_args()


def visualize_embeddings(embeddings, labels, cluster_ids, image_path, colors):
    """Visualize 2d embeddings and color them by labels.
    """
    font = {'size'   : 16}
    matplotlib.rc('font', **font)   

    # list of the clusters/classes
    cluster_set = list(set(labels)) 
    cluster_set.sort()

    embedding_list = []
    label_list = []    
    for cluster in cluster_set:
        idx = np.nonzero(labels == cluster)[0]
        embedding_list.append(embeddings[idx,:])
        label_list.append(labels[idx])

    embedding_list = np.vstack(embedding_list)
    label_list = list(np.hstack(label_list))
    cluster_id_list = [cluster_ids[x] for x in label_list]

    fig = plt.figure(figsize=(14, 12))

    #cust_palette = sns.color_palette("RdBu_r", len(list(set(cluster_id_list))))
    cust_palette = sns.xkcd_palette(colors)
    ax = sns.scatterplot(x=embedding_list[:,0], 
                         y=embedding_list[:,1], 
                         hue=cluster_id_list, 
                         markers='.', 
                         palette= cust_palette
                         )
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.tight_layout()
    plt.savefig(image_path)


if __name__=="__main__":
    args = get_args()
    embedding_root = args.embedding_dir
    run = args.run
    embedding_dir = embedding_root + 'run_{}/'.format(run)

    #merge_info = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    #merge_info = [[0, 9], 1, 2, [3, 8], 4, 5, 6, 7]
    #merge_info = [[0, 9], [1, 5], 2, [3, 8], 4, 6, 7]
    #merge_info = [[0, 9], [1, 5], 2, [3, 8], 4, 6, 7, 10, 11, 12, 13]
    #merge_info = [[0, 9, 12], [1, 5, 11], 2, [3, 8, 13], 4, 6, 7, 10]
    merge_info = [[0, 9, 12], [1, 5, 11], 2, [3, 8, 13], 4, 6, 7, [10, 16], 15, 17, 18]
    print('how to merge clusters: ', merge_info)
    cluster_ids = [str(x) for x in merge_info] # use original cluster ids

    #colors = ["faded green", "dusty purple", "red", "cyan", "yellow green", "midnight blue", "neon green", "bright pink", "crimson", "bright orange", "windows blue", "amber", "greyish"]
    #colors = ["faded green", "dusty purple", "red", "cyan", "yellow green", "midnight blue", "neon green", "bright pink", "crimson", "bright orange"]
    #colors = ["faded green", "dusty purple", "red", "cyan", "yellow green", "midnight blue", "neon green", "bright pink"]
    colors = ["faded green", "dusty purple", "red", "cyan", "yellow green", "midnight blue",
             "neon green", "bright pink", "crimson", "bright orange", "windows blue"]

    for which_split in ['train', 'val', 'test']:
        print('computing evaluation metrics for {}'.format(which_split))
        embedding_name = which_split + '_embedding' + '.npy'
        label_name = which_split + '_label' + '.npy'
        embedding_path = embedding_dir + embedding_name
        label_path = embedding_dir + label_name
        print('embedding path: ', embedding_path)
        print('label path: ', label_path)

        embeddings = np.load(embedding_path)
        labels = np.load(label_path)
        labels = labels.astype(int)
        
        for which_algorithm in ['tsne', 'umap']:
            image_path = './results/run_{}/{}_{}_{}.png'.format(run, which_algorithm, which_split, run)
            print('image saved to: ', image_path)
            
            if which_algorithm == 'tsne':
                print('computing TSNE...')
                vis_embedding = TSNE(n_components=2).fit_transform(embeddings)
            elif which_algorithm == 'umap':
                print('computing UMAP...')
                umap_inst = umap.UMAP()
                vis_embedding = umap_inst.fit_transform(embeddings)

            visualize_embeddings(vis_embedding, labels, cluster_ids, image_path, colors)