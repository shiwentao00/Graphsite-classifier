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
from sklearn.decomposition import PCA


def get_args():
    parser = argparse.ArgumentParser('python')

    parser.add_argument('-embedding_dir',
                        default='../../embeddings/',
                        required=False,
                        help='text file to get the cluster labels')

    parser.add_argument('-run',
                        required=True,
                        type=int,
                        help='which experiment.')

    parser.add_argument('-which_algorithm',
                        default='umap',
                        help='which algorithm to use for manifold learning.')

    return parser.parse_args()


def visualize_embeddings(embeddings, labels, cluster_ids, image_path, image_path_tif, colors, which_algorithm):
    """Visualize 2d embeddings and color them by labels.
    """
    font = {'size': 8}
    matplotlib.rc('font', **font)

    # list of the clusters/classes
    cluster_set = list(set(labels))
    cluster_set.sort()

    embedding_list = []
    label_list = []
    for cluster in cluster_set:
        idx = np.nonzero(labels == cluster)[0]
        embedding_list.append(embeddings[idx, :])
        label_list.append(labels[idx])

    embedding_list = np.vstack(embedding_list)
    label_list = list(np.hstack(label_list))
    cluster_id_list = [cluster_ids[x] for x in label_list]

    fig = plt.figure(figsize=(5.5, 5), dpi=300)

    #cust_palette = sns.color_palette("RdBu_r", len(list(set(cluster_id_list))))
    cust_palette = sns.xkcd_palette(colors)
    #cust_palette = sns.color_palette("Spectral", len(list(set(cluster_id_list))))
    ax = sns.scatterplot(x=embedding_list[:, 0],
                         y=embedding_list[:, 1],
                         hue=cluster_id_list,
                         alpha=0.8,
                         markers='.',
                         s=15,
                         palette=cust_palette
                         )
    plt.legend(bbox_to_anchor=(0.55, 1), loc=2,
               borderaxespad=0., frameon=False)

    if which_algorithm == 'tsne':
        plt.xlabel('tSNE_1')
        plt.ylabel('tSNE_2')

    plt.tight_layout()
    plt.savefig(image_path)
    plt.savefig(image_path_tif)


if __name__ == "__main__":
    args = get_args()
    embedding_root = args.embedding_dir
    run = args.run
    which_algorithm = args.which_algorithm
    embedding_dir = embedding_root + 'run_{}/'.format(run)

    merge_info = [0, 2, 3, 4, 6, 7, 8, 9]
    print('how to merge clusters: ', merge_info)
    cluster_name_dict = {0: '0: ADP and ANP',
                         2: '2: heme',
                         3: '3: glucopyranose and fructose',
                         4: '4: benzene ring',
                         6: '6: chlorophyll',
                         7: '7: lipid',
                         8: '8: glucopyranose',
                         9: '9: UMP and TMP'}
    cluster_ids = [cluster_name_dict[x]
                   for x in merge_info]  # use original cluster ids

    # 8 colors
    colors = ["green", "midnight blue", "red", "windows blue",
              "grey", "orchid", "amber", "bright orange"]

    # 13 colors
    # colors = ["faded green", "dusty purple", "red", "cyan", "yellow green", "midnight blue",
    #          "neon green", "bright pink", "crimson", "bright orange", "windows blue", "amber",
    #          "greyish"]

    # 18 colors
    # colors = ["faded green", "dusty purple", "red", "cyan", "yellow green", "midnight blue",
    #          "neon green", "bright pink", "crimson", "bright orange", "windows blue", "amber", "greyish",
    #          "yellow", "tomato", "navy", "turquoise", "azure"]

    for which_split in ['test']:
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

        print('denoising with PCA...')
        pca_model = PCA(n_components=96)
        embeddings = pca_model.fit_transform(embeddings)
        print('shape after PCA: ', embeddings.shape)
        print('total explained variance ratio:', np.sum(
            pca_model.explained_variance_ratio_))

        image_path = './results_siamese/run_{}/{}_{}_{}.png'.format(
            run, which_algorithm, which_split, run)
        image_path_tif = './results_siamese/run_{}/{}_{}_{}.tif'.format(
            run, which_algorithm, which_split, run)
        print('image saved to: ', image_path)

        if which_algorithm == 'tsne':
            print('computing TSNE...')
            tsne = TSNE(n_components=2, perplexity=40,
                        learning_rate=200, n_iter=1500, random_state=8)
            vis_embedding = tsne.fit_transform(embeddings)
            print('KL divergence after optimizaton: ', tsne.kl_divergence_)

        elif which_algorithm == 'umap':
            print('computing UMAP...')
            umap_inst = umap.UMAP(
                n_components=2, n_neighbors=200, min_dist=1, metric='euclidean')
            vis_embedding = umap_inst.fit_transform(embeddings)

        visualize_embeddings(vis_embedding, labels, cluster_ids,
                             image_path, image_path_tif, colors, which_algorithm)
