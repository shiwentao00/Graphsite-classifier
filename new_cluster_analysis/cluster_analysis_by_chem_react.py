"""
Divide the large clusters into subclusters.
"""
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
from itertools import combinations, product


if __name__=="__main__":
    # read clusters
    cluster_file_dir = "../../data/clusters_after_remove_files_with_no_popsa.yaml"
    with open(cluster_file_dir) as file:
        clusters = yaml.full_load(file)
    clusters = clusters[0:30]
    print('lengths of clusters:')
    print([len(x) for x in clusters])

    # get similarities as ditionary
    tic = time.perf_counter()
    sim = pd.read_csv('../../data/googlenet-ssc.dat', sep=' ', names=['pair', 'pair_b', 'value'], engine='python')
    sim['pair'] = sim.pair + '-' + sim.pair_b
    sim = dict(zip(sim.pair, sim.value))
    toc = time.perf_counter()
    print(f"Loaded all similarities in {toc - tic:0.4f} seconds")

    # numpy array for similarities
    similarity_mat = np.empty([30, 30])

    # intra-cluster similarity
    for cluster_num in range(30):
        print('computing intra-cluster similarity of cluster {}...'.format(cluster_num))
        cluster = clusters[cluster_num]
        combs = list(combinations(cluster, 2))

        cluster_similarity = 0
        num_comb = 0 # include valid combinations only
        for comb in combs:
            if '{}-{}'.format(comb[0], comb[1]) in sim:
                cluster_similarity += sim['{}-{}'.format(comb[0], comb[1])]
                num_comb += 1
            elif '{}-{}'.format(comb[1], comb[0]) in sim:
                cluster_similarity += sim['{}-{}'.format(comb[1], comb[0])]
                num_comb += 1

        print('valid number of pairs: ', num_comb)
        cluster_similarity = cluster_similarity/float(num_comb+0.000000000000000001)
        print('cluster similarity of cluster {}: {}'.format(cluster_num, cluster_similarity))
        similarity_mat[cluster_num][cluster_num] = cluster_similarity
        print('**************************************************************\n')

    # inter-cluster similarity
    for a in range(30):
        for b in range(a + 1, 30):
            print('computing inter-cluster similarity of cluster {} and {}...'.format(a, b))
            cluster_a = clusters[a]
            cluster_b = clusters[b]
            combs = list(product(cluster_a, cluster_b))
            
            cluster_similarity = 0
            num_comb = 0 # include valid combinations only
            for comb in combs:
                if '{}-{}'.format(comb[0], comb[1]) in sim:
                    cluster_similarity += sim['{}-{}'.format(comb[0], comb[1])]
                    num_comb += 1
                elif '{}-{}'.format(comb[1], comb[0]) in sim:
                    cluster_similarity += sim['{}-{}'.format(comb[1], comb[0])]
                    num_comb += 1

            print('valid number of pairs: ', num_comb)
            cluster_similarity = cluster_similarity/float(num_comb+0.000000000000000001)
            print('cluster similarity of cluster {} and {}: {}'.format(a, b, cluster_similarity))
            similarity_mat[a][b] = cluster_similarity
            similarity_mat[b][a] = cluster_similarity
            print('**************************************************************\n')  

    # save the matrix to plot a heatmap
    np.save('./cluster_similarity_matrix.npy', similarity_mat)              