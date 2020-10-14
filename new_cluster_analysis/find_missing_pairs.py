"""
Find the missing pairs in the similarity file and print out
"""
#import argparse
import yaml
from itertools import combinations, product 
import pandas as pd
import time
import numpy as np
from tqdm import tqdm
    


if __name__ == "__main__":
    cluster_file_dir = "../../data/clusters_after_remove_files_with_no_popsa.yaml"
    with open(cluster_file_dir) as file:
        clusters = yaml.full_load(file) 
    clusters = clusters[0:30]
    #print('lengths of clusters:')
    #print([len(x) for x in clusters])

    # get similarities as ditionary
    tic = time.perf_counter()
    #sim = pd.read_csv('./test.csv', sep=' ', names=['pair', 'value'], engine='python')
    sim = pd.read_csv('../../similarity-coeff.csv', sep=' ', names=['pair', 'value'], engine='python')
    sim = dict(zip(sim.pair, sim.value))
    toc = time.perf_counter()
    #print(f"Loaded all similarities in {toc - tic:0.4f} seconds")


    # intra-cluster pairs
    for cluster_num in range(30):
        cluster = clusters[cluster_num]
        combs = list(combinations(cluster, 2))

        for comb in combs:
            if ('{}-{}'.format(comb[0], comb[1]) not in sim) and ('{}-{}'.format(comb[1], comb[0]) not in sim):
                print('{}-{}'.format(comb[0], comb[1]))

    # inter-cluster similarity
    for a in range(30):
        for b in range(a + 1, 30):
            cluster_a = clusters[a]
            cluster_b = clusters[b]
            combs = list(product(cluster_a, cluster_b))
            
            for comb in combs:
                if ('{}-{}'.format(comb[0], comb[1]) not in sim) and ('{}-{}'.format(comb[1], comb[0]) not in sim):
                    print('{}-{}'.format(comb[0], comb[1]))



  

