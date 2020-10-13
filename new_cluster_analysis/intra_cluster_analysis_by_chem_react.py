"""
Divide the large clusters into subclusters.
"""
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
import time


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
    #sim = pd.read_csv('../../data/googlenet-ssc.dat', sep=' ', names=['pair', 'value'], engine='python')
    sim = pd.read_csv('./chem_react_test.dat', sep=' ', names=['pair', 'pair_b', 'value'], engine='python')
    sim['pair'] = sim.pair + '-' + sim.pair_b
    sim = dict(zip(sim.pair, sim.value))
    toc = time.perf_counter()
    print(f"Loaded all similarities in {toc - tic:0.4f} seconds")

    print(sim)

    for clusters_num in range(30):
        print('generating similarity matrix for cluster {}...'.format(clusters_num))
        cluster = clusters[clusters_num]



