"""
Read the similarity file and 
"""
import argparse
import yaml
from itertools import combinations 
import pandas as pd
from multiprocessing import Pool
import time
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser('python')              

    parser.add_argument('-cluster',
                        required=True,
                        type=int,
                        help='which cluster (0 to 29) to compute intra-similarity.')     

    parser.add_argument('-parallel',
                        default=20,
                        type=int,
                        help='how many processes to compute in parallel.')     

    parser.add_argument('-chunk_size',
                        default=50000,
                        type=int,
                        help='how many processes to compute in parallel.')   

    return parser.parse_args()
    

def accumulate_sim(combs):
    """Accumulate similarities by traversing through the combinations"""
    accum = 0
    num_comb = 0
    for comb in combs:
        if '{}-{}'.format(comb[0], comb[1]) in sim:
            accum += sim['{}-{}'.format(comb[0], comb[1])]
            num_comb += 1
        elif '{}-{}'.format(comb[1], comb[0]) in sim:
            accum += sim['{}-{}'.format(comb[1], comb[0])]
            num_comb += 1
    return accum, num_comb

if __name__ == "__main__":
    args = get_args()
    #cluster_num = int(args.cluster)
    parallel = int(args.parallel)
    chunk_size = int(args.chunk_size)
    #chunk_size = 5

    cluster_file_dir = "../../data/clusters_after_remove_files_with_no_popsa.yaml"
    with open(cluster_file_dir) as file:
        clusters = yaml.full_load(file) 
    clusters = clusters[0:30]
    print('lengths of clusters:')
    print([len(x) for x in clusters])

    # get similarities as ditionary
    #sim = pd.read_csv('./test.csv', sep=' ', names=['pair', 'value'], engine='python')
    tic = time.perf_counter()
    sim = pd.read_csv('../../similarity-coeff.csv', sep=' ', names=['pair', 'value'], engine='python')
    sim = dict(zip(sim.pair, sim.value))
    toc = time.perf_counter()
    print(f"Loaded all similarities in {toc - tic:0.4f} seconds")

    for cluster_num in range(0, 30):
        print('processing cluster {}...'.format(cluster_num))
        cluster = clusters[cluster_num]

        # combinations of all pockets in this cluster
        combs = list(combinations(cluster, 2))
        #num_comb = len(combs)
        #print('number of combinations: ', num_comb)

        cluster_similarity = 0
        num_comb = 0 # include valid combinations only
        for comb in tqdm(combs):
            if '{}-{}'.format(comb[0], comb[1]) in sim:
                cluster_similarity += sim['{}-{}'.format(comb[0], comb[1])]
                num_comb += 1
            elif '{}-{}'.format(comb[1], comb[0]) in sim:
                cluster_similarity += sim['{}-{}'.format(comb[1], comb[0])]
                num_comb += 1

        print('valid number of pairs: ', num_comb)
        cluster_similarity = cluster_similarity/float(num_comb+0.000000000000000001)

        print('cluster similarity of cluster {}: {}'.format(cluster_num, cluster_similarity))
        print('**************************************************************\n')

        """
        # divide the combination list into list of lists
        comb_chunks = [comb[x:x+chunk_size] for x in range(0, num_comb, chunk_size)]
        print('length of chunks: ')
        print([len(x) for x in comb_chunks])

        # use multiprocessing to process chunks in parallel
        with Pool(processes=parallel) as pool:
            result = pool.map(accumulate_sim, comb_chunks)

        # sum up the similarites of each process, and average
        cluster_similarity = sum(result)
        """

    

