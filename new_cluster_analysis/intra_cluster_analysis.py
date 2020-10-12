"""
Read the similarity file and 
"""
import argparse
import yaml
from itertools import combinations 
import pandas as pd
from multiprocessing import Pool


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
    for comb in combs:
        if '{}-{}'.format(comb[0], comb[1]) in sim:
            accum += sim['{}-{}'.format(comb[0], comb[1])]
        elif '{}-{}'.format(comb[1], comb[0]) in sim:
            accum += sim['{}-{}'.format(comb[1], comb[0])]
    return accum

if __name__ == "__main__":
    args = get_args()
    cluster_num = int(args.cluster)
    parallel = int(args.parallel)
    chunk_size = int(args.chunk_size)
    #chunk_size = 5

    cluster_file_dir = "../../data/clusters_after_remove_files_with_no_popsa.yaml"
    with open(cluster_file_dir) as file:
        clusters = yaml.full_load(file) 
    clusters = clusters[0:30]
    cluster = clusters[cluster_num]
    print('lengths of clusters:')
    print([len(x) for x in clusters])

    # get similarities as ditionary
    #sim = pd.read_csv('./test.csv', sep=' ', names=['pair', 'value'], engine='python')
    sim = pd.read_csv('../../similarity-coeff.csv.', sep=' ', names=['pair', 'value'], engine='python')
    sim = dict(zip(sim.pair, sim.value))
    #print(sim['2q7gA01-3ze8A02'])
    #print(sim['aa'])
    print(sim)

    # combinations of all pockets in this cluster
    comb = list(combinations(cluster, 2))
    num_comb = len(comb)
    print('number of combinations: ', num_comb)

    # divide the combination list into list of lists
    comb_chunks = [comb[x:x+chunk_size] for x in range(0, num_comb, chunk_size)]

    # use multiprocessing to process chunks in parallel
    with Pool(processes=parallel) as pool:
        result = pool.map(accumulate_sim, comb_chunks)

    # sum up the similarites of each process, and average
    cluster_similarity = sum(result)
    print('cluster similarity of cluster {}: {}'.format(cluster_num, cluster_similarity))

    

