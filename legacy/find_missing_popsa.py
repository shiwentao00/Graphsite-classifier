"""
Find the missing popsa files. Then we manually remove them from train/val/test.
"""
from dataloader import read_cluster_file
from os import listdir
from os.path import isfile, join
import yaml


def protein_list_by_popsa(pop_dir):
    """returns a list of protein codes in the popsa folder
    """
    protein_list = [f for f in listdir(pop_dir) if isfile(join(pop_dir, f))]
    protein_list = [x[0:-5] for x in protein_list]
    return protein_list

if __name__=="__main__":
    cluster_file_dir = '../data/googlenet-classes'
    pop_dir = '../data/pops-googlenet/'    
    output_file = '../data/clusters_after_remove_files_with_no_popsa.yaml'

    clusters = read_cluster_file(cluster_file_dir)

    protein_list = protein_list_by_popsa(pop_dir)

    num_pockets = sum([len(x) for x in clusters])
    print('number of pockets in original clusters: ', num_pockets)

    for cluster in clusters:
        for pocket in cluster:
            if pocket[0:-2] not in protein_list:
                cluster.remove(pocket)
                print('can not find popsa file for {}'.format(pocket))

    num_pockets = sum([len(x) for x in clusters])
    print('number of pockets after removing: ', num_pockets)

    print('saving new clusters to {}...'.format(output_file))
    with open(output_file, 'w') as file:
        documents = yaml.dump(clusters, file)

    with open(output_file) as file:
        loaded_clusters = yaml.full_load(file)   
    print(loaded_clusters)

