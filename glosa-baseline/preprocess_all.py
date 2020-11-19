"""
Replace the last chracters with "TER". Originally it is "END"
"""
import yaml
from os import listdir
from os.path import isfile, join
from tqdm import tqdm


def read_cluster_file_from_yaml(cluster_file_dir):
    """Read the clustered pockets as a list of lists.
    """
    with open(cluster_file_dir) as file:
        clusters = yaml.full_load(file)    
    return clusters


def merge_clusters(clusters, merge_info):
    """Merge some clusters according to merge_info.
    
    Arguments:   
    clusters - list of lists of pockets to represent the pocket clusters.   

    merge_info - new combination of clusters. e.g., [[0,3], [1,2], 4].
    
    Return:   
    new_clusters -  list of lists of pockets to represent the pocket clusters after merging.
    """
    new_clusters = []
    for item in merge_info:
        if type(item) == int:
            new_clusters.append(clusters[item])
        elif type(item) == list:
            temp = []
            for idx in item:
                temp.extend(clusters[idx])
            new_clusters.append(temp)
        else:
            raise TypeError("'merge_info' muse be a list with elements that are either int or list of int.")

    return new_clusters


def preprocess(in_path, out_path):
    in_pocket = open(in_path, 'r')
    pocket = in_pocket.readlines()
    in_pocket.close()
    new_pocket = []
    for line in pocket[0:-1]:
        new_pocket.append(process_line(line))
    new_pocket.append('TER')

    with open(out_path, 'w') as f:
        for x in new_pocket:
            f.write(x)


def process_line(line):
    line = line.strip()
    line = line + '  1.00  0.00           {}  \n'.format(line[13])
    return line

if __name__ == "__main__":
    # recreate dataset with the same split as when training
    with open('../train_classifier.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    cluster_file_dir = '../../data/clusters_after_remove_files_with_no_popsa.yaml'
    pocket_dir = config['pocket_dir']    
    merge_info = config['merge_info']

    clusters = read_cluster_file_from_yaml(cluster_file_dir)
    clusters = merge_clusters(clusters, merge_info)

    # get the input files
    in_dir = '../../data/googlenet-dataset/'
    infiles = []
    for cluster in clusters:
        for x in cluster:
            infiles.append('{}/{}.pdb'.format(x, x))

    # generate modified files
    out_dir = '../../glosa/glosa_v2.2/all_pockets/'
    for f in tqdm(infiles):
        in_path = in_dir + f
        out_path = out_dir + f.split('/')[-1]
        preprocess(in_path, out_path)