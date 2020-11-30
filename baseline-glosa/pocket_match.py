"""
Pocket matching with G-LoSA.
"""
import yaml
from preprocess_all import read_cluster_file_from_yaml
from preprocess_all import merge_clusters
import subprocess
from tqdm import tqdm
import numpy as np
import sklearn.metrics as metrics


def pocket_match(pocket, pocket_cf, label_pocket, label_pocket_cf):
    """call C++ VINA program to do pocket matching"""
    #p = subprocess.run(['./glosa', '-s1 {}'.format(pocket), '-s1cf {}'.format(pocket_cf), '-s2 {}'.format(label_pocket), '-s2cf {}'.format(label_pocket_cf), '-o 0'], 
    p = subprocess.run('./glosa -s1 {} -s1cf {} -s2 {} -s2cf {} -o 0'.format(pocket, pocket_cf, label_pocket, label_pocket_cf),
                        shell=True,
                        stdout=subprocess.PIPE,
                        text=True,
                        #check=True,
                        cwd='/home/wentao/Desktop/local-workspace/siamese-monet-project/glosa/glosa_v2.2/')
    
    # when there is no error
    if p.returncode == 0: 
        result = p.stdout
        return parse_result(result)
    else:
        global error_cnt
        error_cnt += 1
        return 0

def parse_result(result):
    """parse the results and return the score as a float"""
    return float(result.split()[-1])


if __name__ == "__main__":
    # recreate dataset with the same split as when training
    with open('../train_classifier.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    cluster_file_dir = '../../data/clusters_after_remove_files_with_no_popsa.yaml'
    pocket_dir = config['pocket_dir']    
    merge_info = config['merge_info']

    clusters = read_cluster_file_from_yaml(cluster_file_dir)
    clusters = merge_clusters(clusters, merge_info)

    # label pockets
    #label_dir = '../../glosa/glosa_v2.2/label_pockets/'
    label_dir = './label_pockets/'
    label_to_pocket = {
        0: '2ddoA00', # ATP
        1: '1bcfE00', # Heme
        2: '1e55A00', # Carbonhydrate
        3: '5frvB00', # Benzene ring
        4: '5oy0A03', # Chlorophyll
        5: '6rfcA03', # lipid 
        6: '4iu5A00', # Essential amino acid/citric acid/tartaric acid
        7: '4ineA00', # S-adenosyl-L-homocysteine
        8: '6hxiD01', # CoenzymeA
        9: '5ce8A00', # pyridoxal phosphate
        10: '5im2A00', # benzoic acid
        11: '1t57A00', # flavin mononucleotide
        12: '6frlB01', # morpholine ring
        13: '4ymzB00' # phosphate
    }   

    global error_cnt
    error_cnt = 0

    pocket_dir = './all_pockets/'
    target = [] # target labels of all data points
    prediction = [] # prediction by pocket matching
    for label, cluster in enumerate(clusters):
        print('computing class {}...'.format(label))
        for pocket in cluster:
            scores = []
            for out in range(14):
                label_pocket = label_dir + label_to_pocket[out] + '.pdb'
                label_pocket_cf = label_dir + label_to_pocket[out] + '-cf.pdb' # chemical feature file
                in_pocket = pocket_dir + pocket + '.pdb'
                in_pocket_cf = pocket_dir + pocket + '-cf.pdb'
                #print(label_pocket)
                #print(label_pocket_cf)
                #print(in_pocket)
                #print(in_pocket_cf)

                scores.append(pocket_match(in_pocket, in_pocket_cf, label_pocket, label_pocket_cf))

            # get index of max score as predicted class
            pred = np.argmax(np.array(scores))

            # store label and predicted class
            target.append(label)
            prediction.append(pred)
            
    # compute the metrics
    report = metrics.classification_report(np.array(target), np.array(prediction), digits=4)
    print('classification report:')
    print(report)

    # report number of error cases
    print('number of pairs that error occurs: ', error_cnt)