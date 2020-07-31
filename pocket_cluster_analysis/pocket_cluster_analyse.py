"""
Analyse the pocket clusters and use a subset of the data as our dataset.
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def main():
    cluster_file = '../../data/googlenet-classes'
    
    print('loading data...')
    f = open(cluster_file, 'r')
    data_text = f.read()
    f.close()
    data_text = data_text.split('\n')
    
    while('' in data_text): 
        data_text.remove('')
    
    pocket_classes = [int(x.split()[1]) for x in data_text]
    class_sizes = [int(x.split()[2]) for x in data_text]
    print('total number of classes:', len(class_sizes))
    print('total number of pockets:', sum(class_sizes))

    small_classes = [x for x in class_sizes if x < 30]
    print('number of small classes (1-29):', len(small_classes))
    print('number of small class pockets:', sum(small_classes))

    middle_classes = [x for x in class_sizes if x >= 30 and x < 200]
    print('number of middle classes (30-199):', len(middle_classes))
    print('number of middle class pockets:', sum(middle_classes))

    large_classes = [x for x in class_sizes if x >= 200 and x < 1000]
    print('number of large classes (200-999)', len(large_classes))
    print('number of large class pockets:', sum(large_classes))

    super_large_classes = [x for x in class_sizes if x >= 1000]
    print('number of super large classes (>=1000)', len(super_large_classes))
    print('number of super large class pockets:', sum(super_large_classes))


if __name__=="__main__":
    main()