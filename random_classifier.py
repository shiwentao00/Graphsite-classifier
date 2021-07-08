"""compute the performance of a random classifier"""
import yaml
from dataloader import read_cluster_file_from_yaml
from dataloader import merge_clusters
import random
import sklearn.metrics as metrics


if __name__ == "__main__":
    with open('./train_classifier.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    cluster_file_dir = config['cluster_file_dir']
    merge_info = config['merge_info']

    clusters = read_cluster_file_from_yaml(cluster_file_dir)
    clusters = merge_clusters(clusters, merge_info)

    print("lengths:")
    for cluster in clusters:
        print(len(cluster))

    # true labels
    labels = []
    for i, cluster in enumerate(clusters):
        for _ in cluster:
            labels.append(i)

    # random prediction, probability weighted by class size
    predictions = []
    for _ in range(len(labels)):
        predictions.append(random.choice(labels))

    report = metrics.classification_report(labels, predictions, digits=4)

    for i in range(14):
        num = 0
        for x in predictions:
            if x == i:
                num += 1
        print('number of predictions of {} class: {}'.format(i, num))

    print(report)

    '''
              precision    recall  f1-score   support

           0     0.3551    0.3599    0.3575      7625
           1     0.0570    0.0544    0.0557      1158
           2     0.1462    0.1406    0.1433      3001
           3     0.0516    0.0541    0.0528      1054
           4     0.0491    0.0486    0.0488       968
           5     0.0883    0.0915    0.0899      1890
           6     0.0905    0.0932    0.0918      1663
           7     0.0310    0.0299    0.0304       602
           8     0.0351    0.0349    0.0350       573
           9     0.0207    0.0194    0.0201       566
          10     0.0359    0.0357    0.0358       897
          11     0.0206    0.0192    0.0199       417
          12     0.0162    0.0160    0.0161       374
          13     0.0147    0.0148    0.0147       337

    accuracy                         0.1780     21125
   macro avg     0.0723    0.0723    0.0723     21125
weighted avg     0.1768    0.1780    0.1774     21125    
    '''
