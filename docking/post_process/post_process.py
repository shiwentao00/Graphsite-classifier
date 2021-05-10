"""
Take the classificaton results and combine them together.
"""
import yaml
from os import listdir
from os.path import isfile, join
import sklearn.metrics as metrics

def get_class(file_name, normalize):
    if normalize:
        label = file_name.split('-')[2]
    else:
        label = file_name.split('-')[1]
    
    return int(label[5:])

if __name__ == "__main__":
    normalize = True
    result_dir = '../../../smina/output/'
    
    files = []
    if normalize:
        for f in listdir(result_dir):
            if isfile(join(result_dir, f)) and f.startswith('norm'):
                files.append(f)
    else:
        for f in listdir(result_dir):
            if isfile(join(result_dir, f)) and f.startswith('preds'):
                files.append(f)

    # load the predictions of audock-vina
    targets = []
    predictions = []
    for x in files:
        file_path = result_dir + x
        label = get_class(x, normalize)

        # load the predictions in this file
        with open(file_path, 'r') as f:
            file_predictions = yaml.full_load(f)
            for prediction in file_predictions:
                predictions.append(prediction)
                targets.append(label)

    # generate final classification report
    report = metrics.classification_report(targets, predictions, digits=4)
    print(report)


"""
Classification results without normalization

              precision    recall  f1-score   support

           0     0.4951    0.2856    0.3623      7625
           1     0.0984    0.4836    0.1636      1158
           2     0.3521    0.0083    0.0163      3001
           3     0.7000    0.0066    0.0132      1054
           4     0.1424    0.2975    0.1926       968
           5     1.0000    0.0005    0.0011      1890
           6     0.7500    0.0018    0.0036      1663
           7     0.1646    0.2625    0.2023       602
           8     0.0360    0.1728    0.0596       573
           9     0.0156    0.0018    0.0032       566
          10     0.0833    0.0022    0.0043       896
          11     0.0406    0.4988    0.0751       417
          12     0.0000    0.0000    0.0000       374
          13     0.0000    0.0000    0.0000       337

    accuracy                         0.1671     21124
   macro avg     0.2770    0.1444    0.0784     21124
weighted avg     0.4345    0.1671    0.1610     21124
"""


"""
Classification results with normalization

              precision    recall  f1-score   support

           0     0.0000    0.0000    0.0000      7625
           1     0.0000    0.0000    0.0000      1158
           2     0.1667    0.0037    0.0072      3001
           3     0.0557    0.8378    0.1044      1054
           4     0.0000    0.0000    0.0000       968
           5     0.4133    0.0492    0.0879      1890
           6     0.1342    0.0902    0.1079      1663
           7     0.0000    0.0000    0.0000       602
           8     0.0000    0.0000    0.0000       573
           9     0.0000    0.0000    0.0000       566
          10     0.0451    0.1931    0.0731       896
          11     0.0000    0.0000    0.0000       417
          12     0.0000    0.0000    0.0000       374
          13     0.0000    0.0000    0.0000       337

    accuracy                         0.0620     21124
   macro avg     0.0582    0.0839    0.0272     21124
weighted avg     0.0759    0.0620    0.0257     21124
"""


