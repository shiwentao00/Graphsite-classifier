"""
Post-process the docking result to get the final classification result.
"""
from os import listdir
from os.path import isfile, join
import yaml
import sklearn.metrics as metrics


if __name__ == "__main__":
    # root directory
    result_dir = '../../vina/vina-classification-result/'

    # list of files
    files = [f for f in listdir(result_dir) if isfile(join(result_dir, f))]

    # load the predictions of audock-vina
    targets = []
    predictions = []
    for x in files:
        file_path = result_dir + x
        label = int(x.split('-')[1])

        # load the predictions in this file
        with open(file_path) as f:
            file_predictions = yaml.load(f, Loader=yaml.FullLoader)
            for prediction in file_predictions:
                predictions.append(prediction)
                targets.append(label)

    # generate final classification report
    report = metrics.classification_report(targets, predictions, digits=4)
    print(report)

    """
               precision    recall  f1-score   support

           0     0.5177    0.2801    0.3635      7625
           1     0.1025    0.5682    0.1737      1158
           2     0.2453    0.0049    0.0096      2651
           3     0.3158    0.0085    0.0166       704
           4     0.1520    0.2004    0.1729       968
           5     0.6000    0.0063    0.0126      1890
           6     0.5714    0.0083    0.0164       963
           7     0.2230    0.3056    0.2579       602
           8     0.0519    0.1291    0.0740       573
           9     0.0226    0.0186    0.0204       215
          10     0.0925    0.0178    0.0299       897
          11     0.0560    0.6451    0.1030       417
          12     0.3333    0.0027    0.0053       374
          13     0.1429    0.0148    0.0269       337

    accuracy                         0.1848     19374
   macro avg     0.2448    0.1579    0.0916     19374
weighted avg     0.3726    0.1848    0.1807     19374
    """
