"""
Post-process the docking result to get the final classification result.
"""
from os import listdir
from os.path import isfile, join
import yaml


if __name__ == "__main__":
    # root directory
    result_dir = '../../vina/vina-classification-result/'

    # list of files
    files = [f for f in listdir(result_dir) if isfile(join(result_dir, f))]
    print(files)

    target = []
    prediction = []
    # load the predictions
    with open(result_dir + files[0]) as f:
        prediction = yaml.load(f, Loader=yaml.FullLoader)

    print(prediction)
