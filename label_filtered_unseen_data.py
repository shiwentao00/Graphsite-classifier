"""
Label the filtered unseen data that filtered by Manali
"""
import yaml
import pandas as pd


if __name__ == "__main__":
    with open('../unseen-data/unseen-pocket-list_new.yaml') as f:
        clusters = yaml.load(f, Loader=yaml.FullLoader)  
    cluster_sets = [set(x) for x in clusters]
    num_classes = len(cluster_sets)

    df = pd.read_csv('../unseen-data/filtered_test.csv')
    protein1 = df['protein1'].tolist()
    protein2 = df['protein2'].tolist()
    protein = protein1 + protein2
    pockets = list(set(protein))
    print(len(pockets))

    unseen_data = [[] for _ in range(num_classes)]
    for pocket in pockets:
        for cnt, cluster in enumerate(cluster_sets):
            if pocket in cluster:
                unseen_data[cnt].append(pocket)

    print(sum([len(x) for x in unseen_data]))