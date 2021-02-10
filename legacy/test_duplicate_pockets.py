import yaml
with open("../data/clusters_after_remove_files_with_no_popsa.yaml") as f:
    clusters = yaml.load(f, Loader=yaml.FullLoader)

"""
total_pockets = 0
for cluster in clusters[0:30]:
    total_pockets += len(cluster)
print('total number of pockets:', total_pockets)

pocket_set = set()
for cluster in clusters[0:30]:
    for pocket in cluster:
        pocket_set.add(pocket[0:-2])
print('total number of pockets:', len(list(pocket_set)))
"""

pocket_set = set()
new_cluster_file = "../data/clusters_after_remove_duplicates.yaml"
new_clusters = []
for cluster in clusters:
    new_cluster = []
    for pocket in cluster:
        if pocket[0:-2] not in pocket_set:
            pocket_set.add(pocket[0:-2])
            new_cluster.append(pocket)
    new_clusters.append(new_cluster)

# report number of pockets after removing duplicates
total_pockets = 0
for cluster in new_clusters[0:30]:
    total_pockets += len(cluster)
print('total number of pockets:', total_pockets)
