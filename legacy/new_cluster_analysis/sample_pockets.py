import random
import yaml

if __name__ == "__main__":
    random.seed(666)
    cluster_file_dir = "../../data/clusters_after_remove_files_with_no_popsa.yaml"
    with open(cluster_file_dir) as file:
        clusters = yaml.full_load(file)
    clusters = clusters[0:30]

    # atp
    atp = clusters[0]
    atp_sample = random.sample(atp, 20)
    print('atp pockets:')
    for x in atp_sample:
        print(x)
    print('--------------------')

    # heme
    heme = clusters[2]
    heme_sample = random.sample(heme, 20)
    print('heme pockets:')
    for x in heme_sample:
        print(x)
    print('--------------------')