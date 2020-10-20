"""
Plot the cluster similarity before merging as a heatmap.
"""
import numpy as np
import seaborn as sns; sns.set_theme()
import matplotlib.pyplot as plt
import matplotlib

if __name__ == "__main__":
    similarity_mat = np.load('./result/cluster_similarity_matrix.npy')
    #print(similarity_mat)  

    fig, ax = plt.subplots(figsize=(20, 16))
    ax = sns.heatmap(similarity_mat, linewidths=.5, cmap="YlGnBu", annot=True, fmt='.2f')
    plt.savefig('./result/cluster_similarity.png')