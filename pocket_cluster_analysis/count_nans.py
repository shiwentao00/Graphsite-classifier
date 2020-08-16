"""
Number of nans: 1015124
Ratio of nans: 1015124/32821441 = 0.03092868469729894
"""
import numpy as np

sim_mat = np.load('../../similarity_matrices/similarity_matrix_cluster_0_with_nan.npy')
cnt = np.isnan(sim_mat)
print(cnt)
cnt = np.count_nonzero(cnt)
print(cnt)

print(sim_mat.shape)
print(sim_mat.shape[0]*sim_mat.shape[1])
print(1015124/32821441)