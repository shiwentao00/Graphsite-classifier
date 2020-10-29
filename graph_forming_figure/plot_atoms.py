"""
Plot the atoms in 3d Euclidean space to visualize the graph representaion.
Pocket: 5x06F00
"""
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance


if __name__ == "__main__":
    # parse the file and read data
    atoms = open('./atoms_1.txt', 'r')
    lines = atoms.readlines()

    atom_types = []
    x = []
    y = []
    z = []
    for line in lines:
        line = line.strip()
        line = line.split() 
        atom_types.append(line[1][0])
        x.append(float(line[2]))
        y.append(float(line[3]))
        z.append(float(line[4]))
        #print(line)

    print(atom_types)
    print(x)
    print(y)
    print(z)

    # plot the atoms
    color_map = {'N':'green', 'O':'blue', 'C':'red'}
    colors = [color_map[x] for x in atom_types]
    fig = plt.figure(dpi=600)
    ax = plt.axes(projection='3d')
    ax.scatter3D(x, y, z, c=colors, cmap='Greens')
    ax.view_init(20, 40)

    # compute the pair-wise distances
    coord = np.transpose(np.vstack([x,y,z]))
    atom_dist = distance.cdist(coord, coord, 'euclidean') # the distance matrix
    threshold_condition = atom_dist > 4.5 # set the element whose value is larger than threshold to 0
    atom_dist[threshold_condition] = 0 # set the element whose value is larger than threshold to 0
    result = np.where(atom_dist > 0)
    print(result)

    # form edges
    for s, t in zip(result[0], result[1]):
        values = np.vstack([coord[s],coord[t]])
        x_values = values[:, 0]
        y_values = values[:, 1]
        z_values = values[:, 2]
        plt.plot(x_values, y_values, z_values, color='gray', linewidth=0.5)

    # add chemical bonds

    # plot edges 

    # save figure
    plt.tight_layout()
    plt.savefig('./atoms.png')

