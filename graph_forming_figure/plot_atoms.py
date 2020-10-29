"""
Plot the atoms in 3d Euclidean space to visualize the graph representaion.
Pocket: 5x06F00
"""
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from matplotlib import rcParams
import matplotlib


if __name__ == "__main__":
    rcParams['xtick.color'] = 'white'
    rcParams['ytick.color'] = 'white'    
    font = {'size': 8}
    matplotlib.rc('font', **font)  

    # parse the file and read data
    atoms = open('./atoms_1.txt', 'r')
    lines = atoms.readlines()

    # plot atoms
    fig = plt.figure(dpi=600)
    ax = plt.axes(projection='3d')
    color_map = {'N':'green', 'O':'blue', 'C':'red'}
    name_map = {'N':'Nitrogen', 'O':'Oxygen', 'C':'Carbon'}
    x = []
    y = []
    z = []
    for atom_type in ['C', 'O', 'N']:
        x_ = []
        y_ = []
        z_ = []
        for line in lines:
            line = line.strip()
            line = line.split() 
            if line[1][0] == atom_type:
                x_.append(float(line[2]))
                y_.append(float(line[3]))
                z_.append(float(line[4]))
        x.extend(x_)
        y.extend(y_)
        z.extend(z_)
        ax.scatter(x_, y_, z_, c=color_map[atom_type], label=name_map[atom_type]) 
    ax.legend(bbox_to_anchor=(0.05, 0.8), loc=2, borderaxespad=0.)

    # compute the pair-wise distances
    coord = np.transpose(np.vstack([x,y,z]))
    atom_dist = distance.cdist(coord, coord, 'euclidean') # the distance matrix
    threshold_condition = atom_dist > 4.5 # set the element whose value is larger than threshold to 0
    atom_dist[threshold_condition] = 0 # set the element whose value is larger than threshold to 0
    result = np.where(atom_dist > 0)
    #print(result)

    # form edges
    for s, t in zip(result[0], result[1]):
        values = np.vstack([coord[s],coord[t]])
        x_values = values[:, 0]
        y_values = values[:, 1]
        z_values = values[:, 2]
        plt.plot(x_values, y_values, z_values, color='gray', linewidth=0.5)

    # add chemical bonds

    # plot edges 

    # figure configurations
    #ax = plt.gca()

    ax.xaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('white')
    ax.yaxis.pane.fill = False
    ax.yaxis.pane.set_edgecolor('white')
    ax.zaxis.pane.fill = False
    ax.zaxis.pane.set_edgecolor('white')

    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.zaxis.set_ticklabels([])

    #ax.set_xticks([])          # removes the ticks... great now the rest of it
    #ax.set_yticks([])
    #ax.set_zticks([])

    ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))    

    #ax.grid(False)
    ax.grid(True)
    #for axi in (ax.w_xaxis, ax.w_yaxis, ax.w_zaxis):
    #    for tic in axi.get_major_ticks():
    #        tic.tick1On = tic.tick2On = False
    #        tic.label1On = tic.label2On = False    

    # save figure
    ax.view_init(10, 35)
    plt.tight_layout()
    plt.savefig('./atoms.png', bbox_inches='tight')

