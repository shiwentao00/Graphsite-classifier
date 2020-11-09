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
import pandas as pd


def bond_parser(pocket_path):
    f = open(pocket_path,'r')
    f_text = f.read()
    f.close()
    bond_start = f_text.find('@<TRIPOS>BOND')
    bond_end = -1
    df_bonds = f_text[bond_start:bond_end].replace('@<TRIPOS>BOND\n','')
    df_bonds = df_bonds.replace('am', '1') # amide
    df_bonds = df_bonds.replace('ar', '1.5') # aromatic
    df_bonds = df_bonds.replace('du', '1') # dummy
    df_bonds = df_bonds.replace('un', '1') # unknown
    df_bonds = df_bonds.replace('nc', '0') # not connected
    df_bonds = df_bonds.replace('\n',' ')
    df_bonds = np.array([np.float(x) for x in df_bonds.split()]).reshape((-1,4)) # convert the the elements to integer
    df_bonds = pd.DataFrame(df_bonds, columns=['bond_id', 'atom1', 'atom2', 'bond_type'])
    df_bonds.set_index(['bond_id'], inplace=True)
    return df_bonds


if __name__ == "__main__":
    rcParams['xtick.color'] = 'white'
    rcParams['ytick.color'] = 'white'    
    font = {'size': 8}
    matplotlib.rc('font', **font)  

    # parse the file and read data
    atoms = open('./residues.txt', 'r')
    lines = atoms.readlines()

    # atoms
    fig = plt.figure(dpi=600)
    ax = plt.axes(projection='3d')
    color_map = {'N':'green', 'O':'blue', 'C':'red'}
    name_map = {'N':'Nitrogen', 'O':'Oxygen', 'C':'Carbon'}
    x = []
    y = []
    z = []
    cnt = 0
    atom_idx_map = {} # dictionary that maps atom number to index
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
                atom_idx_map[int(line[0])] = cnt # map atom number to index
                cnt += 1
        x.extend(x_)
        y.extend(y_)
        z.extend(z_)
        ax.scatter(x_, y_, z_, c=color_map[atom_type], label=name_map[atom_type]) 

    # compute the pair-wise distances
    coord = np.transpose(np.vstack([x,y,z]))
    atom_dist = distance.cdist(coord, coord, 'euclidean') # the distance matrix
    threshold_condition = atom_dist > 4.5 # set the element whose value is larger than threshold to 0
    atom_dist[threshold_condition] = 0 # set the element whose value is larger than threshold to 0
    result = np.where(atom_dist > 0)

    # form edges
    for s, t in zip(result[0], result[1]):
        values = np.vstack([coord[s],coord[t]])
        x_values = values[:, 0]
        y_values = values[:, 1]
        z_values = values[:, 2]
        plt.plot(x_values, y_values, z_values, color='gray', linewidth=0.5, label='Unbonded')

    # parse chemical bonds
    bonds = bond_parser('./residue_bonds.txt')
    print(bonds)

    # plot chemical bonds
    print(atom_idx_map)
    for _, row in bonds.iterrows():
        a = int(row['atom1'])
        b = int(row['atom2'])
        if a in atom_idx_map and b in atom_idx_map:
            idx_a = atom_idx_map[a]
            idx_b = atom_idx_map[b]
            coord_a = coord[idx_a]
            coord_b = coord[idx_b]
            values = np.vstack([coord_a,coord_b])
            x_values = values[:, 0]
            y_values = values[:, 1]
            z_values = values[:, 2]
            plt.plot(x_values, y_values, z_values, color='orange', linewidth=1, label='Bonded')

    # figure configurations
    #ax = plt.gca()
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(0.05, 0.8), loc=2, borderaxespad=0.)
    #ax.legend(bbox_to_anchor=(0.05, 0.8), loc=2, borderaxespad=0.)

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
    plt.savefig('./atoms.tif', bbox_inches='tight')

