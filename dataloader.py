import random
import itertools
import numpy as np
import pandas as pd
from biopandas.mol2 import PandasMol2
from scipy.spatial import distance
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.data import DataLoader, DataListLoader
import statistics
import yaml


def dataloader_gen(pocket_dir, pop_dir, pos_pairs, neg_pairs, features_to_use, batch_size, shuffle=True, num_workers=1):
    """Dataloader used to wrap PairDataset. Used for training and validation """
    dataset = PairDataset(pocket_dir=pocket_dir, pop_dir=pop_dir, pos_pairs=pos_pairs, neg_pairs=neg_pairs, features_to_use=features_to_use)
    #val_set = PairDataset(pocket_dir=pocket_dir, pop_dir=pop_dir, pos_pairs=val_pos_pairs, neg_pairs=val_neg_pairs, features_to_use=features_to_use)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, follow_batch=['x_a', 'x_b'], drop_last=True)
    #val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, follow_batch=['x_a', 'x_b'], drop_last=True)

    return dataloader


def dataloader_gen_multi_gpu(pocket_dir, train_pos_pairs, train_neg_pairs, val_pos_pairs, val_neg_pairs, test_pos_pairs, test_neg_pairs, features_to_use, batch_size, shuffle=True, num_workers=1):
    """Not used"""
    train_set = PairDataset(pocket_dir=pocket_dir, pos_pairs=train_pos_pairs, neg_pairs=train_neg_pairs, features_to_use=features_to_use)
    val_set = PairDataset(pocket_dir=pocket_dir, pos_pairs=val_pos_pairs, neg_pairs=val_neg_pairs, features_to_use=features_to_use)
    test_set = PairDataset(pocket_dir=pocket_dir, pos_pairs=test_pos_pairs, neg_pairs=test_neg_pairs, features_to_use=features_to_use)
    train_loader = DataListLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    val_loader = DataListLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    test_loader = DataListLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    return train_loader, val_loader, test_loader


def pocket_loader_gen(pocket_dir, pop_dir, clusters, features_to_use, batch_size, shuffle=True, num_workers=1):
    """Dataloader used to wrap Pocket Dataset. Used for inference/testing."""
    pocketset = PocketDataset(pocket_dir=pocket_dir, pop_dir=pop_dir, clusters=clusters, features_to_use=features_to_use)
    pocketloader = DataLoader(pocketset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=True)
    return pocketloader, len(pocketset)


class PocketDataset(Dataset):
    """Dataset to generate single pocket graphs for inference/testing."""
    def __init__(self, pocket_dir, pop_dir, clusters, features_to_use):
        self.pocket_dir = pocket_dir
        self.pop_dir = pop_dir
        self.clusters = clusters 
        self.threshold = 4.5 # distance threshold to form an undirected edge between two atoms
        
        # hard coded info to generate 2 node features
        self.hydrophobicity = {'ALA':1.8,'ARG':-4.5,'ASN':-3.5,'ASP':-3.5,
                               'CYS':2.5,'GLN':-3.5,'GLU':-3.5,'GLY':-0.4,
                               'HIS':-3.2,'ILE':4.5,'LEU':3.8,'LYS':-3.9,
                               'MET':1.9,'PHE':2.8,'PRO':-1.6,'SER':-0.8,
                               'THR':-0.7,'TRP':-0.9,'TYR':-1.3,'VAL':4.2}
        self.binding_probability = {'ALA':0.701,'ARG':0.916,'ASN':0.811,'ASP':1.015,
                                    'CYS':1.650,'GLN':0.669,'GLU':0.956,'GLY':0.788,
                                    'HIS':2.286,'ILE':1.006,'LEU':1.045,'LYS':0.468,
                                    'MET':1.894,'PHE':1.952,'PRO':0.212,'SER':0.883,
                                    'THR':0.730,'TRP':3.084,'TYR':1.672,'VAL':0.884}

        total_features = ['x', 'y', 'z', 'r', 'theta', 'phi', 'sasa', 'charge',
                          'hydrophobicity', 'binding_probability', 'sequence_entropy']
                          
        assert(set(features_to_use).issubset(set(total_features))) # features to use should be subset of total_features
        self.features_to_use = features_to_use

        self.class_labels = []
        self.pockets = []
        for label, cluster in enumerate(self.clusters):
            self.pockets.extend(cluster) # flatten the clusters list
            for pocket in cluster:
                self.class_labels.append(label) # class labels for all the pockets

    def __len__(self):
        cluster_lengths = [len(x) for x in self.clusters]
        return sum(cluster_lengths)

    def __getitem__(self, idx):
        pocket = self.pockets[idx]
        label = self.class_labels[idx]
        pocket_dir = self.pocket_dir + pocket + '/' + pocket + '.mol2'
        profile_dir = self.pocket_dir + pocket + '/' + pocket[0:-2] + '.profile'
        pop_dir = self.pop_dir + pocket[0:-2] + '.pops'

        x, edge_index, edge_attr = read_pocket(pocket_dir, profile_dir, pop_dir, self.hydrophobicity, self.binding_probability, self.features_to_use, self.threshold)
        data =  Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=torch.tensor([label]))
        return data


class PairDataset(Dataset):
    """Dataset to generate pairs of data for training and validation."""
    def __init__(self, pocket_dir, pop_dir, pos_pairs, neg_pairs, features_to_use):
        self.pocket_dir = pocket_dir
        self.pop_dir = pop_dir
        self.pos_pairs = pos_pairs
        self.neg_pairs = neg_pairs
        self.num_pos_pairs = len(pos_pairs)
        self.num_neg_pairs = len(neg_pairs)
        self.threshold = 4.5 # distance threshold to form an undirected edge between two atoms

        # hard coded info to generate 2 node features
        self.hydrophobicity = {'ALA':1.8,'ARG':-4.5,'ASN':-3.5,'ASP':-3.5,
                               'CYS':2.5,'GLN':-3.5,'GLU':-3.5,'GLY':-0.4,
                               'HIS':-3.2,'ILE':4.5,'LEU':3.8,'LYS':-3.9,
                               'MET':1.9,'PHE':2.8,'PRO':-1.6,'SER':-0.8,
                               'THR':-0.7,'TRP':-0.9,'TYR':-1.3,'VAL':4.2}
        self.binding_probability = {'ALA':0.701,'ARG':0.916,'ASN':0.811,'ASP':1.015,
                                    'CYS':1.650,'GLN':0.669,'GLU':0.956,'GLY':0.788,
                                    'HIS':2.286,'ILE':1.006,'LEU':1.045,'LYS':0.468,
                                    'MET':1.894,'PHE':1.952,'PRO':0.212,'SER':0.883,
                                    'THR':0.730,'TRP':3.084,'TYR':1.672,'VAL':0.884}

        total_features = ['x', 'y', 'z', 'charge', 'hydrophobicity', 'binding_probability', 'r', 'theta', 'phi', 'sasa', 'sequence_entropy']
        assert(set(features_to_use).issubset(set(total_features))) # features to use should be subset of total_features
        self.features_to_use = features_to_use

    def __len__(self):
        return len(self.pos_pairs) + len(self.neg_pairs)

    def __getitem__(self, idx):
        # get the pair
        assert(idx>=0 and idx<(self.num_pos_pairs + self.num_neg_pairs))
        if idx >= self.num_pos_pairs:
            idx = idx - self.num_pos_pairs
            pair = self.neg_pairs[idx]
            y = torch.tensor([0]) # pair label
        else:
            pair = self.pos_pairs[idx]
            y = torch.tensor([1], dtype=torch.long) # pair label

        # pocket a location
        pocket_a_dir = self.pocket_dir + pair[0] + '/' + pair[0] + '.mol2'
        profile_a_dir = self.pocket_dir + pair[0] + '/' + pair[0][0:-2] + '.profile'
        pop_a_dir = self.pop_dir + pair[0][0:-2] + '.pops'

        # pocket b location
        pocket_b_dir = self.pocket_dir + pair[1] + '/' + pair[1] + '.mol2'
        profile_b_dir = self.pocket_dir + pair[1] + '/' + pair[1][0:-2] + '.profile'
        pop_b_dir = self.pop_dir + pair[1][0:-2] + '.pops' 

        x_a, edge_index_a, edge_attr_a = read_pocket(pocket_a_dir, profile_a_dir, pop_a_dir, self.hydrophobicity, self.binding_probability, self.features_to_use, self.threshold)
        x_b, edge_index_b, edge_attr_b = read_pocket(pocket_b_dir, profile_b_dir, pop_b_dir, self.hydrophobicity, self.binding_probability, self.features_to_use, self.threshold)

        data =  PairData(x_a=x_a, edge_index_a=edge_index_a, edge_attr_a=edge_attr_a, x_b=x_b, edge_index_b=edge_index_b, edge_attr_b=edge_attr_b, y=y)
        return data

def read_pocket(mol_path, profile_path, pop_path, hydrophobicity, binding_probability, features_to_use, threshold):
    """
    Read the mol2 file as a dataframe.
    """
    atoms = PandasMol2().read_mol2(mol_path)
    atoms = atoms.df[['atom_id','subst_name', 'atom_type', 'atom_name', 'x', 'y', 'z', 'charge']]
    atoms['residue'] = atoms['subst_name'].apply(lambda x: x[0:3])
    atoms['hydrophobicity'] = atoms['residue'].apply(lambda x: hydrophobicity[x])
    atoms['binding_probability'] = atoms['residue'].apply(lambda x: binding_probability[x])

    r, theta, phi = compute_spherical_coord(atoms[['x', 'y', 'z']].to_numpy())
    if 'r' in features_to_use:
        atoms['r'] = r
    if 'theta' in features_to_use:
        atoms['theta'] = theta
    if 'phi' in features_to_use:
        atoms['phi'] = phi 
    
    siteresidue_list = atoms['subst_name'].tolist()
    
    if 'sasa' in features_to_use:
        qsasa_data = extract_sasa_data(siteresidue_list, pop_path)
        atoms['sasa'] = qsasa_data
    
    if 'sequence_entropy' in features_to_use:
        seq_entropy_data = extract_seq_entropy_data(siteresidue_list, profile_path) # sequence entropy data with subst_name as keys
        atoms['sequence_entropy'] = atoms['subst_name'].apply(lambda x: seq_entropy_data[x])
    
    if atoms.isnull().values.any():
        print('invalid input data (containing nan):')
        print(mol_path)

    bonds = bond_parser(mol_path)
    node_features, edge_index, edge_attr = form_graph(atoms, bonds, features_to_use, threshold)
    return node_features, edge_index, edge_attr


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


def compute_edge_attr(edge_index, bonds):
    """
    Compute the edge attributes according to the chemical bonds. 
    """
    sources = edge_index[0,:]
    targets = edge_index[1,:]
    edge_attr = np.zeros((edge_index.shape[1], 1))
    for index, row in bonds.iterrows():
        # find source == row[1], target == row[0]
        source_locations = set(list(np.where(sources==(row[1]-1))[0])) # minus one because in new setting atom id starts with 0
        target_locations = set(list(np.where(targets==(row[0]-1))[0]))
        edge_location = list(source_locations.intersection(target_locations))[0]
        edge_attr[edge_location] = row[2]

        # find source == row[0], target == row[1]
        source_locations = set(list(np.where(sources==(row[0]-1))[0]))
        target_locations = set(list(np.where(targets==(row[1]-1))[0]))
        edge_location = list(source_locations.intersection(target_locations))[0]
        edge_attr[edge_location] = row[2]
    return edge_attr


def form_graph(atoms, bonds, features_to_use, threshold):
    """
    Form a graph data structure (Pytorch geometric) according to the input data frame.
    Rule: Each atom represents a node. If the distance between two atoms are less than or 
    equal to 4.5 Angstrom (may become a tunable hyper-parameter in the future), then an 
    undirected edge is formed between these two atoms. 
    
    Input:
    atoms: dataframe containing the 3-d coordinates of atoms.
    bonds: dataframe of bond info.
    threshold: distance threshold to form the edge (chemical bond).
    
    Output:
    A Pytorch-gemometric graph data with following contents:
        - node_attr (Pytorch Tensor): Node feature matrix with shape [num_nodes, num_node_features]. e.g.,
          x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
        - edge_index (Pytorch LongTensor): Graph connectivity in COO format with shape [2, num_edges*2]. e.g.,
          edge_index = torch.tensor([[0, 1, 1, 2],
                                     [1, 0, 2, 1]], dtype=torch.long)
    
    Forming the final output graph:
        data = Data(x=x, edge_index=edge_index)
    """
    A = atoms.loc[:,'x':'z'] # sample matrix
    A_dist = distance.cdist(A, A, 'euclidean') # the distance matrix
    threshold_condition = A_dist > threshold # set the element whose value is larger than threshold to 0
    A_dist[threshold_condition] = 0 # set the element whose value is larger than threshold to 0
    result = np.where(A_dist > 0)
    result = np.vstack((result[0],result[1]))
    edge_attr = compute_edge_attr(result, bonds)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    edge_index = torch.tensor(result, dtype=torch.long)

    # normalize large features
    atoms['x'] = atoms['x']/300
    atoms['y'] = atoms['y']/300
    atoms['z'] = atoms['z']/300

    node_features = torch.tensor(atoms[features_to_use].to_numpy(), dtype=torch.float32)
    return node_features, edge_index, edge_attr


def compute_spherical_coord(data):
    """
    Shift the geometric center of the pocket to origin, then compute its spherical coordinates.             
    """
    center = np.mean(data, axis=0)
    shifted_data = data - center # center the data around origin
    
    r, theta, phi = cartesian_to_spherical(shifted_data)
    return r, theta, phi


def cartesian_to_spherical(data):
    """Convert cartesian coordinates to spherical coordinates.
    
    Arguments:   
    data - numpy array with shape (n, 3) which is the 
    cartesian coordinates (x, y, z) of n points.

    Returns:   
    numpy array with shape (n, 3) which is the spherical 
    coordinates (r, theta, phi) of n points.
    """
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]

    # distances to origin
    r = np.sqrt(x**2 + y**2 + z**2)

    # angle between x-y plane and z
    theta = np.arccos(z/r)/np.pi

    # angle on x-y plane
    phi = np.arctan2(y, x)/np.pi

    #spherical_coord = np.vstack([r, theta, phi])
    #spherical_coord = np.transpose(spherical_coord)
    return r, theta, phi


def extract_sasa_data(siteresidue_list, pop):
    '''extracts accessible surface area data from .out file generated by POPSlegacy.
        then matches the data in the .out file to the binding site in the mol2 file.
        Used POPSlegacy https://github.com/Fraternalilab/POPSlegacy '''
    # Extracting sasa data from .out file
    residue_list = []
    qsasa_list = []
    with open(pop) as popsa:  # opening .out file
        for line in popsa:
            line_list = line.split()
            if len(line_list) == 12:  # extracting relevant information
                residue_type = line_list[2] + line_list[4]
                if residue_type in siteresidue_list:
                    qsasa = line_list[7]
                    residue_list.append(residue_type)
                    qsasa_list.append(qsasa)
    qsasa_list = [float(x) for x in qsasa_list]
    median = statistics.median(qsasa_list)
    qsasa_new = [median if x == '-nan' else x for x in qsasa_list]

    # Matching amino acids from .mol2 and .out files and creating dictionary
    qsasa_data = []
    fullprotein_data = list(zip(residue_list, qsasa_new))
    for i in range(len(fullprotein_data)):
        if fullprotein_data[i][0] in siteresidue_list:
            qsasa_data.append(float(fullprotein_data[i][1]))
    return qsasa_data


def extract_seq_entropy_data(siteresidue_list, profile):
    '''extracts sequence entropy data from .profile'''
    # Opening and formatting lists of the probabilities and residues
    with open(profile) as profile:  # opening .profile file
        ressingle_list = []
        probdata_list = []
        for line in profile:    # extracting relevant information
            line_list = line.split()
            residue_type = line_list[0]
            prob_data = line_list[1:]
            prob_data = list(map(float, prob_data))
            ressingle_list.append(residue_type)
            probdata_list.append(prob_data)
    ressingle_list = ressingle_list[1:]
    probdata_list = probdata_list[1:]

    # Changing single letter amino acid to triple letter with its corresponding number
    count = 0
    restriple_list = []
    for res in ressingle_list:
        newres = res.replace(res, amino_single_to_triple(res))
        count += 1
        restriple_list.append(newres + str(count))

    # Calculating information entropy
    with np.errstate(divide='ignore'):      # suppress warning
        prob_array = np.asarray(probdata_list)
        log_array = np.log2(prob_array)
        log_array[~np.isfinite(log_array)] = 0  # change all infinite values to 0
        entropy_array = log_array * prob_array
        entropydata_array = np.sum(a=entropy_array, axis=1) * -1
        entropydata_list = entropydata_array.tolist()

    # Matching amino acids from .mol2 and .profile files and creating dictionary
    fullprotein_data = dict(zip(restriple_list, entropydata_list))
    seq_entropy_data = {k: float(fullprotein_data[k]) for k in siteresidue_list if k in fullprotein_data}
    return seq_entropy_data


def amino_single_to_triple(single):
    """
    converts the single letter amino acid abbreviation to the triple letter abbreviation
    """
    
    single_to_triple_dict = {'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS',
                             'G': 'GLY', 'Q': 'GLN', 'E': 'GLU', 'H': 'HIS', 'I': 'ILE',
                             'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO',
                             'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL'}
    
    for i in single_to_triple_dict.keys():
        if i == single:
            triple = single_to_triple_dict[i]
    return triple


class PairData(Data):
    """
    Paired data type. Each object has 2 graphs.
    """
    def __init__(self,x_a, edge_index_a, edge_attr_a, x_b, edge_index_b, edge_attr_b, y):
        super(PairData, self).__init__(y=y)
        self.x_a = x_a
        self.edge_index_a = edge_index_a
        self.edge_attr_a = edge_attr_a

        self.x_b = x_b
        self.edge_index_b = edge_index_b        
        self.edge_attr_b = edge_attr_b

    def __inc__(self, key, value):
        if key == 'edge_index_a':
            return self.x_a.size(0)
        if key == 'edge_index_b':
            return self.x_b.size(0)
        else:
            return super(PairData, self).__inc__(key, value)


def divide_and_gen_pairs(cluster_file_dir, subcluster_dict, num_classes, cluster_th, train_pos_th, train_neg_th, val_pos_th, val_neg_th):
    """
    Divide the dataset and generate pairs of pockets for train, validation, and test.

    Arguments:
        cluster_file_dir: directory of the cluster file.
        num_classes: number of classes to keep.
        th: threshold of maximum number of pockets in one class.
    """
    # read the original clustered pockets
    clusters = read_cluster_file_from_yaml(cluster_file_dir)

    # select clusters according to rank of sizes and sample large clusters
    clusters = select_classes(clusters, num_classes, cluster_th)

    # replace some clusters with their subclusters
    clusters, cluster_ids = cluster_by_chem_react(clusters, subcluster_dict)
    
    # divide the clusters into train, validation and test
    train_clusters, val_clusters, test_clusters = divide_clusters(clusters)
    num_train_pockets = sum([len(x) for x in train_clusters])
    num_val_pockets = sum([len(x) for x in val_clusters])
    num_test_pockets = sum([len(x) for x in test_clusters])
    print('number of pockets in training set: ', num_train_pockets)
    print('number of pockets in validation set: ', num_val_pockets)
    print('number of pockets in test set: ', num_test_pockets)

    # train pairs
    train_pos_pairs, train_neg_pairs = gen_pairs(clusters=train_clusters, pos_pair_th=train_pos_th, neg_pair_th=train_neg_th)

    # validation pairs
    val_pos_pairs, val_neg_pairs = gen_pairs(clusters=val_clusters, pos_pair_th=val_pos_th, neg_pair_th=val_neg_th)

    return train_pos_pairs, train_neg_pairs, val_pos_pairs, val_neg_pairs


def read_cluster_file(cluster_file_dir):
    """Read the clustered pockets as a list of lists.
    """
    f = open(cluster_file_dir, 'r')
    data_text = f.read()
    f.close()
    data_text = data_text.split('\n')
    
    while('' in data_text): 
        data_text.remove('')

    clusters = []
    #cluster_sizes = []
    for x in data_text:
        cluster = x.split()[3:]
        cluster = [x.split(',')[0] for x in cluster]
        clusters.append(cluster)
        #cluster_sizes.append(len(cluster))

    return clusters
    #return clusters[1:] # !!!!!! testing performance without first large cluster.


def read_cluster_file_from_yaml(cluster_file_dir):
    """Read the clustered pockets as a list of lists.
    """
    with open(cluster_file_dir) as file:
        clusters = yaml.full_load(file)    
    return clusters


def select_classes(clusters, num_classes, th):
    """
    Keep the relatively large clusters and limit the number of pockets in super-large clusters.

    Arguments:
        clusters: list of pocket lists. The lists are already ranked according to length.
        num_classes: number of classes to keep.
        th: threshold of maximum number of pockets in one class.
    """
    selected_classes = []
    for i in range(num_classes):
        selected_classes.append(sample_from_list(clusters[i], th))

    #select_classe_lengths = [len(x) for x in selected_classes]

    return selected_classes
    

def sample_from_list(input_list, th):
    """
    Randomly samples th pockets if number of pockets larger than th. 
    """
    if len(input_list) > th:
        return random.sample(input_list, th)
    else:
        return input_list


def divide_clusters(clusters):
    """
    Shuffle and divide the clusters into train, validation and test
    """
    # shuffle the pockets in each cluster
    [random.shuffle(x) for x in clusters] # random.shuffle happens inplace

    # sizes of the clusters
    cluster_sizes = [len(x) for x in clusters]
    #print('total pockets:{}'.format(sum(cluster_sizes)))

    # train
    train_sizes = [int(0.7 * x) for x in cluster_sizes]
    #print('train pockets:{}'.format(sum(train_sizes)))

    # validation
    val_sizes = [int(0.15 * x) for x in cluster_sizes]
    #print('val pockets:{}'.format(sum(val_sizes)))

    # test
    train_val_sizes = [sum(x) for x in zip(train_sizes, val_sizes)]
    test_sizes = [a - b for a, b in zip(cluster_sizes, train_val_sizes)]
    #print('test pockets:{}'.format(sum(test_sizes)))

    train_clusters = []
    val_clusters = []
    test_clusters = []
    for i in range(len(clusters)):
        train_clusters.append(clusters[i][0:train_sizes[i]])
        val_clusters.append(clusters[i][train_sizes[i]: train_sizes[i]+val_sizes[i]])
        test_clusters.append(clusters[i][train_sizes[i]+val_sizes[i]:])
    return train_clusters, val_clusters, test_clusters


def gen_pairs(clusters, pos_pair_th=1000, neg_pair_th=20):
    """
    Generate pairs of pockets from input clusters.
    """
    # generate pairs of pockets in the same cluster
    pos_pairs = []
    for cluster in clusters:
        pairs = list(itertools.combinations(cluster, 2))
        pairs = sample_from_list(pairs, pos_pair_th)
        pos_pairs.extend(pairs)
        
    # generate pairs of pockets in different clusters
    neg_pairs = []
    class_pairs = list(itertools.combinations(clusters, 2))
    for class_pair in class_pairs:
        pairs = [(x, y) for x in class_pair[0] for y in class_pair[1]] 
        pairs = sample_from_list(pairs, neg_pair_th)
        neg_pairs.extend(pairs)
    
    return pos_pairs, neg_pairs


def cluster_by_chem_react(clusters, subcluster_dict):
    """
    Replace the original clusters with their subclusters.

    Arguments:

        clusters: list of lists of pockts.

        subcluster_dict: dictionary contains subclusters. Keys are the original cluster ids. 
    """
    num_original_clusters = len(clusters)
    dict_keys = list(subcluster_dict.keys())
    
    # store the unchanged clusters first
    new_cluster_dict = {}
    for original_cluster_id in range(num_original_clusters):
        if original_cluster_id not in dict_keys: # if this cluster is not further divided
            new_cluster_dict.update({'{}-0'.format(original_cluster_id):clusters[original_cluster_id]})

    # update the subclusters
    for key in dict_keys:
        subclusters = subcluster_dict[key]
        for idx, subcluster in enumerate(subclusters):
            new_cluster_dict.update({'{}-{}'.format(key, idx):subcluster})

    new_cluster_ids = list(new_cluster_dict.keys())
    new_cluster_ids.sort(key=lambda x: int(x.split('-')[0]))

    # for a list of lists of pockts according to new_cluster_ids
    new_clusters = []
    for cluster_id in new_cluster_ids:
        new_clusters.append(new_cluster_dict[cluster_id])

    return new_clusters, new_cluster_ids
    

def merge_clusters(clusters, merge_info):
    """Merge some clusters according to merge_info.
    
    Arguments:   
    clusters - list of lists of pockets to represent the pocket clusters.   

    merge_info - new combination of clusters. e.g., [[0,3], [1,2], 4].
    
    Return:   
    new_clusters -  list of lists of pockets to represent the pocket clusters after merging.
    """
    new_clusters = []
    for item in merge_info:
        if type(item) == int:
            new_clusters.append(clusters[item])
        elif type(item) == list:
            temp = []
            for idx in item:
                temp.extend(clusters[idx])
            new_clusters.append(temp)
        else:
            raise TypeError("'merge_info' muse be a list with elements that are either int or list of int.")

    return new_clusters


if __name__=="__main__":
    """
    Main function for testing and debugging only.
    """
    cluster_file_dir = '../data/googlenet-classes'
    pocket_dir = '../data/googlenet-dataset/'
    pop_dir = '../data/pops-googlenet/'    
    num_classes = 10
    cluster_th = 10000
    features_to_use = ['x', 'y', 'z', 'charge', 'hydrophobicity', 'binding_probability', 'r', 'theta', 'phi', 'sequence_entropy']
    batch_size = 1 # number of pairs in a mini-batch

    train_pos_th = 3000 # threshold of number of positive train pairs for each class
    train_neg_th = 100 # threshold of number of negative train pairs for each combination
    val_pos_th = 1000 # threshold of number of positive validation pairs for each class
    val_neg_th = 25 # threshold of number of negative validation pairs for each combination
    print('positive training pair sampling threshold: ', train_pos_th)
    print('negative training pair sampling threshold: ', train_neg_th)
    print('positive validation pair sampling threshold: ', val_pos_th)
    print('negative validation pair sampling threshold: ', val_neg_th)


    subcluster_file = './pocket_cluster_analysis/results/subclusters.yaml'
    with open(subcluster_file) as file:
        subcluster_dict = yaml.full_load(file)

    train_pos_pairs, train_neg_pairs, val_pos_pairs, val_neg_pairs = divide_and_gen_pairs(cluster_file_dir=cluster_file_dir, 
                                                                                          subcluster_dict = subcluster_dict,
                                                                                          num_classes=num_classes, 
                                                                                          cluster_th=cluster_th,
                                                                                          train_pos_th=train_pos_th,
                                                                                          train_neg_th=train_neg_th,
                                                                                          val_pos_th=val_pos_th,
                                                                                          val_neg_th=val_neg_th)
    
    print('number of classes:', num_classes)
    print('max number of data of each class:', cluster_th)
    print('number of train positive pairs:', len(train_pos_pairs))
    print('number of train negative pairs:', len(train_neg_pairs))
    print('number of validation positive pairs:', len(val_pos_pairs))
    print('number of validation negative pairs:', len(val_neg_pairs))

    train_loader, val_loader = dataloader_gen(pocket_dir, 
                                              pop_dir,
                                              train_pos_pairs, 
                                              train_neg_pairs, 
                                              val_pos_pairs, 
                                              val_neg_pairs, 
                                              features_to_use, 
                                              batch_size, 
                                              shuffle=False,
                                              num_workers=1)

    for data in train_loader:
        break
