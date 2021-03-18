"""
Compute the statistics of the dataset.
"""
import yaml
import torch
from dataloader import read_cluster_file_from_yaml
from dataloader import merge_clusters
from dataloader import pocket_loader_gen
from tqdm import tqdm
from torch_geometric.utils import to_networkx
from networkx.classes.function import info
import statistics 
import numpy as np
import networkx as nx


def compute_graph_stat(graph):
    """
    Compute the following statistics of graph:
        1. number of nodes.
        2. number of edges.
        4. number of isolated nodes.
        5. diameter of graph.
        6. density of graph.
    """
    graph = to_networkx(graph)
    graph = graph.to_undirected()
    graph_stat = {}
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()
    dens = nx.classes.function.density(graph)
    try:
        dia = nx.algorithms.distance_measures.diameter(graph) # can be inf when graph is not connected
    except:
        dia = float('inf')
    graph_stat['num_nodes'] = num_nodes
    graph_stat['num_edges'] = num_edges
    graph_stat['density'] = dens
    graph_stat['diameter'] = dia

    degrees = list(graph.degree())
    degrees = [x[1] for x in degrees]
    sum_of_edges = sum(degrees)
    avg_degree = sum_of_edges/num_nodes
    graph_stat['avg_degree'] = avg_degree

    #print('average degree:', avg_degree)
    #print(info(graph))
    #print('density of graph:', dens)
    #print('diameter of graph:', dia)
    return graph_stat


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # detect cpu or gpu
    
    with open('./train_classifier.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)  

    cluster_file_dir = config['cluster_file_dir']
    pocket_dir = config['pocket_dir']
    pop_dir = config['pop_dir']
    merge_info = config['merge_info']
    features_to_use = config['features_to_use']

    # pocket clusters
    clusters = read_cluster_file_from_yaml(cluster_file_dir)
    clusters = merge_clusters(clusters, merge_info)
    num_classes = len(clusters)
    print('number of classes after merging: ', num_classes)   

    # create dataloader without splitting
    dataloader, data_size, dataset = pocket_loader_gen(pocket_dir=pocket_dir, 
                                                       pop_dir=pop_dir,
                                                       clusters=clusters, 
                                                       features_to_use=features_to_use, 
                                                       batch_size=1, 
                                                       shuffle=False, 
                                                       num_workers=1)


    # measurements
    num_nodes = []
    num_edges = []
    density = []
    diameter = []
    avg_degree = []

    # collect information
    for graph in tqdm(dataloader):
        graph_stat = compute_graph_stat(graph)
        num_nodes.append(graph_stat['num_nodes'])
        num_edges.append(graph_stat['num_edges'])
        density.append(graph_stat['density'])
        diameter.append(graph_stat['diameter'])
        avg_degree.append(graph_stat['avg_degree'])

    # results
    num_nodes_mean = statistics.mean(num_nodes) 
    num_edges_mean = statistics.mean(num_edges)
    density_mean = statistics.mean(density) 
    diameter_mean = statistics.mean(diameter)
    avg_degree_mean = statistics.mean(avg_degree)

    num_nodes_median = statistics.median(num_nodes) 
    num_edges_median = statistics.median(num_edges)
    density_median = statistics.median(density) 
    diameter_median = statistics.median(diameter)
    avg_degree_median = statistics.median(avg_degree)

    print('mean number of nodes:', num_nodes_mean)
    print('mean number of edges:', num_edges_mean)
    print('mean density:', density_mean)
    print('mean diameter:', diameter_mean)
    print('mean average degree:', avg_degree_mean)
    print('median number of nodes:', num_nodes_median)
    print('median number of edges:', num_edges_median)
    print('median density:', density_median)
    print('median diameter:', diameter_median)
    print('median average degree:', avg_degree_median)