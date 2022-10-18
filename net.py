import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import math

def developmental_time_window(N, k, gamma, ):
    '''
    implementation of developmental time window
    to generate multiple-cluster small-world networks
    
    Nisbach F, Kaiser M. Developmental time windows for spatial growth generate multiple-cluster small-world networks[J]. The European Physical Journal B, 2007, 58(2): 185-191.
    
    N: total number of nodes in network
    k: number of pioneer nodes
    gamma(beta): parameter of distance function
    
    '''
    G = nx.Graph()
    
    # add pioneer nodes to graph
    for i in range(k):
        G.add_node(i)
    G
    

def ErdosRenyi(N, p):
    '''
    generate erdos-renyi random network
    N: number of nodes
    p: probability of edge creation
    return binary adjacency matrix
    '''
    # H = nx.fast_gnp_random_graph(N, p)
    H = nx.erdos_renyi_graph(N, p)
    A = nx.to_numpy_matrix(H)
    return A

def BarabasiAlbert(N, m):
    '''
    generate barabasi albert scale-free network
    
    N: number of nodes
    
    m: number of edges to attach from a new node to existing nodes
    
    return binary adjacency matrix
    '''
    H = nx.barabasi_albert_graph(N, m)
    A = nx.to_numpy_matrix(H)
    return A

if __name__ == '__main__':
    pass