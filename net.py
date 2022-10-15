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
    

if __name__ == '__main__':
    