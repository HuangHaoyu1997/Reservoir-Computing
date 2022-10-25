import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import math


def TimeWindowFunction(t, i, k, omega):
    '''
    P_time(t=0) = P_time(t=1) = 0
    P_time(t=mu^i) = 1
    ∫_0^1 P_time(t) dt = omega
    '''
    mu = i / (k+1)
    lamda = -math.log(2, mu) # -np.log2(2)/np.log2(mu), 换底公式
    return (16 * t**(2*lamda) * ((t**lamda-1)**2))**(omega/mu)

def DistanceFunction(A, B, beta=1, gamma=1):
    '''
    two node A B
    beta := gamma to ensure the integration be 1
    smaller beta (and gamma) indicates higher probability of larger distance
    
    return probability
    '''
    distance = np.sqrt(np.sum((A-B)**2))
    return beta * np.exp(-gamma * distance)

def Developmental_Time_Window(N, k, beta, R, r, p_self, omega):
    '''
    implementation of developmental time window
    to generate multiple-cluster small-world networks
    
    Nisbach F, Kaiser M. Developmental time windows for spatial growth generate multiple-cluster small-world networks[J]. The European Physical Journal B, 2007, 58(2): 185-191.
    
    N: total number of nodes in network
    k: number of pioneer nodes
    beta(gamma): parameter of distance function
    R:
    r:
    p_self:
    
    
    算法目前问题: 
    早期节点(0,1,2,...)的引入非常耗时,后随着节点数量增加而逐渐加快
    应该使初始节点数量增加至10个左右,给其随机分配
    '''
    
    A = np.zeros((N, N), dtype=np.float32) # adjacency matrix
    
    theta = np.random.uniform(0,1)
    pioneer_nodes = np.array([[r*np.cos(2*np.pi*(theta + i/k)), r*np.sin(2*np.pi*(theta + i/k))] for i in range(k)])
    
    # first node in A
    A[0, 0] = 1 if np.random.rand()<p_self else 0 # self-connection 
    N_coordinates = [np.random.uniform(-1, 1, size=2)]
    
    # add nodes
    i = 1
    t = 1 / N # timestep [0, 1]
    while i < N:
        # STEP1: random coordinates of new nodes
        U_theta = np.random.uniform(0,1)
        l = np.random.uniform(0, R)
        xyi = np.array([l*np.cos(2*np.pi*U_theta), l*np.sin(2*np.pi*U_theta)])
        
        # STEP2: associated to nearest pioneer node with same time window w(U)
        dis_pioneer = np.sqrt(np.sum((pioneer_nodes - xyi)**2, 1))
        nearest_i = np.argmin(dis_pioneer) + 1
        P_time_U = TimeWindowFunction(t, nearest_i, k, omega)
        
        # STEP3: decide each edge with former nodes
        edge_sum = 0
        for j in range(0, i):
            xyj = N_coordinates[j]
            P_dist = DistanceFunction(xyi, xyj, beta=beta, gamma=beta) # P distance between U and V
            
            # associated to nearest pioneer node with same time window w(V)
            dis_pioneer = np.sqrt(np.sum((pioneer_nodes - xyj)**2, 1))
            nearest_j = np.argmin(dis_pioneer) + 1
            P_time_V = TimeWindowFunction(t, nearest_j, k, omega)
            print(i, P_dist , P_time_U , P_time_V)
            if np.random.rand() < P_dist * P_time_U * P_time_V:
                edge_sum += 1
                A[i, j] = 1
                A[j, i] = 1
        
        # STEP4: if no edge to any existing nodes can be established, a new node will be resampled.
        if edge_sum > 0: 
            # self-connection
            if np.random.rand() < p_self: A[i, i] = 1
            t += 1/N
            i += 1 # add next node
            N_coordinates.append(xyi)
    return A


def ErdosRenyi(N, p):
    '''
    generate erdos-renyi random network
    N: number of nodes
    p: probability of edge creation
    return binary adjacency matrix
    '''
    # H = nx.fast_gnp_random_graph(N, p)
    H = nx.erdos_renyi_graph(N, p)
    A = nx.to_numpy_matrix(H, dtype=np.float32)
    return A

def BarabasiAlbert(N, m):
    '''
    generate barabasi albert scale-free network
    
    N: number of nodes
    
    m: number of edges to attach from a new node to existing nodes
    
    return binary adjacency matrix
    '''
    H = nx.barabasi_albert_graph(N, m)
    A = nx.to_numpy_matrix(H, dtype=np.float32)
    return A

def RandomNetwork(N_hid, R):
    '''
    initialize random weights for matrix A
    p: ratio of inhibitory neurons 抑制性
    R: distance factor
    gamma: shape factor of gamma distribution
    
    '''
    
    # random allocate 3D coordinate to all reservoir neurons
    # V = allocation(X=10, Y=10, Z=10)
    V = np.random.uniform(low=0, high=10, size=(N_hid, 3))
    A = np.zeros((N_hid, N_hid), dtype=np.float32)
    
    for i in range(N_hid):
        for j in range(N_hid):
            p_distance = np.exp(-np.sqrt((V[i][0]-V[j][0])**2+
                                         (V[i][1]-V[j][1])**2+
                                         (V[i][2]-V[j][2])**2)*R)
            if np.random.rand() < p_distance:
                A[i,j] = 1
    
    return A
if __name__ == '__main__':
    
    from config import Config
    A = Developmental_Time_Window(Config.N_hid,
                                Config.k,
                                Config.beta,
                                Config.R_,
                                Config.r,
                                Config.p_self,
                                Config.omega,)