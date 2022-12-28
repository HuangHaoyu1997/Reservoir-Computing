class Config:
    seed = 123
    verbose = True
    
    data = 'mnist'                                # 'mnist', 'cifar10', 'poisson'
    
    if data == 'poisson':
        rate = [5, 50]                                 # firing rate of Poisson spike trains, for true and false samples
        train_num = 500
        test_num = 500
        N_in = 50                                   # input dim
        N_out = 2                                   # output dim
    elif data == 'cifar10':
        train_num = 20000
        test_num = 10000
        N_in = 32*32
        N_out = 10
    elif data == 'mnist':
        train_num = 5000
        test_num = 1000
        N_in = 28*28
        N_out = 10
    
    frames = 30       # static img to event-based frames
    episode_len = 50   # episode length when attacking reservoir model
    
    ##############################
    # MLP parameters
    ##############################
    mlp_hid = 32
    epoch = 300                                     # training epoch for readout layer mlp
    lr = 5e-5                                       # learning rate for mlp
    batch_size = 50                                # batch size for inference and training
    
    ##############################
    # transformer model parameters
    ##############################
    d_model = 32
    encoder_layer = 2
    n_heads = 4
    assert d_model % n_heads == 0
    d_ff = 64 # dim for feedforward linear layer
    
    ##############################
    # GNN model parameters
    ##############################
    egat_hid = 64     # hidden node feature dim of Edge GAT layer
    egat_out = 32     # output node feature dim of Edge GAT layer
    egat_heads = 3    # multi-head attention
    epoch_egat = 3000 # epoch for EGAT training
    lr_egat = 1e-3    # learning rate for mlp
    
    
    
    device = 'cuda:1'                                 # 'cpu', 'cuda'
    

    Win = 1.0         # strength of input linear weights, unif(-Win, Win)
    bias = 1          # bias distribution in reservoir layers
    mem_init = 0.1   # initial membrane potential distribution
    N_hid = 1000                                     # hidden dim / number of reservoir neuron
    alpha = 0.5                                     # memory factor
    
    p_in = 0.2        # ratio of inhibitory neurons
    gamma = 1.0       # shape factor of gamma distribution
    
    sub_thr = False   # subtract thr to mem potential when firing
    
    # topology settings
    binary = False    # binary matrix of reservoir A
    net_type = 'BAC'  # type of reservoir connection topology
                      # 'ER',  # Erdos-Renyi Random Network
                      # 'ERC', # Clusters of Erdos-Renyi Networks
                      # 'BA',  # Barabasi-Albert Network
                      # 'BAC', # Clusters of Barabasi-Albert networks
                      # 'WS',  # Watts Strogatz small world networks
                      # 'WSC', # Clusters of Watts Strogatz small world networks
                      # 'RAN', # random network
                      # 'DTW', # Developmental Time Window for multi-cluster small-world network
    
    layers = 2 # number of reservoir layers
    
    scale = False     # rescale matrix A with spectral radius
    noise = True      # add noise in A
    noise_str = 0.05  # noise strength
    p_ER = 0.2        # connection probability when creating edges, for ER and WS graphs
    m_BA = 3          # number of edges to attach from a new node to existing nodes
    k = 5             # number of clusters in A
    R = 0.2           # distance factor when deciding connections in random network
    
    R_ = 1.5          # radius of all nodes in DTW algorithm
    r = 1.2             # radius of pioneer nodes in DTW 
    omega = 0.2       # shape factor of time window function, bigger for wider shape
    p_self = 0.8      # self-connection in DTW algorithm
    beta = 1          # beta=gamma, in DTW distance probability function
    
    
    
    # dynamics
    neuro_type = 'LIF'# 'LIF', 'IZH', 'HH', 'Hybrid'
    Izh_a = 0.02      # [0.01, 0.15]
    Izh_b = 0.2       # [0.1, 0.3]
    Izh_c = -65       # [-70, -45]
    Izh_d = 8         # [0.02, 10]
    Izh_thr = 30      # [20, 40]
    
    LIF_decay = 0 # 0.5       # LIF membrane potential decay factor, 0 for random decay
    LIF_thr = 0.7         # firing threshold

if __name__ == '__main__':
    from copy import deepcopy
    a = Config()
    b = deepcopy(a)
    b.N_hid = 234
    print(a.N_hid, b.N_hid)
