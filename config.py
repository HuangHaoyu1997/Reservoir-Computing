class Config:
    N_in = 28*28      # input dim
    N_hid = 200       # hidden dim / number of reservoir neuron
    N_out = 10        # output dim
    alpha = 0.2       # memory factor
    decay = 0.5       # membrane potential decay factor
    thr = 0.5         # firing threshold
    R = 0.4           # distance factor when deciding connections
    p_in = 0.2        # ratio of inhibitory neurons
    gamma = 1.0       # shape factor of gamma distribution
    frames = 30       # static img to event-based frames
    device = 'cuda'   # 'cpu', 'cuda'
    batch_size = 2000 # batch size for inference and training
    sub_thr = False   # subtract thr to mem potential when firing
    binary = False    # binary matrix of reservoir A
    type = 'BA'      # type of A topology
                      # 'ER',  # Erdos-Renyi Random Network
                      # 'ERC', # Clusters with Erdos-Renyi Networks
                      # 'BA',  # Barabasi-Albert Network
                      # 'BAC', # Clusters with Barabasi-Albert networks
                      # 'DTW', # Developmental Time Window for multi-cluster small-world network
    
    noise = False      # add noise in A
    noise_str = 0.05  # noise strength
    
    p_ER = 0.2        # connection probability when creating ER graph
    m_BA = 2          # number of edges to attach from a new node to existing nodes
    k = 3             # number of clusters in A
    
    epoch = 200        # training epoch for readout layer mlp
    lr = 2e-3         # learning rate for mlp