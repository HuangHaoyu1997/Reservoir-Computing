from scipy.special import beta


class Config:
    seed = 123
    verbose = True
    
    
    data = 'cifar10'                                # 'mnist'
    train_num = 20000
    test_num = 10000
    epoch = 1000                                     # training epoch for readout layer mlp
    lr = 5e-3                                       # learning rate for mlp
    
    N_in = 32*32 if data=='cifar10' else 28*28      # input dim
    N_hid = 1000                                     # hidden dim / number of reservoir neuron
    N_out = 10                                      # output dim
    alpha = 0.2                                     # memory factor

    layer = 2         # number of reservoir layers
    p_in = 0.2        # ratio of inhibitory neurons
    gamma = 1.0       # shape factor of gamma distribution
    frames = 80       # static img to event-based frames
    device = 'cuda'   # 'cpu', 'cuda'
    batch_size = 500 # batch size for inference and training
    sub_thr = False   # subtract thr to mem potential when firing
    
    
    # topology settings
    binary = False    # binary matrix of reservoir A
    type = ['BAC', 'BAC']      # type of reservoir connection topology
                      # 'ER',  # Erdos-Renyi Random Network
                      # 'ERC', # Clusters with Erdos-Renyi Networks
                      # 'BA',  # Barabasi-Albert Network
                      # 'BAC', # Clusters with Barabasi-Albert networks
                      # 'DTW', # Developmental Time Window for multi-cluster small-world network
    scale = False     # rescale matrix A with spectral radius
    noise = True      # add noise in A
    noise_str = 0.05  # noise strength
    p_ER = 0.2        # connection probability when creating ER graph
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
    
    LIF_decay = 0.5       # LIF membrane potential decay factor
    LIF_thr = 0.7         # firing threshold