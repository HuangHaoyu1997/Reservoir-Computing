import numpy as np
import torch
import matplotlib.pyplot as plt
from config import Config
from net import ErdosRenyi, BarabasiAlbert, Developmental_Time_Window, RandomNetwork, WattsStrogatz
import networkx as nx

def set_seed(config:Config):
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_default_dtype(torch.float32)

def torchUniform(low, high, size):
    '''
    return pytorch uniform ranging [low, high]
    '''
    tmp = torch.rand(size)
    tmp *= (high - low)
    tmp += low
    return tmp

def spectral_radius(M):
    '''
    计算矩阵的谱半径
    '''
    a, b = np.linalg.eig(M) #a为特征值集合，b为特征值向量
    return np.max(np.abs(a)) #返回谱半径

def softmax(x):
    # print(x.shape)
    # print(np.exp(x)/np.exp(x).sum(-1))
    return np.exp(x)/np.exp(x).sum(-1)
    
def encoding(image, frames):
    '''
    image pixel value控制的随机分布编码
    frames:动态帧长度
    return: [batch, frames, N_in]
    '''
    batch = image.shape[0]
    sample = []
    for _ in range(frames):
        img = (image > torch.rand(image.size())).float().reshape(batch, -1)
        img = img.numpy()
        
        sample.append(img)
    samples = np.array(sample) # [frames, batch, N_in]
    samples = np.transpose(samples,[1,0,-1]) # [batch, frames, N_in]
    
    return samples

def allocation(X, Y, Z):
    '''
    randomly allocate coordinates for reservoir neurons
    N_hid = X*Y*Z
    '''
    V = np.zeros((X, Y, Z), [('x', float), ('y', float), ('z', float)])
    V['x'], V['y'], V['z'] = np.meshgrid(np.linspace(0, Y - 1, Y), 
                                         np.linspace(0, X - 1, X),
                                         np.linspace(0, Z - 1, Z))
    V = V.reshape(X * Y * Z)
    np.random.shuffle(V)
    return V

def A_cluster(config:Config):
    '''
    generate A with multiple topologies
    
    N_hid: number of neurons in reservoir
    p: ratio of inhibitory neyrons
    gamma: shape factor of gamma distribution in A weights
    binary: binary matrix A
    net_type: ['ER',  # Erdos-Renyi Random Network
               'ERC', # Clusters with Erdos-Renyi Networks
               'BA',  # Barabasi-Albert Network
               'BAC', # Clusters with Barabasi-Albert networks
               'DTW', # Developmental Time Window for multi-cluster small-world network
               'RAN', # random network
               'WS',  # Watts Strogatz small world networks
               'WSC', # Clusters of Watts Strogatz small world networks
               ]
    noise: add noise or not
    noise_strength: probability of creating a noise connection
    kwargs:[
        'k',
        'p_ER',
        'm_BA',
        
    ]
    k: number of clusters
    
    '''
    N_hid = config.N_hid
    p_in = config.p_in
    gamma = config.gamma
    binary = config.binary
    net_type = config.net_type
    noise = config.noise
    noise_strength = config.noise_str
    N_cluster = config.k
    
    if net_type == 'ER':
        A = ErdosRenyi(N_hid, config.p_ER)

    elif net_type == 'BA':
        A = BarabasiAlbert(N_hid, config.m_BA)
    
    elif net_type == 'WS':
        A = WattsStrogatz(N_hid, config.p_ER, config.m_BA)

    elif net_type == 'WSC':
        A = np.zeros((N_hid, N_hid), dtype=np.float32)
        npc = int(N_hid / N_cluster) # number of nodes per cluster
        for k in range(N_cluster):
            WS = WattsStrogatz(npc, config.p_ER, config.m_BA)
            A[k*npc:(k+1)*npc, k*npc:(k+1)*npc] = WS
            
    elif net_type == 'ERC':
        A = np.zeros((N_hid, N_hid), dtype=np.float32)
        npc = int(N_hid / N_cluster) # number of nodes per cluster
        for k in range(N_cluster):
            ER = ErdosRenyi(npc, config.p_ER)
            A[k*npc:(k+1)*npc, k*npc:(k+1)*npc] = ER
        
    elif net_type == 'BAC':
        A = np.zeros((N_hid, N_hid), dtype=np.float32)
        npc = int(N_hid / N_cluster) # number of nodes per cluster
        for k in range(N_cluster):
            BA = BarabasiAlbert(npc, config.m_BA)
            A[k*npc:(k+1)*npc, k*npc:(k+1)*npc] = BA
    
    elif net_type == 'DTW':
        A = Developmental_Time_Window(N_hid, 
                                      N_cluster, 
                                      config.beta,
                                      config.R_,
                                      config.r,
                                      config.p_self,
                                      config.omega)
    elif net_type == 'RAN':
        A = RandomNetwork(N_hid, config.R)

    # add inhibitory synapses
    for i in range(N_hid):
        for j in range(N_hid):
            if A[i,j] != 0:
                if np.random.rand() <= p_in: # inhibitory
                    if binary: A[i,j] = -1
                    else:      A[i,j] = -np.random.gamma(gamma)
                else: # excitatory
                    if not binary: A[i,j] = np.random.gamma(gamma)
    
    # add noise
    if binary and noise:
        # only add noise to disconnected area
        zeros = np.array(A==0, dtype=np.float32)
        salt_noise = np.random.choice([-1,0,1], 
                                      size=(N_hid,N_hid), 
                                      p=[noise_strength,1-2*noise_strength,noise_strength])
        A += salt_noise * zeros
    
    if (not binary) and noise:
        tmp = np.random.uniform(low=-noise_strength, high=noise_strength, size=(N_hid, N_hid))
        idx = np.abs(tmp) < 0.8*noise_strength
        tmp[idx] = 0
        A += tmp
    if config.scale:
        A /= spectral_radius(A)
    return A


def act(x):
    '''
    pytorch activation function
    '''
    # return 1/(1+torch.exp(-x))
    return torch.tanh(x)

def activation(x):
    # return np.maximum(x, 0)
    # return 1/(1+np.exp(-x))
    # return np.tanh(x)
    return x

def cross_entropy(p, q):
    '''
    CELoss = -∑ p(x)*log(q(x))
    '''
    return -(p * np.log(q)).sum()

if __name__ == '__main__':
    A = A_cluster(N_hid=100,
                  p_in=0.2,
                  gamma=1.0,
                  binary=False,
                  type='ER',
                  p_ER=0.2,)
    print(A.shape)