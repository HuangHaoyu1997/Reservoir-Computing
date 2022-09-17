import numpy as np
import torch
import matplotlib.pyplot as plt

def spectral_radius(M):
    '''
    计算矩阵的谱半径
    '''
    a, b = np.linalg.eig(M) #a为特征值集合，b为特征值向量
    return np.max(np.abs(a)) #返回谱半径

def softmax(x):
    print(x.shape)
    print(np.exp(x)/np.exp(x).sum(-1))
    return np.exp(x)/np.exp(x).sum(-1)
    
def encoding(image, frames):
    '''
    image pixel value控制的随机分布编码
    frames:动态帧长度
    return: [batch, frames, 784]
    '''
    num_img = image.shape[0]
    sample = []
    for _ in range(frames):
        img = (image > torch.rand(image.size())).float().reshape(num_img, -1)
        img = img.numpy()
        sample.append(img)
    samples = np.array(sample)
    samples = np.transpose(samples,[1,0,-1])
    
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

def A_initial(N_hid, R, p, gamma):
    '''
    initialize random weights for matrix A
    p: ratio of inhibitory neurons 抑制性
    R: distance factor
    gamma: shape factor of gamma distribution
    
    '''
    length = N_hid * N_hid
    V = allocation(X=10, Y=10, Z=10)
    A = np.zeros((N_hid, N_hid), dtype=np.float32)
    
    for i in range(N_hid):
        for j in range(N_hid):
            # p_distance与distance成反比
            p_distance = np.exp(-np.sqrt((V[i][0]-V[j][0])**2+
                                            (V[i][1]-V[j][1])**2+
                                            (V[i][2]-V[j][2])**2)*R
                                )
            # connection
            if np.random.rand() < p_distance:
                # 抑制性神经元
                if np.random.rand() < p:
                    A[i,j] = -np.random.gamma(gamma)
                # 兴奋性神经元
                else:
                    A[i,j] = np.random.gamma(gamma)
    
    # cancel the self-connection
    for i in range(N_hid):
        A[i,i] = 0.
    
    # weights = []
    # for _ in range(length):
    #     if np.random.rand() < self.p:
    #         weights.append(np.random.uniform(-1, 0))
    #     else:
    #         weights.append(np.random.uniform(0, 1))
    
    # weights = np.array(weights).reshape((self.N_hid, self.N_hid))
    return A

def activation(x):
    # return np.maximum(x, 0)
    # return 1/(1+np.exp(-x))
    return np.tanh(x)

def cross_entropy(p, q):
    '''
    CELoss = -∑ p(x)*log(q(x))
    '''
    return -(p * np.log(q)).sum()