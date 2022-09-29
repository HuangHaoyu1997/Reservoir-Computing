import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import os, time, torch, pickle
from scipy.linalg import pinv
from utils import encoding, A_initial, activation, softmax
import torch
import torch.nn as nn

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class torchRC(nn.Module):
    '''
    Reservoir Computing Model in pytorch version
    '''
    def __init__(self,
                 N_input, # 输入维度
                 N_hidden, # reservoir神经元数量
                 N_output, # 输出维度
                 alpha, # memory factor
                 decay, # membrane potential decay factor
                 threshold, # firing threshold
                 R, # distance factor
                 p, # ratio of inhibitory neurons
                 gamma,
                 sub_thr, # when firing, subtract threshold to membrane potential
                 binary, # binary A matrix
                 ) -> None:
        super(torchRC, self).__init__()
        
        self.N_in = N_input
        self.N_hid = N_hidden
        self.N_out = N_output
        self.alpha = alpha
        self.decay = decay
        self.thr = threshold
        self.R = R
        self.p = p
        self.gamma = gamma
        self.random_init = True, # 初始状态是否随机初始化
        self.sub_thr = sub_thr
        self.binary = binary
        
        self.reset()
    
    def reset(self,):
        '''
        random initialization:
        W_in:      input weight matrix
        A:         reservoir weight matrix
        W_out:     readout weight matrix
        r_history: state of reservoir neurons
        mem:       membrane potential of reservoir neurons
        
        '''
        W_in = nn.Parameter(torch.rand(self.N_hid, self.N_in) * 0.2-0.1) # unif(-0.1, 0.1)
        A = nn.Parameter(A_initial(self.N_hid, self.R, self.p, self.gamma, self.binary))
        # zero element mask
        zero_mask = A==0
        bias = nn.Parameter(torch.rand(self.N_hid) * 2-1) # unif(-1,1)
        
        # 用系数0.0533缩放，以保证谱半径ρ(A)=1.0
        self.W_out = np.random.uniform(low=-0.0533*np.ones((self.N_out, self.N_hid)), 
                                       high=0.0533*np.ones((self.N_out, self.N_hid)))
        
        # 如果decay不是一个非零实数,则初始化为随机向量
        if not self.decay:
            self.decay = np.random.uniform(0.2, 1.0, size=(self.N_hid)) # np.random.rand(self.N_hid)

class RC:
    '''
    Reservoir Computing Model
    '''
    def __init__(self,
                 N_input, # 输入维度
                 N_hidden, # reservoir神经元数量
                 N_output, # 输出维度
                 alpha, # memory factor
                 decay, # membrane potential decay factor
                 threshold, # firing threshold
                 R, # distance factor
                 p, # 
                 gamma,
                 sub_thr, # when firing, subtract threshold to membrane potential
                 ) -> None:
        
        self.N_in = N_input
        self.N_hid = N_hidden
        self.N_out = N_output
        self.alpha = alpha
        self.decay = decay
        self.thr = threshold
        self.R = R
        self.p = p
        self.gamma = gamma
        self.random_init = True, # 初始状态是否随机初始化
        self.sub_thr = sub_thr
        
        self.reset()
    
    def reset(self,):
        '''
        random initialization:
        W_in:      input weight matrix
        A:         reservoir weight matrix
        W_out:     readout weight matrix
        r_history: state of reservoir neurons
        mem:       membrane potential of reservoir neurons
        
        '''
        self.W_in = np.random.uniform(-0.1, 0.1, size=(self.N_hid, self.N_in))
        self.A = A_initial(self.N_hid, self.R, self.p, self.gamma)
        self.bias = np.random.uniform(-1, 1, size=(self.N_hid))
        
        # 用系数0.0533缩放，以保证谱半径ρ(A)=1.0
        self.W_out = np.random.uniform(low=-0.0533*np.ones((self.N_out, self.N_hid)), 
                                       high=0.0533*np.ones((self.N_out, self.N_hid)))
        
        # 如果decay不是一个非零实数,则初始化为随机向量
        if not self.decay:
            self.decay = np.random.uniform(0.2, 1.0, size=(self.N_hid)) # np.random.rand(self.N_hid)
    
    def state_dict(self,):
        return {
            'W_in': self.W_in,
            'A': self.A,
            'bias': self.bias,
            'W_out': self.W_out,
            'N_input': self.N_in,
            'N_hidden': self.N_hid,
            'N_output': self.N_out,
            'alpha': self.alpha,
            'decay': self.decay,
            'threshold': self.thr,
            'R': self.R,
            'p': self.p,
        }
        
    def membrane(self, mem, x, spike):
        '''
        update membrane voltage for reservoir neurons
        
        mem   [batch, N_hid]
        x     [batch, N_hid]
        spike [batch, N_hid]
        '''
        # print(mem.shape, spike.shape, x.shape)
        batch = mem.shape[0]
        
        # homogeneous decay factor
        if isinstance(self.decay, float):
            if self.sub_thr:
                mem = mem * self.decay - self.thr * (1-spike) + x # 
            else:
                mem = mem * self.decay * (1-spike) + x
        
        # heterogeneous decay factor
        else:
            decay = np.array([self.decay for _ in range(batch)])
            if self.sub_thr:
                mem = mem * decay - self.thr * (1-spike) + x # 
            else:
                mem = mem * decay * (1-spike) + x
                
        spike = np.array(mem>self.thr, dtype=np.float32)
        return mem, spike
    
    def forward_(self, x):
        '''
        inference function of spiking version
        
        x: input vector
        
        return
        
        mems: [frames, batch, N_hid]
        
        spike_train: [frames, batch, N_hid]
        
        '''
        batch = x.shape[0]
        frames = x.shape[1]
        spike_train = []
        spike = np.zeros((batch, self.N_hid), dtype=np.float32)
        mem = np.random.uniform(0, 0.2, size=(batch, self.N_hid))
        # mem = np.zeros((batch, self.N_hid), dtype=np.float32)
        bias = np.array([self.bias for _ in range(batch)])
        mems = []
        for t in range(frames):
            # x[:,t,:].shape (batch, N_in)
            U = np.matmul(x[:,t,:], self.W_in.T) # (batch, N_hid)
            r = np.matmul(spike, self.A) # (batch, N_hid)
            y = self.alpha * r + (1-self.alpha) * (U + bias)
            y = activation(y)
            mem, spike = self.membrane(mem, y, spike)
            spike_train.append(spike)
            mems.append(mem)
        
        return np.array(mems), np.array(spike_train)
    
    def forward(self, x):
        '''
        一个样本的长度应该超过1,即由多帧动态数据构成
        
        r.shape (batch, N_hid)
        
        y.shape (batch, N_out)
        
        spike_train.shape (frame, batch, N_hid)
        
        '''
        assert x.shape[1]>1
        
        batch = x.shape[0]
        frames = x.shape[1]
        
        spike_train = []
        spike = np.zeros((batch, self.N_hid), dtype=np.float32)
        r_history = np.zeros((batch, self.N_hid), dtype=np.float32)
        mem = np.zeros((batch, self.N_hid), dtype=np.float32)
        
        for t in range(frames):
            
            Ar = np.matmul(r_history, self.A) # (batch, N_hid)
            
            U = np.matmul(x[:,t,:], self.W_in.T) # (batch, N_hid)
            r = (1 - self.alpha) * r_history + self.alpha * activation(Ar + U)
            mem, spike = self.membrane(mem, r, spike)
            spike_train.append(spike)
            r_history = r
            
        y = np.matmul(r, self.W_out.T)
        # y = softmax(y)
        return r, y, np.array(spike_train)


class RCagent:
    def __init__(self) -> None:
        pass
    

if __name__ == '__main__':
    
    from data import MNIST_generation
    # ray.init()
    
    train_loader, test_loader = MNIST_generation(train_num=32,
                                                 test_num=250,
                                                 batch_size=16)
    
    model = RC(N_input=28*28,
               N_hidden=1000,
               N_output=10,
               alpha=0.8,
               decay=None, # None for random decay of neurons
               threshold=0.3,
               R=0.2,
               p=0.25,
               gamma=1.0,
               sub_thr=False
               )
    for i, (images, lables) in enumerate(train_loader):
        enc_img = encoding(images, frames=20)
        mems, spike_train = model.forward_(enc_img)
        # r, y, spike_train = model.forward(enc_img)
        firing_rate = spike_train.sum(0)/20
    
    
    plt.figure()
    for i in range(4):
        for j in range(4):
            plt.subplot(4,4,4*i+j+1)
            plt.hist(firing_rate[4*i+j,:])
    plt.show()
    
    # learn(model, train_loader, frames=10)
    
    # from cma import CMAEvolutionStrategy
    # es = CMAEvolutionStrategy(x0=np.zeros((model.N_hid*model.N_out)),
    #                             sigma0=0.5,
    #                             #   inopts={
    #                             #             'popsize':100,
    #                             #           },
    #                             )
    # N_gen = 100
    # for g in range(N_gen):
    #     solutions = es.ask()
    #     task_list = [train.remote(model,
    #                                 solution,
    #                                 train_loader, 
    #                                 test_loader, 
    #                                 batch_size=1,
    #                                 frames=10,
    #                                 ) for solution in solutions]
    #     fitness = ray.get(task_list)
    #     es.tell(solutions, fitness)
    #     with open('ckpt_'+str(g)+'.pkl', 'wb') as f:
    #         pickle.dump([solutions, fitness], f)
    #     print(np.min(fitness))
    
    # labels = []
    # for i, (image, label) in enumerate(train_loader):
    #     label_ = torch.zeros(1, 10).scatter_(1, label.view(-1, 1), 1).squeeze().numpy()
    #     labels.append(label_)
    # labels = np.array(labels, dtype=np.float).T
    
    # with open('train_labels.pkl', 'rb') as f:
    #     labels = pickle.load( f)
        
    # with open('rs.pkl', 'rb') as f:
    #     R_T = pickle.load(f)
    
    # R = R_T.T
    # R_inv = pinv(R)
    # W_out = np.matmul(labels, R_inv)
    # print(W_out.shape)
    # model.W_out = W_out
    
    # correct = 0
    # for i, (image, label) in enumerate(train_loader):
    #     image = encoding(image.squeeze(), 10) # shape=(30,784)
    #     r, y, _ = model.forward(image)
    #     predict = y.argmin()
    #     correct += float(predict == label.item())
    # print(correct / len(train_loader))
    
    # rs = inference.remote(model, train_loader, 10)
    # rs = ray.get(rs)
    # 
    # print(rs.shape)
    # with open('rs.pkl', 'wb') as f:
    #     pickle.dump(rs, f)
    
    # y, spike_train = model.forward(data)
    # spike_train = np.array(spike_train)
    # print(y, spike_train.shape)
    # plt.imshow(spike_train[:,0:100])
    # plt.pause(10)
    # ray.shutdown() 
