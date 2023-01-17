# -*- coding: UTF-8 -*-
import warnings

from config import Config
warnings.filterwarnings("ignore")
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import os, time, torch, pickle
from scipy.linalg import pinv
import torch
import torch.nn as nn
from utils import *

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class Reservoir(nn.Module):
    '''reservoir layer'''
    
    def __init__(self, config:Config):
        '''
        input: x, spike
        output: y
        '''
        super(Reservoir, self).__init__()
        Win = config.Win
        b = config.bias
        N_in = config.N_in
        N_hid = config.N_hid
        device = config.device
        self.alpha = config.alpha
        
        self.W_in = torchUniform(-Win, Win, size=(N_in, N_hid)).to(device)
        self.A = torch.tensor(A_cluster(config)).to(device)
        self.bias = torchUniform(-b, b, size=(N_hid)).to(device) # unif(-1,1)
        
    def forward(self, x, spike):
        '''
        x: (batch, N_in)
        W_in: (N_in, N_hid)
        spike: (batch, )
        '''
        U = torch.mm(x, self.W_in) # (batch, N_hid)
        r = torch.mm(spike, self.A)
        y = act(self.alpha * r + (1 - self.alpha) * (U + self.bias))
        return y

class AnnRC(nn.Module):
    '''
    Artificial-Neuron Version of Reservoir Computing Model in pytorch implementation
    Multi-layer with serial structure
    TODO how to train self.ff1 and self.ff2?
    '''
    def __init__(self, config:Config) -> None:
        super(AnnRC, self).__init__()
        self.config = config
        self.N_in = config.N_in
        self.N_hid = config.N_hid
        self.mem_init = config.mem_init
        self.alpha = config.alpha
        self.frames = config.frames
        self.device = config.device
        self.layers = config.layers
        
        
        Win = config.Win
        self.W_in1 = torchUniform(-Win, Win, size=(self.N_in, self.N_hid)).to(self.device)
        self.A1 = torch.tensor(A_cluster(config)).to(self.device)
        self.bias1 = torchUniform(-config.bias, config.bias, size=(self.N_hid)).to(self.device)
        
        
        self.W_in2 = torchUniform(-Win, Win, size=(self.N_hid, self.N_hid)).to(self.device)
        self.A2 = torch.tensor(A_cluster(config)).to(self.device)
        self.bias2 = torchUniform(-config.bias, config.bias, size=(self.N_hid)).to(self.device)
        
        self.gelu = nn.GELU()
        self.ff1 = nn.Linear(self.N_hid, self.N_hid)
        self.ff2 = nn.Linear(self.N_hid, self.N_hid)
        
        # self.layernorm = nn.LayerNorm()
        set_seed(config)
    
    def forward(self, x):
        '''
        inference function of spiking version
        x: input tensor [batch, frames, N_in]
        
        return
        r:        [batch, frames, N_hid]
        '''
        batch = x.shape[0]
        device = self.device
        
        r = torch.zeros(self.layers, batch, self.frames+1, self.N_hid).to(device)
        r[0,:,0,:] = torchUniform(-self.mem_init, self.mem_init, size=(batch, self.N_hid)).to(device)
        r[1,:,0,:] = torchUniform(-self.mem_init, self.mem_init, size=(batch, self.N_hid)).to(device)
        
        for t in range(self.frames):
            # layer 1
            U = torch.mm(x[:,t,:], self.W_in1) # (batch, N_hid)
            r_ = torch.mm(r[0,:,t,:], self.A1) # information from neighbors (batch, N_hid)
            y = self.alpha * r_ + (1-self.alpha) * act(U + self.bias1)
            y = self.gelu(self.ff1(y)) + y
            r[0,:,t+1,:] = y
            
            # layer 2
            # change r[0,:,t+1,:] to x[:,t,:] for parallel version
            U = torch.mm(r[0,:,t+1,:], self.W_in2) 
            r_ = torch.mm(r[1,:,t,:], self.A2) # information from neighbors (batch, N_hid)
            y = self.alpha * r_ + (1-self.alpha) * act(U + self.bias2)
            y = self.gelu(self.ff2(y)) + y
            r[1,:,t+1,:] = y
            
        return r
    
        # if self.decay is not a non-negative real number, initialize it to random vector
        if not self.decay:
            self.decay = torchUniform(low=0.2, high=1.0, size=(1, self.N_hid)).to(self.device)
        return W_ins, As, Bias

class HybridRC(nn.Module):
    '''
    Reservoir Computing Model with multiple hybrid neuron models in pytorch version
    '''
    def __init__(self, config:Config) -> None:
        super(HybridRC, self).__init__()
        self.config = config
        self.N_in = config.N_in
        self.N_hid = config.N_hid
        self.mem_init = config.mem_init
        self.decay = config.LIF_decay
        self.thr = config.LIF_thr
        self.frames = config.frames
        self.device = config.device
        
        res1_config = deepcopy(config)
        self.res1 = Reservoir(res1_config)
        
        res2_config = deepcopy(config)
        # res2_config.N_in = res1_config.N_hid
        self.res2 = Reservoir(res2_config)
        
        # self.layernorm = nn.LayerNorm()
        set_seed(config)
        
    def membrane(self, mem, x, spike):
        '''
        update membrane voltage for reservoir neurons
        mem   [batch, N_hid]
        x     [batch, N_hid]
        spike [batch, N_hid]
        '''
        mem = mem * self.decay - self.thr * (1-spike) + x # 
        # mem = mem * self.decay * (1-spike) + x
        spike = torch.tensor(mem>self.thr, dtype=torch.float32)
        return mem, spike
    
    def forward(self, x):
        '''
        inference function of spiking version
        x: input tensor [batch, frames, N_in]
        
        return
        mems:        [batch, frames, N_hid]
        spike_train: [batch, frames, N_hid]
        '''
        batch = x.shape[0]
        layers = self.config.layers
        device = self.device
        
        spikes = torch.zeros(layers, batch, self.frames+1, self.N_hid).to(device) # including time=-1 initial spike vector
        mems = torch.zeros(layers, batch, self.frames+1, self.N_hid).to(device)
        mems[0,:,0,:] = torchUniform(-self.mem_init, self.mem_init, size=(batch, self.N_hid)).to(device) # layer 1 initial membrane potential
        mems[1,:,0,:] = torchUniform(-self.mem_init, self.mem_init, size=(batch, self.N_hid)).to(device) # layer 2
        
        
        for t in range(self.frames):
            # layer 1
            y = self.res1(x[:,t,:], spikes[0,:,t,:])
            mem, spike = self.membrane(mems[0,:,t,:], y, spikes[0,:,t,:])
            mems[0,:,t+1,:] = mem
            spikes[0,:,t+1,:] = spike
            
            # layer 2
            y = self.res2(x[:,t,:], spikes[1,:,t,:])
            mem, spike = self.membrane(mems[1,:,t,:], y, spikes[1,:,t,:])
            mems[1,:,t+1,:] = mem
            spikes[1,:,t+1,:] = spike
            
        return mems, spikes
    
    def reset(self, config:Config):
        '''
        random initialization:
        W_in:      input weight matrix
        A:         reservoir weight matrix
        W_out:     readout weight matrix
        r_history: state of reservoir neurons
        mem:       membrane potential of reservoir neurons
        
        '''
        assert len(config.type) == config.layer
        W_ins, As, Bias = [], [], []
        for i in range(config.layer):
            if i == 0: # first layer, input dim -> hidden dim
                # W_in = nn.Parameter(torchUniform(-0.1, 0.1, size=(self.N_in, self.N_hid))).to(self.device) # unif(-0.1, 0.1)
                W_in = torchUniform(-0.1, 0.1, size=(self.N_in, self.N_hid)).to(self.device) # unif(-0.1, 0.1)
            else:      # second layer, hidden dim -> hidden dim
                # W_in = nn.Parameter(torchUniform(-0.1, 0.1, size=(self.N_hid, self.N_hid))).to(self.device) # unif(-0.1, 0.1)
                W_in = torchUniform(-0.1, 0.1, size=(self.N_hid, self.N_hid)).to(self.device) # unif(-0.1, 0.1)
            # A = nn.Parameter(torch.tensor(A_cluster(self.N_hid,
            #                                         config.p_in,
            #                                         config.gamma,
            #                                         config.binary,
            #                                         config.type[i],
            #                                         config.noise,
            #                                         config.noise_str,
            #                                         config,
            #                                         ))).to(self.device)
            A = torch.tensor(A_cluster(self.N_hid, config.p_in, config.gamma, config.binary, config.type[i], config.noise,
                                       config.noise_str,config,)).to(self.device)
            
            # bias = nn.Parameter(torchUniform(-1, 1, size=(self.N_hid))).to(self.device) # unif(-1,1)
            bias = torchUniform(-1, 1, size=(self.N_hid)).to(self.device) # unif(-1,1)
            
            W_ins.append(W_in)
            As.append(A)
            Bias.append(bias)
            
        # if self.decay is not a non-negative real number, initialize it to random vector
        if not self.decay:
            self.decay = torchUniform(low=0.2, high=1.0, size=(1, self.N_hid)).to(self.device)
        return W_ins, As, Bias


class MultiRC(nn.Module):
    '''
    Multi-layer Reservoir Computing Model in pytorch version
    '''
    def __init__(self, config:Config) -> None:
        super(MultiRC, self).__init__()
        self.config = config
        self.N_in = config.N_in
        self.N_hid = config.N_hid
        self.mem_init = config.mem_init
        self.decay = config.LIF_decay
        self.thr = config.LIF_thr
        self.frames = config.frames
        self.device = config.device
        
        res1_config = deepcopy(config)
        self.res1 = Reservoir(res1_config).to(self.device)
        
        res2_config = deepcopy(config)
        # res2_config.N_in = res1_config.N_hid
        self.res2 = Reservoir(res2_config).to(self.device)
        
        # self.layernorm = nn.LayerNorm()
        set_seed(config)
        
    def membrane(self, mem, x, spike):
        '''
        update membrane voltage for reservoir neurons
        mem   [batch, N_hid]
        x     [batch, N_hid]
        spike [batch, N_hid]
        '''
        mem = mem * self.decay - self.thr * (spike) + x # 
        # mem = mem * self.decay * (1-spike) + x
        spike = torch.tensor(mem>self.thr, dtype=torch.float32)
        return mem, spike
    
    def forward(self, x):
        '''
        inference function of spiking version
        x: input tensor [batch, frames, N_in]
        
        return
        mems:        [batch, frames, N_hid]
        spike_train: [batch, frames, N_hid]
        '''
        batch = x.shape[0]
        layers = self.config.layers
        device = self.device
        
        spikes = torch.zeros(layers, batch, self.frames+1, self.N_hid).to(device) # including time=-1 initial spike vector
        mems = torch.zeros(layers, batch, self.frames+1, self.N_hid).to(device)
        mems[0,:,0,:] = torchUniform(-self.mem_init, self.mem_init, size=(batch, self.N_hid)).to(device) # layer 1 initial membrane potential
        mems[1,:,0,:] = torchUniform(-self.mem_init, self.mem_init, size=(batch, self.N_hid)).to(device) # layer 2
        
        
        for t in range(self.frames):
            # layer 1
            y = self.res1(x[:,t,:], spikes[0,:,t,:])
            mem, spike = self.membrane(mems[0,:,t,:], y, spikes[0,:,t,:])
            mems[0,:,t+1,:] = mem
            spikes[0,:,t+1,:] = spike
            
            # layer 2
            y = self.res2(x[:,t,:], spikes[1,:,t,:])
            mem, spike = self.membrane(mems[1,:,t,:], y, spikes[1,:,t,:])
            mems[1,:,t+1,:] = mem
            spikes[1,:,t+1,:] = spike
            
        return mems, spikes
    

class torchRC(nn.Module):
    '''
    Reservoir Computing Model in pytorch version
    for single layer version
    '''
    def __init__(self, config:Config) -> None:
        super(torchRC, self).__init__()
        self.config = config
        self.N_in = config.N_in
        self.N_hid = config.N_hid
        self.mem_init = config.mem_init
        self.decay = config.LIF_decay
        # if self.decay is not a non-negative real number, initialize it to random vector
        if not self.decay:
            self.decay = torchUniform(low=0.2, high=1.0, size=(1, self.N_hid)).to(config.device)
            
        self.thr = config.LIF_thr
        self.R = config.R
        self.sub_thr = config.sub_thr
        self.frames = config.frames
        self.device = config.device
        
        # self.W_ins, self.As, self.Bias = self.reset(config)
        self.reservoir = Reservoir(config).to(self.device)
        set_seed(config)
        
    def membrane(self, mem, x, spike):
        '''
        update membrane voltage for reservoir neurons
        mem   [batch, N_hid]
        x     [batch, N_hid]
        spike [batch, N_hid]
        '''
        mem = mem * self.decay - self.thr * (spike) + x # 
        # mem = mem * self.decay * (1-spike) + x
        spike = torch.tensor(mem>self.thr, dtype=torch.float32)
        return mem, spike
    
    def forward(self, x):
        '''
        inference function of spiking version
        x: input tensor [batch, frames, N_in]
        
        return
        mems:        [batch, frames, N_hid]
        spike_train: [batch, frames, N_hid]
        '''
        batch = x.shape[0]
        layers = 1 # self.config.layers
        device = self.device
        
        spikes = torch.zeros(layers, batch, self.frames+1, self.N_hid).to(device) # including time=-1 initial spike vector
        mems = torch.zeros(layers, batch, self.frames+1, self.N_hid).to(device)
        mems[0,:,0,:] = torchUniform(-self.mem_init, self.mem_init, size=(batch, self.N_hid)).to(device) # layer 1 initial membrane potential
        
        for t in range(self.frames):
            y = self.reservoir(x[:,t,:], spikes[0,:,t,:])
            mem, spike = self.membrane(mems[0,:,t,:], y, spikes[0,:,t,:])
            mems[0,:,t+1,:] = mem
            spikes[0,:,t+1,:] = spike
            
        return mems[0], spikes[0]
    
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

import dgl
from dgl.nn import EGATConv, GraphConv
import torch.nn.functional as F

class EGAT(nn.Module):
    '''
    Graph attention layer that handles edge features
    https://docs.dgl.ai/generated/dgl.nn.pytorch.conv.EGATConv.html#dgl.nn.pytorch.conv.EGATConv
    '''
    def __init__(self, config:Config):
        super(EGAT, self).__init__()
        self.config = config
        self.egat1 = EGATConv(in_node_feats=config.frames,
                              in_edge_feats=1, # edge特征=Aij
                              out_node_feats=config.egat_hid,
                              out_edge_feats=1,
                              num_heads=config.egat_heads)
        
        self.egat2 = EGATConv(in_node_feats=config.egat_hid*config.egat_heads,
                              in_edge_feats=1, # edge特征=Aij
                              out_node_feats=config.egat_out,
                              out_edge_feats=1,
                              num_heads=config.egat_heads)
        self.fc = nn.Linear(config.egat_out * config.egat_heads, config.N_out)
        
    def forward(self, g, node_feats, edge_feats):
        h, _ = self.egat1(g, node_feats, edge_feats) # h.shape [nodes, heads, feats]
        h = h.view(self.config.N_hid, -1)
        h = F.relu(h) # h.shape [nodes, heads*feats]
        h, _ = self.egat2(g, h, edge_feats)
        # h = h.mean(0) # average on node dim
        g.ndata['h'] = h.view(self.config.N_hid, -1)
        node_sum_vec = dgl.mean_nodes(g, 'h')
        # print(node_sum_vec.shape)
        out = self.fc(node_sum_vec)
        # 
        return out
        
class EGCN(nn.Module):
    '''
    Graph convolution network using edge weights
    '''
    def __init__(self, config:Config) -> None:
        super(EGCN, self).__init__()
        self.config = config
        self.gconv1 = GraphConv(in_feats=config.frames, 
                                out_feats=config.egat_hid, 
                                norm='none', 
                                weight=True, 
                                bias=True,
                                activation=nn.ReLU())
        self.gconv2 = GraphConv(in_feats=config.egat_hid, 
                                out_feats=config.egat_out, 
                                norm='none', 
                                weight=True, 
                                bias=True,
                                activation=nn.ReLU()) # nn.Softmax()
        
        self.gconv_single = GraphConv(in_feats=config.frames, 
                                out_feats=config.N_out, 
                                norm='none', 
                                weight=True, 
                                bias=True,
                                activation=nn.ReLU()) # nn.Softmax()
        
        self.fc = nn.Linear(config.egat_out, config.N_out)
    
    def forward(self, g, node_feats, edge_w):
        # h = self.gconv1(g, node_feats, edge_weight=edge_w) # h.shape [nodes, feats]
        # h = self.gconv2(g, h, edge_weight=edge_w)
        h = self.gconv_single(g, node_feats, edge_weight=edge_w)
        g.ndata['h'] = h
        node_sum_vec = dgl.mean_nodes(g, 'h')
        return node_sum_vec
        # print(node_sum_vec.shape)
        # out = self.fc(node_sum_vec)
        # return out

class ConvNet(nn.Module):
    def __init__(self, input_size, config:Config):
        super(ConvNet, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dim = self.fc_dim(input_size)
        self.fc = nn.Linear(self.dim, config.N_hid)
    
    def fc_dim(self, input_size):
        x = self.layer1(torch.rand(input_size))
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        return x.shape[1] * x.shape[2] * x.shape[3]
    
    def forward(self, x):
        x = self.layer1(x)
        # print('cnn',x.dtype)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = self.fc(x.view(-1, self.dim))
        # out = out.reshape(out.size(0), -1)
        # out = self.fc(out)
        return x

class MLP(nn.Module):
    def __init__(self,
                 N_in,
                 N_hid,
                 N_out,
                 ) -> None:
        super(MLP, self).__init__()
        
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(N_in, N_hid)
        self.fc2=nn.Linear(N_hid, N_out) 
    
    def forward(self, x):
        
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class Transformer(nn.Module):
    '''
    transformer encoder for reservoir readout layer
    '''
    def __init__(self, config:Config) -> None:
        super(Transformer, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=config.d_model, 
                                                   nhead=config.n_heads,
                                                   dim_feedforward=config.d_ff,)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.encoder_layer)
        self.avgpool = nn.AdaptiveAvgPool1d(output_size=1)
        self.fc = nn.Linear(config.N_hid, config.d_model)
    
    def forward(self, x):
        '''
        input x shape = [batch, channel, N_hid]
        channel = config.frames
        '''
        x = self.fc(x)                  # out shape [batch, channel, d_model]
        x = self.transformer_encoder(x) # out shape [batch, channel, d_model]
        x = self.avgpool(x)             # out shape [batch, channel, 1]
        x = torch.flatten(x, 1)         # out shape [batch, channel]
        return x

class RCagent:
    def __init__(self) -> None:
        pass
    

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.input_map = nn.Linear(input_size, 4 * hidden_size, bias=True)
        self.recurrent_map = nn.Linear(hidden_size, 4 * hidden_size, bias=False)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        for w in self.input_map.parameters():
            if w.dim() == 1:
                torch.nn.init.uniform_(w, -0.1, 0.1)
            else:
                torch.nn.init.xavier_uniform_(w)
        for w in self.recurrent_map.parameters():
            if w.dim() == 1:
                torch.nn.init.uniform_(w, -0.1, 0.1)
            else:
                torch.nn.init.orthogonal_(w)

    def forward(self, inputs, states):
        output_state, cell_state = states

        z = self.input_map(inputs) + self.recurrent_map(output_state)
        i, ig, fg, og = z.chunk(4, 1)

        input_activation = self.tanh(i)
        input_gate = self.sigmoid(ig)
        forget_gate = self.sigmoid(fg + 1.0)
        output_gate = self.sigmoid(og)

        new_cell = cell_state * forget_gate + input_activation * input_gate
        output_state = self.tanh(new_cell) * output_gate

        return output_state, new_cell


class LeCun(nn.Module):
    def __init__(self):
        super(LeCun, self).__init__()
        self.tanh = nn.Tanh()
    def forward(self, x):
        return 1.7159 * self.tanh(0.666 * x)

class CfcCell(nn.Module):
    def __init__(self, input_size, hidden_size, hparams:Config):
        super(CfcCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hparams = hparams
        self._no_gate = False
        self._no_gate = self.hparams.no_gate
        self._minimal = self.hparams.minimal

        if self.hparams.backbone_activation == "silu":
            backbone_activation = nn.SiLU
        elif self.hparams.backbone_activation == "relu":
            backbone_activation = nn.ReLU
        elif self.hparams.backbone_activation == "tanh":
            backbone_activation = nn.Tanh
        elif self.hparams.backbone_activation == "gelu":
            backbone_activation = nn.GELU
        elif self.hparams.backbone_activation == "lecun":
            backbone_activation = LeCun
        else:
            raise ValueError("Unknown activation")
        
        layer_list = [nn.Linear(input_size + hidden_size, self.hparams.backbone_units),
                      backbone_activation(),]
        
        for _ in range(1, self.hparams.backbone_layers):
            layer_list.append(nn.Linear(self.hparams.backbone_units, self.hparams.backbone_units))
            layer_list.append(backbone_activation())
            layer_list.append(torch.nn.Dropout(self.hparams.backbone_dr))
        
        self.backbone = nn.Sequential(*layer_list)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.ff1 = nn.Linear(self.hparams.backbone_units, hidden_size)
        if self._minimal:
            self.w_tau = torch.nn.Parameter(data=torch.zeros(1, self.hidden_size), requires_grad=True)
            self.A = torch.nn.Parameter(data=torch.ones(1, self.hidden_size), requires_grad=True)
        else:
            self.ff2 = nn.Linear(self.hparams.backbone_units, hidden_size)
            self.time_a = nn.Linear(self.hparams.backbone_units, hidden_size)
            self.time_b = nn.Linear(self.hparams.backbone_units, hidden_size)
        self.init_weights()

    def init_weights(self):
        init_gain = self.hparams.init
        for w in self.parameters():
            if w.dim() == 2:
                torch.nn.init.xavier_uniform_(w, gain=init_gain)

    def forward(self, input, hx, ts):
        batch_size = input.size(0)
        ts = ts.view(batch_size, 1)
        x = torch.cat([input, hx], 1)

        x = self.backbone(x)
        if self._minimal:
            # Solution
            ff1 = self.ff1(x)
            new_hidden = (
                -self.A
                * torch.exp(-ts * (torch.abs(self.w_tau) + torch.abs(ff1)))
                * ff1
                + self.A
            )
        else:
            # Cfc
            ff1 = self.tanh(self.ff1(x))
            ff2 = self.tanh(self.ff2(x))
            t_a = self.time_a(x)
            t_b = self.time_b(x)
            t_interp = self.sigmoid(t_a * ts + t_b)
            if self._no_gate:
                new_hidden = ff1 + t_interp * ff2
            else:
                new_hidden = ff1 * (1.0 - t_interp) + t_interp * ff2
        return new_hidden

class Cfc(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_size,
                 out_feature,
                 hparams:Config,
                 return_sequences=False,
                 use_mixed=False,
                 use_ltc=False,):
        super(Cfc, self).__init__()
        self.in_features = in_features
        self.hidden_size = hidden_size
        self.out_feature = out_feature
        self.return_sequences = return_sequences

        if use_ltc:
            self.rnn_cell = LTCCell(in_features, hidden_size)
        else:
            self.rnn_cell = CfcCell(in_features, hidden_size, hparams)
        self.use_mixed = use_mixed
        if self.use_mixed:
            self.lstm = LSTMCell(in_features, hidden_size)
        self.fc = nn.Linear(self.hidden_size, self.out_feature)

    def forward(self, x, timespans=None, mask=None):
        device = x.device
        batch_size = x.size(0)
        seq_len = x.size(1)
        true_in_features = x.size(2)
        h_state = torch.zeros((batch_size, self.hidden_size), device=device)
        if self.use_mixed:
            c_state = torch.zeros((batch_size, self.hidden_size), device=device)
        output_sequence = []
        if mask is not None:
            forwarded_output = torch.zeros((batch_size, self.out_feature), device=device)
            forwarded_input = torch.zeros((batch_size, true_in_features), device=device)
            time_since_update = torch.zeros((batch_size, true_in_features), device=device)
        for t in range(seq_len):
            inputs = x[:, t]
            ts = timespans[:, t].squeeze()
            if mask is not None:
                if mask.size(-1) == true_in_features:
                    forwarded_input = mask[:, t] * inputs + (1 - mask[:, t]) * forwarded_input
                    time_since_update = (ts.view(batch_size, 1) + time_since_update) * (1 - mask[:, t])
                else:
                    forwarded_input = inputs
                if (true_in_features * 2 < self.in_features and mask.size(-1) == true_in_features):
                    # we have 3x in-features
                    inputs = torch.cat((forwarded_input, time_since_update, mask[:, t]), dim=1)
                else:
                    # we have 2x in-feature
                    inputs = torch.cat((forwarded_input, mask[:, t]), dim=1)
            if self.use_mixed:
                h_state, c_state = self.lstm(inputs, (h_state, c_state))
            h_state = self.rnn_cell.forward(inputs, h_state, ts)
            if mask is not None:
                cur_mask, _ = torch.max(mask[:, t], dim=1)
                cur_mask = cur_mask.view(batch_size, 1)
                current_output = self.fc(h_state)
                forwarded_output = (
                    cur_mask * current_output + (1.0 - cur_mask) * forwarded_output
                )
            if self.return_sequences:
                output_sequence.append(self.fc(h_state))

        if self.return_sequences:
            readout = torch.stack(output_sequence, dim=1)
        elif mask is not None:
            readout = forwarded_output
        else:
            readout = self.fc(h_state)
        return readout


class LTCCell(nn.Module):
    def __init__(
        self,
        in_features,
        units,
        ode_unfolds=6,
        epsilon=1e-8,
    ):
        super(LTCCell, self).__init__()
        self.in_features = in_features
        self.units = units
        self._init_ranges = {
            "gleak": (0.001, 1.0),
            "vleak": (-0.2, 0.2),
            "cm": (0.4, 0.6),
            "w": (0.001, 1.0),
            "sigma": (3, 8),
            "mu": (0.3, 0.8),
            "sensory_w": (0.001, 1.0),
            "sensory_sigma": (3, 8),
            "sensory_mu": (0.3, 0.8),
        }
        self._ode_unfolds = ode_unfolds
        self._epsilon = epsilon
        # self.softplus = nn.Softplus()
        self.softplus = nn.Identity()
        self._allocate_parameters()

    @property
    def state_size(self):
        return self.units

    @property
    def sensory_size(self):
        return self.in_features

    def add_weight(self, name, init_value):
        param = torch.nn.Parameter(init_value)
        self.register_parameter(name, param)
        return param

    def _get_init_value(self, shape, param_name):
        minval, maxval = self._init_ranges[param_name]
        if minval == maxval:
            return torch.ones(shape) * minval
        else:
            return torch.rand(*shape) * (maxval - minval) + minval

    def _erev_initializer(self, shape=None):
        return np.random.default_rng().choice([-1, 1], size=shape)

    def _allocate_parameters(self):
        self._params = {}
        self._params["gleak"] = self.add_weight(
            name="gleak", init_value=self._get_init_value((self.state_size,), "gleak")
        )
        self._params["vleak"] = self.add_weight(
            name="vleak", init_value=self._get_init_value((self.state_size,), "vleak")
        )
        self._params["cm"] = self.add_weight(
            name="cm", init_value=self._get_init_value((self.state_size,), "cm")
        )
        self._params["sigma"] = self.add_weight(
            name="sigma",
            init_value=self._get_init_value(
                (self.state_size, self.state_size), "sigma"
            ),
        )
        self._params["mu"] = self.add_weight(
            name="mu",
            init_value=self._get_init_value((self.state_size, self.state_size), "mu"),
        )
        self._params["w"] = self.add_weight(
            name="w",
            init_value=self._get_init_value((self.state_size, self.state_size), "w"),
        )
        self._params["erev"] = self.add_weight(
            name="erev",
            init_value=torch.Tensor(
                self._erev_initializer((self.state_size, self.state_size))
            ),
        )
        self._params["sensory_sigma"] = self.add_weight(
            name="sensory_sigma",
            init_value=self._get_init_value(
                (self.sensory_size, self.state_size), "sensory_sigma"
            ),
        )
        self._params["sensory_mu"] = self.add_weight(
            name="sensory_mu",
            init_value=self._get_init_value(
                (self.sensory_size, self.state_size), "sensory_mu"
            ),
        )
        self._params["sensory_w"] = self.add_weight(
            name="sensory_w",
            init_value=self._get_init_value(
                (self.sensory_size, self.state_size), "sensory_w"
            ),
        )
        self._params["sensory_erev"] = self.add_weight(
            name="sensory_erev",
            init_value=torch.Tensor(
                self._erev_initializer((self.sensory_size, self.state_size))
            ),
        )

        self._params["input_w"] = self.add_weight(
            name="input_w",
            init_value=torch.ones((self.sensory_size,)),
        )
        self._params["input_b"] = self.add_weight(
            name="input_b",
            init_value=torch.zeros((self.sensory_size,)),
        )


    def _sigmoid(self, v_pre, mu, sigma):
        v_pre = torch.unsqueeze(v_pre, -1)  # For broadcasting
        mues = v_pre - mu
        x = sigma * mues
        return torch.sigmoid(x)

    def _ode_solver(self, inputs, state, elapsed_time):
        v_pre = state

        # We can pre-compute the effects of the sensory neurons here
        sensory_w_activation = self.softplus(self._params["sensory_w"]) * self._sigmoid(
            inputs, self._params["sensory_mu"], self._params["sensory_sigma"]
        )

        sensory_rev_activation = sensory_w_activation * self._params["sensory_erev"]

        # Reduce over dimension 1 (=source sensory neurons)
        w_numerator_sensory = torch.sum(sensory_rev_activation, dim=1)
        w_denominator_sensory = torch.sum(sensory_w_activation, dim=1)

        # cm/t is loop invariant
        cm_t = self.softplus(self._params["cm"]).view(1, -1) / (
            (elapsed_time + 1) / self._ode_unfolds
        )

        # Unfold the multiply ODE multiple times into one RNN step
        for t in range(self._ode_unfolds):
            w_activation = self.softplus(self._params["w"]) * self._sigmoid(
                v_pre, self._params["mu"], self._params["sigma"]
            )

            rev_activation = w_activation * self._params["erev"]

            # Reduce over dimension 1 (=source neurons)
            w_numerator = torch.sum(rev_activation, dim=1) + w_numerator_sensory
            w_denominator = torch.sum(w_activation, dim=1) + w_denominator_sensory

            numerator = (
                cm_t * v_pre
                + self.softplus(self._params["gleak"]) * self._params["vleak"]
                + w_numerator
            )
            denominator = cm_t + self.softplus(self._params["gleak"]) + w_denominator
            # Avoid dividing by 0
            v_pre = numerator / (denominator + self._epsilon)
            if torch.any(torch.isnan(v_pre)):
                breakpoint()
        return v_pre

    def _map_inputs(self, inputs):
        inputs = inputs * self._params["input_w"]
        inputs = inputs + self._params["input_b"]
        return inputs

    def _map_outputs(self, state):
        output = state
        output = output * self._params["output_w"]
        output = output + self._params["output_b"]
        return output

    def _clip(self, w):
        return torch.nn.ReLU()(w)

    def apply_weight_constraints(self):
        self._params["w"].data = self._clip(self._params["w"].data)
        self._params["sensory_w"].data = self._clip(self._params["sensory_w"].data)
        self._params["cm"].data = self._clip(self._params["cm"].data)
        self._params["gleak"].data = self._clip(self._params["gleak"].data)

    def forward(self, input, hx, ts):
        # Regularly sampled mode (elapsed time = 1 second)
        ts = ts.view((-1, 1))
        inputs = self._map_inputs(input)
        next_state = self._ode_solver(inputs, hx, ts)
        # outputs = self._map_outputs(next_state)
        return next_state

if __name__ == '__main__':
    
    from data import Poisson_samples
    from config import Config as config
    import torch
    Rlayer = Reservoir(config).to(config.device)
    x = torch.rand(config.batch_size, config.N_in, dtype=torch.float32).to('cuda')
    spike = torch.rand(config.batch_size, config.N_hid, dtype=torch.float32).to('cuda')
    layernorm = nn.LayerNorm([50, 200]).to('cuda')
    y = Rlayer(x, spike)
    print(y, layernorm(y))
    
    # torch.set_default_dtype(torch.float32)
    # config.batch_size = 32
    # config.frames = 100
    # config.N_in = 50
    # config.device = 'cpu'
    # sample_pos = Poisson_samples(N_samples=config.batch_size,
    #                              N_in=config.N_in,
    #                              T=config.frames,
    #                              rate=10)
    # model = torchRC(config)
    # mems, spikes = model(sample_pos.to(config.device))
    # repre = torch.cat((mems, spikes), dim=-1)
    # print(repre.shape)
    
    
    
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
