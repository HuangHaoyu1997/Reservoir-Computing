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
                                activation=nn.ReLU(),)
        self.gconv2 = GraphConv(in_feats=config.egat_hid, 
                                out_feats=config.egat_out, 
                                norm='none', 
                                weight=True, 
                                bias=True,
                                activation=nn.ReLU(),)
        self.fc = nn.Linear(config.egat_out, config.N_out)
    def forward(self, g, node_feats):
        h = self.gconv1(g, node_feats) # h.shape [nodes, feats]
        h = self.gconv2(g, h)
        # h = h.mean(0) # average on node dim
        g.ndata['h'] = h.view(self.config.N_hid, -1)
        node_sum_vec = dgl.mean_nodes(g, 'h')
        # print(node_sum_vec.shape)
        out = self.fc(node_sum_vec)
        
class ConvNet(nn.Module):
    def __init__(self, config:Config):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7*7*32, config.N_hid)
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

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


class RCagent:
    def __init__(self) -> None:
        pass
    

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
