# -*- coding: UTF-8 -*-
import warnings
warnings.filterwarnings("ignore")

from config import Config
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
    '''
    reservoir layer
    '''
    
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

class MultiRC(nn.Module):
    '''
    Multi-layer Reservoir Computing Model in pytorch version
    '''
    def __init__(self, config:Config) -> None:
        super(MultiRC, self).__init__()
        set_seed(config)
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
        
        if not self.decay:
            self.decay = torchUniform(low=0.2, high=1.0, size=(1, self.N_hid)).to(self.device)
        
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
    


if __name__ == '__main__':
    
    from data import Poisson_samples
    from config import Config as config
    import torch
    
    data = Poisson_samples(32, config.N_in, config.frames, rate=10)
    print(data.shape)
    model = MultiRC(config)
    output = model(data)
    print(output[0].shape)
    
