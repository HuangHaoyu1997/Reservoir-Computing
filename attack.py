'''
每次打开都要记得更新
latest update: 2023年1月17日21:33:28

改变RC网络的拓扑结构, 找到影响网络性能的关键节点
注意，删除储备池中某个节点的全部连接时，也要把输入层对应位置mask

'''

from config import Config
import numpy as np
import torch
import random
from RC import torchRC
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from dgl.nn import EGATConv, EdgeWeightNorm, GraphConv
from dgl.utils import expand_as_pair
import time
from config import Config
torch.set_default_dtype(torch.float32)
import networkx as nx

from data import part_DATA
from utils import torchUniform, A_cluster, set_seed, act

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
        self.ff_in = nn.Linear(self.N_in, self.N_hid, bias=True)
        # self.W_in1 = nn.Parameter(torchUniform(-Win, Win, size=(self.N_in, self.N_hid))).to(self.device)
        self.ff_A = nn.Linear(self.N_hid, self.N_hid, bias=False)
        self.ff_A.weight = nn.Parameter(torch.tensor(A_cluster(config)))
        # self.A1 = nn.Parameter(torch.tensor(A_cluster(config))).to(self.device)
        # self.bias1 = nn.Parameter(torchUniform(-config.bias, config.bias, size=(self.N_hid))).to(self.device)
        
        self.fc = nn.Linear(self.N_hid, config.N_out)
        self.fc1 = nn.Linear(self.N_hid, 128)
        self.fc2 = nn.Linear(128, config.N_out)
        
        # self.W_in2 = torchUniform(-Win, Win, size=(self.N_hid, self.N_hid)).to(self.device)
        # self.A2 = torch.tensor(A_cluster(config)).to(self.device)
        # self.bias2 = torchUniform(-config.bias, config.bias, size=(self.N_hid)).to(self.device)
        
        # self.gelu = nn.GELU()
        # self.ff1 = nn.Linear(self.N_hid, self.N_hid)
        # self.ff2 = nn.Linear(self.N_hid, self.N_hid)
        
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
        r[0,:,0,:] = torchUniform(-self.mem_init, self.mem_init, size=(batch, self.N_hid)).to(device) # layer1
        a = torchUniform(-self.mem_init, self.mem_init, size=(batch, self.N_hid)).to(device)
        # r[1,:,0,:] = torchUniform(-self.mem_init, self.mem_init, size=(batch, self.N_hid)).to(device) # layer2
        
        for t in range(self.frames):
            # layer 1
            # U = torch.mm(x[:,t,:], self.W_in1) # (batch, N_hid)
            U = self.ff_in(x[:,t,:])
            # r_ = torch.mm(r[0,:,t,:], self.A1) # information from neighbors (batch, N_hid)
            # y = self.alpha * r[0,:,t,:] + (1-self.alpha) * act(r_ +  U + self.bias1)
            # r[0,:,t+1,:] = y
            
            # r_ = torch.mm(a, self.A1) # information from neighbors (batch, N_hid)
            r_ = self.ff_A(a)
            y = self.alpha * a + (1-self.alpha) * act(r_ + 0.1*U) # + self.bias1
            a = y
        out = F.relu(self.fc1(y))
        out = self.fc2(out)
        return out

class AttackGym:
    def __init__(self, config:Config):
        self.config = config
        
        self.reset()
        
    def reset(self,):
        config = self.config
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self.model = torchRC(config)
        self.T = 0
    
    def step(self, action):
        # reservoir attacking
        self.model.W_ins[0] = None
        
        obs, reward = rollout_attack(self.config, self.model)
        
        if self.T > self.config.episode_len:
            done = True
        else:
            done = False
        return obs, reward, done



def test():
    '''
    测试，把图中某个神经元移除，具体做法是屏蔽邻接矩阵中，该神经元的输入权重和输出权重
    对于随机输入，被屏蔽的神经元的膜电位模式与其他神经元，具有较为显著的差别。
    '''
    config.N_in = 10
    config.N_hid = 100
    config.frames = 13
    model = torchRC(config)
    A = model.As[0].detach().numpy()
    print(A.shape)
    G = nx.from_numpy_array(A, create_using=nx.DiGraph)

    model.As[0] = model.As[0].detach()
    model.As[0][:,15] = 0
    model.As[0][15,:] = 0

    model.As[0][:,17] = 0
    model.As[0][17,:] = 0

    out = model(torch.rand(12,config.frames, config.N_in))
    means = []
    vars = []
    for i in range(100):
        if i==15: continue
        elif i==17: continue
        means.append(out[0][:,:, i].mean().item())
        vars.append(out[0][:,:,i].std().item())
    print(np.mean(means))
    print(np.mean(vars))
    print(out[0][:,:,15].mean())
    print(out[0][:,:,15].std())
    print(out[0][:,:,17].mean())
    print(out[0][:,:,17].std())

def attack(model:torchRC,
           config,
           
           ):
    # attack
    model.W_ins[0]
    
    # rollout
    reward = rollout_attack(config, model)

if __name__ == '__main__':
    Config.N_hid = 100
    Config.batch_size = 20
    Config.train_num = 1000
    Config.test_num = 1000
    Config.frames = 10
    Config.lr = 1e-4
    Config.epoch = 200
    train_loader, test_loader = part_DATA(Config)
    model = AnnRC(Config).to(Config.device)

    A_before = model.ff_A.weight.cpu().detach().numpy()
    Win_before = model.ff_in.weight.cpu().detach().numpy()
    fcw_before = model.fc1.weight.cpu().detach().numpy()
    fcb_before = model.fc1.bias.cpu().detach().numpy()
    index = model.ff_A.weight==0
    plt.imshow(fcw_before)

    G = nx.from_numpy_matrix(A_before, create_using=nx.DiGraph)
    ecen = nx.degree_centrality(G)
    sorted_ecen = sorted(ecen.items(), key = lambda kv:(kv[1], kv[0]))
    node_list = [i[0] for i in sorted_ecen[-10:]]
    print(node_list)
    
    cost = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.lr)
    # torch.autograd.set_detect_anomaly(True)
    for e in range(Config.epoch):
        train_loss = 0
        train_correct = 0
        for i, (img, label) in enumerate(train_loader):
            batch = img.shape[0]
            x_enc = None
            for _ in range(Config.frames):
                spike = (img > torch.rand(img.size())).float()
                if x_enc is None: x_enc = spike
                else: x_enc = torch.cat((x_enc, spike), dim=1)
            x_enc = x_enc.view(batch, Config.frames, Config.N_in) # [batch, frames, N_in]
            out = model(x_enc.to(Config.device))    
            optimizer.zero_grad()
            loss = cost(out, label.to(Config.device))
            loss.backward()
            optimizer.step()
            train_loss += loss.cpu().item()
            with torch.no_grad():
                model.ff_A.weight[index] = 0
                for i in node_list:
                    model.ff_A.weight[i,:] = 0
                    model.ff_A.weight[:,i] = 0
            
            _, id = torch.max(out.data, 1)
            train_correct += torch.sum(id.cpu()==label.data)
        
        test_loss = 0
        test_correct = 0
        with torch.no_grad():
            model.ff_A.weight[index] = 0
            for i, (img, label) in enumerate(test_loader):
                batch = img.shape[0]
                x_enc = None
                for _ in range(Config.frames):
                    spike = (img > torch.rand(img.size())).float()
                    if x_enc is None: x_enc = spike
                    else: x_enc = torch.cat((x_enc, spike), dim=1)
                x_enc = x_enc.view(batch, Config.frames, Config.N_in) # [batch, frames, N_in]
                out = model(x_enc.to(Config.device))
                loss = cost(out, label.to(Config.device))
                test_loss += loss.cpu().item()
                
                _, id = torch.max(out.data, 1)
                test_correct += torch.sum(id.cpu()==label.data)
        print(e, 
            train_loss/(Config.train_num/Config.batch_size), 
            test_loss/(Config.test_num/Config.batch_size), 
            train_correct.item()/Config.train_num, 
            test_correct.item()/Config.test_num)
    A_after = model.ff_A.weight.cpu().detach().numpy()
    Win_after = model.ff_in.weight.cpu().detach().numpy()
    fcw_after = model.fc1.weight.cpu().detach().numpy()
    fcb_after = model.fc1.bias.cpu().detach().numpy()
    # plt.imshow(A_after_train)
    plt.imshow(fcw_before-fcw_after)