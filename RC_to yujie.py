import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import time
from utils import A_cluster
from tqdm.auto import tqdm
from torchvision import transforms, datasets
from copy import deepcopy

class config:
    input = 700
    output = 20
    hid = 128    # RC Neurons
    thr = 0.5
    decay = 0.5
    rst = 0.05
    
    N_hid = hid
    p_in = 0.2        # ratio of inhibitory neurons
    gamma = 1.0       # shape factor of gamma distribution
    binary = True    # binary matrix of reservoir A
    net_type = 'BAC'  # type of reservoir connection topology
                      # 'ER',  # Erdos-Renyi Random Network
                      # 'ERC', # Clusters of Erdos-Renyi Networks
                      # 'BA',  # Barabasi-Albert Network
                      # 'BAC', # Clusters of Barabasi-Albert networks
                      # 'WS',  # Watts Strogatz small world networks
                      # 'WSC', # Clusters of Watts Strogatz small world networks
                      # 'RAN', # random network
                      # 'DTW', # Developmental Time Window for multi-cluster small-world network
    noise = True      # add noise in A
    noise_str = 0.05  # noise strength
    p_ER = 0.2        # connection probability when creating edges, for ER and WS graphs
    m_BA = 3          # number of edges to attach from a new node to existing nodes
    k = 5             # number of clusters in A
    R = 0.2           # distance factor when deciding connections in random network
    scale = False     # rescale matrix A with spectral radius
    
    input_learn = True # learnable input layer
    seed = 123
    trials = 5        # try on 5 different seeds
    num_minibatch = 10000
    num_per_label_minibatch = 20 # number of samples of each label in one mini-batch
    batch = 256
    epoch = 40
    lr = 0.005
    l1 = 0.0003
    l1_targ = 2000
    dropout = 0.7
    dropout_stepping = 0.01
    dropout_stop = 0.90
    weight_decay = 1e-4
    label_smoothing = False
    smoothing = 0.1
    norm = False      # add layer norm before each layer
    shortcut = False
    small_init = True
    device = torch.device('cuda')


from spikingjelly.datasets.shd import SpikingHeidelbergDigits
SHD_train = SpikingHeidelbergDigits('D:\Ph.D\Research\SNN-SRT数据\SHD', train=True, data_type='frame', frames_number=20, split_by='number')

# 数据增强
SHD_train_aug = []
for x, y in SHD_train:
    SHD_train_aug.append([x, y])
    for _ in range(2):
        noise = np.array(np.random.random(x.shape)>0.95, dtype=np.float)
        SHD_train_aug.append([x + noise, y])

SHD_test = SpikingHeidelbergDigits('D:\Ph.D\Research\SNN-SRT数据\SHD', train=False, data_type='frame', frames_number=20, split_by='number')

train_loader = torch.utils.data.DataLoader(dataset=SHD_train_aug, batch_size=config.batch, shuffle=True, drop_last=False, num_workers=0)
test_loader = torch.utils.data.DataLoader(dataset=SHD_test, batch_size=config.batch, shuffle=False, drop_last=False, num_workers=0)

class ActFun(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(0).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input - 0) < 0.5 # lens
        return grad_input * temp.float()

act_fun = ActFun.apply

def mem_update(input, mem, spk, thr, decay, rst):
    mem = rst * spk + mem * decay * (1-spk) + input
    spike = act_fun(mem - thr)
    return mem, spike

def RadLIF_update(input, mem, spk, thr, alpha, beta, a, b, W):
    mem_ = alpha * mem + (1-alpha) * (input-W) - thr * spk
    W_ = beta * W + (1-beta) * a * mem + b * spk
    spike = act_fun(mem_ - thr)
    return mem_, spike, W_

class RC(nn.Module):
    def __init__(self) -> None:
        super(RC, self).__init__()
        input = config.input
        hid = config.hid
        out = config.output
        self.fc_in = nn.Linear(input, hid)
        self.A1 = nn.Linear(hid, hid, bias=True) # random initialized adjacency matrix
        
        # 用于保证input强度大于lateral connection的强度，不然没法学到东西
        # 因此邻接矩阵的权重尽可能小
        # 0初始化也行
        if config.small_init:
            self.A1.weight.data = 0.9 * self.A1.weight.data
        self.fc_out = nn.Linear(hid, out)
        
        self.thr = nn.Parameter(torch.rand(config.hid)*config.thr, requires_grad=False)
        self.decay = nn.Parameter(torch.rand(config.hid)*config.decay, requires_grad=True)
        self.rst = nn.Parameter(torch.rand(config.hid)*config.rst, requires_grad=False)
        
        self.ln1 = nn.LayerNorm(hid)
        
        if not config.input_learn:
            for name, p in self.named_parameters():
                if 'conv' in name or 'fc_in' in name:
                    p.requires_grad = False
        
    def forward(self, input, mask, device='cuda', mode='parallel'):
        A1_mask, A2_mask, A3_mask, A4_mask = mask
        batch = input.shape[0]
        time_step = input.shape[1]
        
        hid1_mem = torch.zeros(batch, time_step+1, config.hid).uniform_(0, 0.1).to(device)
        hid1_spk = torch.zeros(batch, time_step+1, config.hid).to(device)
        sum1_spk = torch.zeros(batch, config.hid).to(device)
        self.A1.weight.data = self.A1.weight.data * A1_mask.T.to(device)
        
        for t in range(time_step):
            input_t = F.relu(self.fc_in(input[:,t,:].float()))
            
            ########## Layer 1 ##########
            #############################
            if config.norm:
                input_t = self.ln1(input_t)
            x = self.A1(input_t)

            hid1_mem_tmp, hid1_spk_tmp = mem_update(x, hid1_mem[:,t,:], hid1_spk[:,t,:], self.thr, self.decay, self.rst)
            hid1_mem[:,t+1,:] = hid1_mem_tmp
            hid1_spk[:,t+1,:] = hid1_spk_tmp
            sum1_spk += hid1_spk_tmp
            
        sum1_spk /= time_step
        out = self.fc_out(sum1_spk)
        A_norm = torch.norm(self.A1.weight, p=1)
        return out, hid1_mem, hid1_spk, A_norm
    
class RC_RadLIF(nn.Module):
    def __init__(self) -> None:
        super(RC_RadLIF, self).__init__()
        input = config.input
        hid = config.hid
        out = config.output
        self.fc_in = nn.Linear(input, hid)
        self.A1 = nn.Linear(hid, hid, bias=True) # random initialized adjacency matrix
        
        # 用于保证input强度大于lateral connection的强度，不然没法学到东西
        # 因此邻接矩阵的权重尽可能小
        # 0初始化也行
        if config.small_init:
            self.A1.weight.data = 0.9 * self.A1.weight.data
        self.fc_out = nn.Linear(hid, out)
        
        self.thr = nn.Parameter(torch.rand(config.hid).uniform_(0.9, 1.1), requires_grad=False)
        self.a = nn.Parameter(torch.rand(config.hid).uniform_(-1, 1), requires_grad=False)
        self.b = nn.Parameter(torch.rand(config.hid).uniform_(0, 2), requires_grad=False)
        self.alpha = nn.Parameter(torch.rand(config.hid).uniform_(0.6, 0.96), requires_grad=False)
        self.beta = nn.Parameter(torch.rand(config.hid).uniform_(0.96, 0.99), requires_grad=False)
        
        self.ln1 = nn.LayerNorm(hid)
        
        if not config.input_learn:
            for name, p in self.named_parameters():
                if 'conv' in name or 'fc_in' in name:
                    p.requires_grad = False
        
    def forward(self, input, mask, device='cuda', mode='parallel'):
        A1_mask, A2_mask, A3_mask, A4_mask = mask
        batch = input.shape[0]
        time_step = input.shape[1]
        
        hid1_mem = torch.zeros(batch, time_step+1, config.hid).uniform_(0, 0.1).to(device)
        hid1_spk = torch.zeros(batch, time_step+1, config.hid).to(device)
        hid1_W = torch.zeros(batch, time_step+1, config.hid).to(device)
        sum1_spk = torch.zeros(batch, config.hid).to(device)
        self.A1.weight.data = self.A1.weight.data * A1_mask.T.to(device)
        
        for t in range(time_step):
            input_t = F.relu(self.fc_in(input[:,t,:].float()))
            
            ########## Layer 1 ##########
            #############################
            if config.norm:
                input_t = self.ln1(input_t)
            x = self.A1(input_t)
            
            hid1_mem_tmp, hid1_spk_tmp, hid1_W_tmp = RadLIF_update(x, hid1_mem[:,t,:], hid1_spk[:,t,:], self.thr, self.alpha, self.beta, self.a, self.b, hid1_W[:,t,:])
            
            hid1_mem[:,t+1,:] = hid1_mem_tmp
            hid1_spk[:,t+1,:] = hid1_spk_tmp
            hid1_W[:,t+1,:] = hid1_W_tmp
            sum1_spk += hid1_spk_tmp
            
        sum1_spk /= time_step
        out = self.fc_out(sum1_spk)
        A_norm = torch.norm(self.A1.weight, p=1)
        return out, hid1_mem, hid1_spk, A_norm


def train(model, optimizer, criterion, num_epochs, train_loader, test_loader, device, mode):
    train_accs, test_accs = [], []
    
    m1 = (torch.rand(config.hid, config.hid) > config.dropout).int() * (1-torch.eye(config.hid, config.hid)).int()
    m2 = (torch.rand(config.hid, config.hid) > config.dropout).int() * (1-torch.eye(config.hid, config.hid)).int()
    m3 = (torch.rand(config.hid, config.hid) > config.dropout).int() * (1-torch.eye(config.hid, config.hid)).int()
    m4 = (torch.rand(config.hid, config.hid) > config.dropout).int() * (1-torch.eye(config.hid, config.hid)).int()
    mask = [m1.float(), m2.float(), m3.float(), m4.float()]
    
    for epoch in range(num_epochs):
        now = time.time()
        correct, total = 0, 0
        for i, (samples, labels) in enumerate(train_loader): # 
            # samples = samples.requires_grad_().to(device)
            labels = labels.long().to(device)
            optimizer.zero_grad()
            
            outputs, _, _, A_norm = model(samples.to(device), mask, device, mode)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted.cpu() == labels.long().cpu()).sum()
            
            loss = criterion(outputs, labels) + config.l1*F.relu(A_norm-config.l1_targ) # , torch.max(A_norm-6000, 0) 规定一个区间，
            loss.backward()
            optimizer.step()
        tr_acc = 100. * correct.numpy() / total
        ts_acc = test(model, test_loader, mask, device)
        train_accs.append(tr_acc)
        test_accs.append(ts_acc)
        # res_str = 'epoch: ' + str(epoch) \
        #             + ' Loss: ' + str(loss.item())      \
        #             + '. Tr Acc: ' + str(tr_acc)        \
        #             + '. Ts Acc: ' + str(ts_acc)        \
        #             + '. Time:' + str(time.time()-now)  \
        #             + '. A norm:' + str(A_norm.item())
        print('epoch:%d,\tLoss:%.4f,\tTr Acc:%.4f,\tTs Acc:%.4f,\tTime:%.4f,\tA Norm:%.4f'%(epoch, loss.item(), tr_acc, ts_acc, time.time()-now, A_norm.item()))
        
        if (m1==0).sum().item()/config.hid**2 <= config.dropout_stop or \
            (m2==0).sum().item()/config.hid**2 <= config.dropout_stop or \
            (m3==0).sum().item()/config.hid**2 <= config.dropout_stop or \
            (m4==0).sum().item()/config.hid**2 <= config.dropout_stop:
            m1 = m1&((torch.rand(config.hid, config.hid) > config.dropout_stepping).int() * (1-torch.eye(config.hid, config.hid)).int())
            m2 = m2&((torch.rand(config.hid, config.hid) > config.dropout_stepping).int() * (1-torch.eye(config.hid, config.hid)).int())
            m3 = m3&((torch.rand(config.hid, config.hid) > config.dropout_stepping).int() * (1-torch.eye(config.hid, config.hid)).int())
            m4 = m4&((torch.rand(config.hid, config.hid) > config.dropout_stepping).int() * (1-torch.eye(config.hid, config.hid)).int())
            mask = [m1.float(), m2.float(), m3.float(), m4.float()]
    return np.array(train_accs), np.array(test_accs)

def test(model, dataloader, mask, device='cuda', mode='serial'):
    with torch.no_grad():
        correct, total = 0, 0
        for images, labels in dataloader:
            outputs, _, _, _ = model(images.to(device), mask, device=device, mode=mode)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted.cpu() == labels.long().cpu()).sum()
        accuracy = 100. * correct.numpy() / total
    return accuracy


model = RC().to(config.device)
# model = RC_RadLIF().to(config.device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                              lr=config.lr,
                              weight_decay=config.weight_decay)
acc = train(model, optimizer, criterion, config.epoch, train_loader, test_loader, 'cuda', mode='serial')