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
    
    
    num_minibatch = 10000
    num_per_label_minibatch = 20 # number of samples of each label in one mini-batch
    batch = 256
    epoch = 300
    lr = 0.005
    l1 = 0.0002
    l1_targ = 600
    device = torch.device('cuda')

from spikingjelly.datasets.shd import SpikingHeidelbergDigits
from spikingjelly.datasets.n_mnist import NMNIST

# nmnist_train = NMNIST('D:\Ph.D\Research\SNN-SRT数据/N-MNIST', train=True, data_type='frame', frames_number=20, split_by='number')
# nmnist_test = NMNIST('D:\Ph.D\Research\SNN-SRT数据/N-MNIST', train=False, data_type='frame', frames_number=20, split_by='number')
# train_loader = torch.utils.data.DataLoader(dataset=nmnist_train, batch_size=config.batch, shuffle=True, drop_last=False, num_workers=0)
# test_loader = torch.utils.data.DataLoader(dataset=nmnist_test, batch_size=config.batch, shuffle=False, drop_last=False, num_workers=0)

SHD_train = SpikingHeidelbergDigits('D:\Ph.D\Research\SNN-SRT数据\SHD', train=True, data_type='frame', frames_number=20, split_by='number')
SHD_test = SpikingHeidelbergDigits('D:\Ph.D\Research\SNN-SRT数据\SHD', train=False, data_type='frame', frames_number=20, split_by='number')
train_loader = torch.utils.data.DataLoader(dataset=SHD_train, batch_size=config.batch, shuffle=True, drop_last=False, num_workers=0)
test_loader = torch.utils.data.DataLoader(dataset=SHD_test, batch_size=config.batch, shuffle=False, drop_last=False, num_workers=0)


# transform = transforms.Compose([
#     transforms.ToTensor(),
#     # transforms.Normalize((0.1307,), (0.3081,))
# ])

# train_dataset = datasets.MNIST(root='./data/', train=True, transform=transform, download=True)
# test_dataset = datasets.MNIST(root='./data/', train=False, transform=transform, download=True)
# train_dataset = datasets.CIFAR10(root='./data', train=True, download=True)
# test_dataset = datasets.CIFAR10(root='./data', train=False, download=True)
# train_dataset = datasets.CIFAR100(root='./data', train=True, download=True)
# test_dataset = datasets.CIFAR100(root='./data', train=False, download=True)
# train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=config.batch, shuffle=True)
# test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=config.batch, shuffle=False)

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

class RC(nn.Module):
    def __init__(self) -> None:
        super(RC, self).__init__()
        input = config.input
        hid = config.hid
        out = config.output
        self.fc_in = nn.Linear(input, hid)
        self.conv_in = nn.Sequential(
                                    nn.Conv2d(1, 16, 3),
                                    nn.ReLU(),
                                    nn.AvgPool2d(2, 2),
                                    nn.Conv2d(16, 16, 3),
                                    nn.ReLU(),
                                    nn.AvgPool2d(2, 2),
                                    nn.Flatten(),
                                    self.fc_in,
                                    )
        self.fc_in.requires_grad_ = False
        self.conv_in.requires_grad_ = False
        
        # self.A = nn.Parameter(torch.tensor(A_cluster(config)), requires_grad=False) # adjacency matrix
        self.A1 = nn.Linear(config.hid, config.hid, bias=True) # random initialized adjacency matrix
        self.A2 = nn.Linear(config.hid, config.hid, bias=True)
        self.A3 = nn.Linear(config.hid, config.hid, bias=True)
        self.A4 = nn.Linear(config.hid, config.hid, bias=True)
        # self.A1.weight.data = nn.Parameter(torch.tensor(A_cluster(config)))
        # self.A2.weight.data = nn.Parameter(torch.tensor(A_cluster(config)))
        nn.init.uniform_(self.A1.weight, a=0, b=1)
        nn.init.uniform_(self.A2.weight, a=0, b=1)
        nn.init.uniform_(self.A3.weight, a=0, b=1)
        nn.init.uniform_(self.A4.weight, a=0, b=1)
        self.fc_out = nn.Linear(hid*4, out)
        
        self.thr = nn.Parameter(torch.rand(config.hid)*config.thr, requires_grad=False)
        self.decay = nn.Parameter(torch.rand(config.hid)*config.decay, requires_grad=False)
        self.rst = nn.Parameter(torch.rand(config.hid)*config.rst, requires_grad=False)
        
        for name, p in self.named_parameters():
            if 'conv' in name or 'fc_in' in name:
                p.requires_grad = False
        
    def forward(self, input, device='cuda'):
        batch = input.shape[0]
        time_step = input.shape[1]
        # input = input.view(batch, config.input)
        # if len(input.shape)>2:
        #     batch, time_step = input.shape[0:2]
        
        hid1_mem = torch.zeros(batch, time_step+1, config.hid).uniform_(0, 0.1).to(device)
        hid1_spk = torch.zeros(batch, time_step+1, config.hid).to(device)
        
        hid2_mem = torch.zeros(batch, time_step+1, config.hid).uniform_(0, 0.1).to(device)
        hid2_spk = torch.zeros(batch, time_step+1, config.hid).to(device)
        
        hid3_mem = torch.zeros(batch, time_step+1, config.hid).uniform_(0, 0.1).to(device)
        hid3_spk = torch.zeros(batch, time_step+1, config.hid).to(device)
        
        hid4_mem = torch.zeros(batch, time_step+1, config.hid).uniform_(0, 0.1).to(device)
        hid4_spk = torch.zeros(batch, time_step+1, config.hid).to(device)
        
        sum1_spk = torch.zeros(batch, config.hid).to(device)
        sum2_spk = torch.zeros(batch, config.hid).to(device)
        sum3_spk = torch.zeros(batch, config.hid).to(device)
        sum4_spk = torch.zeros(batch, config.hid).to(device)
        
        for t in range(time_step):
            # x = input > torch.rand(input.size()).to(device)
            # x = self.fc_in(x.float())
            # x = self.conv_in(x.float())
            x = self.fc_in(input[:,t,:])
            # x = self.conv_in(input[:, t, :, :, :])
            
            # x = x @ self.A
            x = self.A1(x)
            # x = F.sigmoid(self.A1(x))
            hid1_mem_tmp, hid1_spk_tmp = mem_update(x, hid1_mem[:,t,:], hid1_spk[:,t,:], self.thr, self.decay, self.rst)
            hid1_mem[:,t+1,:] = hid1_mem_tmp
            hid1_spk[:,t+1,:] = hid1_spk_tmp
            sum1_spk += hid1_spk_tmp
            
            x = self.A2(x)
            # x = F.sigmoid(self.A2(hid1_spk_tmp))
            hid2_mem_tmp, hid2_spk_tmp = mem_update(x, hid2_mem[:,t,:], hid2_spk[:,t,:], self.thr, self.decay, self.rst)
            hid2_mem[:,t+1,:] = hid2_mem_tmp
            hid2_spk[:,t+1,:] = hid2_spk_tmp
            sum2_spk += hid2_spk_tmp
            
            x = self.A3(x)
            # x = F.sigmoid(self.A3(hid2_spk_tmp))
            hid3_mem_tmp, hid3_spk_tmp = mem_update(x, hid3_mem[:,t,:], hid3_spk[:,t,:], self.thr, self.decay, self.rst)
            hid3_mem[:,t+1,:] = hid3_mem_tmp
            hid3_spk[:,t+1,:] = hid3_spk_tmp
            sum3_spk += hid3_spk_tmp
            
            x = self.A4(x)
            # x = F.sigmoid(self.A4(hid3_spk_tmp))
            hid4_mem_tmp, hid4_spk_tmp = mem_update(x, hid4_mem[:,t,:], hid4_spk[:,t,:], self.thr, self.decay, self.rst)
            hid4_mem[:,t+1,:] = hid4_mem_tmp
            hid4_spk[:,t+1,:] = hid4_spk_tmp
            sum4_spk += hid4_spk_tmp
            
        sum1_spk /= time_step
        sum2_spk /= time_step
        sum3_spk /= time_step
        sum4_spk /= time_step
        out = self.fc_out(torch.cat((sum1_spk, sum2_spk, sum3_spk, sum4_spk, hid1_mem.mean(1), hid2_mem.mean(1), hid3_mem.mean(1), hid4_mem.mean(1)), dim=1))
        A_norm = torch.norm(self.A1.weight, p=1) + \
                 torch.norm(self.A2.weight, p=1) + \
                 torch.norm(self.A3.weight, p=1) + \
                 torch.norm(self.A4.weight, p=1) 
        return out, hid4_mem, hid4_spk, A_norm

def train(model, optimizer, criterion, num_epochs, train_loader, test_loader, device):
    train_accs, test_accs = [], []
    for epoch in range(num_epochs):
        now = time.time()
        correct, total = 0, 0
        for i, (samples, labels) in enumerate(tqdm(train_loader)): # 
            # samples = samples.requires_grad_().to(device)
            labels = labels.long().to(device)
            optimizer.zero_grad()
            outputs, _, _, A_norm = model(samples.to(device), device)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted.cpu() == labels.long().cpu()).sum()
            
            loss = criterion(outputs, labels) + config.l1*F.relu(A_norm-config.l1_targ) # , torch.max(A_norm-6000, 0) 规定一个区间，
            loss.backward()
            optimizer.step()
        tr_acc = 100. * correct.numpy() / total
        ts_acc = test(model, test_loader, device)
        train_accs.append(tr_acc)
        test_accs.append(ts_acc)
        res_str = 'epoch: ' + str(epoch) \
                    + ' Loss: ' + str(loss.item())      \
                    + '. Tr Acc: ' + str(tr_acc)        \
                    + '. Ts Acc: ' + str(ts_acc)        \
                    + '. Time:' + str(time.time()-now)  \
                    + '. A norm:' + str(A_norm.item())
        print(res_str)
    return train_accs, test_accs

def test(model, dataloader, device='cuda'):
    correct, total = 0, 0
    for images, labels in dataloader:
        outputs, _, _, _ = model(images.to(device), device=device)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted.cpu() == labels.long().cpu()).sum()
    accuracy = 100. * correct.numpy() / total
    return accuracy

model = RC().to(config.device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr)
acc = train(model, optimizer, criterion, config.epoch, train_loader, test_loader, 'cuda')
accuracy = test(model, test_loader)

plt.plot(acc[0], label='train')
plt.plot(acc[1], label='test')
plt.grid()
# plt.axis([-5, 105, 75, 95])
plt.legend()
plt.savefig('SHD_mem.pdf')