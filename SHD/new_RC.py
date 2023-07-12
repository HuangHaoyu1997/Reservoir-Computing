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
    
    seed = 123
    trials = 5        # try on 5 different seeds
    num_minibatch = 10000
    num_per_label_minibatch = 20 # number of samples of each label in one mini-batch
    batch = 256
    epoch = 100
    lr = 0.005
    l1 = 0.0003
    l1_targ = 2000
    dropout = 0.7
    norm = False      # add layer norm before each layer
    shortcut = False
    device = torch.device('cuda')



from spikingjelly.datasets.shd import SpikingHeidelbergDigits
# from spikingjelly.datasets.n_mnist import NMNIST

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
        self.A1 = nn.Linear(hid, hid, bias=True) # random initialized adjacency matrix
        self.A2 = nn.Linear(hid, hid, bias=True)
        self.A3 = nn.Linear(hid, hid, bias=True)
        self.A4 = nn.Linear(hid, hid, bias=True)
        # self.A1.weight.data = nn.Parameter(torch.tensor(A_cluster(config)))
        # self.A2.weight.data = nn.Parameter(torch.tensor(A_cluster(config)))
        # nn.init.uniform_(self.A1.weight, a=-1, b=1)
        # nn.init.uniform_(self.A2.weight, a=-1, b=1)
        # nn.init.uniform_(self.A3.weight, a=-1, b=1)
        # nn.init.uniform_(self.A4.weight, a=-1, b=1)
        
        self.A1_mask = (torch.rand(hid, hid) > config.dropout).float() * (1-torch.eye(hid, hid))
        self.A2_mask = (torch.rand(hid, hid) > config.dropout).float() * (1-torch.eye(hid, hid))
        self.A3_mask = (torch.rand(hid, hid) > config.dropout).float() * (1-torch.eye(hid, hid))
        self.A4_mask = (torch.rand(hid, hid) > config.dropout).float() * (1-torch.eye(hid, hid))
        # TODO 在训练过程中逐渐增加mask的稀疏度
        
        self.fc1 = nn.Linear(hid, hid)
        self.fc2 = nn.Linear(hid, hid)
        self.fc3 = nn.Linear(hid, hid)
        self.fc4 = nn.Linear(hid, hid)
        
        self.fc_out = nn.Linear(hid*4, out)
        
        self.thr = nn.Parameter(torch.rand(config.hid)*config.thr, requires_grad=False)
        self.decay = nn.Parameter(torch.rand(config.hid)*config.decay, requires_grad=False)
        self.rst = nn.Parameter(torch.rand(config.hid)*config.rst, requires_grad=False)
        
        self.ln1 = nn.LayerNorm(hid)
        self.ln2 = nn.LayerNorm(hid)
        self.ln3 = nn.LayerNorm(hid)
        self.ln4 = nn.LayerNorm(hid)
        self.ln5 = nn.LayerNorm(hid)
        self.ln6 = nn.LayerNorm(hid)
        self.ln7 = nn.LayerNorm(hid)
        self.ln8 = nn.LayerNorm(hid)
        
        for name, p in self.named_parameters():
            if 'conv' in name or 'fc_in' in name:
                p.requires_grad = False
        
    def forward(self, input, device='cuda', mode='parallel'):
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
        
        self.A1.weight.data = self.A1.weight.data * self.A1_mask.T.to(device)
        self.A2.weight.data = self.A2.weight.data * self.A2_mask.T.to(device)
        self.A3.weight.data = self.A3.weight.data * self.A3_mask.T.to(device)
        self.A4.weight.data = self.A4.weight.data * self.A4_mask.T.to(device)
        for t in range(time_step):
            # x = input > torch.rand(input.size()).to(device)
            # x = self.fc_in(x.float())
            # x = self.conv_in(x.float())
            input_t = self.fc_in(input[:,t,:])
            # x = self.conv_in(input[:, t, :, :, :])
            
            ########## Layer 1 ##########
            #############################
            if config.norm:
                input_t = self.ln1(input_t)
            x = self.A1(input_t)
            # x = x @ self.A
            # x = F.sigmoid(self.A1(x))
            # x = F.tanh(self.A1(self.ln1(input_t)))
            hid1_mem_tmp, hid1_spk_tmp = mem_update(x, hid1_mem[:,t,:], hid1_spk[:,t,:], self.thr, self.decay, self.rst)
            hid1_mem[:,t+1,:] = hid1_mem_tmp
            hid1_spk[:,t+1,:] = hid1_spk_tmp
            sum1_spk += hid1_spk_tmp
            
            ########## Layer 2 ##########
            #############################
            if mode == 'parallel':
                x = self.A2(input_t)
            elif mode == 'serial':
                input_l2 = hid1_spk_tmp
                if config.shortcut:
                    input_l2 += x # +x  or  + input_t?
                if config.norm:
                    input_l2 = self.ln2(input_l2)
                
                x = self.A2(input_l2)
                # x = F.sigmoid(self.A2(hid1_spk_tmp))
                # x = F.tanh(self.A2(self.ln2(hid1_spk_tmp)+x))
            
            hid2_mem_tmp, hid2_spk_tmp = mem_update(x, hid2_mem[:,t,:], hid2_spk[:,t,:], self.thr, self.decay, self.rst)
            hid2_mem[:,t+1,:] = hid2_mem_tmp
            hid2_spk[:,t+1,:] = hid2_spk_tmp
            sum2_spk += hid2_spk_tmp
            
            ########## Layer 3 ##########
            #############################
            if mode == 'parallel':
                x = self.A3(input_t)
            elif mode == 'serial':
                input_l3 = hid2_spk_tmp
                if config.shortcut:
                    input_l3 += x
                if config.norm:
                    input_l3 = self.ln3(input_l3)
                x = self.A3(input_l3)
                # x = F.tanh(self.A3(self.ln3(hid2_spk_tmp)+x))
                # x = F.sigmoid(self.A3(hid2_spk_tmp))
            hid3_mem_tmp, hid3_spk_tmp = mem_update(x, hid3_mem[:,t,:], hid3_spk[:,t,:], self.thr, self.decay, self.rst)
            hid3_mem[:,t+1,:] = hid3_mem_tmp
            hid3_spk[:,t+1,:] = hid3_spk_tmp
            sum3_spk += hid3_spk_tmp
            
            ########## Layer 4 ##########
            #############################
            if mode == 'parallel':
                x = self.A4(input_t)
            elif mode == 'serial':
                input_l4 = hid3_spk_tmp
                if config.shortcut:
                    input_l4 += x
                if config.norm:
                    input_l4 = self.ln4(input_l4)
                x = self.A4(input_l4)
                # x = F.tanh(self.A4(self.ln4(hid3_spk_tmp)+x))
                # x = F.sigmoid(self.A4(hid3_spk_tmp))
            hid4_mem_tmp, hid4_spk_tmp = mem_update(x, hid4_mem[:,t,:], hid4_spk[:,t,:], self.thr, self.decay, self.rst)
            hid4_mem[:,t+1,:] = hid4_mem_tmp
            hid4_spk[:,t+1,:] = hid4_spk_tmp
            sum4_spk += hid4_spk_tmp
            
        sum1_spk /= time_step
        sum2_spk /= time_step
        sum3_spk /= time_step
        sum4_spk /= time_step
        
        out1 = sum1_spk
        out2 = sum2_spk
        out3 = sum3_spk
        out4 = sum4_spk
        if config.norm:
            out1 = self.ln5(sum1_spk)
            out2 = self.ln6(sum2_spk)
            out3 = self.ln7(sum3_spk)
            out4 = self.ln8(sum4_spk)
        out1 = F.relu(self.fc1(out1))
        out2 = F.relu(self.fc2(out2))
        out3 = F.relu(self.fc3(out3))
        out4 = F.relu(self.fc4(out4))
        # out = self.fc_out(torch.cat((sum1_spk, sum2_spk, sum3_spk, sum4_spk, ), dim=1)) # hid1_mem.mean(1), hid2_mem.mean(1), hid3_mem.mean(1), hid4_mem.mean(1)
        out = self.fc_out(torch.cat((out1, out2, out3, out4), dim=1))
        A_norm = torch.norm(self.A1.weight, p=1) + \
                 torch.norm(self.A2.weight, p=1) + \
                 torch.norm(self.A3.weight, p=1) + \
                 torch.norm(self.A4.weight, p=1) 
        return out, hid4_mem, hid4_spk, A_norm

def train(model, optimizer, criterion, num_epochs, train_loader, test_loader, device, mode):
    train_accs, test_accs = [], []
    for epoch in range(num_epochs):
        now = time.time()
        correct, total = 0, 0
        for i, (samples, labels) in enumerate(tqdm(train_loader)): # 
            # samples = samples.requires_grad_().to(device)
            labels = labels.long().to(device)
            optimizer.zero_grad()
            outputs, _, _, A_norm = model(samples.to(device), device, mode)
            
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
        # res_str = 'epoch: ' + str(epoch) \
        #             + ' Loss: ' + str(loss.item())      \
        #             + '. Tr Acc: ' + str(tr_acc)        \
        #             + '. Ts Acc: ' + str(ts_acc)        \
        #             + '. Time:' + str(time.time()-now)  \
        #             + '. A norm:' + str(A_norm.item())
        print('epoch:%d,\tLoss:%.4f,\tTr Acc:%.4f,\tTs Acc:%.4f,\tTime:%.4f,\tA Norm:%.4f'%(epoch, loss.item(), tr_acc, ts_acc, time.time()-now, A_norm.item()))
    return np.array(train_accs), np.array(test_accs)

def test(model, dataloader, device='cuda', mode='serial'):
    with torch.no_grad():
        correct, total = 0, 0
        for images, labels in dataloader:
            outputs, _, _, _ = model(images.to(device), device=device, mode=mode)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted.cpu() == labels.long().cpu()).sum()
        accuracy = 100. * correct.numpy() / total
    return accuracy

train_acc_log = np.zeros((config.epoch, config.trials))
test_acc_log = np.zeros((config.epoch, config.trials))
for i in range(config.trials):
    print('************** ', i, ' **************')
    np.random.seed(config.seed+i)
    torch.manual_seed(config.seed+i)
    torch.cuda.manual_seed_all(config.seed+i)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model = RC().to(config.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr)
    train_acc, test_acc = train(model, optimizer, criterion, config.epoch, train_loader, test_loader, 'cuda', 'serial')
    accuracy = test(model, test_loader, mode='serial')
    train_acc_log[:,i] = train_acc
    test_acc_log[:,i] = test_acc

train_mean = np.mean(train_acc_log, axis=1)
train_std = np.std(train_acc_log, axis=1)
train_var = np.var(train_acc_log, axis=1)
train_max = np.max(train_acc_log, axis=1)
train_min = np.min(train_acc_log, axis=1)

test_mean = np.mean(test_acc_log, axis=1)
test_std = np.std(test_acc_log, axis=1)
test_var = np.var(test_acc_log, axis=1)
test_max = np.max(test_acc_log, axis=1)
test_min = np.min(test_acc_log, axis=1)

plt.plot(list(range(config.epoch)), train_mean, color='deeppink', label='train mean')
# plt.fill_between(list(range(config.epoch)), data_mean-data_std, data_mean+data_std, color='violet', alpha=0.2)
plt.fill_between(list(range(config.epoch)), train_min, train_max, color='violet', alpha=0.2)

plt.plot(list(range(config.epoch)), test_mean, color='blue', label='test mean')
# plt.fill_between(list(range(config.epoch)), data_mean-data_std, data_mean+data_std, color='violet', alpha=0.2)
plt.fill_between(list(range(config.epoch)), test_min, test_max, color='blue', alpha=0.2)

plt.legend()
plt.grid()
# plt.axis([-5, 105, 75, 95])
plt.savefig('SHD_dropout.pdf')
plt.show()