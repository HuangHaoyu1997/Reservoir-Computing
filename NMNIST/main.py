'''
首次修改2023年7月13日22:10:55

'''
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import time, warnings
warnings.filterwarnings("ignore")
# from utils import A_cluster
from tqdm.auto import tqdm
from torchvision import transforms, datasets
from copy import deepcopy
from spikingjelly.datasets.n_mnist import NMNIST


class config:
    date = time.strftime("%Y-%m-%d-%H-%M-%S/", time.localtime(time.time()))[5:16]
    save_dir = './log/' + date
        
    input = 700
    output = 10
    hid = 300         # number of RC Neurons
    thr = 0.5
    b_j0 = 0.01       # thr baseline
    dt = 1
    R_m = 1
    decay = 0.5
    rst = 0.05
    lens = 0.5
    gamma = 0.5       # gradient scale 
    gradient_type = 'G' # 'MG', 'slayer', 'linear' 窗型函数
    scale = 6.        # special for 'MG'
    hight = 0.15      # special for 'MG'

    N_hid = hid
    p_in = 0.2        # ratio of inhibitory neurons
    # gamma = 1.0       # shape factor of gamma distribution
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
    
    input_learn = False # learnable input layer
    seed = 123
    trials = 5        # try on 5 different seeds
    batch = 512
    epoch = 100
    lr = 0.02
    l1 = 0.0003
    l1_targ = 5000
    fr_norm = 0.01
    fr_targ = 0.05
    dropout = 0.75
    dropout_stepping = 0.02
    dropout_stop = 0.98
    weight_decay = 1e-4
    label_smoothing = False
    smoothing = 0.15
    noise_test = 0.1
    norm = False      # add layer norm before each layer
    shortcut = False
    small_init = True
    ckpt_freq = 10    # every 10 epoch save model
    device = torch.device('cuda')

##########################################################
########### define surrogate gradient function ###########
def gaussian(x, mu=0., sigma=.5):
    return torch.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / torch.sqrt(2 * torch.tensor(torch.pi)) / sigma

class ActFun_adp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):  # input = membrane potential- threshold
        ctx.save_for_backward(input)
        return input.gt(0).float()

    @staticmethod
    def backward(ctx, grad_output):  # approximate the gradients
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        # temp = abs(input) < lens
        if config.gradient_type == 'G':
            temp = torch.exp(-(input**2)/(2*config.lens**2))/torch.sqrt(2*torch.tensor(torch.pi))/config.lens
        elif config.gradient_type == 'MG':
            temp = gaussian(input, mu=0., sigma=config.lens) * (1. + config.hight) \
                - gaussian(input, mu=config.lens, sigma=config.scale * config.lens) * config.hight \
                - gaussian(input, mu=-config.lens, sigma=config.scale * config.lens) * config.hight
        elif config.gradient_type =='linear':
            temp = F.relu(1-input.abs())
        elif config.gradient_type == 'slayer':
            temp = torch.exp(-5*input.abs())
        return grad_input * temp.float() * config.gamma
act_fun_adp = ActFun_adp.apply

#######################################
########### define RC model ###########

class RC(nn.Module):
    def __init__(self):
        super(RC, self).__init__()
        self.conv1 = nn.Conv2d(2, 8, 3, stride=2)
        self.conv2 =  nn.Conv2d(8, 8, 3, stride=2)
        
        self.inpt_hid1 = nn.Linear(392, config.hid)
        self.hid1_hid1 = nn.Linear(config.hid, config.hid) # A1
        self.hid1_hid2 = nn.Linear(config.hid, config.hid)
        self.hid2_hid2 = nn.Linear(config.hid, config.hid) # A2
        self.hid2_out = nn.Linear(config.hid, config.output)
        if config.small_init:
            self.hid1_hid1.weight.data = 0.2 * self.hid1_hid1.weight.data
            self.hid2_hid2.weight.data = 0.2 * self.hid2_hid2.weight.data
        
        nn.init.orthogonal_(self.inpt_hid1.weight)  # 主要用以解决深度网络的梯度消失爆炸问题，在RNN中经常使用
        nn.init.orthogonal_(self.hid2_hid2.weight)
        nn.init.xavier_uniform_(self.inpt_hid1.weight) # 保持输入输出的方差一致，避免所有输出值都趋向于0。通用方法，适用于任何激活函数
        nn.init.xavier_uniform_(self.hid1_hid2.weight)
        nn.init.xavier_uniform_(self.hid2_out.weight)
        
        nn.init.constant_(self.inpt_hid1.bias, 0)
        nn.init.constant_(self.hid1_hid2.bias, 0)
        nn.init.constant_(self.hid1_hid1.bias, 0)
        nn.init.constant_(self.hid2_hid2.bias, 0)
        
        self.tau_adp_h1 = nn.Parameter(torch.Tensor(config.hid))
        self.tau_adp_h2 = nn.Parameter(torch.Tensor(config.hid))
        self.tau_adp_o = nn.Parameter(torch.Tensor(config.output))
        self.tau_m_h1 = nn.Parameter(torch.Tensor(config.hid))
        self.tau_m_h2 = nn.Parameter(torch.Tensor(config.hid))
        self.tau_m_o = nn.Parameter(torch.Tensor(config.output))
        
        nn.init.normal_(self.tau_adp_h1, 150, 10)
        nn.init.normal_(self.tau_adp_h2, 150, 10)
        nn.init.normal_(self.tau_adp_o, 150, 10)
        nn.init.normal_(self.tau_m_h1, 20., 5)
        nn.init.normal_(self.tau_m_h2, 20., 5)
        nn.init.normal_(self.tau_m_o, 20., 5)
        
        self.b_hid1 = self.b_hid2 = self.b_o = 0
        self.dp = nn.Dropout(0.1)
        
        if not config.input_learn:
            for name, p in self.named_parameters():
                if 'conv1' in name or 'conv2' in name:
                    p.requires_grad = False
    
    def output_Neuron(self, inputs, mem, tau_m, dt=1):
        """The read out neuron is leaky integrator without spike"""
        # alpha = torch.exp(-1. * dt / torch.FloatTensor([30.])).to(config.device)
        alpha = torch.exp(-1. * dt / tau_m).to(config.device)
        mem = mem * alpha + (1. - alpha) * config.R_m * inputs
        return mem
    
    def mem_update_adp(self, inputs, mem, spike, tau_adp, b, tau_m, dt=1, isAdapt=1):
        alpha = torch.exp(-1. * dt / tau_m).to(config.device)
        ro = torch.exp(-1. * dt / tau_adp).to(config.device)
        if isAdapt: beta = 1.8
        else:       beta = 0.
        b = ro * b + (1 - ro) * spike
        B = config.b_j0 + beta * b
        mem = mem * alpha + (1 - alpha) * config.R_m * inputs - B * spike * dt
        spike = act_fun_adp(mem - B)
        return mem, spike, B, b
    
    def forward(self, input, mask):
        
        batch = input.shape[0]
        time_step = input.shape[1]
        self.b_hid1 = self.b_hid2 = self.b_out = config.b_j0
        
        hid1_mem = torch.zeros(batch, config.hid).uniform_(0, 0.1).to(config.device)
        hid1_spk = torch.zeros(batch, config.hid).to(config.device)
        
        hid2_mem = torch.zeros(batch, config.hid).uniform_(0, 0.1).to(config.device)
        hid2_spk = torch.zeros(batch, config.hid).to(config.device)
        
        out_mem = torch.zeros(batch, config.output).uniform_(0, 0.1).to(config.device)
        output = torch.zeros(batch, config.output).to(config.device)
        
        sum1_spk = torch.zeros(batch, config.hid).to(config.device)
        sum2_spk = torch.zeros(batch, config.hid).to(config.device)
        
        if config.dropout>0:
            self.hid1_hid1.weight.data = self.hid1_hid1.weight.data * mask[0].T
            self.hid2_hid2.weight.data = self.hid2_hid2.weight.data * mask[1].T
        for t in range(time_step):
            input_t = input[:,t,:,:,:].float()
            ########## Layer 0 ##########
            x = F.relu(self.conv1(input_t))
            x = F.relu(self.conv2(x))
            x = x.view(batch, -1)
            
            ########## Layer 1 ##########
            inpt_hid1 = self.inpt_hid1(x) + self.hid1_hid1(hid1_spk)
            hid1_mem, hid1_spk, theta_h1, self.b_h1 = self.mem_update_adp(inpt_hid1, hid1_mem, hid1_spk, self.tau_adp_h1, self.b_hid1,self.tau_m_h1)
            sum1_spk += hid1_spk
            # hid1_spk = self.dp(hid1_spk)
            
            ########## Layer 2 ##########
            # inpt_hid2 = self.hid1_hid2(hid1_spk) + self.hid2_hid2(hid2_spk)
            # hid2_mem, hid2_spk, theta_h2, self.b_h2 = self.mem_update_adp(inpt_hid2, hid2_mem, hid2_spk, self.tau_adp_h2, self.b_hid2,self.tau_m_h2)
            # sum2_spk += hid2_spk
            # hid2_spk = self.dp(hid2_spk)
            
            ########## Layer out ########
            inpt_out = self.hid2_out(hid1_spk)
            output += inpt_out
            # out_mem = self.output_Neuron(inpt_out, out_mem, self.tau_m_o)
            # if t >= 0:
            #     output += F.softmax(out_mem, dim=1)
            
        sum1_spk /= time_step
        # sum2_spk /= time_step
        output /= time_step

        A_norm = torch.norm(self.hid1_hid1.weight, p=1) # + torch.norm(self.hid2_hid2.weight, p=1)
        
        cluster_in = 0 # 簇内聚类程度
        cluster_out = 0
        global_mean = 0 # 全局中心位置
        for i in range(config.output):
            center = self.hid1_hid1.weight[i*30:(i+1)*30].mean(0)
            cluster_in += ((self.hid1_hid1.weight[i*30:(i+1)*30] - center)**2).mean()
            global_mean += 0.1*center
        # for i in range(config.output):
        #     a = self.hid1_hid1.weight[i*30:(i+1)*30, i*30:(i+1)*30].std(dim=0).sum()
        #     b = self.hid1_hid1.weight[i*30:(i+1)*30, i*30:(i+1)*30].mean(1)
        #     print(torch.std(a, dim=1), torch.std(a, dim=1).shape)
        #     global_mean += 0.1*b
        #     c = (a-b)**2
        #     c = c.mean()
        #     cluster_in += c
        for i in range(config.output):
            d = ((global_mean - self.hid1_hid1.weight[i*30:(i+1)*30].mean(0))**2).mean()
            cluster_out += d
            # print(a[0,0:5], b[0:5], c[0,0:5], (c**2)[0,0:5])
        # print(cluster_in, cluster_out)
        return output, sum1_spk, sum1_spk, A_norm, cluster_in, cluster_out

################################################
########### define training pipeline ###########
def train(trial, model, optimizer, criterion, num_epochs, train_loader, test_loader):
    model.train()
    train_accs, test_accs = [], []

    a = torch.zeros((config.hid, config.hid), dtype=torch.int)
    for i in range(config.output):
        a[i*30:(i+1)*30, i*30:(i+1)*30] = 1.
    invalid_zeros = 1-(a==1).sum().item()/config.hid**2
    if invalid_zeros < config.dropout:
        b = (torch.rand(config.hid, config.hid) > (config.dropout-invalid_zeros)/(1-invalid_zeros)).int() * (1-torch.eye(config.hid, config.hid, dtype=int))
        m1 = a & b
        m1 += torch.eye(config.hid, config.hid, dtype=int)
    else: m1 = a
    
    # m1 = (torch.rand(config.hid, config.hid) > config.dropout).int() * (1-torch.eye(config.hid, config.hid)).int()
    m2 = (torch.rand(config.hid, config.hid) > config.dropout).int() * (1-torch.eye(config.hid, config.hid)).int()
    mask = [m1.float().to(config.device), m2.float().to(config.device)]
    
    torch.save([model.state_dict(), mask], config.save_dir+'/before-train-{:d}-{:d}-{:.2f}.tar'.format(trial, 0, 0))
    
    for epoch in range(num_epochs):
        now = time.time()
        correct, total = 0, 0
        for i, (samples, labels) in enumerate(tqdm(train_loader)): # 
            # samples = samples.requires_grad_().to(device)
            labels = labels.long().to(config.device)
            optimizer.zero_grad()
            
            
            samples = torch.sign(samples.clamp(min=0)) # all pixels should be 0 or 1
            outputs, sum1_spk, sum2_spk, A_norm, cluster_in, cluster_out = model(samples.to(config.device), mask)
            firing_rate = sum1_spk.mean()*0.5 + sum2_spk.mean()*0.5
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted.cpu() == labels.long().cpu()).sum()
            
            # label smoothing
            if config.smoothing>0:
                with torch.no_grad():
                    true_dist = torch.zeros_like(outputs)
                    true_dist.fill_(config.smoothing / (config.output - 1))
                    true_dist.scatter_(1, labels.unsqueeze(1), 1.0 - config.smoothing)
                
                loss = criterion(outputs, true_dist)
            else:
                loss = criterion(outputs, labels)
            
            loss = loss + config.fr_norm * F.relu(firing_rate - config.fr_targ) + \
                    (0.02*cluster_in - 0.04*cluster_out) + \
                    config.l1 * F.relu(A_norm - config.l1_targ) # , torch.max(A_norm-6000, 0) 规定一个区间，
            loss.backward()
            optimizer.step()
        tr_acc = 100. * correct.numpy() / total
        ts_acc = test(model, test_loader, mask)
        train_accs.append(tr_acc)
        test_accs.append(ts_acc)
        # res_str = 'epoch: ' + str(epoch) \
        #             + ' Loss: ' + str(loss.item())      \
        #             + '. Tr Acc: ' + str(tr_acc)        \
        #             + '. Ts Acc: ' + str(ts_acc)        \
        #             + '. Time:' + str(time.time()-now)  \
        #             + '. A norm:' + str(A_norm.item())
        print('epoch:%d, Loss:%.4f, Tr Acc:%.4f, Ts Acc:%.2f, Time:%.4f,\tA Norm:%.4f,\tFr:%.4f, Mask:%.4f, Cin:%.4f, Cout:%.4f'%\
            (epoch, loss.item(), tr_acc, ts_acc, time.time()-now, A_norm.item(), firing_rate, (m1==0).sum().item()/config.hid**2, cluster_in.item(), cluster_out.item()))
        if epoch % config.ckpt_freq==0:
            torch.save([model.state_dict(), mask], config.save_dir+'/model-{:d}-{:d}-{:.2f}.tar'.format(trial, epoch, ts_acc))
        
        
        if (m1==0).sum().item()/config.hid**2 <= config.dropout_stop: # or (m2==0).sum().item()/config.hid**2 <= config.dropout_stop:
            m1 = m1&((torch.rand(config.hid, config.hid) > config.dropout_stepping).int() * (1-torch.eye(config.hid, config.hid)).int())
            m2 = m2&((torch.rand(config.hid, config.hid) > config.dropout_stepping).int() * (1-torch.eye(config.hid, config.hid)).int())
            mask = [m1.float().to(config.device), m2.float().to(config.device)]
    return np.array(train_accs), np.array(test_accs)

def test(model, dataloader, mask):
    model.eval()
    with torch.no_grad():
        correct, total = 0, 0
        for images, labels in dataloader:
            images = torch.sign(images.clamp(min=0)) # all pixels should be 0 or 1
            if config.noise_test>0:
                images += torch.rand_like(images) * config.noise_test
            outputs, _, _, _, _, _ = model(images.to(config.device), mask)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted.cpu() == labels.long().cpu()).sum()
        accuracy = 100. * correct.numpy() / total
    return accuracy

def multiple_trial():
    train_acc_log = np.zeros((config.epoch, config.trials))
    test_acc_log = np.zeros((config.epoch, config.trials))
    for i in range(config.trials):
        print('************** Trial ', i, ' **************')
        np.random.seed(config.seed+i)
        torch.manual_seed(config.seed+i)
        torch.cuda.manual_seed_all(config.seed+i)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        model = RC().to(config.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr, weight_decay=config.weight_decay)
        # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr)
        train_acc, test_acc = train(i, model, optimizer, criterion, config.epoch, train_loader, test_loader)
        
        train_acc_log[:,i] = train_acc
        test_acc_log[:,i] = test_acc
    return train_acc_log, test_acc_log

def plot_errorbar(train_acc_log, test_acc_log, file_name):
    train_mean = np.mean(train_acc_log, axis=1)
    train_std = np.std(train_acc_log, axis=1)
    # train_var = np.var(train_acc_log, axis=1)
    # train_max = np.max(train_acc_log, axis=1)
    # train_min = np.min(train_acc_log, axis=1)

    test_mean = np.mean(test_acc_log, axis=1)
    test_std = np.std(test_acc_log, axis=1)
    # test_var = np.var(test_acc_log, axis=1)
    # test_max = np.max(test_acc_log, axis=1)
    # test_min = np.min(test_acc_log, axis=1)

    plt.plot(list(range(config.epoch)), train_mean, color='deeppink', label='train')
    plt.fill_between(list(range(config.epoch)), train_mean-train_std, train_mean+train_std, color='deeppink', alpha=0.2)
    # plt.fill_between(list(range(config.epoch)), train_min, train_max, color='violet', alpha=0.2)

    plt.plot(list(range(config.epoch)), test_mean, color='blue', label='test')
    plt.fill_between(list(range(config.epoch)), test_mean-test_std, test_mean+test_std, color='blue', alpha=0.2)
    # plt.fill_between(list(range(config.epoch)), test_min, test_max, color='blue', alpha=0.2)

    plt.legend()
    plt.grid()
    # plt.axis([-5, 105, 75, 95])
    plt.savefig(file_name)
    # plt.show()


###################################
########### start train ###########
if __name__ == "__main__":
    os.makedirs(config.save_dir) if not os.path.exists(config.save_dir) else None
    
    #####################################
    ########### load SHD data ###########
    nmnist_train = NMNIST('./data/', train=True, data_type='frame', frames_number=10, split_by='number')
    nmnist_test = NMNIST('./data/', train=False, data_type='frame', frames_number=10, split_by='number')
    train_loader = torch.utils.data.DataLoader(dataset=nmnist_train, batch_size=config.batch, shuffle=True, drop_last=False, num_workers=0)
    test_loader = torch.utils.data.DataLoader(dataset=nmnist_test, batch_size=config.batch, shuffle=False, drop_last=False, num_workers=0)
    
    train_acc_log, test_acc_log = multiple_trial()
    plot_errorbar(train_acc_log, test_acc_log, './fig/'+ config.date +'.pdf')