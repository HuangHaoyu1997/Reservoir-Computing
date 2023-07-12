'''
2023年7月12日23:59:14

'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import time, warnings
warnings.filterwarnings("ignore")
from utils import A_cluster
from tqdm.auto import tqdm
from torchvision import transforms, datasets
from copy import deepcopy

class config:
    input = 700
    output = 20
    hid = 512         # number of RC Neurons
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
    
    input_learn = True # learnable input layer
    seed = 123
    trials = 5        # try on 5 different seeds
    num_minibatch = 10000
    num_per_label_minibatch = 20 # number of samples of each label in one mini-batch
    batch = 256
    epoch = 40
    lr = 0.005
    l1 = 0.0003
    l1_targ = 20000
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

#####################################
########### load SHD data ###########

train_X = np.load('data/trainX_4ms.npy')
train_y = np.load('data/trainY_4ms.npy').astype(float)

test_X = np.load('data/testX_4ms.npy')
test_y = np.load('data/testY_4ms.npy').astype(float)

print('dataset shape: ', train_X.shape)
print('dataset shape: ', test_X.shape)

tensor_trainX = torch.Tensor(train_X)  # transform to torch tensor
tensor_trainY = torch.Tensor(train_y)
train_dataset = torch.utils.data.TensorDataset(tensor_trainX, tensor_trainY)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch, shuffle=True)
tensor_testX = torch.Tensor(test_X)  # transform to torch tensor
tensor_testY = torch.Tensor(test_y)
test_dataset = torch.utils.data.TensorDataset(tensor_testX, tensor_testY)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch, shuffle=False)

##########################################################
########### define surrogate gradient function ###########
def gaussian(x, mu=0., sigma=.5):
    return torch.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / torch.sqrt(2 * torch.tensor(torch.pi)) / sigma

class ActFun_adp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):  # input = membrane potential- threshold
        ctx.save_for_backward(input)
        return input.gt(0).float()  # is firing ???

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
        self.inpt_hid1 = nn.Linear(config.input, config.hid)
        self.hid1_hid1 = nn.Linear(config.hid, config.hid) # A1
        self.hid1_hid2 = nn.Linear(config.hid, config.hid)
        self.hid2_hid2 = nn.Linear(config.hid, config.hid) # A2
        self.hid2_out = nn.Linear(config.hid, config.output)
        
        self.hid1_hid1.weight.data = 0.2 * self.hid1_hid1.weight.data
        self.hid2_hid2.weight.data = 0.2 * self.hid2_hid2.weight.data
        
        nn.init.orthogonal_(self.inpt_hid1.weight)  # 主要用以解决深度网络的梯度消失爆炸问题，在RNN中经常使用
        nn.init.orthogonal_(self.hid2_hid2.weight)
        nn.init.xavier_uniform_(self.inpt_hid1.weight) # 保持输入输出的方差一致，避免所有输出值都趋向于0。通用方法，适用于任何激活函数
        nn.init.xavier_uniform_(self.hid1_hid2.weight)
        nn.init.xavier_uniform_(self.hid2_out.weight)
        
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
        
        if not config.input_learn:
            for name, p in self.named_parameters():
                if 'inpt' in name or 'fc_in' in name:
                    p.requires_grad = False
    
    def output_Neuron(self, inputs, mem, tau_m, dt=1):
        """The read out neuron is leaky integrator without spike"""
        # alpha = torch.exp(-1. * dt / torch.FloatTensor([30.])).cuda()
        alpha = torch.exp(-1. * dt / tau_m).cuda()
        mem = mem * alpha + (1. - alpha) * config.R_m * inputs
        return mem
    
    def mem_update_adp(self, inputs, mem, spike, tau_adp, b, tau_m, dt=1, isAdapt=1):
        alpha = torch.exp(-1. * dt / tau_m).cuda()
        ro = torch.exp(-1. * dt / tau_adp).cuda()
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
        
        self.hid1_hid1.weight.data = self.hid1_hid1.weight.data * mask[0].T.to(config.device)
        self.hid2_hid2.weight.data = self.hid2_hid2.weight.data * mask[1].T.to(config.device)
        for t in range(time_step):
            input_t = input[:,t,:].float()
            
            ########## Layer 1 ##########
            inpt_hid1 = self.inpt_hid1(input_t) + self.hid1_hid1(hid1_spk)
            hid1_mem, hid1_spk, theta_h1, self.b_h1 = self.mem_update_adp(inpt_hid1, hid1_mem, hid1_spk, self.tau_adp_h1, self.b_hid1,self.tau_m_h1)
            sum1_spk += hid1_spk
            
            ########## Layer 2 ##########
            inpt_hid2 = self.hid1_hid2(hid1_spk) + self.hid2_hid2(hid2_spk)
            hid2_mem, hid2_spk, theta_h2, self.b_h2 = self.mem_update_adp(inpt_hid2, hid2_mem, hid2_spk, self.tau_adp_h2, self.b_hid2,self.tau_m_h2)
            sum2_spk += hid2_spk
            
            ########## Layer out ########
            inpt_out = self.hid2_out(hid2_spk)
            out_mem = self.output_Neuron(inpt_out, out_mem, self.tau_m_o)
            output += F.softmax(out_mem, dim=1)
            
        sum1_spk /= time_step
        sum2_spk /= time_step

        A_norm = torch.norm(self.hid1_hid1.weight, p=1) + torch.norm(self.hid2_hid2.weight, p=1)
        return output, sum1_spk, sum2_spk, A_norm

################################################
########### define training pipeline ###########
def train(model, optimizer, criterion, num_epochs, train_loader, test_loader):
    train_accs, test_accs = [], []
    
    m1 = (torch.rand(config.hid, config.hid) > config.dropout).int() * (1-torch.eye(config.hid, config.hid)).int()
    m2 = (torch.rand(config.hid, config.hid) > config.dropout).int() * (1-torch.eye(config.hid, config.hid)).int()
    mask = [m1.float(), m2.float()]
    
    for epoch in range(num_epochs):
        now = time.time()
        correct, total = 0, 0
        for i, (samples, labels) in enumerate(train_loader): # 
            # samples = samples.requires_grad_().to(device)
            labels = labels.long().to(config.device)
            optimizer.zero_grad()
            
            outputs, sum1_spk, sum2_spk, A_norm = model(samples.to(config.device), mask)
            firing_rate = sum1_spk.mean()*0.5 + sum2_spk.mean()*0.5
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted.cpu() == labels.long().cpu()).sum()
            
            loss = criterion(outputs, labels) + \
                   0.0001 * F.relu(firing_rate - 0.03) + \
                   config.l1*F.relu(A_norm-config.l1_targ) # , torch.max(A_norm-6000, 0) 规定一个区间，
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
        print('epoch:%d,\tLoss:%.4f,\tTr Acc:%.4f,\tTs Acc:%.4f,\tTime:%.4f,\tA Norm:%.4f,\tFr:%.4f'%\
            (epoch, loss.item(), tr_acc, ts_acc, time.time()-now, A_norm.item(), firing_rate))
        
        if (m1==0).sum().item()/config.hid**2 <= config.dropout_stop or \
            (m2==0).sum().item()/config.hid**2 <= config.dropout_stop:
            m1 = m1&((torch.rand(config.hid, config.hid) > config.dropout_stepping).int() * (1-torch.eye(config.hid, config.hid)).int())
            m2 = m2&((torch.rand(config.hid, config.hid) > config.dropout_stepping).int() * (1-torch.eye(config.hid, config.hid)).int())
            mask = [m1.float(), m2.float()]
    return np.array(train_accs), np.array(test_accs)

def test(model, dataloader, mask):
    with torch.no_grad():
        correct, total = 0, 0
        for images, labels in dataloader:
            outputs, _, _, _ = model(images.to(config.device), mask)
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
        train_acc, test_acc = train(model, optimizer, criterion, config.epoch, train_loader, test_loader)
        
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
train_acc_log, test_acc_log = multiple_trial()
plot_errorbar(train_acc_log, test_acc_log, './fig/test.pdf')