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
    data_augment = True
    augment_noise = 0.9
    input_learn = True # learnable input layer
    seed = 123
    trials = 5        # try on 5 different seeds
    batch = 256
    epoch = 40
    lr = 0.005
    l1_loss = 0.0003
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
    


from spikingjelly.datasets.shd import SpikingHeidelbergDigits
SHD_train = SpikingHeidelbergDigits('D:\Ph.D\Research\SNN-SRT数据\SHD', train=True, data_type='frame', frames_number=80, split_by='number')
if config.data_augment:
    SHD_train_aug = []
    for x, y in SHD_train:
        SHD_train_aug.append([x, y])
        for _ in range(2):
            noise = np.array(np.random.random(x.shape)>0.95, dtype=np.float)
            SHD_train_aug.append([x + noise, y])
    train_loader = torch.utils.data.DataLoader(dataset=SHD_train_aug, batch_size=config.batch, shuffle=True, drop_last=False, num_workers=0)
else:
    train_loader = torch.utils.data.DataLoader(dataset=SHD_train, batch_size=config.batch, shuffle=True, drop_last=False, num_workers=0)
SHD_test = SpikingHeidelbergDigits('D:\Ph.D\Research\SNN-SRT数据\SHD', train=False, data_type='frame', frames_number=80, split_by='number')
test_loader = torch.utils.data.DataLoader(dataset=SHD_test, batch_size=config.batch, shuffle=False, drop_last=False, num_workers=0)


def gaussian(x, mu=0., sigma=.5):
    return torch.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / torch.sqrt(2 * torch.tensor(torch.pi)) / sigma

class ActFun(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(0).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input - 0) < config.lens 
        return grad_input * temp.float()

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

act_fun = ActFun.apply
act_fun_adp = ActFun_adp.apply

def mem_update(op1, op2, input, mem, spk, thr, decay, rst):
    mem = rst * spk + mem * decay * (1-spk) + F.sigmoid(op2(input + op1(spk)))
    spike = act_fun(mem - thr)
    return mem, spike

def RadLIF_update(input, mem, spk, thr, alpha, beta, a, b, W):
    mem_ = alpha * mem + (1-alpha) * (input-W) - thr * spk
    W_ = beta * W + (1-beta) * a * mem + b * spk
    spike = act_fun(mem_ - thr)
    return mem_, spike, W_

def mem_update_adp(inputs, mem, spike, tau_adp, b, tau_m, dt=1, isAdapt=1):
    alpha = torch.exp(-1. * dt / tau_m).cuda()
    ro = torch.exp(-1. * dt / tau_adp).cuda()
    if isAdapt:
        beta = 1.8
    else:
        beta = 0.

    b = ro * b + (1 - ro) * spike
    B = config.b_j0 + beta * b

    mem = mem * alpha + (1 - alpha) * config.R_m * inputs - B * spike * dt
    inputs_ = mem - B
    spike = act_fun_adp(inputs_)  # act_fun : approximation firing function
    return mem, spike, B, b

def output_Neuron(inputs, mem, tau_m, dt=1):
    """
    The read out neuron is leaky integrator without spike
    """
    # alpha = torch.exp(-1. * dt / torch.FloatTensor([30.])).cuda()
    alpha = torch.exp(-1. * dt / tau_m).cuda()
    mem = mem * alpha + (1. - alpha) * config.R_m * inputs
    return mem

class RC_revise(nn.Module):
    def __init__(self) -> None:
        super(RC_revise, self).__init__()
        input = config.input
        hid = config.hid
        out = config.output
        self.inpt_hid1 = nn.Linear(input, hid)
        self.hid1_hid1 = nn.Linear(hid, hid) # A1
        self.hid1_hid2 = nn.Linear(hid, hid)
        self.hid2_hid2 = nn.Linear(hid, hid) # A2
        self.hid2_out = nn.Linear(hid, out)
        
        self.hid1_hid1.weight.data = 0.2 * self.hid1_hid1.weight.data
        self.hid2_hid2.weight.data = 0.2 * self.hid2_hid2.weight.data
        
        nn.init.orthogonal_(self.inpt_hid1.weight)  # 主要用以解决深度网络的梯度消失爆炸问题，在RNN中经常使用
        nn.init.orthogonal_(self.hid2_hid2.weight)
        nn.init.xavier_uniform_(self.inpt_hid1.weight) # 保持输入输出的方差一致，避免所有输出值都趋向于0。通用方法，适用于任何激活函数
        nn.init.xavier_uniform_(self.hid1_hid2.weight)
        nn.init.xavier_uniform_(self.hid2_out.weight)
        
        self.tau_adp_h1 = nn.Parameter(torch.Tensor(hid))
        self.tau_adp_h2 = nn.Parameter(torch.Tensor(hid))
        self.tau_adp_o = nn.Parameter(torch.Tensor(out))
        self.tau_m_h1 = nn.Parameter(torch.Tensor(hid))
        self.tau_m_h2 = nn.Parameter(torch.Tensor(hid))
        self.tau_m_o = nn.Parameter(torch.Tensor(out))
        
        nn.init.normal_(self.tau_adp_h1, 150, 10)
        nn.init.normal_(self.tau_adp_h2, 150, 10)
        nn.init.normal_(self.tau_adp_o, 150, 10)
        nn.init.normal_(self.tau_m_h1, 20., 5)
        nn.init.normal_(self.tau_m_h2, 20., 5)
        nn.init.normal_(self.tau_m_o, 20., 5)
        
        # self.A1.weight.data = nn.Parameter(torch.tensor(A_cluster(config)))
        # self.A2.weight.data = nn.Parameter(torch.tensor(A_cluster(config)))
        # nn.init.uniform_(self.A1.weight, a=-1, b=1)
        # nn.init.uniform_(self.A2.weight, a=-1, b=1)
        # nn.init.uniform_(self.A3.weight, a=-1, b=1)
        # nn.init.uniform_(self.A4.weight, a=-1, b=1)
        
        self.fc_out = nn.Linear(hid*2, out)
        
        self.thr = nn.Parameter(torch.rand(config.hid)*config.thr, requires_grad=True)
        self.decay = nn.Parameter(torch.rand(config.hid)*config.decay, requires_grad=True)
        self.rst = nn.Parameter(torch.rand(config.hid)*config.rst, requires_grad=True)
        
        
        if not config.input_learn:
            for name, p in self.named_parameters():
                if 'conv' in name or 'fc_in' in name:
                    p.requires_grad = False
    
    def mem_update_adp(self, inputs, mem, spike, tau_adp, b, tau_m, dt=1, isAdapt=1):
        alpha = torch.exp(-1. * dt / tau_m).cuda()
        ro = torch.exp(-1. * dt / tau_adp).cuda()
        if isAdapt:
            beta = 1.8
        else:
            beta = 0.

        b = ro * b + (1 - ro) * spike
        B = config.b_j0 + beta * b

        mem = mem * alpha + (1 - alpha) * config.R_m * inputs - B * spike * dt
        inputs_ = mem - B
        spike = act_fun_adp(inputs_)  # act_fun : approximation firing function
        return mem, spike, B, b
    
    def forward(self, input, mask, device='cuda', mode='serial'):
        A1_mask, A2_mask, _, _ = mask
        batch = input.shape[0]
        time_step = input.shape[1]
        self.b_hid1 = self.b_hid2 = self.b_out = config.b_j0
        
        hid1_mem = torch.zeros(batch, config.hid).uniform_(0, 0.1).to(device)
        hid1_spk = torch.zeros(batch, config.hid).to(device)
        
        hid2_mem = torch.zeros(batch, config.hid).uniform_(0, 0.1).to(device)
        hid2_spk = torch.zeros(batch, config.hid).to(device)
        
        out_mem = torch.zeros(batch, config.output).uniform_(0, 0.1).to(device)
        output = torch.zeros(batch, config.output).to(device)
        
        sum1_spk = torch.zeros(batch, config.hid).to(device)
        sum2_spk = torch.zeros(batch, config.hid).to(device)
        
        self.hid1_hid1.weight.data = self.hid1_hid1.weight.data * A1_mask.T.to(device)
        self.hid2_hid2.weight.data = self.hid2_hid2.weight.data * A2_mask.T.to(device)
        for t in range(time_step):
            input_t = input[:,t,:].float()
            
            # x = self.conv_in(input[:, t, :, :, :])
            
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
            out_mem = output_Neuron(inpt_out, out_mem, self.tau_m_o)
            output += F.softmax(out_mem, dim=1)
            
        sum1_spk /= time_step
        sum2_spk /= time_step

        A_norm = torch.norm(self.hid1_hid1.weight, p=1) + \
                 torch.norm(self.hid2_hid2.weight, p=1)
        return output, sum1_spk, sum2_spk, A_norm

def train(model, optimizer, criterion, num_epochs, train_loader, test_loader, device):
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
            
            outputs, _, _, A_norm = model(samples.to(device), mask, device)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted.cpu() == labels.long().cpu()).sum()
            
            loss = criterion(outputs, labels) + config.l1_loss*F.relu(A_norm-config.l1_targ) # , torch.max(A_norm-6000, 0) 规定一个区间，
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

def test(model, dataloader, mask, device='cuda'):
    with torch.no_grad():
        correct, total = 0, 0
        for images, labels in dataloader:
            outputs, _, _, _ = model(images.to(device), mask, device=device)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted.cpu() == labels.long().cpu()).sum()
        accuracy = 100. * correct.numpy() / total
    return accuracy

config.l1_targ = 50000
config.dropout = 0.01
config.dropout_stepping = 0.01
config.dropout_stop = 0.05
# model = RC().to(config.device)
model = RC_revise().to(config.device)
# model = RC_RadLIF().to(config.device)
# model = RC_parallel().to(config.device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                              lr=config.lr,
                              weight_decay=config.weight_decay)
acc = train(model, optimizer, criterion, config.epoch, train_loader, test_loader, 'cuda')