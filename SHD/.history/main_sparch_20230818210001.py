'''
首次修改2023年8月18日20:53:29

'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import time, warnings, os, h5py, logging
warnings.filterwarnings("ignore")
from datetime import timedelta
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
logger = logging.getLogger(__name__)

class config:
    date = time.strftime("%Y-%m-%d-%H-%M-%S/", time.localtime(time.time()))[5:16]
    new_exp_folder = './log/' + date
    dataset_name = 'shd'
    data_folder = './data/raw/'
    input_dim = 700
    output_dim = 20
    
    batch_size = 512
    nb_epochs = 30
    lr = 1e-2
    scheduler_patience = 1
    scheduler_factor = 0.7
    reg_factor = 0.5
    reg_fmin = 0.01
    reg_fmax = 0.2
    nb_steps = 50
    trial = 5
    seed = round(time.time())
    ckpt_freq = 5
    threshold = 1.0
    
    pdrop = 0.1
    normalization = 'batchnorm'
    train_input = True
    nb_hiddens = 1024
    noise_test = 0.0
    
    device = 'cuda'

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
class SpikeFunctionBoxcar(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.gt(0).float()

    def backward(ctx, grad_spikes):
        (x,) = ctx.saved_tensors
        grad_x = grad_spikes.clone()
        grad_x[x <= -0.5] = 0
        grad_x[x > 0.5] = 0
        return grad_x
#######################################
########### define RC model ###########

class RC(nn.Module):
    def __init__(self):
        super(RC, self).__init__()
        # Fixed params
        self.alpha_lim = [np.exp(-1 / 5), np.exp(-1 / 25)]
        self.beta_lim = [np.exp(-1 / 30), np.exp(-1 / 120)]
        self.a_lim = [-1.0, 1.0]
        self.b_lim = [0.0, 2.0]
        self.spike_fct = SpikeFunctionBoxcar.apply
        
        # Trainable parameters
        self.W = nn.Linear(Config.input_dim, Config.nb_hiddens, bias=True)
        self.V = nn.Linear(Config.nb_hiddens, Config.nb_hiddens, bias=False)
        self.alpha = nn.Parameter(torch.Tensor(Config.nb_hiddens))
        self.beta = nn.Parameter(torch.Tensor(Config.nb_hiddens))
        self.a = nn.Parameter(torch.Tensor(Config.nb_hiddens))
        self.b = nn.Parameter(torch.Tensor(Config.nb_hiddens))
        
        self.W_read = nn.Linear(Config.nb_hiddens, Config.output_dim, bias=True)
        self.alpha_read = nn.Parameter(torch.Tensor(Config.output_dim))

        nn.init.uniform_(self.alpha, self.alpha_lim[0], self.alpha_lim[1])
        nn.init.uniform_(self.beta, self.beta_lim[0], self.beta_lim[1])
        nn.init.uniform_(self.a, self.a_lim[0], self.a_lim[1])
        nn.init.uniform_(self.b, self.b_lim[0], self.b_lim[1])
        nn.init.orthogonal_(self.V.weight)
        nn.init.uniform_(self.alpha_read, self.alpha_lim[0], self.alpha_lim[1])

        # Initialize normalization
        self.normalize = False
        if Config.normalization == "batchnorm":
            self.norm = nn.BatchNorm1d(Config.nb_hiddens, momentum=0.05)
            self.norm_read = nn.BatchNorm1d(Config.output_dim, momentum=0.05)
            self.normalize = True
        elif Config.normalization == "layernorm":
            self.norm = nn.LayerNorm(Config.nb_hiddens)
            self.norm_read = nn.LayerNorm(Config.output_dim)
            self.normalize = True
        self.drop = nn.Dropout(p=Config.pdrop)
        
        if not Config.train_input:
            for name, p in self.named_parameters():
                if 'W' in name: p.requires_grad = False
    
    def radlif_cell(self, Wx, mask, alpha, beta, a, b, V):
        ut = torch.rand(Wx.shape[0], Wx.shape[2]).to(Wx.device)
        wt = torch.rand(Wx.shape[0], Wx.shape[2]).to(Wx.device)
        st = torch.rand(Wx.shape[0], Wx.shape[2]).to(Wx.device)
        s = []
        # Bound values of parameters to plausible ranges
        alpha_ = torch.clamp(alpha, min=self.alpha_lim[0], max=self.alpha_lim[1])
        beta_ = torch.clamp(beta, min=self.beta_lim[0], max=self.beta_lim[1])
        a_ = torch.clamp(a, min=self.a_lim[0], max=self.a_lim[1])
        b_ = torch.clamp(b, min=self.b_lim[0], max=self.b_lim[1])
        # if self.dropout > 0: self.V.weight.data = self.V.weight.data * mask.T
        V_ = V.weight.clone().fill_diagonal_(0)
        for t in range(Wx.shape[1]):
            wt = beta_ * wt + a_ * ut + b_ * st
            ut = alpha_ * (ut - st) + (1 - alpha_) * (Wx[:, t, :] + torch.matmul(st, V_) - wt)
            st = self.spike_fct(ut - Config.threshold)
            s.append(st)
        return torch.stack(s, dim=1)
    
    def readout_cell(self, Wx, alpha):
        ut = torch.rand(Wx.shape[0], Wx.shape[2]).to(Wx.device)
        out = torch.zeros(Wx.shape[0], Wx.shape[2]).to(Wx.device)
        alpha_ = torch.clamp(alpha, min=self.alpha_lim[0], max=self.alpha_lim[1]) # Bound values of the neuron parameters to plausible ranges
        for t in range(Wx.shape[1]):
            ut = alpha * ut + (1 - alpha_) * Wx[:, t, :]
            out = out + F.softmax(ut, dim=1)
        return out
    
    def forward(self, x, mask):
        all_spikes = []
        
        Wx = self.W(x) # (all steps in parallel)
        if self.normalize:
            _Wx = self.norm(Wx.reshape(Wx.shape[0] * Wx.shape[1], Wx.shape[2]))
            Wx = _Wx.reshape(Wx.shape[0], Wx.shape[1], Wx.shape[2])
        s = self.radlif_cell(Wx, mask, self.alpha, self.beta, self.a, self.b, self.V)
        s = self.drop(s)
        all_spikes.append(s)
        
        Wx_ = self.W_read(s)
        if self.normalize:
            _Wx_ = self.norm_read(Wx_.reshape(Wx_.shape[0] * Wx_.shape[1], Wx_.shape[2]))
            Wx_ = _Wx_.reshape(Wx_.shape[0], Wx_.shape[1], Wx_.shape[2])
        out = self.readout_cell(Wx_, self.alpha_read)

        firing_rates = torch.cat(all_spikes, dim=2).mean(dim=(0, 1)) # Compute mean firing rate of each spiking neuron
        return out, firing_rates, all_spikes