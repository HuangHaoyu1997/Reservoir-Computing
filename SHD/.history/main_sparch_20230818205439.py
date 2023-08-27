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
    