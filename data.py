import torch
import random
import numpy as np
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.distributions import Poisson
from torch.utils.data import DataLoader, Dataset
from config import Config

def Rossler(a, b, c, dt, T):
    '''
    Implementation of chaotic Rossler system
    x' = -y-z
    y' = x+ay
    z' = b+z(x-c)
    a=0.5, b=2.0, c=4.0
    '''
    s = [[np.random.uniform(0,1) for _ in range(3)],]
    T = np.arange(0, T, dt)
    for t in T:
        x = s[-1][0] + dt * (-s[-1][1] - s[-1][2])
        y = s[-1][1] + dt * (s[-1][0] + a * s[-1][1])
        z = s[-1][2] + dt * (b + s[-1][2] * (s[-1][0] - c))
        s.append([x, y, z])
    return np.array(s)

def MackeyGlass(x0, tau, dt, T):
    '''
    implementation of Mackey-Glass System
    
    '''
    raise NotImplementedError

def Lorenz63(train_num=1000):
    '''
    implementation for Lorenz 63
    '''
    dt = 0.01 # delta t
    
    traj = np.zeros((train_num, 3), dtype=np.float32)
    traj[0] = [0.1,0.1,0.1] # init point
    
    for i in range(train_num-1):
        traj[i+1, 0] = traj[i, 0] + dt * 10 * (traj[i, 1] - traj[i, 0])
        traj[i+1, 1] = traj[i, 1] + dt * (traj[i, 0] * (28 - traj[i, 2]) - traj[i, 1])
        traj[i+1, 2] = traj[i, 2] + dt * (traj[i, 0] * traj[i, 1] - 8/3 * traj[i, 2])
        
    return traj


class PoissonData(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label
        self.length = data.shape[0]
    
    def __getitem__(self, index):
        label = self.label[index]
        data = self.data[index]
        return data, label
    
    def __len__(self):
        return self.length

def Poisson_samples_fast(N_samples, N_in, T, rate):
    samples = torch.zeros(N_samples, T, N_in, dtype=torch.float32)
    for i in range(N_samples):
        # average time interval for two next spikes
        interval_mean = (T / rate) * torch.ones(N_in)
        interval_generate = Poisson(interval_mean)
        interval_sum = torch.zeros(N_in, dtype=torch.int32)
        spike = torch.zeros(T, N_in, dtype=torch.int32)
        while True:
            # sample the next spiking interval
            interval = interval_generate.sample().int()
            interval_sum += interval
            if (interval_sum>T-1).sum()==N_in:
                break
            # if interval_sum > T-1:
            #     break
            for i, interval in enumerate(interval_sum):
                if interval < T-1:
                    spike[interval, i] = 1.
        samples[i, :, :] = spike
    return samples

def Poisson_samples(N_samples, N_in=50, T=100, rate=10):
    '''
    Generate dataset of Poisson spike trains with specific firing rates
    N_in: dimension of a spike train
    '''
    assert T > rate and rate > 0
    def Poisson_spike_train(T, rate):
        '''
        Generate a poisson spike train
        T: length of spike trains
        rate: large rate for more frequent spikes
        '''
        
        # average time interval for two next spikes
        interval_mean = int(T / rate)
        interval_generate = Poisson(interval_mean)
        
        interval_sum = 0
        spike = [0. for _ in range(T)]
        while True:
            # sample the next spiking interval
            interval = interval_generate.sample().item()
            interval_sum += int(interval)
            if interval_sum > T-1:
                break
            spike[interval_sum] = 1.
        return np.array(spike)
    
    samples = []
    for i in range(N_samples):
        sample = np.array([Poisson_spike_train(T, rate) for _ in range(N_in)]).T
        samples.append(sample)
    # samples = np.array(samples).T
    samples = torch.tensor(samples, dtype=torch.float32)
    return samples

def PoissonDataset(config:Config):
    # training set
    true_num = int(config.train_num/2)
    false_num = int(config.train_num/2)
    true_data = Poisson_samples_fast(true_num, config.N_in, config.frames, config.rate[0])
    false_data = Poisson_samples_fast(false_num, config.N_in, config.frames, config.rate[1])
    data = torch.cat((true_data, false_data), dim=0)
    label = torch.cat((torch.ones(true_num, dtype=torch.long), torch.zeros(false_num, dtype=torch.long)), dim=0)
    
    dataset = PoissonData(data, label)
    trainloader = DataLoader(dataset=dataset, batch_size=config.batch_size, shuffle=True, drop_last=False)
    
    # testing set
    true_num = int(config.test_num/2)
    false_num = int(config.test_num/2)
    true_data = Poisson_samples(true_num, config.N_in, config.frames, config.rate[0])
    false_data = Poisson_samples(false_num, config.N_in, config.frames, config.rate[1])
    data = torch.cat((true_data, false_data), dim=0)
    label = torch.cat((torch.ones(true_num, dtype=torch.long), torch.zeros(false_num, dtype=torch.long)), dim=0)
    
    dataset = PoissonData(data, label)
    testloader = DataLoader(dataset=dataset, batch_size=config.batch_size, shuffle=True, drop_last=False)
    if config.verbose:
        print('Data Generation Finish')
    return trainloader, testloader

def part_DATA(config:Config):
    '''
    load dataset
    '''
    if config.data == 'cifar10':
        transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(), # convert to tensor and rascale to [0,1]
            ])
        train_dataset = torchvision.datasets.CIFAR10(root='./data/',
                                                    train=True,
                                                    download=False,
                                                    transform=transform)
        test_dataset = torchvision.datasets.CIFAR10(root='./data/',
                                                    train=False,
                                                    download=False,
                                                    transform=transform)
    elif config.data == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(), # convert to tensor and rascale to [0,1]
            ])
        train_dataset = torchvision.datasets.MNIST(root='./data/',
                                                    train=True,
                                                    download=False,
                                                    transform=transform)
        test_dataset = torchvision.datasets.MNIST(root='./data/',
                                                    train=False,
                                                    download=False,
                                                    transform=transform)
        
    train_dataset, _ = torch.utils.data.random_split(train_dataset, 
                                                     [config.train_num, len(train_dataset)-config.train_num])
    test_dataset, _ = torch.utils.data.random_split(test_dataset, 
                                                    [config.test_num, len(test_dataset)-config.test_num])
    
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                               batch_size=config.batch_size, 
                                               shuffle=True, 
                                               num_workers=0,)
    test_loader = torch.utils.data.DataLoader(test_dataset, 
                                               batch_size=config.batch_size, 
                                               shuffle=False, 
                                               num_workers=0,)
    if config.verbose:
        print('Data Generation Finish')
    return train_loader, test_loader


def part_MNIST(config:Config):
    train_dataset = torchvision.datasets.MNIST(root='./data/', 
                                               train=True, 
                                               download=False, 
                                               transform=transforms.ToTensor())
    test_dataset = torchvision.datasets.MNIST(root='./data/', 
                                          train=False, 
                                          download=False, 
                                          transform=transforms.ToTensor())
    
    random_list = random.sample(list(range(len(train_dataset))), config.train_num)
    train_data = train_dataset.train_data[random_list]
    train_label = train_dataset.targets[random_list]
    
    random_list = random.sample(list(range(len(test_dataset))), config.test_num)
    test_data = test_dataset.train_data[random_list]
    test_label = test_dataset.targets[random_list]
    
    return train_data, train_label, test_data, test_label
    
def MNIST_generation(batch_size=1):
    '''
    生成随机编码的MNIST动态数据集
    train_num: 训练集样本数
    '''
    
    train_dataset = torchvision.datasets.MNIST(root='./data/', 
                                               train=True, 
                                               download=False, 
                                               transform=transforms.ToTensor())
    
    # 只取一部分数据
    # assert train_num<= len(train_dataset)
    # idx = random.sample(list(range(len(train_dataset))), train_num)
    # train_dataset.data = train_dataset.data[idx]
    
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                               batch_size=batch_size, 
                                               shuffle=True, 
                                               num_workers=0)

    test_dataset = torchvision.datasets.MNIST(root='./data/', 
                                          train=False, 
                                          download=False, 
                                          transform=transforms.ToTensor())
    
    # 只取一部分数据
    # assert test_num<= len(test_dataset)
    # idx = random.sample(list(range(len(test_dataset))), test_num)
    # test_dataset.data = test_dataset.data[idx]
    
    test_loader = torch.utils.data.DataLoader(test_dataset, 
                                              batch_size=batch_size, 
                                              shuffle=False, 
                                              num_workers=0)
    return train_loader, test_loader

if __name__ == '__main__':
    # train_loader, test_loader = MNIST_generation()
    train_data, train_label, test_data, test_label = part_CIFAR10(train_num=6000, test_num=1000)
    # train_data = Lorenz63(train_num=1000)
    # train_data = Rossler(a=0.5, b=2.0, c=4.0, dt=0.01, T=10000)
    
    # plt.plot(train_data[:,0])
    # plt.plot(train_data[:,1])
    # plt.plot(train_data[:,2])
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # ax.plot(train_data[:,0],train_data[:,1],train_data[:,2])
    # plt.show()
    