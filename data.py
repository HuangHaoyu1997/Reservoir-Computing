import torch
import random
import numpy as np
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

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
    train_data = Lorenz63(train_num=1000)
    
    
    # plt.plot(train_data[:,0])
    # plt.plot(train_data[:,1])
    # plt.plot(train_data[:,2])
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(train_data[:,0],train_data[:,1],train_data[:,2])
    plt.show()
    