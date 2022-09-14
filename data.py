import torch
import random
import torchvision
import torchvision.transforms as transforms

def MNIST_generation(train_num=1000, test_num=250, batch_size=1):
    '''
    生成随机编码的MNIST动态数据集
    train_num: 训练集样本数
    '''
    
    train_dataset = torchvision.datasets.MNIST(root='./reservoir/data/', 
                                               train=True, 
                                               download=False, 
                                               transform=transforms.ToTensor())
    
    # 只取一部分数据
    assert train_num<= len(train_dataset)
    idx = random.sample(list(range(len(train_dataset))), train_num)
    train_dataset.data = train_dataset.data[idx]
    
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                               batch_size=batch_size, 
                                               shuffle=True, 
                                               num_workers=0)

    test_dataset = torchvision.datasets.MNIST(root='./reservoir/data/', 
                                          train=False, 
                                          download=False, 
                                          transform=transforms.ToTensor())
    
    # 只取一部分数据
    assert test_num<= len(test_dataset)
    idx = random.sample(list(range(len(test_dataset))), test_num)
    test_dataset.data = test_dataset.data[idx]
    
    test_loader = torch.utils.data.DataLoader(test_dataset, 
                                              batch_size=batch_size, 
                                              shuffle=False, 
                                              num_workers=0)
    return train_loader, test_loader