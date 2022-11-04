from config import Config
import numpy as np
import torch
import random
from RC import torchRC



class AttackGym:
    def __init__(self, config:Config):
        self.config = config
        
        self.reset()
        
    def reset(self,):
        config = self.config
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self.model = torchRC(config)
        self.T = 0
    
    def step(self, action):
        # reservoir attacking
        self.model.W_ins[0] = None
        
        obs, reward = rollout_attack(self.config, self.model)
        
        if self.T > self.config.episode_len:
            done = True
        else:
            done = False
        return obs, reward, done

def test():
    '''
    测试，把图中某个神经元移除，具体做法是屏蔽邻接矩阵中，该神经元的输入权重和输出权重
    对于随机输入，被屏蔽的神经元的膜电位模式与其他神经元，具有较为显著的差别。
    '''
    config.N_in = 10
    config.N_hid = 100
    config.frames = 13
    model = torchRC(config)
    A = model.As[0].detach().numpy()
    print(A.shape)
    G = nx.from_numpy_array(A, create_using=nx.DiGraph)

    model.As[0] = model.As[0].detach()
    model.As[0][:,15] = 0
    model.As[0][15,:] = 0

    model.As[0][:,17] = 0
    model.As[0][17,:] = 0

    out = model(torch.rand(12,config.frames, config.N_in))
    means = []
    vars = []
    for i in range(100):
        if i==15: continue
        elif i==17: continue
        means.append(out[0][:,:, i].mean().item())
        vars.append(out[0][:,:,i].std().item())
    print(np.mean(means))
    print(np.mean(vars))
    print(out[0][:,:,15].mean())
    print(out[0][:,:,15].std())
    print(out[0][:,:,17].mean())
    print(out[0][:,:,17].std())

def attack(model:torchRC,
           config,
           
           ):
    # attack
    model.W_ins[0]
    
    # rollout
    reward = rollout_attack(config, model)

if __name__ == '__main__':
    pass