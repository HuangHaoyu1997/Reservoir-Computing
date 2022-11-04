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

def attack(model:torchRC,
           config,
           
           ):
    # attack
    model.W_ins[0]
    
    # rollout
    reward = rollout_attack(config, model)

if __name__ == '__main__':
    pass