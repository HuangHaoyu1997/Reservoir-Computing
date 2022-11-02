from train_gpu import *
import matplotlib.pyplot as plt
from utils import *

if __name__ == '__main__':
    from config import Config as config
    t = time.time()
    # print(1000**2-(model.A==0).sum())
    # print(spectral_radius(model.A))
    
    # plt.imshow(model.A)
    # plt.show()
    # train_loader, test_loader = MNIST_generation(train_num=100,
    #                                              test_num=250,
    #                                              batch_size=13)
    
    # inference(model, train_loader, frames=20)
    
    
    rollouts(config)
    rollout_attack
    print(time.time()-t)

